import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
import numpy as np
import logging
import torch
from utils_log.utils import set_logger
import attack_generator as attack
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=2/255, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="wrn28",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--dataset', type=str, default="dtd", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=True, help='whether to use dynamic tau')
parser.add_argument('--depth', type=int, default=28, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./results/AT', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--nes', type=bool, default=False)
parser.add_argument('--ams', type=bool, default=False)

parser.add_argument('--linear', action='store_true', help='if specified, use pgd dual mode,(cal both adversarial and clean)')


args = parser.parse_args()

class batch_norm_multiple(nn.Module):
    def __init__(self, norm, inplanes, bn_names=None):
        super(batch_norm_multiple, self).__init__()

        # if no bn name input, by default use single bn
        self.bn_names = bn_names
        if self.bn_names is None:
            self.bn_list = norm(inplanes)
            return

        len_bn_names = len(bn_names)
        self.bn_list = nn.ModuleList([norm(inplanes) for _ in range(len_bn_names)])
        self.bn_names_dict = {bn_name: i for i, bn_name in enumerate(bn_names)}
        return

    def forward(self, x):
        out = x[0]
        name_bn = x[1]

        if name_bn is None:
            out = self.bn_list(out)
        else:
            bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out)

        return out

class proj_head_module(nn.Module):
    def __init__(self, ch, twoLayerProj=False, bn_names=None):
        super(proj_head_module, self).__init__()
        self.in_features = ch
        self.twoLayerProj = twoLayerProj

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)
        self.fc2 = nn.Linear(ch, ch, bias=False)
        self.bn2 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_name):
        # debug
        # print("adv attack: {}".format(flag_adv))
        x = self.fc1(x)
        x = self.bn1([x,bn_name])
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2([x, bn_name])
        return x

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

out_dir = args.out_dir + '_{}_{}'.format(args.net,args.dataset)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

set_logger(os.path.join(out_dir, 'training.log'))

logging.info(out_dir)
logging.info(args)

import torch.nn.functional as F
def pgd(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_logit = model(data, 'pgd')
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv, 'pgd')
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=True).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_logit, dim=1))
        loss_adv.backward(retain_graph=True)
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init)
            output = model(x_adv, 'pgd')
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy
    
def train(model, train_loader, optimizer):
    starttime = datetime.datetime.now()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data = pgd(model,data,target,epsilon=args.epsilon,step_size=args.step_size,num_steps=args.num_steps,
                            loss_fn='cent',category='Madry',rand_init=True)
        # print(data.shape)
        model.train()
        scheduler.step()
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return time, loss_sum

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += torch.nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

logging.info('==> Load Data')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform_test)
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='/data', train=False, download=True, transform=transform_test)
    num_classes = 100
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root='/data', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='/data', split='test', download=True, transform=transform_test)
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# logging.info(num_classes)
logging.info('==> Load Model')
bn_names = ['pgd', 'normal']
from models.wrn_with_bn_finetune import WideResNet
model = WideResNet(28, num_classes, 10, dropRate=0, bn_names=bn_names).cuda()

h_size = model.nChannels
proj_head = proj_head_module(h_size, bn_names=bn_names).cuda()
model.linear = proj_head
model = torch.nn.DataParallel(model).cuda()

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])

h_size = model.module.nChannels
classifier = torch.nn.Linear(h_size, num_classes).cuda()
model.module.linear = classifier

if args.linear:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.linear.parameters():
        param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=True)


def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def save_best_checkpoint(state, checkpoint=out_dir, filename='best_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    try:
        load_dict = torch.load(filepath)
        if state['test_pgd10_acc'] > load_dict['test_pgd10_acc']:
            torch.save(state, filepath)
    except:
        torch.save(state, filepath)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.lr))  # originally 1e-6

start_epoch = 0
best_test_nat_acc = 0
best_epoch = 0  
test_natloss_list = []
test_natacc_list = []
best_test_rob_acc = 0

for epoch in range(start_epoch, args.epochs):

    train_time, train_loss = train(model, train_loader, optimizer)
    model.eval()
    loss, test_nat_acc = eval_clean(model, test_loader)
    test_natloss_list.append(loss)
    loss, test_rob_acc = eval_robust(model, test_loader, perturb_steps=20, epsilon=8 / 255, step_size=2 / 255,
                                loss_fn="cent", category="Madry", rand_init=True)

    save_best_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'test_nat_acc': test_rob_acc,
        'test_natloss_list': test_natloss_list,
        'test_natacc_list': test_natacc_list,
    })

    best_test_rob_acc = max(best_test_rob_acc, test_rob_acc)

    logging.info(
        'Epoch: [%d | %d] | Train Time: %.2f s | Natural Test Acc %.4f | PGD20 Acc %.4f | Best PGD20 Acc %.4f \n' % (
        epoch + 1,
        args.epochs,
        train_time,
        test_nat_acc,
        test_rob_acc,
        best_test_rob_acc)
        )
    