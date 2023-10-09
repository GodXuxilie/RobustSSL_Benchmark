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
from transfer_utils import *

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=8/255, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=2/255, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="ResNet18",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--dataset', type=str, default="dtd", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=True, help='whether to use dynamic tau')
parser.add_argument('--depth', type=int, default=32, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./results/AT', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--nes', type=bool, default=False)
parser.add_argument('--ams', type=bool, default=False)

parser.add_argument('--linear', action='store_true', help='if specified, use pgd dual mode,(cal both adversarial and clean)')


args = parser.parse_args()
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

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            if args.net == 'prer50' or args.net == 'prer50_eps05':
                output = output[0]
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
# from data_loader import get_loader
from transfer_utils.transfer_datasets import *
if args.dataset == 'dtd':
    num_classes, loader = make_loaders_dtd(args.batch_size, workers=4)
    train_loader = loader[0]
    test_loader = loader[1]
elif args.dataset == 'bird':
    num_classes, loader = make_loaders_birds(args.batch_size, workers=4)
    train_loader = loader[0]
    test_loader = loader[1]
elif args.dataset == 'food':
    num_classes, loader = make_loaders_food(args.batch_size, workers=4)
    train_loader = loader[0]
    test_loader = loader[1]
elif args.dataset == 'car':
    num_classes, loader = make_loaders_cars(args.batch_size, workers=4)
    train_loader = loader[0]
    test_loader = loader[1]
elif args.dataset == 'sun':
    num_classes, loader = make_loaders_SUN(args.batch_size, workers=4)
    train_loader = loader[0]
    test_loader = loader[1]

transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

if args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    num_classes = 100
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

if args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)


print('==> Load Model')
from torchvision.models.resnet import ResNet, Bottleneck
class ResNet_new(ResNet):
    def __init__(
    self,
    block,
    layers,
    num_classes = 1000,
    zero_init_residual = False,
    groups = 1,
    width_per_group = 64,
    replace_stride_with_dilation = None,
    norm_layer = None,
    ) -> None:
        super().__init__(block,
    layers,
    num_classes,
    zero_init_residual,
    groups,
    width_per_group,
    replace_stride_with_dilation,
    norm_layer)

    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

print('==> Load Model')
model = ResNet_new(Bottleneck, [3, 4, 6, 3]).cuda()
model = torch.nn.DataParallel(model).cuda()

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])

classifier = torch.nn.Linear(2048, num_classes).cuda()
model.module.fc = classifier

if args.linear:
    for param in model.parameters():
        param.requires_grad = False  
    for param in model.module.fc.parameters():
            param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = True

print(model)


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nes)

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


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr 

start_epoch = 0
best_test_nat_acc = 0
best_epoch = 0  
test_natloss_list = []
test_natacc_list = []

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    starttime = datetime.datetime.now()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    endtime = datetime.datetime.now()
    train_time = (endtime - starttime).seconds

    model.eval()
    loss, test_nat_acc = eval_clean(model, test_loader)
    test_natloss_list.append(loss)

    save_best_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'test_nat_acc': test_nat_acc,
        'test_natloss_list': test_natloss_list,
        'test_natacc_list': test_natacc_list,
    })

    best_test_nat_acc = max(best_test_nat_acc, test_nat_acc)

    logging.info(
        'Epoch: [%d | %d] | Train Time: %.2f s | Natural Test Acc %.4f | Best Natural Test Acc %.4f \n' % (
        epoch + 1,
        args.epochs,
        train_time,
        test_nat_acc,
        best_test_nat_acc)
        )