import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import copy
from pdb import set_trace
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from autoattack import AutoAttack
from robustbench.data import load_cifar10c,load_cifar100c
import math
from torch.utils.data import Dataset
from typing import TypeVar, Sequence
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:

        self.dataset = dataset
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
# DeACL
def load_BN_checkpoint_DeACL(args,state_dict):
    new_state_dict = {}
    new_state_dict_normal = {}
    for k, v in state_dict.items():
        if 'backbone.' in k:
            k = k.replace('backbone.', '')
        new_state_dict_normal[k] = v
        new_state_dict[k] = v
    return new_state_dict

# AdvCL and A-InfoNCE
def load_BN_checkpoint_AdvCL(args, state_dict):  
    new_state_dict = {}
    new_state_dict_normal = {}
    for k, v in state_dict.items():
        if 'downsample.bn.bn_list.0' in k:
            k = k.replace('downsample.bn.bn_list.0', 'downsample.0')
            new_state_dict_normal[k] = v
        elif 'downsample.bn.bn_list.1' in k:
            k = k.replace('downsample.bn.bn_list.1', 'downsample.1')
            new_state_dict[k] = v
        elif '.bn_list.0' in k:
            k = k.replace('.bn_list.0', '')
            new_state_dict_normal[k] = v
        elif '.bn_list.1' in k:
            k = k.replace('.bn_list.1', '')
            new_state_dict[k] = v
        elif 'downsample.conv' in k:
            k = k.replace('downsample.conv', 'downsample.0')
            new_state_dict_normal[k] = v
            new_state_dict[k] = v
        else:
            new_state_dict_normal[k] = v
            new_state_dict[k] = v
    
    return new_state_dict

# DeACL
def load_BN_checkpoint_lora(state_dict):
    new_state_dict = {}
    new_state_dict_normal = {}
    for k, v in state_dict.items():
        if 'conv1.weight' in k and k != 'conv1.weight':
            k = k.replace('conv1.weight', 'conv1.conv.weight')
        elif 'conv2.weight' in k:
            k = k.replace('conv2.weight', 'conv2.conv.weight')
        elif 'conv3.weight' in k:
            k = k.replace('conv3.weight', 'conv3.conv.weight')
        elif 'conv4.weight' in k:
            k = k.replace('conv4.weight', 'conv4.conv.weight')
        elif 'downsample.conv.weight' in k:
            k = k.replace('downsample.conv.weight', 'downsample.conv.conv.weight')
        new_state_dict_normal[k] = v
        new_state_dict[k] = v
    return new_state_dict, new_state_dict_normal

def cvt_state_dict(state_dict, args, num_classes):
    state_dict_new = copy.deepcopy(state_dict)
    if args.bnNameCnt >= 0:
        for name, item in state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace(
                    '.bn_list.{}'.format(args.bnNameCnt), '')] = item

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace(
                'downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace(
                'downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]
    state_dict_new['fc.weight'] = torch.zeros(num_classes, 512).cuda()
    state_dict_new['fc.bias'] = torch.zeros(num_classes).cuda()
    return state_dict_new

def cvt_state_dict_AFF(state_dict, args):
    state_dict_new = copy.deepcopy(state_dict)

    if args.bnNameCnt >= 0:
        for name, item in state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace(
                    '.bn_list.{}'.format(args.bnNameCnt), '')] = item

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace(
                'downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace(
                'downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    state_dict_new['fc.weight'] = state_dict['fc.weight']
    state_dict_new['fc.bias'] = state_dict['fc.bias']
    return state_dict_new


def pgd_attack(model, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    loss = nn.CrossEntropyLoss()

    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        if advFlag is None:
            if forceEval:
                model.eval()
            outputs = model(images + delta)
        else:
            if forceEval:
                model.eval()
            outputs = model(images + delta, advFlag)

        model.zero_grad()
        cost = loss(outputs, labels)
        # cost.backward()
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the speaccuracycified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_adv_test(model, device, test_loader, epsilon, alpha, criterion, log, attack_iter=40, bn_name=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input_adv = pgd_attack(model, input, target, device,
                               eps=epsilon, iters=attack_iter, alpha=alpha, advFlag=bn_name).data
        with torch.no_grad():
            if bn_name is not None:
                output = model.eval()(input_adv, bn_name=bn_name)
            else:
                output = model.eval()(input_adv)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   advFlag=None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            if advFlag is not None:
                output = model.eval()(x_curr, advFlag)
            else:
                output = model.eval()(x_curr)
            # output = model(x_curr,advFlag=advFlag)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

def eval_test_nat(model, test_loader, device, advFlag=None,natural_mode=None):
    # torch.manual_seed(1)
    model.eval()
    acc = 0.
    print(natural_mode)
    for images, labels in test_loader:
         images = images.to(device)
         labels = labels.to(device)
         with torch.no_grad():
            if natural_mode is not None:
                output = model.eval()(images,bn_name=natural_mode)
            else:
                output = model.eval()(images)
            acc += (output.max(1)[1] == labels).float().sum()
    print(acc, len(test_loader.dataset))
    return acc.item() / len(test_loader.dataset)

def eval_test_OOD(model, test_loader, log, device, advFlag=None):
    CORRUPTIONS = ["shot_noise", "gaussian_noise", "impulse_noise","contrast",
                   "zoom_blur", "motion_blur", "glass_blur", "defocus_blur",
                   "snow", "pixelate", "brightness", "fog", 
                    "jpeg_compression", "elastic_transform", "frost"]
    # fix random seed for testing
    # torch.manual_seed(1)
    model.eval()
    acc_list = []
    for i in range(5):
        acc_sum = 0
        count = 0
        for j in range(len(CORRUPTIONS)):
            corruptions = [CORRUPTIONS[j]]
            if test_loader == 'cifar100':
                x_test, y_test = load_cifar100c(n_examples=10000, corruptions=corruptions, severity=i+1, data_dir='../data')
            elif test_loader == 'cifar10':
                x_test, y_test = load_cifar10c(n_examples=10000, corruptions=corruptions, severity=i+1, data_dir='../data')
            acc = clean_accuracy(model, x_test, y_test, device=device,advFlag=advFlag)
            log.info('{}-{}, Acc: {}'.format(CORRUPTIONS[j], i+1, acc))
            acc_sum += acc
            count += 1
        log.info('Severity: {}, Acc: {}'.format(i+1, acc_sum/count))
        acc_list.append(acc_sum/count)
    return acc_list, np.mean(acc_list)

def runAA(args, model, test_loader, log_path, advFlag=None):
    model.eval()
    acc = 0.
    # test_loader.batch_size = 8000 if args.dataset == 'stl10' else 10000
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=8000 if args.dataset == 'stl10' else 10000, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', log_path=log_path)
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        xadv = adversary.run_standard_evaluation(images, labels, bs=128)
        with torch.no_grad():
            if advFlag is not None:
                output = model.eval()(xadv, advFlag)
            else:
                output = model.eval()(xadv)
        acc += (output.max(1)[1] == labels).float().sum()
    return acc.item() / len(test_loader.dataset)

def get_loader(args):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.RandomCrop(args.resize if args.dataset == 'stl10' else 32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
    ])

    if args.dataset == 'cifar10':
        train_datasets = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=transform_train)
        vali_datasets = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=transform_test)
        num_classes = 10
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        train_datasets = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_train)
        vali_datasets = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=transform_test)
        num_classes = 100
        args.num_classes = 100
    elif args.dataset == 'stl10':
        train_datasets = torchvision.datasets.STL10(
            root=args.data, split='train', transform=transform_train, download=True)
        vali_datasets = datasets.STL10(
            root=args.data, split='train', transform=transform_test, download=True)
        testset = datasets.STL10(
            root=args.data, split='test', transform=transform_test, download=True)
        num_classes = 10     
        args.num_classes = 10
    else:
        print("dataset {} is not supported".format(args.dataset))
        assert False

    if args.val>0:
        val = 1 - args.val
        full_indices = np.arange(0,len(train_datasets),1)
        train_indices = np.random.choice(len(train_datasets), size=int(len(train_datasets) * val), replace=False)
        val_indices = np.delete(full_indices, train_indices)
        vali_datasets = Subset(vali_datasets, val_indices)
        train_datasets = Subset(train_datasets, train_indices)
        print('train num: {} val num: {}'.format(len(train_datasets), len(vali_datasets)))

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(vali_datasets, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, vali_loader, test_loader, num_classes, args

def get_model(args, num_classes, mode, log, device='cuda'):
    if not args.dualBN or mode in ['eval', 'ALF', 'SLF']:
        if args.dataset == 'stl10' and args.resize == 96:
            from models.resnet_stl import resnet18, resnet34, resnet50
        else:
            from models.resnet import resnet18, resnet34, resnet50
    else:
        bn_names = ['normal', 'pgd']
        if args.dataset == 'stl10' and args.resize == 96:
            from models.resnet_multi_bn_stl import resnet18, resnet34, resnet50
        else:
            from models.resnet_multi_bn import resnet18, resnet34, resnet50

    ####### set do_normalize=1 if your model needs to normalize the input, otherwise set do_normalize=0 ########
    if args.pretraining in ['AdvCL', 'A-InfoNCE']:
        do_normalize = 0
    else:
        do_normalize = 1

    if args.model == 'r18':
        model_arch = resnet18
    elif args.model == 'r34':
        model_arch = resnet34
    elif args.model == 'r50':
        model_arch = resnet50

    if args.dualBN and mode == 'AFF':
        model = model_arch(pretrained=False, bn_names=bn_names, num_classes=num_classes).to(device)
    else:
        model = model_arch(num_classes=num_classes, do_normalize=do_normalize).to(device)

    if mode in ['SLF', 'ALF']:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert (len(parameters)) == 2  # fc.weight, fc.bias
    elif mode in ['AFF', 'eval']:
        parameters = list(model.parameters())
    
    log.info('len of parameters: {}'.format(len(parameters)))

    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if 'state_dict_dual' in checkpoint:
            state_dict = checkpoint['state_dict_dual']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        if mode in ['SLF', 'ALF']:
            ####### convert to single batchnorm batch ########
            log.info('convert to single batchnorm batch.')
            if args.pretraining in ['AdvCL', 'A-InfoNCE']:
                state_dict= load_BN_checkpoint_AdvCL(args, state_dict)
            elif args.pretraining == 'DeACL':
                state_dict = load_BN_checkpoint_DeACL(args, state_dict)
            elif args.pretraining in ['ACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++', 'DynACL-RCS'] :
                args.bnNameCnt = 1
                state_dict = cvt_state_dict(state_dict, args, num_classes=num_classes)
            state_dict['fc.weight'] = torch.zeros(num_classes, 512).to(device)
            state_dict['fc.bias'] = torch.zeros(num_classes).to(device)
            model.load_state_dict(state_dict, strict=False)
            log.info('read checkpoint {}'.format(args.checkpoint))
        elif mode == 'eval':
            if args.dualBN:
                args.bnNameCnt = 1
                state_dict = cvt_state_dict_AFF(state_dict,args)
            model.load_state_dict(state_dict, strict=False)
            log.info('read checkpoint {}'.format(args.checkpoint))
        elif mode == 'AFF':
            if args.pretraining in ['AdvCL', 'A-InfoNCE']:
                state_dict = load_BN_checkpoint_AdvCL(args, state_dict)
            elif args.pretraining == 'DeACL':
                state_dict = load_BN_checkpoint_DeACL(args, state_dict)
            elif not args.dualBN and args.pretraining in ['ACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++','DynACL-RCS'] :
                args.bnNameCnt = 1
                state_dict = cvt_state_dict(state_dict, args, num_classes=num_classes)
            state_dict['fc.weight'] = torch.zeros(num_classes, 512).to(device)
            state_dict['fc.bias'] = torch.zeros(num_classes).to(device)   
            model.load_state_dict(state_dict, strict=False)
            log.info('read checkpoint {}'.format(args.checkpoint))

        for name, param in model.named_parameters():
            if name not in state_dict.keys():
                log.info('Warning: Missing {} when loading state dict.'.format(name))
    return model, optimizer, scheduler

def trades_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, distance='l_inf'):
    batch_size = len(x_natural)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # nat_logit = model([x_natural,'nat']).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                model.eval()
                nat_logit = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax((nat_logit), dim=1))
                # loss_kl = F.cross_entropy(model(x_adv, thread=None), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                              epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        assert False

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    model.zero_grad()
    optimizer.zero_grad()
    model.train()
    nat_output = model(x_natural)
    adv_output= model(x_adv)

    LAMBDA1 = 1
    LAMBDA2 = 6
    loss = LAMBDA1 * F.cross_entropy(nat_output, y) \
        + LAMBDA2 *  (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1), F.softmax(nat_output, dim=1))
    
    return loss

def trades_loss_dual(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, distance='l_inf', natural_mode='normal'):
    batch_size = len(x_natural)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                model.eval()
                loss_kl = criterion_kl(F.log_softmax(model(x_adv,'pgd'), dim=1),
                                       F.softmax(model(x_natural,'normal'), dim=1))
                # loss_kl = F.cross_entropy(model(x_adv,bn_name='pgd',thread=None), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                              epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        assert False

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    model.zero_grad()
    optimizer.zero_grad()
    model.train()
    # calculate robust loss
    nat_output = model(x_natural, bn_name=natural_mode)
    adv_output= model(x_adv, bn_name='pgd')

    LAMBDA1 = 1.0
    LAMBDA2 = 6.0
    loss = LAMBDA1 * F.cross_entropy(nat_output, y) \
        + LAMBDA2 *  (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1), F.softmax(nat_output, dim=1))
    
    return loss


def train_loop(args, model, device, train_loader, optimizer, epoch, log, mode='ALF'):
    if mode in ['SLF', 'ALF']:
        model.eval()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
    if mode in ['AFF', 'SFF']:
        model.train()

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    loss_sum = 0
    count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        if mode == 'SLF':
            output = model.eval()(data)
            loss = criterion(output, target)
        if mode == 'ALF':
            data = pgd_attack(model, data, target, device, eps=args.epsilon,
                                      alpha=args.step_size, iters=args.num_steps_train, forceEval=True).data
            output = model.eval()(data)
            loss = criterion(output, target)
            
        if mode == 'AFF':
            ### Vanilla finetuning ###
            if args.dualBN:
                loss = trades_loss_dual(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps_train)
            else:
                loss = trades_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps_train)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        count += len(target)
        loss_sum += loss.item() * len(target)
        # break
        if batch_idx % int(len(train_loader)/4) == 0:
            log.info('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal time: {:.3f}'.format(
                mode, epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss_sum/count, totalTimeAve.avg))
        
    return model

def train(args, model, optimizer, scheduler, train_loader, test_loader, mode, device, log, model_dir, vali_loader):
    best_atacc = 0.0
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        
        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        # finetuning
        model = train_loop(args, model, device, train_loader, optimizer, epoch, log, mode)
        
       
        scheduler.step()

        # evaluation
        if (not args.test_frequency == 0) and (epoch % args.test_frequency == 1 or args.test_frequency == 1):
            log.info('================================================================')
            if args.dualBN and mode == 'AFF':
                state_dict = model.state_dict()
                args.bnNameCnt = 1
                state_dict = cvt_state_dict_AFF(state_dict, args)
                if args.dataset == 'stl10' and args.resize == 96:
                    from models.resnet_stl import resnet18, resnet34, resnet50
                else:
                    from models.resnet import resnet18, resnet34, resnet50
                if args.model == 'r18':
                    temp_model = resnet18
                if args.model == 'r34':
                    temp_model = resnet34
                if args.model == 'r50':
                    temp_model = resnet50
                temp_model = temp_model(num_classes=args.num_classes, do_normalize=1, r_nat=args.r_nat, r_adv=0).to(device)
                temp_model.load_state_dict(state_dict)
                temp_model.eval()
                nat_acc = eval_test_nat(temp_model, test_loader, device, None)
                vali_atacc = eval_adv_test(temp_model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
            else:
                model.eval()
                nat_acc = eval_test_nat(model, test_loader, device, None)
                vali_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
            if vali_atacc > best_atacc:
                best_atacc = vali_atacc
                best_acc = nat_acc
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                }, os.path.join(model_dir, '{}_model_bestAT.pt'.format(mode)))
            log.info('best_atacc: {} nat_acc: {}'.format(best_atacc, best_acc*100))
            log.info('================================================================')

    # save checkpoint
    model = save_checkpoint(args, model, model_dir, mode, device)

    return model


def save_checkpoint(args, model, model_dir, mode, device):
    if args.dualBN and mode == 'AFF':
        state_dict = model.state_dict()
        args.bnNameCnt = 1
        state_dict = cvt_state_dict_AFF(state_dict, args)
        if args.dataset == 'stl10' and args.resize == 96:
            print('ssss')
            from models.resnet_stl import resnet18, resnet34, resnet50
        else:
            from models.resnet import resnet18, resnet34, resnet50
        if args.model == 'r18':
            model = resnet18(num_classes=args.num_classes, do_normalize=1).to(device)
            model.load_state_dict(state_dict)
            model.eval()
    else:
        state_dict = model.state_dict()
        model.eval()

    torch.save({
        'state_dict': state_dict,
    }, os.path.join(model_dir, '{}_model_finetune.pt'.format(mode)))

    if args.test_frequency > 0:
        if args.dualBN and mode == 'AFF':
            args.bnNameCnt = 1
            state_dict = torch.load(os.path.join(model_dir, '{}_model_bestAT.pt'.format(mode)))['state_dict']
            state_dict = cvt_state_dict_AFF(state_dict, args)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(os.path.join(model_dir, '{}_model_bestAT.pt'.format(mode)))['state_dict'])

    return model

def setup_hyperparameter(args, mode):
    ####### Hyperparameter of ACL ########
    if args.pretraining  == 'ACL':
        if mode in ['SLF', 'ALF']:
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.05
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 32:
                args.batch_size = 128
                args.lr = 0.01
            args.decreasing_lr = '10,20'
        elif mode == 'AFF':
            args.lr = 0.1
            args.batch_size = 128
            args.epochs = 100
            args.decreasing_lr = '40,60' 
        args.bnNameCnt = 1

    ####### Hyperparameter of DynACL ########
    elif args.pretraining in ['DynACL', 'DynACL++', 'DynACL_RCS']:
        if mode in ['SLF', 'ALF']:
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.05
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 32:
                args.batch_size = 128
                args.lr = 0.01
            args.decreasing_lr = '10,20'
        elif mode == 'AFF':
            if args.dataset == 'stl10' and args.resize == 96:
                args.decreasing_lr = '10,20'
            else:
                args.decreasing_lr = '15,20'
            args.batch_size = 128
            args.lr = 0.1
        args.bnNameCnt = 1
        
    
    ####### Hyperparameter of AdvCL ########
    elif args.pretraining == 'AdvCL':
        if mode in ['SLF', 'ALF']:
            args.decreasing_lr = '15,20'
            args.batch_size = 512
            args.lr = 0.01
        elif mode == 'AFF':
            args.batch_size = 128
            args.decreasing_lr = '15,20'
            args.lr = 0.1

    ####### Hyperparameter of A-InfoNCE ########
    elif args.pretraining == 'A-InfoNCE':
        if mode in ['SLF', 'ALF']:
            args.decreasing_lr = '15,20'
            args.batch_size = 512
            args.lr = 0.01
        elif mode == 'AFF':
            args.batch_size = 128
            args.decreasing_lr = '15,20'
            args.lr = 0.1

    ####### Hyperparameter of DeACL ########
    elif args.pretraining == 'DeACL':
        if mode in ['SLF', 'ALF']:
            args.decreasing_lr = '15,20'
            args.batch_size = 512
            args.lr = 0.01
        elif mode == 'AFF':
            args.batch_size = 128
            args.decreasing_lr = '15,20'
            args.lr = 0.1

    ####### Hyperparameter of ACL_IR ########
    elif args.pretraining == 'ACL_IR':
        if mode == 'SLF':
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.005
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'stl10' and args.resize == 32:
                args.batch_size = 128
                args.lr = 0.01
            args.decreasing_lr = '10,20'
        elif mode == 'ALF':
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'stl10' and args.resize == 32:
                args.batch_size = 128
                args.lr = 0.01 
            args.decreasing_lr = '10,20'
        elif mode == 'AFF':
            args.lr = 0.1
            args.batch_size = 128
            args.epochs = 100
            args.decreasing_lr = '40,60' 
        args.bnNameCnt = 1   

    ####### Hyperparameter of DynACL_IR ########
    elif args.pretraining == 'DynACL-AIR':
        if mode == 'SLF':
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
                args.epochs = 5
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 32:
                args.batch_size = 128
                args.lr = 0.01 
            args.decreasing_lr = '10,20'
        elif mode == 'ALF':
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 32:
                args.batch_size = 128
                args.lr = 0.01 
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 128
                args.lr = 0.1 
            args.decreasing_lr = '10,20'
        elif mode == 'AFF':
            args.batch_size = 128
            args.decreasing_lr = '15,20'
            args.lr = 0.1
        args.bnNameCnt = 1

    ####### Hyperparameter of DynACL++_IR ########
    elif args.pretraining == 'DynACL-AIR++':
        if mode == 'SLF':
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.09
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 512
                args.lr = 0.1
            args.decreasing_lr = '10,20'
        elif mode == 'ALF':
            if args.dataset == 'cifar10':
                args.batch_size = 512
                args.lr = 0.01
                args.epochs = 4
            if args.dataset == 'cifar100':
                args.batch_size = 512
                args.lr = 0.1
            if args.dataset == 'stl10' and args.resize == 96:
                args.batch_size = 128
                args.lr = 0.1 
            args.decreasing_lr = '10,20'
        elif mode == 'AFF':
            if args.dataset == 'cifar10':
                args.batch_size = 128
                args.decreasing_lr = '15,20'
                args.lr = 0.1
                args.epoch = 19
            else:
                args.batch_size = 128
                args.decreasing_lr = '15,20'
                args.lr = 0.1
        args.bnNameCnt = 0

    else:
        args.batch_size = 128
        args.decreasing_lr = '15,20'
        args.lr = 0.1

    return args

