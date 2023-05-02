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


def pgd_attack(model, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    # init
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


def eval_adv_test(model, device, test_loader, epsilon, alpha, criterion, log, attack_iter=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input_adv = pgd_attack(model, input, target, device,
                               eps=epsilon, iters=attack_iter, alpha=alpha).data

        # compute output
        with torch.no_grad():
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


def eval_adv_test_dist(model, device, test_loader, epsilon, alpha, criterion, log, world_size, attack_iter=40, randomInit=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.cuda(
            non_blocking=True), target.cuda(non_blocking=True)
        input_adv = pgd_attack(model, input, target, device, eps=epsilon,
                               iters=attack_iter, alpha=alpha, randomInit=randomInit).data

        # compute output
        output = model(input_adv)
        output_list = [torch.zeros_like(output) for _ in range(world_size)]
        torch.distributed.all_gather(output_list, output)
        output = torch.cat(output_list)

        target_list = [torch.zeros_like(target) for _ in range(world_size)]
        torch.distributed.all_gather(target_list, target)
        target = torch.cat(target_list)

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

def trades_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, distance='l_inf'):
    batch_size = len(x_natural)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                model.eval()
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                              epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        assert False

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    model.train()
    # calculate robust loss
    logits = model(x_natural)

    loss = F.cross_entropy(logits, y)

    logits_adv = model(x_adv)

    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits, dim=1))
    loss += beta * loss_robust

    return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


def fix_bn(model, fixmode):
    if fixmode == 'f1':
        # fix none
        pass
    elif fixmode == 'f2':
        # fix previous three layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixmode == 'f3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


# loss
def pair_cosine_similarity(x, y=None, eps=1e-8):
    if(y == None):
        n = x.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps)
    else:
        n1 = x.norm(p=2, dim=1, keepdim=True)
        n2 = y.norm(p=2, dim=1, keepdim=True)
        return (x @ y.t()) / (n1 * n2.t()).clamp(min=eps)


def nt_xent(x, y=None, t=0.5):
    if(y != None):
        x = pair_cosine_similarity(x, y)
    else:
        # print("device of x is {}".format(x.device))
        x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()

def cvtPrevious2bnToCurrent2bn(state_dict):
    """
    :param state_dict: old state dict with bn and bn adv
    :return:
    """
    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        if ('bn1' in name) and ('adv' not in name):
            newName = name.replace('bn1.', 'bn1.bn_list.0.')
        elif ('bn1' in name) and ('adv' in name):
            newName = name.replace('bn1_adv.', 'bn1.bn_list.1.')
        elif ('bn2' in name) and ('adv' not in name):
            newName = name.replace('bn2.', 'bn2.bn_list.0.')
        elif ('bn2' in name) and ('adv' in name):
            newName = name.replace('bn2_adv.', 'bn2.bn_list.1.')
        elif ('bn.' in name):
            newName = name.replace('bn.', 'bn.bn_list.0.')
        elif ('bn_adv.' in name):
            newName = name.replace('bn_adv.', 'bn.bn_list.1.')
        elif 'bn3' in name:
            assert False
        else:
            newName = name

        print("convert name: {} to {}".format(name, newName))
        new_state_dict[newName] = value
    return new_state_dict


class augStrengthScheduler(object):
    """Computes and stores the average and current value"""

    def __init__(self, aug_dif_scheduler_strength_range, aug_dif_scheduler_epoch_range, transGeneFun):
        if ',' in aug_dif_scheduler_strength_range:
            self.aug_dif_scheduler_strength_range = list(
                map(float, aug_dif_scheduler_strength_range.split(',')))
        else:
            self.aug_dif_scheduler_strength_range = []

        if ',' in aug_dif_scheduler_epoch_range:
            self.aug_dif_scheduler_epoch_range = list(
                map(int, aug_dif_scheduler_epoch_range.split(',')))
        else:
            self.aug_dif_scheduler_epoch_range = []
        self.transGeneFun = transGeneFun
        self.epoch = 0

        assert (len(self.aug_dif_scheduler_strength_range) == 2 and len(self.aug_dif_scheduler_epoch_range) == 2) or \
               (len(self.aug_dif_scheduler_strength_range) ==
                0 and len(self.aug_dif_scheduler_epoch_range) == 0)

    def step(self):
        self.epoch += 1

        if len(self.aug_dif_scheduler_strength_range) == 0 and len(self.aug_dif_scheduler_epoch_range) == 0:
            return self.transGeneFun(1.0)
        else:
            startStrength, endStrength = self.aug_dif_scheduler_strength_range
            startEpoch, endEpoch = self.aug_dif_scheduler_epoch_range
            strength = min(max(0, self.epoch - startEpoch), endEpoch - startEpoch) / (
                endEpoch - startEpoch) * (endStrength - startStrength) + startStrength
            return self.transGeneFun(strength)

# new_state_dict = cvtPrevious2bnToCurrent2bn(checkpoint['state_dict'])
# model.load_state_dict(new_state_dict)

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = torch.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask.cuda()

def generate_high(Images, r):
    # Image: bsxcxhxw, input batched images
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * (1.-mask)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd

def trades_loss_dual(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, distance='l_inf', natural_mode='pgd'):
    batch_size = len(x_natural)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                model.eval()
                loss_kl = criterion_kl(F.log_softmax(model(x_adv,'pgd'), dim=1),
                                       F.softmax(model(x_natural,natural_mode), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                              epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        assert False

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    model.train()
    # calculate robust loss
    logits = model(x_natural, natural_mode)
    loss = F.cross_entropy(logits, y)
    logits_adv = model(x_adv, 'pgd')
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits, dim=1))
    loss += beta * loss_robust

    return loss

def eval_test(model, device, loader, log, advFlag = 'pgd'):
    model.eval()
    test_loss = 0
    correct = 0
    whole = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if advFlag is not None:
                output = model.eval()(data, 'pgd')
            else:
                output = model.eval()(data)
            test_loss += F.cross_entropy(output,
                                         target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)
    test_loss /= len(loader.dataset)
    log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, whole,
        100. * correct / whole))
    test_accuracy = correct / whole
    return test_loss, test_accuracy * 100

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * (step + 1) / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step -
                                                             warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

def setup_hyperparameter(args, mode):
    if mode == 'SLF' or mode == 'ALF':
        if args.dataset == 'cifar10':
            args.lr = 0.01
        if args.dataset == 'cifar100':
            args.lr = 0.005
        if args.dataset == 'stl10' and args.resize == 96:
            args.lr = 0.1
        if args.dataset == 'stl10' and args.resize == 32:
            args.lr = 0.01
    else:
        args.lr = 0.1
    return args

import torchvision
from torchvision import datasets, transforms
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
    elif args.dataset == 'cifar100':
        train_datasets = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_train)
        vali_datasets = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif args.dataset == 'stl10':
        train_datasets = torchvision.datasets.STL10(
            root=args.data, split='train', transform=transform_train, download=True)
        vali_datasets = datasets.STL10(
            root=args.data, split='train', transform=transform_test, download=True)
        testset = datasets.STL10(
            root=args.data, split='test', transform=transform_test, download=True)
        num_classes = 10     
    else:
        print("dataset {} is not supported".format(args.dataset))
        assert False

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)

    vali_loader = torch.utils.data.DataLoader(vali_datasets, batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader( testset, batch_size=args.test_batch_size, shuffle=True)

    return train_loader, vali_loader, test_loader, num_classes


import torch.optim as optim
def get_model(args, num_classes, mode, log, device='cuda'):
    if args.dataset == 'stl10' and args.resize == 96:
        from models.resnet_stl import resnet18, resnet34, resnet50
    else:
        from models.resnet import resnet18, resnet34, resnet50
    if args.model == 'r18':
        model = resnet18(num_classes=num_classes).to(device)
    if args.model == 'r34':
        model = resnet34(num_classes=num_classes).to(device)
    if args.model == 'r50':
        model = resnet50(num_classes=num_classes).to(device)
        model = torch.nn.DataParallel(model)
    if mode == 'ALF' or mode == 'SLF':
        for name, param in model.named_parameters():
            if args.model == 'r50':
                if name not in ['module.fc.weight', 'module.fc.bias']:
                    param.requires_grad = False
            else:
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
    if mode == 'AFF':
        parameters = model.parameters()


    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = args.start_epoch

    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if mode == 'SLF' or mode == 'ALF':
            state_dict = cvt_state_dict(
                state_dict, args, num_classes=num_classes)
        elif not args.eval_only:
            # zero init fc
            state_dict['fc.weight'] = torch.zeros(num_classes, 512).to(device)
            state_dict['fc.bias'] = torch.zeros(num_classes).to(device)

        model.load_state_dict(state_dict, strict=False)
        log.info('read checkpoint {}'.format(args.checkpoint))

    elif args.resume:
        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range(start_epoch):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(
                args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    return model, optimizer, scheduler

def train(args, model, device, train_loader, optimizer, epoch, log, mode='ALF'):
    # model.train()
    if mode == 'SLF' or mode == 'ALF':
        model.eval()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        print(len(parameters))
        assert len(parameters) == 2  # fc.weight, fc.bias
    if mode == 'AFF':
        model.train()

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    criterion = torch.nn.CrossEntropyLoss().cuda()

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
            loss = trades_loss_dual(model=model,
                                   x_natural=data,
                                   y=target,
                                   optimizer=optimizer,
                                   step_size=args.step_size,
                                   epsilon=args.epsilon,
                                   perturb_steps=args.num_steps_train,
                                   natural_mode='normal')

        loss.backward()
        optimizer.step()

        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % 10 == 0:
            log.info('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
                mode, epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))


def train_loop(args, model, optimizer, scheduler, train_loader, test_loader, mode, device, log, model_dir):
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        log.info("current lr is {}".format(
            optimizer.state_dict()['param_groups'][0]['lr']))

        # linear classification
        train(args, model, device, train_loader, optimizer, epoch, log, mode)
        scheduler.step()

         # evaluation
        if (not args.test_frequency == 0) and (epoch % args.test_frequency == 1 or args.test_frequency == 1):
            print('================================================================')
            eval_test(model, device, test_loader, log)
            vali_atacc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
            if vali_atacc > best_atacc:
                best_atacc = vali_atacc
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, os.path.join(model_dir, 'model_bestAT.pt'))
            print('================================================================')

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, os.path.join(model_dir, '{}_model_finetune.pt'.format(mode)))

def cvt_state_dict(state_dict, args, num_classes):
    # deal with adv bn
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

    # zero init fc
    state_dict_new['fc.weight'] = torch.zeros(num_classes, 512).cuda()
    state_dict_new['fc.bias'] = torch.zeros(num_classes).cuda()

    return state_dict_new



from robustbench.data import load_cifar10c,load_cifar100c
# from robustbench.utils import clean_accuracy
import math
def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,
                   advFlag='pgd'):
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
                output = model.eval()(x_curr, 'pgd')
            else:
                output = model.eval()(x_curr)
            # output = model(x_curr,advFlag=advFlag)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

def eval_test_nat(model, test_loader, device, advFlag='pgd'):
    torch.manual_seed(1)
    model.eval()
    acc = 0.
    for images, labels in test_loader:
         images = images.to(device)
         labels = labels.to(device)
         with torch.no_grad():
            if advFlag is not None:
                output = model.eval()(images, 'pgd')
            else:
                output = model.eval()(images)
            acc += (output.max(1)[1] == labels).float().sum()
    print(acc, len(test_loader.dataset))
    return acc.item() / len(test_loader.dataset)

def eval_test_OOD(model, test_loader, log, device, advFlag='pgd'):
    CORRUPTIONS = ["shot_noise", "gaussian_noise", "impulse_noise","contrast",
                   "zoom_blur", "motion_blur", "glass_blur", "defocus_blur",
                   "snow", "pixelate", "brightness", "fog", 
                    "jpeg_compression", "elastic_transform", "frost"]
    # fix random seed for testing
    torch.manual_seed(1)
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

from autoattack import AutoAttack
def runAA(model, test_loader, log_path, advFlag='pgd'):
    model.eval()
    acc = 0.
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', log_path=log_path)
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        xadv = adversary.run_standard_evaluation(images, labels, bs=256)
        with torch.no_grad():
            if advFlag is not None:
                output = model.eval()(xadv, 'pgd')
            else:
                output = model.eval()(xadv)
        acc += (output.max(1)[1] == labels).float().sum()
    return acc.item() / len(test_loader.dataset)


