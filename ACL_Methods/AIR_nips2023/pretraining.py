### Implementation is based on ACL https://github.com/VITA-Group/Adversarial-Contrastive-Learning ###

import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import *
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from PIL import Image
from optimizer.lars import LARS
from losses import ACL_IR_Loss

parser = argparse.ArgumentParser(description='PyTorch Self-Supervised Robust Training with Adversarial Invariant Regularization')
parser.add_argument('experiment', type=str, help='location for saving trained models')
parser.add_argument('--data', type=str, default='../data', help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to be used, (cifar10 or cifar100)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', action='store_true', help='if resume training')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--ACL_DS', action='store_true', help='if specified, use pgd dual mode,(cal both adversarial and clean)')
parser.add_argument('--twoLayerProj', action='store_true', help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model', default='r18', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--eps', default=8.0, type=float)
parser.add_argument('--DynAug', action='store_true', help='whether to use DynACL')

parser.add_argument('--lambda1', default=0.5, type=float, help='weight of SIR')
parser.add_argument('--lambda2', default=0.5, type=float, help='weight of AIR')

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

class CustomCIFAR10(CIFAR10):
    def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

        if labelSubSet is not None:
            self.data = self.data[labelSubSet]
        self.ori_transfor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs), self.ori_transfor(img)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label


class CustomCIFAR100(CIFAR100):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        self.ori_transfor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs),self.ori_transfor(img)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label

class CustomSTL10(STL10):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        self.ori_transfor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs), self.ori_transfor(img)
        else:
            assert False

def get_transform(strength):
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(96 if args.dataset == 'stl10' else 32, scale=(1.0 - 0.9 * strength, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])
    return tfs_train

def get_trainset(args, strength):
    tfs_train = get_transform(strength)
    if args.dataset == 'cifar10':
        train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True)
    elif args.dataset == 'cifar100':
        train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True)
    elif args.dataset == 'stl10':
        train_datasets = CustomSTL10(root=args.data, split='unlabeled', transform=tfs_train, download=True)
    else:
        print("unknow dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)
    return train_loader

def main():
    global args
    args = parser.parse_args()

    assert args.dataset in ['cifar100', 'cifar10', 'stl10']

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert args.dataset in ['cifar100', 'cifar10', 'stl10']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    setup_seed(args.seed)

    # different attack corresponding to different bn settings
    if not args.ACL_DS:
        bn_names = ['normal', ]
    else:
        bn_names = ['normal', 'pgd']

    # define model
    if args.dataset != 'stl10':
        from models.resnet_multi_bn import resnet18, proj_head, resnet34, resnet50
    else:
        from models.resnet_multi_bn_stl import resnet18, proj_head, resnet34, resnet50
    if args.model == 'r18':
        model = resnet18(pretrained=False, bn_names=bn_names)
    if args.model == 'r34':
        model = resnet34(pretrained=False, bn_names=bn_names)
    if args.model == 'r50':
        model = resnet50(pretrained=False, bn_names=bn_names)
    
    ch = model.fc.in_features
    model.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    if args.model == 'r50':
        model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    strength = 1.0
    tfs_train = get_transform(strength)
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset process
    if args.dataset == 'cifar10':
        train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True)
    elif args.dataset == 'cifar100':
        train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True)
    elif args.dataset == 'stl10':
        train_datasets = CustomSTL10(
                root=args.data, split='unlabeled', transform=tfs_train, download=True)
    else:
        print("unknow dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ], gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=10 * len(train_loader))
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    
    for epoch in range(start_epoch, args.epochs + 1):
        if args.DynAug:
            K = 50
            strength = 1 - int((epoch-1)/K) * K/args.epochs
        else:
            strength = 1
        train_loader = get_trainset(args, strength)

        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        train(train_loader, model, optimizer, scheduler, epoch, log, strength,LAMBDA=2/3)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, filename=os.path.join(save_dir, 'model.pt'))

        if epoch % 100 == 0 and epoch > 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))


import datetime
def train(train_loader, model, optimizer, scheduler, epoch, log, strength, LAMBDA):

    losses = AverageMeter()
    losses.reset()
    nat_contrastive_losses = AverageMeter()
    nat_contrastive_losses.reset()
    adv_contrastive_losses = AverageMeter()
    adv_contrastive_losses.reset()
    SIR_losses = AverageMeter()
    SIR_losses.reset()
    AIR_losses = AverageMeter()
    AIR_losses.reset()
    starttime = datetime.datetime.now()

    for i, (inputs, ori_inputs) in enumerate(train_loader):
        scheduler.step()
        d = inputs.size()
        print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
        ori_inputs = ori_inputs.cuda()
        if not args.ACL_DS:
            features = model.train()(inputs, 'normal')
            loss = nt_xent(features)
        else:
            inputs_adv = PGD_contrastive(model, inputs, eps=args.eps/255, alpha=(args.eps/4)/255, iters=args.pgd_iter, singleImg=False)
            features_adv = model.train()(inputs_adv, 'pgd')
            features = model.train()(inputs, 'normal')
            z_orig = model.train()(ori_inputs, 'normal')
            loss_fn = ACL_IR_Loss(normalize=True, temperature=0.5, lambda1=args.lambda1, lambda2=args.lambda2)
            zi = features[[int(2 * i) for i in range(d[0])]]
            zj = features[[int(2 * i + 1) for i in range(d[0])]]
            zi_adv = features_adv[[int(2 * i) for i in range(d[0])]]
            zj_adv = features_adv[[int(2 * i + 1) for i in range(d[0])]]

            if args.DynAug:
                weight = LAMBDA * (1 - strength)
            else:
                weight = 0

            loss,l1,l2,l3,l4 = loss_fn(zi,zj,z_orig,zi_adv,zj_adv,weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu()), inputs.shape[0])
        nat_contrastive_losses.update(float(l1), inputs.shape[0])
        adv_contrastive_losses.update(float(l2), inputs.shape[0])
        SIR_losses.update(float(l3), inputs.shape[0])
        AIR_losses.update(float(l4), inputs.shape[0])

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'nat_CL_Loss {nat_contrastive_loss.val:.4f} ({nat_contrastive_loss.avg:.4f})\t'
                     'adv_CL_Loss {adv_contrastive_loss.val:.4f} ({adv_contrastive_loss.avg:.4f})\t'
                     'SIR_Loss {SIR_loss.val:.4f} ({SIR_loss.avg:.4f})\t'
                     'AIR_Loss {AIR_loss.val:.4f} ({AIR_loss.avg:.4f})\t'.format(
                          epoch, i, len(train_loader), loss=losses,nat_contrastive_loss=nat_contrastive_losses,
                          adv_contrastive_loss=adv_contrastive_losses, SIR_loss = SIR_losses, AIR_loss = AIR_losses))

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    log.info('Train time total: {}'.format(time))

    return losses.avg


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def PGD_contrastive(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    if singleImg:
        idx = [i for i in range(1, delta.data.shape[0], 2)]
        delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    for i in range(iters):
        if feature_gene is None:
            if sameBN:
                features = model.eval()(inputs + delta, 'normal')
            else:
                features = model.eval()(inputs + delta, 'pgd')
        else:
            features = feature_gene(model, inputs + delta, 'eval')

        model.zero_grad()
        loss = nt_xent(features)
        loss.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

        if singleImg:
            # project half of the delta to be zero
            idx = [i for i in range(1, delta.data.shape[0], 2)]
            delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    return (inputs + delta).detach()


if __name__ == '__main__':
    main()


