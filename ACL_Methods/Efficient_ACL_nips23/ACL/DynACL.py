import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
from models.resnet_multi_bn import resnet18, proj_head
from utils import *
import torchvision.transforms as transforms
import os
import numpy as np
from optimizer.lars import LARS
import datetime
from coreset_util import RCS

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str, help='location for saving trained models')
parser.add_argument('--data', type=str, default='./data', help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to be used, (cifar10 or cifar100 or stl10)')
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
parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--fre', type=int, default=20, help='')
parser.add_argument('--warmup', type=int, default=100, help='')
parser.add_argument('--fraction', type=float, default=0.2, help='')
parser.add_argument('--CoresetLoss', type=str, default='KL', help='if specified, use pgd dual mode,(cal both adversarial and clean)', choices=['KL', 'JS', 'ot'])
parser.add_argument('--Coreset_pgd_iter',  default=3, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--Coreset_lr',  default=0.01, type=float, help='how many iterations employed to attack the model')


from torch.utils.data import Dataset
from typing import TypeVar, Sequence
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, STL10
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

class CustomCIFAR10(CIFAR10):
    def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

        if labelSubSet is not None:
            self.data = self.data[labelSubSet]

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[index]
            return torch.stack(imgs), imgLabelTrans, label

class CustomCIFAR100(CIFAR100):
    def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

        if labelSubSet is not None:
            self.data = self.data[labelSubSet]

    def __getitem__(self, index):
        # to return a PIL Image
        img = self.data[index]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            # return self.transform(img), self.transform(img)
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[index]
            return torch.stack(imgs), imgLabelTrans, label
            
class CustomSTL10(STL10):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        self.ori_transfor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            assert False

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    return lr

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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    save_dir = os.path.join('checkpoints_valtest', args.experiment)
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
    model = resnet18(pretrained=False, bn_names=bn_names)

    ch = model.fc.in_features
    model.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    model.cuda()
    cudnn.benchmark = True

    strength = 1.0
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(1.0 - 0.9 * strength, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset process
    if args.dataset == 'cifar10':
        train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True)
        val_dataset = CustomCIFAR10(root=args.data, train=True, transform=tfs_test, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True)
        val_dataset = CustomCIFAR100(root=args.data, train=True, transform=tfs_test, download=True)
        num_classes = 100
    elif args.dataset == 'stl10':
        train_datasets = CustomSTL10(root=args.data, split='unlabeled', transform=tfs_train, download=True)
        val_dataset = CustomSTL10(root=args.data, split='unlabeled', transform=tfs_train, download=True)
        num_classes = 10
    else:
        print("unknow dataset")
        assert False

    full_indices = np.arange(0,len(train_datasets),1)
    train_indices = np.random.choice(len(train_datasets), size=int(len(train_datasets) * 0.99), replace=False)
    val_indices = np.delete(full_indices, train_indices)
    validation_datasets = Subset(val_dataset, val_indices)
    train_datasets = Subset(train_datasets, train_indices)
    print(len(train_datasets), len(validation_datasets))

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)
    
    if args.CoresetLoss == 'ot':
        args.batch_size = 256
        validation_loader = torch.utils.data.DataLoader(
                validation_datasets,
                num_workers=4,
                batch_size=args.batch_size,
                shuffle=False)
    else:
        validation_loader = torch.utils.data.DataLoader(
                validation_datasets,
                num_workers=4,
                batch_size=args.batch_size,
                shuffle=False)

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
    if args.resume:
        if args.checkpoint != '':
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['state_dict'])

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    coreset_class = RCS(train_datasets, fraction=args.fraction, validation_loader=validation_loader, model=model, args=args, log=log)
    
    valid_loss_list = []
    test_loss_list = []
    for epoch in range(start_epoch, args.epochs + 1):
        starttime = datetime.datetime.now()
        K = 50
        strength = 1 - int((epoch-1)/K) * K/args.epochs
        tfs_train = get_transform(strength)
        train_loader = get_trainset(args, strength)
        coreset_class.dataset = train_loader.dataset
        if epoch >= args.warmup and (epoch-1) % args.fre == 0:
            tmp_state_dict = model.state_dict()
            coreset_class.model.load_state_dict(tmp_state_dict)
            coreset_class.lr = args.Coreset_lr
            train_loader = coreset_class.get_subset_loader()
            model.load_state_dict(tmp_state_dict)
            for param in model.parameters():
                param.requires_grad = True
        elif epoch > args.warmup:
            train_loader = coreset_class.load_subset_loader()
            log.info('train on the previously selected subset')
        else:
            log.info('train on the entire set')
        
        if args.scheduler == 'cosine' and epoch >=2:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(step,
                                                        args.epochs * len(train_loader),
                                                        1,  # since lr_lambda computes multiplicative factor
                                                        1e-6 / args.lr,
                                                        warmup_steps=10 * len(train_loader))
            )
            for i in range((epoch - 1) * len(train_loader)):
                scheduler.step()
                
        train_loss = train(train_loader, model, optimizer, scheduler, epoch, log, num_classes=num_classes, strength=strength)
        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            'valid_loss_list': valid_loss_list,
            'test_loss_list': test_loss_list,
        }, filename=os.path.join(save_dir, 'model.pt'))

        log.info('[Epoch: {}] [Train loss: {}] [Train time: {}]'.format(epoch, train_loss, time))

def train(train_loader, model, optimizer, scheduler, epoch, log, num_classes, strength):

    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()

        if not args.ACL_DS:
            features = model.train()(inputs, 'normal')
            loss = nt_xent(features)
        else:
            inputs_adv = PGD_contrastive(model, inputs, iters=args.pgd_iter, singleImg=False)
            features_adv = model.train()(inputs_adv, 'pgd')
            features = model.train()(inputs, 'normal')
            LAMBDA = 2/3
            weight = LAMBDA * (1 - strength)
            # print(weight)
            loss = ((1 - weight) * nt_xent(features) + (1 + weight) * nt_xent(features_adv))/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # break

        # torch.cuda.empty_cache()
        print_freq = max(int(len(train_loader) / 5), 1)
        print(print_freq)
        if i % print_freq == 0:
        # if i % 1 == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f}\t'
                     'iter_train_time: {train_time.avg:.2f}\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))

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
        # project half of the delta to be zero
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
        # print("loss is {}".format(loss))

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


