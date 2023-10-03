import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import torchvision as tv
import pickle
from time import time
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model
from src.argument import parser, print_args
from transfer_utils import fine_tunify, transfer_datasets
from robustness import datasets, model_utils
from robustness import data_augmentation
import copy
from torch.utils.data import Dataset
from typing import TypeVar, Sequence
from autoattack import AutoAttack
from transfer_utils.DOG import dogs
from transfer_utils.CUB import Cub2011
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

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model, tr_loader, va_loader=None, adv_train=False, val_loader=None):
        args = self.args
        logger = self.logger
        tune_sum = 0
        _iter = 0
        test_acc_track = []
        adv_test_acc_track = []
        begin_time = time()
        best_va_adv_acc = 0.0
        criterion_kl = nn.KLDivLoss(size_average=False)
        W = torch.LongTensor([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56])
        loss_list = torch.zeros(size=(1, args.max_epoch+1)).squeeze().cuda()
        loss_min_list = torch.Tensor([0]*len(W))
        w_index = 1
        eta = 0.01
        eta_list = torch.Tensor([0]*len(W))
        eta_list[0] = eta
        max_acc = -1
        rho = 0.75

        for n,p in model.named_parameters():
            if 'bn' in n:
                p.requires_grad = True
            elif 'fc_downstream' in n:
                p.requires_grad = True
            elif 'nat' in n:
                p.requires_grad = True
                tune_sum += p.numel()
            elif 'adv' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        logger.info('tunable params: {} '.format(tune_sum))

        va_acc, va_adv_acc = self.test(model, val_loader, True, False)
        va_acc_nat_branch = self.test_nat_branch(model, tr_loader, True, False)
        LAMBDA1 = 1 - va_acc_nat_branch
        LAMBDA2 = va_acc_nat_branch * 6.0

        opt = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), eta, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)

        for epoch in range(1, args.max_epoch+1):
            logger.info('current epoch: {} lr: {} LAMBDA1: {} LAMBDA2: {}'.format(epoch, eta, LAMBDA1, LAMBDA2))
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.train()
                batch_size = len(data)
                adv_data = self.attack.perturb(data, label, 'mean', True)
                for n,p in model.named_parameters():
                    if 'bn' in n:
                        p.requires_grad = False
                    elif 'fc_downstream' in n:
                        p.requires_grad = True
                    elif 'nat' in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                nat_output, nat_latent = model([data, True, 'nat'])
                for n,p in model.named_parameters():
                    if 'bn' in n:
                        p.requires_grad = True
                    elif 'fc_downstream' in n:
                        p.requires_grad = True
                    elif 'lora' in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                output, adv_latent = model([adv_data, True, 'adv'])
                

                loss = LAMBDA1 * F.cross_entropy(nat_output, label) + \
                        (1 - LAMBDA1) * F.cross_entropy(output, label) + \
                        LAMBDA2 * (1.0 / args.batch_size) * criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
                
                for n,p in model.named_parameters():
                    if 'bn' in n:
                        p.requires_grad = True
                    elif 'fc_downstream' in n:
                        p.requires_grad = True
                    elif 'nat' in n:
                        p.requires_grad = True
                    elif 'adv' in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                        
                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    t1 = time()
                    with torch.no_grad():
                        model.eval()
                        stand_output = model([data, False, 'adv'])
                        model.train()
                    pred = torch.max(stand_output, dim=1)[1]
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                    pred = torch.max(output, dim=1)[1]
                    adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time()
                    logger.info(f'epoch: {epoch}, iter: {_iter}, lr={opt.param_groups[0]["lr"]}, '
                                f'spent {time()-begin_time:.2f} s, tr_loss: {loss.item():.3f}')
                    logger.info(f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%')
                    begin_time = time()

                _iter += 1
            
            va_acc, va_adv_acc = self.test(model, val_loader, True, False)
            va_acc_nat_branch = self.test_nat_branch(model, tr_loader, True, False)
            LAMBDA1 = 1 - va_acc_nat_branch
            LAMBDA2 = va_acc_nat_branch * 6.0
            k3 = 1 - va_acc / (1 - va_acc + 1 - va_adv_acc)



            loss_list[epoch-1] = va_adv_acc
            if  va_adv_acc > max_acc:
                loss_min_list[w_index] = va_adv_acc
                max_acc = va_adv_acc
                file_name = os.path.join(args.model_folder, f'checkpoint_best_nat_adv.pth')
                save_model(model, file_name)

            if epoch -1 == W[w_index]:
                w_j_0 = W[w_index-1].item()
                w_j_1 = W[w_index].item() 
                count = 0
                for j in range(w_j_0, w_j_1, 1):
                    if loss_list[j+1] > loss_list[j]:
                        count += 1
                if count < rho * (w_j_1 - w_j_0):
                    eta /= 2
                    eta_list[w_index] = eta
                    model.load_state_dict(torch.load(os.path.join(args.model_folder, f'checkpoint_best_nat_adv.pth')))
                    logger.info('Satisfy condition 1!')
                elif eta_list[w_index] == eta_list[w_index-1] and loss_min_list[w_index] == loss_min_list[w_index-1]:
                    eta /= 2
                    eta_list[w_index] = eta
                    model.load_state_dict(torch.load(os.path.join(args.model_folder, f'checkpoint_best_nat_adv.pth')))
                    logger.info('Satisfy condition 2!')

                if w_index < len(W) - 1:
                    w_index += 1
                    loss_min_list[w_index] = loss_min_list[w_index-1]
                    eta_list[w_index] = eta_list[w_index-1]

                for param_group in opt.param_groups:
                    param_group["lr"] = eta 

            va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0
            va_acc_nat_branch = va_acc_nat_branch * 100.0
            logger.info(f'val acc: {va_acc:.3f}%, val adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s, val nat_branch acc: {va_acc_nat_branch:.3f}%')
               
            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True, False)
                va_acc_nat_branch = self.test_nat_branch(model, va_loader, True, False)
                t2 = time()
                
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0
                va_acc_nat_branch = va_acc_nat_branch * 100.0
                logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s, test nat_branch acc: {va_acc_nat_branch:.3f}%')
                if va_adv_acc > best_va_adv_acc:
                    best_va_adv_acc = va_adv_acc
                    file_name = os.path.join(args.model_folder, f'checkpoint_best.pth')
                    save_model(model, file_name)
                logger.info('best adv acc: {}'.format(best_va_adv_acc))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')
                test_acc_track.append(va_acc)
                adv_test_acc_track.append(va_adv_acc)
                pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_track.pkl','wb'))
                pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_track.pkl','wb'))
                
                if args.save_all:
                    file_name = os.path.join(args.model_folder, 'checkpoint_epoch{}.pth'.format(epoch))
                    save_model(model, file_name)
            if eta < 0.00001:
                break
        file_name = os.path.join(args.model_folder, f'checkpoint_final.pth')
        save_model(model, file_name)

    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 
        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        model.eval()
        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                output = model([data, False, 'adv'])
                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                total_acc += te_acc
                num += output.shape[0]
                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       False)
                    model.eval()
                    adv_output = model([adv_data, False, 'adv'])
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num
        model.train()

        return total_acc / num , total_adv_acc / num
    
    def test_nat_branch(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 
        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        model.eval()
        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                output = model([data, False, 'nat'])
                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                total_acc += te_acc
                num += output.shape[0]
        model.train()
        return total_acc / num 
    
    def test_AA(self, model, test_loader, eps, log_path, bs=128, norm='Linf', version='standard'):
        model.eval()
        adversary = AutoAttack(model, norm=norm, eps=eps, log_path=log_path,version=version, verbose=True)
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)
        with torch.no_grad():
            adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
        return 0
    
import numpy as np
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_folder = '%s_%s' % (args.dataset, args.affix)

    model_folder = os.path.join(args.model_root, save_folder)
    log_folder = model_folder

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)

    model_arch = args.model_arch  
    model, _ = model_utils.make_and_restore_model(
                arch=model_arch,
                dataset=datasets.ImageNet('',r_adv=args.r_adv, r_nat=args.r_nat), resume_path=args.model_path, pytorch_pretrained=False,
                add_custom_forward=True, r=args.r)
    
    while hasattr(model, 'model'):
        model = model.model
    model = fine_tunify.ft(
                model_arch, model, args.num_classes, 0)
    ds, (_,_) = transfer_datasets.make_loaders('cifar10', batch_size=10, workers=8, subset=50000)
    if type(ds) == int:
        print('new ds')
        new_ds = datasets.CIFAR(args.data_root)
        new_ds.num_classes = ds
        new_ds.mean = torch.tensor([0., 0., 0.])
        new_ds.std = torch.tensor([1.0, 1.0, 1.0])
        ds = new_ds
    ds.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    ds.std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, add_custom_forward=True, r=args.r, parallel=True)
    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    if torch.cuda.is_available():
        model.cuda()
    trainer = Trainer(args, logger, attack)


    if args.todo == 'train':
        val = 1 - args.val
        if args.dataset == 'cifar10' :
            transform_train = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
            tr_dataset = tv.datasets.CIFAR10(args.data_root,
                                       train=True,
                                       transform=transform_train,
                                       download=True)
            val_dataset = tv.datasets.CIFAR10(args.data_root,
                                       train=True,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)
            te_dataset = tv.datasets.CIFAR10(args.data_root,
                                       train=False,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)
            
            if args.val>0:
                full_indices = np.arange(0,len(tr_dataset),1)
                train_indices = np.random.choice(len(tr_dataset), size=int(len(tr_dataset) * val), replace=False)
                val_indices = np.delete(full_indices, train_indices)
                val_dataset = Subset(val_dataset, val_indices)
                tr_dataset = Subset(tr_dataset, train_indices)
                print(len(tr_dataset), len(val_dataset))

            tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        elif args.dataset == 'cifar100' :
            transform_train = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
            tr_dataset = tv.datasets.CIFAR100(args.data_root,
                                       train=True,
                                       transform=transform_train,
                                       download=True)
            val_dataset = tv.datasets.CIFAR100(args.data_root,
                                       train=True,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)
            te_dataset = tv.datasets.CIFAR100(args.data_root,
                                       train=False,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)
            
            if args.val>0:
                full_indices = np.arange(0,len(tr_dataset),1)
                train_indices = np.random.choice(len(tr_dataset), size=int(len(tr_dataset) * val), replace=False)
                val_indices = np.delete(full_indices, train_indices)
                val_dataset = Subset(val_dataset, val_indices)
                tr_dataset = Subset(tr_dataset, train_indices)
                
            tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        elif args.dataset == 'cub':
            transform_train = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.RandomResizedCrop(224),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor()
            ])
        
            transform_test = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor()
            ])
            from CUB import Cub2011
            tr_dataset = Cub2011(root=args.data_root, train=True, download=False, transform=transform_train)
            val_dataset = Cub2011(root=args.data_root, train=True, download=False, transform=transform_test)
            te_dataset = Cub2011(root=args.data_root, train=False, download=False, transform=transform_test)

            if args.val>0:
                full_indices = np.arange(0,len(tr_dataset),1)
                train_indices = np.random.choice(len(tr_dataset), size=int(len(tr_dataset) * val), replace=False)
                val_indices = np.delete(full_indices, train_indices)
                val_dataset = Subset(val_dataset, val_indices)
                tr_dataset = Subset(tr_dataset, train_indices)

            tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        elif args.dataset == 'dog':
            transform_train = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.RandomResizedCrop(224),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor()
            ])
        
            transform_test = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor()
            ])
            
            tr_dataset = dogs(root=args.data_root,
                                 train=True,
                                 cropped=False,
                                 transform=transform_train,
                                 download=True)
            val_dataset = dogs(root=args.data_root,
                                 train=True,
                                 cropped=False,
                                 transform=transform_test,
                                 download=True)
            te_dataset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=transform_test,
                                download=True)

            if args.val>0:
                full_indices = np.arange(0,len(tr_dataset),1)
                train_indices = np.random.choice(len(tr_dataset), size=int(len(tr_dataset) * val), replace=False)
                val_indices = np.delete(full_indices, train_indices)
                val_dataset = Subset(val_dataset, val_indices)
                tr_dataset = Subset(tr_dataset, train_indices)

            tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        else:
            ds, tr_dataset, te_dataset = get_dataset_and_loaders(args)
            print(len(tr_dataset), len(te_dataset))
            val_dataset = copy.deepcopy(te_dataset)

            tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
        trainer.train(model, tr_loader, te_loader, args.adv_train, val_loader)
        
    elif args.todo == 'test':
        if args.dataset == 'cifar10' :
            te_dataset = tv.datasets.CIFAR10(args.data_root,
                                       train=False,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        elif args.dataset == 'cifar100' :
            te_dataset = tv.datasets.CIFAR100(args.data_root,
                                       train=False,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        elif args.dataset == 'cub':
            transform_test = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor()
            ])
            te_dataset = Cub2011(root=args.data_root, train=False, download=False, transform=transform_test)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        elif args.dataset == 'dog':
            transform_test = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor()
            ])
            te_dataset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=transform_test,
                                download=True)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        else:
            ds, tr_dataset, te_dataset = get_dataset_and_loaders(args)
            te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        sd= torch.load(args.load_checkpoint)
        model.load_state_dict(sd)
        log_path = log_folder + '/robustness.txt'
        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)
        logger.info(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")
        trainer.test_AA(model, te_loader, eps=0.0314, log_path=log_path, bs=args.batch_size, norm='Linf', version='standard')
    else:
        raise NotImplementedError


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, 8, subset=0)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = torch.tensor([0., 0., 0.])
            new_ds.std = torch.tensor([1., 1., 1.])
            ds = new_ds
    return ds, train_loader, validation_loader

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
