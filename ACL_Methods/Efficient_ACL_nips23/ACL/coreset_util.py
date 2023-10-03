from operator import index
from pickletools import optimize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import TypeVar, Sequence
import datetime
import torch.nn.functional as F

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class IndexSubset(Dataset[T_co]):
    dataset: Dataset[T_co]
    def __init__(self, dataset: Dataset[T_co]) -> None:
        self.dataset = dataset

    def __getitem__(self, idx):
        tmp_list = []
        tmp_list.append(idx)
        tmp_list.append(self.dataset[idx])
        return tmp_list

    def __len__(self):
        return len(self.dataset)
        
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)

def nt_xent(x, t=0.5, weight=None):
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
    return -torch.log(x)

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def normalize(AA):
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    return AA

def JS_loss(P, Q, reduction='mean'):
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    P = F.softmax(P, dim=1)
    Q = F.softmax(Q, dim=1)
    M = 0.5 * (P + Q)
    return 0.5 * (kld(torch.log(P), M) + kld(torch.log(Q), M))
    
def kl_loss(nat, adv, reduction='mean'):
    P = torch.log(normalize(adv) + 1e-8)
    Q = normalize(nat) + 1e-8
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    return kld(P, Q)

import ot
def ot_loss(P, Q, reduction='sum'):
    batch_size = P.size(0)
    m = batch_size
    n = batch_size
    loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, F.softmax(P, dim=1),F.softmax(Q, dim=1), None, None, 0.01, m, n)
    return loss
    
def PGD(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False, loss_type='JS'):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    if singleImg:
        # project half of the delta to be zero
        idx = [i for i in range(1, delta.data.shape[0], 2)]
        delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    nat_feature = model.eval()(inputs, 'normal')
    nat_feature = nat_feature.detach()

    for i in range(iters):
        features = model.eval()(inputs + delta, 'pgd')

        model.zero_grad()
        if loss_type == 'JS':
            loss = JS_loss(nat_feature, features) 
        elif loss_type == 'KL':
            loss = kl_loss(features, nat_feature)
        elif loss_type == 'ot':
            loss = ot_loss(nat_feature, features)
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


def PGD_contrastive(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)
    alpha = 2/255 * 5 / iters

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
            features = feature_gene(model, inputs + delta)

        model.zero_grad()
        loss = torch.mean(nt_xent(features))
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

class Coreset:
    def __init__(self, full_data, fraction, log, args) -> None:
        super(Coreset, self).__init__()
        self.dataset = full_data
        self.len_full = len(full_data)
        self.fraction = fraction
        self.budget = int(self.len_full * self.fraction)
        self.indices = np.random.choice(self.len_full, size=self.budget, replace=False)
        self.subset_loader = None
        self.log = log
        self.args = args
    
    def update_subset_indice(self):
        pass 

    def get_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        self.log.info('begin subset selection')
        starttime = datetime.datetime.now()
        self.indices = self.update_subset_indice()
        self.subset_loader = DataLoader(Subset(self.dataset, self.indices), num_workers=4,batch_size=self.args.batch_size,shuffle=True, pin_memory=True)
        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        self.log.info('finish subset selection. subset train number:{} \t spent time: {}'.format(len(self.indices), time))
        return self.subset_loader
    
    def load_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        self.log.info('begin load a subset list')
        self.subset_loader = DataLoader(Subset(self.dataset, self.indices), num_workers=4,batch_size=self.args.batch_size,shuffle=True, pin_memory=True)
        self.log.info('finish load a subset list. subset train number: {} \t '.format(len(self.indices)))
        return self.subset_loader

class RCS(Coreset):
    def __init__(self, full_data, fraction, log,  args, validation_loader, model) -> None:
        super().__init__(full_data, fraction, log, args)
        self.validation_loader = validation_loader
        self.model = model
        self.lr = 0.001
        if self.args.CoresetLoss == 'JS':
            self.loss_fn = JS_loss
        elif self.args.CoresetLoss == 'KL':
            self.loss_fn = kl_loss
        elif self.args.CoresetLoss == 'ot':
            self.loss_fn = ot_loss

    def update_subset_indice(self):
        # initialize validation dataset
        self.log.info('use {} loss!'.format(self.args.CoresetLoss))
        valid_loss = 0
        feature_val_nat = None
        feature_val_adv = None
        for param in self.model.parameters():
            param.requires_grad = False
        linear_layer = self.model.fc
        ori_linear_layer_state_dict = linear_layer.state_dict()

        for name,param in linear_layer.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.train()
        linear_layer.train()
        for i, (valid_inputs) in enumerate(self.validation_loader):
            d = valid_inputs.size()
            
            valid_inputs = valid_inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
            valid_inputs = valid_inputs[:d[0]]
            valid_inputs_adv = PGD(self.model, valid_inputs, iters=self.args.Coreset_pgd_iter, alpha=(10./255.)/int(self.args.Coreset_pgd_iter), 
                                        singleImg=False, loss_type=self.args.CoresetLoss)
            with torch.no_grad():
                features_adv_before_fc = self.model.get_feature(valid_inputs_adv, 'pgd')
                features_before_fc = self.model.get_feature(valid_inputs, 'normal')
                if feature_val_nat is None:
                    feature_val_nat = features_before_fc.detach()
                    feature_val_adv = features_adv_before_fc.detach()
                else:
                    feature_val_nat = torch.cat([feature_val_nat, features_before_fc.detach()], dim=0)
                    feature_val_adv = torch.cat([feature_val_adv, features_adv_before_fc.detach()], dim=0)
          
        linear_layer.zero_grad()
        features = linear_layer(feature_val_nat, 'normal')
        features_adv = linear_layer(feature_val_adv, 'pgd')
        valid_loss = self.loss_fn(features_adv, features)
        valid_loss.backward()

        valid_grad_list = []
        for name,param in linear_layer.named_parameters():
            g = param.grad
            if 'fc' in name:
                print(name)
                valid_grad_list.append(g.detach().mean(dim=0).view(1, -1))
        grad_val = torch.cat(valid_grad_list, dim=1)
        
        subset_index = []
        train_loader = DataLoader(IndexSubset(self.dataset), num_workers=4,batch_size=self.args.Coreset_bs,shuffle=True)

        self.model.eval()
        linear_layer.eval()
        batch_index_list = []
        per_batch_grads = []
        per_batch_ori_grads = []
        batch_index_list = []

        # begin to find the subset in each batch
        for i, (idx, inputs) in enumerate(train_loader):
            linear_layer.load_state_dict(ori_linear_layer_state_dict)
            
            d = inputs.size()
            # print("inputs origin shape is {}".format(d))
            inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
            inputs_adv = PGD_contrastive(self.model, inputs, iters=self.args.pgd_iter, singleImg=False)
            with torch.no_grad():
                features_adv_before_fc = self.model.get_feature(inputs_adv, 'pgd').detach()
                features_before_fc = self.model.get_feature(inputs, 'normal').detach()
            # initialize the gradient of each unlabel data
            linear_layer.zero_grad()
            features = linear_layer(features_before_fc, 'normal')
            features_adv = linear_layer(features_adv_before_fc, 'pgd')
            batch_loss = (nt_xent(features) + nt_xent(features_adv))/2
            batch_loss = torch.mean(batch_loss)
            batch_loss.backward()

            batch_grad_list = []
            batch_grad_ori_list = []
            for name,param in linear_layer.named_parameters():
                g = param.grad
                if 'fc' in name:
                    batch_grad_ori_list.append(g.detach())
                    batch_grad_list.append(g.detach().mean(dim=0).view(1, -1))
            grad_batch = torch.cat(batch_grad_list, dim=1)
            
            per_batch_ori_grads.append(batch_grad_ori_list)
            per_batch_grads.append(grad_batch)
            batch_index_list.append(idx)

        per_batch_grads = torch.cat(per_batch_grads, dim=0)
        index_list = torch.LongTensor([q for q in range(len(batch_index_list))]).cuda()
        batch_num = int(self.budget / self.args.batch_size)  

        # Greedy search
        for j in range(batch_num):
            # compute the gain function
            grad_sample_list_curr = per_batch_grads[index_list]
            gain = torch.matmul(grad_sample_list_curr, grad_val.reshape(-1,1)).squeeze()
            r = torch.argmax(gain, dim=0)
            print(len(batch_index_list[index_list[r]]))
            subset_index.extend(batch_index_list[index_list[r]])

            if j == batch_num - 1:
                break

            linear_layer.fc1.weight.data = linear_layer.fc1.weight.data - (self.lr) * per_batch_ori_grads[index_list[r]][0]
            linear_layer.fc1.bias.data = linear_layer.fc1.bias.data - (self.lr) * per_batch_ori_grads[index_list[r]][1]
            linear_layer.fc2.weight.data = linear_layer.fc2.weight.data - (self.lr) * per_batch_ori_grads[index_list[r]][2]
            linear_layer.fc3.weight.data = linear_layer.fc3.weight.data - (self.lr) * per_batch_ori_grads[index_list[r]][3]

            self.model.fc.fc1.weight.data = self.model.fc.fc1.weight.data - self.lr * per_batch_ori_grads[index_list[r]][0]
            self.model.fc.fc1.bias.data = self.model.fc.fc1.bias.data - self.lr * per_batch_ori_grads[index_list[r]][1]
            self.model.fc.fc2.weight.data = self.model.fc.fc2.weight.data - self.lr * per_batch_ori_grads[index_list[r]][2]
            self.model.fc.fc3.weight.data = self.model.fc.fc3.weight.data - self.lr * per_batch_ori_grads[index_list[r]][3]

            self.model.train()
            linear_layer.train()
            feature_val_nat = None
            feature_val_adv = None
            for i, (valid_inputs) in enumerate(self.validation_loader):
                d = valid_inputs.size()
                
                valid_inputs = valid_inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
                valid_inputs = valid_inputs[:d[0]]
                valid_inputs_adv = PGD(self.model, valid_inputs, iters=self.args.Coreset_pgd_iter, alpha=(10./255.)/int(self.args.Coreset_pgd_iter), 
                                            singleImg=False, loss_type=self.args.CoresetLoss)
                with torch.no_grad():
                    features_adv_before_fc = self.model.get_feature(valid_inputs_adv, 'pgd')
                    features_before_fc = self.model.get_feature(valid_inputs, 'normal')
                    if feature_val_nat is None:
                        feature_val_nat = features_before_fc.detach()
                        feature_val_adv = features_adv_before_fc.detach()
                    else:
                        feature_val_nat = torch.cat([feature_val_nat, features_before_fc.detach()], dim=0)
                        feature_val_adv = torch.cat([feature_val_adv, features_adv_before_fc.detach()], dim=0)

            # update grad_val with the new parameter
            linear_layer.zero_grad()
            features = linear_layer(feature_val_nat, 'normal')
            features_adv = linear_layer(feature_val_adv, 'pgd')
            valid_loss = self.loss_fn(features_adv, features)
            
            linear_layer.zero_grad()
            valid_loss.backward()
            valid_grad_list = []
            for name,param in linear_layer.named_parameters():
                g = param.grad
                if 'fc' in name:
                    valid_grad_list.append(g.detach().mean(dim=0).view(1, -1))
            grad_val = torch.cat(valid_grad_list, dim=1)
            index_list = del_tensor_ele(index_list, r)

        self.model.fc.load_state_dict(ori_linear_layer_state_dict)
        return subset_index
