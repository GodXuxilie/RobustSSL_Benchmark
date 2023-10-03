from operator import index
from pickletools import optimize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import TypeVar, Sequence
import datetime
import copy
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
        # except:
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

def normalize(AA):
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    return AA

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def JS_loss(P, Q, reduction='mean'):
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    M = 0.5 * (P + Q)
    return 0.5 * (kld(P, M) + kld(Q, M))

import ot
def ot_loss(P, Q, reduction='sum'):
    batch_size = P.size(0)
    m = batch_size
    n = batch_size
    loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, F.softmax(P, dim=1), F.softmax(Q, dim=2), None, None, 0.01, m, n)
    return loss

def kl_loss(nat, adv, reduction='mean'):
    P = F.log_softmax(adv, dim=1)
    Q = F.softmax(nat, dim=1)
    kld = torch.nn.KLDivLoss(reduction=reduction).cuda()
    return kld(P,Q)

def PGD(model, data, epsilon, step_size, num_steps, loss_type='JS'):
    model.eval()
    x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
    # x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
    nat_feature = model(data, 'pgd').detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        output_feature = model(x_adv, 'pgd')
        model.zero_grad()
        with torch.enable_grad():
            if loss_type == 'JS':
                loss_adv = JS_loss(nat_feature, output_feature)
            elif loss_type == 'KL':
                loss_adv = kl_loss(nat_feature, output_feature)
            elif loss_type == 'ot':
                loss_adv = ot_loss(nat_feature, output_feature)
        # print(loss_adv)
        loss_adv.backward(retain_graph=True)
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def PGD_contrastive(model, inputs, eps=4. / 255., alpha=1. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False):
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
                features = model.eval()(inputs + delta)
            else:
                features = model.eval()(inputs + delta, 'pgd')
        else:
            features = feature_gene(model, inputs + delta)

        model.zero_grad()
        loss = nt_xent(features, t=0.1)
        loss = torch.mean(loss)
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

class Coreset:
    def __init__(self, full_data, fraction, log, args) -> None:
        super(Coreset, self).__init__()
        self.dataset = full_data
        self.len_full = len(full_data)
        self.fraction = fraction
        self.budget = int(self.len_full * self.fraction)
        self.subset_loader = None
        self.log = log
        self.args = args
        self.indices = np.random.choice(self.len_full, size=self.budget, replace=False)
        self.lr = None

    def update_subset_indice(self):
        pass 

    def get_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        self.log.info('begin subset selection')
        starttime = datetime.datetime.now()
        self.indices = self.update_subset_indice()
        self.subset_loader = DataLoader(Subset(self.dataset, self.indices), num_workers=8,batch_size=self.args.batch_size,shuffle=True, pin_memory=True)
        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        self.log.info('finish subset selection. subset train number: {} \t spent time: {}s'.format(len(self.indices), time))
        return self.subset_loader

    def load_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        self.log.info('begin load a subset list')
        self.subset_loader = DataLoader(Subset(self.dataset, self.indices), num_workers=8,batch_size=self.args.batch_size,shuffle=True, pin_memory=True)
        self.log.info('finish load a subset list. subset train number: {} \t '.format(len(self.indices)))
        return self.subset_loader

class RCS(Coreset):
    def __init__(self, full_data, fraction, log,  args, validation_loader, model) -> None:
        super().__init__(full_data, fraction, log, args)
        self.validation_loader = validation_loader
        self.model = model
        if self.args.CoresetLoss == 'JS':
            self.loss_fn = JS_loss
        elif self.args.CoresetLoss == 'KL':
            self.loss_fn = kl_loss
        elif self.args.CoresetLoss == 'ot':
            self.loss_fn = ot_loss
            
    def update_subset_indice(self):
        self.log.info('use {} loss'.format(self.args.CoresetLoss))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        linear_layer = self.model.module.linear
        state_dict_linear = linear_layer.state_dict()

        for name,param in linear_layer.named_parameters():
            param.requires_grad = True

        feature_val_nat = None
        feature_val_adv = None
        
        for i, (valid_inputs) in enumerate(self.validation_loader):
            d = valid_inputs.size()
            
            valid_inputs = valid_inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
            valid_inputs = valid_inputs[:d[0]]
            valid_inputs_adv = PGD(self.model, valid_inputs,epsilon=self.args.epsilon/255, num_steps=self.args.Coreset_num_steps,
                                    step_size=(self.args.epsilon/4)/255 * (5 / self.args.Coreset_num_steps),loss_type=self.args.CoresetLoss)
            with torch.no_grad():
                features_adv_before_linear = self.model.module.get_feature(valid_inputs_adv, 'pgd')
                features_before_linear = self.model.module.get_feature(valid_inputs, 'normal')
                if feature_val_nat is None:
                    feature_val_nat = features_before_linear.detach()
                    feature_val_adv = features_adv_before_linear.detach()
                else:
                    feature_val_nat = torch.cat([feature_val_nat, features_before_linear.detach()], dim=0)
                    feature_val_adv = torch.cat([feature_val_adv, features_adv_before_linear.detach()], dim=0)
        
        linear_layer.zero_grad()
        features = linear_layer(feature_val_nat, 'normal')
        features_adv = linear_layer(feature_val_adv, 'pgd')
        valid_loss = self.loss_fn(features, features_adv)
        valid_loss.backward()

        valid_grad_list = []
        for name, param in linear_layer.named_parameters():
            g = param.grad
            if 'fc' in name:
                print(name)
                valid_grad_list.append(g.detach().mean(dim=0).view(1, -1))
                param.grad = None
        grad_val = torch.cat(valid_grad_list, dim=1).cuda()
        ori_grad_val = copy.deepcopy(grad_val)
        
        subset_index = []
        train_loader = DataLoader(IndexSubset(self.dataset), num_workers=4,batch_size=self.args.batch_size,shuffle=True, pin_memory=True)
        
        batch_index_list = []
        per_batch_grads = []
        per_batch_ori_grads = []
        starttime = datetime.datetime.now()

        for i, (idx, inputs) in enumerate(train_loader):
            d = inputs.size()
            inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
            inputs_adv = PGD_contrastive(self.model, inputs, iters=self.args.Coreset_num_steps, singleImg=False,eps=self.args.epsilon/255,alpha=(self.args.epsilon/4)/255)
            with torch.no_grad():
                features_adv_before_fc = self.model.module.get_feature(inputs_adv, 'pgd').detach()
                features_before_fc = self.model.module.get_feature(inputs, 'normal').detach()
            linear_layer.zero_grad()
            features = linear_layer(features_before_fc, 'normal')
            features_adv = linear_layer(features_adv_before_fc, 'pgd')
            batch_loss = (nt_xent(features, t=0.1) + nt_xent(features_adv, t=0.1))/2
            batch_loss = torch.mean(batch_loss)
            batch_loss.backward()

            batch_grad_list = []
            batch_grad_ori_list = []
            for name, param in linear_layer.named_parameters():
                g = param.grad
                if 'fc' in name:
                    batch_grad_ori_list.append(g.detach())
                    batch_grad_list.append(g.detach().mean(dim=0).view(1, -1))
                    param.grad = None
            grad_batch = torch.cat(batch_grad_list, dim=1).cuda()

            per_batch_ori_grads.append(batch_grad_ori_list)
            per_batch_grads.append(grad_batch)
            batch_index_list.append(idx)

            # We conduct RCS every 100 minibatches of training data to enable RCS on large-scale datasets
            if (i+1) % 100 == 0 or (i+1) == len(train_loader):
                per_batch_grads = torch.cat(per_batch_grads, dim=0)
                index_list = torch.LongTensor([q for q in range(len(batch_index_list))]).cuda()
                batch_num = int((self.budget / self.args.batch_size) * (len(index_list) / len(train_loader)))

                # Greedy search
                for j in range(batch_num):
                    # compute the gain function
                    grad_batch_list_curr = per_batch_grads[index_list]
                    gain = torch.matmul(grad_batch_list_curr.cuda(), grad_val.reshape(-1,1).cuda()).squeeze()
                    print(gain.shape)
                    r = torch.argmax(gain, dim=0)
                    print(gain[r])
                    subset_index.extend(batch_index_list[index_list[r]])

                    if j == batch_num - 1:
                        break

                    linear_layer.fc1.weight.data = linear_layer.fc1.weight.data - (self.lr) * per_batch_ori_grads[index_list[r]][0]
                    linear_layer.fc1.bias.data = linear_layer.fc1.bias.data - (self.lr) * per_batch_ori_grads[index_list[r]][1]
                    linear_layer.fc2.weight.data = linear_layer.fc2.weight.data - (self.lr) * per_batch_ori_grads[index_list[r]][2]

                    self.model.module.linear.fc1.weight.data = self.model.module.linear.fc1.weight.data - self.lr * per_batch_ori_grads[index_list[r]][0]
                    self.model.module.linear.fc1.bias.data = self.model.module.linear.fc1.bias.data - self.lr * per_batch_ori_grads[index_list[r]][1]
                    self.model.module.linear.fc2.weight.data = self.model.module.linear.fc2.weight.data - self.lr * per_batch_ori_grads[index_list[r]][2]
        
                    feature_val_nat = None
                    feature_val_adv = None

                    for i_, (valid_inputs) in enumerate(self.validation_loader):
                        d = valid_inputs.size()
            
                        valid_inputs = valid_inputs.view(d[0]*2, d[2], d[3], d[4]).cuda()
                        valid_inputs = valid_inputs[:d[0]]
                        valid_inputs_adv = PGD(self.model, valid_inputs,epsilon=8/255, num_steps=self.args.Coreset_num_steps,
                                                step_size=2/255 * (5 / self.args.Coreset_num_steps),loss_type=self.args.CoresetLoss)
                        with torch.no_grad():
                            features_adv_before_linear = self.model.module.get_feature(valid_inputs_adv, 'pgd')
                            features_before_linear = self.model.module.get_feature(valid_inputs, 'normal')
                            if feature_val_nat is None:
                                feature_val_nat = features_before_linear.detach()
                                feature_val_adv = features_adv_before_linear.detach()
                            else:
                                feature_val_nat = torch.cat([feature_val_nat, features_before_linear.detach()], dim=0)
                                feature_val_adv = torch.cat([feature_val_adv, features_adv_before_linear.detach()], dim=0)

                    # update grad_val with the new parameter
                    linear_layer = self.model.module.linear
                    linear_layer.zero_grad()
                    features = linear_layer(feature_val_nat, 'normal')
                    features_adv = linear_layer(feature_val_adv, 'pgd')
                    valid_loss = self.loss_fn(features, features_adv)

                    print(valid_loss)
                    valid_loss.backward()
                    valid_grad_list = []
                    for name,param in linear_layer.named_parameters():
                        g = param.grad
                        if 'fc' in name:
                            print(name)
                            valid_grad_list.append(g.detach().mean(dim=0).view(1, -1))
                            param.grad = None

                    grad_val = torch.cat(valid_grad_list, dim=1).cuda()
                    index_list = del_tensor_ele(index_list, r)

                    if (i+1) % 100 == 0:
                        endtime = datetime.datetime.now()
                        time = (endtime - starttime).seconds
                        self.log.info('Batch num: {}, time: {}'.format(i, time))

                    linear_layer.load_state_dict(state_dict_linear)
                    self.model.module.linear.load_state_dict(state_dict_linear)

                    batch_index_list = []
                    per_batch_grads = []
                    per_batch_ori_grads = []
                    grad_val = copy.deepcopy(ori_grad_val)

                    print(len(subset_index), len(subset_index)/self.len_full)

        return subset_index
