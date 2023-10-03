"""
this code is modified from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

original author: Utku Ozbulak - github.com/utkuozbulak
"""
import sys
sys.path.append("..")

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.utils import tensor2cuda

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)

        dist = dist.view(x.shape[0], -1)

        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)

        # dist = F.normalize(dist, p=2, dim=1)

        dist = dist / dist_norm

        dist *= epsilon

        dist = dist.view(x.shape)

        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x

class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model
        # self.model.eval()

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True 

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        self.model.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                if len(labels.size())==1:
                    outputs = self.model(x)
                    loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
                if len(labels.size())==2:
                    outputs = self.model.forward_IN(x)
                    loss = -torch.mean(torch.sum(torch.log(F.softmax(outputs,dim=1)) * labels, dim=1))

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data) 

                # the adversaries' pixel value should within max_x and min_x due 
                # to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)
                # the adversaries' value should be valid pixel value
                x.clamp_(self.min_val, self.max_val)

        self.model.train()

        return x
    
    def perturb_IN(self, original_images, labels, reduction4loss='mean', random_start=False):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        self.model.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                if len(labels.size())==1:
                    outputs = self.model.forward_IN(x)
                    loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
                if len(labels.size())==2:
                    outputs = self.model.forward_IN(x)
                    loss = -torch.mean(torch.sum(torch.log(F.softmax(outputs,dim=1)) * labels, dim=1))

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))

                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data)

                # the adversaries' pixel value should within max_x and min_x due 
                # to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)
                # the adversaries' value should be valid pixel value
                x.clamp_(self.min_val, self.max_val)

        self.model.train()

        return x

    def perturb_fat(self, data, target, reduction4loss='mean', random_start=False, s=32):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        K = self.max_iters
        count = 0
        output_target = []
        output_adv = []
        output_natural = []
        tau = 0
        omega = 0.0
        control = (torch.ones(len(target)) * tau).cuda()

        # Initialize the adversarial data with random noise
        if random_start:
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        else:
            iter_adv = data.cuda().detach()

        iter_clean_data = data.cuda().detach()
        iter_target = target.cuda().detach()
        output_iter_clean_data = self.model(data)

        # x.requires_grad = True 

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        self.model.eval()

        while K>0:
            iter_adv.requires_grad_()
            output = self.model(iter_adv)
            pred = output.max(1, keepdim=True)[1]
            output_index = []
            iter_index = []

            # Calculate the indexes of adversarial data those still needs to be iterated
            for idx in range(len(pred)):
                if pred[idx] != iter_target[idx]:
                    if control[idx] == 0:
                        output_index.append(idx)
                    else:
                        control[idx] -= 1
                        iter_index.append(idx)
                else:
                    iter_index.append(idx)

            # Add adversarial data those do not need any more iteration into set output_adv
            if len(output_index) != 0:
                if len(output_target) == 0:
                    # incorrect adv data should not keep iterated
                    output_adv = iter_adv[output_index].reshape(-1, 3, s, s).cuda()
                    output_natural = iter_clean_data[output_index].reshape(-1, 3, s, s).cuda()
                    output_target = iter_target[output_index].reshape(-1).cuda()
                else:
                    # incorrect adv data should not keep iterated
                    output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, s, s).cuda()), dim=0)
                    output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, s, s).cuda()), dim=0)
                    output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)

            # calculate gradient
            self.model.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            loss_adv.backward(retain_graph=True)
            grad = iter_adv.grad

            # update iter adv
            if len(iter_index) != 0:
                control = control[iter_index]
                iter_adv = iter_adv[iter_index]
                iter_clean_data = iter_clean_data[iter_index]
                iter_target = iter_target[iter_index]
                output_iter_clean_data = output_iter_clean_data[iter_index]
                grad = grad[iter_index]
                eta = self.alpha * grad.sign()

                iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
                iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - self.epsilon), iter_clean_data + self.epsilon)
                iter_adv = torch.clamp(iter_adv, 0, 1)
                count += len(iter_target)
            else:
                output_adv = output_adv.detach()
                self.model.train()
                return output_adv, output_target, output_natural

            K = K-1

            if len(output_target) == 0:
                output_target = iter_target.reshape(-1).squeeze().cuda()
                output_adv = iter_adv.reshape(-1, 3, s,s).cuda()
                output_natural = iter_clean_data.reshape(-1, 3, s,s).cuda()
            else:
                output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, s,s)), dim=0).cuda()
                output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
                output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, s,s).cuda()),dim=0).cuda()

            output_adv = output_adv.detach()
    # return output_adv, output_target, output_natural, count

        self.model.train()

        return output_adv, output_target, output_natural

    
