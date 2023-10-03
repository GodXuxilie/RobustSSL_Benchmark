"""

OT using IPOT and Sinkhorn algorithm
Code is copied from https://github.com/Haichao-Zhang/FeatureScatter.

"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# from utils import softCrossEntropy

import numpy as np

'''Some utility functions
'''
import os
import sys
import time
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import random
import scipy.io

import torch

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


def label_smoothing(y_batch_tensor, num_classes, delta):
    y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * \
        y_batch_tensor + delta / (num_classes - 1)
    return y_batch_smooth


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param targets: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = targets.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, targets)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, targets), 1)

        return loss


class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


def sinkhorn_loss_joint_IPOT(alpha, beta, x_feature, y_feature, x_label,
                             y_label, epsilon, m, n):

    C_fea = get_cost_matrix(x_feature, y_feature)
    C = C_fea
    T = sinkhorn(C, 0.01, 100)
    # T = IPOT(C, 1)
    batch_size = C.size(0)
    cost_ot = torch.sum(T * C)
    return cost_ot


def sinkhorn(C, epsilon, niter=50, device='cuda'):
    m = C.size(0)
    n = C.size(1)
    mu = Variable(1. / m * torch.FloatTensor(m).fill_(1).to('cuda'),
                  requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1).to('cuda'),
                  requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) +
                         1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).cpu().data.numpy():
            break
    U, V = u, v

    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    pi = pi.to('cuda').float()
    return pi  # return the transport


def IPOT(cost_matrix, beta=1, device='cuda'):
    m = cost_matrix.size(0)
    n = cost_matrix.size(1)
    sigma = 1.0 / n * torch.ones([n, 1]).to(device)

    T = torch.ones([m, n]).to(device)
    A = torch.exp(-cost_matrix / beta)

    for t in range(50):
        # BUG: should be elementwise product, * in numpy
        #Q = torch.mm(A, T)
        Q = A * T  # Hardmard product
        for k in range(1):
            delta = 1.0 / (m * torch.mm(Q, sigma))
            sigma = 1.0 / (n * torch.mm(delta.t(), Q)).t()
            #sigma = 1.0 / (n * torch.mv(Q, delta))
        tmp = torch.mm(construct_diag(torch.squeeze(delta)), Q)
        T = torch.mm(tmp, construct_diag(torch.squeeze(sigma)))

    return T


def construct_diag(d):
    n = d.size(0)
    x = torch.zeros([n, n]).to(d.device)
    x[range(n), range(n)] = d.view(-1)
    return x


def get_cost_matrix(x_feature, y_feature):
    C_fea = cost_matrix_cos(x_feature, y_feature)  # Wasserstein cost function
    return C_fea


def cost_matrix_cos(x, y, p=2):
    # return the m*n sized cost matrix
    "Returns the matrix of $|x_i-y_j|^p$."
    # un squeeze differently so that the tensors can broadcast
    # dim-2 (summed over) is the feature dim
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    c = torch.clamp(1 - cos(x_col, y_lin), min=0)

    return c