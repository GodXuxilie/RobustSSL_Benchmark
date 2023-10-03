import numpy as np
from autoattack import AutoAttack
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def pgd(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_logit = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=True).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1), F.softmax(nat_logit, dim=1))
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
        loss_adv.backward(retain_graph=True)
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def multitarget_pgd(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_logit = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    N = int(num_steps / 9)

    for i in range(N):
        for j in range(9):
            x_adv.requires_grad_()
            output = model(x_adv)
            model.zero_grad()
            with torch.enable_grad():
                index = (i*9+j)%9
                temp_target = torch.Tensor([index if target[k] != index else index + 1 for k in range(len(target))]).long().cuda()
                target = target.data
                target_onehot = torch.zeros(target.size() + (10,))
                target_onehot = target_onehot.cuda()
                target_onehot.scatter_(1, target.unsqueeze(1), 1.)
                target_var = Variable(target_onehot, requires_grad=False)
                real = (target_var * output).sum(1)

                target = temp_target.data
                target_onehot = torch.zeros(target.size() + (10,))
                target_onehot = target_onehot.cuda()
                target_onehot.scatter_(1, target.unsqueeze(1), 1.)
                target_var = Variable(target_onehot, requires_grad=False)
                target_real = (target_var * output).sum(1)

                loss_adv = torch.sum(target_real - real)

            loss_adv.backward(retain_graph=True)
            eta = step_size * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    test_loss = 0
    correct = 0
    # blockPrint()
    AA = AutoAttack(model, eps=epsilon)
    AA.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            if category == 'AA':
                x_adv = AA.run_standard_evaluation(data, target, bs=len(target))
            else:
                x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss_fn,perturb_steps,epsilon,step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    # enablePrint()
    return test_loss, test_accuracy

def eval_clean_class(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc_list = [0]*10
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(pred)):
                if pred[i] == target[i]:
                    acc_list[target[i]] += 1
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy, acc_list

def eval_robust_class(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    test_loss = 0
    correct = 0
    acc_list = [0] * 10
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            if category == 'AA':
                AA = AutoAttack(model, eps=8 / 255)
                AA.attacks_to_run = ['apgd-ce', 'apgd-t']
                x_adv = AA.run_standard_evaluation(data, target)
            else:
                x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(pred)):
                if pred[i] == target[i]:
                    acc_list[target[i]] += 1
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss_fn,perturb_steps,epsilon,step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy, acc_list

# def eval_rr_loss()

def max_margin_loss(x, y):
    B = y.size(0)
    corr = x[range(B), y]

    x_new = x - 1000 * torch.eye(10)[y].cuda()
    tar = x[range(B), x_new.argmax(dim=1)]
    loss = tar - corr
    loss = torch.mean(loss)

    return loss

