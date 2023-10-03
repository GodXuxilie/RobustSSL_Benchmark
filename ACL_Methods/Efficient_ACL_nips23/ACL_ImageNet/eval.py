from __future__ import print_function
import argparse
import torchvision
from torchvision import transforms
import pickle
import attack_generator as attack
import numpy as np
import os
import torch

parser = argparse.ArgumentParser(description='PyTorch test')
parser.add_argument('--epsilon', type=int, default=8, help='perturbation')
parser.add_argument('--num-steps', type=int, default=10, help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.007, help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--net', type=str, default="resnet18",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100")
parser.add_argument('--random', type=bool, default=True, help="whether to initiat adversarial sample with random noise")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width-factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop-rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--dir', type=str, default=None, help='pt model dir')
parser.add_argument('--all_epoch', action='store_true')
parser.add_argument('--pt-name',type=str,default='')
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--end_epoch', type=int, default=20)
parser.add_argument('--test_aa', action='store_true')
parser.add_argument('--mode', type=str, default='final', help='final, singal, fast')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# settings
seed = args.seed
epsilon = args.epsilon
step_size = args.step_size
num_steps = args.num_steps
random = args.random
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
dir = args.dir
pt_name = args.pt_name

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setup data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='/data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100(root='/data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 100

from models.wrn import WideResNet
model = WideResNet(28, num_classes, 10, dropRate=0).cuda()
# coreset_model = WideResNet(28, num_classes, 10, dropRate=0).cuda()

# print(net)
state = dict()
natural_acc = []
fgsm_eps8_acc = []
xent_pgd20_eps8_acc = []
cw_pgd20_eps8_acc = []
aa_acc = []
eps = args.epsilon
alpha = eps/4


import datetime
starttime = datetime.datetime.now()
model.load_state_dict(torch.load(pt_name, map_location="cuda:0")['state_dict'])
model.eval()
loss, acc = attack.eval_clean(model, test_loader)
natural_acc.append(acc)
print('natural: ', acc)
loss, acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8 / 255, step_size=8 / 2550,
                                loss_fn="cent", category="Madry", rand_init=True)
print('PGD20: ', acc)

# loss, acc5 = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=eps / 255,
#                                     step_size=alpha / 2550,
#                                     loss_fn='rand', category="AA", rand_init=True)
# print('AA: ', acc)
