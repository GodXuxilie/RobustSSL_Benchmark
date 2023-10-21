# The code is modified from the code of DynACL <https://github.com/PKU-ML/DYNACL>.
import os
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
from utils import train, get_loader, get_model,setup_hyperparameter, eval_test_nat, runAA, logger, eval_test_OOD, eval_adv_test
import torch

parser = argparse.ArgumentParser(
    description='Finetuning (SLF, ALF, AFF) and Evaluation')
parser.add_argument('--experiment', type=str,
                    help='location for saving trained models,\
                    we recommend to specify it as a subdirectory of the pretraining export path',
                    required=True)

parser.add_argument('--model', type=str, default='r18')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--dualBN', type=int, default=0, 
                    help='Whether to use dual BN module during fine-tuning')

parser.add_argument('--data', type=str, default='../data',
                    help='location of the data')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset to be used (cifar10 or cifar100)')
parser.add_argument('--resize', type=int, default=32,
                    help='location of the data')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 512)')

parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--decreasing_lr', default='15,20',
                    help='decreasing strategy')

parser.add_argument('--epsilon', type=float, default=8. / 255.,
                    help='perturbation')
parser.add_argument('--step-size', type=float, default=2. / 255.,
                    help='perturb step size')
parser.add_argument('--num-steps-train', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--num-steps-test', type=int, default=20,
                    help='perturb number of steps')

parser.add_argument('--eval-only', action='store_true',
                    help='if specified, eval the loaded model')
parser.add_argument('--eval-AA', action='store_true',
                    help='if specified, eval the loaded model')
parser.add_argument('--eval-OOD', action='store_true',
                    help='if specified, eval the loaded model')

parser.add_argument('--pretraining', type=str, default='ACL',
                    choices=['ACL', 'AdvCL', 'A-InfoNCE', 'DeACL', 'DynACL', 'DynACL++', 'DynACL_AIR', 'DynACL_AIR++', 'DynACL_RCS'])
parser.add_argument('--mode', type=str, default='ALL',
                    choices=['ALL', 'SLF', 'ALF', 'AFF'])


parser.add_argument('--autolora', action='store_true',
                    help='if specified, eval the loaded model')
parser.add_argument('--r_nat', type=int, default=0, 
                    help='rank of the low-rank branch. If it is above 0, it uses AutoLoRa')
parser.add_argument('--val', type=float, default=0.0, help='validation size')
parser.add_argument('--scale', type=int, default=6, help='The scale of weight for the KL loss')
parser.add_argument('--autoLR', type=int, default=0, 
                    help='Whether to use the automatic learning rate scheduler')
parser.add_argument('--divide', type=int, default=2, 
                    help='The decay rate of learning rate in the automatic learning rate scheduler')


parser.add_argument('--seed', type=int, default=1, help='the directory to save the models')
parser.add_argument('--test_frequency', type=int, default=0,
                    help='validation frequency during finetuning, 0 for no evaluation')   

parser.add_argument('--gpu', type=str, default='0', help='Set up the GPU id')

args = parser.parse_args()


# settings
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model_dir = os.path.join('checkpoints', args.experiment)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
device = 'cuda'
cudnn.benchmark = True

### Setting AutoLoRa ###
if args.autolora:
    args.r_nat = 8
    args.autoLR = 1
    args.epochs = 50
    args.test_frequency = 1
    args.scale = 6

if args.eval_only:
    log = logger(os.path.join(model_dir))
    common_corrup_dir = 'checkpoints/' + args.experiment + '/common_corruptions'
    if not os.path.exists(common_corrup_dir):
        os.makedirs(common_corrup_dir)
    common_corrup_log = logger(os.path.join(common_corrup_dir))

    robust_dir = 'checkpoints/' + args.experiment + '/robust'
    if not os.path.exists(robust_dir):
        os.makedirs(robust_dir)
    robust_log = logger(os.path.join(robust_dir))
    AA_log = robust_dir + '/AA_details.txt'

    result_dir = 'checkpoints/' + args.experiment + '/result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_log = logger(os.path.join(result_dir))
    result_log.info('test checkpoint: {}'.format(args.checkpoint))

    mode = 'eval'
    advFlag = None

    train_loader, vali_loader, test_loader, num_classes, args = get_loader(args)
    model, optimizer, scheduler = get_model(args, num_classes, mode, log, device=device)
    model.eval()

    # eval natural accuracy
    nat_acc = eval_test_nat(model, test_loader, device, advFlag)
    log.info('{} standard acc: {:.2f}'.format(mode, nat_acc * 100))
    
    # eval robustness against adversarial attacks
    if args.eval_AA:
        PGD20_acc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=torch.nn.functional.cross_entropy, log=log, attack_iter=args.num_steps_test)
        log.info('{} PGD20 acc: {:.2f}'.format(mode, PGD20_acc))
        
        AA_acc = runAA(args, model, test_loader, AA_log, advFlag=None)
        log.info('{} robust acc: {:.2f}'.format(mode, AA_acc * 100))

    # eval robustness againt common corruptions
    if args.eval_OOD and args.dataset in ['cifar10', 'cifar100']:
        common_corrup_dir = os.path.join(model_dir, 'common_corruptions')
        common_corrup_log = logger(os.path.join(common_corrup_dir))
        ood_acc_list, ood_acc_mean = eval_test_OOD(model, args.dataset, common_corrup_log, device, advFlag=None)
        log.info('{} mean corruption acc: {:.2f}'.format(mode, ood_acc_mean * 100))
        for i in range(5):
            log.info('{} corruption severity-{} acc: {:.2f}'.format(mode, i+1, ood_acc_list[i] * 100))
    
else:
    common_corrup_dir = 'checkpoints/' + args.experiment + '/common_corruptions'
    if not os.path.exists(common_corrup_dir):
        os.makedirs(common_corrup_dir)
    common_corrup_log = logger(os.path.join(common_corrup_dir))

    robust_dir = 'checkpoints/' + args.experiment + '/robust'
    if not os.path.exists(robust_dir):
        os.makedirs(robust_dir)
    robust_log = logger(os.path.join(robust_dir))
    AA_log = robust_dir + '/AA_details.txt'

    result_dir = 'checkpoints/' + args.experiment + '/result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_log = logger(os.path.join(result_dir))
    result_log.info('Finetuning checkpoint: {}'.format(args.checkpoint))

    if args.mode in ['ALL', 'SLF']:
        mode = 'SLF'
        model_dir = os.path.join('checkpoints', args.experiment, mode)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log = logger(os.path.join(model_dir))
        log.info(str(args))
        advFlag = None
        log.info('Finetuning mode: {}'.format(mode))
        # SLF finetuning
        args = setup_hyperparameter(args, mode)
        
        train_loader, vali_loader, test_loader, num_classes, args = get_loader(args)
        model, optimizer, scheduler = get_model(args, num_classes, mode, log, device=device)
        model = train(args, model, optimizer, scheduler, train_loader, test_loader, mode, device, log, model_dir, vali_loader)

        # eval natural accuracy
        SLF_nat_acc = eval_test_nat(model, test_loader, device, advFlag)
        result_log.info('{} standard acc: {:.2f}'.format(mode, SLF_nat_acc * 100))

        # eval robustness against adversarial attacks (PGD20)
        SLF_PGD20_acc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=torch.nn.functional.cross_entropy, log=log, attack_iter=args.num_steps_test)
        result_log.info('{} PGD20 acc: {:.2f}'.format(mode, SLF_PGD20_acc))

        # eval robustness against adversarial attacks (AutoAttack)
        if args.eval_AA:
            SLF_AA_acc = runAA(args, model, test_loader, AA_log, advFlag)
            result_log.info('{} robust acc: {:.2f}'.format(mode, SLF_AA_acc * 100))

        # eval robustness againt common corruptions
        if args.eval_OOD and args.dataset in ['cifar10', 'cifar100']:
            SLF_ood_acc_list, SLF_ood_acc_mean = eval_test_OOD(model, args.dataset, common_corrup_log, device, advFlag)
            result_log.info('{} mean corruption acc: {:.2f}'.format(mode, SLF_ood_acc_mean * 100))
            for i in range(5):
                result_log.info('{} corruption severity-{} acc: {:.2f}'.format(mode, i+1, SLF_ood_acc_list[i] * 100))

    if args.mode in ['ALL', 'ALF']:
        mode = 'ALF'
        model_dir = os.path.join('checkpoints', args.experiment, mode)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log = logger(os.path.join(model_dir))
        log.info(str(args))
        
        log.info('Finetuning mode: {}'.format(mode))

        advFlag = None
        args = setup_hyperparameter(args, mode)

        train_loader, vali_loader, test_loader, num_classes, args = get_loader(args)
        model, optimizer, scheduler = get_model(args, num_classes, mode, log, device=device)
        model = train(args, model, optimizer, scheduler, train_loader, test_loader, mode, device, log, model_dir, vali_loader)

        # eval natural accuracy
        ALF_nat_acc = eval_test_nat(model, test_loader, device, advFlag)
        result_log.info('{} standard acc: {:.2f}'.format(mode, ALF_nat_acc * 100))

        # eval robustness against adversarial attacks (PGD20)
        ALF_PGD20_acc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=torch.nn.functional.cross_entropy, log=log, attack_iter=args.num_steps_test)
        result_log.info('{} PGD20 acc: {:.2f}'.format(mode, ALF_PGD20_acc))

        # eval robustness against adversarial attacks (AutoAttack)
        if args.eval_AA:
            ALF_AA_acc = runAA(args, model, test_loader, AA_log, advFlag)
            result_log.info('{} robust acc: {:.2f}'.format(mode, ALF_AA_acc * 100))

        # eval robustness againt common corruptions
        if args.eval_OOD and args.dataset in ['cifar10', 'cifar100']:
            ALF_ood_acc_list, ALF_ood_acc_mean = eval_test_OOD(model, args.dataset, common_corrup_log, device, advFlag)
            result_log.info('{} mean corruption acc: {:.2f}'.format(mode, ALF_ood_acc_mean * 100))
            for i in range(5):
                result_log.info('{} corruption severity-{} acc: {:.2f}'.format(mode, i+1, ALF_ood_acc_list[i] * 100))

    if args.mode in ['ALL', 'AFF']:
        mode = 'AFF'
        model_dir = os.path.join('checkpoints', args.experiment, mode)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log = logger(os.path.join(model_dir))
        log.info(str(args))
        
        log.info('Finetuning mode: {}'.format(mode))
        args = setup_hyperparameter(args, mode)
        advFlag = None

        train_loader, vali_loader, test_loader, num_classes, args = get_loader(args)
        model, optimizer, scheduler = get_model(args, num_classes, mode, log, device=device)
        model = train(args, model, optimizer, scheduler, train_loader, test_loader, mode, device, log, model_dir, vali_loader)

        # eval natural accuracy
        AFF_nat_acc = eval_test_nat(model, test_loader, device, advFlag)
        result_log.info('{} standard acc: {:.2f}'.format(mode, AFF_nat_acc * 100))

        # eval robustness against adversarial attacks (PGD20)
        AFF_PGD20_acc = eval_adv_test(model, device, test_loader, epsilon=args.epsilon, alpha=args.step_size,
                          criterion=torch.nn.functional.cross_entropy, log=log, attack_iter=args.num_steps_test)
        result_log.info('{} PGD20 acc: {:.2f}'.format(mode, AFF_PGD20_acc))

        # eval robustness against adversarial attacks (AutoAttack)
        if args.eval_AA:
            AFF_AA_acc = runAA(args, model, test_loader, AA_log, advFlag)
            result_log.info('{} robust acc: {:.2f}'.format(mode, AFF_AA_acc * 100))

        # eval robustness againt common corruptions
        if args.eval_OOD and args.dataset in ['cifar10', 'cifar100']:
            AFF_ood_acc_list, AFF_ood_acc_mean = eval_test_OOD(model, args.dataset, common_corrup_log, device, advFlag)
            result_log.info('{} mean corruption acc: {:.2f}'.format(mode, AFF_ood_acc_mean * 100))
            for i in range(5):
                result_log.info('{} corruption severity-{} acc: {:.2f}'.format(mode, i+1, AFF_ood_acc_list[i] * 100))

    if args.mode == 'ALL':
        result_log.info('mean robust accuracy: {:.2f}'.format(np.mean([SLF_AA_acc * 100, ALF_AA_acc * 100, AFF_AA_acc * 100])))
        if args.eval_AA:
            result_log.info('mean standard accuracy: {:.2f}'.format(np.mean([SLF_nat_acc * 100, ALF_nat_acc * 100, AFF_nat_acc * 100])))
        if args.eval_OOD and args.dataset in ['cifar10', 'cifar100']:
            result_log.info('mean corruption accuracy: {:.2f}'.format(np.mean([SLF_ood_acc_mean * 100, ALF_ood_acc_mean * 100, AFF_ood_acc_mean * 100])))

