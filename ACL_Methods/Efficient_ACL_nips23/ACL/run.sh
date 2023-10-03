### Pre Training on CIFAR10 ###
nohup python ACL.py results/ACL_RCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL.py results/ACL_RCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL.py results/ACL_RCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL.py results/DynRCS_KL_005 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL.py results/DynRCS_KL_01 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL.py results/DynRCS_KL_02 --gpu 0 --ACL_DS --dataset cifar10 --method RCS --CoresetLoss KL --fraction 0.2 &

### Pre Training on CIFAR100 ###
nohup python ACL.py results/ACL_RCS_KL_005_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL.py results/ACL_RCS_KL_01_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL.py results/ACL_RCS_KL_02_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL.py results/DynRCS_KL_005_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL.py results/DynRCS_KL_01_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL.py results/DynRCS_KL_02_cifar100 --gpu 0 --ACL_DS --dataset cifar100 --method RCS --CoresetLoss KL --fraction 0.2 &

### Pre Training on STL10 ###
nohup python ACL.py results/ACL_RCS_KL_005_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python ACL.py results/ACL_RCS_KL_01_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python ACL.py results/ACL_RCS_KL_02_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.2 &

nohup python DynACL.py results/DynRCS_KL_005_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.05 &
nohup python DynACL.py results/DynRCS_KL_01_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.1 &
nohup python DynACL.py results/DynRCS_KL_02_stl10 --gpu 0 --ACL_DS --dataset stl10 --method RCS --CoresetLoss KL --fraction 0.2 &

