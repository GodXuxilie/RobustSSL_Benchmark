### Pre-training ###
nohup python SAT.py --gpu 0,1,2,3 --out_dir KL_005 --method RCS --fraction 0.05 &
nohup python SAT.py --gpu 0,1,2,3 --out_dir KL_01 --method RCS --fraction 0.1 &
nohup python SAT.py --gpu 0,1,2,3 --out_dir KL_02 --method RCS --fraction 0.2 &

### finetuning ###
PT=KL_005/checkpoint.pth.tar
NAME=KL_005

nohup python adv_tune.py --gpu 0 --out_dir cifar10_robust_full_finetuning/${NAME} --dataset cifar10 --lr 0.01 --resume $PT &
nohup python adv_tune.py --gpu 0 --out_dir cifar100_robust_full_finetuning/${NAME} --dataset cifar100 --lr 0.01 --resume $PT &

nohup python adv_tune.py --gpu 0 --out_dir cifar10_robust_partial_finetuning/${NAME} --dataset cifar10 --lr 0.01 --resume $PT --linear &
nohup python adv_tune.py --gpu 0 --out_dir cifar100_robust_partial_finetuning/${NAME} --dataset cifar100 --lr 0.01 --resume $PT --linear &
