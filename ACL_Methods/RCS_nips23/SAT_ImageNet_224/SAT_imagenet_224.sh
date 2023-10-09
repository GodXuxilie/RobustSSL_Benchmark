### Pre-training ###
nohup python SAT.py --gpu 0,1,2,3 --net r50 --out_dir KL_005 --method RCS --fraction 0.05 &
nohup python SAT.py --gpu 0,1,2,3 --net r50 --out_dir KL_01 --method RCS --fraction 0.1 &
nohup python SAT.py --gpu 0,1,2,3 --net r50 --out_dir KL_02 --method RCS --fraction 0.2 &


### Finetuning ###
NAME = KL_005
PT = KL_005/checkpoint.pth.tar

nohup python transfer.py --gpu 0 --dataset cifar10  --out_dir cifar10_standard_full_finetuning/${NAME} --net r50 --lr 0.001 --resume $PT  &
nohup python transfer.py --gpu 0 --dataset cifar100  --out_dir cifar100_standard_full_finetuning/${NAME} --net r50 --lr 0.001 --resume $PT  &

nohup python transfer.py --gpu 0 --dataset cifar10  --out_dir cifar10_standard_partial_finetuning/${NAME} --net r50 --lr 0.01 --resume $PT  &
nohup python transfer.py --gpu 0 --dataset cifar100  --out_dir cifar100_standard_partial_finetuning/${NAME} --net r50 --lr 0.01 --resume $PT  &
