nohup python finetuning_eval.py --gpu 0 --experiment ACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS.pt --mode Ensemble --eval-AA --eval-OOD &
nohup python finetuning_eval.py --gpu 1 --experiment ACL_cifar100_r18_cifar100 --dataset cifar100 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS_CIFAR100.pt  --mode Ensemble --eval-AA --eval-OOD &
nohup python finetuning_eval.py --gpu 2 --experiment DynACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL.pt --mode Ensemble --eval-AA --eval-OOD &
nohup python finetuning_eval.py --gpu 3 --experiment DynACL_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode Ensemble --eval-AA --eval-OOD &

nohup python finetuning_eval.py --gpu 0 --experiment ACL_cifar10_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS.pt --mode Ensemble --eval-AA &
nohup python finetuning_eval.py --gpu 1 --experiment ACL_cifar100_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS_CIFAR100.pt  --mode Ensemble --eval-AA &
nohup python finetuning_eval.py --gpu 2 --experiment DynACL_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL.pt --mode Ensemble --eval-AA &
nohup python finetuning_eval.py --gpu 3 --experiment DynACL_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode Ensemble --eval-AA &
