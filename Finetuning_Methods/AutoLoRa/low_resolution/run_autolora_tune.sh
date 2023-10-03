# CIFAR-10 task
python vanilla_finetuning.py --gpu 3 --experiment ACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining ACL --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/dropbox_checkpoints/ACL.pt --mode AFF --eval-AA --eval-OOD &
python vanilla_finetuning.py --gpu 3 --experiment DynACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/dropbox_checkpoints/DynACL.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 3 --experiment AdvCL_cifar10_r18_cifar10 --dataset cifar10 --pretraining AdvCL --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/AdvCL.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar10_r18_cifar10 --dataset cifar10 --pretraining A-InfoNCE --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/AInfoNCE.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DeACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining DeACL --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/DeACL.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL++_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL++ --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/DynACL++.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL_IR_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL_IR --model r18 --checkpoint /home/x/xuxilie/RobSSL_benchmark_copy/DynACL_IR.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL++_IR_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL++_IR --model r18 --checkpoint ./home/x/xuxilie/RobSSL_benchmark_copy/DynACL++_IR.pt --mode ALL --eval-AA --eval-OOD &

# CIFAR-100 task
nohup python vanilla_finetuning.py --gpu 1 --experiment ACL_cifar100_r18_cifar100 --dataset cifar100 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS_CIFAR100.pt  --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 3 --experiment DynACL_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL++_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL++ --model r18 --checkpoint ./checkpoints/DynACL++_CIFAR100.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 0 --experiment AdvCL_cifar100_r18_cifar100 --dataset cifar100 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL_CIFAR100.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar100_r18_cifar100 --dataset cifar100 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE_CIFAR100.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL_IR_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL_IR --model r18 --checkpoint ./checkpoints/DynACL_IR_CIFAR100.pt --mode ALL --eval-AA --eval-OOD &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL++_IR_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL++_IR --model r18 --checkpoint ./checkpoints/DynACL++_IR_CIFAR100.pt --mode ALL --eval-AA --eval-OOD &

# STL10 task
nohup python vanilla_finetuning.py --gpu 1 --experiment ACL_stl10_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS_STL10.pt  --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 3 --experiment DynACL_stl10_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_STL10.pt  --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL++_stl10_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL++_STL10.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL_IR_stl10_r18_stl10 --dataset stl10 --pretraining DynACL_IR --model r18 --checkpoint ./checkpoints/DynACL_IR_STL10.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL++_IR_stl10_r18_stl10 --dataset stl10 --pretraining DynACL++_IR --model r18 --checkpoint ./checkpoints/DynACL++_IR_STL10.pt --mode ALL --eval-AA &

# transferability from CIFAR-10 to STL10
nohup python vanilla_finetuning.py --gpu 0 --experiment ACL_cifar10_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 0 --experiment AdvCL_cifar10_r18_stl10 --dataset stl10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar10_r18_stl10 --dataset stl10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment DeACL_cifar10_r18_stl10 --dataset stl10 --pretraining DeACL --model r18 --checkpoint ./checkpoints/DeACL.pt --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 2 --experiment DynACL_IR_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL_IR --model r18 --checkpoint ./checkpoints/DynACL_IR.pt --mode ALL --eval-AA &

# transferability from CIFAR-100 to STL10
nohup python vanilla_finetuning.py --gpu 1 --experiment ACL_cifar100_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_DS_CIFAR100.pt  --mode ALL --eval-AA &
nohup python vanilla_finetuning.py --gpu 3 --experiment DynACL_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode ALL  --eval-AA &
nohup python vanilla_finetuning.py --gpu 0 --experiment AdvCL_cifar100_r18_stl10 --dataset stl10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL_CIFAR100.pt --mode ALL --eval-AA  &
nohup python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar100_r18_stl10 --dataset stl10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE_CIFAR100.pt --mode ALL --eval-AA  &
nohup python vanilla_finetuning.py --gpu 3 --experiment DynACL_IR_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL_IR --model r18 --checkpoint ./checkpoints/DynACL_IR_CIFAR100.pt  --mode ALL --eval-AA &
