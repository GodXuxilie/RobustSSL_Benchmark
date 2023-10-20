# Transferabilit from CIFAR-10 to CIFAR-10
python vanilla_finetuning.py --gpu 3 --experiment ACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL.pt --mode AFF --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 3 --experiment DynACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 3 --experiment AdvCL_cifar10_r18_cifar10 --dataset cifar10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL.pt --mode ALL --eval-AA --eval-OOD --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar10_r18_cifar10 --dataset cifar10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE.pt --mode ALL --eval-AA --eval-OOD --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DeACL_cifar10_r18_cifar10 --dataset cifar10 --pretraining DeACL --model r18 --checkpoint ./checkpoints/DeACL.pt --mode ALL --eval-AA --eval-OOD --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DynACL++_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL++ --model r18 --checkpoint ./checkpoints/DynACL++.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL_AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_RCS_cifar10_r18_cifar10 --dataset cifar10 --pretraining DynACL_RCS --model r18 --checkpoint ./checkpoints/DynACL_RCS.pt --mode ALL --eval-AA --eval-OOD

# Transferabilit from CIFAR-100 to CIFAR-100
python vanilla_finetuning.py --gpu 1 --experiment ACL_cifar100_r18_cifar100 --dataset cifar100 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_CIFAR100.pt  --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 3 --experiment DynACL_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL++_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL++ --model r18 --checkpoint ./checkpoints/DynACL++_CIFAR100.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 0 --experiment AdvCL_cifar100_r18_cifar100 --dataset cifar100 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL_CIFAR100.pt --mode ALL --eval-AA --eval-OOD --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar100_r18_cifar100 --dataset cifar100 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE_CIFAR100.pt --mode ALL --eval-AA --eval-OOD --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_CIFAR100.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL_AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++_CIFAR100.pt --mode ALL --eval-AA --eval-OOD
python vanilla_finetuning.py --gpu 2 --experiment DynACL_RCS_cifar100_r18_cifar100 --dataset cifar100 --pretraining DynACL_RCS --model r18 --checkpoint ./checkpoints/DynACL_RCS_CIFAR100.pt --mode ALL --eval-AA --eval-OOD

# Transferabilit from STL-10 to STL-10
python vanilla_finetuning.py --gpu 1 --experiment ACL_stl10_r18_stl10 --dataset stl10 --resize 96 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_STL10.pt  --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 3 --experiment DynACL_stl10_r18_stl10 --dataset stl10 --resize 96 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_STL10.pt  --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 2 --experiment DynACL++_stl10_r18_stl10 --dataset stl10 --resize 96 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL++_STL10.pt --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_stl10_r18_stl10 --dataset stl10 --resize 96 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_STL10.pt --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR++_stl10_r18_stl10 --dataset stl10 --resize 96 --pretraining DynACL_AIR++ --model r18 --checkpoint ./checkpoints/DynACL_AIR++_STL10.pt --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 2 --experiment DynACL_RCS_stl10_r18_stl10 --dataset stl10 --resize 96 --pretraining DynACL_RCS --model r18 --checkpoint ./checkpoints/DynACL_RCS_STL10.pt --mode ALL --eval-AA

# Transferability from CIFAR-10 to STL10
python vanilla_finetuning.py --gpu 0 --experiment ACL_cifar10_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL.pt --mode ALL --eval-AA --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DynACL_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL.pt --mode ALL --eval-AA --dualBN 0
python vanilla_finetuning.py --gpu 0 --experiment AdvCL_cifar10_r18_stl10 --dataset stl10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL.pt --mode ALL --eval-AA --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar10_r18_stl10 --dataset stl10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE.pt --mode ALL --eval-AA --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DeACL_cifar10_r18_stl10 --dataset stl10 --pretraining DeACL --model r18 --checkpoint ./checkpoints/DeACL.pt --mode ALL --eval-AA --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DynACL_AIR_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR.pt --mode ALL --eval-AA --dualBN 0
python vanilla_finetuning.py --gpu 2 --experiment DynACL_RCS_cifar10_r18_stl10 --dataset stl10 --pretraining DynACL_RCS --model r18 --checkpoint ./checkpoints/DynACL_RCS.pt --mode ALL --eval-AA --dualBN 0

# Transferability from CIFAR-100 to STL10
python vanilla_finetuning.py --gpu 1 --experiment ACL_cifar100_r18_stl10 --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_CIFAR100.pt  --mode ALL --eval-AA
python vanilla_finetuning.py --gpu 3 --experiment DynACL_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode ALL  --eval-AA
python vanilla_finetuning.py --gpu 0 --experiment AdvCL_cifar100_r18_stl10 --dataset stl10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL_CIFAR100.pt --mode ALL --eval-AA 
python vanilla_finetuning.py --gpu 2 --experiment AInfoNCE_cifar100_r18_stl10 --dataset stl10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE_CIFAR100.pt --mode ALL --eval-AA 
python vanilla_finetuning.py --gpu 3 --experiment DynACL_AIR_cifar100_r18_stl10 --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_CIFAR100.pt  --mode ALL --eval-AA
