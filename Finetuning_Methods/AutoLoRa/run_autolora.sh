# AutoLoRa for cross-task robustness transferability from CIFAR-10 to STL-10
python finetuning.py --gpu 3 --experiment AInfoNCE_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE.pt --mode ALL --eval-AA --autolora
python finetuning.py --gpu 0 --experiment DeACL_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining DeACL --model r18 --checkpoint ./checkpoints/DeACL.pt --mode ALL --eval-AA --autolora
python finetuning.py --gpu 1 --experiment ACL_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL.pt --mode ALL --eval-AA --autolora
python finetuning.py --gpu 2 --experiment DynACL_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL.pt --mode ALL --eval-AA --autolora
python finetuning.py --gpu 3 --experiment AdvCL_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL.pt --mode ALL --eval-AA --autolora
python finetuning.py --gpu 3 --experiment DynACL_AIR_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR.pt --mode ALL --eval-AA --autolora
python finetuning.py --gpu 1 --experiment DynACL_RCS_cifar10_r18_stl10_AutoLoRa --dataset stl10 --pretraining DynACL_RCS --model r18 --checkpoint ./checkpoints/DynACL_RCS.pt --mode ALL --eval-AA --autolora

# AutoLoRa for cross-task robustness transferability from CIFAR-100 to STL-10
python finetuning.py --gpu 2 --experiment ACL_cifar100_r18_stl10_AutoLoRa --dataset stl10 --pretraining ACL --model r18 --checkpoint ./checkpoints/ACL_CIFAR100.pt  --mode ALL --eval-AA --autolora
python finetuning.py --gpu 0 --experiment AdvCL_cifar100_r18_stl10_AutoLoRa --dataset stl10 --pretraining AdvCL --model r18 --checkpoint ./checkpoints/AdvCL_CIFAR100.pt --mode ALL --eval-AA  --autolora
python finetuning.py --gpu 1 --experiment DynACL_cifar100_r18_stl10_AutoLoRa --dataset stl10 --pretraining DynACL --model r18 --checkpoint ./checkpoints/DynACL_CIFAR100.pt  --mode ALL  --eval-AA --autolora
python finetuning.py --gpu 1 --experiment AInfoNCE_cifar100_r18_stl10_AutoLoRa --dataset stl10 --pretraining A-InfoNCE --model r18 --checkpoint ./checkpoints/AInfoNCE_CIFAR100.pt --mode AFF --eval-AA --autolora
python finetuning.py --gpu 2 --experiment DynACL_AIR_cifar100_r18_stl10_AutoLoRa --dataset stl10 --pretraining DynACL_AIR --model r18 --checkpoint ./checkpoints/DynACL_AIR_CIFAR100.pt  --mode ALL --eval-AA --autolora
python finetuning.py --gpu 2 --experiment DynACL_RCS_cifar100_r18_stl10_AutoLoRa --dataset stl10 --pretraining DynACL_RCS --model r18 --checkpoint ./checkpoints/DynACL_RCS_CIFAR100.pt --mode ALL --eval-AA --autolora
