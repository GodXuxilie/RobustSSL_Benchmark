# Fine-tuning adversarailly pre-trained models via supervised adversarial training on ImageNet-224

# cifar10
python AutoLoRa.py --adv_train --model_root './cifar10_resnet18_AutoLoRA' --dataset cifar10 --num_classes 10 --model_arch resnet18 --model-path './resnet18_linf_eps4.0.ckpt'
python AutoLoRa.py --adv_train --model_root './cifar10_resnet50_AutoLoRA' --dataset cifar10 --num_classes 10 --model_arch resnet50 --model-path './resnet50_linf_eps4.0.ckpt'

# cifar100
python AutoLoRa.py --adv_train --model_root './cifar100_resnet18_AutoLoRA' --dataset cifar100 --num_classes 100 --model_arch resnet18 --model-path './resnet18_linf_eps4.0.ckpt'
python AutoLoRa.py --adv_train --model_root './cifar100_resnet50_AutoLoRA' --dataset cifar100 --num_classes 100 --model_arch resnet50 --model-path './resnet50_linf_eps4.0.ckpt'

# dtd57
python AutoLoRa.py --adv_train --model_root './dts57_resnet18_AutoLoRA' --dataset dtd --num_classes 57 --model_arch resnet18 --model-path './resnet18_linf_eps4.0.ckpt'
python AutoLoRa.py --adv_train --model_root './dtd57_resnet50_AutoLoRA' --dataset dtd --num_classes 57 --model_arch resnet50 --model-path './resnet50_linf_eps4.0.ckpt'

# dog120
python AutoLoRa.py --adv_train --model_root './dog120_resnet18_AutoLoRA' --dataset dog --num_classes 120 --model_arch resnet18 --model-path './resnet18_linf_eps4.0.ckpt'
python AutoLoRa.py --adv_train --model_root './dog120_resnet50_AutoLoRA' --dataset dog --num_classes 120 --model_arch resnet50 --model-path './resnet50_linf_eps4.0.ckpt'

# cub200
python AutoLoRa.py --adv_train --model_root './cub200_resnet18_AutoLoRA' --dataset cub --num_classes 200 --model_arch resnet18 --model-path './resnet18_linf_eps4.0.ckpt'
python AutoLoRa.py --adv_train --model_root './cub200_resnet50_AutoLoRA' --dataset cub --num_classes 200 --model_arch resnet50 --model-path './resnet50_linf_eps4.0.ckpt'

# cal256
python AutoLoRa.py --adv_train --model_root './caltech256_resnet18_AutoLoRA' --dataset caltech256 --num_classes 257 --model_arch resnet18 --model-path './resnet18_linf_eps4.0.ckpt'
python AutoLoRa.py --adv_train --model_root './caltech256_resnet50_AutoLoRA' --dataset caltech256 --num_classes 257 --model_arch resnet50 --model-path './resnet50_linf_eps4.0.ckpt'


# Fine-tuning adversarailly pre-trained models via self-supervised adversarial training

