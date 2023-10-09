# Efficient Adversarial Contrastive Learning via Robustness-aware Coreset Selection 
---
This repository provides codes for NeurIPS 2023 (Spotlight) paper: **Efficient Adversarial Contrastive Learning via Robustness-aware Coreset Selection** (https://arxiv.org/pdf/2302.03857.pdf) 
<br>Xilie Xu* (NUS), Jingfeng Zhang* (RIKEN-AIP/University of Auckland), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS).

<!-- In this repo, we provide the code and the script for reproduce the experiemtns in the main paper, including ACL/DynACL on CIFAR-10/CIFAR-100/STL10, ACL on ImageNet-1K, and standard adversarial training (SAT) on ImageNet-1K.  -->

## Prerequesite

<!-- ### Dataset preparation -->

### Package
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6


## Script

### [ACL/DynACL with RCS on CIFAR-10/CIFAR-100/STL-10](./ACL/run.sh)
```
cd ./ACL
python ACL.py exp_dir --ACL_DS --dataset dataset --fraction 0.2 &
python DynACL.py exp_dir --ACL_DS --dataset dataset --fraction 0.2 &
```
### [ACL with RCS on ImageNet-1K of $32 \times 32$ resolution](./ACL_ImageNet/ACL_imagnet.sh)
```
cd ./ACL_ImageNet
python ACL.py exp_dir --ACL_DS --fraction 0.05 &
```
As for preparing ImageNet-1K of $32 \times 32$ resolution, we use the following scripts.

```
wget https://image-net.org/data/downsample/Imagenet32_train.zip
wget https://image-net.org/data/downsample/Imagenet32_val.zip
```

### [Standard Adversarial Training (SAT) with RCS on ImageNet-1K of $32 \times 32$ resolution](./SAT_ImegeNet_32/SAT_imagenet_32.sh)
```
cd ./SAT_ImageNet_32
python SAT.py --out_dir exp_dir --fraction 0.05
```

### [Standard Adversarial Training (SAT) with RCS on ImageNet-1K of $224 \times 224$ resolution](./SAT_ImageNet_224/SAT_imagenet_224.sh)
```
cd ./SAT_ImageNet_224
python SAT.py --out_dir exp_dir --fraction 0.05 
```
As for preparing ImageNet-1K of $224\times 224$ resolution, we use the following scripts.
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

## BibTeX
```
@inproceedings{xu2023RCS,
  title={Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection},
  author={Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Contact
Please drop an e-mail to xuxilie@comp.nus.edu.sg and jingfeng.zhang@auckland.ac.nz if you have any issue.