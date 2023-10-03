# Efficient Adversarial Contrastive Learning via Robustness-aware Coreset Selection 
---
This repository provides codes for NeurIPS 2023 (Spotlight) paper: **Efficient Adversarial Contrastive Learning via Robustness-aware Coreset Selection** (https://arxiv.org/pdf/2302.03857.pdf) Xilie Xu* (NUS), Jingfeng Zhang* (RIKEN-AIP/University of Auckland), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS).

<!-- In this repo, we provide the code and the script for reproduce the experiemtns in the main paper, including ACL/DynACL on CIFAR-10/CIFAR-100/STL10, ACL on ImageNet-1K, and standard adversarial training (SAT) on ImageNet-1K.  -->

### Dataset preparation
As for preparing ImageNet-1K of $32 \times 32$ resolution, we use the following scripts.

```
wget https://image-net.org/data/downsample/Imagenet32_train.zip
wget https://image-net.org/data/downsample/Imagenet32_val.zip
```
