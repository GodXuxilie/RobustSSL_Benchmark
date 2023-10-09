
# Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization
---
This repository provides codes for NeurIPS 2023 paper: **Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization** (https://arxiv.org/pdf/2305.00374.pdf) 
<br> Xilie Xu* (NUS), Jingfeng Zhang* (RIKEN-AIP/University of Auckland), Feng Liu (The University of Melbourne), Masashi Sugiyama (RIKEN-AIP/The University of Toyko), Mohan Kankanhalli (NUS).

## Environment
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6


## Script
```
# ACL with IR
python pretraining.py exp_dir --dataset dataset --ACL_DS 
# DynACL with IR
python pretraining.py exp_dir --dataset dataset --ACL_DS --DynAug
```

## BibTeX
```
@inproceedings{xu2023IR,
  title={Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization},
  author={Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Contact
Please drop an e-mail to xuxilie@comp.nus.edu.sg and jingfeng.zhang@auckland.ac.nz if you have any issue.