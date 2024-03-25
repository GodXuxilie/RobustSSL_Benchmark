# Benchmarking Transferability of Robust Self-Supervised Learning (RobustSSL)
The wide-ranging applications of foundation models, espeically in safety-critical areas, necessitates the robust self-supervised learning which can yield strong adversarial robustness in downsteam tasks via fine-tuning.
In this repo, we provide a benchmark for robustness transferability of robust pre-training.


## [Leaderboard](https://robustssl.github.io)

The leaderboard is demonstrates in [robustssl.github.io](https://robustssl.github.io).

## RobustSSL: Methods and Model Zoo

We consider the following RobustSSL methods: 

- [AIR (Xu et al., NeurIPS'23a)](https://github.com/GodXuxilie/Enhancing_ACL_via_AIR)
- [RCS (Xu et al., NeurIPS'23b)](https://github.com/GodXuxilie/Efficient_ACL_via_RCS)
- [DynACL (Luo et al., ICLR'23)](https://github.com/PKU-ML/DYNACL) 
- [A-InfoNCE (Yu et al., ECCV'22)](https://github.com/yqy2001/A-InfoNCE)
- [DeAC (Zhang et al., ECCV'22)](https://github.com/pantheon5100/DeACL/tree/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e)
- [AdvCL (Fan et al., NeruIPS'21)](https://github.com/LijieFan/AdvCL)
- [ACL (Jiang et al., NeurIPS'20)](https://github.com/VITA-Group/Adversarial-Contrastive-Learning)
- [RoCL (Kim et al., NeurIPS'20)](https://github.com/Kim-Minseon/RoCL)

<!-- - [RoCL (NeurIPS'20)](https://github.com/Kim-Minseon/RoCL) -->

<details> <summary><b>Modle Zoo</b>: We released all the pre-trained checkpoints in <a href='https://www.dropbox.com/sh/h1hkv3lt2f2zvi2/AACp5IWNaMcqrYAu6hr__4yea?dl=0'>Dropbox</a>.</summary> 
<!-- Alternatively, you can copy the address of the ```link``` in the following table and then use ```wget link_address``` to download the specific pre-trained weight. -->

| Pre-trained weights of ResNet-18 encoder | ACL ([Jiang et al., NeurIPS'20](https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html)) | AdvCL ([Fan et al., NeurIPS'21](https://arxiv.org/abs/2111.01124)) | A-InfoNCE ([Yu et al., ECCV'22](https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness)) | DeACL ([Zhang et al., ECCV'22](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900716.pdf)) | DynACL ([Luo et al., ICLR'23](https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ)) | DynACL++ ([Luo et al., ICLR'23](https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ)) | DynACL-AIR ([Xu et al., NeurIPS'23a](https://arxiv.org/abs/2305.00374)) | DynACL-AIR++ ([Xu et al., NeurIPS'23a](https://arxiv.org/abs/2305.00374)) | DynACL-RCS ([Xu et al., NeurIPS'23b](https://arxiv.org/pdf/2302.03857)) |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | [link<sup>*</sup>](https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL_DS.pt?dl=0) | [link](https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0) | [link](https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/wo1qrrnybycunn3/DeACL.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/87fhoyrzh33fwrt/DynACL%2B%2B.pt?dl=0) | [link](https://www.dropbox.com/s/gw2yopl0dp0exhg/DynACL_IR.pt?dl=0) | [link](https://www.dropbox.com/s/3mucajb2shgaega/DynACL%2B%2B_IR.pt?dl=0) | [link](https://www.dropbox.com/scl/fi/trhle9cgvtryzyxmjh88s/DynACL_RCS.pt?rlkey=1v8o6jyrm822v3rdlytl4xiph&dl=0) |
| CIFAR-100 | [link<sup>*</sup>](https://www.dropbox.com/s/02hhe679zo6c7da/ACL_DS_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/fcf5y77p87447p9/AdvCL_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/f0ruhgov3dbdcae/AInfoNCE_CIFAR100.pt?dl=0) | - | [link<sup>*</sup>](https://www.dropbox.com/s/mabnrcp7zahp6ke/DynACL_CIFAR100.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/hcjol6ihj0en2fn/DynACL%2B%2B_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/fyilp077jfwom27/DynACL_IR_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/8mbsjwwqqtrvzw4/DynACL%2B%2B_IR_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/scl/fi/cjt1e2kbfzx0y019gy8h3/DynACL_RCS_CIFAR100.pt?rlkey=1iv8u06kiqahvnj5xo2hllurd&dl=0)| 
| STL10 | [link](https://www.dropbox.com/s/6jenhn0bpe5ifle/ACL_DS_STL10.pt?dl=0) | - | - | - | [link<sup>*</sup>](https://www.dropbox.com/s/ydd6lbracw73019/DynACL_STL10.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/gjxlu7woupqjmr2/DynACL%2B%2B_STL10.pt?dl=0) | [link](https://www.dropbox.com/s/78bm6mlqr3xci19/DynACL_IR_STL10.pt?dl=0) | [link](https://www.dropbox.com/s/ktja3i5mmbjm4dw/DynACL%2B%2B_IR_STL10.pt?dl=0) | [link](https://www.dropbox.com/scl/fi/acdsvzvipdfr3nzuhhuzl/DynACL_RCS_STL10.pt?rlkey=2nehgdbgy8oiq1cqjc2tiuyjx&dl=0) |

**Acknowledgements**: The superscript ```*``` denotes that the pre-trained encoders haved been provided in their GitHub and we copied them into our Dropbox directory; otherwise, the encoders were pre-trained by us.
</details> 


To provide a comprehensive benchmark, we welcome incoraporating new self-supervised robust pre-training methods into our repo!

## Fine-Tuning
Here, we provide two kinds of fine-tuning methods:
- [Vanilla Fine-tuning](https://github.com/GodXuxilie/ACL_Benchmark/tree/main/Finetuning_Methods/Vanilla_Finetuning): You need to specify the hyper-parameters such as the learning rate and the batch size for each pre-trained models. We provide all the scripts for finetuning and evalution in the file [```run_vanilla_tune.sh```](./Finetuning_Methods/Vanilla/run_vanilla_tune.sh).
<!-- ```setup_hyperparameter(args, mode)``` of the file ```utils.py```. -->
- [AutoLoRa (Xu et al., ArXiv'23)](https://github.com/GodXuxilie/ACL_Benchmark/tree/main/Finetuning_Methods/AutoLoRa): It is a **parameter-free and automated** robust fine-tuning framework. You *DO NOT* need to search for the appropriate hyper-parameters. We provide all the scripts for finetuning and evalution in the file [```run_autolora.sh```](./Finetuning_Methods/AutoLoRa/run_autolora.sh).

To provide a comprehensive benchmark, we welcome incoraporating new robust fine-tuning methods into our repo!

We consider the following three fine-tuning modes:
- Standard linear fine-tuning (**SLF**): only standardly fine-tuning the classifier while freezing the encoder.
- Adversarial linear fine-tuning (**ALF**): only adversarially fine-tuning the classifier while freezing the encoder.
- Adversarial full fine-tuning (**AFF**): adversarially fine-tuning both the encoder and the classifier.

<!-- To conduct fine-tuning, you can use the following script:
```
python finetuning_eval.py   --gpu GPU_id \
                            --experiment path_of_directory_for_saving_log \
                            --pretraining  pre_training_method: [ACL, AdvCL, A-InfoNCE, DeACL, DynACL, DynACL++, DynACL_IR, DynACL++_IR, Efficient_ACL, Efficient_DynACL] \
                            --model type_of_backbone_network \
                            --checkpoint path_of_pretrained_checkpoint \ 
                            --dataset downstream_dataset: [cifar10, cifar100, stl10] \ 
                            --finetuning finetuning_method: [vanilla, autolora] \
                            --mode finetuning_mode: [ALL, SLF, ALF, AFF] \ 
                            --eval-AA \
                            --eval-OOD 
```

We provide all the scripts for finetuning and evalution in the file ```run_tune_eval.sh```. Please feel free to check the performance of the pre-trained encoders. -->

<!-- If you want to use ```finetuning_eval.py``` to evaluate the performance of your pre-trained weights, you need to first specify the hyper-parameters such as the learning rate and the batch size in the function ```setup_hyperparameter(args, mode)``` of the file ```utils.py``` for your method, and then use the above script to conduct finetuning and evaluation. -->


## Requirement
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6
+ [AutoAttack](https://github.com/fra31/auto-attack) (Install AutoAttack via ```pip install git+https://github.com/fra31/auto-attack```)
+ [robustbench](https://robustbench.github.io/) (Install robustbench via ```pip install git+https://github.com/RobustBench/robustbench.git```)

</details>




## References
If you fine the code is useful to you, please cite the following papers by copying the following BibTeX.
<!-- <details> <summary>BibTeX</summary> -->

```
@inproceedings{
xu2024autolora,
title={AutoLoRa: A Parameter-Free Automated Robust Fine-Tuning Framework},
author={Xilie Xu and Jingfeng Zhang and Mohan Kankanhalli},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=09xFexjhqE}
}

@inproceedings{
xu2023efficient,
title={Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection},
author={Xilie Xu and Jingfeng Zhang and Feng Liu and Masashi Sugiyama and Mohan Kankanhalli},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=fpzA8uRA95}
}

@inproceedings{
xu2023enhancing,
title={Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization},
author={Xilie Xu and Jingfeng Zhang and Feng Liu and Masashi Sugiyama and Mohan Kankanhalli},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=zuXyQsXVLF}
}

@inproceedings{luo2023DynACL,
    title={Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning},
    author={Rundong Luo and Yifei Wang and Yisen Wang},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=0qmwFNJyxCL}
}

@inproceedings{zhang2022DeACL,
  title={Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness},
  author={Zhang, Chaoning and Zhang, Kang and Zhang, Chenshuang and Niu, Axi and Feng, Jiu and Yoo, Chang D and Kweon, In So},
  booktitle={ECCV 2022},
  pages={725--742},
  year={2022},
  organization={Springer}
}

@inproceedings{yu2022AInfoNCE,
  title={Adversarial Contrastive Learning via Asymmetric InfoNCE},
  author={Yu, Qiying and Lou, Jieming and Zhan, Xianyuan and Li, Qizhang and Zuo, Wangmeng and Liu, Yang and Liu, Jingjing},
  booktitle={European Conference on Computer Vision},
  pages={53--69},
  year={2022},
  organization={Springer}
}

@article{fan2021AdvCL,
  title={When Does Contrastive Learning Preserve Adversarial Robustness from Pretraining to Finetuning?},
  author={Fan, Lijie and Liu, Sijia and Chen, Pin-Yu and Zhang, Gaoyuan and Gan, Chuang},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={21480--21492},
  year={2021}
}

@article{jiang2020ACL,
  title={Robust pre-training by adversarial contrastive learning},
  author={Jiang, Ziyu and Chen, Tianlong and Chen, Ting and Wang, Zhangyang},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={16199--16210},
  year={2020}
}

@article{kim2020RoCL,
  title={Adversarial self-supervised contrastive learning},
  author={Kim, Minseon and Tack, Jihoon and Hwang, Sung Ju},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={2983--2994},
  year={2020}
}
```
<!-- </details> -->




<!-- ## Acknowledgements

The pre-training code in the directory of ```ACL_Methods``` are provided by [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning), [RoCL](https://github.com/Kim-Minseon/RoCL), [AdvCL](https://github.com/LijieFan/AdvCL), [A-InfoNCE](https://github.com/yqy2001/A-InfoNCE), [DeACL](https://github.com/pantheon5100/DeACL/tree/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e), and [DynACL](https://github.com/PKU-ML/DYNACL).  -->

## Contact

Please contact xuxilie@comp.nus.edu.sg and jingfeng.zhang@auckland.ac.nz if you have any question on the codes.

