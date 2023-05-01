# Benchmarking Robustness Transferability of Adversarial Contrastive Learning (ACL)
This repository contains the following three kinds of code:
+ Robust pre-training via various ACL methods 
+ Finetuning tools
+ Evaluation protocals

# Requirement
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6
+ AutoAttack
+ robustbench

# Robust pre-training via ACL
We provide a unified pre-training framework to conduct robust pre-training.

```
python pretraining.py   --gpu GPU_id \
                        --experiment path_of_log_to_be_saved \
                        --pretraining pre_training_method: [ACL, AdvCL, AInfoNCE, DynACL] \
                        --dataset pretraining_dataset: [cifar10, cifar100, stl10]
                        --model type_of_backbone_network
```

# Finetuning tools
We provide a unified finetuning tools to applying the adversarially pre-trained models to downstream tasks.

### Finetuning without post-processing
```
python finetuning.py    --gpu GPU_id \
                        --experiment path_of_log_to_be_saved \
                        --pretraining  pre_training_method \
                        --model type_of_backbone_network \
                        --checkpoint path_of_pretrained_checkpoint \ 
                        --dataset downstream_dataset: [cifar10, cifar100, stl10] \ 
                        --tune_mode finetuning_mode: [SLF,ALF,AFF]
```
### Finetuning with post-processing
```
python finetuning.py    --gpu GPU_id \
                        --experiment path_of_log_to_be_saved \
                        --pretraining  pre_training_method \
                        --model type_of_backbone_network \
                        --checkpoint path_of_pretrained_checkpoint \ 
                        --dataset downstream_dataset: [cifar10, cifar100, stl10] \
                        --tune_mode finetuning_mode: [LP_SLF,LP_ALF,LP_AFF]
```
### Finetuning in the semi-supervised setting
```
python finetuning.py    --gpu GPU_id \
                        --experiment path_of_log_to_be_saved \
                        --pretraining  pre_training_method \
                        --model type_of_backbone_network \
                        --checkpoint path_of_pretrained_checkpoint \ 
                        --dataset downstream_dataset: [cifar10, cifar100, stl10] \
                        --tune_mode finetuning_mode: [semi_AT]
```

# Evalution Protocals
We provide a unified evaluation protocals to fairly evaluate the performance of finetuned models on the downstream tasks.
```
python evaluation.py    --gpu GPU_id \
                        --experiment path_of_log_to_be_saved \
                        --pretraining  pre_training_method \
                        --model type_of_backbone_network \
                        --checkpoint path_of_finetuned_checkpoint \ 
                        --dataset downstream_dataset: [cifar10, cifar100, stl10] \
                        --eval_mode evalution_mode: [adv_attack, common_corrup]
```

# Performance Benchmarking Across Tasks
Here, robust pre-training and finetuning are conducted on the same datasets.

## CIFAR-10 task

<table>
<thead>
  <tr>
    <th rowspan="2">Rank</th>
    <th rowspan="2">Pre-training</th>
    <th rowspan="2">Reference</th>
    <th rowspan="2">Mean<br>RA</th>
    <th rowspan="2">Mean<br>SA</th>
    <th colspan="2">SLF</th>
    <th colspan="2">ALF</th>
    <th colspan="2">AFF</th>
  </tr>
  <tr>
    <th>RA</th>
    <th>SA</th>
    <th>RA</th>
    <th>SA</th>
    <th>RA</th>
    <th>SA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>DynACL</td>
    <td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td>A-InfoNCE</td>
    <td><a href="https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness.">Yu et al., ECCV'22</a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td>AdvCL</td>
    <td><a href="https://arxiv.org/abs/2111.01124">Fan et al., NeurIPS'21</a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>4</td>
    <td>ACL</td>
    <td><a href="https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html">Jiang et al., NeurIPS'20</a></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>

## CIFAR-100 task

## STL10 task

# Performance Benchmarking Across Datasets

## From CIFAR-10 to STL10

## From CIFAR-100 to STL10

