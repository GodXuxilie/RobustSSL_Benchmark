# Benchmarking Robustness Transferability of Adversarial Contrastive Learning (ACL)

This repository provides ***a unified finetuning and evaluation tool*** to evaluate the performance of finetuned models on the downstream tasks.

```
python finetuning_eval.py   --gpu GPU_id \
                            --experiment path_of_log_to_be_saved \
                            --pretraining  pre_training_method \
                            --model type_of_backbone_network \
                            --checkpoint path_of_pretrained_checkpoint \ 
                            --dataset downstream_dataset: [cifar10, cifar100, stl10] \ 
                            --mode finetuning_mode: [Ensemble, SLF, ALF, AFF] \
                            --eval-AA \
                            --eval-OOD 
```

# Requirement
+ Python 3.8
+ Pytorch 1.13
+ CUDA 11.6
+ [AutoAttack](https://github.com/fra31/auto-attack) (Install AutoAttack via ```pip install git+https://github.com/fra31/auto-attack```)
+ [robustbench](https://robustbench.github.io/) (Install robustbench via ```pip install git+https://github.com/RobustBench/robustbench.git```)


# ACL pre-training
In the directory of ```Pretraining```, we cloned the pre-training code of [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning), [RoCL](https://github.com/Kim-Minseon/RoCL), [AdvCL](https://github.com/LijieFan/AdvCL), [A-InfoNCE](https://github.com/yqy2001/A-InfoNCE), [DeACL](https://github.com/pantheon5100/DeACL/tree/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e), and [DynACL](https://github.com/PKU-ML/DYNACL). You can follow the script in their corresponding ```readme.md``` to conduct robust pre-training.

We released all the pre-trained weights in [this Dropbox directory](https://www.dropbox.com/sh/h1hkv3lt2f2zvi2/AACp5IWNaMcqrYAu6hr__4yea?dl=0).
Alternatively, you can copy the address of the ```link``` in the following table and then use ```wget link_address``` to download the specific pre-trained weight.

| Weights of pre-trained ResNet-18 | ACL | AdvCL | A-InfoNCE | DeACL | DynACL |
|---|---|---|---|---|---|
| CIFAR-10 | [link<sup>*</sup>](https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL_DS.pt?dl=0) | [link](https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0) | [link](https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/wo1qrrnybycunn3/DeACL.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0) |
| CIFAR-100 | [link<sup>*</sup>](https://www.dropbox.com/s/02hhe679zo6c7da/ACL_DS_CIFAR100.pt?dl=0) | [link]() | [link]() | - | [link<sup>*</sup>](https://www.dropbox.com/s/mabnrcp7zahp6ke/DynACL_CIFAR100.pt?dl=0) |
| STL10 | [link](https://www.dropbox.com/s/6jenhn0bpe5ifle/ACL_DS_STL10.pt?dl=0) | - | - | - | [link<sup>*</sup>](https://www.dropbox.com/s/ydd6lbracw73019/DynACL_STL10.pt?dl=0) |

Acknowledgement: ```*``` denotes the pre-trained encoders haved been provided in their GitHub and we copied them into our Dropbox directory.

More pre-trained weights will be coming soon!


# Performance Benchmarking Across Tasks
Here, robust pre-training and finetuning are conducted on the same datasets. 

## CIFAR-10 task

### Robustness against adversarial attacks
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

### Robustness against common corruptions

<table>
<thead>
  <tr>
    <th rowspan="2">Rank</th>
    <th rowspan="2">Pre-training</th>
    <th rowspan="2">Reference</th>
    <th rowspan="2">Mean<br>Acc</th>
    <th colspan="6">SLF</th>
    <th colspan="6">ALF</th>
    <th colspan="6">AFF</th>
  </tr>
  <tr>
    <th>Mean</th>
    <th>CS-1</th>
    <th>CS-2</th>
    <th>CS-3</th>
    <th>CS-4</th>
    <th>CS-5</th>
    <th>Mean</th>
    <th>CS-1</th>
    <th>CS-2</th>
    <th>CS-3</th>
    <th>CS-4</th>
    <th>CS-5</th>
    <th>Mean</th>
    <th>CS-1</th>
    <th>CS-2</th>
    <th>CS-3</th>
    <th>CS-4</th>
    <th>CS-5</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>ACL</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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
    <td>AdvCL</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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
    <td>A-InfoNCE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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
    <td>DynACL</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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

### Robustness against adversarial attacks
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

### Robustness against common corruptions

<table>
<thead>
  <tr>
    <th rowspan="2">Rank</th>
    <th rowspan="2">Pre-training</th>
    <th rowspan="2">Reference</th>
    <th rowspan="2">Mean<br>Acc</th>
    <th colspan="6">SLF</th>
    <th colspan="6">ALF</th>
    <th colspan="6">AFF</th>
  </tr>
  <tr>
    <th>Mean</th>
    <th>CS-1</th>
    <th>CS-2</th>
    <th>CS-3</th>
    <th>CS-4</th>
    <th>CS-5</th>
    <th>Mean</th>
    <th>CS-1</th>
    <th>CS-2</th>
    <th>CS-3</th>
    <th>CS-4</th>
    <th>CS-5</th>
    <th>Mean</th>
    <th>CS-1</th>
    <th>CS-2</th>
    <th>CS-3</th>
    <th>CS-4</th>
    <th>CS-5</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>ACL</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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
    <td>AdvCL</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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
    <td>A-InfoNCE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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
    <td>DynACL</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
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

## STL10 task
### Robustness against adversarial attacks
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


# Performance Benchmarking Across Datasets

## From CIFAR-10 to STL10

### Robustness against adversarial attacks
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

## From CIFAR-100 to STL10

### Robustness against adversarial attacks
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
