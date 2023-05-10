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
In the directory of ```ACL_Methods```, we cloned the pre-training code of [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning), [RoCL](https://github.com/Kim-Minseon/RoCL), [AdvCL](https://github.com/LijieFan/AdvCL), [A-InfoNCE](https://github.com/yqy2001/A-InfoNCE), [DeACL](https://github.com/pantheon5100/DeACL/tree/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e), and [DynACL](https://github.com/PKU-ML/DYNACL). You can follow the script in their corresponding ```readme.md``` to conduct robust pre-training.

We released all the pre-trained weights in [this Dropbox directory](https://www.dropbox.com/sh/h1hkv3lt2f2zvi2/AACp5IWNaMcqrYAu6hr__4yea?dl=0).
Alternatively, you can copy the address of the ```link``` in the following table and then use ```wget link_address``` to download the specific pre-trained weight.

| Pre-trained weights of ResNet-18 encoder | ACL ([Jiang et al., NeurIPS'20](https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html)) | AdvCL ([Fan et al., NeurIPS'21](https://arxiv.org/abs/2111.01124)) | A-InfoNCE ([Yu et al., ECCV'22](https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness)) | DeACL ([Zhang et al., ECCV'22](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900716.pdf)) | DynACL ([Luo et al., ICLR'23](https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ)) | DynACL++ ([Luo et al., ICLR'23](https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ)) | DynACL-IR ([Xu et al., ArXiv'23](https://arxiv.org/abs/2305.00374)) | DynACL++-IR ([Xu et al., ArXiv'23](https://arxiv.org/abs/2305.00374)) |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | [link<sup>*</sup>](https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL_DS.pt?dl=0) | [link](https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0) | [link](https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/wo1qrrnybycunn3/DeACL.pt?dl=0) | [link<sup>*</sup>](https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0) | [link](https://www.dropbox.com/s/87fhoyrzh33fwrt/DynACL%2B%2B.pt?dl=0) | [link](https://www.dropbox.com/s/z1dcfh0tw7u85iw/DynACL_IR.pt?dl=0) | [link](https://www.dropbox.com/s/rbichiftmu70q6x/DynACL_IR_plus.pt?dl=0) |
| CIFAR-100 | [link<sup>*</sup>](https://www.dropbox.com/s/02hhe679zo6c7da/ACL_DS_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/fcf5y77p87447p9/AdvCL_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/f0ruhgov3dbdcae/AInfoNCE_CIFAR100.pt?dl=0) | - | [link<sup>*</sup>](https://www.dropbox.com/s/mabnrcp7zahp6ke/DynACL_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/hcjol6ihj0en2fn/DynACL%2B%2B_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/1zd6m5pheh5qc6q/DynACL_IR_CIFAR100.pt?dl=0) | [link](https://www.dropbox.com/s/6km587v5bfbugks/DynACL_IR_plus_CIFAR100.pt?dl=0) |
| STL10 | [link](https://www.dropbox.com/s/6jenhn0bpe5ifle/ACL_DS_STL10.pt?dl=0) | - | - | - | [link<sup>*</sup>](https://www.dropbox.com/s/ydd6lbracw73019/DynACL_STL10.pt?dl=0) | [link](https://www.dropbox.com/s/gjxlu7woupqjmr2/DynACL%2B%2B_STL10.pt?dl=0) | [link](https://www.dropbox.com/s/0swf31xq3d1rahn/DynACL_IR_STL10.pt?dl=0) | [link](https://www.dropbox.com/s/vmppj5nhk0rylx6/DyACL_IR_plus_STL10.pt?dl=0) |


**Acknowledgements**: ```*``` denotes the pre-trained encoders haved been provided in their GitHub and we copied them into our Dropbox directory.

# Evaluation Procedure and Metrics

We first conduct finetuning, including standard linear finetuning (SLF), adversarial linear finetuning (ALF), and adversarial full finetuning (AFF) on the downsteam task using the pre-trained models and 

Then, we evaluate the performance of the finetuned model on the downstream task. We use the following 3 types of evaluation metrics:
+ **SA** refers to the standard test accuracy evaluated on the *natural test data*.
+ **RA** refers to the robust test accuracy evaluated on the *adversarial test data* generated via the standard version of AutoAttack.
+ **CA** refers to the mean test accuracy of the *test data under common corruptions* with corruption severity ranging \{1,2,3,4,5\}.
+ **Mean SA/RA/CA** refers to the mean value of SA/RA/CA achieved by 3 types of finetuning.

# Performance Benchmarking Across Tasks
Here, robust pre-training and finetuning are conducted on the same datasets. 

## CIFAR-10 task

<table><thead><tr><th rowspan="2">Rank</th><th rowspan="2">Pre-training</th><th rowspan="2">Reference</th><th rowspan="2">Mean RA</th><th rowspan="2">Mean CA</th><th rowspan="2">Mean SA</th><th colspan="3">SLF</th><th colspan="3">ALF</th><th colspan="3">AFF</th></tr><tr><th>RA</th><th>CA</th><th>SA</th><th>RA</th><th>CA</th><th>SA</th><th>RA</th><th>CA</th><th>SA</th></tr></thead><tbody><tr><td>1</td><td><a href="https://www.dropbox.com/s/rbichiftmu70q6x/DynACL_IR_plus.pt?dl=0">DynACL++-IR</a></td><td><a href="https://arxiv.org/abs/2305.00374">Xu et al., ArXiv'23</a></td><td>48.67</td><td>72.47</td><td>81.24</td><td>46.99</td><td>72.11</td><td>81.80</td><td>48.23</td><td>71.74</td><td>79.56</td><td>50.79</td><td>73.56</td><td>82.36</td></tr><tr><td>2</td><td><a href="https://www.dropbox.com/s/87fhoyrzh33fwrt/DynACL%2B%2B.pt?dl=0">DynACL++</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>48.28</td><td>71.80</td><td>80.19</td><td>46.53</td><td>71.43</td><td>79.82</td><td>47.98</td><td>70.89</td><td>78.81</td><td>50.34</td><td>73.07</td><td>81.93</td></tr><tr><td>3</td><td><a href="https://www.dropbox.com/s/z1dcfh0tw7u85iw/DynACL_IR.pt?dl=0">DynACL-IR</a></td><td><a href="https://arxiv.org/abs/2305.00374">Xu et al., ArXiv'23</a></td><td>47.36</td><td>71.49</td><td>79.41</td><td>45.27</td><td>70.51</td><td>78.08</td><td>46.14</td><td>69.97</td><td>77.42</td><td>50.68</td><td>74.00</td><td>82.74</td></tr><tr><td>4</td><td><a href="https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0">DynACL</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>47.09</td><td>69.39</td><td>76.75</td><td>45.09</td><td>68.67</td><td>75.41</td><td>45.67</td><td>66.69</td><td>72.97</td><td>50.52</td><td>72.81</td><td>81.86</td></tr><tr><td>5</td><td><a href="https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0">AdvCL</a></td><td><a href="https://arxiv.org/abs/2111.01124">Fan et al., NeurIPS'21</a></td><td>45.61</td><td>72.85</td><td>81.80</td><td>43.18</td><td>73.14</td><td>82.36</td><td>44.05</td><td>71.50</td><td>80.04</td><td>49.61</td><td>73.91</td><td>83.00</td></tr><tr><td>6</td><td><a href="https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0">A-InfoNCE</a></td><td><a href="https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness">Yu et al., ECCV'22</a></td><td>45.24</td><td>73.17</td><td>82.33</td><td>42.72</td><td>74.09</td><td>83.70</td><td>43.28</td><td>71.61</td><td>80.30</td><td>49.73</td><td>73.80</td><td>82.99</td></tr><tr><td>7</td><td><a href="https://www.dropbox.com/s/wo1qrrnybycunn3/DeACL.pt?dl=0">DeACL</a></td><td><a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900716.pdf">Zhang et al., ECCV'22</a></td><td>43.49</td><td>71.34</td><td>78.65</td><td>43.27</td><td>73.06</td><td>79.94</td><td>41.99</td><td>71.66</td><td>77.71</td><td>45.21</td><td>69.29</td><td>78.29</td></tr><tr><td>8</td><td><a href="https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL.pt?dl=0">ACL</a></td><td><a href="https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html">Jiang et al., NeurIPS'20</a></td><td>43.13</td><td>70.59</td><td>78.33</td><td>39.17</td><td>70.72</td><td>78.22</td><td>40.60</td><td>68.56</td><td>75.53</td><td>49.62</td><td>72.50</td><td>81.25</td></tr></tbody></table>

## CIFAR-100 task

<table><thead><tr><th rowspan="2">Rank</th><th rowspan="2">Pre-training</th><th rowspan="2">Reference</th><th rowspan="2">Mean RA</th><th rowspan="2">Mean CA</th><th rowspan="2">Mean SA</th><th colspan="3">SLF</th><th colspan="3">ALF</th><th colspan="3">AFF</th></tr><tr><th>RA</th><th>CA</th><th>SA</th><th>RA</th><th>CA</th><th>SA</th><th>RA</th><th>CA</th><th>SA</th></tr></thead><tbody><tr><td>1</td><td><a href="https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0">A-InfoNCE</a></td><td><a href="https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness">Yu et al., ECCV'22</a></td><td>22.02</td><td>42.27</td><td>51.31</td><td>20.41</td><td>41.43</td><td>50.06</td><td>19.53</td><td>37.02</td><td>44.78</td><td>26.11</td><td>48.35</td><td>59.08</td></tr><tr><td>2</td><td><a href="https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0">DynACL</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>21.62</td><td>41.06</td><td>48.95</td><td>19.30</td><td>38.96</td><td>45.83</td><td>20.69</td><td>37.44</td><td>43.58</td><td>24.88</td><td>46.77</td><td>57.43</td></tr><tr><td>3</td><td><a href="https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0">AdvCL</a></td><td><a href="https://arxiv.org/abs/2111.01124">Fan et al., NeurIPS'21</a></td><td>20.85</td><td>39.17</td><td>47.58</td><td>19.10</td><td>36.90</td><td>44.60</td><td>18.19</td><td>33.07</td><td>39.85</td><td>25.26</td><td>47.54</td><td>58.29</td></tr><tr><td>4</td><td><a href="https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL.pt?dl=0">ACL</a></td><td><a href="https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html">Jiang et al., NeurIPS'20</a></td><td>20.38</td><td>40.79</td><td>49.02</td><td>17.30</td><td>38.59</td><td>45.82</td><td>19.07</td><td>36.78</td><td>43.49</td><td>24.78</td><td>47.00</td><td>57.74</td></tr><tr><td>5</td><td><a href="https://www.dropbox.com/s/hcjol6ihj0en2fn/DynACL%2B%2B_CIFAR100.pt?dl=0">DynACL++</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>20.07</td><td>43.46</td><td>51.04</td><td>20.07</td><td>43.46</td><td>52.16</td><td></td><td></td><td>49.92</td><td></td><td></td><td></td></tr></tbody></table>

## STL10 task

<table><thead><tr><th>Rank</th><th>Pre-training</th><th>Reference</th><th>RA</th><th>SA</th></tr></thead><tbody><tr><td>1</td><td><a href="https://www.dropbox.com/s/vmppj5nhk0rylx6/DyACL_IR_plus_STL10.pt?dl=0">DynACL++-IR</a></td><td><a href="https://arxiv.org/abs/2305.00374">Xu et al., ArXiv'23</a></td><td>47.90</td><td>71.44</td></tr><tr><td>2</td><td><a href="https://www.dropbox.com/s/0swf31xq3d1rahn/DynACL_IR_STL10.pt?dl=0">DynACL-IR</a></td><td><a href="https://arxiv.org/abs/2305.00374">Xu et al., ArXiv'23</a></td><td>47.71</td><td>72.25</td></tr><tr><td>3</td><td><a href="https://www.dropbox.com/s/gjxlu7woupqjmr2/DynACL%2B%2B_STL10.pt?dl=0">DynACL++</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>47.24</td><td>70.91</td></tr><tr><td>4</td><td><a href="https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0">DynACL</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>46.61</td><td>69.56</td></tr><tr><td>5</td><td><a href="https://www.dropbox.com/s/6jenhn0bpe5ifle/ACL_STL10.pt?dl=0">ACL</a></td><td><a href="https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html">Jiang et al., NeurIPS'20</a></td><td>32.45</td><td>74.72</td></tr></tbody></table>

# Performance Benchmarking Across Datasets

## From CIFAR-10 to STL10

<table><thead><tr><th rowspan="2">Rank</th><th rowspan="2">Pre-training</th><th rowspan="2">Reference</th><th rowspan="2">Mean RA</th><th rowspan="2">Mean SA</th><th colspan="2">SLF</th><th colspan="2">ALF</th><th colspan="2">AFF</th></tr><tr><th>RA</th><th>SA</th><th>RA</th><th>SA</th><th>RA</th><th>SA</th></tr></thead><tbody><tr><td>1</td><td><a href="https://www.dropbox.com/s/z1dcfh0tw7u85iw/DynACL_IR.pt?dl=0">DynACL-IR</a></td><td><a href="https://arxiv.org/abs/2305.00374">Xu et al., ArXiv'23</a></td><td>32.31</td><td>61.57</td><td>29.64</td><td>60.84</td><td>31.24</td><td>57.14</td><td>36.06</td><td>66.74</td></tr><tr><td>2</td><td><a href="https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0">DynACL</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>31.34</td><td>55.50</td><td>29.17</td><td>52.41</td><td>29.59</td><td>49.55</td><td>35.25</td><td>64.53</td></tr><tr><td>3</td><td><a href="https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0">AdvCL</a></td><td><a href="https://arxiv.org/abs/2111.01124">Fan et al., NeurIPS'21</a></td><td>30.03</td><td>63.22</td><td>28.38</td><td>62.78</td><td>27.88</td><td>58.73</td><td>33.84</td><td>68.14</td></tr><tr><td>4</td><td><a href="https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0">A-InfoNCE</a></td><td><a href="https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness">Yu et al., ECCV'22</a></td><td>29.61</td><td>64.87</td><td>27.18</td><td>65.75</td><td>28.16</td><td>60.86</td><td>33.50</td><td>67.99</td></tr><tr><td>5</td><td><a href="https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL.pt?dl=0">ACL</a></td><td><a href="https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html">Jiang et al., NeurIPS'20</a></td><td>29.14</td><td>56.57</td><td>27.52</td><td>56.50</td><td>27.24</td><td>51.81</td><td>32.66</td><td>61.41</td></tr><tr><td>6</td><td><a href="https://www.dropbox.com/s/wo1qrrnybycunn3/DeACL.pt?dl=0">DeACL</a></td><td><a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900716.pdf">Zhang et al., ECCV'22</a></td><td>16.13</td><td>41.97</td><td>11.04</td><td>38.49</td><td>12.70</td><td>39.41</td><td>24.66</td><td>48.01</td></tr></tbody></table>


## From CIFAR-100 to STL10

<table><thead><tr><th rowspan="2">Rank</th><th rowspan="2">Pre-training</th><th rowspan="2">Reference</th><th rowspan="2">Mean RA</th><th rowspan="2">Mean SA</th><th colspan="2">SLF</th><th colspan="2">ALF</th><th colspan="2">AFF</th></tr><tr><th>RA</th><th>SA</th><th>RA</th><th>SA</th><th>RA</th><th>SA</th></tr></thead><tbody><tr><td>1</td><td><a href="https://www.dropbox.com/s/1zd6m5pheh5qc6q/DynACL_IR_CIFAR100.pt?dl=0">DynACL-IR</a></td><td><a href="https://arxiv.org/abs/2305.00374">Xu et al., ArXiv'23</a></td><td>27.32</td><td>53.43</td><td>23.94</td><td>50.55</td><td>26.60</td><td>48.55</td><td>31.42</td><td>61.19</td></tr><tr><td>2</td><td><a href="https://www.dropbox.com/s/vhxt1hkrtpz2bf9/DynACL.pt?dl=0">DynACL</a></td><td><a href="https://openreview.net/forum?id=0qmwFNJyxCL&noteId=ZXhFXELOcQ">Luo et al., ICLR'23</a></td><td>27.06</td><td>50.70</td><td>23.77</td><td>47.54</td><td>26.24</td><td>45.70</td><td>31.17</td><td>58.85</td></tr><tr><td>3</td><td><a href="https://www.dropbox.com/s/cq8c0a5u06mxnoj/ACL.pt?dl=0">ACL</a></td><td><a href="https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html">Jiang et al., NeurIPS'20</a></td><td>25.02</td><td>49.69</td><td>21.91</td><td>47.59</td><td>24.40</td><td>45.24</td><td>28.76</td><td>56.23</td></tr><tr><td>4</td><td><a href="https://www.dropbox.com/s/bk8eu96ppcj44sz/AInfoNCE.pt?dl=0">A-InfoNCE</a></td><td><a href="https://arxiv.org/abs/2207.08374#:~:text=Contrastive%20learning%20(CL)%20has%20recently,other%2C%20yields%20better%20adversarial%20robustness">Yu et al., ECCV'22</a></td><td>23.24</td><td>54.54</td><td>18.22</td><td>51.14</td><td>20.40</td><td>47.60</td><td>31.10</td><td>64.88</td></tr><tr><td>5</td><td><a href="https://www.dropbox.com/s/fzwg9gcf4ty5oji/AdvCL.pt?dl=0">AdvCL</a></td><td><a href="https://arxiv.org/abs/2111.01124">Fan et al., NeurIPS'21</a></td><td>22.72</td><td>51.86</td><td>18.06</td><td>49.80</td><td>19.60</td><td>44.54</td><td>30.50</td><td>61.24</td></tr></tbody></table>


# References
```

@article{xu2023AIR,
  title={Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization},
  author={Xu, Xilie and Zhang, Jingfeng and Liu, Feng and Sugiyama, Masashi and Kankanhalli, Mohan},
  journal={arXiv preprint arXiv:2305.00374},
  year={2023}
}

@inproceedings{luo2023DynACL,
    title = {Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning},
    author = {Luo, Rundong and Wang, Yifei and Wang, Yisen},
    booktitle = {ICLR},
    year = {2023},
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

@inproceedings{fan2021AdvCL,
  title={When Does Contrastive Learning Preserve Adversarial Robustness from Pretraining to Finetuning?},
  author={Lijie Fan, Sijia Liu, Pin-Yu Chen, Gaoyuan Zhang and Chuang Gan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

@article{ACL2020,
    title = {Robust Pre-Training by Adversarial Contrastive Learning},
    author = {Ziyu Jiang and Tianlong Chen and Ting Chen and Zhangyang Wang},
    booktitle = {Advances in Neural Information Processing Systems 34},
    year = {2020}
    }

@inproceedings{kim2020RoCL,
  title={Adversarial Self-Supervised Contrastive Learning},
  author={Minseon Kim and Jihoon Tack and Sung Ju Hwang},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}

```

# Acknowledgements

The pre-training codes in the directory of ```ACL_Methods``` are provided by [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning), [RoCL](https://github.com/Kim-Minseon/RoCL), [AdvCL](https://github.com/LijieFan/AdvCL), [A-InfoNCE](https://github.com/yqy2001/A-InfoNCE), [DeACL](https://github.com/pantheon5100/DeACL/tree/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e), and [DynACL](https://github.com/PKU-ML/DYNACL). 
