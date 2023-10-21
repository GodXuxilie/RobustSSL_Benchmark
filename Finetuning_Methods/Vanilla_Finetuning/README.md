# Vanilla Fine-Tuning

The vanilla fine-tuning utilizes standard training, [standard adversarial training](https://github.com/MadryLab/mnist_challenge), and [TRADES](https://github.com/yaodongyu/TRADES) in the SLF, ALF, and AFF mode, respectively.

To obtain satisfactory performance in downstream tasks, you need to modify the the function ```setup_hyperparameter(args,mode)``` of the file [```utils.py```](../AutoLoRa/utils.py) where you can specify hyper-parameters such as the initial learning rate (LR), the scheduler of the LR, the weight decay, the batch size, <i>etc</i>.

We provide the running script in [```run_vanilla_tune.sh```](./run_vanilla_tune.sh).

If you would like to fine-tuning your own pre-trained weights, please use the following script:
```
python vanilla_finetuning.py --gpu gpu_id
                             --experiment exp_name
                             --dataset downstream_dataset: ['cifar10', 'cifar100', 'stl10'] 
                             --pretraining pre_training_method_name: ['ACL', 'AdvCL', 'A-InfoNCE', 'DeACL', 'DynACL', 'DynACL++', 'DynACL_AIR', 'DynACL_AIR++', 'DynACL_RCS'] 
                             --model model_arch: ['r18', 'r34', 'r50'] 
                             --checkpoint path_of_checkpoint
                             --mode finetuning_mode: ['ALL', 'SLF', 'ALF', 'AFF']
                             --eval-AA 
                             --eval-OOD
```