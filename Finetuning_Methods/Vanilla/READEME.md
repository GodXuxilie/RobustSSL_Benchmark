# Vanilla Fine-Tuning

The vanilla fine-tuning utilizes standard training, [standard adversarial training](https://github.com/MadryLab/mnist_challenge), and [TRADES](https://github.com/yaodongyu/TRADES) in the SLF, ALF, and AFF mode, respectively.

To obtain satisfactory performance in downstream tasks, you need to modify the the function ```setup_hyperparameter(args,mode)``` of the file ```utils.py``` where you can specify hyper-parameters such as the initial learning rate (LR), the scheduler of the LR, the weight decay, the batch size, <i>etc</i>.

