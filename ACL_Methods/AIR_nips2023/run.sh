### pre-training ###


### ACL with IR ###
python pretraining.py ACL_IR_cifar10 --lambda1 0.5 --lambda2 0.5 --dataset cifar10 --ACL_DS
python pretraining.py ACL_IR_cifar100 --lambda1 0.5 --lambda2 0.5 --dataset cifar100 --ACL_DS
python pretraining.py ACL_IR_stl10 --lambda1 0.5 --lambda2 0.5 --dataset stl10 --ACL_DS

### DynACL with IR ###
python pretraining.py DynACL_IR_cifar10 --lambda1 0.5 --lambda2 0.5 --dataset cifar10 --ACL_DS --DynAug
python pretraining.py DynACL_IR_cifar100 --lambda1 0.5 --lambda2 0.5 --dataset cifar100 --ACL_DS --DynAug
python pretraining.py DynACL_IR_stl10 --lambda1 0.5 --lambda2 0.5 --dataset stl10 --ACL_DS --DynAug

