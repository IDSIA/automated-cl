# Training & Eval scripts

#### Training

* For Omniglot/Mini-ImageNet, two-task training: `train_2tasks.sh`

* For Omniglot/Mini-ImageNet/FC100, three-task training: `train_3tasks.sh`

* 5-task training for domain incremental settings: `train_split_domain_incremental.sh`

* 5-task training for class incremental settings: `train_split_class_incremental.sh`

#### Evaluation

* `eval.sh` for 2-task/3-task general model on their few shot test sets or MNIST/CIFAR10/FashionMNIST etc.

* Split-MNIST, domain incremental: `eval_splitmnist_domain_incremental.sh` and `eval_splitmnist_domain_incremental_outofbox.sh`

* Split-MNIST, class incremental: `eval_splitmnist_class_incremental.sh` and `eval_splitmnist_class_incremental_2task_outofbox.sh`
(note that out-of-the-box models can only be evaluated for the 2-task setting)

