# Training & Eval scripts

#### Training

* For Omniglot/Mini-ImageNet, two-task training.
* For Omniglot/Mini-ImageNet/FC100, three-task training.
* 5-task training for domain incremental settings
* 5-task training for class incremental settings

#### Evaluation

* `eval.sh` for 2-task/3-task general model on their few shot test sets or MNIST/CIFAR10/FashionMNIST etc.

* Split-MNIST, domain incremental: `eval_splitmnist_domain_incremental.sh` (single script for both OOB and meta-finetuned models)

* Split-MNIST, class incremental: `eval_splitmnist_class_incremental.sh` and `eval_splitmnist_class_incremental_2task_outofbox.sh`
(note that out-of-the-box models can only be evaluated for the 2-task setting)

