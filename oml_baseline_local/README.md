## OML baselines

We forked and modified from the official code repository: https://github.com/khurramjaved96/mrcl

We refer to the original repo for the requirements etc. The out-of-the-box model checkpoint can also be downloaded from the link in their repository (`omniglot_oml.model` bug-free omniglot one)

There are two separate executable files (modified from `oml_omniglot.py`) for the domain-incremental and class-incremental cases:
* `evaluate_splitmnist_domain_incremental.py`
* `evaluate_splitmnist_class_incremental.py`

In the current version of the code, we manually change:
* `NUM_ITER`
* `best_lr`

to tune these meta-test hyper-parameters.

For the class-incremental case, we have in addition:
* `NUM_TASKS`
to be set to either `2` or `5` to produce 2-task and 5-task results.

LICENSE of the original code: authors of https://github.com/khurramjaved96/mrcl
