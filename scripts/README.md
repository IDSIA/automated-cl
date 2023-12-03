# Training & Eval scripts

#### Training

* For Omniglot/Mini-ImageNet, two-task training.
```
SEED=1

export CUDA_VISIBLE_DEVICES=0

export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions"

CODE=
DATA=

python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 32 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 15 \
  --test_per_class 1 \
  --use_warmup \
  --use_wandb \
  --ood_eval \
  --ood_eval2 \
  --ood_eval3 \
  --extra_label \
  --use_fs \
  --use_ab_v2 \
  --use_acl \
  --use_instance_norm \
  --loss_scale_task_a 0.1 \
  --use_cache \
  --project_name 'my_project' \
```

* For Omniglot/Mini-ImageNet/FC100, three-task training.
```
python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 32 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 15 \
  --test_per_class 1 \
  --use_warmup \
  --use_wandb \
  --ood_eval \
  --ood_eval2 \
  --ood_eval3 \
  --extra_label \
  --use_fs \
  --use_abc_v2 \
  --use_acl \
  --use_instance_norm \
  --loss_scale_task_a 1 \
  --use_cache \
  --project_name 'my_project' \
```

* 5-task training for domain incremental settings
```
# out of the box model path:
OB_MODEL = 

python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --train_splitmnist_style \
  --init_model_except_output_from ${OB_MODEL} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 64 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 5 \
  --n_way 2 \
  --report_every 10 \
  --validate_every 50 \
  --test_per_class 1 \
  --extra_label \
  --use_fs \
  --use_acl \
  --use_instance_norm \
  --use_cache \
  --project_name 'my_project' \
```

* 5-task training for class incremental settings
```
python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --train_splitmnist_style_class_incremental \
  --init_model_except_output_from_class_incremental ${OB_MODEL} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 64 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 5 \
  --n_way 10 \
  --report_every 10 \
  --validate_every 50 \
  --test_per_class 1 \
  --extra_label \
  --use_fs \
  --use_acl \
  --use_instance_norm \
  --use_cache \
  --project_name 'my_project' \
```

#### Evaluation

* Split-MNIST, domain incremental
```
# trained model path:
OB_MODEL = 

python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --eval_splitmnist \
  --eval_only_dir ${OB_MODEL} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 32 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 15 \
  --n_way 5 \
  --test_per_class 1 \
  --extra_label \
  --use_fs \
  --use_ab_v2 \
  --use_acl \
  --use_instance_norm \
  --use_cache \
```

* Split-MNIST, class incremental, 2-task
```
python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --eval_splitmnist_incremental_class_2task \
  --eval_only_dir ${OB_MODEL} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 32 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 15 \
  --n_way 5 \
  --test_per_class 1 \
  --extra_label \
  --use_fs \
  --use_ab_v2 \
  --use_acl \
  --use_instance_norm \
  --use_cache \
```

* Split-MNIST, class incremental, 5-task

Replace `--eval_splitmnist_incremental_class_2task` by `--eval_splitmnist_incremental_class`

* eval on 2-task few-shot learning test sets (Omniglot/Mini-ImageNet)

```
# trained model path:
OB_MODEL = 

python3 ${CODE}/main.py \
  --data_dir ${DATA} \
  --eval_only_dir ${OB_MODEL} \
  --name_dataset miniimagenet_32_norm_cache \
  --seed ${SEED} \
  --num_worker 12 \
  --test_per_class 1 \
  --model_type 'compat_stateful_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 32 \
  --num_layer 2 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 2 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 15 \
  --test_per_class 1 \
  --ood_eval \
  --ood_eval2 \
  --ood_eval3 \
  --extra_label \
  --use_fs \
  --use_ab_v2 \
  --use_acl \
  --use_instance_norm \
  --loss_scale_task_a 0.1 \
  --use_cache \
```

* To the above, add `--eval_extra_only` for evaluation on MNIST/CIFAR10 or `--eval_extra_4_tasks` 

