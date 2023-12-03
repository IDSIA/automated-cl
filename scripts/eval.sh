#!/bin/bash
SEED=1

export CUDA_VISIBLE_DEVICES=0

export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions"

CODE=
DATA=

# trained model path:
# for example: ../automated_cl_checkpoints_Nov2023/2task/best_model.pt
OB_MODEL= 

# change --use_ab_v2 to --use_abc_v2 for 3 task model eval

# Add --eval_extra_only for evaluation on MNIST/CIFAR10
# --eval_extra_4_tasks for 4 task eval

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
