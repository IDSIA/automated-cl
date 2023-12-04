#!/bin/bash
# 2-task class-incremental with out-of-the-box model

SEED=1

export CUDA_VISIBLE_DEVICES=0

export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions"

CODE=
DATA=

# trained model path:
# for example: ../automated_cl_checkpoints_Nov2023/2task
OB_MODEL=

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
