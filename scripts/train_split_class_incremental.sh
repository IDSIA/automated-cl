#!/bin/bash
SEED=1

export CUDA_VISIBLE_DEVICES=0

export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions"

CODE=
DATA=

# out of the box model path:
# for example: ../automated_cl_checkpoints_Nov2023/2task/best_model.pt
OB_MODEL= 

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
  --use_ab_v2 \
  --use_acl \
  --use_instance_norm \
  --use_cache \
  --project_name 'my_project' \
