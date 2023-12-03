# Main file to be executed
# NB: current version of the code is not at all optimized;
# e.g., the ACL loss computation can be largely accelerated by batchfying computation

import os
import sys
import json
import time
import hashlib
from datetime import datetime
import argparse
import logging
import numpy as np
import random
from packaging import version
from itertools import cycle

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, Normalize

from torchmeta_local.utils.data import BatchMetaDataLoader
from warmup_lr import WarmupWrapper

from model_few_shot import (
    ConvLSTMModel, ConvDeltaModel, ConvSRWMModel,
    Res12LSTMModel, Res12DeltaModel, Res12SRWMModel,
    MixerSRWMModel, SRMixerModel,
    CompatStatefulMixerSRWMModel, CompatStatefulSelfModMixerModel,
    CompatStatefulConvSRWMModel, StatefulConvDeltaModel)
from utils_few_shot import eval_model_label_sync, eval_acl_ab_model_label_sync


def zero_div(nume, denom):
    return nume / denom if denom else 0.0


parser = argparse.ArgumentParser(
    description='N-way K-shot learning based on label synchronous '
                'seq-processing NNs with only predicting (N*K+1)th image.')
parser.add_argument('--data_dir', type=str,
                    default='./data', help='location of the data corpus')
parser.add_argument('--name_dataset', type=str, default='omniglot',
                    choices=['omniglot', 'miniimagenet', 'omniglot_rgb84x84',
                             'omniglot_rgb84x84_norm', 'omniglot_norm', 'omniglot_32_norm',
                             'miniimagenet_norm', 'tieredimagenet',
                             'cifar_fs', 'cifar_fs_norm', 'cifar_fs_rfs',
                             'miniimagenet_32_norm_cache',
                             'fc100', 'fc100_norm', 'fc100_rfs', 'miniimagenet_32_norm'])
parser.add_argument('--num_worker', default=12, type=int,
                    help='for dataloader.')
parser.add_argument('--work_dir', default='save_models', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--init_model_from', default=None, type=str,
                    help='e.g. save_models/aaa/best_model.pt.')
parser.add_argument('--init_model_except_output_from', default=None, type=str,
                    help='e.g. save_models/aaa/best_model.pt.')
parser.add_argument('--init_model_except_output_from_class_incremental', default=None, type=str,
                    help='e.g. save_models/aaa/best_model.pt.')

parser.add_argument('--model_type', type=str, default='lstm',
                    choices=['lstm', 'deltanet', 'srwm',
                             'res12_lstm', 'res12_deltanet', 'res12_srwm',
                             'mixer_srwm', 'srwm_mixer',
                             'compat_stateful_srwm_res12', 'compat_stateful_srwm_mixer',
                             'compat_stateful_self_mod_mixer',
                             'compat_stateful_srwm', 'stateful_deltanet'],
                    help='0: LSTM, 1: DeltaNet, 2: SRWM')
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--valid_seed', default=0, type=int, help='Seed.')
parser.add_argument('--test_seed', default=0, type=int, help='Seed.')
parser.add_argument('--fixed_valid', action='store_true',
                    help='use fixed validation set.')
parser.add_argument('--fixed_test', action='store_true',
                    help='[debug mode] use fixed test set.')
parser.add_argument('--num_test', default=10, type=int,
                    help='test size.')
parser.add_argument('--total_epoch', default=1, type=int,
                    help='iterate more than one epoch.')
parser.add_argument('--train_acc_stop', default=120, type=int,
                    help='stopping based on train acc.')
parser.add_argument('--ood_eval', action='store_true',
                    help='eval on extra unrelated set.')
parser.add_argument('--ood_eval2', action='store_true',
                    help='eval on extra unrelated set.')
parser.add_argument('--ood_eval3', action='store_true',
                    help='eval on extra unrelated set.')
parser.add_argument('--ood_eval5', action='store_true',
                    help='fashion mnist.')
parser.add_argument('--extra_label', action='store_true',
                    help='eval on extra unrelated set.')
parser.add_argument('--use_84', action='store_true',
                    help='eval on extra unrelated set.')
parser.add_argument('--use_cache', action='store_true',
                    help='eval on extra unrelated set.')
parser.add_argument('--use_fc', action='store_true',
                    help='use fc100 for ab.')
parser.add_argument('--disable_ct', action='store_true',
                    help='train in non-sequential mode.')
parser.add_argument('--disable_multi', action='store_true',
                    help='train in single-task mode.')
parser.add_argument('--loss_scale_task_a', default=1., type=float,
                    help='multiplier for all losses for TASK A.')
parser.add_argument('--prioritize_last', default=1., type=float,
                    help='multiplier for all other losses than the last task.')
parser.add_argument('--prioritize_last_factor', default=1., type=float,
                    help='multiplier for all other losses than the last task.')
parser.add_argument('--ab_acl_scaler', default=1., type=float,
                    help='multiplier for ab acl losses than the last task.')
parser.add_argument('--scale_first', default=1., type=float,
                    help='multiplier for the first task.')
parser.add_argument('--drop_last_batch', action='store_true',
                    help='dataloader.')
parser.add_argument('--cycle_dataloader', action='store_true',
                    help='cycle dataloader.')
parser.add_argument('--eval_only_dir', default=None,
                    help='skip training and eval ckpts in dir.')
parser.add_argument('--eval_extra_only', action='store_true',
                    help='train in single-task mode.')
parser.add_argument('--eval_extra_4_tasks', action='store_true',
                    help='4 task eval.')

# split mnist
parser.add_argument('--eval_splitmnist', action='store_true',
                    help='train in single-task mode.')
parser.add_argument('--eval_splitfashion', action='store_true',
                    help='train in single-task mode.')
parser.add_argument('--eval_splitcifar10', action='store_true',
                    help='train in single-task mode.')
parser.add_argument('--eval_splitmnist_incremental_class', action='store_true',
                    help='train in single-task mode.')
parser.add_argument('--eval_splitmnist_incremental_class_2task', action='store_true',
                    help='train in single-task mode.')

# model hyper-parameters:
parser.add_argument('--num_layer', default=1, type=int,
                    help='number of layers. for both LSTM and Trafo.')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--n_head', default=8, type=int,
                    help='Transformer number of heads.')
parser.add_argument('--ff_factor', default=4, type=int,
                    help='Transformer ff dim to hidden dim ratio.')
parser.add_argument('--ff_factor_tk', default=0.5, type=float,
                    help='mixer token proj ff dim to hidden dim ratio.')
parser.add_argument('--patch_size', default=16, type=int,
                    help='mixer, patch size.')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout rate.')
parser.add_argument('--state_dropout', default=0.0, type=float,
                    help='state reset rate.')
parser.add_argument('--input_dropout', default=0.0, type=float,
                    help='input dropout rate.')
parser.add_argument('--vision_dropout', default=0.0, type=float,
                    help='dropout rate in the vision feat extractor.')
parser.add_argument('--dropout_type', type=str, default='base',
                    choices=['base', 'inblock', '2d', '2d_inblock'])
parser.add_argument('--use_big_res12', action='store_true',
                    help='use big Res-12.')
parser.add_argument('--not_use_ln', action='store_true',
                    help='not use layer norm.')
parser.add_argument('--not_use_res', action='store_true',
                    help='not use residual connections.')
parser.add_argument('--not_use_ff', action='store_true',
                    help='not use ff block.')
parser.add_argument('--srwm_beta_init', default=0.0, type=float,
                    help='beta bias for srwm.')
parser.add_argument('--srwm_init_scaler', default=1.0, type=float,
                    help='init for srwm.')
parser.add_argument('--srwm_q_init_scaler', default=0.01, type=float,
                    help='q init for srwm.')
parser.add_argument('--unif_init', action='store_true',
                    help='use unif for init.')
parser.add_argument('--use_input_softmax', action='store_true',
                    help='input softmax for srwm.')
parser.add_argument('--context_carry_over', action='store_true',
                    help='state carry over.')
parser.add_argument('--context_carry_over_k', default=1, type=int)
parser.add_argument('--context_carry_over_double', action='store_true',
                    help='state carry over two segments.')
parser.add_argument('--single_state_training', action='store_true',
                    help='carry state from batch 0.')
parser.add_argument('--no_softmax_on_y', action='store_true',
                    help='srwm; apply no softmax on y.')
parser.add_argument('--remove_bn', action='store_true',
                    help='remove bn in certain models.')
parser.add_argument('--use_instance_norm', action='store_true',
                    help='use InstanceNorm2d in certain models.')
parser.add_argument('--no_load_optimizer', action='store_true',
                    help='use InstanceNorm2d in certain models.')

# few shot learning setting
parser.add_argument('--n_way', default=5, type=int,
                    help='number of possible classes per train/test episode.')
parser.add_argument('--k_shot', default=1, type=int,
                    help='number of examples in the `train` part of torchmeta')
parser.add_argument('--test_per_class', default=1, type=int,
                    help='param for torchmeta')
parser.add_argument('--use_fs', action='store_true',
                    help='auxiliary first shot.')

# use automated continual learning loss
parser.add_argument('--use_ab', action='store_true',
                    help='in-context-train/test on a then b.')
parser.add_argument('--old_use_ab', action='store_true',
                    help='in-context-train/test on a then b.')
parser.add_argument('--use_ab_v2', action='store_true',
                    help='another variant of above.')
parser.add_argument('--use_abc_v2', action='store_true',
                    help='another variant of above.')
parser.add_argument('--use_b_first', action='store_true',
                    help='in-context-train/test on b then a.')
parser.add_argument('--use_abab', action='store_true')  # TODO
parser.add_argument('--use_acl', action='store_true',
                    help='use the ACL loss.')

parser.add_argument('--train_splitmnist_style', action='store_true',
                    help='domain incremental.')
parser.add_argument('--train_splitmnist_style_class_incremental', action='store_true',
                    help='class incremental.')
parser.add_argument('--metaval_fashion', action='store_true',
                    help='another variant of above.')
parser.add_argument('--metaval_cifar', action='store_true',
                    help='another variant of above.')
parser.add_argument('--mix_metafinetuning', action='store_true',
                    help='use om and im for splitmnist style training.')

# training hyper-parameters:
parser.add_argument('--total_train_steps', default=100000, type=int,
                    help='Number of training steps to train on')
parser.add_argument('--valid_size', default=100, type=int,
                    help='Number of valid batches to validate on')
parser.add_argument('--test_size', default=100, type=int,
                    help='Number of test batches to test on')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size.')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='batch size.')
parser.add_argument('--warmup_steps', default=5000, type=int)
parser.add_argument('--freeze_after_steps', default=200000, type=int)
parser.add_argument('--freeze_after', action='store_true',
                    help='freeze the conv stem.')
parser.add_argument('--freeze_out_emb', action='store_true',
                    help='freeze the output embeddings.')
parser.add_argument('--use_radam', action='store_true',
                    help='use radam.')
parser.add_argument('--use_sgd', action='store_true',
                    help='use radam.')
parser.add_argument('--use_adamw', action='store_true',
                    help='use radam.')
parser.add_argument('--use_dropblock', action='store_true',
                    help='use dropblock.')
parser.add_argument('--sgd_momentum', default=0.9, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay term.')
parser.add_argument('--use_exp', action='store_true',
                    help='use exp warm up.')
parser.add_argument('--use_warmup', action='store_true',
                    help='use warmup scheduling.')
parser.add_argument('--grad_cummulate', default=1, type=int,
                    help='number of gradient accumulation steps.')
parser.add_argument('--report_every', default=100, type=int,
                    help='Report log every this steps (not used).')
parser.add_argument('--validate_every', default=1000, type=int,
                    help='Report log every this steps (not used).')
parser.add_argument('--clip', default=0.0, type=float,
                    help='global norm clipping threshold.')
parser.add_argument('--job_id', default=0, type=int)
# for wandb
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

model_name = args.model_type

exp_str = ''
for arg_key in vars(args):
    exp_str += str(getattr(args, arg_key)) + '-'

# taken from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
exp_hash = str(int(hashlib.sha1(exp_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
job_id = args.job_id

# Set work directory
args.work_dir = os.path.join(
    args.work_dir, f"{job_id}-{exp_hash}-{time.strftime('%Y%m%d-%H%M%S')}")
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

work_dir_key = '/'.join(os.path.abspath(args.work_dir).split('/')[-3:])

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
loginf(f"Work dir: {args.work_dir}")

# wandb settings
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True
    # fix to remove extra HTTPS connection logs
    # https://stackoverflow.com/questions/11029717/how-do-i-disable-log-messages-from-the-requests-library
    logging.getLogger("requests").setLevel(logging.WARNING)

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(
        project=project_name, settings=wandb.Settings(start_method='fork'))
    # or `settings=wandb.Settings(start_method='thread')`
    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//" \
                         f"{model_name}-{args.name_dataset}//" \
                         f"seed{args.seed}//radam{args.use_radam}/" \
                         f"wd{args.weight_decay}/ip{args.input_dropout}/" \
                         f"{args.dropout_type}/ls{args.label_smoothing}/" \
                         f"adamw{args.use_adamw}/dropb{args.use_dropblock}/" \
                         f"freeze{args.freeze_after}/e{args.freeze_out_emb}/" \
                         f"use_warm{args.use_warmup}/exp{args.use_exp}/" \
                         f"psz{args.patch_size}/tk{args.ff_factor_tk}/" \
                         f"fzstep{args.freeze_after_steps}/" \
                         f"{args.test_per_class}-test_per_cl/" \
                         f"{args.n_way}way-{args.k_shot}shot/" \
                         f"L{args.num_layer}/h{args.hidden_size}/" \
                         f"n{args.n_head}/ff{args.ff_factor}/" \
                         f"d{args.dropout}/vd{args.vision_dropout}/" \
                         f"bigres{args.use_big_res12}/b{args.batch_size}/" \
                         f"lr{args.learning_rate}/warm{args.warmup_steps}" \
                         f"g{args.grad_cummulate}/bias{args.srwm_beta_init}" \
                         f"softmax{args.use_input_softmax}" \
                         f"//PATH'{work_dir_key}'//"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.seed = args.seed
    config.test_per_class = args.test_per_class
    config.extra_label = args.extra_label
    config.use_ab = args.use_ab
    config.use_ab_v2 = args.use_ab_v2
    config.use_abc_v2 = args.use_abc_v2
    config.disable_ct = args.disable_ct
    config.disable_multi = args.disable_multi
    config.use_fs = args.use_fs
    config.use_fc = args.use_fc
    config.use_cache = args.use_cache
    config.use_acl = args.use_acl
    config.loss_scale_task_a = args.loss_scale_task_a
    config.use_b_first = args.use_b_first
    config.remove_bn = args.remove_bn
    config.use_instance_norm = args.use_instance_norm
    config.n_way = args.n_way
    config.k_shot = args.k_shot
    config.srwm_beta_init = args.srwm_beta_init
    config.use_input_softmax = args.use_input_softmax
    config.context_carry_over = args.context_carry_over
    config.context_carry_over_double = args.context_carry_over_double
    config.context_carry_over_k = args.context_carry_over_k
    config.single_state_training = args.single_state_training
    config.name_dataset = args.name_dataset
    config.work_dir = args.work_dir
    config.model_type = args.model_type
    config.hidden_size = args.hidden_size
    config.n_head = args.n_head
    config.ff_factor = args.ff_factor
    config.dropout = args.dropout
    config.vision_dropout = args.vision_dropout
    config.use_big_res12 = args.use_big_res12
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.freeze_after = args.freeze_after
    config.freeze_out_emb = args.freeze_out_emb
    config.freeze_after_steps = args.freeze_after_steps
    config.grad_cummulate = args.grad_cummulate
    config.use_radam = args.use_radam
    config.use_sgd = args.use_sgd
    config.use_adamw = args.use_adamw
    config.sgd_momentum = args.sgd_momentum
    config.input_dropout = args.input_dropout
    config.dropout_type = args.dropout_type
    config.use_dropblock = args.use_dropblock
    config.weight_decay = args.weight_decay
    config.label_smoothing = args.label_smoothing
    config.report_every = args.report_every
    config.not_use_ln = args.not_use_ln
    config.not_use_res = args.not_use_res
    config.not_use_ff = args.not_use_ff
    config.patch_size = args.patch_size
    config.ff_factor_tk = args.ff_factor_tk
else:
    use_wandb = False
# end wandb

# save args
loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

with open(f'{args.work_dir}/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# set seed
loginf(f"Seed: {args.seed}")
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

valid_seed = args.valid_seed
test_seed = args.test_seed
loginf(f"Valid seed: {valid_seed}, Test seed: {test_seed}")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# set dataset
batch_size = args.batch_size
n_way = args.n_way
k_shot_train = args.k_shot
test_per_class = args.test_per_class

# Let's hard code this
args.drop_last_batch = True

if args.use_ab or args.use_ab_v2:
    if args.use_cache:
        if args.use_fc:
            if args.use_b_first:
                loginf(f"A-B training on miniimagenet_32_norm_cache and fc100_norm")
                from torchmeta_local.datasets.helpers import fc100_norm as data_cls_b
                from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_a
            else:
                loginf(f"A-B training on fc100_norm and miniimagenet_32_norm_cache")
                from torchmeta_local.datasets.helpers import fc100_norm as data_cls_a
                from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_b
        else:
            if args.use_b_first:
                loginf(f"A-B training on miniimagenet_32_norm_cache and omniglot_32_norm")
                from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_b
                from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_a 
            else:
                loginf(f"A-B training on omniglot_32_norm and miniimagenet_32_norm_cache")
                from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a
                from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_b 
    elif args.use_84:
        loginf(f"A-B training on omniglot_84_norm and miniimagenet_84_norm")
        from torchmeta_local.datasets.helpers import omniglot_rgb84x84_norm as data_cls_a
        from torchmeta_local.datasets.helpers import miniimagenet_norm as data_cls_b
    else:
        if args.use_fc:
            loginf(f"A-B training on fc100_norm and miniimagenet_32_norm")
            from torchmeta_local.datasets.helpers import fc100_norm as data_cls_a
            from torchmeta_local.datasets.helpers import miniimagenet_32_norm as data_cls_b
        else:
            loginf(f"A-B training on omniglot_32_norm and miniimagenet_32_norm")
            from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a
            from torchmeta_local.datasets.helpers import miniimagenet_32_norm as data_cls_b    

    # use first shot loss
    use_fs = args.use_fs

    if args.use_fs:
        assert model_name in [
            'compat_stateful_srwm', 'stateful_deltanet', 'compat_stateful_deltanet',
            'compat_stateful_srwm_res12', 'compat_stateful_srwm_mixer',
            'compat_stateful_self_mod_mixer']
        num_samples_per_class = {
            'first_shot': 1, 'train': k_shot_train, 'test': test_per_class}
        dataset_a = data_cls_a(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
        dataset_b = data_cls_b(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
    else:
        dataset_a = data_cls_a(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed)
        dataset_b = data_cls_b(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed)

    dataloader_a = BatchMetaDataLoader(
        dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)
    dataloader_b = BatchMetaDataLoader(
        dataset_b, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    # valid
    val_dataset_a = data_cls_a(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_val=True,
                        shuffle=True, seed=valid_seed)  # fixed validation set
    val_dataset_b = data_cls_b(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_val=True,
                        shuffle=True, seed=valid_seed)  # fixed validation set

    val_dataloader_a = BatchMetaDataLoader(
        val_dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    val_dataloader_b = BatchMetaDataLoader(
        val_dataset_b, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    # test
    test_dataset_a = data_cls_a(
        args.data_dir, ways=n_way, shots=k_shot_train,
        test_shots=test_per_class, meta_test=True,
        download=True, shuffle=True, seed=test_seed)

    test_dataset_b = data_cls_b(
        args.data_dir, ways=n_way, shots=k_shot_train,
        test_shots=test_per_class, meta_test=True,
        download=True, shuffle=True, seed=test_seed)

    test_dataloader_a = BatchMetaDataLoader(
        test_dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    test_dataloader_b = BatchMetaDataLoader(
        test_dataset_b, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    if args.cycle_dataloader:
        zip_dataloader_a_b = zip(cycle(dataloader_a), cycle(dataloader_b))
    else:
        zip_dataloader_a_b = zip(dataloader_a, dataloader_b)

    # end use_ab or use_ab_v2
elif args.use_abc_v2:
    loginf(f"A-B-C training on omniglot_32_norm, miniimagenet_32_norm_cache and fc100_norm")
    from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a
    from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_b 
    from torchmeta_local.datasets.helpers import fc100_norm as data_cls_c

    # use first shot loss
    use_fs = args.use_fs

    if args.use_fs:
        assert model_name in [
            'compat_stateful_srwm', 'stateful_deltanet', 'compat_stateful_deltanet',
            'compat_stateful_srwm_res12', 'compat_stateful_srwm_mixer',
            'compat_stateful_self_mod_mixer']
        num_samples_per_class = {
            'first_shot': 1, 'train': k_shot_train, 'test': test_per_class}
        dataset_a = data_cls_a(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
        dataset_b = data_cls_b(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
        dataset_c = data_cls_c(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
    else:
        dataset_a = data_cls_a(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed)
        dataset_b = data_cls_b(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed)
        dataset_c = data_cls_c(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed)

    dataloader_a = BatchMetaDataLoader(
        dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)
    dataloader_b = BatchMetaDataLoader(
        dataset_b, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)
    dataloader_c = BatchMetaDataLoader(
        dataset_c, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    # valid
    val_dataset_a = data_cls_a(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_val=True,
                        shuffle=True, seed=valid_seed)  # fixed validation set
    val_dataset_b = data_cls_b(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_val=True,
                        shuffle=True, seed=valid_seed)  # fixed validation set
    val_dataset_c = data_cls_c(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_val=True,
                        shuffle=True, seed=valid_seed)  # fixed validation set

    val_dataloader_a = BatchMetaDataLoader(
        val_dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    val_dataloader_b = BatchMetaDataLoader(
        val_dataset_b, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    val_dataloader_c = BatchMetaDataLoader(
        val_dataset_c, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    # test
    test_dataset_a = data_cls_a(
        args.data_dir, ways=n_way, shots=k_shot_train,
        test_shots=test_per_class, meta_test=True,
        download=True, shuffle=True, seed=test_seed)

    test_dataset_b = data_cls_b(
        args.data_dir, ways=n_way, shots=k_shot_train,
        test_shots=test_per_class, meta_test=True,
        download=True, shuffle=True, seed=test_seed)

    test_dataset_c = data_cls_c(
        args.data_dir, ways=n_way, shots=k_shot_train,
        test_shots=test_per_class, meta_test=True,
        download=True, shuffle=True, seed=test_seed)

    test_dataloader_a = BatchMetaDataLoader(
        test_dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    test_dataloader_b = BatchMetaDataLoader(
        test_dataset_b, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    test_dataloader_c = BatchMetaDataLoader(
        test_dataset_c, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    if args.cycle_dataloader:
        zip_dataloader_a_b_c = zip(
            cycle(dataloader_a), cycle(dataloader_b), cycle(dataloader_c))
    else:
        zip_dataloader_a_b_c = zip(dataloader_a, dataloader_b, dataloader_c)

elif args.train_splitmnist_style:
    # lazy implementation:
    # we independently draw 5 2-way tasks; this might end up in an ill-conditioned setting where the
    # certain classes are assigned to two different labels within the same sequence.
    # Ideally, we should draw one 10-way task instead and split it.
    loginf(f"Split-MNIST-like domain-incremental 5-task training")
    if args.mix_metafinetuning:
        from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a
        from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_b
    else:
        from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a

    # use first shot loss
    use_fs = args.use_fs

    if args.use_fs:
        assert model_name in [
            'compat_stateful_srwm', 'stateful_deltanet', 'compat_stateful_deltanet',
            'compat_stateful_srwm_res12', 'compat_stateful_srwm_mixer',
            'compat_stateful_self_mod_mixer']
        num_samples_per_class = {
            'first_shot': 1, 'train': k_shot_train, 'test': test_per_class}
        dataset_a = data_cls_a(
            args.data_dir, ways=n_way, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
        if args.mix_metafinetuning:
            dataset_b = data_cls_b(
                args.data_dir, ways=n_way, shots=k_shot_train,
                test_shots=test_per_class, meta_train=True,
                download=True, shuffle=True, seed=seed,
                num_samples_per_class=num_samples_per_class)
    else:
        assert False

    dataloader_a = BatchMetaDataLoader(
        dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    if args.mix_metafinetuning:
        dataloader_b = BatchMetaDataLoader(
            dataset_b, batch_size=batch_size, num_workers=args.num_worker,
            pin_memory=True, drop_last=args.drop_last_batch)


elif args.train_splitmnist_style_class_incremental:
    # lazy implementation:
    # we draw 5 2-way tasks and shift the target labels by task_id * 2
    # this might end up in an ill-conditioned setting where certain classes are assigned to
    # two different labels within the same sequence.
    # Ideally, we should draw one 10-way task instead and split it.
    loginf(f"Split-MNIST-like class-incremental 5-task training")
    if args.mix_metafinetuning:
        from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a
        from torchmeta_local.datasets.helpers import miniimagenet_32_norm_cache as data_cls_b
    else:
        from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls_a
    # use first shot loss
    use_fs = args.use_fs

    if args.use_fs:
        assert model_name in [
            'compat_stateful_srwm', 'stateful_deltanet', 'compat_stateful_deltanet',
            'compat_stateful_srwm_res12', 'compat_stateful_srwm_mixer',
            'compat_stateful_self_mod_mixer']
        num_samples_per_class = {
            'first_shot': 1, 'train': k_shot_train, 'test': test_per_class}
        dataset_a = data_cls_a(
            args.data_dir, ways=2, shots=k_shot_train,
            test_shots=test_per_class, meta_train=True,
            download=True, shuffle=True, seed=seed,
            num_samples_per_class=num_samples_per_class)
        if args.mix_metafinetuning:
            dataset_b = data_cls_b(
                args.data_dir, ways=2, shots=k_shot_train,
                test_shots=test_per_class, meta_train=True,
                download=True, shuffle=True, seed=seed,
                num_samples_per_class=num_samples_per_class)            
    else:
        assert False

    dataloader_a = BatchMetaDataLoader(
        dataset_a, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    if args.mix_metafinetuning:
        dataloader_b = BatchMetaDataLoader(
            dataset_b, batch_size=batch_size, num_workers=args.num_worker,
            pin_memory=True, drop_last=args.drop_last_batch)

else:
    loginf(f"Dataset/Task: {args.name_dataset}")
    if args.name_dataset == 'omniglot':
        from torchmeta_local.datasets.helpers import omniglot as data_cls
    elif args.name_dataset == 'omniglot_norm':
        from torchmeta_local.datasets.helpers import omniglot_norm as data_cls
    elif args.name_dataset == 'omniglot_32_norm':
        from torchmeta_local.datasets.helpers import omniglot_32_norm as data_cls
    elif args.name_dataset == 'miniimagenet':
        from torchmeta_local.datasets.helpers import miniimagenet as data_cls
    elif args.name_dataset == 'tieredimagenet':
        from torchmeta_local.datasets.helpers import tieredimagenet as data_cls
    elif args.name_dataset == 'miniimagenet_norm':  # mean/std normalized
        from torchmeta_local.datasets.helpers import (
            miniimagenet_norm as data_cls)
    elif args.name_dataset == 'miniimagenet_32_norm':
        from torchmeta_local.datasets.helpers import (
            miniimagenet_32_norm as data_cls)
    elif args.name_dataset == 'miniimagenet_32_norm_cache':
        from torchmeta_local.datasets.helpers import (
            miniimagenet_32_norm_cache as data_cls)
    elif args.name_dataset == 'omniglot_rgb84x84':
        from torchmeta_local.datasets.helpers import omniglot_rgb84x84 as data_cls
    elif args.name_dataset == 'omniglot_rgb84x84_norm':  # mean/std normalized
        from torchmeta_local.datasets.helpers import (
            omniglot_rgb84x84_norm as data_cls)
    elif args.name_dataset == 'fc100':
        from torchmeta_local.datasets.helpers import fc100 as data_cls
    elif args.name_dataset == 'cifar_fs':
        from torchmeta_local.datasets.helpers import cifar_fs as data_cls
    elif args.name_dataset == 'cifar_fs_norm':
        from torchmeta_local.datasets.helpers import cifar_fs_norm as data_cls
    elif args.name_dataset == 'cifar_fs_rfs':
        from torchmeta_local.datasets.helpers import cifar_fs_rfs as data_cls
    elif args.name_dataset == 'fc100_norm':
        from torchmeta_local.datasets.helpers import fc100_norm as data_cls
    elif args.name_dataset == 'fc100_rfs':
        from torchmeta_local.datasets.helpers import fc100_rfs as data_cls
    else:
        assert False, f'Unknown dataset: {args.name_dataset}'

    # use first shot loss
    use_fs = args.use_fs

    if args.use_fs:
        assert model_name in [
            'compat_stateful_srwm', 'stateful_deltanet', 'compat_stateful_deltanet',
            'compat_stateful_srwm_res12', 'compat_stateful_srwm_mixer',
            'compat_stateful_self_mod_mixer']
        num_samples_per_class = {
            'first_shot': 1, 'train': k_shot_train, 'test': test_per_class}
        dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_train=True,
                        download=True, shuffle=True, seed=seed,
                        num_samples_per_class=num_samples_per_class)
    else:
        dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                    test_shots=test_per_class, meta_train=True,
                    download=True, shuffle=True, seed=seed)

    dataloader = BatchMetaDataLoader(
        dataset, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    if args.name_dataset == 'fc100_rfs':
        from torchmeta_local.datasets.helpers import fc100_norm as data_cls

    if args.name_dataset == 'cifar_fs_rfs':
        from torchmeta_local.datasets.helpers import cifar_fs_norm as data_cls

    val_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_val=True,
                        shuffle=True, seed=valid_seed)  # fixed validation set

    if args.fixed_valid:
        # https://github.com/tristandeleu/pytorch-meta/issues/132
        valid_class_size = len(val_dataset.dataset)  # num classes in valid
        # `dataset` here is torchmeta ClassDataset
        import itertools
        from torch.utils.data import Subset
        cls_indices = np.array(range(valid_class_size))
        all_indices = []
        for subset in itertools.combinations(cls_indices, args.n_way):
            all_indices.append(subset)
        val_total_size = args.valid_size * batch_size
        val_indices = random.sample(all_indices, val_total_size)
        val_dataset = Subset(val_dataset, val_indices)

    val_dataloader = BatchMetaDataLoader(
        val_dataset, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

    test_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                            test_shots=test_per_class, meta_test=True,
                            download=True, shuffle=True, seed=test_seed)

    if args.fixed_test:
        # https://github.com/tristandeleu/pytorch-meta/issues/132
        test_class_size = len(test_dataset.dataset)  # num classes in valid
        # `dataset` here is torchmeta ClassDataset
        import itertools
        from torch.utils.data import Subset
        cls_indices = np.array(range(test_class_size))
        all_indices = []
        for subset in itertools.combinations(cls_indices, args.n_way):
            all_indices.append(subset)
        test_total_size = args.test_size * batch_size
        test_indices = random.sample(all_indices, test_total_size)
        test_dataset = Subset(test_dataset, test_indices)

    test_dataloader = BatchMetaDataLoader(
        test_dataset, batch_size=batch_size, num_workers=args.num_worker,
        pin_memory=True, drop_last=args.drop_last_batch)

device = 'cuda'

# setting model

hidden_size = args.hidden_size
num_classes = args.n_way

num_layer = args.num_layer
n_head = args.n_head
dim_head = hidden_size // n_head
dim_ff = hidden_size * args.ff_factor
dropout_rate = args.dropout
vision_dropout = args.vision_dropout

# is_imagenet = args.name_dataset != 'omniglot'
is_imagenet = args.name_dataset not in ['omniglot', 'omniglot_norm']
is_fc100 = False

if args.name_dataset in ['fc100', 'fc100_norm', 'fc100_rfs', 'cifar_fs', 'cifar_fs_norm', 'cifar_fs_rfs',
                         'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
    is_fc100 = True
    is_imagenet = False

if model_name == 'lstm':  # conv lstm
    loginf("Model: LSTM")
    model = ConvLSTMModel(hidden_size, num_classes, num_layer=num_layer,
                          vision_dropout=vision_dropout,
                          imagenet=is_imagenet, fc100=is_fc100)
elif model_name == 'deltanet':
    loginf("Model: DeltaNet")
    model = ConvDeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           imagenet=is_imagenet, fc100=is_fc100)
elif model_name in ['stateful_deltanet', 'compat_stateful_deltanet']:
    loginf("Model: Stateful DeltaNet")
    model = StatefulConvDeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           imagenet=is_imagenet, fc100=is_fc100,
                           single_state_training=args.single_state_training,
                           extra_label=args.extra_label,
                           remove_bn=args.remove_bn,
                           use_instance_norm=args.use_instance_norm)
elif model_name == 'srwm':
    loginf("Model: Self-Referential learning")
    model = ConvSRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                          num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                          dropout=dropout_rate, num_classes=num_classes,
                          vision_dropout=vision_dropout,
                          use_ln=True, beta_init=args.srwm_beta_init,
                          use_input_softmax=args.use_input_softmax,
                          input_dropout=args.input_dropout,
                          dropout_type=args.dropout_type,
                          imagenet=is_imagenet, fc100=is_fc100,
                          init_scaler=args.srwm_init_scaler,
                          q_init_scaler=args.srwm_q_init_scaler,
                          unif_init=args.unif_init,
                          no_softmax_on_y=args.no_softmax_on_y,
                          remove_bn=args.remove_bn,
                          use_instance_norm=args.use_instance_norm)
elif model_name == 'compat_stateful_srwm':
    loginf("Model: Self-Referential learning")
    model = CompatStatefulConvSRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                          num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                          dropout=dropout_rate, num_classes=num_classes,
                          vision_dropout=vision_dropout,
                          use_ln=True, beta_init=args.srwm_beta_init,
                          use_input_softmax=args.use_input_softmax,
                          input_dropout=args.input_dropout,
                          dropout_type=args.dropout_type,
                          imagenet=is_imagenet, fc100=is_fc100,
                          init_scaler=args.srwm_init_scaler,
                          q_init_scaler=args.srwm_q_init_scaler,
                          unif_init=args.unif_init,
                          single_state_training=args.single_state_training,
                          no_softmax_on_y=args.no_softmax_on_y,
                          extra_label=args.extra_label,
                          remove_bn=args.remove_bn,
                          use_instance_norm=args.use_instance_norm)
elif model_name == 'compat_stateful_srwm_mixer':
    loginf("Model: Mixer, Self-Referential learning")
    model = CompatStatefulMixerSRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                          num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                          dropout=dropout_rate, num_classes=num_classes,
                          vision_dropout=vision_dropout,
                          use_ln=True, beta_init=args.srwm_beta_init,
                          use_input_softmax=args.use_input_softmax,
                          input_dropout=args.input_dropout,
                          dropout_type=args.dropout_type,
                          patch_size=args.patch_size,
                          expansion_factor=dim_ff,
                          expansion_factor_token=args.ff_factor_tk,
                          imagenet=is_imagenet, fc100=is_fc100,
                          init_scaler=args.srwm_init_scaler,
                          q_init_scaler=args.srwm_q_init_scaler,
                          unif_init=args.unif_init,
                          single_state_training=args.single_state_training,
                          no_softmax_on_y=args.no_softmax_on_y,
                          extra_label=args.extra_label)
elif model_name == 'compat_stateful_self_mod_mixer':
    loginf("Model: Mixer, Self-Referential learning")
    model = CompatStatefulSelfModMixerModel(hidden_size=hidden_size, num_layers=num_layer,
                          num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                          dropout=dropout_rate, num_classes=num_classes,
                          vision_dropout=vision_dropout,
                          use_ln=True, beta_init=args.srwm_beta_init,
                          use_input_softmax=args.use_input_softmax,
                          input_dropout=args.input_dropout,
                          dropout_type=args.dropout_type,
                          patch_size=args.patch_size,
                          expansion_factor=dim_ff,
                          expansion_factor_token=args.ff_factor_tk,
                          imagenet=is_imagenet, fc100=is_fc100,
                          init_scaler=args.srwm_init_scaler,
                          q_init_scaler=args.srwm_q_init_scaler,
                          unif_init=args.unif_init,
                          single_state_training=args.single_state_training,
                          no_softmax_on_y=args.no_softmax_on_y,
                          extra_label=args.extra_label)
elif model_name == 'mixer_srwm':
    loginf("Model: Self-Referential learning")
    model = MixerSRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           use_ln=True, beta_init=args.srwm_beta_init,
                           use_input_softmax=args.use_input_softmax,
                           input_dropout=args.input_dropout,
                           imagenet=is_imagenet, fc100=is_fc100,
                           init_scaler=args.srwm_init_scaler,
                           q_init_scaler=args.srwm_q_init_scaler,
                           unif_init=args.unif_init,
                           no_softmax_on_y=args.no_softmax_on_y)
elif model_name == 'srwm_mixer':
    loginf("Model: Self-Referential learning")
    model = SRMixerModel(        
        hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           use_ln=True, beta_init=args.srwm_beta_init,
                           use_input_softmax=args.use_input_softmax,
                           patch_size=args.patch_size,
                           expansion_factor=dim_ff,
                           expansion_factor_token=args.ff_factor_tk,
                           imagenet=is_imagenet, fc100=is_fc100,
                           init_scaler=args.srwm_init_scaler,
                           q_init_scaler=args.srwm_q_init_scaler,
                           unif_init=args.unif_init,
                           no_softmax_on_y=args.no_softmax_on_y)
elif model_name == 'res12_lstm':
    loginf("Model: Resnet12 + LSTM")
    model = Res12LSTMModel(hidden_size=hidden_size, num_layers=num_layer,
                           dropout=dropout_rate,
                           vision_dropout=vision_dropout,
                           use_big=args.use_big_res12,
                           num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_deltanet':
    # assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + Deltanet")
    model = Res12DeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                            num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                            dropout=dropout_rate,
                            vision_dropout=vision_dropout,
                            use_big=args.use_big_res12,
                            num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_srwm':
    # assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + SRWM")
    model = Res12SRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           use_big=args.use_big_res12,
                           use_ln=not args.not_use_ln,
                           use_res=not args.not_use_res,
                           use_ff=not args.not_use_ff,
                           beta_init=args.srwm_beta_init,
                           use_input_softmax=args.use_input_softmax,
                           input_dropout=args.input_dropout,
                           dropout_type=args.dropout_type,
                           use_dropblock=args.use_dropblock,
                           imagenet=is_imagenet)

loginf(f"Number of trainable params: {model.num_params()}")
loginf(f"{model}")

model = model.to(device)

# NB: these are just for references.
# Checkpoints found via these ood evals (validated on **test** set of extra/external datasets)
# should NOT be used for final eval (unless we test on sub-sets of MNIST etc based on heldout classes
# that are not used in validation; e.g., our initial plan was to validate on MNIST 0-4 and test on MNIST 5-9)
# Could be replaced by the external validation set in the final version (see how we do it for split-mnist)
# extra task for eval
if args.ood_eval:
    loginf("Preparing extra eval dataset MNIST...")
    norm_params = {'mean': [0.1307], 'std': [0.3081]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.MNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    extra_train_data = []
    extra_train_labels = []

    extra_train_data_part2 = []
    extra_train_labels_part2 = []

    for class_id in range(num_classes):
        indices = extra_dataset.targets == class_id
        extra_train_data.append(extra_dataset.data[indices][:k_shot_train].to(device))
        extra_train_labels.append(extra_dataset.targets[indices][:k_shot_train].to(device))

        # part 2 is 2k-shot
        extra_train_data_part2.append(extra_dataset.data[indices][k_shot_train:3*k_shot_train].to(device))
        extra_train_labels_part2.append(extra_dataset.targets[indices][k_shot_train:3*k_shot_train].to(device))

    # class appears nth time only once all classes were seen for n-1 times for all n
    # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
    extra_train_data = torch.stack(extra_train_data, dim=1)
    extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

    extra_train_data_part2 = torch.stack(extra_train_data_part2, dim=1)
    extra_train_data_part2 = extra_train_data_part2.reshape(num_classes * k_shot_train * 2, *compat_shape)

    extra_train_labels = torch.stack(extra_train_labels, dim=1)
    extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

    extra_train_labels_part2 = torch.stack(extra_train_labels_part2, dim=1)
    extra_train_labels_part2 = extra_train_labels_part2.reshape(num_classes * k_shot_train * 2)

    # test set
    test_set = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=mnist_transform, download=True)

    # restrict number of classes
    idx = test_set.train_labels<num_classes
    test_set.targets = test_set.targets[idx]
    test_set.data = test_set.data[idx]

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker)
    loginf("done.")


# also add CIFAR10
if args.ood_eval2:
    loginf("Preparing extra eval dataset 2 CIFAR10...")
    norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params)])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params)])
    else:
        assert 'omni' in args.name_dataset
        loginf("Transforming to Grayscale.")
        from torchvision.transforms import Grayscale
        compat_shape = [1, 28, 28]
        norm_params = {'mean': [0.5], 'std': [0.25]}
        mnist_transform = Compose(
            [Resize(28), Grayscale(num_output_channels=1), ToTensor(), Normalize(**norm_params)])

    extra_dataset2 = torchvision.datasets.CIFAR10(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset2(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset2 = TransformedDataset2(extra_dataset2, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    extra_train_data2 = []
    extra_train_labels2 = []

    for class_id in range(num_classes):
        indices = extra_dataset2.targets == class_id
        extra_train_data2.append(extra_dataset2.data[indices][:k_shot_train].to(device))
        extra_train_labels2.append(extra_dataset2.targets[indices][:k_shot_train].to(device))

    # class appears nth time only once all classes were seen for n-1 times for all n
    # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
    extra_train_data2 = torch.stack(extra_train_data2, dim=1)
    extra_train_data2 = extra_train_data2.reshape(num_classes * k_shot_train, *compat_shape)

    # from torchvision.utils import save_image
    # save_image(extra_train_data, 'img2.png')

    extra_train_labels2 = torch.stack(extra_train_labels2, dim=1)
    extra_train_labels2 = extra_train_labels2.reshape(num_classes * k_shot_train)

    # test set
    test_set2 = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, transform=mnist_transform, download=True)
    # test_set = torchvision.datasets.CIFAR10(
    #     root=args.data_dir, train=False, transform=mnist_transform, download=True)

    # restrict number of classes
    tmp_targets = torch.ByteTensor(test_set2.targets)
    idx = tmp_targets<num_classes

    test_set2.targets = tmp_targets[idx].tolist()
    test_set2.data = test_set2.data[idx]

    # print(test_set.data[0].unsqueeze(0).shape)
    # save_image(test_set.data[0].unsqueeze(0).float(), 'img3.png')

    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker)
    loginf("done.")


# also add SVHN
if args.ood_eval3:
    loginf("Preparing extra eval dataset 3 SVHN...")
    norm_params = {'mean': [0.4376821, 0.4437697, 0.47280442], 'std': [0.19803012, 0.20101562, 0.19703614]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params)])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [ToTensor(), Normalize(**norm_params)])
    else:
        assert 'omni' in args.name_dataset
        loginf("Transforming to Grayscale.")
        from torchvision.transforms import Grayscale
        compat_shape = [1, 28, 28]
        norm_params = {'mean': [0.5], 'std': [0.25]}
        mnist_transform = Compose(
            [Resize(28), Grayscale(num_output_channels=1), ToTensor(), Normalize(**norm_params)])

    extra_dataset3 = torchvision.datasets.SVHN(
        root=args.data_dir, split='train', download=True)

    class TransformedDataset3(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.labels[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int) or isinstance(label, np.int64):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset3 = TransformedDataset3(extra_dataset3, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    extra_train_data3 = []
    extra_train_labels3 = []

    for class_id in range(num_classes):
        indices = extra_dataset3.targets == class_id
        extra_train_data3.append(extra_dataset3.data[indices][:k_shot_train].to(device))
        extra_train_labels3.append(extra_dataset3.targets[indices][:k_shot_train].to(device))

    # class appears nth time only once all classes were seen for n-1 times for all n
    # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
    extra_train_data3 = torch.stack(extra_train_data3, dim=1)
    extra_train_data3 = extra_train_data3.reshape(num_classes * k_shot_train, *compat_shape)

    extra_train_labels3 = torch.stack(extra_train_labels3, dim=1)
    extra_train_labels3 = extra_train_labels3.reshape(num_classes * k_shot_train)

    # test set
    test_set3 = torchvision.datasets.SVHN(
        root=args.data_dir, split='test', transform=mnist_transform, download=True)

    # restrict number of classes
    tmp_targets = torch.ByteTensor(test_set3.labels)
    idx = tmp_targets<num_classes

    test_set3.labels = tmp_targets[idx].tolist()
    test_set3.data = test_set3.data[idx]

    extra_test_loader3 = torch.utils.data.DataLoader(
        dataset=test_set3, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker)
    loginf("done.")


if args.ood_eval5:
    loginf("Preparing extra eval dataset Fashion MNIST...")
    norm_params = {'mean': [0.286], 'std': [0.353]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset5 = torchvision.datasets.FashionMNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset5(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset5 = TransformedDataset5(extra_dataset5, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    extra_train_data5 = []
    extra_train_labels5 = []

    for class_id in range(num_classes):
        indices = extra_dataset5.targets == class_id
        extra_train_data5.append(extra_dataset5.data[indices][:k_shot_train].to(device))
        extra_train_labels5.append(extra_dataset5.targets[indices][:k_shot_train].to(device))

    # class appears nth time only once all classes were seen for n-1 times for all n
    # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
    extra_train_data5 = torch.stack(extra_train_data5, dim=1)
    extra_train_data5 = extra_train_data5.reshape(num_classes * k_shot_train, *compat_shape)

    extra_train_labels5 = torch.stack(extra_train_labels5, dim=1)
    extra_train_labels5 = extra_train_labels5.reshape(num_classes * k_shot_train)

    # test set
    test_set5 = torchvision.datasets.FashionMNIST(
        root=args.data_dir, train=False, transform=mnist_transform, download=True)

    # restrict number of classes
    idx = test_set5.train_labels<num_classes
    test_set5.targets = test_set5.targets[idx]
    test_set5.data = test_set5.data[idx]

    extra_test_loader5 = torch.utils.data.DataLoader(
        dataset=test_set5, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker)
    loginf("done.")

#############################

# Set optimiser
learning_rate = args.learning_rate
clip = args.clip

loginf(f"Learning rate: {learning_rate}")
loginf(f"clip at: {clip}")

loginf(f"Batch size: {args.batch_size}")
loginf(f"Gradient accumulation for {args.grad_cummulate} steps.")

if version.parse(torch.__version__) >= version.parse("1.10.0"):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
else:
    loginf(f"Ignoring label_smoothing. Torch version {torch.__version__}")
    loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             betas=(0.9, 0.995), eps=1e-9)
loginf(f"{optimizer}")

if args.use_warmup:
    loginf("Using Warmup. Ignoring `learning_rate`.")
    optimizer = WarmupWrapper(args.hidden_size, args.warmup_steps, optimizer)

# load if needed
if args.init_model_from is not None:
    loginf(f"loading model from {args.init_model_from}")
    checkpoint = torch.load(args.init_model_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not args.no_load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if args.context_carry_over:
        state = checkpoint['state']
    else:
        state = None

elif args.init_model_except_output_from is not None:
    loginf(f"loading model from {args.init_model_except_output_from}")
    checkpoint = torch.load(args.init_model_except_output_from)
    checkpoint_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    new_dict = {}
    # we assume that the dim of the old/checkpoint model is bigger.
    for key, value in checkpoint_dict.items():  # 2-dim
        if key == 'input_proj.weight':
            cur_val = model_dict[key]
            new_len = cur_val.shape[-1]
            new_dict[key] = value[:, -new_len:]  # ideally the last one for extra ouput should be taken TODO
        elif key == 'out_layer.weight':
            cur_val = model_dict[key]
            new_len = cur_val.shape[0]
            new_dict[key] = value[:new_len, :]  
        elif key == 'out_layer.bias':  # 1-dim
            cur_val = model_dict[key]
            new_len = cur_val.shape[-1]
            new_dict[key] = value[:new_len]
        else:
            new_dict[key] = value

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

elif args.init_model_except_output_from_class_incremental is not None:
    loginf(f"loading model from {args.init_model_except_output_from_class_incremental}")
    checkpoint = torch.load(args.init_model_except_output_from_class_incremental)
    checkpoint_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    new_dict = {}
    # we assume that the dim of the new one is bigger than old/checkpoint.
    for key, value in checkpoint_dict.items():  # 2-dim
        if key == 'input_proj.weight':
            new_dict[key] = model_dict[key]
            new_len = value.shape[-1]
            new_dict[key][:, -new_len:] = value
        elif key == 'out_layer.weight':
            new_dict[key] = model_dict[key]
            new_len = value.shape[0]
            new_dict[key][:new_len, :] = value  
        elif key == 'out_layer.bias':  # 1-dim
            new_dict[key] = model_dict[key]
            new_len = value.shape[-1]
            new_dict[key][:new_len] = value
        else:
            new_dict[key] = value

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    if args.context_carry_over:
        state = checkpoint['state']
    else:
        state = None
else:
    state = None

model.reset_grad()
############
skip_train = False if args.eval_only_dir is None else True

if skip_train:
    ckpt_path = args.eval_only_dir
    best_model_path = os.path.join(ckpt_path, 'best_model.pt')
    lastest_model_path = os.path.join(ckpt_path, 'lastest_model.pt')
    best_ext_model_path = os.path.join(ckpt_path, 'best_ext_model.pt')
    best_ext2_model_path = os.path.join(ckpt_path, 'best_ext2_model.pt')
    best_ext3_model_path = os.path.join(ckpt_path, 'best_ext3_model.pt')
else:
    best_model_path = os.path.join(args.work_dir, 'best_model.pt')
    lastest_model_path = os.path.join(args.work_dir, 'lastest_model.pt')
    best_ext_model_path = os.path.join(args.work_dir, 'best_ext_model.pt')
    best_ext2_model_path = os.path.join(args.work_dir, 'best_ext2_model.pt')
    best_ext3_model_path = os.path.join(args.work_dir, 'best_ext3_model.pt')

loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Start training")
start_time = time.time()
interval_start_time = time.time()
train_timer = time.time()
last_batch_logged = 0

best_val_first_shot_acc = 0.0
best_valid_test_first_shot_acc = 0.0
best_test_first_shot_acc = 0.0
best_external_acc = 0.0
best_external_acc2 = 0.0
best_external_acc3 = 0.0

num_seq = 0
running_loss = 0.0
fs_running_loss = 0.0
running_total = 0

# To be cleaned up...
if args.use_abc_v2:
    running_loss_a_1 = 0.0
    running_loss_a_2 = 0.0
    running_loss_a_3 = 0.0

    running_correct_a_1 = 0.0
    running_correct_a_2 = 0.0
    running_correct_a_3 = 0.0
    running_total_a_1 = 0
    running_total_a_2 = 0
    running_total_a_3 = 0

    running_loss_b_1 = 0.0
    running_loss_b_2 = 0.0
    running_loss_b_3 = 0.0

    running_correct_b_1 = 0.0
    running_correct_b_2 = 0.0
    running_correct_b_3 = 0.0
    running_total_b_1 = 0
    running_total_b_2 = 0
    running_total_b_3 = 0

    running_loss_c_1 = 0.0
    running_loss_c_2 = 0.0
    running_loss_c_3 = 0.0

    running_correct_c_1 = 0.0
    running_correct_c_2 = 0.0
    running_correct_c_3 = 0.0
    running_total_c_1 = 0
    running_total_c_2 = 0
    running_total_c_3 = 0

    fs_running_loss_a_1 = 0.0
    fs_running_loss_a_2 = 0.0
    fs_running_loss_a_3 = 0.0

    fs_running_loss_b_1 = 0.0
    fs_running_loss_b_2 = 0.0
    fs_running_loss_b_3 = 0.0

    fs_running_loss_c_1 = 0.0
    fs_running_loss_c_2 = 0.0
    fs_running_loss_c_3 = 0.0

    fs_running_correct_a_1 = 0.0
    fs_running_correct_a_2 = 0
    fs_running_correct_a_3 = 0

    fs_running_correct_b_1 = 0.0
    fs_running_correct_b_2 = 0
    fs_running_correct_b_3 = 0

    fs_running_correct_c_1 = 0.0
    fs_running_correct_c_2 = 0
    fs_running_correct_c_3 = 0

    running_loss_acl_1 = 0.0
    running_correct_acl_1 = 0.0
    running_correct_acl_ab_1 = 0.0
    running_total_acl_1 = 0.0
    running_total_acl_ab_1 = 0.0

    running_loss_acl_2 = 0.0
    running_correct_acl_2 = 0.0
    running_correct_acl_ab_2 = 0.0
    running_total_acl_2 = 0.0
    running_total_acl_ab_2 = 0.0

    running_loss_acl_3 = 0.0
    running_correct_acl_3 = 0.0
    running_correct_acl_ab_3 = 0.0
    running_total_acl_3 = 0.0
    running_total_acl_ab_3 = 0.0

    running_loss_acl_ab_1 = 0.0
    running_loss_acl_ab_2 = 0.0
    running_loss_acl_ab_3 = 0.0

running_loss_a_1 = 0.0
running_loss_a_2 = 0.0
running_loss_b_1 = 0.0
running_loss_b_2 = 0.0

fs_running_loss_1 = 0.0
fs_running_loss_2 = 0.0

fs_running_loss_a_1 = 0.0
fs_running_loss_a_2 = 0.0
fs_running_loss_b_1 = 0.0
fs_running_loss_b_2 = 0.0

running_loss_acl_1 = 0.0
running_loss_acl_2 = 0.0

running_correct = 0
fs_running_correct = 0

running_correct_a_1 = 0
running_correct_a_2 = 0
running_correct_b_1 = 0
running_correct_b_2 = 0

running_total_a_1 = 0
running_total_a_2 = 0
running_total_b_1 = 0
running_total_b_2 = 0

fs_running_correct_1 = 0
fs_running_correct_2 = 0

fs_running_correct_a_1 = 0
fs_running_correct_a_2 = 0
fs_running_correct_b_1 = 0
fs_running_correct_b_2 = 0

running_correct_acl_1 = 0
running_correct_acl_2 = 0
running_total_acl_1 = 0
running_total_acl_2 = 0

running_correct_v2 = 0
count_first_first = 0
first_first = 0
run_step = 0

train_running_correct = 0
train_running_total = 0

test_running_correct = 0
test_running_total = 0

one_running_total = 0
one_correct = 0
two_correct = 0
five_correct = 0
eight_correct = 0
ten_correct = 0

offset_step = 0
total_steps = 0
end_training = False
cur_train_acc = 0

# random state reset
drop2d_layer = nn.Dropout2d(args.state_dropout)
drop2d_layer.train()

if args.context_carry_over:
    assert 'stateful' in model_name

### A-B training
# version 1. Not used in the end. Use v2 below.
if args.use_ab and not skip_train:
    for ep in range(args.total_epoch):
        loginf(f'epoch {ep} ====================')
        for i, (batch_1, batch_2) in enumerate(zip_dataloader_a_b):
            model.train()
            # state = None
            if args.context_carry_over_double:
                if i % 2 == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif args.context_carry_over_k > 1:
                if i % args.context_carry_over_k == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif not args.context_carry_over:
                state = None
            elif state is not None:
                state = model.clone_state(state, detach=True)

            # shuffle order
            if i % 2 == 0:
                batch_a = batch_1
                batch_b = batch_2
            else:
                batch_a = batch_2
                batch_b = batch_1

            # Extract test examples:
            # TASK A ##########################################################
            test_inputs_a, test_targets_a = batch_a['test']
            test_inputs_a = test_inputs_a.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_a = test_targets_a.to(device=device)

            test_inputs_a = test_inputs_a.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_a = test_targets_a.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_a = test_inputs_a[0].unsqueeze(0)
            test_targets_a = test_targets_a[0].unsqueeze(0)

            # TASK B ##########################################################
            # same for test
            test_inputs_b, test_targets_b = batch_b['test']
            test_inputs_b = test_inputs_b.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_b = test_targets_b.to(device=device)

            test_inputs_b = test_inputs_b.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_b = test_targets_b.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_b = test_inputs_b[0].unsqueeze(0)
            test_targets_b = test_targets_b[0].unsqueeze(0)

            # Do the first-shot part before updating the state ================
            if use_fs:
                # TASK A
                fs_train_inputs, fs_train_targets = batch_a['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([fs_train_inputs, test_inputs_a], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_a], dim=0)
                target_labels_a = target_labels[-1].reshape(-1)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                # TASK B
                fs_train_inputs, fs_train_targets = batch_b['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([net_input, fs_train_inputs, test_inputs_b], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_b], dim=0)
                target_labels_b = target_labels[-1].reshape(-1)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback = torch.cat([label_feedback, sync_labels, dummy_last_token], dim=0)

                # do not update BN stats on this small batch
                model.set_bn_in_eval_mode()
                outputs, _ = model(net_input, label_feedback, state)
                model.set_bn_in_train_mode()
                # not carrying states forward from one-shot learning

                outputs_b = outputs[-1]
                outputs_a = outputs[fs_slen]  # 'predict' position for A

                outputs_a = outputs_a.reshape(-1, num_classes)
                outputs_b = outputs_b.reshape(-1, num_classes)
                
                loss_fs_a = loss_fn(outputs_a, target_labels_a)
                loss_fs_b = loss_fn(outputs_b, target_labels_b)

                with torch.no_grad():
                    _, predicted = outputs_a.max(-1)
                bool_correct_pred = (predicted == target_labels_a)
                # fs_running_correct_1 += bool_correct_pred.sum().item()
                if i % 2 == 0:
                    fs_running_correct_a_1 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_b_1 += bool_correct_pred.sum().item()

                with torch.no_grad():
                    _, predicted = outputs_b.max(-1)
                bool_correct_pred = (predicted == target_labels_b)
                # fs_running_correct_2 += bool_correct_pred.sum().item()
                if i % 2 == 0:
                    fs_running_correct_b_2 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_a_2 += bool_correct_pred.sum().item()

            # Extract train examples ##########################################
            train_inputs, train_targets = batch_a['train']
            train_inputs = train_inputs.to(device=device)  # (B, len, 1, 28, 28)
            train_targets = train_targets.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs.shape
            bsz, slen = train_shape[0], train_shape[1]

            num_seq += bsz

            train_inputs = train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets = train_targets.transpose(0, 1)  # (len, B)

            # do the main part
            net_input = torch.cat([train_inputs, test_inputs_a], dim=0)
            target_labels = torch.cat([train_targets, test_targets_a], dim=0)
            target_labels_a = target_labels[-1].reshape(-1)  # used in loss later

            target_labels_shape = target_labels.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels[:-1]
            # does not matter which label to feed for the last position.
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

            # net_input and label_feedback for TASK A ready
            # do the same for TASK B
            train_inputs, train_targets = batch_b['train']
            bsz_b = train_inputs.shape[0]
            # TODO hard coded assuming that B has more examples than A
            if bsz_b > bsz:
                train_inputs = train_inputs[:bsz].to(device=device)  # (B, len, 1, 28, 28)
                train_targets = train_targets[:bsz].to(device=device)  # (B, len)
            else:
                train_inputs = train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                train_targets = train_targets.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs.shape
            bsz, slen = train_shape[0], train_shape[1]

            train_inputs = train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets = train_targets.transpose(0, 1)  # (len, B)

            net_input = torch.cat([net_input, train_inputs, test_inputs_b], dim=0)
            target_labels = torch.cat([train_targets, test_targets_b], dim=0)
            target_labels_b = target_labels[-1].reshape(-1)

            target_labels_shape = target_labels.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels[:-1]

            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            label_feedback = torch.cat([label_feedback, sync_labels, dummy_last_token], dim=0)

            # -- forward all: A_context, A_predict, B_context, B_predict
            outputs, state = model(net_input, label_feedback, state)
            state = model.clone_state(state)

            outputs_b = outputs[-1]
            outputs_a = outputs[slen]  # 'predict' position for A

            outputs_a = outputs_a.reshape(-1, num_classes)
            outputs_b = outputs_b.reshape(-1, num_classes)

            loss_main_a = loss_fn(outputs_a, target_labels_a)
            loss_main_b = loss_fn(outputs_b, target_labels_b)

            # TASK A
            with torch.no_grad():
                _, predicted = outputs_a.max(-1)
            bool_correct_pred = (predicted == target_labels_a)
            if i % 2 == 0:
                running_correct_a_1 += bool_correct_pred.sum().item()
                running_total_a_1 += target_labels_a.size(0)
            else:
                running_correct_b_1 += bool_correct_pred.sum().item()
                running_total_b_1 += target_labels_a.size(0)

            # TASK B
            with torch.no_grad():
                _, predicted = outputs_b.max(-1)
            bool_correct_pred = (predicted == target_labels_b)
            if i % 2 == 0:
                running_correct_b_2 += bool_correct_pred.sum().item()
                running_total_b_2 += target_labels_b.size(0)
            else:
                running_correct_a_2 += bool_correct_pred.sum().item()
                running_total_a_2 += target_labels_b.size(0)

            # ACL part, evaluate forgetting
            net_input = test_inputs_a
            target_labels = test_targets_a

            dummy_last_token = torch.zeros_like(target_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            model.set_bn_in_eval_mode()
            outputs, state = model(net_input, dummy_last_token, state)
            model.set_bn_in_train_mode()
            state = model.clone_state(state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            target_labels = target_labels[-1].reshape(-1)

            loss_acl_a = loss_fn(outputs, target_labels)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels)

            if i % 2 == 0:
                running_correct_acl_1 += bool_correct_pred.sum().item()
                running_total_acl_1 += target_labels.size(0)
            else:
                running_correct_acl_2 += bool_correct_pred.sum().item()
                running_total_acl_2 += target_labels.size(0)

            # loss scaler
            if i % 2 == 0:
                a_scale = args.loss_scale_task_a 
                b_scale = 1.0
            else:
                a_scale = 1.0
                b_scale = args.loss_scale_task_a                

            # total loss
            if args.disable_multi:
                if use_fs:
                    if i % 2 == 0:
                        loss = (loss_fs_a + loss_main_a) * 0.5
                    else:
                        loss = (loss_fs_b + loss_main_b) * 0.5
                else:
                    if i % 2 == 0:
                        loss = loss_main_a
                    else:
                        loss = loss_main_b
            else:
                if args.use_acl:
                    if use_fs:
                        loss = ((loss_fs_a + loss_main_a) * a_scale + (loss_fs_b + loss_main_b) * b_scale + loss_acl_a) * 0.2
                    else:
                        loss = (loss_main_a * a_scale + loss_main_b * b_scale + loss_acl_a) * 0.33
                else:
                    if use_fs:
                        loss = ((loss_fs_a + loss_main_a) * a_scale + (loss_fs_b + loss_main_b) * b_scale) * 0.25
                    else:
                        loss = (loss_main_a * a_scale + loss_main_b * b_scale) * 0.5

            loss.backward()

            if i % args.grad_cummulate == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                model.reset_grad()

            # global loss
            if i % 2 == 0:
                running_loss_a_1 += loss_main_a.item()
                running_loss_b_2 += loss_main_b.item()
                if use_fs:
                    fs_running_loss_a_1 += loss_fs_a.item()
                    fs_running_loss_b_2 += loss_fs_b.item()
            else:
                running_loss_b_1 += loss_main_a.item()
                running_loss_a_2 += loss_main_b.item()
                if use_fs:
                    fs_running_loss_a_2 += loss_fs_a.item()
                    fs_running_loss_b_1 += loss_fs_b.item()
            # if args.use_acl:
            if i % 2 == 0:
                running_loss_acl_1 += loss_acl_a.item()
            else:
                running_loss_acl_2 += loss_acl_a.item()

            running_total += target_labels.size(0)
            model.eval()

            run_step += 1
            if i % args.report_every == 0:

                if use_wandb:
                    if use_fs:
                        wandb.log({
                            "train_loss_a_1": running_loss_a_1 / run_step,
                            "train_loss_a_2": running_loss_a_2 / run_step,
                            "train_loss_fs_a_1": fs_running_loss_a_1 / run_step,
                            "train_loss_fs_a_2": fs_running_loss_a_2 / run_step,
                            "running_acc_a_1": 100 * zero_div(running_correct_a_1, running_total_a_1),
                            "running_acc_a_2": 100 * zero_div(running_correct_a_2, running_total_a_2),
                            "running_acc_fs_a_1": 100 * zero_div(fs_running_correct_a_1, running_total_a_1),
                            "running_acc_fs_a_2": 100 * zero_div(fs_running_correct_a_2, running_total_a_2),
                            "train_loss_b_1": running_loss_b_1 / run_step,
                            "train_loss_b_2": running_loss_b_2 / run_step,
                            "train_loss_fs_b_1": fs_running_loss_b_1 / run_step,
                            "train_loss_fs_b_2": fs_running_loss_b_2 / run_step,
                            "running_acc_b_1": 100 * zero_div(running_correct_b_1, running_total_b_1),
                            "running_acc_b_2": 100 * zero_div(running_correct_b_2, running_total_b_2),
                            "running_acc_fs_b_1": 100 * zero_div(fs_running_correct_b_1, running_total_b_1),
                            "running_acc_fs_b_2": 100 * zero_div(fs_running_correct_b_2, running_total_b_2),
                            "train_loss_acl_a": running_loss_acl_1 / run_step,
                            "train_loss_acl_b": running_loss_acl_2 / run_step,
                            "running_acc_acl_a": 100 * zero_div(running_correct_acl_1, running_total_acl_1),
                            "running_acc_acl_b": 100 * zero_div(running_correct_acl_2, running_total_acl_2),
                        })
                    else:
                        wandb.log({
                            "train_loss_a_1": running_loss_a_1 / run_step,
                            "train_loss_a_2": running_loss_a_2 / run_step,
                            "running_acc_a_1": 100 * zero_div(running_correct_a_1, running_total_a_1),
                            "running_acc_a_2": 100 * zero_div(running_correct_a_2, running_total_a_2),
                            "train_loss_b_1": running_loss_b_1 / run_step,
                            "train_loss_b_2": running_loss_b_2 / run_step,
                            "running_acc_b_1": 100 * zero_div(running_correct_b_1, running_total_b_1),
                            "running_acc_b_2": 100 * zero_div(running_correct_b_2, running_total_b_2),
                            "train_loss_acl_a": running_loss_acl_1 / run_step,
                            "train_loss_acl_b": running_loss_acl_2 / run_step,
                            "running_acc_acl_a": 100 * zero_div(running_correct_acl_1, running_total_acl_1),
                            "running_acc_acl_b": 100 * zero_div(running_correct_acl_2, running_total_acl_2),
                        })
                train_elapsed = time.time() - train_timer
                train_timer = time.time()
                num_images_per_sec = (
                    (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                    // train_elapsed)
                last_batch_logged = i

                # check accurary on the batch.
                # writer.add_scalar("Loss/train", running_loss / run_step, i)
                if use_fs:
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss_a_1: {running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_a_2: {running_loss_a_2 / run_step :.3f}, '
                        f'train_loss_fs_a_1: {fs_running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_fs_a_2: {fs_running_loss_a_2 / run_step :.3f}, '
                        f'running_acc_a_1: {100 * zero_div(running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_a_2: {100 * zero_div(running_correct_a_2, running_total_a_2):.2f} % '
                        f'running_acc_fs_a_1: {100 * zero_div(fs_running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_fs_a_2: {100 * zero_div(fs_running_correct_a_2, running_total_a_2):.2f} % '
                        f'train_loss_b_1: {running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_b_2: {running_loss_b_2 / run_step :.3f}, '
                        f'train_loss_fs_b_1: {fs_running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_fs_b_2: {fs_running_loss_b_2 / run_step :.3f}, '
                        f'running_acc_b_1: {100 * zero_div(running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_b_2: {100 * zero_div(running_correct_b_2, running_total_b_2):.2f} % '
                        f'running_acc_fs_b_1: {100 * zero_div(fs_running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_fs_b_2: {100 * zero_div(fs_running_correct_b_2, running_total_b_2):.2f} % '
                        f'train_loss_acl_a: {running_loss_acl_1 / run_step :.3f}, '
                        f'running_acc_acl_a: {100 * zero_div(running_correct_acl_1, running_total_acl_1):.2f} %, '
                        f'train_loss_acl_b: {running_loss_acl_2 / run_step :.3f}, '
                        f'running_acc_acl_b: {100 * zero_div(running_correct_acl_2, running_total_acl_2):.2f} %, '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')
                else:
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss_a_1: {running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_a_2: {running_loss_a_2 / run_step :.3f}, '
                        f'running_acc_a_1: {100 * zero_div(running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_a_2: {100 * zero_div(running_correct_a_2, running_total_a_2):.2f} % '
                        f'train_loss_b_1: {running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_b_2: {running_loss_b_2 / run_step :.3f}, '
                        f'running_acc_b_1: {100 * zero_div(running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_b_2: {100 * zero_div(running_correct_b_2, running_total_b_2):.2f} % '
                        f'train_loss_acl_a: {running_loss_acl_1 / run_step :.3f}, '
                        f'running_acc_acl_a: {100 * zero_div(running_correct_acl_1, running_total_acl_1):.2f} %, '
                        f'train_loss_acl_b: {running_loss_acl_2 / run_step :.3f}, '
                        f'running_acc_acl_b: {100 * zero_div(running_correct_acl_2, running_total_acl_2):.2f} %, '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')

                running_loss_a_1 = 0.0
                running_loss_a_2 = 0.0

                running_correct_a_1 = 0.0
                running_correct_a_2 = 0.0
                running_total_a_1 = 0
                running_total_a_2 = 0

                running_loss_b_1 = 0.0
                running_loss_b_2 = 0.0

                running_correct_b_1 = 0
                running_correct_b_2 = 0
                running_total_b_1 = 0
                running_total_b_2 = 0

                fs_running_loss_a_1 = 0.0
                fs_running_loss_a_2 = 0.0
                fs_running_loss_b_1 = 0.0
                fs_running_loss_b_2 = 0.0

                fs_running_correct_a_1 = 0.0
                fs_running_correct_a_2 = 0
                fs_running_correct_b_1 = 0.0
                fs_running_correct_b_2 = 0

                running_loss_acl_1 = 0.0
                running_correct_acl_1 = 0.0
                running_total_acl_1 = 0.0

                running_loss_acl_2 = 0.0
                running_correct_acl_2 = 0.0
                running_total_acl_2 = 0.0

                running_total = 0
                run_step = 0

            # ======================================================================

            if i % args.validate_every == 0:  # run validation
                model.eval()
                with torch.no_grad():
                    if args.disable_multi:
                        v_total_a = eval_model_label_sync(
                            model, val_dataloader_a, num_steps=args.valid_size)
                        test_total_a = eval_model_label_sync(
                            model, test_dataloader_a, num_steps=args.test_size)

                        v_total_b = eval_model_label_sync(
                            model, val_dataloader_b, num_steps=args.valid_size)
                        test_total_b = eval_model_label_sync(
                            model, test_dataloader_b, num_steps=args.test_size)
                    else:
                        v_total_a, v_state = eval_model_label_sync(
                            model, val_dataloader_a, num_steps=args.valid_size,
                            get_state=True)
                        test_total_a, test_state = eval_model_label_sync(
                            model, test_dataloader_a, num_steps=args.test_size,
                            get_state=True)

                        v_total_b = eval_model_label_sync(
                            model, val_dataloader_b, num_steps=args.valid_size,
                            state=v_state, get_state=False)
                        test_total_b = eval_model_label_sync(
                            model, test_dataloader_b, num_steps=args.test_size,
                            state=test_state, get_state=False)

                loginf(
                    f"[val {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
                    f'val total A {100 * v_total_a :.2f} %, '
                    f'val total B {100 * v_total_b :.2f} %, ')

                loginf(
                    f'test acc A {100 * test_total_a :.2f} %, '
                    f'test acc B {100 * test_total_b :.2f} %')  # debugging

                if use_wandb:
                    wandb.log({
                        "val_acc_a": 100 * v_total_a,
                        "test_acc_a": 100 * test_total_a,
                        "val_acc_b": 100 * v_total_b,
                        "test_acc_b": 100 * test_total_b,
                    })

                avg_v = (v_total_a + v_total_b) * 0.5
                if avg_v > best_val_first_shot_acc:
                    best_val_first_shot_acc = avg_v
                    best_step = i + offset_step
                    # Save the best model
                    loginf("The best model so far.")
                    if args.context_carry_over:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'state': state,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    else:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    loginf("Saved.")
                    if test_total_b > best_valid_test_first_shot_acc:
                        best_valid_test_first_shot_acc = test_total_b
                if test_total_b > best_test_first_shot_acc:
                    best_test_first_shot_acc = test_total_b
                loginf(
                    f'current best valid_acc: {100 * best_val_first_shot_acc :.2f} '
                    f'%\ncurrent best valid test_acc on B: '
                    f'{100 * best_valid_test_first_shot_acc :.2f} %\n'
                    f'current best test_acc on B: {100 * best_test_first_shot_acc :.2f} ')
                # Save the latest model
                if args.context_carry_over:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'state': state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': avg_v}, lastest_model_path)
                else:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': avg_v}, lastest_model_path)

                if args.ood_eval:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                            # double shot
                            self_train_input = extra_train_data_part2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels_part2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct_doubleshot += bool_correct_pred.sum().item()

                    external_acc = 100 * extra_running_correct / total_test_samples
                    external_acc_doubleshot = 100 * extra_running_correct_doubleshot / total_test_samples
                    loginf(f'Extra test acc: {external_acc:.2f} %')
                    loginf(f'Extra test double shot acc: {external_acc_doubleshot:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_acc": external_acc,
                            "extra_double_acc": external_acc_doubleshot,
                        })
                    if best_external_acc < external_acc:
                        best_external_acc = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)	

                if args.ood_eval2:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader2):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'CIFAR10 test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_cifar10_acc": external_acc,
                        })
                    if best_external_acc2 < external_acc:
                        best_external_acc2 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)	

                if args.ood_eval3:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader3):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data3.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels3.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'SVHN test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_svhn_acc": external_acc,
                        })
                    if best_external_acc3 < external_acc:
                        best_external_acc3 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)	

                elapsed = time.time() - interval_start_time
                loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
                interval_start_time = time.time()
                train_timer = time.time()

            if cur_train_acc > args.train_acc_stop:
                loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
                end_training = True
                break
            if i + offset_step > args.total_train_steps:
                end_training = True
                loginf(f'reached {args.total_train_steps} steps')
                break
            if args.freeze_out_emb:
                if args.freeze_after_steps < i:
                    for param in model.out_layer.parameters():
                        param.requires_grad = False      
                    # loginf(f"Step {i}: freezing output embeddings")  
            if args.freeze_after:
                if args.freeze_after_steps < i:
                    # loginf(f"Step {i}: freezing conv stem")  
                    if model_name in ['srwm', 'deltanet']:
                        for param in model.conv_layers.parameters():
                            param.requires_grad = False
                    elif model_name in ['res12_srwm', 'res12_deltanet']:
                        for param in model.stem_resnet12.parameters():
                            param.requires_grad = False
        if end_training:
            break
        offset_step += i

# version 2.
# create the seq. do one forward pass first to update the BN stats.
# then fix the BN stats for the rest
elif args.use_ab_v2 and not skip_train:
    for ep in range(args.total_epoch):
        loginf(f'EPOCH {ep} ====================')
        for i, (batch_1, batch_2) in enumerate(zip_dataloader_a_b):
            model.train()
            # state = None
            if args.context_carry_over_double:
                if i % 2 == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif args.context_carry_over_k > 1:
                if i % args.context_carry_over_k == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif not args.context_carry_over:
                state = None
            elif state is not None:
                state = model.clone_state(state, detach=True)

            # shuffle order
            if i % 2 == 0:
                batch_a = batch_1
                batch_b = batch_2
            else:
                batch_a = batch_2
                batch_b = batch_1

            # Extract test examples:
            # TASK A ##########################################################
            test_inputs_a, test_targets_a = batch_a['test']
            test_inputs_a = test_inputs_a.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_a = test_targets_a.to(device=device)

            test_inputs_a = test_inputs_a.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_a = test_targets_a.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_a = test_inputs_a[0].unsqueeze(0)
            test_targets_a = test_targets_a[0].unsqueeze(0)

            # TASK B ##########################################################
            # same for test
            test_inputs_b, test_targets_b = batch_b['test']
            test_inputs_b = test_inputs_b.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_b = test_targets_b.to(device=device)

            test_inputs_b = test_inputs_b.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_b = test_targets_b.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_b = test_inputs_b[0].unsqueeze(0)
            test_targets_b = test_targets_b[0].unsqueeze(0)

            # Extract train examples ##########################################
            train_inputs_a, train_targets_a = batch_a['train']
            train_inputs_a = train_inputs_a.to(device=device)  # (B, len, 1, 28, 28)
            train_targets_a = train_targets_a.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs_a.shape
            bsz, slen = train_shape[0], train_shape[1]

            num_seq += bsz

            train_inputs_a = train_inputs_a.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets_a = train_targets_a.transpose(0, 1)  # (len, B)

            # do the main part
            net_input_a = torch.cat([train_inputs_a, test_inputs_a], dim=0)
            target_labels_a = torch.cat([train_targets_a, test_targets_a], dim=0)
            # target_labels_a = target_labels_a[-1].reshape(-1)  # used in loss later

            target_labels_shape = target_labels_a.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels_a[:-1]
            # does not matter which label to feed for the last position.
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            label_feedback_a = torch.cat([sync_labels, dummy_last_token], dim=0)

            # net_input and label_feedback for TASK A ready
            # do the same for TASK B
            train_inputs_b, train_targets_b = batch_b['train']
            train_inputs_b = train_inputs_b.to(device=device)  # (B, len, 1, 28, 28)
            train_targets_b = train_targets_b.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs_b.shape
            bsz, slen = train_shape[0], train_shape[1]

            train_inputs_b = train_inputs_b.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets_b = train_targets_b.transpose(0, 1)  # (len, B)

            net_input_b = torch.cat([train_inputs_b, test_inputs_b], dim=0)
            target_labels_b = torch.cat([train_targets_b, test_targets_b], dim=0)
            # target_labels_b = target_labels_b[-1].reshape(-1)

            target_labels_shape = target_labels_b.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels_b[:-1]

            # dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            # if model.extra_label:
            #     dummy_last_token = dummy_last_token + model.num_classes
            label_feedback_b = torch.cat([sync_labels, dummy_last_token], dim=0)

            # -- forward all: A_context, A_predict, B_context, B_predict
            # one forward pass to update BN stats
            if not args.use_instance_norm:
                with torch.no_grad():
                    net_input_dummy = torch.cat([net_input_a, net_input_b], dim=0)
                    label_feedback_dummy = torch.cat([label_feedback_a, label_feedback_b], dim=0)
                    outputs_dummy, _ = model(net_input_dummy, label_feedback_dummy, state)

            target_labels_a = target_labels_a[-1].reshape(-1)  # used in loss later
            target_labels_b = target_labels_b[-1].reshape(-1)  # used in loss later

            # TASK A
            if use_fs:
                # TASK A
                fs_train_inputs, fs_train_targets = batch_a['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([fs_train_inputs, test_inputs_a], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_a], dim=0)
                # target_labels = target_labels[-1].reshape(-1)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                # do not update BN stats on this small batch
                model.set_bn_in_eval_mode()
                outputs, _ = model(net_input, label_feedback, state)
                model.set_bn_in_train_mode()
                # not carrying states forward from one-shot learning

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)

                loss_fs_a = loss_fn(outputs, target_labels_a)

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels_a)
                # fs_running_correct_1 += bool_correct_pred.sum().item()
                if i % 2 == 0:
                    fs_running_correct_a_1 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_b_1 += bool_correct_pred.sum().item()

            model.set_bn_in_eval_mode()
            _, state = model(train_inputs_a, train_targets_a, state)
            state = model.clone_state(state)
            outputs, _ = model(test_inputs_a, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            loss_main_a = loss_fn(outputs, target_labels_a)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels_a)
            if i % 2 == 0:
                running_correct_a_1 += bool_correct_pred.sum().item()
                running_total_a_1 += target_labels_a.size(0)
            else:
                running_correct_b_1 += bool_correct_pred.sum().item()
                running_total_b_1 += target_labels_a.size(0)

            # TASK B

            # Do the first-shot part before updating the state ================
            if use_fs:
                fs_train_inputs, fs_train_targets = batch_b['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([fs_train_inputs, test_inputs_b], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_b], dim=0)
                # target_labels = target_labels[-1].reshape(-1)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                # do not update BN stats on this small batch
                model.set_bn_in_eval_mode()
                outputs, _ = model(net_input, label_feedback, state)
                model.set_bn_in_train_mode()
                # not carrying states forward from one-shot learning

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                
                loss_fs_b = loss_fn(outputs, target_labels_b)

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels_b)
                if i % 2 == 0:
                    fs_running_correct_b_2 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_a_2 += bool_correct_pred.sum().item()

            model.set_bn_in_eval_mode()
            _, state = model(train_inputs_b, train_targets_b, state)
            state = model.clone_state(state)
            outputs, _ = model(test_inputs_b, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            loss_main_b = loss_fn(outputs, target_labels_b)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels_b)
            if i % 2 == 0:
                running_correct_b_2 += bool_correct_pred.sum().item()
                running_total_b_2 += target_labels_b.size(0)
            else:
                running_correct_a_2 += bool_correct_pred.sum().item()
                running_total_a_2 += target_labels_b.size(0)

            # ACL part, evaluate forgetting
            net_input = test_inputs_a
            target_labels = test_targets_a

            model.set_bn_in_eval_mode()
            outputs, _ = model(net_input, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            target_labels = target_labels[-1].reshape(-1)

            loss_acl_a = loss_fn(outputs, target_labels)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels)

            if i % 2 == 0:
                running_correct_acl_1 += bool_correct_pred.sum().item()
                running_total_acl_1 += target_labels.size(0)
            else:
                running_correct_acl_2 += bool_correct_pred.sum().item()
                running_total_acl_2 += target_labels.size(0)

            # loss scale
            if i % 2 == 0:
                a_scale = args.loss_scale_task_a 
                b_scale = 1.0
            else:
                a_scale = 1.0
                b_scale = args.loss_scale_task_a

            # total loss
            if args.disable_multi:
                if use_fs:
                    if i % 2 == 0:
                        loss = (loss_fs_a + loss_main_a) * 0.5
                    else:
                        loss = (loss_fs_b + loss_main_b) * 0.5
                else:
                    if i % 2 == 0:
                        loss = loss_main_a
                    else:
                        loss = loss_main_b
            else:
                if args.use_acl:
                    if use_fs:
                        loss = ((loss_fs_a + loss_main_a) * a_scale + (loss_fs_b + loss_main_b) * b_scale + loss_acl_a) * 0.2
                    else:
                        loss = (loss_main_a * a_scale + loss_main_b * b_scale + loss_acl_a) * 0.33
                else:
                    if use_fs:
                        loss = ((loss_fs_a + loss_main_a) * a_scale + (loss_fs_b + loss_main_b) * b_scale) * 0.25
                    else:
                        loss = (loss_main_a * a_scale + loss_main_b * b_scale) * 0.5
            loss.backward()

            if i % args.grad_cummulate == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                model.reset_grad()

            # global loss
            if i % 2 == 0:
                running_loss_a_1 += loss_main_a.item()
                running_loss_b_2 += loss_main_b.item()
                if use_fs:
                    fs_running_loss_a_1 += loss_fs_a.item()
                    fs_running_loss_b_2 += loss_fs_b.item()
            else:
                running_loss_b_1 += loss_main_a.item()
                running_loss_a_2 += loss_main_b.item()
                if use_fs:
                    fs_running_loss_a_2 += loss_fs_a.item()
                    fs_running_loss_b_1 += loss_fs_b.item()
            # if args.use_acl:
            if i % 2 == 0:
                running_loss_acl_1 += loss_acl_a.item()
            else:
                running_loss_acl_2 += loss_acl_a.item()

            running_total += target_labels.size(0)
            model.eval()

            run_step += 1
            if i % args.report_every == 0:
                if use_wandb:
                    if use_fs:
                        wandb.log({
                            "train_loss_a_1": running_loss_a_1 / run_step,
                            "train_loss_a_2": running_loss_a_2 / run_step,
                            "train_loss_fs_a_1": fs_running_loss_a_1 / run_step,
                            "train_loss_fs_a_2": fs_running_loss_a_2 / run_step,
                            "running_acc_a_1": 100 * zero_div(running_correct_a_1, running_total_a_1),
                            "running_acc_a_2": 100 * zero_div(running_correct_a_2, running_total_a_2),
                            "running_acc_fs_a_1": 100 * zero_div(fs_running_correct_a_1, running_total_a_1),
                            "running_acc_fs_a_2": 100 * zero_div(fs_running_correct_a_2, running_total_a_2),
                            "train_loss_b_1": running_loss_b_1 / run_step,
                            "train_loss_b_2": running_loss_b_2 / run_step,
                            "train_loss_fs_b_1": fs_running_loss_b_1 / run_step,
                            "train_loss_fs_b_2": fs_running_loss_b_2 / run_step,
                            "running_acc_b_1": 100 * zero_div(running_correct_b_1, running_total_b_1),
                            "running_acc_b_2": 100 * zero_div(running_correct_b_2, running_total_b_2),
                            "running_acc_fs_b_1": 100 * zero_div(fs_running_correct_b_1, running_total_b_1),
                            "running_acc_fs_b_2": 100 * zero_div(fs_running_correct_b_2, running_total_b_2),
                            "train_loss_acl_a": running_loss_acl_1 / run_step,
                            "train_loss_acl_b": running_loss_acl_2 / run_step,
                            "running_acc_acl_a": 100 * zero_div(running_correct_acl_1, running_total_acl_1),
                            "running_acc_acl_b": 100 * zero_div(running_correct_acl_2, running_total_acl_2),
                        })
                    else:
                        wandb.log({
                            "train_loss_a_1": running_loss_a_1 / run_step,
                            "train_loss_a_2": running_loss_a_2 / run_step,
                            "running_acc_a_1": 100 * zero_div(running_correct_a_1, running_total_a_1),
                            "running_acc_a_2": 100 * zero_div(running_correct_a_2, running_total_a_2),
                            "train_loss_b_1": running_loss_b_1 / run_step,
                            "train_loss_b_2": running_loss_b_2 / run_step,
                            "running_acc_b_1": 100 * zero_div(running_correct_b_1, running_total_b_1),
                            "running_acc_b_2": 100 * zero_div(running_correct_b_2, running_total_b_2),
                            "train_loss_acl_a": running_loss_acl_1 / run_step,
                            "train_loss_acl_b": running_loss_acl_2 / run_step,
                            "running_acc_acl_a": 100 * zero_div(running_correct_acl_1, running_total_acl_1),
                            "running_acc_acl_b": 100 * zero_div(running_correct_acl_2, running_total_acl_2),
                        })
                train_elapsed = time.time() - train_timer
                train_timer = time.time()
                num_images_per_sec = (
                    (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                    // train_elapsed)
                last_batch_logged = i

                # check accurary on the batch.
                # writer.add_scalar("Loss/train", running_loss / run_step, i)
                if use_fs:
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss_a_1: {running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_a_2: {running_loss_a_2 / run_step :.3f}, '
                        f'train_loss_fs_a_1: {fs_running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_fs_a_2: {fs_running_loss_a_2 / run_step :.3f}, '
                        f'running_acc_a_1: {100 * zero_div(running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_a_2: {100 * zero_div(running_correct_a_2, running_total_a_2):.2f} % '
                        f'running_acc_fs_a_1: {100 * zero_div(fs_running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_fs_a_2: {100 * zero_div(fs_running_correct_a_2, running_total_a_2):.2f} % '
                        f'train_loss_b_1: {running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_b_2: {running_loss_b_2 / run_step :.3f}, '
                        f'train_loss_fs_b_1: {fs_running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_fs_b_2: {fs_running_loss_b_2 / run_step :.3f}, '
                        f'running_acc_b_1: {100 * zero_div(running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_b_2: {100 * zero_div(running_correct_b_2, running_total_b_2):.2f} % '
                        f'running_acc_fs_b_1: {100 * zero_div(fs_running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_fs_b_2: {100 * zero_div(fs_running_correct_b_2, running_total_b_2):.2f} % '
                        f'train_loss_acl_a: {running_loss_acl_1 / run_step :.3f}, '
                        f'running_acc_acl_a: {100 * zero_div(running_correct_acl_1, running_total_acl_1):.2f} %, '
                        f'train_loss_acl_b: {running_loss_acl_2 / run_step :.3f}, '
                        f'running_acc_acl_b: {100 * zero_div(running_correct_acl_2, running_total_acl_2):.2f} %, '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')
                else:
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss_a_1: {running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_a_2: {running_loss_a_2 / run_step :.3f}, '
                        f'running_acc_a_1: {100 * zero_div(running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_a_2: {100 * zero_div(running_correct_a_2, running_total_a_2):.2f} % '
                        f'train_loss_b_1: {running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_b_2: {running_loss_b_2 / run_step :.3f}, '
                        f'running_acc_b_1: {100 * zero_div(running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_b_2: {100 * zero_div(running_correct_b_2, running_total_b_2):.2f} % '
                        f'train_loss_acl_a: {running_loss_acl_1 / run_step :.3f}, '
                        f'running_acc_acl_a: {100 * zero_div(running_correct_acl_1, running_total_acl_1):.2f} %, '
                        f'train_loss_acl_b: {running_loss_acl_2 / run_step :.3f}, '
                        f'running_acc_acl_b: {100 * zero_div(running_correct_acl_2, running_total_acl_2):.2f} %, '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')

                running_loss_a_1 = 0.0
                running_loss_a_2 = 0.0

                running_correct_a_1 = 0.0
                running_correct_a_2 = 0.0
                running_total_a_1 = 0
                running_total_a_2 = 0

                running_loss_b_1 = 0.0
                running_loss_b_2 = 0.0

                running_correct_b_1 = 0
                running_correct_b_2 = 0
                running_total_b_1 = 0
                running_total_b_2 = 0

                fs_running_loss_a_1 = 0.0
                fs_running_loss_a_2 = 0.0
                fs_running_loss_b_1 = 0.0
                fs_running_loss_b_2 = 0.0

                fs_running_correct_a_1 = 0.0
                fs_running_correct_a_2 = 0
                fs_running_correct_b_1 = 0.0
                fs_running_correct_b_2 = 0

                running_loss_acl_1 = 0.0
                running_correct_acl_1 = 0.0
                running_total_acl_1 = 0.0

                running_loss_acl_2 = 0.0
                running_correct_acl_2 = 0.0
                running_total_acl_2 = 0.0

                running_total = 0
                run_step = 0

            # ======================================================================
            if i % args.validate_every == 0:  # run validation
                model.eval()
                with torch.no_grad():
                    if args.disable_multi:
                        v_total_a = eval_model_label_sync(
                            model, val_dataloader_a, num_steps=args.valid_size,
                            get_second_last_state=True)
                        test_total_a = eval_model_label_sync(
                            model, test_dataloader_a, num_steps=args.test_size,
                            get_second_last_state=True)

                        v_total_b = eval_model_label_sync(
                            model, val_dataloader_b, num_steps=args.valid_size)
                        test_total_b = eval_model_label_sync(
                            model, test_dataloader_b, num_steps=args.test_size)
                    else:
                        v_total_a, v_state = eval_model_label_sync(
                            model, val_dataloader_a, num_steps=args.valid_size,
                            get_state=True, get_second_last_state=True)
                        test_total_a, test_state = eval_model_label_sync(
                            model, test_dataloader_a, num_steps=args.test_size,
                            get_state=True, get_second_last_state=True)

                        v_total_b = eval_model_label_sync(
                            model, val_dataloader_b, num_steps=args.valid_size,
                            state=v_state, get_state=False)
                        test_total_b = eval_model_label_sync(
                            model, test_dataloader_b, num_steps=args.test_size,
                            state=test_state, get_state=False)

                loginf(
                    f"[val {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
                    f'val total A {100 * v_total_a :.2f} %, '
                    f'val total B {100 * v_total_b :.2f} %, ')

                loginf(
                    f'test acc A {100 * test_total_a :.2f} %, '
                    f'test acc B {100 * test_total_b :.2f} %')  # debugging

                if use_wandb:
                    wandb.log({
                        "val_acc_a": 100 * v_total_a,
                        "test_acc_a": 100 * test_total_a,
                        "val_acc_b": 100 * v_total_b,
                        "test_acc_b": 100 * test_total_b,
                    })
                avg_v = (v_total_a + v_total_b) * 0.5
                if avg_v > best_val_first_shot_acc:
                    best_val_first_shot_acc = avg_v
                    best_step = i + offset_step
                    # Save the best model
                    loginf("The best model so far.")
                    if args.context_carry_over:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'state': state,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    else:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    loginf("Saved.")
                    if test_total_b > best_valid_test_first_shot_acc:
                        best_valid_test_first_shot_acc = test_total_b
                if test_total_b > best_test_first_shot_acc:
                    best_test_first_shot_acc = test_total_b
                loginf(
                    f'current best valid_acc: {100 * best_val_first_shot_acc :.2f} '
                    f'%\ncurrent best valid test_acc on B: '
                    f'{100 * best_valid_test_first_shot_acc :.2f} %\n'
                    f'current best test_acc on B: {100 * best_test_first_shot_acc :.2f} ')
                # Save the latest model
                if args.context_carry_over:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'state': state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': v_total_b}, lastest_model_path)
                else:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': v_total_b}, lastest_model_path)

                if args.ood_eval:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                            # double shot
                            self_train_input = extra_train_data_part2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels_part2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct_doubleshot += bool_correct_pred.sum().item()

                    external_acc = 100 * extra_running_correct / total_test_samples
                    external_acc_doubleshot = 100 * extra_running_correct_doubleshot / total_test_samples
                    loginf(f'Extra test acc: {external_acc:.2f} %')
                    loginf(f'Extra test double shot acc: {external_acc_doubleshot:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_acc": external_acc,
                            "extra_double_acc": external_acc_doubleshot,
                        })
                    if best_external_acc < external_acc:
                        best_external_acc = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)	

                if args.ood_eval2:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader2):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'CIFAR10 test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_cifar10_acc": external_acc,
                        })
                    if best_external_acc2 < external_acc:
                        best_external_acc2 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)	

                if args.ood_eval3:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader3):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data3.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels3.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'SVHN test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_svhn_acc": external_acc,
                        })
                    if best_external_acc3 < external_acc:
                        best_external_acc3 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)	

                elapsed = time.time() - interval_start_time
                loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
                interval_start_time = time.time()
                train_timer = time.time()

            if cur_train_acc > args.train_acc_stop:
                loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
                end_training = True
                break
            if i + offset_step > args.total_train_steps:
                end_training = True
                loginf(f'reached {args.total_train_steps} steps')
                break
            if args.freeze_out_emb:
                if args.freeze_after_steps < i:
                    for param in model.out_layer.parameters():
                        param.requires_grad = False      
                    # loginf(f"Step {i}: freezing output embeddings")  
            if args.freeze_after:
                if args.freeze_after_steps < i:
                    # loginf(f"Step {i}: freezing conv stem")  
                    if model_name in ['srwm', 'deltanet']:
                        for param in model.conv_layers.parameters():
                            param.requires_grad = False
                    elif model_name in ['res12_srwm', 'res12_deltanet']:
                        for param in model.stem_resnet12.parameters():
                            param.requires_grad = False
        if end_training:
            break
        offset_step += i


# version 2 with three tasks
elif args.use_abc_v2 and not skip_train:
    for ep in range(args.total_epoch):
        loginf(f'EPOCH {ep} ====================')
        # for i, (batch_1, batch_2) in enumerate(zip_dataloader_a_b):
        for i, (batch_1, batch_2, batch_3) in enumerate(zip_dataloader_a_b_c):
            model.train()
            # state = None
            if args.context_carry_over_double:
                if i % 2 == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif args.context_carry_over_k > 1:
                if i % args.context_carry_over_k == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif not args.context_carry_over:
                state = None
            elif state is not None:
                state = model.clone_state(state, detach=True)

            # shuffle order
            if i % 3 == 0:
                batch_a = batch_1
                batch_b = batch_2
                batch_c = batch_3
            elif i % 3 == 1:
                batch_a = batch_3
                batch_b = batch_1
                batch_c = batch_2
            else:
                batch_a = batch_2
                batch_b = batch_3
                batch_c = batch_1

            # Extract test examples:
            # TASK A ##########################################################
            test_inputs_a, test_targets_a = batch_a['test']
            test_inputs_a = test_inputs_a.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_a = test_targets_a.to(device=device)

            test_inputs_a = test_inputs_a.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_a = test_targets_a.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_a = test_inputs_a[0].unsqueeze(0)
            test_targets_a = test_targets_a[0].unsqueeze(0)

            # TASK B ##########################################################
            # same for test
            test_inputs_b, test_targets_b = batch_b['test']
            test_inputs_b = test_inputs_b.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_b = test_targets_b.to(device=device)

            test_inputs_b = test_inputs_b.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_b = test_targets_b.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_b = test_inputs_b[0].unsqueeze(0)
            test_targets_b = test_targets_b[0].unsqueeze(0)

            # TASK C ##########################################################
            # same for test
            test_inputs_c, test_targets_c = batch_c['test']
            test_inputs_c = test_inputs_c.to(device=device)  # (B, test_len, 28 * 28)
            test_targets_c = test_targets_c.to(device=device)

            test_inputs_c = test_inputs_c.transpose(0, 1)  # (test_len, B, 28 * 28)
            test_targets_c = test_targets_c.transpose(0, 1)  # (test_len, B)

            # take only the fist element (randomized already)
            test_inputs_c = test_inputs_c[0].unsqueeze(0)
            test_targets_c = test_targets_c[0].unsqueeze(0)

            # Extract train examples ##########################################
            train_inputs_a, train_targets_a = batch_a['train']
            train_inputs_a = train_inputs_a.to(device=device)  # (B, len, 1, 28, 28)
            train_targets_a = train_targets_a.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs_a.shape
            bsz, slen = train_shape[0], train_shape[1]

            num_seq += bsz

            train_inputs_a = train_inputs_a.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets_a = train_targets_a.transpose(0, 1)  # (len, B)

            # do the main part
            net_input_a = torch.cat([train_inputs_a, test_inputs_a], dim=0)
            target_labels_a = torch.cat([train_targets_a, test_targets_a], dim=0)
            # target_labels_a = target_labels_a[-1].reshape(-1)  # used in loss later

            target_labels_shape = target_labels_a.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels_a[:-1]
            # does not matter which label to feed for the last position.
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes
            label_feedback_a = torch.cat([sync_labels, dummy_last_token], dim=0)

            ###  net_input and label_feedback for TASK A ready
            # do the same for TASK B
            train_inputs_b, train_targets_b = batch_b['train']
            train_inputs_b = train_inputs_b.to(device=device)  # (B, len, 1, 28, 28)
            train_targets_b = train_targets_b.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs_b.shape
            bsz, slen = train_shape[0], train_shape[1]

            train_inputs_b = train_inputs_b.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets_b = train_targets_b.transpose(0, 1)  # (len, B)

            net_input_b = torch.cat([train_inputs_b, test_inputs_b], dim=0)
            target_labels_b = torch.cat([train_targets_b, test_targets_b], dim=0)
            # target_labels_b = target_labels_b[-1].reshape(-1)

            target_labels_shape = target_labels_b.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels_b[:-1]

            # dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            # if model.extra_label:
            #     dummy_last_token = dummy_last_token + model.num_classes
            label_feedback_b = torch.cat([sync_labels, dummy_last_token], dim=0)

            ###  net_input and label_feedback for TASK A and B ready
            # do the same for TASK C
            train_inputs_c, train_targets_c = batch_c['train']
            train_inputs_c = train_inputs_c.to(device=device)  # (B, len, 1, 28, 28)
            train_targets_c = train_targets_c.to(device=device)  # (B, len)

            # shuffle and reshape
            train_shape = train_inputs_c.shape
            bsz, slen = train_shape[0], train_shape[1]

            train_inputs_c = train_inputs_c.transpose(0, 1)  # (len, B, 28 * 28)
            train_targets_c = train_targets_c.transpose(0, 1)  # (len, B)

            net_input_c = torch.cat([train_inputs_c, test_inputs_c], dim=0)
            target_labels_c = torch.cat([train_targets_c, test_targets_c], dim=0)
            # target_labels_b = target_labels_b[-1].reshape(-1)

            target_labels_shape = target_labels_c.shape
            assert target_labels_shape[0] == slen + 1
            assert target_labels_shape[1] == bsz

            sync_labels = target_labels_c[:-1]

            label_feedback_c = torch.cat([sync_labels, dummy_last_token], dim=0)

            # -- forward all: A_context, A_predict, B_context, B_predict
            # one forward pass to update BN stats
            if not args.use_instance_norm:
                with torch.no_grad():
                    net_input_dummy = torch.cat([net_input_a, net_input_b, net_input_c], dim=0)
                    label_feedback_dummy = torch.cat([label_feedback_a, label_feedback_b, label_feedback_c], dim=0)
                    outputs_dummy, _ = model(net_input_dummy, label_feedback_dummy, state)

            target_labels_a = target_labels_a[-1].reshape(-1)  # used in loss later
            target_labels_b = target_labels_b[-1].reshape(-1)  # used in loss later
            target_labels_c = target_labels_c[-1].reshape(-1)  # used in loss later

            # TASK A
            if use_fs:
                # TASK A
                fs_train_inputs, fs_train_targets = batch_a['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([fs_train_inputs, test_inputs_a], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_a], dim=0)
                # target_labels = target_labels[-1].reshape(-1)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                # do not update BN stats on this small batch
                model.set_bn_in_eval_mode()
                outputs, _ = model(net_input, label_feedback, state)
                model.set_bn_in_train_mode()
                # not carrying states forward from one-shot learning

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)

                loss_fs_a = loss_fn(outputs, target_labels_a)

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels_a)
                if i % 3 == 0:
                    fs_running_correct_a_1 += bool_correct_pred.sum().item()
                elif i % 3 == 1:
                    fs_running_correct_b_1 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_c_1 += bool_correct_pred.sum().item()

            model.set_bn_in_eval_mode()
            _, state = model(train_inputs_a, train_targets_a, state)
            state = model.clone_state(state)
            outputs, _ = model(test_inputs_a, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            loss_main_a = loss_fn(outputs, target_labels_a)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels_a)
            if i % 3 == 0:
                running_correct_a_1 += bool_correct_pred.sum().item()
                running_total_a_1 += target_labels_a.size(0)
            elif i % 3 == 1:
                running_correct_b_1 += bool_correct_pred.sum().item()
                running_total_b_1 += target_labels_a.size(0)
            else:
                running_correct_c_1 += bool_correct_pred.sum().item()
                running_total_c_1 += target_labels_a.size(0)

            # TASK B

            # Do the first-shot part before updating the state ================
            if use_fs:
                fs_train_inputs, fs_train_targets = batch_b['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([fs_train_inputs, test_inputs_b], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_b], dim=0)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                # do not update BN stats on this small batch
                model.set_bn_in_eval_mode()
                outputs, _ = model(net_input, label_feedback, state)
                model.set_bn_in_train_mode()
                # not carrying states forward from one-shot learning

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                
                loss_fs_b = loss_fn(outputs, target_labels_b)

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels_b)
                # fs_running_correct_2 += bool_correct_pred.sum().item()
                if i % 3 == 0:
                    fs_running_correct_b_2 += bool_correct_pred.sum().item()
                elif i % 3 == 1:
                    fs_running_correct_c_2 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_a_2 += bool_correct_pred.sum().item()

            model.set_bn_in_eval_mode()
            _, state = model(train_inputs_b, train_targets_b, state)
            state = model.clone_state(state)
            outputs, _ = model(test_inputs_b, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            loss_main_b = loss_fn(outputs, target_labels_b)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels_b)
            if i % 3 == 0:
                running_correct_b_2 += bool_correct_pred.sum().item()
                running_total_b_2 += target_labels_b.size(0)
            elif i % 3 == 1:
                running_correct_c_2 += bool_correct_pred.sum().item()
                running_total_c_2 += target_labels_b.size(0)            
            else:
                running_correct_a_2 += bool_correct_pred.sum().item()
                running_total_a_2 += target_labels_b.size(0)

            # ACL part, evaluate forgetting
            net_input = test_inputs_a
            target_labels = test_targets_a

            model.set_bn_in_eval_mode()
            outputs, _ = model(net_input, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            target_labels = target_labels[-1].reshape(-1)

            loss_acl_a = loss_fn(outputs, target_labels)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels)

            if i % 3 == 0:
                running_correct_acl_1 += bool_correct_pred.sum().item()
                running_total_acl_1 += target_labels.size(0)
            elif i % 3 == 1:
                running_correct_acl_2 += bool_correct_pred.sum().item()
                running_total_acl_2 += target_labels.size(0)       
            else:
                running_correct_acl_3 += bool_correct_pred.sum().item()
                running_total_acl_3 += target_labels.size(0)

            # TASK C

            # Do the first-shot part before updating the state ================
            if use_fs:
                fs_train_inputs, fs_train_targets = batch_c['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                net_input = torch.cat([fs_train_inputs, test_inputs_c], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets_c], dim=0)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1

                sync_labels = target_labels[:-1]
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                # do not update BN stats on this small batch
                model.set_bn_in_eval_mode()
                outputs, _ = model(net_input, label_feedback, state)
                model.set_bn_in_train_mode()
                # not carrying states forward from one-shot learning

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                
                loss_fs_c = loss_fn(outputs, target_labels_c)

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels_c)
                # fs_running_correct_2 += bool_correct_pred.sum().item()
                if i % 3 == 0:
                    fs_running_correct_c_3 += bool_correct_pred.sum().item()
                elif i % 3 == 1:
                    fs_running_correct_a_3 += bool_correct_pred.sum().item()
                else:
                    fs_running_correct_b_3 += bool_correct_pred.sum().item()

            model.set_bn_in_eval_mode()
            _, state = model(train_inputs_c, train_targets_c, state)
            state = model.clone_state(state)
            outputs, _ = model(test_inputs_c, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            loss_main_c = loss_fn(outputs, target_labels_c)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels_c)
            if i % 3 == 0:
                running_correct_c_3 += bool_correct_pred.sum().item()
                running_total_c_3 += target_labels_c.size(0)
            elif i % 3 == 1:
                running_correct_a_3 += bool_correct_pred.sum().item()
                running_total_a_3 += target_labels_c.size(0)
            else:
                running_correct_b_3 += bool_correct_pred.sum().item()
                running_total_b_3 += target_labels_c.size(0)

            # ACL PART **1**, evaluate forgetting, there are TWO, A and B.
            net_input = test_inputs_a
            target_labels = test_targets_a

            model.set_bn_in_eval_mode()
            outputs, _ = model(net_input, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            target_labels = target_labels[-1].reshape(-1)

            loss_acl_ab_a = loss_fn(outputs, target_labels)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels)

            if i % 3 == 0:
                running_correct_acl_ab_1 += bool_correct_pred.sum().item()
                running_total_acl_ab_1 += target_labels.size(0)
            elif i % 3 == 1:
                running_correct_acl_ab_2 += bool_correct_pred.sum().item()
                running_total_acl_ab_2 += target_labels.size(0)
            else:
                running_correct_acl_ab_3 += bool_correct_pred.sum().item()
                running_total_acl_ab_3 += target_labels.size(0)

            # ACL PART **2**, evaluate forgetting, there are TWO, A and B.
            net_input = test_inputs_b
            target_labels = test_targets_b

            # dummy_last_token = torch.zeros_like(target_labels[0].unsqueeze(0))
            # if model.extra_label:
            #     dummy_last_token = dummy_last_token + model.num_classes

            model.set_bn_in_eval_mode()
            outputs, _ = model(net_input, dummy_last_token, state)
            model.set_bn_in_train_mode()

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            target_labels = target_labels[-1].reshape(-1)

            loss_acl_ab_b = loss_fn(outputs, target_labels)

            with torch.no_grad():
                _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == target_labels)

            if i % 3 == 0:
                running_correct_acl_ab_2 += bool_correct_pred.sum().item()
                running_total_acl_ab_2 += target_labels.size(0)
            elif i % 3 == 1:
                running_correct_acl_ab_3 += bool_correct_pred.sum().item()
                running_total_acl_ab_3 += target_labels.size(0)
            else:
                running_correct_acl_ab_1 += bool_correct_pred.sum().item()
                running_total_acl_ab_1 += target_labels.size(0)

            # loss scale
            if i % 3 == 0:
                a_scale = args.loss_scale_task_a
                b_scale = 1.0
                c_scale = 1.0
            elif i % 3 == 1:
                a_scale = 1.0
                b_scale = args.loss_scale_task_a
                c_scale = 1.0
            else:
                a_scale = 1.0
                b_scale = 1.0
                c_scale = args.loss_scale_task_a

            a_scaler = args.scale_first
            ab_scaler = args.prioritize_last
            c_scaler = args.prioritize_last_factor
            ab_acl_scaler = args.ab_acl_scaler

            # total loss
            if args.use_acl:
                if use_fs:
                    loss = (((loss_fs_a + loss_main_a) * a_scale * a_scaler + (loss_fs_b + loss_main_b) * b_scale) * ab_scaler + (loss_fs_c + loss_main_c) * c_scale * c_scaler + (loss_acl_a + loss_acl_ab_a + loss_acl_ab_b) * ab_acl_scaler) * 0.11
                else:
                    loss = ((loss_main_a * a_scale * a_scaler + loss_main_b * b_scale) * ab_scaler + loss_main_c * c_scale * c_scaler + (loss_acl_a + loss_acl_ab_a + loss_acl_ab_b) * ab_acl_scaler) * 0.16
            else:
                if use_fs:
                    loss = (((loss_fs_a + loss_main_a) * a_scale * a_scaler + (loss_fs_b + loss_main_b) * b_scale) * ab_scaler + (loss_fs_c + loss_main_c) * c_scale * c_scaler) * 0.16
                else:
                    loss = ((loss_main_a * a_scale * a_scaler + loss_main_b * b_scale) * ab_scaler + loss_main_c * c_scale * c_scaler) * 0.33
            loss.backward()

            if i % args.grad_cummulate == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                model.reset_grad()

            # global loss
            if i % 3 == 0:
                running_loss_a_1 += loss_main_a.item()
                running_loss_b_2 += loss_main_b.item()
                running_loss_c_3 += loss_main_c.item()
                if use_fs:
                    fs_running_loss_a_1 += loss_fs_a.item()
                    fs_running_loss_b_2 += loss_fs_b.item()
                    fs_running_loss_c_3 += loss_fs_c.item()
            elif i % 3 == 1:
                running_loss_b_1 += loss_main_a.item()
                running_loss_c_2 += loss_main_b.item()
                running_loss_a_3 += loss_main_c.item()
                if use_fs:
                    fs_running_loss_b_1 += loss_fs_a.item()
                    fs_running_loss_c_2 += loss_fs_b.item()
                    fs_running_loss_a_3 += loss_fs_c.item()
            else:
                running_loss_c_1 += loss_main_a.item()
                running_loss_a_2 += loss_main_b.item()
                running_loss_b_3 += loss_main_c.item()
                if use_fs:
                    fs_running_loss_c_1 += loss_fs_a.item()
                    fs_running_loss_a_2 += loss_fs_b.item()
                    fs_running_loss_b_3 += loss_fs_c.item()
            # if args.use_acl:
            if i % 3 == 0:
                running_loss_acl_1 += loss_acl_a.item()
                running_loss_acl_ab_1 += loss_acl_ab_a.item()
                running_loss_acl_ab_2 += loss_acl_ab_b.item()
            if i % 3 == 1:
                running_loss_acl_2 += loss_acl_a.item()
                running_loss_acl_ab_2 += loss_acl_ab_a.item()
                running_loss_acl_ab_3 += loss_acl_ab_b.item()
            else:
                running_loss_acl_3 += loss_acl_a.item()
                running_loss_acl_ab_3 += loss_acl_ab_a.item()
                running_loss_acl_ab_1 += loss_acl_ab_b.item()

            running_total += target_labels.size(0)
            model.eval()

            run_step += 1
            if i % args.report_every == 0:

                if use_wandb:
                    if use_fs:
                        wandb.log({
                            "train_loss_a_1": running_loss_a_1 / run_step,
                            "train_loss_a_2": running_loss_a_2 / run_step,
                            "train_loss_a_3": running_loss_a_3 / run_step,
                            "train_loss_fs_a_1": fs_running_loss_a_1 / run_step,
                            "train_loss_fs_a_2": fs_running_loss_a_2 / run_step,
                            "train_loss_fs_a_3": fs_running_loss_a_3 / run_step,
                            "running_acc_a_1": 100 * zero_div(running_correct_a_1, running_total_a_1),
                            "running_acc_a_2": 100 * zero_div(running_correct_a_2, running_total_a_2),
                            "running_acc_a_3": 100 * zero_div(running_correct_a_3, running_total_a_3),
                            "running_acc_fs_a_1": 100 * zero_div(fs_running_correct_a_1, running_total_a_1),
                            "running_acc_fs_a_2": 100 * zero_div(fs_running_correct_a_2, running_total_a_2),
                            "running_acc_fs_a_3": 100 * zero_div(fs_running_correct_a_3, running_total_a_3),
                            "train_loss_acl_a": running_loss_acl_1 / run_step,
                            "running_acc_acl_a": 100 * zero_div(running_correct_acl_1, running_total_acl_1),
                            "train_loss_acl_ab_a": running_loss_acl_ab_1 / run_step,
                            "running_acc_acl_ab_a": 100 * zero_div(running_correct_acl_ab_1, running_total_acl_ab_1),
                            "train_loss_b_1": running_loss_b_1 / run_step,
                            "train_loss_b_2": running_loss_b_2 / run_step,
                            "train_loss_b_3": running_loss_b_3 / run_step,
                            "train_loss_fs_b_1": fs_running_loss_b_1 / run_step,
                            "train_loss_fs_b_2": fs_running_loss_b_2 / run_step,
                            "train_loss_fs_b_3": fs_running_loss_b_3 / run_step,
                            "running_acc_b_1": 100 * zero_div(running_correct_b_1, running_total_b_1),
                            "running_acc_b_2": 100 * zero_div(running_correct_b_2, running_total_b_2),
                            "running_acc_b_3": 100 * zero_div(running_correct_b_3, running_total_b_3),
                            "running_acc_fs_b_1": 100 * zero_div(fs_running_correct_b_1, running_total_b_1),
                            "running_acc_fs_b_2": 100 * zero_div(fs_running_correct_b_2, running_total_b_2),
                            "running_acc_fs_b_3": 100 * zero_div(fs_running_correct_b_3, running_total_b_3),
                            "train_loss_acl_b": running_loss_acl_2 / run_step,
                            "running_acc_acl_b": 100 * zero_div(running_correct_acl_2, running_total_acl_2),
                            "train_loss_acl_ab_b": running_loss_acl_ab_2 / run_step,
                            "running_acc_acl_ab_b": 100 * zero_div(running_correct_acl_ab_2, running_total_acl_ab_2),
                            "train_loss_c_1": running_loss_c_1 / run_step,
                            "train_loss_c_2": running_loss_c_2 / run_step,
                            "train_loss_c_3": running_loss_c_3 / run_step,
                            "train_loss_fs_c_1": fs_running_loss_c_1 / run_step,
                            "train_loss_fs_c_2": fs_running_loss_c_2 / run_step,
                            "train_loss_fs_c_3": fs_running_loss_c_3 / run_step,
                            "running_acc_c_1": 100 * zero_div(running_correct_c_1, running_total_c_1),
                            "running_acc_c_2": 100 * zero_div(running_correct_c_2, running_total_c_2),
                            "running_acc_c_3": 100 * zero_div(running_correct_c_3, running_total_c_3),
                            "running_acc_fs_c_1": 100 * zero_div(fs_running_correct_c_1, running_total_c_1),
                            "running_acc_fs_c_2": 100 * zero_div(fs_running_correct_c_2, running_total_c_2),
                            "running_acc_fs_c_3": 100 * zero_div(fs_running_correct_c_3, running_total_c_3),
                            "train_loss_acl_c": running_loss_acl_3 / run_step,
                            "running_acc_acl_c": 100 * zero_div(running_correct_acl_3, running_total_acl_3),
                            "train_loss_acl_ab_c": running_loss_acl_ab_3 / run_step,
                            "running_acc_acl_ab_c": 100 * zero_div(running_correct_acl_ab_3, running_total_acl_ab_3),
                        })
                    else:
                        wandb.log({
                            "train_loss_a_1": running_loss_a_1 / run_step,
                            "train_loss_a_2": running_loss_a_2 / run_step,
                            "train_loss_a_3": running_loss_a_3 / run_step,
                            "running_acc_a_1": 100 * zero_div(running_correct_a_1, running_total_a_1),
                            "running_acc_a_2": 100 * zero_div(running_correct_a_2, running_total_a_2),
                            "running_acc_a_3": 100 * zero_div(running_correct_a_3, running_total_a_3),
                            "train_loss_acl_a": running_loss_acl_1 / run_step,
                            "running_acc_acl_a": 100 * zero_div(running_correct_acl_1, running_total_acl_1),
                            "train_loss_acl_ab_a": running_loss_acl_ab_1 / run_step,
                            "running_acc_acl_ab_a": 100 * zero_div(running_correct_acl_ab_1, running_total_acl_ab_1),
                            "train_loss_b_1": running_loss_b_1 / run_step,
                            "train_loss_b_2": running_loss_b_2 / run_step,
                            "train_loss_b_3": running_loss_b_3 / run_step,
                            "running_acc_b_1": 100 * zero_div(running_correct_b_1, running_total_b_1),
                            "running_acc_b_2": 100 * zero_div(running_correct_b_2, running_total_b_2),
                            "running_acc_b_3": 100 * zero_div(running_correct_b_3, running_total_b_3),
                            "train_loss_acl_b": running_loss_acl_2 / run_step,
                            "running_acc_acl_b": 100 * zero_div(running_correct_acl_2, running_total_acl_2),
                            "train_loss_acl_ab_b": running_loss_acl_ab_2 / run_step,
                            "running_acc_acl_ab_b": 100 * zero_div(running_correct_acl_ab_2, running_total_acl_ab_2),
                            "train_loss_c_1": running_loss_c_1 / run_step,
                            "train_loss_c_2": running_loss_c_2 / run_step,
                            "train_loss_c_3": running_loss_c_3 / run_step,
                            "running_acc_c_1": 100 * zero_div(running_correct_c_1, running_total_c_1),
                            "running_acc_c_2": 100 * zero_div(running_correct_c_2, running_total_c_2),
                            "running_acc_c_3": 100 * zero_div(running_correct_c_3, running_total_c_3),
                            "train_loss_acl_c": running_loss_acl_3 / run_step,
                            "running_acc_acl_c": 100 * zero_div(running_correct_acl_3, running_total_acl_3),
                            "train_loss_acl_ab_c": running_loss_acl_ab_3 / run_step,
                            "running_acc_acl_ab_c": 100 * zero_div(running_correct_acl_ab_3, running_total_acl_ab_3),
                        })
                train_elapsed = time.time() - train_timer
                train_timer = time.time()
                num_images_per_sec = (
                    (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                    // train_elapsed)
                last_batch_logged = i

                # check accurary on the batch.
                # writer.add_scalar("Loss/train", running_loss / run_step, i)
                if use_fs:
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss_a_1: {running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_a_2: {running_loss_a_2 / run_step :.3f}, '
                        f'train_loss_a_3: {running_loss_a_3 / run_step :.3f}, '
                        f'train_loss_fs_a_1: {fs_running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_fs_a_2: {fs_running_loss_a_2 / run_step :.3f}, '
                        f'train_loss_fs_a_3: {fs_running_loss_a_3 / run_step :.3f}, '
                        f'running_acc_a_1: {100 * zero_div(running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_a_2: {100 * zero_div(running_correct_a_2, running_total_a_2):.2f} % '
                        f'running_acc_a_3: {100 * zero_div(running_correct_a_3, running_total_a_3):.2f} % '
                        f'running_acc_fs_a_1: {100 * zero_div(fs_running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_fs_a_2: {100 * zero_div(fs_running_correct_a_2, running_total_a_2):.2f} % '
                        f'running_acc_fs_a_3: {100 * zero_div(fs_running_correct_a_3, running_total_a_3):.2f} % '
                        f'train_loss_ACL_a: {running_loss_acl_1 / run_step :.3f}, '
                        f'train_loss_ACL_ab_a: {running_loss_acl_ab_1 / run_step :.3f}, '
                        f'running_acc_ACL_a: {100 * zero_div(running_correct_acl_1, running_total_acl_1):.2f} %, '
                        f'running_acc_ACL_ab_a: {100 * zero_div(running_correct_acl_ab_1, running_total_acl_ab_1):.2f} %, '
                        f'train_loss_b_1: {running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_b_2: {running_loss_b_2 / run_step :.3f}, '
                        f'train_loss_b_3: {running_loss_b_3 / run_step :.3f}, '
                        f'train_loss_fs_b_1: {fs_running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_fs_b_2: {fs_running_loss_b_2 / run_step :.3f}, '
                        f'train_loss_fs_b_3: {fs_running_loss_b_3 / run_step :.3f}, '
                        f'running_acc_b_1: {100 * zero_div(running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_b_2: {100 * zero_div(running_correct_b_2, running_total_b_2):.2f} % '
                        f'running_acc_b_3: {100 * zero_div(running_correct_b_3, running_total_b_3):.2f} % '
                        f'running_acc_fs_b_1: {100 * zero_div(fs_running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_fs_b_2: {100 * zero_div(fs_running_correct_b_2, running_total_b_2):.2f} % '
                        f'running_acc_fs_b_3: {100 * zero_div(fs_running_correct_b_3, running_total_b_3):.2f} % '
                        f'train_loss_ACL_b: {running_loss_acl_2 / run_step :.3f}, '
                        f'train_loss_ACL_ab_b: {running_loss_acl_ab_2 / run_step :.3f}, '
                        f'running_acc_ACL_b: {100 * zero_div(running_correct_acl_2, running_total_acl_2):.2f} %, '
                        f'running_acc_ACL_ab_b: {100 * zero_div(running_correct_acl_ab_2, running_total_acl_ab_2):.2f} %, '
                        f'train_loss_c_1: {running_loss_c_1 / run_step :.3f}, '
                        f'train_loss_c_2: {running_loss_c_2 / run_step :.3f}, '
                        f'train_loss_c_3: {running_loss_c_3 / run_step :.3f}, '
                        f'train_loss_fs_c_1: {fs_running_loss_c_1 / run_step :.3f}, '
                        f'train_loss_fs_c_2: {fs_running_loss_c_2 / run_step :.3f}, '
                        f'train_loss_fs_c_3: {fs_running_loss_c_3 / run_step :.3f}, '
                        f'running_acc_c_1: {100 * zero_div(running_correct_c_1, running_total_c_1):.2f} % '
                        f'running_acc_c_2: {100 * zero_div(running_correct_c_2, running_total_c_2):.2f} % '
                        f'running_acc_c_3: {100 * zero_div(running_correct_c_3, running_total_c_3):.2f} % '
                        f'running_acc_fs_c_1: {100 * zero_div(fs_running_correct_c_1, running_total_c_1):.2f} % '
                        f'running_acc_fs_c_2: {100 * zero_div(fs_running_correct_c_2, running_total_c_2):.2f} % '
                        f'running_acc_fs_c_3: {100 * zero_div(fs_running_correct_c_3, running_total_c_3):.2f} % '
                        f'train_loss_ACL_c: {running_loss_acl_3 / run_step :.3f}, '
                        f'train_loss_ACL_ab_c: {running_loss_acl_ab_3 / run_step :.3f}, '
                        f'running_acc_ACL_c: {100 * zero_div(running_correct_acl_3, running_total_acl_3):.2f} %, '
                        f'running_acc_ACL_ab_c: {100 * zero_div(running_correct_acl_ab_3, running_total_acl_ab_3):.2f} %, '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')
                else:
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss_a_1: {running_loss_a_1 / run_step :.3f}, '
                        f'train_loss_a_2: {running_loss_a_2 / run_step :.3f}, '
                        f'train_loss_a_3: {running_loss_a_3 / run_step :.3f}, '
                        f'running_acc_a_1: {100 * zero_div(running_correct_a_1, running_total_a_1):.2f} % '
                        f'running_acc_a_2: {100 * zero_div(running_correct_a_2, running_total_a_2):.2f} % '
                        f'running_acc_a_3: {100 * zero_div(running_correct_a_3, running_total_a_3):.2f} % '
                        f'train_loss_ACL_a: {running_loss_acl_1 / run_step :.3f}, '
                        f'train_loss_ACL_ab_a: {running_loss_acl_ab_1 / run_step :.3f}, '
                        f'running_acc_ACL_a: {100 * zero_div(running_correct_acl_1, running_total_acl_1):.2f} %, '
                        f'running_acc_ACL_ab_a: {100 * zero_div(running_correct_acl_ab_1, running_total_acl_ab_1):.2f} %, '
                        f'train_loss_b_1: {running_loss_b_1 / run_step :.3f}, '
                        f'train_loss_b_2: {running_loss_b_2 / run_step :.3f}, '
                        f'train_loss_b_3: {running_loss_b_3 / run_step :.3f}, '
                        f'running_acc_b_1: {100 * zero_div(running_correct_b_1, running_total_b_1):.2f} % '
                        f'running_acc_b_2: {100 * zero_div(running_correct_b_2, running_total_b_2):.2f} % '
                        f'running_acc_b_3: {100 * zero_div(running_correct_b_3, running_total_b_3):.2f} % '
                        f'train_loss_ACL_b: {running_loss_acl_2 / run_step :.3f}, '
                        f'train_loss_ACL_ab_b: {running_loss_acl_ab_2 / run_step :.3f}, '
                        f'running_acc_ACL_b: {100 * zero_div(running_correct_acl_2, running_total_acl_2):.2f} %, '
                        f'running_acc_ACL_ab_b: {100 * zero_div(running_correct_acl_ab_2, running_total_acl_ab_2):.2f} %, '
                        f'train_loss_c_1: {running_loss_c_1 / run_step :.3f}, '
                        f'train_loss_c_2: {running_loss_c_2 / run_step :.3f}, '
                        f'train_loss_c_3: {running_loss_c_3 / run_step :.3f}, '
                        f'running_acc_c_1: {100 * zero_div(running_correct_c_1, running_total_c_1):.2f} % '
                        f'running_acc_c_2: {100 * zero_div(running_correct_c_2, running_total_c_2):.2f} % '
                        f'running_acc_c_3: {100 * zero_div(running_correct_c_3, running_total_c_3):.2f} % '
                        f'train_loss_ACL_c: {running_loss_acl_3 / run_step :.3f}, '
                        f'train_loss_ACL_ab_c: {running_loss_acl_ab_3 / run_step :.3f}, '
                        f'running_acc_ACL_c: {100 * zero_div(running_correct_acl_3, running_total_acl_3):.2f} %, '
                        f'running_acc_ACL_ab_c: {100 * zero_div(running_correct_acl_ab_3, running_total_acl_ab_3):.2f} %, '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')

                running_loss_a_1 = 0.0
                running_loss_a_2 = 0.0
                running_loss_a_3 = 0.0

                running_correct_a_1 = 0.0
                running_correct_a_2 = 0.0
                running_correct_a_3 = 0.0
                running_total_a_1 = 0
                running_total_a_2 = 0
                running_total_a_3 = 0

                running_loss_b_1 = 0.0
                running_loss_b_2 = 0.0
                running_loss_b_3 = 0.0

                running_correct_b_1 = 0.0
                running_correct_b_2 = 0.0
                running_correct_b_3 = 0.0
                running_total_b_1 = 0
                running_total_b_2 = 0
                running_total_b_3 = 0

                running_loss_c_1 = 0.0
                running_loss_c_2 = 0.0
                running_loss_c_3 = 0.0

                running_correct_c_1 = 0.0
                running_correct_c_2 = 0.0
                running_correct_c_3 = 0.0
                running_total_c_1 = 0
                running_total_c_2 = 0
                running_total_c_3 = 0


                fs_running_loss_a_1 = 0.0
                fs_running_loss_a_2 = 0.0
                fs_running_loss_a_3 = 0.0

                fs_running_loss_b_1 = 0.0
                fs_running_loss_b_2 = 0.0
                fs_running_loss_b_3 = 0.0

                fs_running_loss_c_1 = 0.0
                fs_running_loss_c_2 = 0.0
                fs_running_loss_c_3 = 0.0

                fs_running_correct_a_1 = 0.0
                fs_running_correct_a_2 = 0
                fs_running_correct_a_3 = 0

                fs_running_correct_b_1 = 0.0
                fs_running_correct_b_2 = 0
                fs_running_correct_b_3 = 0

                fs_running_correct_c_1 = 0.0
                fs_running_correct_c_2 = 0
                fs_running_correct_c_3 = 0

                running_loss_acl_1 = 0.0
                running_correct_acl_1 = 0.0
                running_correct_acl_ab_1 = 0.0
                running_total_acl_1 = 0.0
                running_total_acl_ab_1 = 0.0

                running_loss_acl_2 = 0.0
                running_correct_acl_2 = 0.0
                running_correct_acl_ab_2 = 0.0
                running_total_acl_2 = 0.0
                running_total_acl_ab_2 = 0.0

                running_loss_acl_3 = 0.0
                running_correct_acl_3 = 0.0
                running_correct_acl_ab_3 = 0.0
                running_total_acl_3 = 0.0
                running_total_acl_ab_3 = 0.0

                running_loss_acl_ab_1 = 0.0
                running_loss_acl_ab_2 = 0.0
                running_loss_acl_ab_3 = 0.0

                running_total = 0
                run_step = 0

            # ======================================================================

            if i % args.validate_every == 0:  # run validation
                model.eval()
                with torch.no_grad():
                    v_total_a, v_state = eval_model_label_sync(
                        model, val_dataloader_a, num_steps=args.valid_size,
                        get_state=True, get_second_last_state=True)
                    test_total_a, test_state = eval_model_label_sync(
                        model, test_dataloader_a, num_steps=args.test_size,
                        get_state=True, get_second_last_state=True)

                    v_total_b, v_state = eval_model_label_sync(
                        model, val_dataloader_b, num_steps=args.valid_size,
                        state=v_state, get_state=True,
                        get_second_last_state=True)
                    test_total_b, test_state = eval_model_label_sync(
                        model, test_dataloader_b, num_steps=args.test_size,
                        state=test_state, get_state=True,
                        get_second_last_state=True)

                    v_total_c = eval_model_label_sync(
                        model, val_dataloader_c, num_steps=args.valid_size,
                        state=v_state, get_state=False)
                    test_total_c = eval_model_label_sync(
                        model, test_dataloader_c, num_steps=args.test_size,
                        state=test_state, get_state=False)

                loginf(
                    f"[val {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
                    f'val total A {100 * v_total_a :.2f} %, '
                    f'val total B {100 * v_total_b :.2f} %, '
                    f'val total C {100 * v_total_c :.2f} %, ')

                loginf(
                    f'test acc A {100 * test_total_a :.2f} %, '
                    f'test acc B {100 * test_total_b :.2f} %, '
                    f'test acc C {100 * test_total_c :.2f} %')  # debugging

                if use_wandb:
                    wandb.log({
                        "val_acc_a": 100 * v_total_a,
                        "test_acc_a": 100 * test_total_a,
                        "val_acc_b": 100 * v_total_b,
                        "test_acc_b": 100 * test_total_b,
                        "val_acc_c": 100 * v_total_c,
                        "test_acc_c": 100 * test_total_c,
                    })

                avg_v = (v_total_a + v_total_b + v_total_c) * 0.333
                if avg_v > best_val_first_shot_acc:
                    best_val_first_shot_acc = avg_v
                    best_step = i + offset_step
                    # Save the best model
                    loginf("The best model so far.")
                    if args.context_carry_over:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'state': state,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    else:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    loginf("Saved.")
                    if test_total_b > best_valid_test_first_shot_acc:
                        best_valid_test_first_shot_acc = test_total_b
                if test_total_b > best_test_first_shot_acc:
                    best_test_first_shot_acc = test_total_b
                loginf(
                    f'current best valid_acc: {100 * best_val_first_shot_acc :.2f} '
                    f'%\ncurrent best valid test_acc on B: '
                    f'{100 * best_valid_test_first_shot_acc :.2f} %\n'
                    f'current best test_acc on B: {100 * best_test_first_shot_acc :.2f} ')
                # Save the latest model
                if args.context_carry_over:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'state': state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': v_total_b}, lastest_model_path)
                else:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': v_total_b}, lastest_model_path)

                if args.ood_eval:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                            # double shot
                            self_train_input = extra_train_data_part2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels_part2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct_doubleshot += bool_correct_pred.sum().item()

                    external_acc = 100 * extra_running_correct / total_test_samples
                    external_acc_doubleshot = 100 * extra_running_correct_doubleshot / total_test_samples
                    loginf(f'Extra test acc: {external_acc:.2f} %')
                    loginf(f'Extra test double shot acc: {external_acc_doubleshot:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_acc": external_acc,
                            "extra_double_acc": external_acc_doubleshot,
                        })
                    if best_external_acc < external_acc:
                        best_external_acc = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)	

                if args.ood_eval2:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader2):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'CIFAR10 test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_cifar10_acc": external_acc,
                        })
                    if best_external_acc2 < external_acc:
                        best_external_acc2 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)	

                if args.ood_eval3:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader3):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data3.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels3.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'SVHN test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_svhn_acc": external_acc,
                        })
                    if best_external_acc3 < external_acc:
                        best_external_acc3 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)	

                elapsed = time.time() - interval_start_time
                loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
                interval_start_time = time.time()
                train_timer = time.time()

            if cur_train_acc > args.train_acc_stop:
                loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
                end_training = True
                break
            if i + offset_step > args.total_train_steps:
                end_training = True
                loginf(f'reached {args.total_train_steps} steps')
                break
            if args.freeze_out_emb:
                if args.freeze_after_steps < i:
                    for param in model.out_layer.parameters():
                        param.requires_grad = False      
                    # loginf(f"Step {i}: freezing output embeddings")  
            if args.freeze_after:
                if args.freeze_after_steps < i:
                    # loginf(f"Step {i}: freezing conv stem")  
                    if model_name in ['srwm', 'deltanet']:
                        for param in model.conv_layers.parameters():
                            param.requires_grad = False
                    elif model_name in ['res12_srwm', 'res12_deltanet']:
                        for param in model.stem_resnet12.parameters():
                            param.requires_grad = False
        if end_training:
            break
        offset_step += i


# splitmnist domain incremental
elif args.train_splitmnist_style and not skip_train:
    if args.metaval_fashion:
        loginf("Preparing Split-FashionMNIST...")
        norm_params = {'mean': [0.286], 'std': [0.353]}
        MetavalDataset = torchvision.datasets.FashionMNIST
    elif args.metaval_cifar:
        loginf("Preparing Split-CIFAR10...")
        norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
        MetavalDataset = torchvision.datasets.CIFAR10
    else:
        loginf("Preparing Split-MNIST...")
        norm_params = {'mean': [0.1307], 'std': [0.3081]}
        MetavalDataset = torchvision.datasets.MNIST

    extra_dataset = MetavalDataset(
        download=True, root=args.data_dir, train=True)

    if args.metaval_cifar:
        idx = np.arange(extra_dataset.__len__())
        val_indices = idx[50000-5000:]
        train_indices= idx[:-5000]
        if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
            compat_shape = [3, 84, 84]
            mnist_transform = Compose(
                [Resize(84), ToTensor(), Normalize(**norm_params)])
        elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
            compat_shape = [3, 32, 32]
            mnist_transform = Compose(
                [Resize(32), ToTensor(), Normalize(**norm_params)])
        else:
            assert 'omni' in args.name_dataset
            compat_shape = [1, 28, 28]
            mnist_transform = Compose(
                [Resize(28), ToTensor(), Normalize(**norm_params)])
    else:
        idx = np.arange(extra_dataset.__len__())
        val_indices = idx[60000-5000:]
        train_indices= idx[:-5000]
        if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
            compat_shape = [3, 84, 84]
            mnist_transform = Compose(
                [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
        elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
            compat_shape = [3, 32, 32]
            mnist_transform = Compose(
                [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
        else:
            assert 'omni' in args.name_dataset
            compat_shape = [1, 28, 28]
            mnist_transform = Compose(
                [Resize(28), ToTensor(), Normalize(**norm_params)])

    from torch.utils.data import Dataset
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

        def __len__(self):
            return len(self.data)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_valid_loaders = {}

    for split_id in range(5):  # 5 tasks
        extra_val_dataset = MetavalDataset(
            download=True, root=args.data_dir, train=True, transform=mnist_transform)

        if args.metaval_cifar:
            # extra_val_dataset.targets = extra_val_dataset.targets[val_indices]
            tmp_targets = torch.ByteTensor(extra_val_dataset.targets)
            tmp_targets = tmp_targets[val_indices]
            idx_0 = tmp_targets == (split_id * 2)
            idx_1 = tmp_targets == (split_id * 2+1)
            idx = torch.logical_or(idx_0, idx_1)
            extra_val_dataset.targets = (tmp_targets[idx] - split_id * 2).tolist() 
        else:
            extra_val_dataset.targets = extra_val_dataset.targets[val_indices]
            idx_0 = extra_val_dataset.train_labels == (split_id * 2)
            idx_1 = extra_val_dataset.train_labels == (split_id * 2+1)
            idx = torch.logical_or(idx_0, idx_1)
            extra_val_dataset.targets = extra_val_dataset.targets[idx] - split_id * 2

        extra_val_dataset.data = extra_val_dataset.data[val_indices][idx]

        extra_valid_loader = torch.utils.data.DataLoader(
            dataset=extra_val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_valid_loaders[split_id] = extra_valid_loader

    after_all_tasks_correct = {}
    after_all_tasks_total = {}
    for task_id in range(5):
        after_all_tasks_correct[task_id] = 0
        after_all_tasks_total[task_id] = 0

    for ep in range(args.total_epoch):
        loginf(f'EPOCH {ep} ====================')
        i = -1
        # for i, (batch_1, batch_2) in enumerate(zip_dataloader_a_b):
        while True:  # TODO fix me, now it should continue running while the dataloader is not empty
            i += 1
            task_batch = {}
            model.train()
            for task_id in range(5):  # 5 tasks
                # get batch
                task_batch[task_id] = iter(dataloader_a).next()

            # state = None
            if args.context_carry_over_double:
                if i % 2 == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif args.context_carry_over_k > 1:
                if i % args.context_carry_over_k == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif not args.context_carry_over:
                state = None
            elif state is not None:
                state = model.clone_state(state, detach=True)

            test_input_dict = []
            test_target_dict = []

            train_input_dict = []
            train_target_dict = []

            net_input_dict = []
            net_target_labels_dict = []
            label_feedback_dict = []

            # prepare data batches
            for task_id in range(5):
                batch_a = task_batch[task_id]

                # Extract test examples:
                test_inputs_a, test_targets_a = batch_a['test']
                test_inputs_a = test_inputs_a.to(device=device)  # (B, test_len, 28 * 28)
                test_targets_a = test_targets_a.to(device=device)

                test_inputs_a = test_inputs_a.transpose(0, 1)  # (test_len, B, 28 * 28)
                test_targets_a = test_targets_a.transpose(0, 1)  # (test_len, B)

                # take only the fist element (randomized already)
                test_inputs_a = test_inputs_a[0].unsqueeze(0)
                test_targets_a = test_targets_a[0].unsqueeze(0)

                # better with dict? let's see
                test_input_dict.append(test_inputs_a)
                test_target_dict.append(test_targets_a)

                # Extract train examples ##########################################
                train_inputs_a, train_targets_a = batch_a['train']
                train_inputs_a = train_inputs_a.to(device=device)  # (B, len, 1, 28, 28)
                train_targets_a = train_targets_a.to(device=device)  # (B, len)

                # shuffle and reshape
                train_shape = train_inputs_a.shape
                bsz, slen = train_shape[0], train_shape[1]

                num_seq += bsz

                train_inputs_a = train_inputs_a.transpose(0, 1)  # (len, B, 28 * 28)
                train_targets_a = train_targets_a.transpose(0, 1)  # (len, B)

                train_input_dict.append(train_inputs_a)
                train_target_dict.append(train_targets_a)

                # do the main part
                net_input_a = torch.cat([train_inputs_a, test_inputs_a], dim=0)
                target_labels_a = torch.cat([train_targets_a, test_targets_a], dim=0)

                target_labels_shape = target_labels_a.shape
                assert target_labels_shape[0] == slen + 1
                assert target_labels_shape[1] == bsz

                sync_labels = target_labels_a[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    assert model.num_classes == 2
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback_a = torch.cat([sync_labels, dummy_last_token], dim=0)

                net_input_dict.append(net_input_a)
                net_target_labels_dict.append(target_labels_a[-1].reshape(-1))
                label_feedback_dict.append(label_feedback_a)

            # -- forward all: A_context, A_predict, B_context, B_predict
            # one forward pass to update BN stats
            if not args.use_instance_norm:
                with torch.no_grad():
                    net_input_dummy = torch.cat([net_input_dict], dim=0)
                    label_feedback_dummy = torch.cat([label_feedback_dict], dim=0)
                    outputs_dummy, _ = model(net_input_dummy, label_feedback_dummy, state)

            # Go through each tasks
            acl_loss_list = []
            loss_list = []
            after_all_tasks_acl_acc = []

            for task_id in range(5):
                batch_a = task_batch[task_id]
                if use_fs:
                    fs_train_inputs, fs_train_targets = batch_a['first_shot']
                    fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                    fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                    # shuffle and reshape
                    fs_train_shape = fs_train_inputs.shape
                    fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                    fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                    _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                    net_input = torch.cat([fs_train_inputs, test_input_dict[task_id]], dim=0)
                    target_labels = torch.cat([fs_train_targets, test_target_dict[task_id]], dim=0)
                    # target_labels = target_labels[-1].reshape(-1)

                    target_labels_shape = target_labels.shape
                    assert target_labels_shape[0] == fs_slen + 1

                    sync_labels = target_labels[:-1]
                    label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                    # if state is not None:
                    #     state = model.clone_state(state)

                    # do not update BN stats on this small batch
                    model.set_bn_in_eval_mode()
                    outputs, _ = model(net_input, label_feedback, state)
                    model.set_bn_in_train_mode()
                    # not carrying states forward from one-shot learning

                    outputs = outputs[-1]
                    outputs = outputs.reshape(-1, num_classes)

                    loss_fs_a = loss_fn(
                        outputs, net_target_labels_dict[task_id])

                    with torch.no_grad():
                        _, predicted = outputs.max(-1)
                    bool_correct_pred = (
                        predicted == net_target_labels_dict[task_id])

                model.set_bn_in_eval_mode()
                _, state = model(
                    train_input_dict[task_id], train_target_dict[task_id], state)
                state = model.clone_state(state)
                outputs, _ = model(
                    test_input_dict[task_id], dummy_last_token, state)
                model.set_bn_in_train_mode()

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                # TODO loss factor
                loss_list.append(
                    loss_fn(outputs, net_target_labels_dict[task_id]))

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == net_target_labels_dict[task_id])

                if task_id == 4:
                    after_all_tasks_correct[task_id] += bool_correct_pred.sum().item()
                    after_all_tasks_total[task_id] += net_target_labels_dict[task_id].size(0)

                if task_id > 0:  # ACL part, evaluate forgetting
                    for prev_task_id in range(0, task_id):
                        net_input = test_input_dict[prev_task_id]
                        target_labels = test_target_dict[prev_task_id]

                        model.set_bn_in_eval_mode()
                        outputs, _ = model(net_input, dummy_last_token, state)
                        model.set_bn_in_train_mode()

                        outputs = outputs[-1]
                        outputs = outputs.reshape(-1, num_classes)
                        target_labels = target_labels[-1].reshape(-1)

                        acl_loss_list.append(
                            loss_fn(outputs, target_labels))

                        with torch.no_grad():
                            _, predicted = outputs.max(-1)
                        bool_correct_pred = (predicted == target_labels)

                        if task_id == 4:
                           after_all_tasks_correct[prev_task_id] += bool_correct_pred.sum().item()
                           after_all_tasks_total[prev_task_id] += target_labels.size(0)

            # compute loss
            # more scaling? for now just mean
            loss = torch.stack(loss_list + acl_loss_list, dim=0).mean()
            loss.backward()

            if i % args.grad_cummulate == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                model.reset_grad()

            model.eval()

            run_step += 1
            if i % args.report_every == 0:
                after_all_tasks_acc = []
                log_msg = {}
                for task_id in range(5):
                    acc = 100 * zero_div(
                        after_all_tasks_correct[task_id],
                        after_all_tasks_total[task_id])
                    log_msg[f'final_train_acc_{task_id}'] = acc

                if use_wandb:
                    wandb.log(log_msg)

                train_elapsed = time.time() - train_timer
                train_timer = time.time()
                num_images_per_sec = (
                    (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                    // train_elapsed)
                last_batch_logged = i

                loginf(
                    f'steps: {i + offset_step}, num_seq: {num_seq} (elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} images/s)')
                loginf(log_msg)

                running_total = 0
                run_step = 0

                for task_id in range(5):
                    after_all_tasks_correct[task_id] = 0
                    after_all_tasks_total[task_id] = 0

            # ======================================================================

            if i % args.validate_every == 0:  # run validation
                model.eval()
                with torch.no_grad():
                    num_extra_test = 5
                    bsz = args.batch_size

                    model.eval()

                    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

                    # Do single task eval first.
                    store_results_a = []
                    store_results_b = []
                    store_results_c = []
                    store_results_d = []
                    store_results_e = []

                    for run_id in range(num_extra_test):
                        # TASK A
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 0
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):
                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
                        store_results_a.append(external_acc)

                        # Task B
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 1
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
                        store_results_b.append(external_acc)

                        # Task C
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 2
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M45, M45] test acc: {external_acc:.2f} %')
                        store_results_c.append(external_acc)

                        # Task D
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 3
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M67, M67] test acc: {external_acc:.2f} %')
                        store_results_d.append(external_acc)


                        # Task E
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 4
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M89, M89] test acc: {external_acc:.2f} %')
                        store_results_e.append(external_acc)


                    mean_a = np.mean(store_results_a)
                    std_a = np.std(store_results_a)
                    mean_b = np.mean(store_results_b)
                    std_b = np.std(store_results_b)
                    mean_c = np.mean(store_results_c)
                    std_c = np.std(store_results_c)
                    mean_d = np.mean(store_results_d)
                    std_d = np.std(store_results_d)
                    mean_e = np.mean(store_results_e)
                    std_e = np.std(store_results_e)

                    loginf(
                        f'[== {num_extra_test} runs: M01, M01 ==] '
                        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
                        f'[== {num_extra_test} runs: M23, M23 ==] '
                        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
                        f'[== {num_extra_test} runs: M45, M45 ==] '
                        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
                        f'[== {num_extra_test} runs: M67, M67 ==] '
                        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
                        f'[== {num_extra_test} runs: M89, M89 ==] '
                        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
                        )

                    # ACL eval

                    store_results_a_a = []
                    store_results_ab_b = []
                    store_results_ab_a = []

                    store_results_abc_c = []
                    store_results_abc_a = []
                    store_results_abc_b = []

                    store_results_abcd_d = []
                    store_results_abcd_c = []
                    store_results_abcd_b = []
                    store_results_abcd_a = []

                    store_results_abcde_e = []
                    store_results_abcde_d = []
                    store_results_abcde_c = []
                    store_results_abcde_b = []
                    store_results_abcde_a = []

                    for run_id in range(num_extra_test):
                        # MNIST -> CIFAR-10, MNIST
                        ########## part 1
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 0
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
                        store_results_a_a.append(external_acc)

                        # MNIST -> F-MNIST, F-MNIST
                        ########## part 2, new data
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 1
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
                        store_results_ab_b.append(external_acc)

                        ########## part 2, ACL 1/1
                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
                        store_results_ab_a.append(external_acc)

                        ########## part 3, new data
                        # MNIST, CIFAR10 ->M59
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 2
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M45] test acc: {external_acc:.2f} %')
                        store_results_abc_c.append(external_acc)

                        ########## part 3, ACL 1/2

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M01] test acc: {external_acc:.2f} %')
                        store_results_abc_a.append(external_acc)

                        ########## part 3, ACL 2/2

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[1]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M23] test acc: {external_acc:.2f} %')
                        store_results_abc_b.append(external_acc)


                        ########## part 4, new data
                        # MNIST, CIFAR10, M59, F-MNIST
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 3
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M67] test acc: {external_acc:.2f} %')
                        store_results_abcd_d.append(external_acc)

                        ########## part 4, ACL 1/3

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M01] test acc: {external_acc:.2f} %')
                        store_results_abcd_a.append(external_acc)

                        ########## part 4, ACL 2/3

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[1]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M23] test acc: {external_acc:.2f} %')
                        store_results_abcd_b.append(external_acc)


                        ########## part 4, ACL 3/3

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[2]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M45] test acc: {external_acc:.2f} %')
                        store_results_abcd_c.append(external_acc)


                        ########## part 5, new data
                        # MNIST, CIFAR10, M59, F-MNIST, SVHN
                        avg_this_run = []
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 4
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M89] test acc: {external_acc:.2f} %')
                        store_results_abcde_e.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 1/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M01] test acc: {external_acc:.2f} %')
                        store_results_abcde_a.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 2/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[1]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M23] test acc: {external_acc:.2f} %')
                        store_results_abcde_b.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 3/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[2]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M45] test acc: {external_acc:.2f} %')
                        store_results_abcde_c.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 4/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[3]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)[:,:2]
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M67] test acc: {external_acc:.2f} %')
                        store_results_abcde_d.append(external_acc)
                        avg_this_run.append(external_acc)
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89] Average acc: {np.mean(avg_this_run):.2f} %')


                    mean_a_a = np.mean(store_results_a_a)
                    std_a_a = np.std(store_results_a_a)

                    mean_ab_b = np.mean(store_results_ab_b)
                    std_ab_b = np.std(store_results_ab_b)
                    mean_ab_a = np.mean(store_results_ab_a)
                    std_ab_a = np.std(store_results_ab_a)

                    mean_abc_c = np.mean(store_results_abc_c)
                    std_abc_c = np.std(store_results_abc_c)
                    mean_abc_b = np.mean(store_results_abc_b)
                    std_abc_b = np.std(store_results_abc_b)
                    mean_abc_a = np.mean(store_results_abc_a)
                    std_abc_a = np.std(store_results_abc_a)

                    mean_abcd_d = np.mean(store_results_abcd_d)
                    std_abcd_d = np.std(store_results_abcd_d)
                    mean_abcd_c = np.mean(store_results_abcd_c)
                    std_abcd_c = np.std(store_results_abcd_c)
                    mean_abcd_b = np.mean(store_results_abcd_b)
                    std_abcd_b = np.std(store_results_abcd_b)
                    mean_abcd_a = np.mean(store_results_abcd_a)
                    std_abcd_a = np.std(store_results_abcd_a)

                    mean_abcde_d = np.mean(store_results_abcde_d)
                    std_abcde_d = np.std(store_results_abcde_d)
                    mean_abcde_c = np.mean(store_results_abcde_c)
                    std_abcde_c = np.std(store_results_abcde_c)
                    mean_abcde_b = np.mean(store_results_abcde_b)
                    std_abcde_b = np.std(store_results_abcde_b)
                    mean_abcde_a = np.mean(store_results_abcde_a)
                    std_abcde_a = np.std(store_results_abcde_a)
                    mean_abcde_e = np.mean(store_results_abcde_e)
                    std_abcde_e = np.std(store_results_abcde_e)

                    avg_acc_overall = []
                    for run_id in range(num_extra_test):
                        final_acc = []
                        final_acc.append(store_results_abcde_a[run_id])
                        final_acc.append(store_results_abcde_b[run_id])
                        final_acc.append(store_results_abcde_c[run_id])
                        final_acc.append(store_results_abcde_d[run_id])
                        final_acc.append(store_results_abcde_e[run_id])

                        task_acc = np.mean(final_acc)
                        avg_acc_overall.append(task_acc)

                    loginf(
                        f'[== {num_extra_test} runs: M01, M01 ==] '
                        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
                        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
                        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

                        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M45 ==] '
                        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M01 ==] '
                        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M23 ==] '
                        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M67 ==] '
                        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
                        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M23 ==] '
                        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
                        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'

                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M89 ==] '
                        f'mean: {mean_abcde_e:.2f}, std: {std_abcde_e:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M01 ==] '
                        f'mean: {mean_abcde_a:.2f}, std: {std_abcde_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M23 ==] '
                        f'mean: {mean_abcde_b:.2f}, std: {std_abcde_b:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M45 ==] '
                        f'mean: {mean_abcde_c:.2f}, std: {std_abcde_c:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M67 ==] '
                        f'mean: {mean_abcde_d:.2f}, std: {std_abcde_d:.2f} \n'
                        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
                        )
                after_all_tasks_acc = []
                log_msg = {}
                for task_id in range(5):
                    acc = 100 * zero_div(
                        after_all_tasks_correct[task_id],
                        after_all_tasks_total[task_id])
                    log_msg[f'final_train_acc_{task_id}'] = acc

                if use_wandb:
                    wandb.log({'valid_all_task_mean_acc': np.mean(avg_acc_overall),
                               'valid_task_0': mean_abcde_a,
                               'valid_task_1': mean_abcde_b,
                               'valid_task_2': mean_abcde_c,
                               'valid_task_3': mean_abcde_d,
                               'valid_task_4': mean_abcde_e,
                               })

                avg_v = np.mean(avg_acc_overall)
                if avg_v > best_val_first_shot_acc:
                    best_val_first_shot_acc = avg_v
                    best_step = i + offset_step
                    # Save the best model
                    loginf("The best model so far.")
                    if args.context_carry_over:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'state': state,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    else:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    loginf("Saved.")

                # Save the latest model
                if args.context_carry_over:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'state': state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': avg_v}, lastest_model_path)
                else:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': avg_v}, lastest_model_path)

                elapsed = time.time() - interval_start_time
                loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
                interval_start_time = time.time()
                train_timer = time.time()

            if cur_train_acc > args.train_acc_stop:
                loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
                end_training = True
                break
            if i + offset_step > args.total_train_steps:
                end_training = True
                loginf(f'reached {args.total_train_steps} steps')
                break
            if args.freeze_out_emb:
                if args.freeze_after_steps < i:
                    for param in model.out_layer.parameters():
                        param.requires_grad = False      
                    # loginf(f"Step {i}: freezing output embeddings")  
            if args.freeze_after:
                if args.freeze_after_steps < i:
                    # loginf(f"Step {i}: freezing conv stem")  
                    if model_name in ['srwm', 'deltanet']:
                        for param in model.conv_layers.parameters():
                            param.requires_grad = False
                    elif model_name in ['res12_srwm', 'res12_deltanet']:
                        for param in model.stem_resnet12.parameters():
                            param.requires_grad = False
        if end_training:
            break
        offset_step += i


# splitmnist
elif args.train_splitmnist_style_class_incremental and not skip_train:
    if args.metaval_fashion:
        loginf("Preparing Split-FashionMNIST...")
        norm_params = {'mean': [0.286], 'std': [0.353]}
        MetavalDataset = torchvision.datasets.FashionMNIST
    elif args.metaval_cifar:
        loginf("Preparing Split-CIFAR10...")
        norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
        MetavalDataset = torchvision.datasets.CIFAR10
    else:
        loginf("Preparing Split-MNIST...")
        norm_params = {'mean': [0.1307], 'std': [0.3081]}
        MetavalDataset = torchvision.datasets.MNIST

    extra_dataset = MetavalDataset(
        download=True, root=args.data_dir, train=True)

    if args.metaval_cifar:
        idx = np.arange(extra_dataset.__len__())
        val_indices = idx[50000-5000:]
        train_indices= idx[:-5000]
        if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
            compat_shape = [3, 84, 84]
            mnist_transform = Compose(
                [Resize(84), ToTensor(), Normalize(**norm_params)])
        elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
            compat_shape = [3, 32, 32]
            mnist_transform = Compose(
                [Resize(32), ToTensor(), Normalize(**norm_params)])
        else:
            assert 'omni' in args.name_dataset
            compat_shape = [1, 28, 28]
            mnist_transform = Compose(
                [Resize(28), ToTensor(), Normalize(**norm_params)])
    else:
        if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
            compat_shape = [3, 84, 84]
            mnist_transform = Compose(
                [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
        elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
            compat_shape = [3, 32, 32]
            mnist_transform = Compose(
                [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
        else:
            assert 'omni' in args.name_dataset
            compat_shape = [1, 28, 28]
            mnist_transform = Compose(
                [Resize(28), ToTensor(), Normalize(**norm_params)])

        idx = np.arange(extra_dataset.__len__())
        val_indices = idx[60000-5000:]
        train_indices= idx[:-5000]

    # extra_dataset.targets = extra_dataset.targets[train_indices]
    # extra_dataset.data = extra_dataset.data[train_indices]

    from torch.utils.data import Dataset
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

        def __len__(self):
            return self.data

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_valid_loaders = {}

    for split_id in range(5):  # 5 tasks
        extra_val_dataset = MetavalDataset(
            download=True, root=args.data_dir, train=True, transform=mnist_transform)

        if args.metaval_cifar:
            # extra_val_dataset.targets = extra_val_dataset.targets[val_indices]
            tmp_targets = torch.ByteTensor(extra_val_dataset.targets)
            tmp_targets = tmp_targets[val_indices]
            idx_0 = tmp_targets == (split_id * 2)
            idx_1 = tmp_targets == (split_id * 2+1)
            idx = torch.logical_or(idx_0, idx_1)
            extra_val_dataset.targets = (tmp_targets[idx] - split_id * 2).tolist() 
        else:
            extra_val_dataset.targets = extra_val_dataset.targets[val_indices]
            idx_0 = extra_val_dataset.train_labels == (split_id * 2)
            idx_1 = extra_val_dataset.train_labels == (split_id * 2+1)
            idx = torch.logical_or(idx_0, idx_1)
            extra_val_dataset.targets = extra_val_dataset.targets[idx] - split_id * 2

        extra_val_dataset.data = extra_val_dataset.data[val_indices][idx]

        extra_valid_loader = torch.utils.data.DataLoader(
            dataset=extra_val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_valid_loaders[split_id] = extra_valid_loader

    after_all_tasks_correct = {}
    after_all_tasks_total = {}
    for task_id in range(5):
        after_all_tasks_correct[task_id] = 0
        after_all_tasks_total[task_id] = 0

    for ep in range(args.total_epoch):
        loginf(f'EPOCH {ep} ====================')
        i = -1
        # for i, (batch_1, batch_2) in enumerate(zip_dataloader_a_b):
        while True:  # TODO fix me, now it should continue running while the dataloader is not empty
            i += 1
            task_batch = {}
            model.train()
            for task_id in range(5):  # 5 tasks
                # get batch
                task_batch[task_id] = iter(dataloader_a).next()

            # state = None
            if args.context_carry_over_double:
                if i % 2 == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif args.context_carry_over_k > 1:
                if i % args.context_carry_over_k == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif not args.context_carry_over:
                state = None
            elif state is not None:
                state = model.clone_state(state, detach=True)

            test_input_dict = []
            test_target_dict = []

            train_input_dict = []
            train_target_dict = []

            net_input_dict = []
            net_target_labels_dict = []
            label_feedback_dict = []

            # prepare data batches
            for task_id in range(5):
                batch_a = task_batch[task_id]

                # Extract test examples:
                test_inputs_a, test_targets_a = batch_a['test']
                test_inputs_a = test_inputs_a.to(device=device)  # (B, test_len, 28 * 28)
                test_targets_a = test_targets_a.to(device=device)

                test_inputs_a = test_inputs_a.transpose(0, 1)  # (test_len, B, 28 * 28)
                test_targets_a = test_targets_a.transpose(0, 1)  # (test_len, B)

                # take only the fist element (randomized already)
                test_inputs_a = test_inputs_a[0].unsqueeze(0)
                test_targets_a = test_targets_a[0].unsqueeze(0)

                # class incremental
                test_targets_a = test_targets_a + 2 * task_id

                # better with dict? let's see
                test_input_dict.append(test_inputs_a)
                test_target_dict.append(test_targets_a)

                # Extract train examples ##########################################
                train_inputs_a, train_targets_a = batch_a['train']
                train_inputs_a = train_inputs_a.to(device=device)  # (B, len, 1, 28, 28)
                train_targets_a = train_targets_a.to(device=device)  # (B, len)

                # shuffle and reshape
                train_shape = train_inputs_a.shape
                bsz, slen = train_shape[0], train_shape[1]

                num_seq += bsz

                train_inputs_a = train_inputs_a.transpose(0, 1)  # (len, B, 28 * 28)
                train_targets_a = train_targets_a.transpose(0, 1)  # (len, B)

                train_targets_a = train_targets_a + 2 * task_id

                train_input_dict.append(train_inputs_a)
                train_target_dict.append(train_targets_a)

                # do the main part
                net_input_a = torch.cat([train_inputs_a, test_inputs_a], dim=0)
                target_labels_a = torch.cat([train_targets_a, test_targets_a], dim=0)

                target_labels_shape = target_labels_a.shape
                assert target_labels_shape[0] == slen + 1
                assert target_labels_shape[1] == bsz

                sync_labels = target_labels_a[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    assert model.num_classes == 10
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback_a = torch.cat([sync_labels, dummy_last_token], dim=0)

                net_input_dict.append(net_input_a)
                net_target_labels_dict.append(target_labels_a[-1].reshape(-1))
                label_feedback_dict.append(label_feedback_a)

            # -- forward all: A_context, A_predict, B_context, B_predict
            # one forward pass to update BN stats
            if not args.use_instance_norm:
                with torch.no_grad():
                    net_input_dummy = torch.cat([net_input_dict], dim=0)
                    label_feedback_dummy = torch.cat([label_feedback_dict], dim=0)
                    outputs_dummy, _ = model(net_input_dummy, label_feedback_dummy, state)

            # Go through each tasks
            acl_loss_list = []
            loss_list = []
            after_all_tasks_acl_acc = []

            for task_id in range(5):
                batch_a = task_batch[task_id]
                if use_fs:
                    fs_train_inputs, fs_train_targets = batch_a['first_shot']
                    fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                    fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                    # shuffle and reshape
                    fs_train_shape = fs_train_inputs.shape
                    fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                    fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                    _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                    fs_train_targets = fs_train_targets + 2 * task_id

                    net_input = torch.cat([fs_train_inputs, test_input_dict[task_id]], dim=0)
                    target_labels = torch.cat([fs_train_targets, test_target_dict[task_id]], dim=0)
                    # target_labels = target_labels[-1].reshape(-1)

                    target_labels_shape = target_labels.shape
                    assert target_labels_shape[0] == fs_slen + 1

                    sync_labels = target_labels[:-1]
                    label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                    # do not update BN stats on this small batch
                    model.set_bn_in_eval_mode()
                    outputs, _ = model(net_input, label_feedback, state)
                    model.set_bn_in_train_mode()
                    # not carrying states forward from one-shot learning

                    outputs = outputs[-1]
                    outputs = outputs.reshape(-1, num_classes)

                    loss_fs_a = loss_fn(
                        outputs, net_target_labels_dict[task_id])

                    with torch.no_grad():
                        _, predicted = outputs.max(-1)
                    bool_correct_pred = (
                        predicted == net_target_labels_dict[task_id])

                    # TODO dict
                    # dict_fs_running_correct_a[task_id] += bool_correct_pred.sum().item()


                model.set_bn_in_eval_mode()
                _, state = model(
                    train_input_dict[task_id], train_target_dict[task_id], state)
                state = model.clone_state(state)
                outputs, _ = model(
                    test_input_dict[task_id], dummy_last_token, state)
                model.set_bn_in_train_mode()

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                # TODO loss factor
                loss_list.append(
                    loss_fn(outputs, net_target_labels_dict[task_id]))

                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == net_target_labels_dict[task_id])

                if task_id == 4:
                    after_all_tasks_correct[task_id] += bool_correct_pred.sum().item()
                    after_all_tasks_total[task_id] += net_target_labels_dict[task_id].size(0)

                if task_id > 0:  # ACL part, evaluate forgetting
                    for prev_task_id in range(0, task_id):
                        net_input = test_input_dict[prev_task_id]
                        target_labels = test_target_dict[prev_task_id]

                        model.set_bn_in_eval_mode()
                        outputs, _ = model(net_input, dummy_last_token, state)
                        model.set_bn_in_train_mode()

                        outputs = outputs[-1]
                        outputs = outputs.reshape(-1, num_classes)
                        target_labels = target_labels[-1].reshape(-1)

                        acl_loss_list.append(
                            loss_fn(outputs, target_labels))

                        with torch.no_grad():
                            _, predicted = outputs.max(-1)
                        bool_correct_pred = (predicted == target_labels)

                        if task_id == 4:
                           after_all_tasks_correct[prev_task_id] += bool_correct_pred.sum().item()
                           after_all_tasks_total[prev_task_id] += target_labels.size(0)

            # compute loss
            # TODO more scaling? for now just mean
            loss = torch.stack(loss_list + acl_loss_list, dim=0).mean()
            loss.backward()

            if i % args.grad_cummulate == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                model.reset_grad()
            model.eval()

            run_step += 1
            if i % args.report_every == 0:
                after_all_tasks_acc = []
                log_msg = {}
                for task_id in range(5):
                    acc = 100 * zero_div(
                        after_all_tasks_correct[task_id],
                        after_all_tasks_total[task_id])
                    log_msg[f'final_train_acc_{task_id}'] = acc

                if use_wandb:
                    wandb.log(log_msg)

                train_elapsed = time.time() - train_timer
                train_timer = time.time()
                num_images_per_sec = (
                    (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                    // train_elapsed)
                last_batch_logged = i

                loginf(
                    f'steps: {i + offset_step}, num_seq: {num_seq} (elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} images/s)')
                loginf(log_msg)

                running_total = 0
                run_step = 0

                for task_id in range(5):
                    after_all_tasks_correct[task_id] = 0
                    after_all_tasks_total[task_id] = 0

            # ======================================================================

            if i % args.validate_every == 0:  # run validation
                model.eval()
                with torch.no_grad():
                    num_extra_test = 5
                    bsz = args.batch_size

                    model.eval()

                    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

                    # Do single task eval first.
                    store_results_a = []
                    store_results_b = []
                    store_results_c = []
                    store_results_d = []
                    store_results_e = []

                    for run_id in range(num_extra_test):
                        # TASK A
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 0
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):
                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
                        store_results_a.append(external_acc)

                        # Task B
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 1
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
                        store_results_b.append(external_acc)

                        # Task C
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 2
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M45, M45] test acc: {external_acc:.2f} %')
                        store_results_c.append(external_acc)

                        # Task D
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 3
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M67, M67] test acc: {external_acc:.2f} %')
                        store_results_d.append(external_acc)


                        # Task E
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 4
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M89, M89] test acc: {external_acc:.2f} %')
                        store_results_e.append(external_acc)


                    mean_a = np.mean(store_results_a)
                    std_a = np.std(store_results_a)
                    mean_b = np.mean(store_results_b)
                    std_b = np.std(store_results_b)
                    mean_c = np.mean(store_results_c)
                    std_c = np.std(store_results_c)
                    mean_d = np.mean(store_results_d)
                    std_d = np.std(store_results_d)
                    mean_e = np.mean(store_results_e)
                    std_e = np.std(store_results_e)

                    loginf(
                        f'[== {num_extra_test} runs: M01, M01 ==] '
                        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
                        f'[== {num_extra_test} runs: M23, M23 ==] '
                        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
                        f'[== {num_extra_test} runs: M45, M45 ==] '
                        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
                        f'[== {num_extra_test} runs: M67, M67 ==] '
                        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
                        f'[== {num_extra_test} runs: M89, M89 ==] '
                        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
                        )

                    # ACL eval

                    store_results_a_a = []
                    store_results_ab_b = []
                    store_results_ab_a = []

                    store_results_abc_c = []
                    store_results_abc_a = []
                    store_results_abc_b = []

                    store_results_abcd_d = []
                    store_results_abcd_c = []
                    store_results_abcd_b = []
                    store_results_abcd_a = []

                    store_results_abcde_e = []
                    store_results_abcde_d = []
                    store_results_abcde_c = []
                    store_results_abcde_b = []
                    store_results_abcde_a = []

                    for run_id in range(num_extra_test):
                        # MNIST -> CIFAR-10, MNIST
                        ########## part 1
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 0
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
                        store_results_a_a.append(external_acc)

                        # MNIST -> F-MNIST, F-MNIST
                        ########## part 2, new data
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 1
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
                        store_results_ab_b.append(external_acc)

                        ########## part 2, ACL 1/1
                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
                        store_results_ab_a.append(external_acc)

                        ########## part 3, new data
                        # MNIST, CIFAR10 ->M59
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 2
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M45] test acc: {external_acc:.2f} %')
                        store_results_abc_c.append(external_acc)

                        ########## part 3, ACL 1/2

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M01] test acc: {external_acc:.2f} %')
                        store_results_abc_a.append(external_acc)

                        ########## part 3, ACL 2/2

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[1]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M23] test acc: {external_acc:.2f} %')
                        store_results_abc_b.append(external_acc)


                        ########## part 4, new data
                        # MNIST, CIFAR10, M59, F-MNIST
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 3
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M67] test acc: {external_acc:.2f} %')
                        store_results_abcd_d.append(external_acc)

                        ########## part 4, ACL 1/3

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M01] test acc: {external_acc:.2f} %')
                        store_results_abcd_a.append(external_acc)

                        ########## part 4, ACL 2/3

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[1]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M23] test acc: {external_acc:.2f} %')
                        store_results_abcd_b.append(external_acc)


                        ########## part 4, ACL 3/3

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[2]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M45] test acc: {external_acc:.2f} %')
                        store_results_abcd_c.append(external_acc)


                        ########## part 5, new data
                        # MNIST, CIFAR10, M59, F-MNIST, SVHN
                        avg_this_run = []
                        extra_running_correct = 0
                        total_test_samples = 0

                        extra_train_data = []
                        extra_train_labels = []

                        split_id = 4
                        for class_id in range(split_id * 2, split_id * 2 + 2):
                            indices = extra_dataset.targets == class_id
                            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
                            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

                        # class appears nth time only once all classes were seen for n-1 times for all n
                        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
                        extra_train_data = torch.stack(extra_train_data, dim=1)
                        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

                        extra_train_labels = torch.stack(extra_train_labels, dim=1)
                        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

                        with torch.no_grad():
                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            # forward to get state
                            _, context_state = model(self_train_input, self_train_labels, context_state)

                            for _, batch in enumerate(split_mnist_valid_loaders[split_id]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                                if model.extra_label:
                                    dummy_last_token = dummy_last_token + model.num_classes

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M89] test acc: {external_acc:.2f} %')
                        store_results_abcde_e.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 1/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[0]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M01] test acc: {external_acc:.2f} %')
                        store_results_abcde_a.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 2/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[1]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M23] test acc: {external_acc:.2f} %')
                        store_results_abcde_b.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 3/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[2]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M45] test acc: {external_acc:.2f} %')
                        store_results_abcde_c.append(external_acc)
                        avg_this_run.append(external_acc)

                        ########## part 5, ACL 4/4

                        extra_running_correct = 0
                        total_test_samples = 0

                        with torch.no_grad():
                            for _, batch in enumerate(split_mnist_valid_loaders[3]):

                                test_input = batch[0].to(device)
                                test_labels = batch[1].to(device)

                                bsz = test_labels.shape[0]

                                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                                outputs = outputs[-1]
                                outputs = outputs.reshape(-1, num_classes)
                                _, predicted = outputs.max(-1)

                                bool_correct_pred = (predicted == test_labels)
                                extra_running_correct += bool_correct_pred.sum().item()
                                total_test_samples += bsz

                        external_acc = 100 * extra_running_correct / total_test_samples
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M67] test acc: {external_acc:.2f} %')
                        store_results_abcde_d.append(external_acc)
                        avg_this_run.append(external_acc)
                        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89] Average acc: {np.mean(avg_this_run):.2f} %')


                    mean_a_a = np.mean(store_results_a_a)
                    std_a_a = np.std(store_results_a_a)

                    mean_ab_b = np.mean(store_results_ab_b)
                    std_ab_b = np.std(store_results_ab_b)
                    mean_ab_a = np.mean(store_results_ab_a)
                    std_ab_a = np.std(store_results_ab_a)

                    mean_abc_c = np.mean(store_results_abc_c)
                    std_abc_c = np.std(store_results_abc_c)
                    mean_abc_b = np.mean(store_results_abc_b)
                    std_abc_b = np.std(store_results_abc_b)
                    mean_abc_a = np.mean(store_results_abc_a)
                    std_abc_a = np.std(store_results_abc_a)

                    mean_abcd_d = np.mean(store_results_abcd_d)
                    std_abcd_d = np.std(store_results_abcd_d)
                    mean_abcd_c = np.mean(store_results_abcd_c)
                    std_abcd_c = np.std(store_results_abcd_c)
                    mean_abcd_b = np.mean(store_results_abcd_b)
                    std_abcd_b = np.std(store_results_abcd_b)
                    mean_abcd_a = np.mean(store_results_abcd_a)
                    std_abcd_a = np.std(store_results_abcd_a)

                    mean_abcde_d = np.mean(store_results_abcde_d)
                    std_abcde_d = np.std(store_results_abcde_d)
                    mean_abcde_c = np.mean(store_results_abcde_c)
                    std_abcde_c = np.std(store_results_abcde_c)
                    mean_abcde_b = np.mean(store_results_abcde_b)
                    std_abcde_b = np.std(store_results_abcde_b)
                    mean_abcde_a = np.mean(store_results_abcde_a)
                    std_abcde_a = np.std(store_results_abcde_a)
                    mean_abcde_e = np.mean(store_results_abcde_e)
                    std_abcde_e = np.std(store_results_abcde_e)

                    avg_acc_overall = []
                    for run_id in range(num_extra_test):
                        final_acc = []
                        final_acc.append(store_results_abcde_a[run_id])
                        final_acc.append(store_results_abcde_b[run_id])
                        final_acc.append(store_results_abcde_c[run_id])
                        final_acc.append(store_results_abcde_d[run_id])
                        final_acc.append(store_results_abcde_e[run_id])

                        task_acc = np.mean(final_acc)
                        avg_acc_overall.append(task_acc)

                    loginf(
                        f'[== {num_extra_test} runs: M01, M01 ==] '
                        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
                        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
                        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

                        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M45 ==] '
                        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M01 ==] '
                        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M23 ==] '
                        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M67 ==] '
                        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
                        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M23 ==] '
                        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
                        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'

                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M89 ==] '
                        f'mean: {mean_abcde_e:.2f}, std: {std_abcde_e:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M01 ==] '
                        f'mean: {mean_abcde_a:.2f}, std: {std_abcde_a:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M23 ==] '
                        f'mean: {mean_abcde_b:.2f}, std: {std_abcde_b:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M45 ==] '
                        f'mean: {mean_abcde_c:.2f}, std: {std_abcde_c:.2f} \n'
                        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M67 ==] '
                        f'mean: {mean_abcde_d:.2f}, std: {std_abcde_d:.2f} \n'
                        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
                        )
                after_all_tasks_acc = []
                log_msg = {}
                for task_id in range(5):
                    acc = 100 * zero_div(
                        after_all_tasks_correct[task_id],
                        after_all_tasks_total[task_id])
                    log_msg[f'final_train_acc_{task_id}'] = acc

                if use_wandb:
                    wandb.log({'valid_all_task_mean_acc': np.mean(avg_acc_overall),
                               'valid_task_0': mean_abcde_a,
                               'valid_task_1': mean_abcde_b,
                               'valid_task_2': mean_abcde_c,
                               'valid_task_3': mean_abcde_d,
                               'valid_task_4': mean_abcde_e,
                               })

                avg_v = np.mean(avg_acc_overall)
                if avg_v > best_val_first_shot_acc:
                    best_val_first_shot_acc = avg_v
                    best_step = i + offset_step
                    # Save the best model
                    loginf("The best model so far.")
                    if args.context_carry_over:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'state': state,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    else:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': avg_v}, best_model_path)
                    loginf("Saved.")

                if args.context_carry_over:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'state': state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': avg_v}, lastest_model_path)
                else:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': avg_v}, lastest_model_path)

                elapsed = time.time() - interval_start_time
                loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
                interval_start_time = time.time()
                train_timer = time.time()

            if cur_train_acc > args.train_acc_stop:
                loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
                end_training = True
                break
            if i + offset_step > args.total_train_steps:
                end_training = True
                loginf(f'reached {args.total_train_steps} steps')
                break
            if args.freeze_out_emb:
                if args.freeze_after_steps < i:
                    for param in model.out_layer.parameters():
                        param.requires_grad = False      
                    # loginf(f"Step {i}: freezing output embeddings")  
            if args.freeze_after:
                if args.freeze_after_steps < i:
                    # loginf(f"Step {i}: freezing conv stem")  
                    if model_name in ['srwm', 'deltanet']:
                        for param in model.conv_layers.parameters():
                            param.requires_grad = False
                    elif model_name in ['res12_srwm', 'res12_deltanet']:
                        for param in model.stem_resnet12.parameters():
                            param.requires_grad = False
        if end_training:
            break
        offset_step += i


### standard single task training.
elif not skip_train:
    for ep in range(args.total_epoch):
        loginf(f'epoch {ep} ====================')
        for i, batch in enumerate(dataloader):
            model.train()
            # state = None
            if args.context_carry_over_double:
                if i % 2 == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif args.context_carry_over_k > 1:
                if i % args.context_carry_over_k == 0:
                    state = None
                else:
                    state = model.clone_state(state, detach=True)
            elif not args.context_carry_over:
                state = None
            elif state is not None:
                state = model.clone_state(state, detach=True)

            if use_fs:
                fs_train_inputs, fs_train_targets = batch['first_shot']
                fs_train_inputs = fs_train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                fs_train_targets = fs_train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                fs_train_shape = fs_train_inputs.shape
                fs_train_inputs = fs_train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                fs_train_targets = fs_train_targets.transpose(0, 1)  # (len, B)
                _, fs_slen = fs_train_shape[0], fs_train_shape[1]

                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                train_targets = train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                train_shape = train_inputs.shape
                bsz, slen = train_shape[0], train_shape[1]

                num_seq += bsz

                train_inputs = train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                train_targets = train_targets.transpose(0, 1)  # (len, B)

                # same for test
                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=device)  # (B, test_len, 28 * 28)
                test_targets = test_targets.to(device=device)

                test_inputs = test_inputs.transpose(0, 1)  # (test_len, B, 28 * 28)
                test_targets = test_targets.transpose(0, 1)  # (test_len, B)

                # take only the fist element (randomized already)
                test_inputs = test_inputs[0].unsqueeze(0)
                test_targets = test_targets[0].unsqueeze(0)

                # do the first shot part
                net_input = torch.cat([fs_train_inputs, test_inputs], dim=0)
                target_labels = torch.cat([fs_train_targets, test_targets], dim=0)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == fs_slen + 1
                assert target_labels_shape[1] == bsz

                sync_labels = target_labels[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                outputs, state = model(net_input, label_feedback, state)
                state = model.clone_state(state)

                # outputs, _ = model(net_input, label_feedback)
                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)

                target_labels = target_labels[-1].reshape(-1)
                loss_fs = loss_fn(outputs, target_labels)
                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels)
                fs_running_correct += bool_correct_pred.sum().item()

                # do the main part
                net_input = torch.cat([train_inputs, test_inputs], dim=0)
                target_labels = torch.cat([train_targets, test_targets], dim=0)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == slen + 1
                assert target_labels_shape[1] == bsz

                sync_labels = target_labels[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                outputs, state = model(net_input, label_feedback, state)
                state = model.clone_state(state, detach=True)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)

                target_labels = target_labels[-1].reshape(-1)
                loss_main = loss_fn(outputs, target_labels)
                loss = (loss_fs + loss_main) * 0.5
                loss.backward()

                if i % args.grad_cummulate == 0:
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    model.reset_grad()

                # global loss
                running_loss += loss_main.item()
                fs_running_loss += loss_fs.item()
                running_total += target_labels.size(0)
                model.eval()
                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels)
                running_correct += bool_correct_pred.sum().item()

                run_step += 1
                if i % args.report_every == 0:

                    cur_train_acc = 100 * running_correct / running_total
                    if use_wandb:
                        wandb.log({
                            "train_loss": running_loss / run_step,
                            "train_loss_fs": fs_running_loss / run_step,
                            "running_acc": 100 * running_correct / running_total,
                            "running_acc_fs": 100 * fs_running_correct / running_total,
                        })

                    train_elapsed = time.time() - train_timer
                    train_timer = time.time()
                    num_images_per_sec = (
                        (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                        // train_elapsed)
                    last_batch_logged = i

                    # check accurary on the batch.
                    # writer.add_scalar("Loss/train", running_loss / run_step, i)
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss: {running_loss / run_step :.3f}, '
                        f'train_loss_fs: {fs_running_loss / run_step :.3f}, '
                        f'running_acc: {100 * running_correct / running_total:.2f} % '
                        f'running_acc_fs: {100 * fs_running_correct / running_total:.2f} % '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')
                    running_loss = 0.0
                    fs_running_loss = 0.0
                    running_total = 0
                    running_correct = 0
                    fs_running_correct = 0
                    run_step = 0
            else:
                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=device)  # (B, len, 1, 28, 28)
                train_targets = train_targets.to(device=device)  # (B, len)

                # shuffle and reshape
                train_shape = train_inputs.shape
                bsz, slen = train_shape[0], train_shape[1]

                num_seq += bsz

                train_inputs = train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
                train_targets = train_targets.transpose(0, 1)  # (len, B)

                # same for test
                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=device)  # (B, test_len, 28 * 28)
                test_targets = test_targets.to(device=device)

                test_inputs = test_inputs.transpose(0, 1)  # (test_len, B, 28 * 28)
                test_targets = test_targets.transpose(0, 1)  # (test_len, B)

                # take only the fist element (randomized already)
                test_inputs = test_inputs[0].unsqueeze(0)
                test_targets = test_targets[0].unsqueeze(0)

                net_input = torch.cat([train_inputs, test_inputs], dim=0)
                target_labels = torch.cat([train_targets, test_targets], dim=0)

                target_labels_shape = target_labels.shape
                assert target_labels_shape[0] == slen + 1
                assert target_labels_shape[1] == bsz

                sync_labels = target_labels[:-1]
                # does not matter which label to feed for the last position.
                dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes
                label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                if args.context_carry_over:
                    outputs, state = model(net_input, label_feedback, state)
                    state = model.clone_state_drop(state, drop2d_layer)
                elif args.context_carry_over_double:
                    if i % 2 == 1:
                        outputs, state = model(net_input, label_feedback, state)
                        state = model.clone_state_drop(state, drop2d_layer)
                    else:
                        outputs, state = model(net_input, label_feedback)
                        state = model.clone_state_drop(state, drop2d_layer)
                elif args.context_carry_over_k > 1:
                    if i % args.context_carry_over_k == 0:
                        outputs, state = model(net_input, label_feedback)
                        state = model.clone_state_drop(state, drop2d_layer)
                    else:
                        outputs, state = model(net_input, label_feedback, state)
                        state = model.clone_state_drop(state, drop2d_layer)
                else:
                    outputs, _ = model(net_input, label_feedback)
                # outputs, _ = model(net_input, label_feedback)
                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)

                target_labels = target_labels[-1].reshape(-1)
                loss = loss_fn(outputs, target_labels)
                loss.backward()

                if i % args.grad_cummulate == 0:
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    model.reset_grad()

                # global loss
                running_loss += loss.item()
                running_total += target_labels.size(0)
                model.eval()
                with torch.no_grad():
                    _, predicted = outputs.max(-1)
                bool_correct_pred = (predicted == target_labels)
                running_correct += bool_correct_pred.sum().item()

                run_step += 1
                if i % args.report_every == 0:

                    cur_train_acc = 100 * running_correct / running_total
                    if use_wandb:
                        wandb.log({
                            "train_loss": running_loss / run_step,
                            "running_acc": 100 * running_correct / running_total,
                        })

                    train_elapsed = time.time() - train_timer
                    train_timer = time.time()
                    num_images_per_sec = (
                        (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                        // train_elapsed)
                    last_batch_logged = i

                    # check accurary on the batch.
                    # writer.add_scalar("Loss/train", running_loss / run_step, i)
                    loginf(f'steps: {i + offset_step}, num_seq: {num_seq}, '
                        f'train_loss: {running_loss / run_step :.3f}, '
                        f'running_acc: {100 * running_correct / running_total:.2f} % '
                        f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                        'images/s)')
                    running_loss = 0.0
                    running_total = 0
                    running_correct = 0
                    run_step = 0

            # ======================================================================

            if i % args.validate_every == 0:  # run validation
                model.eval()

                if i == 3:
                    import sys; sys.exit(0)

                with torch.no_grad():
                    v_total = eval_model_label_sync(
                        model, val_dataloader, num_steps=args.valid_size)
                    test_total = eval_model_label_sync(
                        model, test_dataloader, num_steps=args.test_size)

                loginf(
                    f"[val {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
                    f'val total {100 * v_total :.2f} %, ')

                loginf(f'test acc {100 * test_total :.2f} % ')  # debugging

                if use_wandb:
                    wandb.log({
                        "val_acc": 100 * v_total,
                        "test_acc": 100 * test_total,  # debugging
                    })

                if v_total > best_val_first_shot_acc:
                    best_val_first_shot_acc = v_total
                    best_step = i + offset_step
                    # Save the best model
                    loginf("The best model so far.")
                    if args.context_carry_over:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'state': state,
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': v_total}, best_model_path)
                    else:
                        torch.save({'epoch': best_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'valid_acc': v_total}, best_model_path)
                    loginf("Saved.")
                    if test_total > best_valid_test_first_shot_acc:
                        best_valid_test_first_shot_acc = test_total
                if test_total > best_test_first_shot_acc:
                    best_test_first_shot_acc = test_total
                loginf(
                    f'current best valid_acc {100 * best_val_first_shot_acc :.2f} '
                    f'%\ncurrent best valid test_acc '
                    f'{100 * best_valid_test_first_shot_acc :.2f} %\n'
                    f'current best test_acc {100 * best_test_first_shot_acc :.2f} ')
                # Save the latest model
                if args.context_carry_over:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'state': state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': v_total}, lastest_model_path)
                else:
                    torch.save({'train_step': i + offset_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'valid_total_acc': v_total}, lastest_model_path)

                if args.ood_eval:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                            # double shot
                            self_train_input = extra_train_data_part2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels_part2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct_doubleshot += bool_correct_pred.sum().item()

                    external_acc = 100 * extra_running_correct / total_test_samples
                    external_acc_doubleshot = 100 * extra_running_correct_doubleshot / total_test_samples
                    loginf(f'Extra test acc: {external_acc:.2f} %')
                    loginf(f'Extra test double shot acc: {external_acc_doubleshot:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_acc": external_acc,
                            "extra_double_acc": external_acc_doubleshot,
                        })
                    if best_external_acc < external_acc:
                        best_external_acc = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'ext_acc': best_external_acc}, best_ext_model_path)	

                if args.ood_eval2:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader2):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'CIFAR10 test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_cifar10_acc": external_acc,
                        })
                    if best_external_acc2 < external_acc:
                        best_external_acc2 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc2}, best_ext2_model_path)	

                if args.ood_eval3:
                    extra_running_correct = 0
                    extra_running_correct_doubleshot = 0
                    total_test_samples = 0

                    model.eval()
                    with torch.no_grad():
                        for _, batch in enumerate(extra_test_loader3):

                            test_input = batch[0].to(device)
                            test_labels = batch[1].to(device)

                            bsz = test_labels.shape[0]

                            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
                            self_train_input = extra_train_data3.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
                            self_train_labels = extra_train_labels3.repeat(bsz, 1).transpose(0, 1)  # len, bsz

                            net_input = torch.cat([self_train_input, test_input.unsqueeze(0)], dim=0)
                            
                            sync_labels = self_train_labels
                            # does not matter which label to feed for the last position.
                            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
                            if model.extra_label:
                                dummy_last_token = dummy_last_token + model.num_classes
                            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

                            if args.context_carry_over:
                                outputs, _ = model(net_input, label_feedback, state)
                            else:
                                outputs, _ = model(net_input, label_feedback)
                            outputs = outputs[-1]
                            outputs = outputs.reshape(-1, num_classes)
                            _, predicted = outputs.max(-1)

                            bool_correct_pred = (predicted == test_labels)
                            extra_running_correct += bool_correct_pred.sum().item()
                            total_test_samples += bsz

                    external_acc = 100 * extra_running_correct / total_test_samples
                    loginf(f'SVHN test acc: {external_acc:.2f} %')
                    if use_wandb:
                        wandb.log({
                            "extra_svhn_acc": external_acc,
                        })
                    if best_external_acc3 < external_acc:
                        best_external_acc3 = external_acc
                        if args.context_carry_over:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'state': state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)
                        else:
                            torch.save({'epoch': best_step,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'valid_acc': best_external_acc3}, best_ext3_model_path)	

                elapsed = time.time() - interval_start_time
                loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
                interval_start_time = time.time()
                train_timer = time.time()

            if cur_train_acc > args.train_acc_stop:
                loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
                end_training = True
                break
            if i + offset_step > args.total_train_steps:
                end_training = True
                loginf(f'reached {args.total_train_steps} steps')
                break
            if args.freeze_out_emb:
                if args.freeze_after_steps < i:
                    for param in model.out_layer.parameters():
                        param.requires_grad = False      
                    # loginf(f"Step {i}: freezing output embeddings")  
            if args.freeze_after:
                if args.freeze_after_steps < i:
                    # loginf(f"Step {i}: freezing conv stem")  
                    if model_name in ['srwm', 'deltanet']:
                        for param in model.conv_layers.parameters():
                            param.requires_grad = False
                    elif model_name in ['res12_srwm', 'res12_deltanet']:
                        for param in model.stem_resnet12.parameters():
                            param.requires_grad = False
        if end_training:
            break
        offset_step += i

if not skip_train:
    elapsed = time.time() - start_time
    loginf(f"Finished {i} steps in {elapsed / 60.:.2f} min.")
    loginf(f"Best one shot validation acc: {100 * best_val_first_shot_acc:.2f} % "
        f"at step {best_step}")


### Eval
### NB: the current eval code is largely redundant/repetitive; to be refactored.

if args.use_ab or args.old_use_ab:
    # eval best model
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                get_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                get_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    extra_running_correct = 0
    total_test_samples = 0

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST -> CIFAR-10, CIFAR 10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, state)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    loginf(f"=== CIFAR-10 -> MNIST ===")

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[CIFAR-10 -> MNIST, CIFAR10] test acc: {external_acc:.2f} %')

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, state)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')

    # =======================================
    #  eval also EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' EXTRA checkpoint {best_ext_model_path}")
    checkpoint = torch.load(best_ext_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Extra ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Extra ckpt, A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Extra ckpt, A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Extra ckpt, A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Extra ckpt, Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Extra ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Extra ckpt, B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Extra ckpt, B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Extra ckpt, B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Extra ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    extra_running_correct = 0
    total_test_samples = 0

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST -> CIFAR-10, CIFAR 10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, state)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    loginf(f"=== Extra ckpt, CIFAR-10 -> MNIST ===")

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, CIFAR-10 -> MNIST, CIFAR10] test acc: {external_acc:.2f} %')

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, state)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')


    # =======================================
    #  eval also last checkpoint
    # =======================================
    loginf(f"Evaluating the last checkpoint {lastest_model_path}")
    checkpoint = torch.load(lastest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Last ckpt, Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Last ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Last ckpt, A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Last ckpt, A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Last ckpt, A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Last ckpt, Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Last ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Last ckpt, B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Last ckpt, B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Last ckpt, B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Last ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Last ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    extra_running_correct = 0
    total_test_samples = 0

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST -> CIFAR-10, CIFAR 10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, state)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    loginf(f"=== Last ckpt, CIFAR-10 -> MNIST ===")

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, CIFAR-10 -> MNIST, CIFAR10] test acc: {external_acc:.2f} %')

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, state)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')

    loginf(f'== END ==')


elif args.eval_extra_only:
    # eval best model
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"MNIST -> F-MNINST")
    # EVAL OOD MNIST -> CIFAR-10 -> F-MNINST
    # MNIST -> CIFAR-10, MNIST

    # mnist
    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # cifar10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # Fashion
    extra_test_loader5 = torch.utils.data.DataLoader(
        dataset=test_set5, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    bsz = args.batch_size
    num_extra_test = 5

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> F-MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> F-MNIST, F-MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> F-MNIST, MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f"=== F-MNIST -> MNIST ===")

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # F-MNIST, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST -> MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST -> MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: F-MNIST, F-MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST -> MNIST, MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST -> MNIST, F-MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    # =======================================
    #  eval also EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' EXTRA checkpoint {best_ext_model_path}")
    checkpoint = torch.load(best_ext_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Extra ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> F-MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> F-MNIST, F-MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> F-MNIST, MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f"=== F-MNIST -> MNIST ===")

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # F-MNIST, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST -> MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST -> MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: F-MNIST, F-MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST -> MNIST, MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST -> MNIST, F-MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    # =======================================
    #  eval also last checkpoint
    # =======================================
    loginf(f"Evaluating the last checkpoint {lastest_model_path}")
    checkpoint = torch.load(lastest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Last ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Last ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> F-MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> F-MNIST, F-MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> F-MNIST, MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f"=== F-MNIST -> MNIST ===")

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # F-MNIST, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST -> MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST -> MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: F-MNIST, F-MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST -> MNIST, MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST -> MNIST, F-MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f'== END ==')
    import sys; sys.exit(0)


elif args.eval_extra_4_tasks:
    # eval best model
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"MNIST -> F-MNINST")
    # EVAL OOD MNIST -> CIFAR-10 -> F-MNINST
    # MNIST -> CIFAR-10, MNIST

    # mnist
    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # cifar10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # Fashion
    extra_test_loader5 = torch.utils.data.DataLoader(
        dataset=test_set5, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # SVNH
    extra_test_loader3 = torch.utils.data.DataLoader(
        dataset=test_set3, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # test set
    test_set4 = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=mnist_transform, download=True)

    # restrict number of classes
    idx = test_set4.train_labels>num_classes-1
    test_set4.targets = test_set4.targets[idx] - model.num_classes
    test_set4.data = test_set4.data[idx]

    extra_test_loader4 = torch.utils.data.DataLoader(
        dataset=test_set4, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    bsz = args.batch_size
    num_extra_test = 5

    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id + num_classes
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train) - model.num_classes

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M59, M59] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)

        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset3.targets == class_id
            extra_train_data.append(extra_dataset3.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset3.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader3):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: SVHN, SVHN] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR10, CIFAR10 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M59, M59 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST, F-MNIST ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: SVHN, SVHN ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10, MNIST] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id + num_classes 
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train) - model.num_classes

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, M59] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, MNIST] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, M59] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10, CIFAR10 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10, MNIST ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, M59 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, MNIST ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, CIFAR10 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, F-MNIST ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, MNIST ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, CIFAR10 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, M59 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'
        )

    # =======================================
    #  eval also EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' EXTRA checkpoint {best_ext_model_path}")
    checkpoint = torch.load(best_ext_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'

    # Do single task eval first.
    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id + num_classes
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train) - model.num_classes

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M59, M59] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)


        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset3.targets == class_id
            extra_train_data.append(extra_dataset3.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset3.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader3):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: SVHN, SVHN] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR10, CIFAR10 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M59, M59 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST, F-MNIST ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: SVHN, SVHN ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10, MNIST] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id + num_classes 
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train) - model.num_classes

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, M59] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, MNIST] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, M59] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10, CIFAR10 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10, MNIST ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, M59 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, MNIST ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, CIFAR10 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, F-MNIST ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, MNIST ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, CIFAR10 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, M59 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'
        )

    # =======================================
    #  eval also last checkpoint
    # =======================================
    loginf(f"Evaluating the last checkpoint {lastest_model_path}")
    checkpoint = torch.load(lastest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Last ckpt, Eval on EXTRA datasets ===")

    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id + num_classes
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train) - model.num_classes

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M59, M59] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)


        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset3.targets == class_id
            extra_train_data.append(extra_dataset3.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset3.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader3):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: SVHN, SVHN] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR10, CIFAR10 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M59, M59 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: F-MNIST, F-MNIST ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: SVHN, SVHN ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10, MNIST] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id + num_classes 
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train) - model.num_classes

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, M59] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, MNIST] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset5.targets == class_id
            extra_train_data.append(extra_dataset5.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset5.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader5):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, F-MNIST] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader4):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR10 -> M59 -> F-MNIST, M59] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    mean_abcd_d = np.mean(store_results_abcd_d)
    std_abcd_d = np.std(store_results_abcd_d)
    mean_abcd_c = np.mean(store_results_abcd_c)
    std_abcd_c = np.std(store_results_abcd_c)
    mean_abcd_b = np.mean(store_results_abcd_b)
    std_abcd_b = np.std(store_results_abcd_b)
    mean_abcd_a = np.mean(store_results_abcd_a)
    std_abcd_a = np.std(store_results_abcd_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10, CIFAR10 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10, MNIST ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, M59 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, MNIST ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59, CIFAR10 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, F-MNIST ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, MNIST ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, CIFAR10 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR10 -> M59 -> F-MNIST, M59 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'
        )

    loginf(f'== END ==')
    import sys; sys.exit(0)


elif args.eval_splitmnist or args.train_splitmnist_style:
    num_extra_test = 10
    bsz = args.batch_size

    loginf("Preparing Split-MNIST...")
    norm_params = {'mean': [0.1307], 'std': [0.3081]}
    # norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.MNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_test_loaders = {}

    for split_id in range(5):  # 5 tasks
        # test set
        test_set = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, transform=mnist_transform, download=True)
        # restrict number of classes

        idx_0 = test_set.train_labels == (split_id * 2)
        idx_1 = test_set.train_labels == (split_id * 2+1)
        idx = torch.logical_or(idx_0, idx_1)
        test_set.targets = test_set.targets[idx] - split_id * 2
        test_set.data = test_set.data[idx]

        extra_test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_test_loaders[split_id] = extra_test_loader

    # =======================================
    #  eval EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

    # Do single task eval first.
    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M45, M45] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M67, M67] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)


        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M89, M89] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: M23, M23 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M45, M45 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: M67, M67 ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: M89, M89 ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    store_results_abcde_e = []
    store_results_abcde_d = []
    store_results_abcde_c = []
    store_results_abcde_b = []
    store_results_abcde_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M45] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M01] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M23] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M67] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M01] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M23] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M45] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)


        ########## part 5, new data
        # MNIST, CIFAR10, M59, F-MNIST, SVHN
        avg_this_run = []
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M89] test acc: {external_acc:.2f} %')
        store_results_abcde_e.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 1/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M01] test acc: {external_acc:.2f} %')
        store_results_abcde_a.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 2/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M23] test acc: {external_acc:.2f} %')
        store_results_abcde_b.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 3/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M45] test acc: {external_acc:.2f} %')
        store_results_abcde_c.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 4/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[3]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M67] test acc: {external_acc:.2f} %')
        store_results_abcde_d.append(external_acc)
        avg_this_run.append(external_acc)
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89] Average acc: {np.mean(avg_this_run):.2f} %')


    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    mean_abcd_d = np.mean(store_results_abcd_d)
    std_abcd_d = np.std(store_results_abcd_d)
    mean_abcd_c = np.mean(store_results_abcd_c)
    std_abcd_c = np.std(store_results_abcd_c)
    mean_abcd_b = np.mean(store_results_abcd_b)
    std_abcd_b = np.std(store_results_abcd_b)
    mean_abcd_a = np.mean(store_results_abcd_a)
    std_abcd_a = np.std(store_results_abcd_a)

    mean_abcde_d = np.mean(store_results_abcde_d)
    std_abcde_d = np.std(store_results_abcde_d)
    mean_abcde_c = np.mean(store_results_abcde_c)
    std_abcde_c = np.std(store_results_abcde_c)
    mean_abcde_b = np.mean(store_results_abcde_b)
    std_abcde_b = np.std(store_results_abcde_b)
    mean_abcde_a = np.mean(store_results_abcde_a)
    std_abcde_a = np.std(store_results_abcde_a)
    mean_abcde_e = np.mean(store_results_abcde_e)
    std_abcde_e = np.std(store_results_abcde_e)

    avg_acc_overall = []
    for run_id in range(num_extra_test):
        final_acc = []
        final_acc.append(store_results_abcde_a[run_id])
        final_acc.append(store_results_abcde_b[run_id])
        final_acc.append(store_results_abcde_c[run_id])
        final_acc.append(store_results_abcde_d[run_id])
        final_acc.append(store_results_abcde_e[run_id])

        task_acc = np.mean(final_acc)
        avg_acc_overall.append(task_acc)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M45 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M01 ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M23 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M67 ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M23 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M89 ==] '
        f'mean: {mean_abcde_e:.2f}, std: {std_abcde_e:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M01 ==] '
        f'mean: {mean_abcde_a:.2f}, std: {std_abcde_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M23 ==] '
        f'mean: {mean_abcde_b:.2f}, std: {std_abcde_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M45 ==] '
        f'mean: {mean_abcde_c:.2f}, std: {std_abcde_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M67 ==] '
        f'mean: {mean_abcde_d:.2f}, std: {std_abcde_d:.2f} \n'
        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
        )
    loginf(f'== END ==')
    import sys; sys.exit(0)


elif args.eval_splitmnist_incremental_class:
    num_extra_test = 10
    bsz = args.batch_size

    loginf("Preparing Split-MNIST...")
    norm_params = {'mean': [0.1307], 'std': [0.3081]}
    # norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.MNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_test_loaders = {}

    for split_id in range(5):  # 5 tasks
        # test set
        test_set = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, transform=mnist_transform, download=True)
        # restrict number of classes

        idx_0 = test_set.train_labels == (split_id * 2)  # train_labels and targets are the same here
        idx_1 = test_set.train_labels == (split_id * 2+1)
        idx = torch.logical_or(idx_0, idx_1)
        test_set.targets = test_set.targets[idx]
        test_set.data = test_set.data[idx]

        extra_test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_test_loaders[split_id] = extra_test_loader

    # =======================================
    #  eval EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

    # Do single task eval first.
    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M45, M45] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M67, M67] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)


        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M89, M89] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: M23, M23 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M45, M45 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: M67, M67 ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: M89, M89 ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    store_results_abcde_e = []
    store_results_abcde_d = []
    store_results_abcde_c = []
    store_results_abcde_b = []
    store_results_abcde_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)
                # print(test_labels)
                # print(predicted)
                # import sys; sys.exit(0)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M45] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M01] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M23] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M67] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M01] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M23] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M45] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)


        ########## part 5, new data
        # MNIST, CIFAR10, M59, F-MNIST, SVHN
        avg_this_run = []
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M89] test acc: {external_acc:.2f} %')
        store_results_abcde_e.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 1/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M01] test acc: {external_acc:.2f} %')
        store_results_abcde_a.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 2/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M23] test acc: {external_acc:.2f} %')
        store_results_abcde_b.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 3/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M45] test acc: {external_acc:.2f} %')
        store_results_abcde_c.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 4/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[3]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M67] test acc: {external_acc:.2f} %')
        store_results_abcde_d.append(external_acc)
        avg_this_run.append(external_acc)
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89] Average acc: {np.mean(avg_this_run):.2f} %')


    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    mean_abcd_d = np.mean(store_results_abcd_d)
    std_abcd_d = np.std(store_results_abcd_d)
    mean_abcd_c = np.mean(store_results_abcd_c)
    std_abcd_c = np.std(store_results_abcd_c)
    mean_abcd_b = np.mean(store_results_abcd_b)
    std_abcd_b = np.std(store_results_abcd_b)
    mean_abcd_a = np.mean(store_results_abcd_a)
    std_abcd_a = np.std(store_results_abcd_a)

    mean_abcde_d = np.mean(store_results_abcde_d)
    std_abcde_d = np.std(store_results_abcde_d)
    mean_abcde_c = np.mean(store_results_abcde_c)
    std_abcde_c = np.std(store_results_abcde_c)
    mean_abcde_b = np.mean(store_results_abcde_b)
    std_abcde_b = np.std(store_results_abcde_b)
    mean_abcde_a = np.mean(store_results_abcde_a)
    std_abcde_a = np.std(store_results_abcde_a)
    mean_abcde_e = np.mean(store_results_abcde_e)
    std_abcde_e = np.std(store_results_abcde_e)

    avg_acc_overall = []
    for run_id in range(num_extra_test):
        final_acc = []
        final_acc.append(store_results_abcde_a[run_id])
        final_acc.append(store_results_abcde_b[run_id])
        final_acc.append(store_results_abcde_c[run_id])
        final_acc.append(store_results_abcde_d[run_id])
        final_acc.append(store_results_abcde_e[run_id])

        task_acc = np.mean(final_acc)
        avg_acc_overall.append(task_acc)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M45 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M01 ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M23 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M67 ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M23 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M89 ==] '
        f'mean: {mean_abcde_e:.2f}, std: {std_abcde_e:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M01 ==] '
        f'mean: {mean_abcde_a:.2f}, std: {std_abcde_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M23 ==] '
        f'mean: {mean_abcde_b:.2f}, std: {std_abcde_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M45 ==] '
        f'mean: {mean_abcde_c:.2f}, std: {std_abcde_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M67 ==] '
        f'mean: {mean_abcde_d:.2f}, std: {std_abcde_d:.2f} \n'
        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
        )
    loginf(f'== END ==')
    import sys; sys.exit(0)


# this is incremental-class with 2 tasks
elif args.eval_splitmnist_incremental_class_2task:
    num_extra_test = 10
    bsz = args.batch_size

    loginf("Preparing Split-MNIST...")
    norm_params = {'mean': [0.1307], 'std': [0.3081]}
    # norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.MNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_test_loaders = {}

    for split_id in range(2):  # 5 tasks
        # test set
        test_set = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, transform=mnist_transform, download=True)
        # restrict number of classes

        idx_0 = test_set.train_labels == (split_id * 2)
        idx_1 = test_set.train_labels == (split_id * 2+1)
        idx = torch.logical_or(idx_0, idx_1)
        test_set.targets = test_set.targets[idx]
        test_set.data = test_set.data[idx]

        extra_test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_test_loaders[split_id] = extra_test_loader

    # =======================================
    #  eval EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

    # Do single task eval first.
    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:, :4]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:, :4]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: M23, M23 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:, :4]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:, :4]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:, :4]
                _, predicted = outputs.max(-1)
                # print(test_labels)
                # print(predicted)
                # import sys; sys.exit(0)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    avg_acc_overall = []
    for run_id in range(num_extra_test):
        final_acc = []
        final_acc.append(store_results_ab_a[run_id])
        final_acc.append(store_results_ab_b[run_id])

        task_acc = np.mean(final_acc)
        avg_acc_overall.append(task_acc)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
        )
    loginf(f'== END ==')
    import sys; sys.exit(0)


elif args.eval_splitfashion:
    num_extra_test = args.num_test
    bsz = args.batch_size

    loginf("Preparing Split-FashionMNIST...")
    norm_params = {'mean': [0.286], 'std': [0.353]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.FashionMNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_test_loaders = {}

    for split_id in range(5):  # 5 tasks
        # test set
        test_set = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, transform=mnist_transform, download=True)
        # restrict number of classes

        idx_0 = test_set.train_labels == (split_id * 2)
        idx_1 = test_set.train_labels == (split_id * 2+1)
        idx = torch.logical_or(idx_0, idx_1)
        test_set.targets = test_set.targets[idx] - split_id * 2
        test_set.data = test_set.data[idx]

        extra_test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_test_loaders[split_id] = extra_test_loader

    # =======================================
    #  eval EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

    # Do single task eval first.
    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M45, M45] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M67, M67] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)


        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M89, M89] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: M23, M23 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M45, M45 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: M67, M67 ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: M89, M89 ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    store_results_abcde_e = []
    store_results_abcde_d = []
    store_results_abcde_c = []
    store_results_abcde_b = []
    store_results_abcde_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M45] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M01] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M23] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M67] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M01] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M23] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M45] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)


        ########## part 5, new data
        # MNIST, CIFAR10, M59, F-MNIST, SVHN
        avg_this_run = []
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M89] test acc: {external_acc:.2f} %')
        store_results_abcde_e.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 1/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M01] test acc: {external_acc:.2f} %')
        store_results_abcde_a.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 2/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M23] test acc: {external_acc:.2f} %')
        store_results_abcde_b.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 3/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M45] test acc: {external_acc:.2f} %')
        store_results_abcde_c.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 4/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[3]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M67] test acc: {external_acc:.2f} %')
        store_results_abcde_d.append(external_acc)
        avg_this_run.append(external_acc)
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89] Average acc: {np.mean(avg_this_run):.2f} %')


    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    mean_abcd_d = np.mean(store_results_abcd_d)
    std_abcd_d = np.std(store_results_abcd_d)
    mean_abcd_c = np.mean(store_results_abcd_c)
    std_abcd_c = np.std(store_results_abcd_c)
    mean_abcd_b = np.mean(store_results_abcd_b)
    std_abcd_b = np.std(store_results_abcd_b)
    mean_abcd_a = np.mean(store_results_abcd_a)
    std_abcd_a = np.std(store_results_abcd_a)

    mean_abcde_d = np.mean(store_results_abcde_d)
    std_abcde_d = np.std(store_results_abcde_d)
    mean_abcde_c = np.mean(store_results_abcde_c)
    std_abcde_c = np.std(store_results_abcde_c)
    mean_abcde_b = np.mean(store_results_abcde_b)
    std_abcde_b = np.std(store_results_abcde_b)
    mean_abcde_a = np.mean(store_results_abcde_a)
    std_abcde_a = np.std(store_results_abcde_a)
    mean_abcde_e = np.mean(store_results_abcde_e)
    std_abcde_e = np.std(store_results_abcde_e)

    avg_acc_overall = []
    for run_id in range(num_extra_test):
        final_acc = []
        final_acc.append(store_results_abcde_a[run_id])
        final_acc.append(store_results_abcde_b[run_id])
        final_acc.append(store_results_abcde_c[run_id])
        final_acc.append(store_results_abcde_d[run_id])
        final_acc.append(store_results_abcde_e[run_id])

        task_acc = np.mean(final_acc)
        avg_acc_overall.append(task_acc)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M45 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M01 ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M23 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M67 ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M23 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M89 ==] '
        f'mean: {mean_abcde_e:.2f}, std: {std_abcde_e:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M01 ==] '
        f'mean: {mean_abcde_a:.2f}, std: {std_abcde_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M23 ==] '
        f'mean: {mean_abcde_b:.2f}, std: {std_abcde_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M45 ==] '
        f'mean: {mean_abcde_c:.2f}, std: {std_abcde_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M67 ==] '
        f'mean: {mean_abcde_d:.2f}, std: {std_abcde_d:.2f} \n'
        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
        )
    loginf(f'== END ==')
    import sys; sys.exit(0)


elif args.eval_splitcifar10:
    num_extra_test = args.num_test
    bsz = args.batch_size

    loginf("Preparing extra eval dataset 2 CIFAR10...")
    norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params)])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params)])
    else:
        assert 'omni' in args.name_dataset
        loginf("Transforming to Grayscale.")
        from torchvision.transforms import Grayscale
        compat_shape = [1, 28, 28]
        norm_params = {'mean': [0.5], 'std': [0.25]}
        mnist_transform = Compose(
            [Resize(28), Grayscale(num_output_channels=1), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.CIFAR10(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    split_mnist_test_loaders = {}

    for split_id in range(5):  # 5 tasks
        # test set
        test_set = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, transform=mnist_transform, download=True)

        tmp_targets = torch.ByteTensor(test_set.targets)
        idx_0 = tmp_targets == (split_id * 2)
        idx_1 = tmp_targets == (split_id * 2+1)
        idx = torch.logical_or(idx_0, idx_1)

        test_set.targets = (tmp_targets[idx] - split_id * 2).tolist() 
        test_set.data = test_set.data[idx]

        extra_test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_worker, drop_last=True)

        split_mnist_test_loaders[split_id] = extra_test_loader

    # =======================================
    #  eval EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")

    # Do single task eval first.
    # Do single task eval first.
    store_results_a = []
    store_results_b = []
    store_results_c = []
    store_results_d = []
    store_results_e = []

    for run_id in range(num_extra_test):
        # TASK A
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a.append(external_acc)

        # Task B
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M23, M23] test acc: {external_acc:.2f} %')
        store_results_b.append(external_acc)

        # Task C
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M45, M45] test acc: {external_acc:.2f} %')
        store_results_c.append(external_acc)

        # Task D
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M67, M67] test acc: {external_acc:.2f} %')
        store_results_d.append(external_acc)


        # Task E
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M89, M89] test acc: {external_acc:.2f} %')
        store_results_e.append(external_acc)


    mean_a = np.mean(store_results_a)
    std_a = np.std(store_results_a)
    mean_b = np.mean(store_results_b)
    std_b = np.std(store_results_b)
    mean_c = np.mean(store_results_c)
    std_c = np.std(store_results_c)
    mean_d = np.mean(store_results_d)
    std_d = np.std(store_results_d)
    mean_e = np.mean(store_results_e)
    std_e = np.std(store_results_e)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a:.2f}, std: {std_a:.2f} \n'
        f'[== {num_extra_test} runs: M23, M23 ==] '
        f'mean: {mean_b:.2f}, std: {std_b:.2f} \n'
        f'[== {num_extra_test} runs: M45, M45 ==] '
        f'mean: {mean_c:.2f}, std: {std_c:.2f} \n'
        f'[== {num_extra_test} runs: M67, M67 ==] '
        f'mean: {mean_d:.2f}, std: {std_d:.2f} \n'
        f'[== {num_extra_test} runs: M89, M89 ==] '
        f'mean: {mean_e:.2f}, std: {std_e:.2f} \n'
        )

    # ACL eval

    store_results_a_a = []
    store_results_ab_b = []
    store_results_ab_a = []

    store_results_abc_c = []
    store_results_abc_a = []
    store_results_abc_b = []

    store_results_abcd_d = []
    store_results_abcd_c = []
    store_results_abcd_b = []
    store_results_abcd_a = []

    store_results_abcde_e = []
    store_results_abcde_d = []
    store_results_abcde_c = []
    store_results_abcde_b = []
    store_results_abcde_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        ########## part 1
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 0
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01, M01] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> F-MNIST, F-MNIST
        ########## part 2, new data
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 1
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M23] test acc: {external_acc:.2f} %')
        store_results_ab_b.append(external_acc)

        ########## part 2, ACL 1/1
        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23, M01] test acc: {external_acc:.2f} %')
        store_results_ab_a.append(external_acc)

        ########## part 3, new data
        # MNIST, CIFAR10 ->M59
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 2
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M45] test acc: {external_acc:.2f} %')
        store_results_abc_c.append(external_acc)

        ########## part 3, ACL 1/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M01] test acc: {external_acc:.2f} %')
        store_results_abc_a.append(external_acc)

        ########## part 3, ACL 2/2

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45, M23] test acc: {external_acc:.2f} %')
        store_results_abc_b.append(external_acc)


        ########## part 4, new data
        # MNIST, CIFAR10, M59, F-MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 3
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M67] test acc: {external_acc:.2f} %')
        store_results_abcd_d.append(external_acc)

        ########## part 4, ACL 1/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M01] test acc: {external_acc:.2f} %')
        store_results_abcd_a.append(external_acc)

        ########## part 4, ACL 2/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M23] test acc: {external_acc:.2f} %')
        store_results_abcd_b.append(external_acc)


        ########## part 4, ACL 3/3

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67, M45] test acc: {external_acc:.2f} %')
        store_results_abcd_c.append(external_acc)


        ########## part 5, new data
        # MNIST, CIFAR10, M59, F-MNIST, SVHN
        avg_this_run = []
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        split_id = 4
        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(split_mnist_test_loaders[split_id]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M89] test acc: {external_acc:.2f} %')
        store_results_abcde_e.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 1/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[0]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M01] test acc: {external_acc:.2f} %')
        store_results_abcde_a.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 2/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[1]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M23] test acc: {external_acc:.2f} %')
        store_results_abcde_b.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 3/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[2]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M45] test acc: {external_acc:.2f} %')
        store_results_abcde_c.append(external_acc)
        avg_this_run.append(external_acc)

        ########## part 5, ACL 4/4

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(split_mnist_test_loaders[3]):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)[:,:2]
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89, M67] test acc: {external_acc:.2f} %')
        store_results_abcde_d.append(external_acc)
        avg_this_run.append(external_acc)
        loginf(f'[Run {run_id}: M01 -> M23 -> M45 -> M67 -> M89] Average acc: {np.mean(avg_this_run):.2f} %')


    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_ab_b = np.mean(store_results_ab_b)
    std_ab_b = np.std(store_results_ab_b)
    mean_ab_a = np.mean(store_results_ab_a)
    std_ab_a = np.std(store_results_ab_a)

    mean_abc_c = np.mean(store_results_abc_c)
    std_abc_c = np.std(store_results_abc_c)
    mean_abc_b = np.mean(store_results_abc_b)
    std_abc_b = np.std(store_results_abc_b)
    mean_abc_a = np.mean(store_results_abc_a)
    std_abc_a = np.std(store_results_abc_a)

    mean_abcd_d = np.mean(store_results_abcd_d)
    std_abcd_d = np.std(store_results_abcd_d)
    mean_abcd_c = np.mean(store_results_abcd_c)
    std_abcd_c = np.std(store_results_abcd_c)
    mean_abcd_b = np.mean(store_results_abcd_b)
    std_abcd_b = np.std(store_results_abcd_b)
    mean_abcd_a = np.mean(store_results_abcd_a)
    std_abcd_a = np.std(store_results_abcd_a)

    mean_abcde_d = np.mean(store_results_abcde_d)
    std_abcde_d = np.std(store_results_abcde_d)
    mean_abcde_c = np.mean(store_results_abcde_c)
    std_abcde_c = np.std(store_results_abcde_c)
    mean_abcde_b = np.mean(store_results_abcde_b)
    std_abcde_b = np.std(store_results_abcde_b)
    mean_abcde_a = np.mean(store_results_abcde_a)
    std_abcde_a = np.std(store_results_abcde_a)
    mean_abcde_e = np.mean(store_results_abcde_e)
    std_abcde_e = np.std(store_results_abcde_e)

    avg_acc_overall = []
    for run_id in range(num_extra_test):
        final_acc = []
        final_acc.append(store_results_abcde_a[run_id])
        final_acc.append(store_results_abcde_b[run_id])
        final_acc.append(store_results_abcde_c[run_id])
        final_acc.append(store_results_abcde_d[run_id])
        final_acc.append(store_results_abcde_e[run_id])

        task_acc = np.mean(final_acc)
        avg_acc_overall.append(task_acc)

    loginf(
        f'[== {num_extra_test} runs: M01, M01 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M23 ==] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23, M01 ==] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M45 ==] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M01 ==] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45, M23 ==] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M67 ==] '
        f'mean: {mean_abcd_d:.2f}, std: {std_abcd_d:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_a:.2f}, std: {std_abcd_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M23 ==] '
        f'mean: {mean_abcd_b:.2f}, std: {std_abcd_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67, M01 ==] '
        f'mean: {mean_abcd_c:.2f}, std: {std_abcd_c:.2f} \n'

        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M89 ==] '
        f'mean: {mean_abcde_e:.2f}, std: {std_abcde_e:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M01 ==] '
        f'mean: {mean_abcde_a:.2f}, std: {std_abcde_a:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M23 ==] '
        f'mean: {mean_abcde_b:.2f}, std: {std_abcde_b:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M45 ==] '
        f'mean: {mean_abcde_c:.2f}, std: {std_abcde_c:.2f} \n'
        f'[== {num_extra_test} runs: M01 -> M23 -> M45 -> M67 -> M89, M67 ==] '
        f'mean: {mean_abcde_d:.2f}, std: {std_abcde_d:.2f} \n'
        f'5-task mean: {np.mean(avg_acc_overall):.2f}, std: {np.std(avg_acc_overall):.2f} \n'
        )
    loginf(f'== END ==')
    import sys; sys.exit(0)


###############################################################################
elif args.use_ab_v2:

    # eval best model
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_total_ab_b, test_total_ab_a = eval_acl_ab_model_label_sync(
                model, test_dataloader_a, test_dataloader_b, num_steps=args.test_size)

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_total_ab_b, test_total_ab_a = eval_acl_ab_model_label_sync(
                model, test_dataloader_b, test_dataloader_a, num_steps=args.test_size)

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST

    # mnist
    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    # cifar10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    bsz = args.batch_size
    num_extra_test = 5

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR-10, CIFAR-10 ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR-10, MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f"=== CIFAR-10 -> MNIST ===")

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # CIFAR-10, CIFAR-10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10, CIFAR-10] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: CIFAR-10, CIFAR-10 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR-10 -> MNIST, MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR-10 -> MNIST, CIFAR-10 ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    # =======================================
    #  eval also EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' EXTRA checkpoint {best_ext_model_path}")
    checkpoint = torch.load(best_ext_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_total_ab_b, test_total_ab_a = eval_acl_ab_model_label_sync(
                model, test_dataloader_a, test_dataloader_b, num_steps=args.test_size)

        loginf(
            f"[Extra ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Extra ckpt, A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Extra ckpt, A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Extra ckpt, A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Extra ckpt, Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_total_ab_b, test_total_ab_a = eval_acl_ab_model_label_sync(
                model, test_dataloader_b, test_dataloader_a, num_steps=args.test_size)

        loginf(
            f"[Extra ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Extra ckpt, B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Extra ckpt, B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Extra ckpt, B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Extra ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR-10, CIFAR-10 ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR-10, MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f"=== CIFAR-10 -> MNIST ===")

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # CIFAR-10, CIFAR-10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10, CIFAR-10] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: CIFAR-10, CIFAR-10 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR-10 -> MNIST, MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR-10 -> MNIST, CIFAR-10 ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    # =======================================
    #  eval also last checkpoint
    # =======================================
    loginf(f"Evaluating the last checkpoint {lastest_model_path}")
    checkpoint = torch.load(lastest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Last ckpt, Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_total_ab_b, test_total_ab_a = eval_acl_ab_model_label_sync(
                model, test_dataloader_a, test_dataloader_b, num_steps=args.test_size)

        loginf(
            f"[Last ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Last ckpt, A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Last ckpt, A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Last ckpt, A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Last ckpt, Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_total_ab_b, test_total_ab_a = eval_acl_ab_model_label_sync(
                model, test_dataloader_b, test_dataloader_a, num_steps=args.test_size)

        loginf(
            f"[Last ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Last ckpt, B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Last ckpt, B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Last ckpt, B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Last ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Last ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # MNIST -> CIFAR-10, MNIST
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: MNIST, MNIST ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR-10, CIFAR-10 ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: MNIST -> CIFAR-10, MNIST ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f"=== CIFAR-10 -> MNIST ===")

    store_results_a_a = []
    store_results_a_b_b = []
    store_results_a_b_a = []

    for run_id in range(num_extra_test):
        # CIFAR-10, CIFAR-10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset2.targets == class_id
            extra_train_data.append(extra_dataset2.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset2.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            # self_train_input = train_data.repeat(1, bsz)  # len, bsz
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels)

            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10, CIFAR-10] test acc: {external_acc:.2f} %')
        store_results_a_a.append(external_acc)

        # MNIST -> CIFAR-10, CIFAR 10
        extra_running_correct = 0
        total_test_samples = 0

        extra_train_data = []
        extra_train_labels = []

        for class_id in range(num_classes):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        extra_train_data = extra_train_data.reshape(num_classes * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        extra_train_labels = extra_train_labels.reshape(num_classes * k_shot_train)

        with torch.no_grad():
            self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
            self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

            # forward to get state
            _, context_state = model(self_train_input, self_train_labels, context_state)

            for _, batch in enumerate(extra_test_loader):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
                if model.extra_label:
                    dummy_last_token = dummy_last_token + model.num_classes

                outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')
        store_results_a_b_b.append(external_acc)

        extra_running_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for _, batch in enumerate(extra_test_loader2):

                test_input = batch[0].to(device)
                test_labels = batch[1].to(device)

                bsz = test_labels.shape[0]

                outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

                outputs = outputs[-1]
                outputs = outputs.reshape(-1, num_classes)
                _, predicted = outputs.max(-1)

                bool_correct_pred = (predicted == test_labels)
                extra_running_correct += bool_correct_pred.sum().item()
                total_test_samples += bsz

        external_acc = 100 * extra_running_correct / total_test_samples
        loginf(f'[Run {run_id}: CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')
        store_results_a_b_a.append(external_acc)

    mean_a_a = np.mean(store_results_a_a)
    std_a_a = np.std(store_results_a_a)

    mean_a_b_b = np.mean(store_results_a_b_b)
    std_a_b_b = np.std(store_results_a_b_b)

    mean_a_b_a = np.mean(store_results_a_b_a)
    std_a_b_a = np.std(store_results_a_b_a)

    loginf(
        f'[== {num_extra_test} runs: CIFAR-10, CIFAR-10 ==] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR-10 -> MNIST, MNIST ==] '
        f'mean: {mean_a_b_b:.2f}, std: {std_a_b_b:.2f}, 95%-CI {1.96 * std_a_b_b / num_extra_test:.2f} \n'
        f'[== {num_extra_test} runs: CIFAR-10 -> MNIST, CIFAR-10 ==] '
        f'mean: {mean_a_b_a:.2f}, std: {std_a_b_a:.2f}, 95%-CI {1.96 * std_a_b_a / num_extra_test:.2f}'
        )

    loginf(f'== END ==')

elif args.use_abc_v2:

    # extra task for eval
    loginf("Preparing extra eval dataset MNIST...")
    norm_params = {'mean': [0.1307], 'std': [0.3081]}
    # norm_params = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    if 'imagenet' in args.name_dataset and not ('32' in args.name_dataset):
        compat_shape = [3, 84, 84]
        mnist_transform = Compose(
            [Resize(84), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    elif args.name_dataset in ['fc100', 'fc100_norm', 'miniimagenet_32_norm', 'miniimagenet_32_norm_cache', 'omniglot_32_norm']:
        compat_shape = [3, 32, 32]
        mnist_transform = Compose(
            [Resize(32), ToTensor(), Normalize(**norm_params), Lambda(lambda x: x.repeat(3, 1, 1))])
    else:
        assert 'omni' in args.name_dataset
        compat_shape = [1, 28, 28]
        mnist_transform = Compose(
            [Resize(28), ToTensor(), Normalize(**norm_params)])

    extra_dataset = torchvision.datasets.MNIST(
        download=True, root=args.data_dir, train=True)

    class TransformedDataset(object):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # Construct the self-training examples
    # these are fixed.
    extra_train_data4 = []
    extra_train_labels4 = []

    for class_id in range(num_classes):
        indices = extra_dataset.targets == class_id + num_classes  # shifted by num_classes.
        extra_train_data4.append(extra_dataset.data[indices][:k_shot_train].to(device))
        extra_train_labels4.append(extra_dataset.targets[indices][:k_shot_train].to(device))

    # class appears nth time only once all classes were seen for n-1 times for all n
    # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
    extra_train_data4 = torch.stack(extra_train_data4, dim=1)
    extra_train_data4 = extra_train_data4.reshape(num_classes * k_shot_train, *compat_shape)

    extra_train_labels4 = torch.stack(extra_train_labels4, dim=1)
    extra_train_labels4 = extra_train_labels4.reshape(num_classes * k_shot_train)

    # test set
    test_set4 = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=mnist_transform, download=True)

    # restrict number of classes
    idx = test_set4.train_labels>num_classes-1
    test_set4.targets = test_set4.targets[idx]
    test_set4.data = test_set4.data[idx]

    extra_test_loader4 = torch.utils.data.DataLoader(
        dataset=test_set4, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker)
    loginf("done.")
    # data ready

    # eval best model
    loginf(f"Evaluating the 'best' checkpoint {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Eval A -> B -> C ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []
    results_abc_a = []
    results_abc_b = []
    results_abc_c = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

            test_total_abc_c, test_state = eval_model_label_sync(
                model, test_dataloader_c, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_abc_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

            test_total_abc_b = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %, '
            f'test acc A -> B -> C: C {100 * test_total_abc_c :.2f} %, '
            f'test acc A -> B -> C: A {100 * test_total_abc_a :.2f} %, '
            f'test acc A -> B -> C: B {100 * test_total_abc_b :.2f} %, '
            )

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

        results_abc_a.append(100 * test_total_abc_a)
        results_abc_b.append(100 * test_total_abc_b)
        results_abc_c.append(100 * test_total_abc_c)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    mean_abc_a = np.mean(results_abc_a)
    std_abc_a = np.std(results_abc_a)

    mean_abc_b = np.mean(results_abc_b)
    std_abc_b = np.std(results_abc_b)

    mean_abc_c = np.mean(results_abc_c)
    std_abc_c = np.std(results_abc_c)

    loginf(
        f'[A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    loginf(
        f'[A -> B -> C: C, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_abc_c:.2f}, std: {std_abc_c:.2f}, 95%-CI {1.96 * std_abc_c / num_test:.2f}')

    loginf(
        f'[A -> B -> C: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_abc_a:.2f}, std: {std_abc_a:.2f}, 95%-CI {1.96 * std_abc_a / num_test:.2f}')

    loginf(
        f'[A -> B -> C: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_abc_b:.2f}, std: {std_abc_b:.2f}, 95%-CI {1.96 * std_abc_b / num_test:.2f}')

    # TODO lazy mode, did not change below

    # EVAL other direction
    loginf(f"=== Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"MNIST-04 -> CIFAR-10 -> MNIST-59")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    extra_running_correct = 0
    total_test_samples = 0

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    bsz = args.batch_size
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST -> CIFAR-10, CIFAR 10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST-04 -> CIFAR-10 -> MNIST-59
    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data4.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels4.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader4):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST-04 -> CIFAR-10 -> MNIST-59, MNIST-59] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST-04 -> CIFAR-10 -> MNIST-59, MNIST-04] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[MNIST-04 -> CIFAR-10 -> MNIST-59, CIFAR-10] test acc: {external_acc:.2f} %')

    # TODO lazy mode, did not change below

    loginf(f"=== CIFAR-10 -> MNIST ===")

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[CIFAR-10 -> MNIST, CIFAR10] test acc: {external_acc:.2f} %')

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')

    # =======================================
    #  eval also EXTRA-validated checkpoint
    # =======================================
    loginf(f"Evaluating the 'best' EXTRA checkpoint {best_ext_model_path}")
    checkpoint = torch.load(best_ext_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Extra ckpt, Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Extra ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Extra ckpt, A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Extra ckpt, A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Extra ckpt, A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Extra ckpt, Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Extra ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Extra ckpt, B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Extra ckpt, B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Extra ckpt, B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Extra ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Extra ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    extra_running_correct = 0
    total_test_samples = 0

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST -> CIFAR-10, CIFAR 10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    loginf(f"=== Extra ckpt, CIFAR-10 -> MNIST ===")

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, CIFAR-10 -> MNIST, CIFAR10] test acc: {external_acc:.2f} %')

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Extra ckpt, CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')


    # =======================================
    #  eval also last checkpoint
    # =======================================
    loginf(f"Evaluating the last checkpoint {lastest_model_path}")
    checkpoint = torch.load(lastest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    loginf(f"=== Last ckpt, Eval A -> B ===")

    num_test = args.num_test
    test_size = 1000
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Last ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc A: A {100 * test_total_a_a :.2f} %, '
            f'test acc A -> B: B {100 * test_total_ab_b :.2f} %, '
            f'test acc A -> B: A {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Last ckpt, A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Last ckpt, A -> B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Last ckpt, A -> B: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')

    # EVAL other direction
    loginf(f"=== Last ckpt, Eval B -> A ===")
    results_a_a = []
    results_ab_b = []
    results_ab_a = []

    for i in range(num_test):

        with torch.no_grad():
            test_total_a_a, test_state = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                get_state=True, get_second_last_state=True)

            test_total_ab_b, test_state = eval_model_label_sync(
                model, test_dataloader_a, num_steps=args.test_size,
                state=test_state, get_state=True, get_second_last_state=True)

            test_total_ab_a = eval_model_label_sync(
                model, test_dataloader_b, num_steps=args.test_size,
                state=test_state, get_state=False, eval_no_context=True)  # TODO

        loginf(
            f"[Last ckpt, test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test acc B: B {100 * test_total_a_a :.2f} %, '
            f'test acc B -> A: A {100 * test_total_ab_b :.2f} %, '
            f'test acc B -> A: B {100 * test_total_ab_a :.2f} %')

        results_a_a.append(100 * test_total_a_a)
        results_ab_b.append(100 * test_total_ab_b)
        results_ab_a.append(100 * test_total_ab_a)

    mean_a_a = np.mean(results_a_a)
    std_a_a = np.std(results_a_a)

    mean_ab_b = np.mean(results_ab_b)
    std_ab_b = np.std(results_ab_b)

    mean_ab_a = np.mean(results_ab_a)
    std_ab_a = np.std(results_ab_a)

    loginf(
        f'[Last ckpt, B: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_a_a:.2f}, std: {std_a_a:.2f}, 95%-CI {1.96 * std_a_a / num_test:.2f}')

    loginf(
        f'[Last ckpt, B -> A: A, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_b:.2f}, std: {std_ab_b:.2f}, 95%-CI {1.96 * std_ab_b / num_test:.2f}')

    loginf(
        f'[Last ckpt, B -> A: B, {num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean_ab_a:.2f}, std: {std_ab_a:.2f}, 95%-CI {1.96 * std_ab_a / num_test:.2f}')
    
    loginf(f"=== Last ckpt, Eval on EXTRA datasets ===")
    assert args.ood_eval and args.ood_eval2, 'Turn on extra eval datasets.'
    loginf(f"Last ckpt, MNIST -> CIFAR-10")
    # EVAL OOD MNIST -> CIFAR-10
    # MNIST -> CIFAR-10, MNIST
    extra_running_correct = 0
    total_test_samples = 0

    extra_test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    # MNIST -> CIFAR-10, CIFAR 10
    extra_test_loader2 = torch.utils.data.DataLoader(
        dataset=test_set2, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_worker, drop_last=True)

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, MNIST -> CIFAR-10, CIFAR10] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, MNIST -> CIFAR-10, MNIST] test acc: {external_acc:.2f} %')

    loginf(f"=== Last ckpt, CIFAR-10 -> MNIST ===")

    extra_running_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data2.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels2.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels)

        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, CIFAR-10 -> MNIST, CIFAR10] test acc: {external_acc:.2f} %')

    with torch.no_grad():
        # self_train_input = train_data.repeat(1, bsz)  # len, bsz
        self_train_input = extra_train_data.repeat(bsz, 1, 1, 1, 1).transpose(0, 1)  # len, bsz
        self_train_labels = extra_train_labels.repeat(bsz, 1).transpose(0, 1)  # len, bsz

        # forward to get state
        _, context_state = model(self_train_input, self_train_labels, context_state)

        for _, batch in enumerate(extra_test_loader):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            dummy_last_token = torch.zeros_like(self_train_labels[0].unsqueeze(0))
            if model.extra_label:
                dummy_last_token = dummy_last_token + model.num_classes

            outputs, state = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, CIFAR-10 -> MNIST, MNIST] test acc: {external_acc:.2f} %')

    extra_running_correct = 0
    total_test_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(extra_test_loader2):

            test_input = batch[0].to(device)
            test_labels = batch[1].to(device)

            bsz = test_labels.shape[0]

            outputs, _ = model(test_input.unsqueeze(0), dummy_last_token, context_state)

            outputs = outputs[-1]
            outputs = outputs.reshape(-1, num_classes)
            _, predicted = outputs.max(-1)

            bool_correct_pred = (predicted == test_labels)
            extra_running_correct += bool_correct_pred.sum().item()
            total_test_samples += bsz

    external_acc = 100 * extra_running_correct / total_test_samples
    loginf(f'[Last ckpt, CIFAR-10 -> MNIST, CIFAR-10] test acc: {external_acc:.2f} %')

    loginf(f'== END ==')

else:
    # load the best model and evaluate on the test set
    del dataloader, dataset, val_dataloader, val_dataset

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    with torch.no_grad():
        test_total = eval_model_label_sync(
            model, test_dataloader, num_steps=args.test_size)

    loginf(
        f"[test {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
        f'test total {100 * test_total :.2f} %')

    # eval latest
    checkpoint = torch.load(lastest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    with torch.no_grad():
        test_total = eval_model_label_sync(
            model, test_dataloader, num_steps=args.test_size)

    loginf(
        f"[test latest {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
        f'test total {100 * test_total :.2f} %')

    # final eval
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    results = []

    num_test = args.num_test
    test_size = 1000

    for i in range(num_test):

        with torch.no_grad():
            test_total = eval_model_label_sync(
                model, test_dataloader, num_steps=args.test_size)

        test_total = 100 * test_total

        loginf(
            f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
            f'test total {test_total :.2f} %')

        results.append(test_total)

    mean = np.mean(results)
    std = np.std(results)

    loginf(
        f'[{num_test} tests using {batch_size * test_size} samples each] '
        f'mean: {mean:.2f}, std: {std:.2f}, 95%-CI {1.96 * std / num_test:.2f}')
