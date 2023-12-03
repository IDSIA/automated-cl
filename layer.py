import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_weight import fast_weight_delta, stateful_fast_weight_delta
from self_ref_v0 import self_ref_v0, stateful_self_ref_v0
from self_ref_v3 import self_ref_v3, stateful_self_ref_v3


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True,
                 use_res=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.use_res = use_res

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, res_dim),
            nn.Dropout(dropout),
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        if self.use_res:
            out = self.ff_layers(out) + x
        else:
             out = self.ff_layers(out)
        return out


# Fast weight layer with feed-forward fast net
class FastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, stateful=False,
                 single_state_training=False):
        super(FastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.stateful = stateful
        self.single_state_training = single_state_training
        if stateful:
            self.fw_layer = stateful_fast_weight_delta
        else:
            self.fw_layer = fast_weight_delta

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None, get_state=False):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        head_q = elu_p1(head_q)
        head_k = elu_p1(head_k)

        # normalize k and q, crucial for stable training.
        head_k = sum_norm(head_k)
        head_q = sum_norm(head_q)

        if state is None:
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)
        else:
            fast_weights = state

        if self.stateful:
            out, fast_weights = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        else:
            out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        if get_state:
            assert fast_weights is not None
            if self.single_state_training:
                fast_weights = fast_weights.detach()[0].clone().unsqueeze(0).repeat(bsz, 1, 1, 1)
                return out, fast_weights
            return out, fast_weights.detach().clone()
        else:
            return out


# self referential weight matrix layer
class SRWMlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, use_ln=True,
                 use_input_softmax=False, beta_init=-1.0, use_res=True, stateful=False,
                 init_scaler=1., q_init_scaler=0.01, unif_init=False,
                 single_state_training=False, no_softmax_on_y=False):
        super(SRWMlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.use_ln = use_ln
        self.use_res = use_res
        self.use_input_softmax = use_input_softmax
        self.no_softmax_on_y = no_softmax_on_y
        if no_softmax_on_y:
            assert use_input_softmax, '`no_softmax_on_y` is True but not `use_input_softmax`'

        self.stateful = stateful
        self.single_state_training = single_state_training
        if no_softmax_on_y:
            if stateful:
                self.sr_layer = stateful_self_ref_v3
            else:
                self.sr_layer = self_ref_v3
            self.y_lnorm = nn.LayerNorm(dim_head)
        else:
            if stateful:
                self.sr_layer = stateful_self_ref_v0
            else:
                self.sr_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head

        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4),
                                requires_grad=True)
        if use_ln:
            self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        if unif_init:
            self.reset_parameters_unif(init_scaler, q_init_scaler)
        else:
            self.reset_parameters(beta_init, init_scaler, q_init_scaler)

    def reset_parameters(self, beta_init, init_scaler, q_init_scaler=0.01):
        std = init_scaler / math.sqrt(self.dim_head)
        # std = 0.1 / math.sqrt(self.dim_head)
        std_q = q_init_scaler / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)
        # nn.init.normal_(self.W_q, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std_q)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        # nn.init.normal_(self.w_b, mean=-5., std=std)
        nn.init.normal_(self.w_b, mean=beta_init, std=std)

    def reset_parameters_unif(self, init_scaler, q_init_scaler=0.01):
        # beta_init not used
        nn.init.uniform_(self.W_y, a=-init_scaler, b=init_scaler)
        nn.init.uniform_(self.W_q, a=-q_init_scaler, b=q_init_scaler)
        nn.init.uniform_(self.W_k, a=-init_scaler, b=init_scaler)
        nn.init.uniform_(self.w_b, a=-init_scaler, b=init_scaler)

    def forward(self, h, state=None, get_state=False):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        if self.use_input_softmax:
            if self.no_softmax_on_y:
                x = F.softmax(x, dim=-1)
                input_to_y = x.clone()
            else:
                x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        if state is not None:  # state stores the shift from the base weights.
            W_y_bc, W_q_bc, W_k_bc, w_b_bc = state
            W_y_bc = W_y_bc + self.W_y.clone().repeat(bsz, 1, 1, 1)
            W_q_bc = W_q_bc + self.W_q.clone().repeat(bsz, 1, 1, 1)
            W_k_bc = W_k_bc + self.W_k.clone().repeat(bsz, 1, 1, 1)
            w_b_bc = w_b_bc + self.w_b.clone().repeat(bsz, 1, 1, 1)
        else:
            W_y_bc = self.W_y.clone().repeat(bsz, 1, 1, 1)
            W_q_bc = self.W_q.clone().repeat(bsz, 1, 1, 1)
            W_k_bc = self.W_k.clone().repeat(bsz, 1, 1, 1)
            w_b_bc = self.w_b.clone().repeat(bsz, 1, 1, 1)

        if self.no_softmax_on_y:
            if self.stateful:
                out, W_y_bc, W_q_bc, W_k_bc, w_b_bc = self.sr_layer(x, input_to_y, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
            else:
                out = self.sr_layer(x, input_to_y, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
            out = self.y_lnorm(out)
        else:
            if self.stateful:
                out, W_y_bc, W_q_bc, W_k_bc, w_b_bc = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
            else:
                out = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        if self.use_ln:
            if self.use_res:
                out = self.layer_norm(h) + out
        else:
            if self.use_res:
                out = h + out

        if get_state:
            if self.single_state_training:  # take only batch one.
                W_y_bc = (W_y_bc[0].unsqueeze(0) - self.W_y.detach().clone()).repeat(bsz, 1, 1, 1)
                W_q_bc = (W_q_bc[0].unsqueeze(0) - self.W_q.detach().clone()).repeat(bsz, 1, 1, 1)
                W_k_bc = (W_k_bc[0].unsqueeze(0) - self.W_k.detach().clone()).repeat(bsz, 1, 1, 1)
                w_b_bc = (w_b_bc[0].unsqueeze(0) - self.w_b.detach().clone()).repeat(bsz, 1, 1, 1)
            else:
                W_y_bc = W_y_bc - self.W_y.detach().clone().repeat(bsz, 1, 1, 1)
                W_q_bc = W_q_bc - self.W_q.detach().clone().repeat(bsz, 1, 1, 1)
                W_k_bc = W_k_bc - self.W_k.detach().clone().repeat(bsz, 1, 1, 1)
                w_b_bc = w_b_bc - self.w_b.detach().clone().repeat(bsz, 1, 1, 1)

            state = (W_y_bc, W_q_bc, W_k_bc, w_b_bc)

            return out, state

        return out
