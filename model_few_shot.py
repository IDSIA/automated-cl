from operator import xor
import torch
import torch.nn as nn

from layer import FastFFlayer, TransformerFFlayers, SRWMlayer
from utils_other.resnet_impl import resnet12_base
from utils_other.resnet_dropblock_impl import resnet12_dropblock
from utils_other.mlpmixer_impl import MLPMixer, FeedForward, PreNormResidual
from einops.layers.torch import Rearrange, Reduce


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)

    # set batch norm in eval mode (preventing mean/var updates)
    def set_bn_in_eval_mode(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def set_bn_in_train_mode(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()


class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_classes, emb_dim=10, imagenet=False):
        super(LSTMModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, num_classes)

        self.rnn = nn.LSTM(hidden_size + emb_dim, hidden_size)
        # plus one for the dummy token
        self.fb_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, fb, state=None):
        # Assume linealized input: here `images.view(−1, 28*28)`.
        x = self.fc1(x)
        emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out, _ = self.rnn(out, state)
        out = self.out_layer(out)

        return out, None


# Conv4 by Vynials et al:
# '''
# We used a simple yet powerful CNN as the embedding function – consisting of a stack of modules,
# each of which is a 3 × 3 convolution with 64 filters followed by batch normalization [10], a Relu
# non-linearity and 2 × 2 max-pooling. We resized all the images to 28 × 28 so that, when we stack 4
# modules, the resulting feature map is 1 × 1 × 64, resulting in our embedding function f(x).
# '''
class ConvLSTMModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layer=1, imagenet=False,
                 fc100=False, vision_dropout=0.0, bn_momentum=0.1):
        super(ConvLSTMModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 2, 2)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []

        for i in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_block.append(nn.BatchNorm2d(
                out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        # self.fc1 = nn.Linear(conv_feature_final_size, hidden_size)

        self.rnn = nn.LSTM(self.conv_feature_final_size + num_classes,
                           hidden_size, num_layers=num_layer)
        # self.rnn = nn.LSTM(self.conv_feature_final_size + emb_dim, hidden_size)
        # plus one for the dummy token
        # self.fb_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)

        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out, _ = self.rnn(out, state)
        out = self.out_layer(out)

        return out, None


# Linear Transformer with the delta update rule.
class DeltaNetModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, emb_dim=10, imagenet=False):
        super(DeltaNetModel, self).__init__()
        assert num_head * dim_head == hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.feedback_emb = nn.Embedding(num_classes , emb_dim)
        # input projection takes both image and feedback
        self.input_proj = nn.Linear(hidden_size + emb_dim, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        x = self.fc1(x)
        emb = self.feedback_emb(fb)
        out = torch.cat([x, emb], dim=-1)
        out = self.input_proj(out)

        out = self.layers(out)
        out = self.out_layer(out)
        return out, state


class ConvDeltaModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dim_ff, dropout, vision_dropout=0.0, emb_dim=10,
                 imagenet=False, fc100=False, bn_momentum=0.1, use_pytorch=False):
        super(ConvDeltaModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []

        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_block.append(nn.BatchNorm2d(
                out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class StatefulConvDeltaModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dim_ff, dropout, vision_dropout=0.0, emb_dim=10,
                 imagenet=False, fc100=False, bn_momentum=0.1, use_pytorch=False,
                 single_state_training=False, extra_label=False,
                 remove_bn=False, use_instance_norm=False):
        super(StatefulConvDeltaModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.extra_label = extra_label
        if extra_label:
            self.num_input_classes = num_classes + 1
        else:
            self.num_input_classes = num_classes
        list_conv_layers = []

        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if use_instance_norm:
                conv_block.append(nn.InstanceNorm2d(
                    out_num_channel, affine=True))
            elif not remove_bn:
                conv_block.append(nn.BatchNorm2d(
                    out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + self.num_input_classes, hidden_size)

        layers = []
        assert not use_pytorch, 'not implemented.'
        self.num_layers = num_layers

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout, stateful=True,
                            single_state_training=single_state_training))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.ModuleList(layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def clone_state_drop(self, state, drop2d_layer):
        W_state = state

        B, H, D, _ = W_state[0].shape
        W_state_list = []

        for i in range(self.num_layers):
            W_state_list.append(drop2d_layer(W_state[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))

        W_state_tuple = tuple(W_state_list)

        return W_state_tuple

    def clone_state(self, state, detach=False):
        W_state = state

        B, H, D, _ = W_state[0].shape
        W_state_list = []

        for i in range(self.num_layers):
            if detach:
                W_state_list.append(W_state[i].detach().clone())
            else:
                W_state_list.append(W_state[i].clone())

        W_state_tuple = tuple(W_state_list)

        return W_state_tuple


    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_input_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out = self.input_proj(out)

        # forward main layers
        W_state_list = []
        if state is not None:
            W = state

        for i in range(self.num_layers):
            if state is not None:
                out, out_state = self.layers[2 * i](
                    out,
                    state=W[i],
                    get_state=True)
            else:
                out, out_state = self.layers[2 * i](
                    out,
                    get_state=True)
            W_state_list.append(out_state)
            out = self.layers[2 * i + 1](out)

        out = self.out_layer(out)

        W_state_tuple = tuple(W_state_list)

        return out, W_state_tuple


class ConvSRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, emb_dim=10, use_ln=True, use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False, bn_momentum=0.1, 
                 input_dropout=0.0, dropout_type='base', init_scaler=1., q_init_scaler=0.01,
                 unif_init=False, no_softmax_on_y=False, remove_bn=False,
                 use_instance_norm=False):
        super(ConvSRWMModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []

        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if use_instance_norm:
                conv_block.append(nn.InstanceNorm2d(
                    out_num_channel, affine=True))
            elif not remove_bn:
                conv_block.append(nn.BatchNorm2d(
                    out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if '2d' in dropout_type:
                conv_block.append(nn.Dropout2d(vision_dropout))
            else:
                conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        # self.input_layer_norm = nn.LayerNorm(self.conv_feature_final_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init, no_softmax_on_y=no_softmax_on_y))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)
        if dropout_type == 'base':
            self.input_drop = nn.Dropout(input_dropout)
        else:
            self.input_drop = nn.Dropout2d(input_dropout)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.input_drop(x)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        # TODO remove?
        # x = self.input_layer_norm(x)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class CompatStatefulConvSRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, emb_dim=10, use_ln=True,
                 use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False, bn_momentum=0.1, 
                 input_dropout=0.0, dropout_type='base',
                 init_scaler=1., q_init_scaler=0.01,
                 unif_init=False, single_state_training=False,
                 no_softmax_on_y=False, extra_label=False, remove_bn=False,
                 use_instance_norm=False):
        super().__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.extra_label = extra_label
        if extra_label:
            self.num_input_classes = num_classes + 1
        else:
            self.num_input_classes = num_classes
        list_conv_layers = []

        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if use_instance_norm:
                conv_block.append(nn.InstanceNorm2d(
                    out_num_channel, affine=True))
            elif not remove_bn:
                conv_block.append(nn.BatchNorm2d(
                    out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if '2d' in dropout_type:
                conv_block.append(nn.Dropout2d(vision_dropout))
            else:
                conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + self.num_input_classes, hidden_size)

        # self.input_layer_norm = nn.LayerNorm(self.conv_feature_final_size)

        layers = []
        self.num_layers = num_layers

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init, stateful=True,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init,
                          single_state_training=single_state_training,
                          no_softmax_on_y=no_softmax_on_y))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.layers = nn.ModuleList(layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)
        if dropout_type == 'base':
            self.input_drop = nn.Dropout(input_dropout)
        else:
            self.input_drop = nn.Dropout2d(input_dropout)

    # return clone of input state
    def clone_state(self, state, detach=False):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        for i in range(self.num_layers):
            if detach:
                Wy_state_list.append(Wy_states[i].detach().clone())
                Wq_state_list.append(Wq_states[i].detach().clone())
                Wk_state_list.append(Wk_states[i].detach().clone())
                wb_state_list.append(wb_states[i].detach().clone())
            else:
                Wy_state_list.append(Wy_states[i].clone())
                Wq_state_list.append(Wq_states[i].clone())
                Wk_state_list.append(Wk_states[i].clone())
                wb_state_list.append(wb_states[i].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    # return clone of input state, with drop certain batch
    def clone_state_drop(self, state, drop2d_layer):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        B, H, D, _ = Wy_states[0].shape

        for i in range(self.num_layers):
            Wy_state_list.append(drop2d_layer(Wy_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wq_state_list.append(drop2d_layer(Wq_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wk_state_list.append(drop2d_layer(Wk_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            wb_state_list.append(drop2d_layer(wb_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            # Wy_state_list.append(Wy_states[i].clone())
            # Wq_state_list.append(Wq_states[i].clone())
            # Wk_state_list.append(Wk_states[i].clone())
            # wb_state_list.append(wb_states[i].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.input_drop(x)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_input_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)

        # forward main layers
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        if state is not None:
            Wy_states, Wq_states, Wk_states, wb_states = state

        for i in range(self.num_layers):
            if state is not None:
                out, out_state = self.layers[2 * i](
                    out,
                    state=(Wy_states[i], Wq_states[i],
                           Wk_states[i], wb_states[i]),
                    get_state=True)
            else:
                out, out_state = self.layers[2 * i](
                    out,
                    get_state=True)
            # no cloning here. We do it outside where needed
            Wy_state_list.append(out_state[0])
            Wq_state_list.append(out_state[1])
            Wk_state_list.append(out_state[2])
            wb_state_list.append(out_state[3])

            # Wy_state_list.append(out_state[0].unsqueeze(0))
            # Wq_state_list.append(out_state[1].unsqueeze(0))
            # Wk_state_list.append(out_state[2].unsqueeze(0))
            # wb_state_list.append(out_state[3].unsqueeze(0))
            out = self.layers[2 * i + 1](out)

        out = self.out_layer(out)

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return out, state_tuple


class CompatStatefulRes12SRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, emb_dim=10, use_ln=True,
                 use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False, bn_momentum=0.1, 
                 input_dropout=0.0, dropout_type='base',
                 use_dropblock=False, use_big=False,
                 init_scaler=1., q_init_scaler=0.01,
                 unif_init=False, single_state_training=False,
                 no_softmax_on_y=False, extra_label=False, use_instance_norm=False):
        super().__init__()

        self.input_channels = 3
        self.num_classes = num_classes
        self.extra_label = extra_label
        if extra_label:
            self.num_input_classes = num_classes + 1
        else:
            self.num_input_classes = num_classes

        if use_dropblock:
            dropblock = 5 if imagenet else 2
            self.stem_resnet12 = resnet12_dropblock(
                use_big=use_big, drop_rate=vision_dropout,
                dropblock_size=dropblock)
            if use_big:
                self.conv_feature_final_size = 2560 if imagenet else 640
            else:
                self.conv_feature_final_size = 1024 if imagenet else 256
        else:
            self.stem_resnet12 = resnet12_base(
                vision_dropout, use_big, dropout_type,
                instance_norm=use_instance_norm)

            if use_big:
                self.conv_feature_final_size = 512
            else:
                self.conv_feature_final_size = 256

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + self.num_input_classes, hidden_size)

        # self.input_layer_norm = nn.LayerNorm(self.conv_feature_final_size)

        layers = []
        self.num_layers = num_layers

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init, stateful=True,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init,
                          single_state_training=single_state_training,
                          no_softmax_on_y=no_softmax_on_y))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.layers = nn.ModuleList(layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)
        if dropout_type == 'base':
            self.input_drop = nn.Dropout(input_dropout)
        else:
            self.input_drop = nn.Dropout2d(input_dropout)

    # return clone of input state
    def clone_state(self, state, detach=False):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        for i in range(self.num_layers):
            if detach:
                Wy_state_list.append(Wy_states[i].detach().clone())
                Wq_state_list.append(Wq_states[i].detach().clone())
                Wk_state_list.append(Wk_states[i].detach().clone())
                wb_state_list.append(wb_states[i].detach().clone())
            else:
                Wy_state_list.append(Wy_states[i].clone())
                Wq_state_list.append(Wq_states[i].clone())
                Wk_state_list.append(Wk_states[i].clone())
                wb_state_list.append(wb_states[i].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    # return clone of input state, with drop certain batch
    def clone_state_drop(self, state, drop2d_layer):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        B, H, D, _ = Wy_states[0].shape

        for i in range(self.num_layers):
            Wy_state_list.append(drop2d_layer(Wy_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wq_state_list.append(drop2d_layer(Wq_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wk_state_list.append(drop2d_layer(Wk_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            wb_state_list.append(drop2d_layer(wb_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            # Wy_state_list.append(Wy_states[i].clone())
            # Wq_state_list.append(Wq_states[i].clone())
            # Wk_state_list.append(Wk_states[i].clone())
            # wb_state_list.append(wb_states[i].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.input_drop(x)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_input_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)

        # forward main layers
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        if state is not None:
            Wy_states, Wq_states, Wk_states, wb_states = state

        for i in range(self.num_layers):
            if state is not None:
                out, out_state = self.layers[2 * i](
                    out,
                    state=(Wy_states[i], Wq_states[i],
                           Wk_states[i], wb_states[i]),
                    get_state=True)
            else:
                out, out_state = self.layers[2 * i](
                    out,
                    get_state=True)
            # no cloning here. We do it outside where needed
            Wy_state_list.append(out_state[0])
            Wq_state_list.append(out_state[1])
            Wk_state_list.append(out_state[2])
            wb_state_list.append(out_state[3])
            # Wy_state_list.append(out_state[0].unsqueeze(0))
            # Wq_state_list.append(out_state[1].unsqueeze(0))
            # Wk_state_list.append(out_state[2].unsqueeze(0))
            # wb_state_list.append(out_state[3].unsqueeze(0))
            out = self.layers[2 * i + 1](out)

        out = self.out_layer(out)

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return out, state_tuple


class CompatStatefulMixerSRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, emb_dim=10, use_ln=True,
                 use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False, bn_momentum=0.1, 
                 patch_size=16, expansion_factor = 4, expansion_factor_token = 0.5,
                 input_dropout=0.0, dropout_type='base',
                 init_scaler=1., q_init_scaler=0.01,
                 unif_init=False, single_state_training=False,
                 no_softmax_on_y=False, extra_label=False):
        super().__init__()

        self.num_classes = num_classes
        self.extra_label = extra_label
        if extra_label:
            self.num_input_classes = num_classes + 1
        else:
            self.num_input_classes = num_classes

        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            image_size = 84
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            image_size = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            image_size = 28
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels

        self.vision_model = MLPMixer(
            image_size = image_size,
            channels = input_channels,
            patch_size = patch_size,
            dim = 32,
            depth = 4,
            expansion_factor = expansion_factor,
            expansion_factor_token = expansion_factor_token,
            dropout = vision_dropout,
            num_classes = self.conv_feature_final_size  # just to make it similar to conv baseline
        )

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + self.num_input_classes, hidden_size)

        # self.input_layer_norm = nn.LayerNorm(self.conv_feature_final_size)

        layers = []
        self.num_layers = num_layers

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init, stateful=True,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init,
                          single_state_training=single_state_training,
                          no_softmax_on_y=no_softmax_on_y))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.layers = nn.ModuleList(layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)
        if dropout_type == 'base':
            self.input_drop = nn.Dropout(input_dropout)
        else:
            self.input_drop = nn.Dropout2d(input_dropout)

    # return clone of input state
    def clone_state(self, state, detach=False):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        for i in range(self.num_layers):
            if detach:
                Wy_state_list.append(Wy_states[i].detach().clone())
                Wq_state_list.append(Wq_states[i].detach().clone())
                Wk_state_list.append(Wk_states[i].detach().clone())
                wb_state_list.append(wb_states[i].detach().clone())
            else:
                Wy_state_list.append(Wy_states[i].clone())
                Wq_state_list.append(Wq_states[i].clone())
                Wk_state_list.append(Wk_states[i].clone())
                wb_state_list.append(wb_states[i].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    # return clone of input state, with drop certain batch
    def clone_state_drop(self, state, drop2d_layer):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        for i in range(self.num_layers):
            B, H, D, _ = Wy_states[0].shape
            Wy_state_list.append(drop2d_layer(Wy_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wq_state_list.append(drop2d_layer(Wq_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wk_state_list.append(drop2d_layer(Wk_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            wb_state_list.append(drop2d_layer(wb_states[i].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            # Wy_state_list.append(Wy_states[i].clone())
            # Wq_state_list.append(Wq_states[i].clone())
            # Wk_state_list.append(Wk_states[i].clone())
            # wb_state_list.append(wb_states[i].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.input_drop(x)

        x = self.vision_model(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_input_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)

        # forward main layers
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        if state is not None:
            Wy_states, Wq_states, Wk_states, wb_states = state

        for i in range(self.num_layers):
            if state is not None:
                out, out_state = self.layers[2 * i](
                    out,
                    state=(Wy_states[i], Wq_states[i],
                           Wk_states[i], wb_states[i]),
                    get_state=True)
            else:
                out, out_state = self.layers[2 * i](
                    out,
                    get_state=True)
            # no cloning here. We do it outside where needed
            Wy_state_list.append(out_state[0])
            Wq_state_list.append(out_state[1])
            Wk_state_list.append(out_state[2])
            wb_state_list.append(out_state[3])
            # Wy_state_list.append(out_state[0].unsqueeze(0))
            # Wq_state_list.append(out_state[1].unsqueeze(0))
            # Wk_state_list.append(out_state[2].unsqueeze(0))
            # wb_state_list.append(out_state[3].unsqueeze(0))
            out = self.layers[2 * i + 1](out)

        out = self.out_layer(out)

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return out, state_tuple


# MLP-mixer backend
class MixerSRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, emb_dim=10, use_ln=True,
                 use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False, bn_momentum=0.1, 
                 input_dropout=0.0, patch_size=16, expansion_factor = 4, expansion_factor_token = 0.5,
                 init_scaler=1., q_init_scaler=0.01, unif_init=False,
                 single_state_training=False, no_softmax_on_y=False):
        super(MixerSRWMModel, self).__init__()

        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            image_size = 84
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            image_size = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            image_size = 28
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.vision_model = MLPMixer(
            image_size = image_size,
            channels = input_channels,
            patch_size = patch_size,
            dim = 32,
            depth = 4,
            expansion_factor = expansion_factor,
            expansion_factor_token = expansion_factor_token,
            dropout = vision_dropout,
            num_classes = self.conv_feature_final_size  # just to make it similar to conv baseline
        )

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        # self.input_layer_norm = nn.LayerNorm(self.conv_feature_final_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init,
                          single_state_training=single_state_training,
                          no_softmax_on_y=no_softmax_on_y))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

        self.input_drop = nn.Dropout(input_dropout)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.input_drop(x)

        # for conv_layer in self.conv_layers:
        x = self.vision_model(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        # TODO remove?
        # x = self.input_layer_norm(x)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class CompatStatefulSelfModMixerModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, emb_dim=10, use_ln=True,
                 use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False, bn_momentum=0.1, 
                 patch_size=16, expansion_factor = 4, expansion_factor_token = 0.5,
                 input_dropout=0.0, dropout_type='base',
                 init_scaler=1., q_init_scaler=0.01,
                 unif_init=False, single_state_training=False,
                 no_softmax_on_y=False, extra_label=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.extra_label = extra_label
        if extra_label:
            self.num_input_classes = num_classes + 1
        else:
            self.num_input_classes = num_classes

        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            image_size = 84
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            image_size = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            image_size = 28
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels

        dim = hidden_size

        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, (
            'image must be divisible by patch size')
        num_patches = (image_h // patch_size) * (image_w // patch_size)

        self.patching = Rearrange(
                's b c (h p1) (w p2) -> s b (h w) (p1 p2 c)',
                p1 = patch_size, p2 = patch_size)

        self.input_proj = nn.Linear(
            (patch_size ** 2) * input_channels + self.num_input_classes, dim)

        tk_mixer_layers_list = []
        ch_mixer_layers_list = []

        tk_srwm_layers_list = []
        ch_srwm_layers_list = []

        self.num_layers = num_layers

        for _ in range(num_layers):
            # mixer layers
            tk_mixer_layers_list.append(
                PreNormResidual(
                    dim, FeedForward(
                        num_patches, expansion_factor, vision_dropout, nn.Linear), transpose=True))
            ch_mixer_layers_list.append(
                PreNormResidual(
                    dim,
                    FeedForward(
                        dim, expansion_factor_token, vision_dropout, nn.Linear)))
            tk_srwm_layers_list.append(
                    SRWMlayer(1, num_patches, num_patches,
                              dropout, use_ln, use_input_softmax, beta_init,
                              stateful=True,
                              init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                              unif_init=unif_init,
                              single_state_training=single_state_training,
                              no_softmax_on_y=no_softmax_on_y))

            ch_srwm_layers_list.append(
                SRWMlayer(num_head, dim_head, dim, dropout, use_ln,
                          use_input_softmax, beta_init,
                          stateful=True,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init,
                          single_state_training=single_state_training,
                          no_softmax_on_y=no_softmax_on_y))

        self.tk_mixer_layers = nn.ModuleList(tk_mixer_layers_list)
        self.ch_mixer_layers = nn.ModuleList(ch_mixer_layers_list)

        self.tk_srwm_layers = nn.ModuleList(tk_srwm_layers_list)
        self.ch_srwm_layers = nn.ModuleList(ch_srwm_layers_list)

        self.layer_norm = nn.LayerNorm(dim)
        self.mean_reduce = Reduce('b n c -> b c', 'mean')

        self.out_srwm_layer = SRWMlayer(
            num_head, dim_head, dim, dropout, use_ln, use_input_softmax,
            beta_init, stateful=True,
            init_scaler=init_scaler, q_init_scaler=q_init_scaler,
            unif_init=unif_init, single_state_training=single_state_training,
            no_softmax_on_y=no_softmax_on_y)

        self.out_proj = nn.Linear(dim, num_classes)

    # return clone of input state
    def clone_state(self, state, detach=False):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        if detach:
            for i in range(self.num_layers):
                id = 2 * i
                Wy_state_list.append(Wy_states[id].detach().clone())
                Wq_state_list.append(Wq_states[id].detach().clone())
                Wk_state_list.append(Wk_states[id].detach().clone())
                wb_state_list.append(wb_states[id].detach().clone())

                id = 2 * i + 1
                Wy_state_list.append(Wy_states[id].detach().clone())
                Wq_state_list.append(Wq_states[id].detach().clone())
                Wk_state_list.append(Wk_states[id].detach().clone())
                wb_state_list.append(wb_states[id].detach().clone())

            id = 2 * self.num_layers
            Wy_state_list.append(Wy_states[id].detach().clone())
            Wq_state_list.append(Wq_states[id].detach().clone())
            Wk_state_list.append(Wk_states[id].detach().clone())
            wb_state_list.append(wb_states[id].detach().clone())
        else:
            for i in range(self.num_layers):
                id = 2 * i
                Wy_state_list.append(Wy_states[id].clone())
                Wq_state_list.append(Wq_states[id].clone())
                Wk_state_list.append(Wk_states[id].clone())
                wb_state_list.append(wb_states[id].clone())

                id = 2 * i + 1
                Wy_state_list.append(Wy_states[id].clone())
                Wq_state_list.append(Wq_states[id].clone())
                Wk_state_list.append(Wk_states[id].clone())
                wb_state_list.append(wb_states[id].clone())

            id = 2 * self.num_layers
            Wy_state_list.append(Wy_states[id].clone())
            Wq_state_list.append(Wq_states[id].clone())
            Wk_state_list.append(Wk_states[id].clone())
            wb_state_list.append(wb_states[id].clone())

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    # return clone of input state, with drop certain batch
    def clone_state_drop(self, state, drop2d_layer):
        Wy_states, Wq_states, Wk_states, wb_states = state

        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        for i in range(self.num_layers):
            id = 2 * i
            B, H, D, _ = Wy_states[id].shape
            Wy_state_list.append(drop2d_layer(Wy_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wq_state_list.append(drop2d_layer(Wq_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wk_state_list.append(drop2d_layer(Wk_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            wb_state_list.append(drop2d_layer(wb_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))

            id = 2 * i + 1
            B, H, D, _ = Wy_states[id].shape
            Wy_state_list.append(drop2d_layer(Wy_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wq_state_list.append(drop2d_layer(Wq_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            Wk_state_list.append(drop2d_layer(Wk_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
            wb_state_list.append(drop2d_layer(wb_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))

        id = 2 * self.num_layers
        B, H, D, _ = Wy_states[id].shape
        Wy_state_list.append(drop2d_layer(Wy_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
        Wq_state_list.append(drop2d_layer(Wq_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
        Wk_state_list.append(drop2d_layer(Wk_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))
        wb_state_list.append(drop2d_layer(wb_states[id].clone().detach().reshape(B, 1, H*D, -1).transpose(0, 1)).transpose(0, 1).reshape(B, H, D, -1))

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return state_tuple

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = self.patching(x)  # (len, B, S=num_patch, C=ph*pw*c)
        _, _, num_patch, x_dim = x.shape

        # x = self.input_drop(x)

        emb = fb.unsqueeze(-1).expand(-1, -1, num_patch)
        emb = torch.nn.functional.one_hot(emb, num_classes=self.num_input_classes)
        x = torch.cat([x, emb], dim=-1)

        x = self.input_proj(x)  # (len, B, psz, dim)

        # forward main layers
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        if state is not None:
            Wy_states, Wq_states, Wk_states, wb_states = state

        for i in range(self.num_layers):
            x = x.transpose(-2, -1)
            x = x.reshape(slen, bsz * self.hidden_size, num_patch)

            if state is not None:
                x, out_state_tk = self.tk_srwm_layers[i](
                    x,
                    state=(Wy_states[2 * i], Wq_states[2 * i], Wk_states[2 * i], wb_states[2 * i]),
                    get_state=True)
            else:
                x, out_state_tk = self.tk_srwm_layers[i](
                    x,
                    get_state=True)
            # print(x.shape)
            x = x.reshape(slen * bsz, self.hidden_size, num_patch)
            # print(x.shape)

            x = self.tk_mixer_layers[i](x).transpose(1, 2)
            # x = x.reshape(slen, bsz, num_patch, self.hidden_size)
            x = x.reshape(slen, bsz * num_patch, self.hidden_size)

            if state is not None:
                x, out_state_ch = self.ch_srwm_layers[i](
                    x,
                    state=(Wy_states[2 * i + 1], Wq_states[2 * i + 1], Wk_states[2 * i + 1], wb_states[2 * i + 1]),
                    get_state=True)
            else:
                x, out_state_ch = self.ch_srwm_layers[i](
                    x,
                    get_state=True)
            x = x.reshape(slen * bsz, num_patch, self.hidden_size)

            Wy_state_list.append(out_state_tk[0])
            Wq_state_list.append(out_state_tk[1])
            Wk_state_list.append(out_state_tk[2])
            wb_state_list.append(out_state_tk[3])

            Wy_state_list.append(out_state_ch[0])
            Wq_state_list.append(out_state_ch[1])
            Wk_state_list.append(out_state_ch[2])
            wb_state_list.append(out_state_ch[3])

            x = self.ch_mixer_layers[i](x)

        x = self.mean_reduce(self.layer_norm(x))
        x = x.reshape(slen, bsz, -1)

        final_state_index = 2 * self.num_layers
        if state is not None:
            x, out_state = self.out_srwm_layer(
                x,
                state=(
                    Wy_states[final_state_index],
                    Wq_states[final_state_index],
                    Wk_states[final_state_index],
                    wb_states[final_state_index]),
                get_state=True)
        else:
            x, out_state = self.out_srwm_layer(
                x, get_state=True)

        # x = x.reshape(slen * bsz, -1)
        x = self.out_proj(x)

        Wy_state_list.append(out_state[0])
        Wq_state_list.append(out_state[1])
        Wk_state_list.append(out_state[2])
        wb_state_list.append(out_state[3])

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return x, state_tuple


# Self-modifying MLP-mixer
class SRMixerModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dropout, vision_dropout=0.0, use_ln=True,
                 use_input_softmax=False,
                 beta_init=0., imagenet=False, fc100=False,
                 patch_size=16, expansion_factor = 4, expansion_factor_token = 0.5,
                 init_scaler=1., q_init_scaler=0.01,
                 unif_init=False, single_state_training=False,
                 no_softmax_on_y=False):
        super().__init__()

        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            image_size = 84
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            image_size = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            image_size = 28
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        dim = hidden_size

        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, (
            'image must be divisible by patch size')
        num_patches = (image_h // patch_size) * (image_w // patch_size)

        self.patching = Rearrange(
                's b c (h p1) (w p2) -> s b (h w) (p1 p2 c)',
                p1 = patch_size, p2 = patch_size)

        self.input_proj = nn.Linear(
            (patch_size ** 2) * input_channels + num_classes, dim)

        tk_mixer_layers_list = []
        ch_mixer_layers_list = []

        tk_srwm_layers_list = []
        ch_srwm_layers_list = []

        self.num_layers = num_layers

        for _ in range(num_layers):
            # mixer layers
            tk_mixer_layers_list.append(
                PreNormResidual(
                    dim, FeedForward(
                        num_patches, expansion_factor, vision_dropout, nn.Linear), transpose=True))
            ch_mixer_layers_list.append(
                PreNormResidual(
                    dim,
                    FeedForward(
                        dim, expansion_factor_token, vision_dropout, nn.Linear)))
            # SRWM layers
            tk_srwm_layers_list.append(
                    SRWMlayer(1, num_patches, num_patches,
                              dropout, use_ln, use_input_softmax, beta_init,
                              init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                              unif_init=unif_init,
                              single_state_training=single_state_training,
                              no_softmax_on_y=no_softmax_on_y))

            ch_srwm_layers_list.append(
                SRWMlayer(num_head, dim_head, dim, dropout, use_ln,
                          use_input_softmax, beta_init,
                          init_scaler=init_scaler, q_init_scaler=q_init_scaler,
                          unif_init=unif_init,
                          single_state_training=single_state_training,
                          no_softmax_on_y=no_softmax_on_y))

        self.tk_mixer_layers = nn.ModuleList(tk_mixer_layers_list)
        self.ch_mixer_layers = nn.ModuleList(ch_mixer_layers_list)

        self.tk_srwm_layers = nn.ModuleList(tk_srwm_layers_list)
        self.ch_srwm_layers = nn.ModuleList(ch_srwm_layers_list)

        self.layer_norm = nn.LayerNorm(dim)
        self.mean_reduce = Reduce('b n c -> b c', 'mean')

        self.out_srwm_layer = SRWMlayer(
            num_head, dim_head, dim, dropout, use_ln, use_input_softmax,
            beta_init, init_scaler=init_scaler, q_init_scaler=q_init_scaler,
            unif_init=unif_init, single_state_training=single_state_training,
            no_softmax_on_y=no_softmax_on_y)

        self.out_proj = nn.Linear(dim, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        # x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.patching(x)  # (len, B, S=num_patch, C=ph*pw*c)
        _, _, num_patch, x_dim = x.shape
        # print(x.shape)
        emb = fb.unsqueeze(-1).expand(-1, -1, num_patch)
        emb = torch.nn.functional.one_hot(emb, num_classes=self.num_classes)
        x = torch.cat([x, emb], dim=-1)
        # x = x.reshape(slen * bsz, num_patch, x_dim + self.num_classes)
        # print(x.shape)
        x = self.input_proj(x)  # (len, B, psz, dim)
        for i in range(self.num_layers):
            x = x.transpose(-2, -1)
            x = x.reshape(slen, bsz * self.hidden_size, num_patch)

            x = self.tk_srwm_layers[i](x)
            # print(x.shape)
            x = x.reshape(slen * bsz, self.hidden_size, num_patch)
            # print(x.shape)

            x = self.tk_mixer_layers[i](x).transpose(1, 2)
            # x = x.reshape(slen, bsz, num_patch, self.hidden_size)
            x = x.reshape(slen, bsz * num_patch, self.hidden_size)

            x = self.ch_srwm_layers[i](x)
            x = x.reshape(slen * bsz, num_patch, self.hidden_size)

            x = self.ch_mixer_layers[i](x)
        x = self.mean_reduce(self.layer_norm(x))
        x = x.reshape(slen, bsz, -1)
        x = self.out_srwm_layer(x)
        # x = x.reshape(slen * bsz, -1)
        x = self.out_proj(x)
        return x, None


class Res12LSTMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, dropout, vision_dropout=0.0, use_big=False,
                 emb_dim=10, imagenet=False):
        super(Res12LSTMModel, self).__init__()

        self.stem_resnet12 = resnet12_base(vision_dropout, use_big)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256

        self.rnn = nn.LSTM(self.conv_feature_final_size + num_classes,
                           hidden_size, num_layers=num_layers,
                           dropout=dropout)
        # self.rnn = nn.LSTM(self.conv_feature_final_size + emb_dim, hidden_size)
        # plus one for the dummy token
        # self.fb_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out, _ = self.rnn(out, state)
        out = self.out_layer(out)

        return out, None


class Res12DeltaModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, use_big=False,
                 emb_dim=10, imagenet=False):
        super(Res12DeltaModel, self).__init__()

        self.stem_resnet12 = resnet12_base(vision_dropout, use_big)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class Res12SRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dim_ff, dropout, vision_dropout=0.0,
                 use_big=False, emb_dim=10, use_res=True,
                 use_ff=True,
                 use_ln=True, use_input_softmax=False, beta_init=-1.,
                 imagenet=False, input_dropout=0.0, dropout_type='base',
                 use_dropblock=False):
        super(Res12SRWMModel, self).__init__()
        if use_dropblock:
            dropblock = 5 if imagenet else 2
            self.stem_resnet12 = resnet12_dropblock(
                use_big=use_big, drop_rate=vision_dropout,
                dropblock_size=dropblock)
            if use_big:
                self.conv_feature_final_size = 2560 if imagenet else 640
            else:
                self.conv_feature_final_size = 1024 if imagenet else 256
        else:
            self.stem_resnet12 = resnet12_base(
                vision_dropout, use_big, dropout_type)

            if use_big:
                self.conv_feature_final_size = 512
            else:
                self.conv_feature_final_size = 256

        self.input_channels = 3
        self.num_classes = num_classes

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init, use_res=use_res))
            if use_ff:
                layers.append(
                    TransformerFFlayers(dim_ff, hidden_size, dropout,
                                        use_layernorm=use_ln, use_res=use_res))
        self.layers = nn.Sequential(*layers)

        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_size, num_classes)

        self.input_drop = nn.Dropout(input_dropout)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        # TODO remove?
        # x = self.input_layer_norm(x)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        # emb = self.fb_emb(fb)
        out = torch.cat([x, emb], dim=-1)

        # out = self.activation(out)  # or F.relu.
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None
