# Fork from https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py

from os import X_OK
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, transpose=False):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.transpose = transpose

    def forward(self, x, transpose=False):
        if self.transpose:
            return self.fn(self.norm(x.transpose(-1, -2)).transpose(-1, -2)) + x
        else:
            return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes,
             expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, (
        f'image must be divisible by patch size; image_h: {image_h}, image_w, {image_w}, patch_size {patch_size}')
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(
                dim, FeedForward(
                    num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(
                dim,
                FeedForward(
                    dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


# Everything in nn.Linear (no conv1d), no nn.Sequential 
class MLPMixerv2(nn.Module):

    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes,
             expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
        super().__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, (
            'image must be divisible by patch size')
        num_patches = (image_h // patch_size) * (image_w // patch_size)

        self.patching = Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_size, p2 = patch_size)

        self.input_proj = nn.Linear((patch_size ** 2) * channels, dim)

        tk_mixer_layers_list = []
        ch_mixer_layers_list = []

        self.depth = depth

        for _ in range(depth):
            tk_mixer_layers_list.append(
                PreNormResidual(
                    dim, FeedForward(
                        num_patches, expansion_factor, dropout, nn.Linear)))
            ch_mixer_layers_list.append(
                PreNormResidual(
                    dim,
                    FeedForward(
                        dim, expansion_factor_token, dropout, nn.Linear)))

        self.tk_mixer_layers = nn.ModuleList(tk_mixer_layers_list)
        self.ch_mixer_layers = nn.ModuleList(ch_mixer_layers_list)
        
        self.layer_norm = nn.LayerNorm(dim)
        self.mean_reduce = Reduce('b n c -> b c', 'mean')

        self.out_proj = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patching(x)  # (B, S=num_patch, C=ph*pw*c)
        x = self.input_proj(x)
        for i in range(self.depth):
            x = self.tk_mixer_layers[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.ch_mixer_layers[i](x)
        x = self.out_proj(self.mean_reduce(self.layer_norm(x)))
        return x
