from dataclasses import dataclass
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, Bernoulli
from torch.distributions.one_hot_categorical import OneHotCategorical

from .distributions import SafeTruncatedNormal, ContDist
from .utils import (
    Conv2DBlock,
    ConvTranspose2DBlock,
    Linear,
    MLP,
    GRUCell,
    LayerNormGRUCell,
    LayerNormGRUCellV2,
)


@dataclass
class ImgEncoderConfig:
    c_in: int

@dataclass
class ImgDecoderConfig:
    input_size: int
    c_out: int
    dec_type: str
    rec_sigma: float

class ImgEncoder(nn.Module):
    def __init__(self, config: ImgEncoderConfig):
        super().__init__()

        self.c_in = config.c_in
        depth = 48

        self.enc = nn.Sequential(
            Conv2DBlock(config.c_in, depth, 4, 2, 0, num_groups=0, bias=True,non_linearity=True, act="elu", weight_init="xavier"),
            Conv2DBlock(depth, 2 * depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=True, act="elu", weight_init="xavier"),
            Conv2DBlock(2 * depth, 4 * depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=True, act="elu", weight_init="xavier"),
            Conv2DBlock(4 * depth, 8 * depth, 4, 2, 0, num_groups=0, bias=True, non_linearity=True, act="elu", weight_init="xavier"),
        )

    def forward(self, ipts):
        """
        ipts: tensor, (B, T, 3, 64, 64)
        return: tensor, (B, T, 1024)
        """

        shape = ipts.shape
        o = self.enc(rearrange(ipts, "b t c h w -> (b t) c h w"))
        o = rearrange(o, "(b t) c h w -> b t (c h w)", b=shape[0])

        return o


class ImgDecoder(nn.Module):
    def __init__(self, config: ImgDecoderConfig):
        super().__init__()

        self.c_out = config.c_out
        depth = 48

        self.fc = Linear(config.input_size, 1536, bias=True, weight_init="xavier")
        if config.dec_type == "conv":
            self.dec = nn.Sequential(
                ConvTranspose2DBlock(1536, 4 * depth, 5, 2, 0, 0, num_groups=0, bias=True, non_linearity=True, act="elu", weight_init="xavier"),
                ConvTranspose2DBlock(4 * depth, 2 * depth, 5, 2, 0, 0, num_groups=0, bias=True, non_linearity=True, act="elu", weight_init="xavier"),
                ConvTranspose2DBlock(2 * depth, depth, 5, 2, 0, 1, num_groups=0, bias=True, non_linearity=True, act="elu", weight_init="xavier"),
                ConvTranspose2DBlock(depth, config.c_out, 6, 2, 0, 0, num_groups=0, bias=True, non_linearity=False, weight_init="xavier"),
            )

        elif config.dec_type == "pixelshuffle":
            pass

        else:
            raise ValueError(f"decoder type {config.dec_type} is not supported.")

        self.shape = (config.c_out, 64, 64)
        self.rec_sigma = config.rec_sigma

    def forward(self, ipts):
        """
        ipts: tensor, (B, T, C)
        """

        shape = ipts.shape

        fc_o = self.fc(ipts)
        dec_o = self.dec(rearrange(fc_o, "b t c -> (b t) c 1 1"))
        dec_o = rearrange(dec_o, "(b t) c h w -> b t c h w", b=shape[0])

        dec_pdf = Independent(
            Normal(dec_o, self.rec_sigma * dec_o.new_ones(dec_o.shape)), len(self.shape)
        )

        return dec_pdf


@dataclass
class DenseDecoderConfig:
    input_size: int
    layers: int
    units: int
    output_size: int
    dist: str = "normal"
    act: str = "relu"
    weight_init: str = "xavier"

    @property
    def output_shape(self):
        if isinstance(self.output_size, int):
            return (self.output_size,)
        return self.output_size

class DenseDecoder(nn.Module):
    def __init__(self, config: DenseDecoderConfig):
        super().__init__()

        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "celu": nn.CELU,
        }
        module_list = []

        for i in range(config.layers):
            if i == 0:
                dim_in = config.input_size
            else:
                dim_in = config.units
            dim_out = config.units

            module_list.append(Linear(dim_in, dim_out, weight_init=config.weight_init))
            module_list.append(acts[config.act]())

        module_list.append(Linear(dim_out, 1, weight_init=config.weight_init))
        self.dec = nn.Sequential(*module_list)

        self.dist = config.dist
        self.output_shape = config.output_shape

    def forward(self, inpts):
        logits = self.dec(inpts)
        logits = logits.float()

        if self.dist == "normal":
            pdf = Independent(Normal(logits, 1), len(self.output_shape))

        elif self.dist == "binary":
            pdf = Independent(Bernoulli(logits=logits), len(self.output_shape))

        else:
            raise NotImplementedError(self.dist)

        return pdf


@dataclass
class ActionDecoderConfig:
    input_size: int
    action_size: int
    layers: int
    units: int
    dist: str = "onehot"
    act: str = "relu"
    min_std: float = 0.1
    init_std: float = 5.0
    mean_scale: float = 5.0
    weight_init: str = "xavier"

class ActionDecoder(nn.Module):
    def __init__(self, config: ActionDecoderConfig):
        super().__init__()

        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "celu": nn.CELU,
        }
        module_list = []

        for i in range(config.layers):
            if i == 0:
                dim_in = config.input_size
            else:
                dim_in = config.units
            dim_out = config.units

            module_list.append(Linear(dim_in, dim_out, weight_init=config.weight_init))
            module_list.append(acts[config.act]())

        if config.dist == "trunc_normal":
            module_list.append(Linear(dim_out, 2 * config.action_size, weight_init=config.weight_init))

        elif config.dist == "onehot":
            module_list.append(Linear(dim_out, config.action_size, weight_init=config.weight_init))

        else:
            raise NotImplementedError(self.dist)

        self.dec = nn.Sequential(*module_list)
        self.dist = config.dist
        self.raw_init_std = np.log(np.exp(config.init_std) - 1)
        self.min_std = config.min_std
        self.mean_scale = config.mean_scale

    def forward(self, inpts):
        logits = self.dec(inpts)

        logits = logits.float()

        if self.dist == "trunc_normal":
            mean, std = torch.chunk(logits, 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self.min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(Independent(dist, 1))

        if self.dist == "onehot":
            dist = OneHotCategorical(logits=logits)

        return dist