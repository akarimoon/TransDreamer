import random

import numpy as np
from omegaconf import DictConfig
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def set_hyperparams(cfg: DictConfig) -> DictConfig:
    # dense input size & d_model
    deter_type = cfg.world_model.dynamic.cell.config.deter_type
    stoch_size = cfg.world_model.dynamic.config.stoch_size
    stoch_discrete = cfg.world_model.dynamic.config.stoch_discrete
    n_layers = cfg.world_model.dynamic.cell.config.n_layers

    d_model = cfg.world_model.dynamic.cell.config.d_model
    if deter_type == "concat_o":
        d_model = n_layers * d_model

    if stoch_discrete:
        dense_input_size = d_model + stoch_size * stoch_discrete
    else:
        dense_input_size = d_model + stoch_size
    
    cfg.world_model.img_dec.config.input_size = dense_input_size
    cfg.world_model.reward.config.input_size = dense_input_size
    cfg.world_model.pcont.config.input_size = dense_input_size
    cfg.actor_critic.actor.config.input_size = dense_input_size
    cfg.actor_critic.value.config.input_size = dense_input_size
    cfg.actor_critic.slow_value.config.input_size = dense_input_size

    # c_in & c_out
    cfg.world_model.dynamic.img_enc.config.c_in = 1 if cfg.env.train.grayscale else 3
    cfg.world_model.img_dec.config.c_out = 1 if cfg.env.train.grayscale else 3

    # action size
    cfg.world_model.dynamic.config.action_size = cfg.env.action_size
    cfg.actor_critic.actor.config.action_size = cfg.env.action_size

    # action repeat
    cfg.datasets.train.action_repeat = cfg.env.train.action_repeat
    cfg.datasets.test.action_repeat = cfg.env.test.action_repeat

    return cfg