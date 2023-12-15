from collections import defaultdict
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import shutil

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch import optim
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.autograd.profiler as profiler
from tqdm import tqdm
import wandb

# from utils import Checkpointer
# from solver import get_optimizer
# from envs import make_env, count_steps
# from data import EnvIterDataset
from envs import make_env
from model import TransformerModel
from utils import set_seed


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)

        # Setup environment
        self.train_env = instantiate(cfg.env.train)
        self.test_env = instantiate(cfg.env.test)

        #TODO:
        # Setup dataset

        # Setup model
        self.model = TransformerModel(cfg.arch).to(self.device)
        #TODO:
        # optimizer
        opt_fn = optim.AdamW if cfg.optimize.optimizer == 'adamW' else optim.Adam
        kwargs = {"weight_decay": cfg.optimize.weight_decay, "eps": cfg.optimize.eps}
        self.optimizer_world_model = opt_fn(self.model.world_model.parameters(), cfg.optimize.model_lr, **kwargs)
        self.optimizer_actor = opt_fn(self.model.actor.parameters(), cfg.optimize.actor_lr, **kwargs)
        self.optimizer_value = opt_fn(self.model.value.parameters(), cfg.optimize.value_lr, **kwargs)
        # scheduler

    def run(self) -> None:

        self.global_step = 0
        self.collect_seed_episodes()

        obs = self.train_env.reset()
        state = None
        action_list = torch.zeros(1, 1, self.cfg.env.action_size).float()  # T, C
        action_list[0, 0, 0] = 1.0
        for step in tqdm(range(self.cfg.total_steps)):
            obs, state, action_list = self.collect(obs, state, action_list)

            self.train()

            if self.global_step % self.cfg.train.eval_every_step == 0:
                self.test()

            self.global_step += 1

    def collect_seed_episodes(self) -> None:
        print("Collecting seed episodes...")

        self.train_env.reset()
        length = 0
        for step in tqdm(range(self.cfg.arch.prefill)):
            action = self.train_env.sample_random_action()
            _, _, done = self.train_env.step(action[0])
            length += 1
            steps += done * length
            length = length * (1.0 - done)
            if done:
                self.train_env.reset()

        self.train_dataset = ...
        self.train_dataloader = ...
        self.train_iter = iter(self.train_dataloader)

    @torch.no_grad()
    def collect(self, obs, state, action_list) -> None:
        input_type = self.cfg.arch.world_model.input_type

        self.model.eval()
        next_obs, reward, done = self.train_env.step(
            action_list[0, -1].detach().cpu().numpy()
        )
        prev_image = torch.tensor(obs[input_type])
        next_image = torch.tensor(next_obs[input_type])
        action_list, state = self.model.policy(
            prev_image.to(self.device),
            next_image.to(self.device),
            action_list.to(self.device),
            self.global_step,
            0.1,
            state,
            context_len=self.cfg.train.batch_length,
        )
        obs = next_obs
        if done:
            self.train_env.reset()
            state = None
            action_list = torch.zeros(1, 1, self.cfg.env.action_size).float()  # T, C
            action_list[0, 0, 0] = 1.0

        return obs, state, action_list
    
    def train(self) -> None:
        self.model.train()

        traj = next(self.train_iter)
        traj = self._to_device(traj)

        start_time = time.time()
        to_log = []

        # Train world model
        log, post_state = self.model.compute_world_model_loss(traj, self.model_optimizer, temp)
        for k, v in log.items():
            to_log.append({f'world_model/train/{k}': v})
        to_log += log

        # Train actor-critic
        log = self.model.compute_actor_and_value_loss(traj, post_state, self.actor_optimizer, self.value_optimizer, temp)        
        for k, v in log.items():
            to_log.append({f'actor_critic/train/{k}': v})
        to_log += log

        # Log
        to_log.append({'duration': (time.time() - start_time) / 3600})
        if self.global_step % self.cfg.train.log_every_step == 0:
            for metrics in to_log:
                wandb.log({'step': self.global_step, **metrics})

    @torch.no_grad()
    def test(self) -> None:
        pass

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device).float() for k in batch}