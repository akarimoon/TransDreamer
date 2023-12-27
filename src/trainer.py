from collections import defaultdict
from functools import partial
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import shutil

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch import optim
from tqdm import tqdm
import wandb

from env.wrapper import make_env
from models import TransDreamer
from utils import set_seed, set_hyperparams


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        # Setup hyperparameter dependencies
        cfg = set_hyperparams(cfg)

        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
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

        # Setup dataset
        train_dataset = instantiate(cfg.datasets.train)
        test_dataset = instantiate(cfg.datasets.test)

        # Setup environment
        self.train_env = make_env(**cfg.env.train, dataset=train_dataset)
        self.test_env = make_env(**cfg.env.test, dataset=test_dataset)
        
        world_model = instantiate(cfg.world_model, sequence_length=cfg.training.sequence_length,
                                  discount=cfg.rl.discount, r_transform=cfg.rl.r_transform)
        actor = instantiate(cfg.actor_critic.actor)
        value = instantiate(cfg.actor_critic.value)
        slow_value = instantiate(cfg.actor_critic.value)
        self.model = TransDreamer(world_model, actor, value, slow_value, 
                                  batch_length=cfg.training.batch_num_samples, lambda_=cfg.rl.lambda_
        ).to(self.device)
        self.model._setup_loss_config(cfg.training.loss)
        self.input_type = self.model.world_model.input_type

        opt_fn = optim.AdamW if cfg.training.optimizer == 'adamW' else optim.Adam
        kwargs = {"weight_decay": cfg.training.weight_decay, "eps": cfg.training.eps}
        self.optimizer_world_model = opt_fn(self.model.world_model.parameters(), cfg.training.model_lr, **kwargs)
        self.optimizer_actor = opt_fn(self.model.actor.parameters(), cfg.training.actor_lr, **kwargs)
        self.optimizer_value = opt_fn(self.model.value.parameters(), cfg.training.value_lr, **kwargs)
        
        # scheduler



    def anneal_temp(self) -> float:
        temp_start = self.cfg.training.temp.start
        temp_end = self.cfg.training.temp.end
        decay_steps = self.cfg.training.temp.decay_steps
        temp = (
            temp_start - (temp_start - temp_end) * (self.global_step - self.cfg.training.prefill) / decay_steps
        )

        temp = max(temp, temp_end)

        return temp

    def run(self) -> None:

        self.global_step = 0
        self.collect_seed_episodes()
        steps = self.train_env.dataset.num_seen_steps
        print(f"Collected {steps} steps. Start training...")

        obs = self.train_env.reset()
        state = None
        action_list = torch.zeros(1, 1, self.cfg.env.action_size).float()  # T, C
        action_list[0, 0, 0] = 1.0

        self.global_step = max(self.global_step, steps)
        init_step = self.global_step
        for step in tqdm(range(init_step, self.cfg.common.total_steps)):
            obs, state, action_list = self.collect(obs, state, action_list)

            if self.global_step % self.cfg.training.train_every == 0:
                self.train()

            if self.global_step % self.cfg.evaluation.eval_every == 0:
                self.test()

            if self.global_step % self.cfg.evaluation.save_every == 0:
                self.save()

            self.global_step += 1

    def collect_seed_episodes(self) -> None:
        print("Collecting seed episodes...")

        self.train_env.reset()
        length = 0
        steps = self.train_env.dataset.num_seen_steps
        with tqdm(total=self.cfg.training.prefill) as pbar:
            while steps < self.cfg.training.prefill:
                action = self.train_env.sample_random_action()
                _, _, done = self.train_env.step(action[0])
                length += 1
                steps += done * length
                if done * length > 0:
                    pbar.update(int(done * length))
                length = length * (1.0 - done)
                if done:
                    self.train_env.reset()

    @torch.no_grad()
    def collect(self, obs, state, action_list) -> None:

        self.model.eval()
        next_obs, reward, done = self.train_env.step(action_list[0, -1].detach().cpu().numpy())

        batch = {
            "prev_image": torch.tensor(obs[self.input_type]),
            "next_image": torch.tensor(next_obs[self.input_type]),
            "action_list": action_list,
        }
        batch = self._to_device(batch)
        action_list, state = self.model.policy(
            batch,
            self.global_step,
            0.1,
            state,
            context_len=self.cfg.training.sequence_length,
        )
        obs = next_obs
        if done:
            self.train_env.reset()
            state = None
            action_list = torch.zeros(1, 1, self.cfg.env.action_size).float()  # T, C
            action_list[0, 0, 0] = 1.0

        return obs, state, action_list
    
    def train(self) -> None:
        temp = self.anneal_temp()

        self.model.train()

        traj = self.train_env.sample_batch(self.cfg.training.batch_num_samples, self.cfg.training.sequence_length)
        traj = self._to_device(traj)

        start_time = time.time()
        to_log = []

        # Train world model
        self.optimizer_world_model.zero_grad()
        model_loss, log, post_state = self.model.compute_world_model_loss(traj, temp)
        for k, v in log.items():
            to_log.append({f'world_model/train/{k}': v})
        model_loss.backward()
        if self.cfg.training.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.world_model.parameters(), self.cfg.training.grad_clip)
        self.optimizer_world_model.step()

        # Train actor-critic
        self.optimizer_actor.zero_grad()
        self.optimizer_value.zero_grad()
        actor_loss, value_loss, log = self.model.compute_actor_and_value_loss(traj, post_state, temp)        
        for k, v in log.items():
            to_log.append({f'actor_critic/train/{k}': v})
        actor_loss.backward()
        value_loss.backward()
        if self.cfg.training.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.cfg.training.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.value.parameters(), self.cfg.training.grad_clip)
        self.optimizer_actor.step()
        self.optimizer_value.step()

        # Update target
        if self.model.slow_update % self.cfg.training.slow_update_step == 0:
            self.model.update_slow_target()

        # Log
        to_log.append({'duration': (time.time() - start_time) / 3600})
        if self.global_step % self.cfg.training.log_every == 0:
            for metrics in to_log:
                wandb.log({'step': self.global_step, **metrics})

    @torch.no_grad()
    def test(self) -> None:
        self.model.eval()

        obs = self.test_env.reset()
        action_list = torch.zeros(1, 1, self.cfg.env.action_size).float()  # T, C
        action_list[:, 0, 0] = 1.0  # B, T, C
        state = None
        done = False

        with torch.no_grad():
            while not done:
                next_obs, reward, done = self.test_env.step(action_list[0, -1].detach().cpu().numpy())
                batch = {
                    "prev_image": torch.tensor(obs[self.input_type]),
                    "next_image": torch.tensor(next_obs[self.input_type]),
                    "action_list": action_list,
                }
                batch = self._to_device(batch)
                action_list, state = self.model.policy(
                    batch,
                    self.global_step,
                    0.1,
                    state,
                    training=False,
                    context_len=self.cfg.training.sequence_length,
                )
                obs = next_obs

    def save(self) -> None:
        torch.save(self.model.state_dict(), self.ckpt_dir / 'last.pt')
        torch.save({
            "optimizer_world_model": self.optimizer_world_model.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_value": self.optimizer_value.state_dict(),
        }, self.ckpt_dir / 'optimizer.pt')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device).float() for k in batch}