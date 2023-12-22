# modified from https://github.com/eloialonso/iris/blob/main/src/dataset.py
from collections import deque
from pathlib import Path
import random
from typing import Dict, List, Optional

import psutil
import torch
import wandb

class EpisodesDataset:
    def __init__(self, max_num_episodes, name) -> None:
        self.max_num_episodes = max_num_episodes
        self.name = name
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episodes_metrics = deque()
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def __len__(self) -> int:
        return len(self.episodes)
    
    def add_episode(self, episode) -> int:
        if self.max_num_episodes is not None and len(self.episodes) == self.max_num_episodes:
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id
    
    def get_episode(self, episode_id: int) -> Dict:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]
    
    def _popleft(self) -> Dict:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0]
        assert len(id_to_delete) == 1
        self.newly_deleted_episodes.add(id_to_delete[0])
        self.episode_id_to_queue_idx = {k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0}
        return self.episodes.popleft()
    
    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        to_log = [
            {f"sim/{self.name}/return": float(episode["reward"].sum())},
            {f"sim/{self.name}/length": len(episode["reward"]) - 1},
        ]
        for metrics in to_log:
            wandb.log({'step': self.num_seen_episodes, **metrics})

        self.num_seen_episodes += 1 
        self.newly_modified_episodes.add(episode_id)
        return episode_id
    
    def sample_batch(self, batch_num_samples: int, sequence_length: int, balance: bool = True) -> Dict:
        return self._collate_episodes(self._sample_episodes(batch_num_samples, sequence_length, balance))

    def _sample_episodes(self, batch_num_samples: int, sequence_length: int, balance: bool = True) -> List:
        sampled_episodes = random.choices(self.episodes, k=batch_num_samples)

        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            episode_len = len(sampled_episode["action"])
            available = episode_len - sequence_length
            if available < 1:
                print(f"Skipped short episode of length {available}.")
                continue
            if balance:
                index = min(random.randint(0, episode_len), available)
            else:
                index = int(random.randint(0, available + 1))
            episode = {
                k: v[index:index+sequence_length] for k, v in sampled_episode.items()
            }
            sampled_episodes_segments.append(episode)

        return sampled_episodes_segments
    
    def _collate_episodes(self, episodes: List[Dict]) -> Dict:
        episode_keys = [ep.keys() for ep in episodes]
        batch = {}
        for k in episode_keys[0]:
            batch[k] = torch.stack([torch.tensor(ep[k]) for ep in episodes])

        return batch
    
    def update_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir()
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            torch.save(episode, directory / f'{episode_id}.pt')
        for episode_id in self.newly_deleted_episodes:
            (directory / f'{episode_id}.pt').unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir() and len(self.episodes) == 0
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])
        self.num_seen_episodes = episode_ids[-1] + 1
        for episode_id in episode_ids:
            episode = {**torch.load(directory / f'{episode_id}.pt')}
            self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)

    def save_episodes(self, directory: Path) -> None:
        # append all episodes to list
        observations = []
        for episode_id in range(self.num_seen_episodes):
            observations.append(self.get_episode(episode_id)["image"])
        observations = torch.cat(observations, dim=0)
        print(observations.min(), observations.max(), type(observations))
        torch.save(observations, directory / 'observations.pt')


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """
    Prevent episode dataset from going out of RAM.
    Warning: % looks at system wide RAM usage while G looks only at process RAM usage.
    """
    def __init__(self, max_ram_usage: str, name: str) -> None:
        super().__init__(max_num_episodes=None, name=name)
        self.max_ram_usage = max_ram_usage
        self.num_steps = 0
        self.max_num_steps = None

        max_ram_usage = str(max_ram_usage)
        if max_ram_usage.endswith('%'):
            m = int(max_ram_usage.split('%')[0])
            assert 0 < m < 100
            self.check_ram_usage = lambda: psutil.virtual_memory().percent > m
        else:
            assert max_ram_usage.endswith('G')
            m = float(max_ram_usage.split('G')[0])
            self.check_ram_usage = lambda: psutil.Process().memory_info()[0] / 2 ** 30 > m

    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode) -> int:
        if self.max_num_steps is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps
        self.num_steps += len(episode)
        while (self.max_num_steps is not None) and (self.num_steps > self.max_num_steps):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _popleft(self):
        episode = super()._popleft()
        self.num_steps -= len(episode)
        return episode
