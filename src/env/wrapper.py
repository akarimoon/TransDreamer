import numpy as np

from .atari_env import Atari
from .crafter import Crafter

import numpy as np
import gym
import pdb

class OneHotAction:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def sample_random_action(self):
        action = np.zeros(
            (
                1,
                self._env.action_space.n,
            ),
            dtype=np.float,
        )
        idx = np.random.randint(0, self._env.action_space.n, size=(1,))[0]
        action[0, idx] = 1
        return action


class TimeLimit:
    def __init__(self, env, duration, time_penalty):
        self._env = env
        self._step = None
        self._duration = duration
        self.time_penalty = time_penalty

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self.time_penalty:
            reward = reward - 1.0 / self._duration

        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class Collect:
    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        transition["action"] = action
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        transition["done"] = float(done)
        self._episode.append(transition)
        if done:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
        obs["image"] = obs["image"][None, ...]
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition["action"] = np.zeros(self._env.action_space.n)
        transition["reward"] = 0.0
        transition["discount"] = 1.0
        transition["done"] = 0.0
        self._episode = [transition]
        obs["image"] = obs["image"][None, ...]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            pdb.set_trace()
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert "reward" not in spaces
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["reward"] = reward
        return obs, reward, done

    def reset(self):
        obs = self._env.reset()
        obs["reward"] = 0.0
        return obs


def make_env(cfg):
    suite, task = cfg.env.name.split("_", 1)

    if suite == "atari":
        env = Atari(
            task,
            cfg.env.action_repeat,
            (64, 64),
            grayscale=cfg.env.grayscale,
            life_done=False,
            sticky_actions=True,
            all_actions=cfg.env.all_actions,
        )
        env = OneHotAction(env)

    elif suite == "crafter":
        env = Crafter(task, (64, 64))
        env = OneHotAction(env)

    else:
        raise NotImplementedError(suite)

    env = TimeLimit(env, cfg.env.time_limit, cfg.env.time_penalty)

    callbacks = []
    #TODO: add callbacks
    # if store:
    #     callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    # callbacks.append(lambda ep: summarize_episode(ep, cfg, datadir, writer, prefix))
    env = Collect(env, callbacks, cfg.env.precision)
    env = RewardObs(env)

    return env