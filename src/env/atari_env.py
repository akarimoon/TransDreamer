import threading

import numpy as np
import gym
import pdb


class Atari:
    LOCK = threading.Lock()

    def __init__(
        self,
        id,
        action_repeat=4,
        size=(84, 84),
        grayscale=True,
        noops=30,
        life_done=False,
        sticky_actions=True,
        all_actions=True,
    ):
        assert size[0] == size[1]
        # import gym.wrappers
        # import gym.envs.atari

        # if name == "james_bond":
        #     name = "jamesbond"
        # with self.LOCK:
        #     env = gym.envs.atari.AtariEnv(
        #         game=name,
        #         obs_type="image",
        #         frameskip=1,
        #         repeat_action_probability=0.25 if sticky_actions else 0.0,
        #         full_action_space=all_actions,
        #     )
        # # Avoid unnecessary rendering in inner env.
        # env._get_obs = lambda: None
        # # Tell wrapper that the inner env has no action repeat.
        # env.spec = gym.envs.registration.EnvSpec("NoFrameskip-v0")
        # env = gym.wrappers.AtariPreprocessing(
        #     env, noops, action_repeat, size[0], life_done, grayscale
        # )
        with self.LOCK:
            env = gym.make(
                id=id,
                obs_type="image",
                frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions,
            )
        env._get_obs = lambda: None
        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale
        )
        assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
        self._env = env
        self._grayscale = grayscale
        self._size = size
        self.action_size = 18

    @property
    def observation_space(self):
        shape = (1 if self._grayscale else 3,) + self._size
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({"image": space})
        # return gym.spaces.Dict({
        #     'image': self._env.observation_space,
        # })

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            image, _ = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
        return {"image": image}

    def step(self, action):
        image, reward, done, _, info = self._env.step(action)
        if self._grayscale:
            image = image[..., None]
        image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
        obs = {"image": image}
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)