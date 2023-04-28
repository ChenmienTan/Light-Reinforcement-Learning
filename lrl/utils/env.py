from typing import Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np


class Envs:

    def __init__(self, envs: Sequence[gym.Env]):

        self.envs = envs

    def __getitem__(self, idx: int) -> gym.Env:

        return self.envs[idx]

    def __len__(self) -> int:

        return len(self.envs)

    def reset(
            self,
            indices: Optional[Sequence[int]] = None
        ) -> Tuple[np.ndarray, Tuple[dict]]:

        envs = self.envs if indices is None else [self[indice] for indice in indices]
        batch = [env.reset() for env in envs]
        states, infos = zip(*batch)
        states = np.stack(states)

        return states, infos

    def step(
            self,
            actions: np.ndarray,
            indices: Optional[Sequence[int]] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        envs = self.envs if indices is None else [self[indice] for indice in indices]
        actions = actions.squeeze(-1) if actions.shape[-1] == 1 else actions
        batch = [env.step(action) for env, action in zip(envs, actions)]
        states, rewards, terminated, truncated, infos = zip(*batch)
        states = np.stack(states)
        rewards = np.stack(rewards)[:, None]
        terminated = np.stack(terminated)[:, None]
        truncated = np.stack(truncated)[:, None]

        return states, rewards, terminated, truncated, infos