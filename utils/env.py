import numpy as np


class Envs:

    '''
    vector wrapper of environments.

    Usage:
        states, infos = envs.reset(), where states.shape = (n_envs, state.shape).
        states, rewards, terminated, truncated, infos = envs.step(actions), where states.shape = (n_envs, state.shape), rewards.shape = (n_envs, 1), terminated.shape = (n_envs, 1), truncated.shape = (n_envs, 1).
    '''

    def __init__(self, envs):

        self.envs = envs

    def __getitem__(self, idx):

        return self.envs[idx]

    def __len__(self):

        return len(self.envs)

    def reset(self, indices = None):

        envs = self.envs if indices is None else [self[indice] for indice in indices]
        batch = [env.reset() for env in envs]
        states, infos = zip(*batch)
        states = np.stack(states)

        return states, infos

    def step(self, actions, indices = None):

        envs = self.envs if indices is None else [self[indice] for indice in indices]
        actions = actions.squeeze(-1) if actions.shape[-1] == 1 else actions
        batch = [env.step(action) for env, action in zip(envs, actions)]
        states, rewards, terminated, truncated, infos = zip(*batch)
        states = np.stack(states)
        rewards = np.stack(rewards)[:, None]
        terminated = np.stack(terminated)[:, None]
        truncated = np.stack(truncated)[:, None]

        return states, rewards, terminated, truncated, infos