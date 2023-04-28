from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from lrl.utils import Envs
from lrl.utils import Buffer


def train(train_envs, test_envs, buffer, policy, n_epochs, collect_per_epoch, step_per_collect, **kwargs):

    Trainer(train_envs, test_envs, buffer, policy, **kwargs).train(n_epochs, collect_per_epoch, step_per_collect)


class Trainer:

    def __init__(
        self,
        train_envs: Envs,
        test_envs: Envs,
        buffer: Buffer,
        policy: nn.Module,
        norm_states: Optional[bool] = False,
        scale_rewards: Optional[bool] = False
    ):

        self.train_envs = train_envs
        self.test_envs = test_envs
        self.buffer = buffer
        self.policy = policy
        self.norm_states = norm_states
        self.scale_rewards = scale_rewards

        self.states, _ = self.train_envs.reset()

        if norm_states:
            self.state_normalizer = RunningMeanStd(shape = train_envs[0].observation_space.shape)
            self.states = self.state_normalizer(self.states)
        if scale_rewards:
            self.reward_scaler = RewardScaler(n_envs = len(train_envs), gamma = policy.gamma)
        
    def collect(self, n_steps: int, random: Optional[bool] = False):

        with torch.no_grad():
            for _ in range(n_steps):
                
                if random:
                    actions = np.stack([self.train_envs[0].action_space.sample() for _ in range(len(self.train_envs))])
                else:
                    states = torch.tensor(self.states, dtype = torch.float32).to(self.policy.device)
                    actions = self.policy.actor(states)
                    actions = actions.to("cpu").numpy()

                next_states, rewards, terminated, truncated, _ = self.train_envs.step(actions)

                if self.norm_states:
                    next_states = self.state_normalizer(next_states)
                if self.scale_rewards:
                    rewards = self.reward_scaler(rewards, terminated, truncated)

                self.buffer.add(self.states, actions, rewards, next_states, terminated, truncated)

                dones = np.logical_or(terminated, truncated)
                if dones.any():
                    indices = np.where(dones)[0]
                    states, _ = self.train_envs.reset(indices)
                    if self.norm_states:
                        states = self.state_normalizer(states)
                    next_states[indices] = states
                
                self.states = next_states

    def test(self):

        with torch.no_grad():

            returns, dones = np.zeros((len(self.test_envs), 1)), np.zeros((len(self.test_envs), 1), dtype = bool)
            all_states, _ = self.test_envs.reset()

            while not dones.all():
                indices = np.where(np.logical_not(dones))[0]
                states = all_states[indices]

                if self.norm_states:
                    states = self.state_normalizer(states, update = False)

                states = torch.tensor(states, dtype = torch.float32).to(self.policy.device)
                actions = self.policy.actor(states, deterministic = True)
                actions = actions.to("cpu").numpy()
                states, rewards, terminated, truncated, _ = self.test_envs.step(actions, indices)
                all_states[indices] = states
                returns[indices] += rewards
                dones[indices] = dones[indices] | terminated | truncated

        return np.mean(returns)

    def train(
        self,
        n_epochs: int,
        collect_per_epoch: int,
        step_per_collect: int
    ):

        with tqdm(total = n_epochs * collect_per_epoch) as tbar:

            for n_epoch in range(n_epochs):
                for _ in range(collect_per_epoch):

                    self.collect(n_steps = step_per_collect)
                    self.policy.learn(self.buffer)
                    tbar.update()

                ret = self.test()

                wandb.log({
                    "step": (n_epoch + 1) * collect_per_epoch * step_per_collect * len(self.train_envs),
                    "return": ret
                })


class RunningMeanStd:

    def __init__(self, shape: Sequence[int]):

        self.n = 0
        self.mean = np.zeros(*shape)
        self.S = np.zeros(*shape)
        self.std = np.zeros(*shape)

    def __call__(self, x, update = True):

        if update:
            n = self.n + x.shape[0]
            delta = np.mean(x, axis = 0) - self.mean
            self.mean += x.shape[0] * delta / n
            self.S += x.shape[0] * np.var(x, axis = 0) + self.n * x.shape[0] * np.square(delta) / n
            self.std = np.sqrt(self.S / n)
            self.n = n

        return (x - self.mean) / (self.std + np.finfo(np.float32).eps)


class RewardScaler:

    def __init__(self, n_envs: int, gamma: float):

        self.R = np.zeros(n_envs)
        self.gamma = gamma
        self.normalizer = RunningMeanStd(shape = n_envs)

    def __call__(self, rewards, terminated, truncated):

        self.R = self.gamma * self.R + rewards.squeeze(-1)
        self.normalizer(self.R)
        dones = np.logical_or(terminated, truncated)
        if dones.any():
            indices = np.where(dones)[0]
            self.R[indices] = 0

        return rewards / (self.normalizer.std[:, None] + np.finfo(np.float32).eps)