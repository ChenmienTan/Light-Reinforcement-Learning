from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

import wandb

class Buffer:

    def __init__(
        self,
        state_dim: Tuple[int],
        action_dim: int,
        n_envs: int,
        buffer_size: int,
        batch_size: Optional[int] = None,
        device: Optional[Union[str, int, torch.device]] = "cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        
        self.idx, self.n_trans = 0, 0

        self.states = np.zeros((buffer_size, n_envs, *state_dim), dtype = np.float32)
        self.actions = np.zeros((buffer_size, n_envs, action_dim), dtype = np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.next_states = np.zeros((buffer_size, n_envs, *state_dim), dtype = np.float32)
        self.terminates = np.zeros((buffer_size, n_envs), dtype = np.bool)
        self.truncates = np.zeros((buffer_size, n_envs), dtype = np.bool)

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states : np.ndarray,
        terminates: np.ndarray,
        truncates: np.ndarray
    ):
        
        self.states[self.idx] = states
        self.actions[self.idx] = actions
        self.rewards[self.idx] = rewards
        self.next_states[self.idx] = next_states
        self.terminates[self.idx] = terminates
        self.truncates[self.idx] = truncates
        
        self.idx = (self.idx + 1) % self.buffer_size
        self.n_trans = min(self.n_trans + 1, self.buffer_size)

    def sample(self):

        states = self.states[:self.n_trans]
        actions = self.actions[:self.n_trans]
        rewards = self.rewards[:self.n_trans]
        next_states = self.next_states[:self.n_trans]
        terminates = self.terminates[:self.n_trans]
        truncates = self.truncates[:self.n_trans]
        
        truncates[self.idx - 1] = np.zeros(self.n_envs, dtype = np.bool)
        truncates[-1] = np.zeros(self.n_envs, dtype = np.bool)
        
        states = states.swapaxes(0, 1).reshape(-1, *self.state_dim)
        actions = actions.swapaxes(0, 1).reshape(-1, self.action_dim)
        rewards = rewards.swapaxes(0, 1).flatten()
        next_states = next_states.swapaxes(0, 1).reshape(-1, *self.state_dim)
        terminates = terminates.swapaxes(0, 1).flatten()
        truncates = truncates.swapaxes(0, 1).flatten()

        if self.batch_size is not None:

            indices = np.random.choice(self.n_envs * self.n_trans, size = self.batch_size)

            states = states[indices]
            actions = actions[indices]
            rewards = rewards[indices]
            next_states = next_states[indices]
            terminates = terminates[indices]
            truncates = truncates[indices]

        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(rewards).to(self.device),
            torch.from_numpy(next_states).to(self.device),
            torch.from_numpy(terminates).to(self.device),
            torch.from_numpy(truncates).to(self.device)
        )


class Trainer:

    def __init__(
        self,
        train_envs,
        valid_envs,
        buffer: Buffer,
        policy: nn.Module,
        norm_states: Optional[bool] = False
    ):
        
        self.train_envs = train_envs
        self.valid_envs = valid_envs
        self.buffer = buffer
        self.policy = policy
        self.norm_states = norm_states
        
        self.states, _ = self.train_envs.reset()
        
        if norm_states:
            self.normalizer = RunningMeanStd(train_envs.observation_space.shape[0])
            self.states = self.normalizer(self.states, True)
        
    def collect(self, n_steps: int, random: Optional[bool] = False):
        
        for _ in range(n_steps):
                
            if random:
                actions = np.stack([self.train_envs.action_space.sample() for _ in range(len(self.train_envs))])
            else:
                th_states = torch.from_numpy(self.states).to(torch.float32).to(self.policy.device)
                with torch.no_grad():
                    th_actions = self.policy.actor(th_states, False)
                actions = th_actions.to("cpu").numpy()
                
            next_states, rewards, terminates, truncates, _ = self.train_envs.step(actions)
            if self.norm_states:
                next_states = self.normalizer(next_states, True)
            self.buffer.add(self.states, actions, rewards, next_states, terminates, truncates)
            dones = np.logical_or(terminates, truncates)
            if dones.any():
                indices = np.where(dones)[0]
                states, _ = self.train_envs.reset(indices)
                if self.norm_states:
                    states = self.normalizer(states, True)
                next_states[indices] = states
            self.states = next_states
                
    def valid(self):

        returns = np.zeros(len(self.valid_envs))
        dones = np.zeros(len(self.valid_envs), dtype = bool)
        all_states, _ = self.valid_envs.reset()

        while not dones.all():
            indices = np.where(np.logical_not(dones))[0]
            states = all_states[indices]
            if self.norm_states:
                states = self.normalizer(states, False)
            th_states = torch.from_numpy(states).to(torch.float32).to(self.policy.device)
            with torch.no_grad():
                th_actions = self.policy.actor(th_states, True)
            actions = th_actions.to("cpu").numpy()
            states, rewards, terminates, truncates, _ = self.valid_envs.step(actions, indices)
            all_states[indices] = states
            returns[indices] += rewards
            dones[indices] = dones[indices] | terminates | truncates

        return np.mean(returns)
    
    def run(
        self,
        n_epochs: int,
        step_per_epoch: int
    ):
        
        for n_epoch in tqdm(n_epochs, ncols = 100):
            self.collect(step_per_epoch)
            self.policy.learn(self.buffer)
            ret = self.valid()

            wandb.log({
                "step": (n_epoch + 1) * step_per_epoch * len(self.train_envs),
                "return": ret
            })
                

class RunningMeanStd:

    def __init__(self, dim: int):

        self.n = 0
        self.mean = np.zeros(dim)
        self.var = np.zeros(dim)
        self.std = np.zeros(dim)

    def __call__(self, x: np.ndarray, training: bool) -> np.ndarray:

        if training:
            n = self.n + x.shape[0]
            delta = np.mean(x, 0) - self.mean
            self.mean += x.shape[0] * delta / n
            self.var += x.shape[0] * np.var(x, 0) + self.n * x.shape[0] * np.square(delta) / n
            self.std = np.sqrt(self.var / (n - 1 + np.finfo(np.float32).eps))
            self.n = n

        return (x - self.mean) / (self.std + np.finfo(np.float32).eps)