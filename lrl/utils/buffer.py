from typing import Optional, Sequence

import numpy as np
import torch


class Buffer:

    def __init__(
            self,
            state_dim: Sequence[int],
            action_dim: int,
            n_envs: int,
            buffer_size: int,
            action_dtype = torch.float32,
        ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.buffer_size = buffer_size

        self.states = torch.zeros((n_envs, buffer_size, *state_dim), dtype = torch.float32)
        self.actions = torch.zeros((n_envs, buffer_size, action_dim), dtype = action_dtype)
        self.rewards = torch.zeros((n_envs, buffer_size, 1), dtype = torch.float32)
        self.next_states = torch.zeros((n_envs, buffer_size, *state_dim), dtype = torch.float32)
        self.terminated = torch.zeros((n_envs, buffer_size, 1), dtype = torch.bool)
        self.truncated = torch.zeros((n_envs, buffer_size, 1), dtype = torch.bool)

        self.n = 0
        self.n_transitions = 0

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states : np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray
    ):

        self.states[:, self.n] = torch.Tensor(states)
        self.actions[:, self.n] = torch.Tensor(actions)
        self.rewards[:, self.n] = torch.Tensor(rewards)
        self.next_states[:, self.n] = torch.Tensor(next_states)
        self.terminated[:, self.n] = torch.Tensor(terminated)
        self.truncated[:, self.n] = torch.Tensor(truncated)

        self.n = (self.n + 1) % self.buffer_size
        if self.n_transitions < self.buffer_size:
            self.n_transitions += 1

    def sample(self, batch_size: Optional[int] = None, device: Optional[str] = "cpu"):

        n_transitions = self.n_envs * self.n_transitions

        states = self.states[:, :self.n_transitions].view((n_transitions, *self.state_dim))
        actions = self.actions[:, :self.n_transitions].view((n_transitions, self.action_dim))
        rewards = self.rewards[:, :self.n_transitions].view((n_transitions, 1))
        next_states = self.next_states[:, :self.n_transitions].view((n_transitions, *self.state_dim))
        terminated = self.terminated[:, :self.n_transitions].view((n_transitions, 1))
        truncated = self.truncated[:, :self.n_transitions]

        truncated[:, self.n - 1] = torch.tensor([True for _ in range(self.n_envs)]).unsqueeze(-1)
        truncated[:, -1] = torch.tensor([True for _ in range(self.n_envs)]).unsqueeze(-1)
        truncated = truncated.view((n_transitions, 1))

        if batch_size is not None:

            indices = np.random.choice(n_transitions, size = batch_size)

            states = states[indices]
            actions = actions[indices]
            rewards = rewards[indices]
            next_states = next_states[indices]
            terminated = terminated[indices]
            truncated = truncated[indices]

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        terminated = terminated.to(device)
        truncated = truncated.to(device)

        return states, actions, rewards, next_states, terminated, truncated