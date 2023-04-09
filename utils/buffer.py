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
            state_dtype = torch.float32,
            action_dtype = torch.float32,
        ):

        '''
        for discrete environment, set dtype = torch.long
        '''

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.buffer_size = buffer_size

        self.states = torch.zeros((n_envs, buffer_size, *state_dim), dtype = state_dtype)
        self.actions = torch.zeros((n_envs, buffer_size, action_dim), dtype = action_dtype)
        self.rewards = torch.zeros((n_envs, buffer_size, 1), dtype = torch.float32)
        self.next_states = torch.zeros((n_envs, buffer_size, *state_dim), dtype = state_dtype)
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
        
        '''
        add a tuple (s, a, r, s') into the buffer. Only the most recent buffer_size tuples will be kept.

        self.n: the position of the pointer.
        self.n_transitions: how many tuples are stored in the buffer, at most buffer_size.
        '''

        self.states[:, self.n] = torch.tensor(states, dtype = torch.float32)
        self.actions[:, self.n] = torch.tensor(actions, dtype = torch.float32)
        self.rewards[:, self.n] = torch.tensor(rewards, dtype = torch.float32)
        self.next_states[:, self.n] = torch.tensor(next_states, dtype = torch.float32)
        self.terminated[:, self.n] = torch.tensor(terminated, dtype = torch.bool)
        self.truncated[:, self.n] = torch.tensor(truncated, dtype = torch.bool)

        self.n = (self.n + 1) % self.buffer_size
        if self.n_transitions < self.buffer_size:
            self.n_transitions += 1

    def sample(self, batch_size: Optional[int] = None, device = 'cpu'):

        '''
        sample transitions from the buffer.

        batch_size: number of transitions sampled from the buffer. If None, all transitions will be retrieved. In this case, the transitions are not disrupted so that some quantities, e.g., lambda return, can be computed.
        '''

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