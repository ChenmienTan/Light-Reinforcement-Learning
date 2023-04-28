from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent

def orthogonal_init(net: nn.Module, gain: float):

    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain = gain)
            nn.init.zeros_(module.bias)

def soft_update(tau: float, net: nn.Module, target_net: nn.Module):

    for params, target_params in zip(net.parameters(), target_net.parameters()):

        target_params.data.copy_(tau * params + (1 - tau) * target_params)


class MLP(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            activation_fn: nn.Module = nn.ReLU
        ):
        super().__init__()

        input_sizes = [input_size] + hidden_sizes[:-1]
        modules = []
        for input_size, output_size in zip(input_sizes, hidden_sizes):
            modules += [nn.Linear(input_size, output_size), activation_fn()]
        self.net = nn.Sequential(*modules)

    def forward(
            self,
            states: torch.Tensor,
            actions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        if actions is not None:
            states = torch.cat([states, actions], dim = -1)

        return self.net(states)
    

class ConvNet(nn.Module):

    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        hidden_size: int,
        activation_fn: nn.Module
    ):
        super().__init__()

        modules = []
        for in_channel, out_channel, kernel_size, stride, padding in zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings):
            modules += [nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding), activation_fn()]
        modules += [nn.Flatten(), nn.LazyLinear(hidden_size), activation_fn()]
        self.net = nn.Sequential(*modules)

    def forward(self, states: torch.Tensor) -> torch.Tensor:

        return self.net(states)


class DiscreteActor(nn.Module):

    def __init__(
            self,
            preprocess_net: nn.Module,
            hidden_size: int,
            action_dim: int
        ):
        super().__init__()

        self.preprocess_net = preprocess_net
        self.logits = nn.Linear(hidden_size, action_dim)

    def forward(
            self,
            states: torch.Tensor,
            deterministic: Optional[bool] = False
        ) -> torch.Tensor:

        hidden_states = self.preprocess_net(states)
        logits = self.logits(hidden_states)

        if deterministic:
            actions = logits.argmax(-1).unsqueeze(-1)
        else:
            dists = Categorical(logits = logits)
            actions = dists.sample().unsqueeze(-1)

        return actions
    
    # For PPO
    def compute_dists_and_log_probs(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[Categorical, torch.Tensor]:

        hidden_states = self.preprocess_net(states)
        logits = self.logits(hidden_states)
        dists = Categorical(logits = logits)
        log_probs = dists.log_prob(actions.squeeze(-1)).unsqueeze(-1)

        return dists, log_probs
    
    # For SAC
    def compute_probs(self, states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.preprocess_net(states)
        logits = self.logits(hidden_states)
        probs = logits.softmax(-1)

        return probs
    

class GaussianActor(nn.Module):

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_size: int,
        action_dim: int,
        action_bound: float,
        bound_action_method: str,
        log_sigma: Optional[float] = None,
        trainable_sigma: Optional[bool] = True
    ):
        
        assert log_sigma is not None or trainable_sigma, "the standard variance must be assigned if it is not a learnable parameter."

        super().__init__()

        self.preprocess_net = preprocess_net
        self.action_bound = action_bound
        self.bound_action_method = bound_action_method
        self.log_sigma = log_sigma

        self.mu = nn.Linear(hidden_size, action_dim)

        if log_sigma is None:
            self.sigma = nn.Linear(hidden_size, action_dim)
        else:
            self.sigma = nn.Parameter(log_sigma * torch.ones(action_dim), requires_grad = trainable_sigma)
            
        orthogonal_init(self.preprocess_net, gain = np.sqrt(2))
        orthogonal_init(self.mu, gain = np.sqrt(2) * 1e-2)

    def compute_dists(self, states: torch.Tensor) -> Independent:

        hidden_states = self.preprocess_net(states)
        mus = self.mu(hidden_states)

        if self.log_sigma is None:
            log_sigmas = self.sigma(hidden_states)
        else:
            log_sigmas = self.sigma.expand_as(mus)

        sigmas = log_sigmas.clamp(-20, 2).exp()
        dists = Independent(Normal(mus, sigmas), 1)

        return dists

    def forward(
            self,
            states: torch.Tensor,
            deterministic: Optional[bool] = False
        ) -> torch.Tensor:
            
        if deterministic:
            hidden_states = self.preprocess_net(states)
            actions = self.mu(hidden_states)
        else:
            dists = self.compute_dists(states)
            actions = dists.rsample()

        if self.bound_action_method == "clip":
            actions = actions.clamp(-self.action_bound, self.action_bound)
        elif self.bound_action_method == "tanh":
            actions = self.action_bound * actions.tanh()

        return actions
    
    # for PPO
    def compute_dists_and_log_probs(
            self,
            states: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[Independent, torch.Tensor]:

        dists = self.compute_dists(states)
        if self.bound_action_method == "tanh":
            orginal_actions = torch.atanh(actions / self.action_bound)
        
        log_probs = dists.log_prob(orginal_actions).unsqueeze(-1)

        if self.bound_action_method == "tanh":
            log_probs = (log_probs - torch.log(1 - actions.pow(2) + torch.finfo(torch.float32).eps).sum(-1).unsqueeze(-1)) / self.action_bound

        return dists, log_probs
    
    # for SAC
    def compute_actions_and_log_probs(
            self,
            states: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        dists = self.compute_dists(states)
        actions = dists.rsample()

        if self.bound_action_method == "clip":
            actions = actions.clamp(-self.action_bound, self.action_bound)

        log_probs = dists.log_prob(actions).unsqueeze(-1)

        if self.bound_action_method == "tanh":
            actions = self.action_bound * actions.tanh()
            log_probs = (log_probs - (1 - actions.pow(2) + torch.finfo(torch.float32).eps).log().sum(-1).unsqueeze(-1)) / self.action_bound

        return actions, log_probs
    

class DiscreteCritic(nn.Module):

    def __init__(self, preprocess_net: nn.Module, hidden_size: int, action_dim: int):
        super().__init__()

        self.preprocess_net = preprocess_net
        self.value = nn.Linear(hidden_size, action_dim)

    def forward(
            self,
            states: torch.Tensor,
            actions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        hidden_states = self.preprocess_net(states)
        values = self.value(hidden_states)
        if actions is not None:
            actions = actions.squeeze(-1)
            values = values[torch.arange(values.shape[0]), actions]
            values = values.unsqueeze(-1)

        return values


class ContinuousCritic(nn.Module):

    def __init__(self, preprocess_net: nn.Module, hidden_size: int):
        super().__init__()

        self.preprocess_net = preprocess_net
        self.value = nn.Linear(hidden_size, 1)

    def forward(
            self,
            states: torch.Tensor,
            actions: Optional[torch.Tensor]
        ) -> torch.Tensor:

        hidden_states = self.preprocess_net(states, actions)
        values = self.value(hidden_states)

        return values
