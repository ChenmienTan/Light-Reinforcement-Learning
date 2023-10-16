from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

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
        activation_fn: nn.Module
    ):
        super().__init__()

        input_sizes = [input_size] + hidden_sizes[:-1]
        modules = []
        for input_size, output_size in zip(input_sizes, hidden_sizes):
            modules += [nn.Linear(input_size, output_size), activation_fn()]
        self.net = nn.Sequential(*modules)

    def forward(
        self,
        states: torch.FloatTensor,
        actions: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:

        if actions is not None:
            states = torch.cat([states, actions], dim = -1)

        return self.net(states)
    

class GaussianActor(nn.Module):

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_size: int,
        action_dim: int,
        action_bound: float,
        bound_action_method: str,
        log_sigma: Optional[float] = None,
        trainable_sigma: Optional[bool] = True,
        log_sigma_bound: Optional[Tuple[int]] = (-20, 2)
    ):
        super().__init__()

        self.preprocess_net = preprocess_net
        self.action_bound = action_bound
        self.bound_action_method = bound_action_method
        self.log_sigma = log_sigma
        self.log_sigma_bound = log_sigma_bound

        self.mu = nn.Linear(hidden_size, action_dim)

        if log_sigma is None:
            self.sigma = nn.Linear(hidden_size, action_dim)
        else:
            self.sigma = nn.Parameter(log_sigma * torch.ones(action_dim), requires_grad = trainable_sigma)
            
        orthogonal_init(self.preprocess_net, gain = np.sqrt(2))
        orthogonal_init(self.mu, gain = np.sqrt(2) * 1e-2)

    def compute_dists(self, states: torch.FloatTensor) -> Independent:

        hidden_states = self.preprocess_net(states)
        mus = self.mu(hidden_states)

        if self.log_sigma is None:
            log_sigmas = self.sigma(hidden_states)
        else:
            log_sigmas = self.sigma.expand_as(mus)

        sigmas = log_sigmas.clamp(*self.log_sigma_bound).exp()
        dists = Independent(Normal(mus, sigmas), 1)

        return dists

    def forward(
        self,
        states: torch.FloatTensor,
        deterministic: Optional[bool]
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
        states: torch.FloatTensor,
        actions: torch.FloatTensor
    ) -> Tuple[Independent, torch.FloatTensor]:

        dists = self.compute_dists(states)
        
        log_probs = dists.log_prob(
            (actions / self.action_bound).atanh()
            if self.bound_action_method == "tanh" else actions
        )

        if self.bound_action_method == "tanh":
            log_probs = (log_probs - (1 - actions.pow(2) + torch.finfo(torch.float32).eps).log().sum(-1)) / self.action_bound

        return dists, log_probs
    
    # for SAC
    def compute_actions_and_log_probs(
        self,
        states: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        dists = self.compute_dists(states)
        actions = dists.rsample()

        if self.bound_action_method == "clip":
            actions = actions.clamp(-self.action_bound, self.action_bound)

        log_probs = dists.log_prob(actions)

        if self.bound_action_method == "tanh":
            actions = self.action_bound * actions.tanh()
            log_probs = (log_probs - (1 - actions.pow(2) + torch.finfo(torch.float32).eps).log().sum(-1)) / self.action_bound

        return actions, log_probs


class ContinuousCritic(nn.Module):

    def __init__(self, preprocess_net: nn.Module, hidden_size: int):
        super().__init__()

        self.preprocess_net = preprocess_net
        self.value = nn.Linear(hidden_size, 1)

    def forward(
        self,
        states: torch.FloatTensor,
        actions: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:

        if actions is not None:
            states = torch.cat((states, actions), -1)
        hidden_states = self.preprocess_net(states)
        values = self.value(hidden_states)

        return values
