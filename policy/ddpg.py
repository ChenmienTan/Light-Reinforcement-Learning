
from copy import deepcopy

import torch
import torch.nn as nn
import wandb

from utils import soft_update

import warnings
warnings.filterwarnings('ignore')


class DDPG(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        gamma: float = 0.99,
        tau: float = 5e-3,
        lr: float = 1e-3,
        update_per_collect: int = 1,
        batch_size: int = 256,
        device: str = 'cpu'
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.tau = tau
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.device = device

        self.target_actor = deepcopy(actor)
        self.target_critic = deepcopy(critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr = lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr = lr
        )

    def learn(self, buffer):

        for _ in range(self.update_per_collect):

            states, actions, rewards, next_states, terminated, _ = buffer.sample(batch_size = self.batch_size, device = self.device)

            # compute td_target
            with torch.no_grad():
                next_actions = self.target_actor(next_states, deterministic = True)
                target_q = self.target_critic(next_states, next_actions)

            td_target = rewards + self.gamma * torch.logical_not(terminated) * target_q

            for params in self.critic.parameters():
                params.requires_grad = True

            # compute critic loss
            pred_q = self.critic(states, actions)
            critic_loss = (pred_q - td_target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for params in self.critic.parameters():
                params.requires_grad = False

            # compute actor loss
            actions = self.actor(states, deterministic = True)
            values = self.critic(states, actions)
            actor_loss = - values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.tau, self.actor, self.target_actor)
            soft_update(self.tau, self.critic, self.target_critic)

            wandb.log({
                'actor loss': actor_loss.item(),
                'critic loss': critic_loss.item()
            })


