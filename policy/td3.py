
from copy import deepcopy


import torch
import torch.nn as nn

from utils import soft_update

import wandb

import warnings
warnings.filterwarnings('ignore')

class TD3(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        gamma: float = 0.99,
        tau: float = 5e-3,
        lr: float = 1e-3,
        update_per_collect: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        device: str = 'cpu'
    ):
        super().__init__()

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.gamma = gamma
        self.tau = tau
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.n_update = 0
        self.actor_update_freq = actor_update_freq
        self.device = device

        self.target_actor = deepcopy(actor)
        self.target_critic1 = deepcopy(critic1)
        self.target_critic2 = deepcopy(critic2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr = lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr = lr
        )

    def learn(self, buffer):

        for _ in range(self.update_per_collect):

            states, actions, rewards, next_states, terminated, _ = buffer.sample(batch_size = self.batch_size, device = self.device)

            # compute td target
            with torch.no_grad():
                next_actions = self.actor(next_states) # ignore noise clip
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)

            td_target = rewards + self.gamma * torch.logical_not(terminated) * torch.min(target_q1, target_q2)

            for params in self.critic1.parameters():
                params.requires_grad = True

            # compute critic loss
            pred_q1 = self.critic1(states, actions)
            pred_q2 = self.critic2(states, actions)
            critic_loss = (pred_q1 - td_target).pow(2).mean() + (pred_q2 - td_target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            wandb.log({'critic loss': critic_loss.item()})

            if self.n_update % self.actor_update_freq == 0:

                for params in self.critic1.parameters():
                    params.requires_grad = False

                # compute actor loss
                actions = self.actor(states, deterministic = True)
                values = self.critic1(states, actions)
                actor_loss = - values.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                soft_update(self.tau, self.actor, self.target_actor)
                soft_update(self.tau, self.critic1, self.target_critic1)
                soft_update(self.tau, self.critic2, self.target_critic2)

                wandb.log({
                    'actor loss': actor_loss.item(),
                    'critic loss': critic_loss.item()
                })

            self.n_update += 1