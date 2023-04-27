from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import wandb

import warnings
warnings.filterwarnings("ignore")

class PPO(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 1e-3,
        n_collects: Optional[int] = None,
        update_per_collect: int = 10,
        batch_size: int = 64,
        norm_advantages: bool = True,
        recompute_advantages: bool = True,
        schedule_lr: bool = False,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 1e-2,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.norm_advantages = norm_advantages
        self.recompute_advantages = recompute_advantages
        self.schedule_lr = schedule_lr
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr = lr
        )

        if schedule_lr:
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda = lambda n_collect: 1 - n_collect / n_collects
            )

    def learn(self, buffer):

        n_transitions = buffer.n_envs * buffer.buffer_size
        states, actions, rewards, next_states, terminated, truncated = buffer.sample(device = self.device)
        dones = torch.logical_or(terminated, truncated)

        with torch.no_grad():
            _, old_log_probs = self.actor.compute_dists_and_log_probs(states, actions)

        for update in range(self.update_per_collect):

            if update == 0 or self.recompute_advantages:

                # compute lambda returns and advantages

                with torch.no_grad():
                    values = self.critic(states)
                    next_values = self.critic(next_states)

                advantage = 0
                advantages = torch.zeros((n_transitions, 1)).to(self.device)
                deltas = rewards + self.gamma * torch.logical_not(terminated) * next_values - values
                for n in range(n_transitions - 1, -1, -1):
                    advantage = torch.logical_not(dones[n]) * self.gamma * self.gae_lambda * advantage + deltas[n]
                    advantages[n] = advantage

                lambda_returns = advantages + values

                if self.norm_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            Indices = np.random.permutation(n_transitions)

            for start_indice in range(0, n_transitions, self.batch_size):
                indices = Indices[start_indice: start_indice + self.batch_size]

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_advantages = advantages[indices]
                batch_lambda_returns = lambda_returns[indices]

                # compute actor loss
                dists, log_probs = self.actor.compute_dists_and_log_probs(batch_states, batch_actions)

                ratio = (log_probs - batch_old_log_probs).exp()
                clamped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                actor_loss = - torch.min(ratio * batch_advantages, clamped_ratio * batch_advantages).mean()

                # compute critic loss
                values = self.critic(batch_states)
                critic_loss = (values - batch_lambda_returns).pow(2).mean()

                # compute entropy loss
                entropy_loss = - dists.entropy().mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm
                    )

                self.optimizer.step()

                wandb.log({
                    "actor loss": actor_loss.item(),
                    "critic loss": critic_loss.item(),
                    "entropy loss": entropy_loss.item(),
                })

        if self.schedule_lr:
            self.scheduler.step()