import argparse
from typing import Optional, Union, Sequence

import envpool

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from net import *
from util import *

import wandb

import warnings
warnings.filterwarnings("ignore")

class PPO(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        gamma: Optional[float] = 0.99,
        gae_lambda: Optional[float] = 0.95,
        lr: Optional[float] = 3e-4,
        n_collects: Optional[int] = None,
        update_per_collect: Optional[int] = 10,
        batch_size: Optional[int] = 64,
        norm_advantages: Optional[bool] = True,
        recompute_advantages: Optional[bool] = True,
        schedule_lr: Optional[bool] = True,
        clip_eps: Optional[float] = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: Optional[float] = None,
        vf_coef: Optional[float] = 0.5,
        ent_coef: Optional[float] = 1e-2,
        max_grad_norm: Optional[float] = 0.5,
        device: Optional[Union[str, int, torch.device]] = "cpu",
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
        self.dual_clip = dual_clip
        self.value_clip = value_clip
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
                self.optimizer,
                lr_lambda = lambda n_collect: 1 - n_collect / n_collects
            )

    def learn(self, buffer):

        states, actions, rewards, next_states, terminates, truncates = buffer.sample()
        dones = torch.logical_or(terminates, truncates)

        with torch.no_grad():
            _, old_log_probs = self.actor.compute_dists_and_log_probs(states, actions)

        for update in range(self.update_per_collect):

            if update == 0 or self.recompute_advantages:

                with torch.no_grad():
                    old_values = self.critic(states).squeeze(-1)
                    next_values = self.critic(next_states).squeeze(-1)

                deltas = rewards + self.gamma * torch.logical_not(terminates) * next_values - old_values
                advantage = 0
                advantages = torch.zeros(states.shape[0], device = self.device)
                for n in range(states.shape[0] - 1, -1, -1):
                    advantage = torch.logical_not(dones[n]) * self.gamma * self.gae_lambda * advantage + deltas[n]
                    advantages[n] = advantage

                lambda_returns = advantages + old_values

                if self.norm_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + torch.finfo(torch.float32).eps)

            indices = np.random.permutation(states.shape[0])

            for start_indice in range(0, states.shape[0], self.batch_size):
                idx = indices[start_indice: start_indice + self.batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_lambda_returns = lambda_returns[idx]
                batch_old_values = old_values[idx]

                dists, log_probs = self.actor.compute_dists_and_log_probs(batch_states, batch_actions)

                ratio = (log_probs - batch_old_log_probs).exp()
                clamped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_obj1 = ratio * batch_advantages
                actor_obj2 = clamped_ratio * batch_advantages

                if self.dual_clip is not None:
                    actor_obj1 = torch.min(actor_obj1, actor_obj2)
                    actor_obj2 = torch.max(actor_obj1, self.dual_clip * batch_advantages)
                    actor_loss = - torch.where(batch_advantages < 0, actor_obj2, actor_obj1).mean()
                else:
                    actor_loss = - torch.min(actor_obj1, actor_obj2).mean()

                # compute critic loss
                values = self.critic(batch_states).squeeze(-1)

                if self.value_clip is not None:
                    clamped_values = values.clamp(batch_old_values - self.value_clip, batch_old_values + self.value_clip)
                    critic_obj1 = (values - batch_lambda_returns).pow(2)
                    critic_obj2 = (clamped_values - batch_lambda_returns).pow(2)
                    critic_loss = torch.max(critic_obj1, critic_obj2).mean()
                else:
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
            
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type = str, default = "Hopper-v4")
    parser.add_argument("--n-train-envs", type = int, default = 1)
    parser.add_argument("--n-valid-envs", type = int, default = 10)

    parser.add_argument("--log-sigma", type = float, default = -0.5)
    parser.add_argument("--bound-action-method", type = str, default = "tanh", choices = ["clip", "tanh"])
    parser.add_argument("--hidden-sizes", type = Sequence[int], default = [64, 64])
    parser.add_argument("--activation-fn", type = nn.Module, default = nn.Tanh)

    parser.add_argument("--n-epochs", type = int, default = 500)
    parser.add_argument("--collect-per-epoch", type = int, default = 1)
    parser.add_argument("--step-per-collect", type = int, default = 2048)

    args = parser.parse_args()
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    wandb.init(project = "hopper", name = "ppo", config = args)

    train_envs = envpool.make(
        args.env_name,
        "gymnasium",
        num_envs = args.n_train_envs
    )
    
    valid_envs = envpool.make(
        args.env_name,
        "gymnasium",
        num_envs = args.n_valid_envs
    )

    state_dim = train_envs.observation_space.shape
    action_dim = train_envs.action_space.shape[0]
    action_bound = train_envs.action_space.high[0]

    buffer = Buffer(
        state_dim,
        action_dim,
        args.n_train_envs,
        args.step_per_collect,
        device = device
    )

    actor_preprocess_net = MLP(
        state_dim[0],
        args.hidden_sizes,
        args.activation_fn
    )

    critic_preprocess_net = MLP(
        state_dim[0],
        args.hidden_sizes,
        args.activation_fn
    )

    actor = GaussianActor(
        actor_preprocess_net,
        args.hidden_sizes[-1],
        action_dim,
        action_bound,
        args.bound_action_method,
        args.log_sigma
    )

    critic = ContinuousCritic(
        critic_preprocess_net,
        args.hidden_sizes[-1]
    )

    policy = PPO(
        actor,
        critic,
        n_collects = args.n_epochs * args.collect_per_epoch,
        device = device
    ).to(device)

    Trainer(
        train_envs,
        valid_envs,
        buffer,
        policy,
        True
    ).run(
        args.n_epochs,
        args.collect_per_epoch,
        args.step_per_collect
    )