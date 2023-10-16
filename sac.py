import argparse
from typing import Optional, Union, Sequence
from copy import deepcopy

import envpool

import torch
import torch.nn as nn

from net import *
from util import *

import wandb

import warnings
warnings.filterwarnings("ignore")


class SAC(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        gamma: Optional[float] = 0.99,
        alpha: Optional[float] = 0.2,
        lr: Optional[float] = 1e-3,
        tau: Optional[float] = 5e-3,
        update_per_collect: Optional[int] = 1,
        batch_size: Optional[int] = 256,
        device: Optional[Union[str, int, torch.device]] = "cpu"
    ):
        super().__init__()

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.device = device

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

            states, actions, rewards, next_states, terminates, _ = buffer.sample()

            # compute td target
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.compute_actions_and_log_probs(next_states)
                target_q1 = self.target_critic1(next_states, next_actions).squeeze(-1)
                target_q2 = self.target_critic2(next_states, next_actions).squeeze(-1)

            td_target = rewards + self.gamma * torch.logical_not(terminates) * (torch.min(target_q1, target_q2) - self.alpha * next_log_probs)

            for params in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                params.requires_grad = True

            # compute critic loss
            pred_q1 = self.critic1(states, actions).squeeze(-1)
            pred_q2 = self.critic2(states, actions).squeeze(-1)
            critic_loss = (pred_q1 - td_target).pow(2).mean() + (pred_q2 - td_target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for params in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                params.requires_grad = False

            # compute actor loss
            actions, log_probs = self.actor.compute_actions_and_log_probs(states)
            q1 = self.critic1(states, actions).squeeze(-1)
            q2 = self.critic2(states, actions).squeeze(-1)
            actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.tau, self.critic1, self.target_critic1)
            soft_update(self.tau, self.critic2, self.target_critic2)

            wandb.log({
                "actor loss": actor_loss.item(),
                "critic loss": critic_loss.item()
            })


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type = str, default = "HalfCheetah-v4")
    parser.add_argument("--project-name", type = str, default = "halfcheetah")
    parser.add_argument("--n-train-envs", type = int, default = 1)
    parser.add_argument("--n-valid-envs", type = int, default = 10)

    parser.add_argument("--bound-action-method", type = str, default = "tanh", choices = ["clip", "tanh"])
    parser.add_argument("--hidden-sizes", type = Sequence[int], default = [256, 256])
    parser.add_argument("--activation-fn", type = nn.Module, default = nn.ReLU)

    parser.add_argument("--buffer-size", type = int, default = 300000)
    parser.add_argument("--batch-size", type = int, default = 256)
    
    parser.add_argument("--n-epochs", type = int, default = 1000)
    parser.add_argument("--collect-per-epoch", type = int, default = 1000)                          
    parser.add_argument("--step-per-collect", type = int, default = 1)
    parser.add_argument("--n-start-steps", type = int, default = 10000)
    
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)
    
    wandb.init(project = args.project_name, name = "sac", config = args)

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
        args.buffer_size,
        args.batch_size,
        args.device
    )

    actor_preprocess_net = MLP(
        state_dim[0],
        args.hidden_sizes,
        args.activation_fn
    )

    critic1_preprocess_net = MLP(
        state_dim[0] + action_dim,
        args.hidden_sizes,
        args.activation_fn
    )

    critic2_preprocess_net = MLP(
        state_dim[0] + action_dim,
        args.hidden_sizes,
        args.activation_fn
    )

    actor = GaussianActor(
        actor_preprocess_net,
        args.hidden_sizes[-1],
        action_dim,
        action_bound,
        args.bound_action_method,
    )

    critic1 = ContinuousCritic(
        critic1_preprocess_net,
        args.hidden_sizes[-1]
    )

    critic2 = ContinuousCritic(
        critic2_preprocess_net,
        args.hidden_sizes[-1]
    )

    policy = SAC(
        actor,
        critic1,
        critic2,
        device = args.device
    ).to(args.device)
    
    trainer = Trainer(
        train_envs,
        valid_envs,
        buffer,
        policy
    )

    trainer.collect(args.n_start_steps, True)

    trainer.run(
        args.n_epochs,
        args.collect_per_epoch,
        args.step_per_collect
    )