from typing import Sequence
import argparse

import gymnasium as gym
import torch
import torch.nn as nn

from lrl.utils import (
    Envs,
    Buffer,
    MLP,
    GaussianActor,
    ContinuousCritic,
    Trainer
)
from lrl.policy import TD3


import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type = str, default = "halfcheetah")
    parser.add_argument("--name", type = str, default = "td3")

    parser.add_argument("--env-name", type = str, default = "HalfCheetah-v4")
    parser.add_argument("--n-train-envs", type = int, default = 1)
    parser.add_argument("--n-test-envs", type = int, default = 10)

    parser.add_argument("--gamma", type = float, default = 0.99)
    parser.add_argument("--tau", type = float, default = 5e-3)

    parser.add_argument("--log_sigma", type = float, default = -2.3)
    parser.add_argument("--bound-action-method", type = str, default = "tanh", choices = ["clip", "tanh"])

    parser.add_argument("--hidden-sizes", type = Sequence[int], default = [256, 256])
    parser.add_argument("--activation-fn", type = nn.Module, default = nn.ReLU)
    parser.add_argument("--batch-size", type = int, default = 256)
    parser.add_argument("--lr", type = float, default = 1e-3)

    parser.add_argument("--buffer-size", type = int, default = 300000)
    parser.add_argument("--n-epochs", type = int, default = 1000)
    parser.add_argument("--collect-per-epoch", type = int, default = 1000)                    
    parser.add_argument("--step-per-collect", type = int, default = 1)
    parser.add_argument("--update-per-collect", type = int, default = 1)
    parser.add_argument("--n-start-steps", type = int, default = 25000)
    parser.add_argument("--actor-update-freq", type = int, default = 2)
    
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    wandb.init(project = args.project, name = args.name, config = args)

    train_envs = Envs([gym.make(args.env_name) for _ in range(args.n_train_envs)])
    test_envs = Envs([gym.make(args.env_name) for _ in range(args.n_test_envs)])

    state_dim = train_envs[0].observation_space.shape
    action_dim = train_envs[0].action_space.shape[0]
    action_bound = train_envs[0].action_space.high[0]

    buffer = Buffer(
        state_dim,
        action_dim,
        args.n_train_envs,
        args.buffer_size
    )

    actor_preprocess_net = MLP(
        input_size = state_dim[0],
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn
    )

    critic1_preprocess_net = MLP(
        input_size = state_dim[0] + action_dim,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn
    )

    critic2_preprocess_net = MLP(
        input_size = state_dim[0] + action_dim,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn
    )

    actor = GaussianActor(
        actor_preprocess_net,
        args.hidden_sizes[-1],
        action_dim,
        action_bound,
        args.bound_action_method,
        args.log_sigma,
        trainable_sigma = False
    )

    critic1 = ContinuousCritic(
        critic1_preprocess_net,
        args.hidden_sizes[-1]
    )

    critic2 = ContinuousCritic(
        critic2_preprocess_net,
        args.hidden_sizes[-1]
    )

    policy = TD3(
        actor,
        critic1,
        critic2,
        gamma = args.gamma,
        tau = args.tau,
        lr = args.lr,
        update_per_collect = args.update_per_collect,
        batch_size = args.batch_size,
        actor_update_freq = args.actor_update_freq,
        device = args.device
    ).to(args.device)
    
    trainer = Trainer(
        train_envs = train_envs,
        test_envs = test_envs,
        buffer = buffer,
        policy = policy
    )

    trainer.collect(n_steps = args.n_start_steps, random = True)

    trainer.train(
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        step_per_collect = args.step_per_collect,
    )