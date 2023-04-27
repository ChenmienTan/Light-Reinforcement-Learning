from typing import Sequence

import argparse
import gymnasium as gym

import torch
import torch.nn as nn
import wandb

from lrl.utils import (
    Envs,
    Buffer,
    MLP,
    GaussianActor,
    train
)
from lrl.policy import PPO


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type = str, default = "halfcheetah")
    parser.add_argument("--name", type = str, default = "ppo")

    parser.add_argument("--env-name", type = str, default = "HalfCheetah-v4")
    parser.add_argument("--n-train-envs", type = int, default = 1)
    parser.add_argument("--n-test-envs", type = int, default = 10)

    parser.add_argument("--gamma", type = float, default = 0.99)
    parser.add_argument("--gae-lambda", type = float, default = 0.95)

    parser.add_argument("--log-sigma", type = float, default = -0.5)
    parser.add_argument("--bound-action-method", type = str, default = "tanh", choices = ["clip", "tanh"])

    parser.add_argument("--hidden-sizes", type = Sequence[int], default = [64, 64])
    parser.add_argument("--activation-fn", type = nn.Module, default = nn.Tanh)
    parser.add_argument("--batch-size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 1e-3)

    parser.add_argument("--n-epochs", type = int, default = 500)
    parser.add_argument("--collect-per-epoch", type = int, default = 1)
    parser.add_argument("--step-per-collect", type = int, default = 2048)
    parser.add_argument("--update-per-collect", type = int, default = 10)

    parser.add_argument("--norm-advantages", type = bool, default = True)
    parser.add_argument("--recompute-advantages", type = bool, default = True)

    parser.add_argument("--clip-eps", type = float, default = 0.2)
    parser.add_argument("--vf-coef", type = float, default = 0.5)
    parser.add_argument("--ent-coef", type = float, default = 1e-2)
    parser.add_argument("--max-grad-norm", type = float, default = 0.5)

    parser.add_argument("--norm-states", type = bool, default = True)
    parser.add_argument("--schedule_lr", type = bool, default = True)

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
        args.step_per_collect
    )

    actor_preprocess_net = MLP(
        input_size = state_dim[0],
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn
    ).to(args.device)

    critic_preprocess_net = MLP(
        input_size = state_dim[0],
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn
    )

    actor = GaussianActor(
        actor_preprocess_net,
        args.hidden_sizes[-1],
        action_dim,
        action_bound,
        args.bound_action_method,
        args.log_sigma
    ).to(args.device)

    critic = nn.Sequential(
        critic_preprocess_net,
        nn.Linear(args.hidden_sizes[-1], 1)
    ).to(args.device)

    policy = PPO(
        actor,
        critic,
        gamma = args.gamma,
        gae_lambda = args.gae_lambda,
        lr = args.lr,
        n_collects = args.n_epochs * args.collect_per_epoch,
        update_per_collect = args.update_per_collect,
        batch_size = args.batch_size,
        norm_advantages = args.norm_advantages,
        recompute_advantages = args.recompute_advantages,
        schedule_lr = args.schedule_lr,
        clip_eps = args.clip_eps,
        vf_coef = args.vf_coef,
        ent_coef = args.ent_coef,
        max_grad_norm = args.max_grad_norm,
        device = args.device
    ).to(args.device)

    train(
        train_envs = train_envs,
        test_envs = test_envs,
        buffer = buffer,
        policy = policy,
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        step_per_collect = args.step_per_collect,
        norm_states = args.norm_states
    )