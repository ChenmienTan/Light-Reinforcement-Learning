from typing import Sequence

import argparse
import gymnasium as gym

import torch
import torch.nn as nn
import wandb

from lrl.utils import (
    Envs,
    Buffer,
    ConvNet,
    DiscreteActor,
    train
)
from lrl.policy import PPO


class PreprocessNet(nn.Module):

    def __init__(self, net: nn.Module):
        super().__init__()

        self.net = net

    def forward(self, states: torch.Tensor):
        states = states.transpose(1, -1) / 225

        return self.net(states)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type = str, default = "breakout")
    parser.add_argument("--name", type = str, default = "ppo")

    parser.add_argument("--env-name", type = str, default = "ALE/Breakout-v5")
    parser.add_argument("--n-train-envs", type = int, default = 10)
    parser.add_argument("--n-test-envs", type = int, default = 10)

    parser.add_argument("--gamma", type = float, default = 0.99)
    parser.add_argument("--gae-lambda", type = float, default = 0.95)

    parser.add_argument("--channels", type = Sequence[int], default = [3, 32, 64, 64])
    parser.add_argument("--kernel-sizes", type = Sequence[int], default = [8, 4, 3])
    parser.add_argument("--strides", type = Sequence[int], default = [4, 2, 1])
    parser.add_argument("--paddings", type = Sequence[int], default = [0, 0, 0])
    parser.add_argument("--hidden-size", type = int, default = 512)
    parser.add_argument("--activation-fn", type = nn.Module, default = nn.ReLU)
    parser.add_argument("--batch-size", type = int, default = 256)
    parser.add_argument("--lr", type = float, default = 3e-4)

    parser.add_argument("--n-epochs", type = int, default = 1000)
    parser.add_argument("--collect-per-epoch", type = int, default = 10)
    parser.add_argument("--step-per-collect", type = int, default = 100)
    parser.add_argument("--update-per-collect", type = int, default = 4)
    
    parser.add_argument("--norm-advantages", type = bool, default = True)
    parser.add_argument("--recompute-advantages", type = bool, default = False)

    parser.add_argument("--clip-eps", type = float, default = 0.1)
    parser.add_argument("--vf-coef", type = float, default = 0.25)
    parser.add_argument("--ent-coef", type = float, default = 1e-2)
    parser.add_argument("--max-grad-norm", type = float, default = 0.5)

    parser.add_argument("--schedule_lr", type = bool, default = True)

    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    wandb.init(project = args.project, name = args.name, config = args)

    train_envs = Envs([gym.make(args.env_name) for _ in range(args.n_train_envs)])
    test_envs = Envs([gym.make(args.env_name) for _ in range(args.n_test_envs)])

    state_dim = train_envs[0].observation_space.shape
    action_dim = train_envs[0].action_space.n

    buffer = Buffer(
        state_dim,
        1,
        args.n_train_envs,
        args.step_per_collect,
        action_dtype = torch.int64
    )

    preprocess_net = ConvNet(
        channels = args.channels,
        kernel_sizes = args.kernel_sizes,
        strides = args.strides,
        paddings = args.paddings,
        hidden_size = args.hidden_size,
        activation_fn = args.activation_fn
    )

    preprocess_net = PreprocessNet(preprocess_net)

    actor = DiscreteActor(
        preprocess_net,
        args.hidden_size,
        action_dim
    )

    critic = nn.Sequential(
        preprocess_net,
        nn.Linear(args.hidden_size, 1)
    )

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
        step_per_collect = args.step_per_collect
    )