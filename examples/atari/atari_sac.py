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
    DiscreteCritic,
    train
)
from lrl.policy import DiscreteSAC


class PreprocessNet(nn.Module):

    def __init__(self, net):
        super().__init__()

        self.net = net

    def forward(self, states: torch.tensor):
        states = states.transpose(1, -1) / 225

        return self.net(states)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type = str, default = "breakout")
    parser.add_argument("--name", type = str, default = "sac")

    parser.add_argument("--env-name", type = str, default = "ALE/Breakout-v5")
    parser.add_argument("--n-train-envs", type = int, default = 10)
    parser.add_argument("--n-test-envs", type = int, default = 10)

    parser.add_argument("--alpha", type = float, default = 0.05)
    parser.add_argument("--gamma", type = float, default = 0.99)
    parser.add_argument("--tau", type = float, default = 5e-3)

    parser.add_argument("--channels", type = Sequence[int], default = [3, 32, 64, 64])
    parser.add_argument("--kernel-sizes", type = Sequence[int], default = [8, 4, 3])
    parser.add_argument("--strides", type = Sequence[int], default = [4, 2, 1])
    parser.add_argument("--paddings", type = Sequence[int], default = [0, 0, 0])
    parser.add_argument("--hidden-size", type = int, default = 512)
    parser.add_argument("--activation-fn", type = nn.Module, default = nn.ReLU)
    parser.add_argument("--batch-size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 1e-5)

    parser.add_argument("--buffer-size", type = int, default = 100000)
    parser.add_argument("--n-epochs", type = int, default = 1000)
    parser.add_argument("--collect-per-epoch", type = int, default = 1000)                          
    parser.add_argument("--step-per-collect", type = int, default = 1)
    parser.add_argument("--update-per-collect", type = int, default = 1)

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
    ).to(args.device)

    preprocess_net = PreprocessNet(preprocess_net).to(args.device)

    actor = DiscreteActor(
        preprocess_net,
        args.hidden_size,
        action_dim
    ).to(args.device)

    critic1 = DiscreteCritic(
        preprocess_net,
        args.hidden_size,
        action_dim
    ).to(args.device)

    critic2 = DiscreteCritic(
        preprocess_net,
        args.hidden_size,
        action_dim
    ).to(args.device)

    policy = DiscreteSAC(
        actor,
        critic1,
        critic2,
        gamma = args.gamma,
        alpha = args.alpha,
        tau = args.tau,
        lr = args.lr,
        update_per_collect = args.update_per_collect,
        batch_size = args.batch_size,
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