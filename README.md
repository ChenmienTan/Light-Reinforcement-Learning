# Light Reinforcement Learning

This repo is a light and flexible deep reinforcement learning framework.
It is consisted of four modules: a vector environment wrapper, a replay buffer, Actor and Critic network components, and a train function.
The structure is quite simple so that one should be able to easily modify it for his or her own purpose.
For your reference and convenience, we implemented some baselines, including [PPO](https://arxiv.org/pdf/1707.06347.pdf), [DDPG](https://arxiv.org/pdf/1509.02971.pdf), [TD3](https://arxiv.org/pdf/1802.09477.pdf), and [SAC](https://arxiv.org/pdf/1812.05905.pdf) as well as their instances in MuJoCo and Atari environments.

## Usage

* Open Anaconda Prompt and execute  `cd path`, where "path" is the folder path of this repo.
* Execute `pip install -r requirements.txt` and `pip install dist/lrl-1.0-py3-none-any.whl`.