import torch

from lrl.policy import SAC

from lrl.utils import soft_update

import wandb

import warnings
warnings.filterwarnings("ignore")

class DiscreteSAC(SAC):

    def learn(self, buffer):

        for _ in range(self.update_per_collect):

            states, actions, rewards, next_states, terminated, _ = buffer.sample(batch_size = self.batch_size, device = self.device)

            # compute td target
            with torch.no_grad():
                next_probs = self.actor.compute_probs(states)
                target_q1 = (next_probs * self.target_critic1(next_states)).sum(-1).unsqueeze(-1)
                target_q2 = (next_probs * self.target_critic2(next_states)).sum(-1).unsqueeze(-1)

            td_target = rewards + self.gamma * torch.logical_not(terminated) * (torch.min(target_q1, target_q2) - self.alpha * (next_probs * next_probs.log()).sum(-1))

            for params in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                params.requires_grad = True

            # compute critic loss
            pred_q1 = self.critic1(states, actions)
            pred_q2 = self.critic2(states, actions)
            critic_loss = (pred_q1 - td_target).pow(2).mean() + (pred_q2 - td_target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for params in list(self.critic1.parameters()) + list(self.critic2.parameters()):
                params.requires_grad = False

            # compute actor loss
            probs = self.actor.compute_probs(states)
            q1 = (probs * self.critic1(states)).sum(-1).unsqueeze(-1)
            q2 = (probs * self.critic2(states)).sum(-1).unsqueeze(-1)
            actor_loss = (self.alpha * (probs * probs.log()).sum(-1) - torch.min(q1, q2)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.tau, self.critic1, self.target_critic1)
            soft_update(self.tau, self.critic2, self.target_critic2)

            wandb.log({
                "actor loss": actor_loss.item(),
                "critic loss": critic_loss.item()
            })