import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from collections import OrderedDict
from agent import Agent
import utils


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, pruning_algo=None):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.critic = critic_cfg.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = actor_cfg.to(self.device)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        # -------------------------
        self.pruning_algo = pruning_algo
        if self.pruning_algo in ['rlx2', 'rigl']:
            from DST.DST_Scheduler import DST_Scheduler
            from DST.utils import get_W
            zeta = 0.5              # Initial mask update ratio
            random_grow = False     #Use random grow scheme
            static_actor = False    # Fix the topology of actor
            static_critic = False   # Fix the topology of critic
            delta = 10000           # Mask update interval

            if self.pruning_algo == 'rlx2':
                self.rlx2_nstep = 3
                self.rlx2_delay_nstep = 30000

            self.actor_pruner = DST_Scheduler(model=self.actor, optimizer=self.actor_optimizer,
                                              T_end=int(1000000 / self.actor_update_frequency),
                                              static_topo=static_actor, zeta=zeta, delta=delta, random_grow=random_grow)
            self.critic_pruner = DST_Scheduler(model=self.critic, optimizer=self.critic_optimizer,
                                                T_end=1000000, static_topo=static_critic,
                                               zeta=zeta, delta=delta, random_grow=random_grow)
            self.targer_critic_W, _ = get_W(self.critic_target)
        # ------------------------

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, stats):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        stats['train_critic/loss'] = critic_loss.detach().cpu().numpy()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return stats

    def update_actor_and_alpha(self, obs, stats):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        stats['train_actor/loss'] = actor_loss.detach().cpu().numpy()
        stats['train_actor/target_entropy'] = self.target_entropy
        stats['train_actor/entropy'] = -log_prob.mean().detach().cpu().numpy()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return stats

    def update(self, replay_buffer, step):
        stats = {}
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        stats = self.update_critic(obs, action, reward, next_obs, not_done_no_max, stats)
        if step % self.actor_update_frequency == 0:
            stats = self.update_actor_and_alpha(obs, stats)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        return stats

    def calculate_snip_score(self, replay_buffer, itr=1):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size*itr)
        # ----- critic -----
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # ----  actor  -----
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()


    def update_rigl(self, replay_buffer, step=0, batch_size=256):
        if self.pruning_algo == 'rlx2':
            current_nstep = self.rlx2_nstep if step >= self.rlx2_delay_nstep else 1
        else:
            current_nstep = 1
        state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.sample(batch_size, current_nstep)

        with torch.no_grad():
            accum_reward = torch.zeros(reward[:, 0].shape).to(self.device)
            have_not_done = torch.ones(not_done[:, 0].shape).to(self.device)
            have_not_reset = torch.ones(not_done[:, 0].shape).to(self.device)
            modified_n = torch.zeros(not_done[:, 0].shape).to(self.device)
            nstep_next_action = torch.zeros(action[:, 0].shape).to(self.device)
            for k in range(current_nstep):
                accum_reward += have_not_reset * have_not_done * self.discount ** k * reward[:, k]
                have_not_done *= torch.maximum(not_done[:, k], 1 - have_not_reset)
                dist = self.actor(next_state[:, k])
                next_action = dist.rsample()
                nstep_next_action += have_not_reset * have_not_done * (next_action - nstep_next_action)
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
                accum_reward += have_not_reset * have_not_done * self.discount ** (k + 1) * (
                            - self.alpha.detach() * log_prob)
                if k == current_nstep - 1:
                    break
                have_not_reset *= (1 - reset_flag[:, k])
                modified_n += have_not_reset
            modified_n = modified_n.type(torch.long)
            nstep_next_state = next_state[np.arange(batch_size), modified_n[:, 0]]
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(nstep_next_state, nstep_next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            if current_nstep == 1:
                target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(
                    target_Q.shape) * self.discount * target_Q
            else:
                target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(
                    target_Q.shape) * self.discount ** (modified_n + 1) * target_Q

        if step % self.critic_target_update_frequency == 0:
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state[:, 0], action[:, 0])
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Delayed policy updates
        if step % self.actor_update_frequency == 0:
            dist = self.actor(state[:, 0])
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_Q1, actor_Q2 = self.critic(state[:, 0], action)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        # Update the frozen target models
        if step % self.critic_target_update_frequency == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)

            for w, mask in zip(self.targer_critic_W, self.critic_pruner.backward_masks):
                w.data *= mask

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step)))
