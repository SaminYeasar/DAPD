#!/usr/bin/env python3
import numpy as np
import torch.nn.functional as F
import torch
import copy
import os
import time
from video import VideoRecorder
from replay_buffer import ReplayBuffer
import utils
import dmc2gym
import hydra
import gym
import warnings
import collections
from dapd_utils import class_pathways, mask_network, apply_backward_hook, preprocess_for_mask_update, sparsity_level
warnings.filterwarnings('ignore', category=DeprecationWarning)


import wandb
def wandb_init(cfg):
    project = "neural_pathway"
    name = f'agent={cfg.agent_name}_algo={cfg.algo}_env={cfg.env}'
    group = f'kr={cfg.keep_ratio}_seed={cfg.seed}_lr={cfg.lr}'
    wandb.init(
        project=project,
        group=group,
        name=name,
        config={'agent': cfg.agent_name, 'algo': cfg.algo, 'env': cfg.env, 'env_type': cfg.env_type, 'seed': cfg.seed,
                'h_dim': cfg.diag_gaussian_actor.hidden_dim, 'kr': cfg.keep_ratio, 'lr': cfg.lr, 'continual_pruning': cfg.continual_pruning,
                'm_mavg': cfg.mask_update_mavg, 'mask_init': cfg.mask_init_method, 'ips_thrsh': cfg.ips_threshold, 'iterative_pruning': cfg.iterative_pruning}
    )
    wandb.run.save()

def get_mask_stats(mask):
    percent_active = {}
    for l, m in enumerate(mask):
        x, y = m.shape
        active_neurons = torch.sum(m)
        percent_active[f'actor/layer_{l}'] = (round((active_neurons/(x*y)).cpu().data.numpy()*100, 2))
    return percent_active

def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env_type == 'dm_control':
        if cfg.env == 'ball_in_cup_catch':
            domain_name = 'ball_in_cup'
            task_name = 'catch'
        else:
            domain_name = cfg.env.split('_')[0]
            task_name = '_'.join(cfg.env.split('_')[1:])
        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           seed=cfg.seed,
                           visualize_reward=True)
    elif cfg.env_type == 'gym':
        env = gym.make(cfg.env)
    else:
        print('choose correct env')

    env.seed(cfg.seed)
    env.action_scale_high = env.action_space.high.max()
    env.action_scale_low = env.action_space.high.min()
    return env

def make_agent(obs_dim, action_dim, action_range, cfg):
    cfg.obs_dim = obs_dim
    cfg.action_dim = action_dim
    cfg.action_range = action_range
    return hydra.utils.instantiate(cfg)

class Workspace(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # set workdir
        self.set_work_dir()
        # set seed
        self.set_seed()

        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        self.agent = make_agent(self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())],
                                self.cfg.agent)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)
        self.tmp_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(max(self.env._max_episode_steps, self.cfg.batch_size)),
                                          self.device)
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0


    def set_agent(self):
        self.agent = make_agent(self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())],
                                self.cfg.agent)
    def set_dummy_agent(self):
        self.dummy_agent = make_agent(self.env.observation_space.shape[0],
                                self.env.action_space.shape[0],
                                [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())],
                                self.cfg.agent)
        self.dummy_agent.actor.load_state_dict(copy.deepcopy(self.agent.actor.state_dict()))
        self.dummy_agent.critic.load_state_dict(copy.deepcopy(self.agent.critic.state_dict()))
        self.dummy_agent.critic_target.load_state_dict(copy.deepcopy(self.agent.critic_target.state_dict()))


    def set_work_dir(self):
        self.work_dir = os.getcwd()

    def set_seed(self):
        utils.set_seed_everywhere(self.cfg.seed)

    def reset_episodic_storage(self):
        self.storage = {'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'next_observations': [], 'episodic_returns': [], 'success': []}



    def evaluate(self):
        self.reset_episodic_storage()
        average_episode_reward = 0
        average_episode_len = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_len = 0
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not (done or episode_len >= self.env._max_episode_steps):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                self.storage['observations'].append(obs.astype(np.float32))
                self.storage['actions'].append(action.astype(np.float32))
                obs, reward, done, info = self.env.step(action*self.env.action_scale_high)
                if (episode_len + 1) == self.env._max_episode_steps:
                    done = True
                self.storage['next_observations'].append(obs.astype(np.float32))
                self.storage['rewards'].append(reward)
                self.storage['terminals'].append(int(done))
                self.video_recorder.record(self.env)
                episode_reward += reward
                average_episode_len += 1
                episode_len += 1
            self.storage['episodic_returns'].append(episode_reward)
            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_len /= self.cfg.num_eval_episodes
        eval_stats = {"eval/episode_reward": average_episode_reward, "eval/episode_len": average_episode_len}
        print('Eval at train step:', self.step, eval_stats)
        return eval_stats


    def quick_collect(self):
        done = False
        obs = self.env.reset()
        self.agent.reset()
        episode_step = 1
        episode_rw = 0
        episodes = 10
        timestep = 0
        while timestep < 5000:
            while not done:
                # sample action:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                # take a step
                next_obs, reward, done, _ = self.env.step(action*self.env.action_scale_high)
                if episode_step + 1 == self.env._max_episode_steps: done = True
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
                # collect samples
                self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
                obs = next_obs
                episode_step +=1
                episode_rw += reward
                timestep += 1
            episode_step = 0
            done = False
            obs = self.env.reset()
        print(f'expert_reward {episode_rw/episodes}')

    def prune_network(self):
        # (1) uses random expert samples
        self.quick_collect()
        # find masks
        masks = collections.defaultdict(dict)
        # (2)
        self.set_dummy_agent()
        self.pathway = class_pathways(self.cfg.keep_ratio, history_len=self.cfg.mask_update_mavg)
        masks["actor"][f'task0'], masks["critic"][f'task0'] = self.pathway.get_masks(self.dummy_agent, self.replay_buffer, itr=1)
        # load mask
        self.mask_agent(masks)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(self.cfg.replay_buffer_capacity),
                                          self.device)
        mask_stats = {'train/actor_sparsity': sparsity_level(masks['actor']['task0']),
                      'train/critic_sparsity': sparsity_level(masks['actor']['task0'])}
        print(mask_stats)
        return mask_stats

    def mask_agent(self, keep_masks):
        # unhook
        preprocess_for_mask_update(self.agent.actor)
        preprocess_for_mask_update(self.agent.critic)
        preprocess_for_mask_update(self.agent.critic_target)
        # for forward
        mask_network(self.agent.actor, keep_masks["actor"][f'task0'])
        mask_network(self.agent.critic, keep_masks["critic"][f'task0'])
        mask_network(self.agent.critic_target, keep_masks["critic"][f'task0'])
        # for backward
        apply_backward_hook(self.agent.actor, keep_masks["actor"]['task0'], fixed_weight=-1)
        apply_backward_hook(self.agent.critic, keep_masks["critic"]['task0'], fixed_weight=-1)
        self.layer_activation = get_mask_stats(keep_masks["actor"][f'task0'])
        wandb.log(self.layer_activation, self.step)


    def dapd_mask_update(self):
        return (self.cfg.iterative_pruning and self.step > 0)

    def rlx2_buffer_update(self):
        if self.cfg.use_dynamic_buffer and (self.step + 1) % self.cfg.buffer_adjustment_interval == 0:
            if self.replay_buffer.size == self.replay_buffer.max_size:
                ind = (self.replay_buffer.ptr + np.arange(8 * self.cfg.agent.batch_size)) % self.replay_buffer.max_size
            else:
                ind = (self.replay_buffer.left_ptr + np.arange(
                    8 * self.cfg.agent.batch_size)) % self.replay_buffer.max_size
            batch_state = torch.FloatTensor(self.replay_buffer.state[ind]).to(self.cfg.device)
            batch_action_mean = torch.FloatTensor(self.replay_buffer.action_mean[ind]).to(self.cfg.device)
            with torch.no_grad():
                current_action = self.agent.actor(batch_state).mean
                distance = F.mse_loss(current_action, batch_action_mean) / 2
            if distance > self.cfg.buffer_threshold and self.replay_buffer.size > self.cfg.buffer_min_size:
                self.replay_buffer.shrink()

    def train(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        self.activate_eval = False

        while self.step < self.cfg.num_train_steps:
            if done:
                episode += 1
                wandb.log({'train/episode_reward': episode_reward,
                           'train/episode': episode,
                           'train/duration': time.time() - start_time},
                          self.step)
                # ----------------
                # Evaluate
                # ----------------
                if self.activate_eval:
                    self.activate_eval = False
                    eval_stats = self.evaluate()
                    wandb.log(eval_stats, step=self.step)
                # ----------------
                # Reset environment
                # ----------------
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
            # run training update
            if self.step >= self.cfg.num_seed_steps:
                train_stats = self.agent.update(self.replay_buffer, self.step)
                wandb.log(train_stats, step=self.step)
            # take environment step
            next_obs, reward, done, info = self.env.step(action*self.env.action_scale_high)
            if episode_step + 1 == self.env._max_episode_steps:
                done = True
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            # add collect data to replay
            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
            self.tmp_buffer.add(obs, action, reward, next_obs, done, done_no_max)
            obs = next_obs
            episode_step += 1
            self.step += 1
            if self.step % self.cfg.eval_frequency == 0:
                self.activate_eval = True

        # save the recorder only on the last eval
        self.video_recorder = VideoRecorder(self.work_dir)
        self.agent.save(self.work_dir, step='final')


    def train_DAPD(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        self.activate_eval = False

        while self.step < self.cfg.num_train_steps:
            if done:
                episode += 1
                wandb.log({'train/episode_reward': episode_reward,
                           'train/episode': episode,
                           'train/duration': time.time() - start_time},
                          self.step)
                # ----------------
                # Evaluate
                # ----------------
                if self.activate_eval:
                    self.activate_eval = False
                    eval_stats = self.evaluate()
                    wandb.log(eval_stats, step=self.step)
                    # DAPD: check if reached threshold, TH performance:
                    if (not self.cfg.continual_pruning and eval_stats["eval/episode_reward"] > self.cfg.ips_threshold):
                        self.cfg.iterative_pruning = False
                # ----------------
                # Reset environment
                # ----------------
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                # ---------------
                # DAPD mask update
                # ---------------
                if (self.cfg.algo == 'DAPD' and self.dapd_mask_update()):
                    masks = collections.defaultdict(dict)
                    self.set_dummy_agent()
                    masks["actor"][f'task0'], masks["critic"][f'task0'] = self.pathway.get_masks(self.dummy_agent, self.tmp_buffer) # important: to use the most recent experience, self.tmp_buffer
                    self.mask_agent(masks)
                    mask_stats = {'train/actor_sparsity': sparsity_level(masks['actor']['task0']),
                                  'train/critic_sparsity': sparsity_level(masks['actor']['task0'])}
                    wandb.log(mask_stats, step=self.step)


            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                train_stats = self.agent.update(self.replay_buffer, self.step)
                wandb.log(train_stats, step=self.step)

            # take environment step
            next_obs, reward, done, info = self.env.step(action*self.env.action_scale_high)
            if episode_step + 1 == self.env._max_episode_steps:
                done = True
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            # add collect data to replay
            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
            self.tmp_buffer.add(obs, action, reward, next_obs, done, done_no_max)
            obs = next_obs
            episode_step += 1
            self.step += 1
            if self.step % self.cfg.eval_frequency == 0:
                self.activate_eval = True

        # save the recorder only on the last eval
        self.video_recorder = VideoRecorder(self.work_dir)
        self.agent.save(self.work_dir, step='final')

    def train_rlx2_rigl(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        self.activate_eval = False

        while self.step < self.cfg.num_train_steps:
            if done:
                episode += 1
                wandb.log({'train/episode_reward': episode_reward,
                           'train/episode': episode,
                           'train/duration': time.time() - start_time},
                          self.step)
                # ----------------
                # Evaluate
                # ----------------
                if self.activate_eval:
                    self.activate_eval = False
                    eval_stats = self.evaluate()
                    wandb.log(eval_stats, step=self.step)
                # ----------------
                # Reset environment
                # ----------------
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0


            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                assert self.cfg.algo in ['rlx2', 'rigl']
                train_stats = self.agent.update_rigl(self.replay_buffer, step=self.step)
                wandb.log(train_stats, step=self.step)

            # take environment step
            next_obs, reward, done, info = self.env.step(action * self.env.action_scale_high)
            if episode_step + 1 == self.env._max_episode_steps:
                done = True
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            # add collect data to replay
            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
            obs = next_obs
            episode_step += 1
            self.step += 1
            if self.step % self.cfg.eval_frequency == 0:
                self.activate_eval = True

            # Applies only for RLx2
            if self.cfg.algo == 'rlx2':
                self.rlx2_buffer_update()

        # save the recorder only on the last eval
        self.video_recorder = VideoRecorder(self.work_dir)
        self.agent.save(self.work_dir, step='final')
        wandb.finish()


os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(config_path='./config', config_name='train')
def main(cfg):
    from main import Workspace as W
    workspace = W(cfg)
    wandb_init(cfg)

    if cfg.algo == 'dense':
        workspace.train()

    elif cfg.algo == 'DAPD':
        workspace.prune_network()
        workspace.train_DAPD()

    elif cfg.algo in ['rlx2', 'rigl']:
        workspace.train_rlx2_rigl()


if __name__ == '__main__':
   main()
