import numpy as np
import torch
import os
import tensorflow as tf
import gzip

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.ux = 0

    def __len__(self):
        return self.capacity if self.full else self.idx


    def insert_from_dataset(self, filename):
        with tf.io.gfile.GFile(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                dataset = np.load(outfile, allow_pickle=True).item()
        self.obses = np.array(dataset['observations'], dtype=np.float32)
        self.next_obses = np.array(dataset['next_observations'], dtype=np.float32)
        self.rewards = np.array(dataset['rewards'], dtype=np.float32).reshape(-1, 1)
        self.not_dones = 1 - np.array(dataset['terminals'], dtype=np.float32).reshape(-1, 1)
        self.not_dones_no_max = 1 - np.array(dataset['terminals'], dtype=np.float32).reshape(-1, 1)
        self.actions = np.array(dataset['actions'], dtype=np.float32)

        self.idx = self.obses.shape[0]

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.ux+=1

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


    def save(self, env, data_type):
        storage = {}
        # indicates already collected 1M data and now self.idx replacing new data from top
        if self.ux>=self.capacity:
            storage['observations'] = self.obses
            storage['actions'] = self.actions
            storage['next_observations'] = self.next_obses
            storage['rewards'] = self.rewards
            storage['terminals'] = 1-self.not_dones
        else:
            storage['observations'] = self.obses[0:self.idx]
            storage['actions'] = self.actions[0:self.idx]
            storage['next_observations'] = self.next_obses[0:self.idx]
            storage['rewards'] = self.rewards[0:self.idx]
            storage['terminals'] = 1-self.not_dones[0:self.idx]


        # add info max return, min return
        data_dir = f"../expert_dataset/{data_type}/{env}"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = f'{data_dir}/es_0_{self.idx}'
        with tf.io.gfile.GFile(filename, 'wb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                np.save(outfile, storage)


class ReplayBuffer_VEnv(object):
    """Buffer to store environment transitions."""
    def __init__(self, num_tasks, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 #if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, num_tasks), dtype=np.float32)
        self.not_dones = np.empty((capacity, num_tasks), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, num_tasks), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.ux = 0

    def __len__(self):
        return self.capacity if self.full else self.idx


    def insert_from_dataset(self, filename):
        with tf.io.gfile.GFile(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                dataset = np.load(outfile, allow_pickle=True).item()
        self.obses = np.array(dataset['observations'], dtype=np.float32)
        self.next_obses = np.array(dataset['next_observations'], dtype=np.float32)
        self.rewards = np.array(dataset['rewards'], dtype=np.float32).reshape(-1, 1)
        self.not_dones = 1 - np.array(dataset['terminals'], dtype=np.float32).reshape(-1, 1)
        self.not_dones_no_max = 1 - np.array(dataset['terminals'], dtype=np.float32).reshape(-1, 1)
        self.actions = np.array(dataset['actions'], dtype=np.float32)

        self.idx = self.obses.shape[0]

    def add(self, obs, action, reward, next_obs, not_done, not_done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not_done)
        np.copyto(self.not_dones_no_max[self.idx], not_done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.ux += 1

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float().transpose(0, 1) # (batch, task, obs) -- > (task, batch, obs)
        actions = torch.as_tensor(self.actions[idxs], device=self.device).transpose(0, 1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).transpose(0, 1).unsqueeze(-1)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float().transpose(0, 1)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).transpose(0, 1).unsqueeze(-1)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device).transpose(0, 1).unsqueeze(-1)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


    def save(self, env, data_type):
        storage = {}
        # indicates already collected 1M data and now self.idx replacing new data from top
        if self.ux>=self.capacity:
            storage['observations'] = self.obses
            storage['actions'] = self.actions
            storage['next_observations'] = self.next_obses
            storage['rewards'] = self.rewards
            storage['terminals'] = 1-self.not_dones
        else:
            storage['observations'] = self.obses[0:self.idx]
            storage['actions'] = self.actions[0:self.idx]
            storage['next_observations'] = self.next_obses[0:self.idx]
            storage['rewards'] = self.rewards[0:self.idx]
            storage['terminals'] = 1-self.not_dones[0:self.idx]


        # add info max return, min return
        data_dir = f"../expert_dataset/{data_type}/{env}"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if data_type == 'replay':
            filename = f'{data_dir}/replay_buffer'
        else:filename = f'{data_dir}/es_0_{self.idx}'
        with tf.io.gfile.GFile(filename, 'wb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                np.save(outfile, storage)