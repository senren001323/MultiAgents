import torch
import numpy as np
import json
import random
import utils
from train import trainer
from custom_envs.ThreatClear import ThreatClear
from policy.policy import MADDPG, MADDPGForDiscrete


class MADDPGTrain:

    def __init__(self, 
                 env):
        with open('./train/config.json', 'r') as file:
            config = json.load(file)

        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.num_episodes = config['num_episodes']
        self.hidden_dim = config['hidden_dim']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.sigma = config['sigma']
        self.buffer_size = config['buffer_size']
        self.minimal_size = config['minimal_size']
        self.batch_size = config['batch_size']
        self.update_interval = config['update_interval']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seeds()
        self.replybuffer = utils.ReplayBuffer(self.buffer_size)

        self.env = env
        self.num_agents = self.env.NUM_DRONES

    def set_seeds(self):
        random.seed(44)
        np.random.seed(44)
        torch.manual_seed(44)

    def learn(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.num_episodes

        state_dims = []
        action_dims = []
        action_bounds = []
        for i in range(self.env.NUM_DRONES):
            state_dims.append(self.env.observation_space[i].shape[0])
            action_dims.append(self.env.action_space[i].shape[0])
            action_bounds.append(self.env.action_space[i].high[0])
        total_dims = sum(state_dims) + sum(action_dims)

        agents = MADDPG(state_dims, 
                        self.hidden_dim, 
                        action_dims,
                        total_dims,
                        action_bounds, 
                        self.actor_lr, 
                        self.critic_lr, 
                        self.gamma, 
                        self.tau, 
                        self.sigma, 
                        self.num_agents, 
                        self.device)
        
        return_list = trainer.train_tracking(self.env, 
                                             agents, 
                                             num_episodes,
                                             self.replybuffer, 
                                             self.minimal_size,
                                             self.batch_size, 
                                             self.update_interval)
        return agents, return_list

class MADDPGTrainParticle:

    def __init__(self, 
                 env):
        with open('./train/particle_config.json', 'r') as file:
            config = json.load(file)

        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.num_episodes = config['num_episodes']
        self.hidden_dim = config['hidden_dim']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.buffer_size = config['buffer_size']
        self.minimal_size = config['minimal_size']
        self.batch_size = config['batch_size']
        self.update_interval = config['update_interval']
        self.episode_length = config['episode_length']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seeds()
        self.replybuffer = utils.ReplayBuffer(self.buffer_size)

        self.env = env


    def set_seeds(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def learn(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.num_episodes

        state_dims = []
        action_dims = []
        for action_space in self.env.action_space:
            action_dims.append(action_space.n)
        for state_space in self.env.observation_space:
            state_dims.append(state_space.shape[0])
        total_dims = sum(state_dims) + sum(action_dims)

        agents = MADDPGForDiscrete(state_dims, 
                        self.hidden_dim, 
                        action_dims,
                        total_dims,
                        self.actor_lr, 
                        self.critic_lr, 
                        self.gamma, 
                        self.tau,
                        self.device)
        
        return_list = trainer.train_particle(self.env,
                                             agents, 
                                             num_episodes,
                                             self.replybuffer, 
                                             self.minimal_size,
                                             self.batch_size, 
                                             self.update_interval,
                                             self.episode_length)
        return agents, return_list




