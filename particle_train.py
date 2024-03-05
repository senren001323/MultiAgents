import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import sys
import torch
import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from train.train import MADDPGTrainParticle

sys.path.append("multiagent-particle-envs")
def make_env(scenario_name):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env

env = make_env("simple_adversary")

policy = MADDPGTrainParticle(env)
agents, return_list = policy.learn(num_episodes=4000)

agents.save_model(0)
agents.save_model(1)
agents.save_model(2)

file_dir = './result/'

return_array = np.array(return_list)
print(return_array.shape)

for i, agent_i in enumerate(["adversary_0", "agent_0", "agent_1"]):
    plt.figure()
    plt.plot(
        np.arange(return_array.shape[0]) * 100,
        utils.moving_average(return_array[:, i], 9))
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(agent_i)
    file_path = os.path.join(file_dir, f"maddpg_{agent_i}.png")
    plt.savefig(file_path)
    plt.close()


