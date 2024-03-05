import torch
import numpy as np
import utils
from tqdm import tqdm


def train_tracking(env, agents, num_episodes, replaybuffer, minimal_size, batch_size, update_interval):
    """Train model
        Train multi-agents with off-policy
    Return: 
     -List: rewards of each drones, in total num/10 elements
    """
    return_list = []
    cnt = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                obs = env.reset()
                done = False
                while not done:
                    actions = agents.take_actions(obs)
                    next_obs, rew, done, _ = env.step(actions)
                    replaybuffer.add(obs, actions, rew, next_obs, done)
                    obs = next_obs
                    done = done["__all__"]
                    cnt += 1
                
                    if replaybuffer.size() >= minimal_size and \
                    cnt % update_interval == 0:
                        sample = replaybuffer.sample(batch_size)
                        sample = [utils.stack_array(x, env.NUM_DRONES) for x in sample]
                        b_obs, b_act, b_rew, b_nobs, b_done = sample
                        transition_dict = {'obs': {}, 'actions': {}, 'rewards': {}, 'next_obs': {}, 'dones':{}}
                        transition_dict['obs'] = {i: b_obs[i] for i in range(3)}
                        transition_dict['actions'] = {i: b_act[i] for i in range(3)}
                        transition_dict['rewards'] = {i: b_rew[i] for i in range(3)}
                        transition_dict['next_obs'] = {i: b_nobs[i] for i in range(3)}
                        transition_dict['dones'] = {i: b_done[i] for i in range(3)}
                        for i_agent in range(env.NUM_DRONES):
                            agents.update(transition_dict, i_agent)
                        agents.update_target()
                if (i_episode+1) % 10 == 0:
                    eps_return = evaluate_done(env, agents, num_episodes=10)
                    return_list.append(eps_return)
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.5f' % (np.mean(eps_return))})
                if (i_episode+1) % 50 == 0:
                    eps_return = evaluate_done(env, agents, num_episodes=10)
                    print(eps_return)
                pbar.update(1)
    return return_list

def train_particle(env, agents, num_episodes, replaybuffer, minimal_size, batch_size, update_interval, ep_length):
    return_list = []  # 记录每一轮的回报（return）
    cnt = 0
    for i in range(num_episodes):
        obs = env.reset()
        for _ in range(ep_length):
            actions = agents.take_actions(obs, explore=True)
            next_obs, reward, done, _ = env.step(actions)
            replaybuffer.add(obs, actions, reward, next_obs, done)
            obs = next_obs
            cnt += 1

            if replaybuffer.size(
            ) >= minimal_size and cnt % update_interval == 0:
                sample = replaybuffer.sample(batch_size)
                sample = [utils.stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    agents.update(sample, a_i)
                agents.update_all_targets()
        if (i + 1) % 100 == 0:
            eps_return = evaluate_length(env, agents, n_episode=100)
            return_list.append(eps_return)
            print(f"Episode: {i+1}, {eps_return}")
    return return_list

def evaluate_done(env, agents, num_episodes=10):
    """Policy evaluation for UAVs
    Return:
     -returns: List of drones
         Calculated form average episodes' returns
    """
    env = env
    returns = np.zeros(env.NUM_DRONES)
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            actions = agents.take_actions(obs)
            obs, rew, done, info = env.step(actions)
            rew = np.array(list(rew.values()))
            returns += rew / num_episodes
            done = done["__all__"]
    return returns.tolist()

def evaluate_length(env, agents, n_episode=10, episode_length=25):
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = agents.take_actions(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()





