import torch
import torch.nn.functional as F
import numpy as np
from policy.net import PolicyNet, QvalueNet, PolicyDiscrete


def onehot_from_logits(logits, eps=0.01):
    ''' Generate optimal action's ohe-hot format '''
    argmax_acts = (logits == logits.max(1, keepdim=True)[0]).float()
    rand_acts = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # ε-greedy
    return torch.stack([
        argmax_acts[i] if r > eps else rand_acts[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """ Sample from Gumbel(0,1) distribution"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Sample from Gumbel-Softmax"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """Sample from Gumbel-Softmax and discretized"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y

class DDPGForDiscrete:

    def __init__(self, 
                 state_dim,
                 hidden_dim,
                 action_dim, 
                 total_dim, 
                 actor_lr, 
                 critic_lr, 
                 device,
                 agent_idx,
                 chkpt_file='./param_dict/'):
        self.actor = PolicyDiscrete(state_dim, hidden_dim, action_dim, agent_idx, chkpt_file).to(device)
        self.target_actor = PolicyDiscrete(state_dim, hidden_dim, action_dim, agent_idx, chkpt_file).to(device)
        self.critic = QvalueNet(total_dim, hidden_dim, agent_idx, chkpt_file).to(device)
        self.target_critic = QvalueNet(total_dim, hidden_dim, agent_idx, chkpt_file).to(device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, target_net, net, tau):
        for target_param, param in zip(target_net.parameters(),
                                       net.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)

class MADDPGForDiscrete:
    
    def __init__(self,
                 state_dims,
                 hidden_dim,
                 action_dims,
                 total_dims,    
                 actor_lr, 
                 critic_lr, 
                 gamma, 
                 tau,
                 device):
        self.agents = []
        for i in range(3):
            self.agents.append(
                DDPGForDiscrete(state_dims[i], hidden_dim, action_dims[i], total_dims,
                      actor_lr, critic_lr, device, i))
        self.gamma = gamma
        self.tau = tau
        self.critic_lossfunc = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agent.actor for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_actor for agent in self.agents]

    def take_actions(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(3)
        ] # List of [(1, state_dim)_1, ..., (1, state_dim)_i]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, agent_i):
        obs, act, rew, next_obs, done = sample
        curr_agent = self.agents[agent_i]

        # update critic net for each agent 
        curr_agent.critic_optimizer.zero_grad()
        all_target_actor_acts = [
            onehot_from_logits(target_actor(_next_obs))
            for target_actor, _next_obs in zip(self.target_policies, next_obs)
        ]
        # td-target computation
        td_target = rew[agent_i].view(-1, 1) + \
                              self.gamma * \
                              curr_agent.target_critic(torch.cat((*next_obs, *all_target_actor_acts), dim=1)) * \
                              (1 - done[agent_i].view(-1, 1))
        qvalue = curr_agent.critic(
            torch.cat((*obs, *act), dim=1)
        )
        critic_loss = self.critic_lossfunc(qvalue, td_target.detach()) # MSE loss
        critic_loss.backward()
        curr_agent.critic_optimizer.step()

        # update actor net
        curr_agent.actor_optimizer.zero_grad()
        curr_actor_policy = curr_agent.actor(obs[agent_i])
        all_actor_acts = []
        for i, (actor, _obs) in enumerate(zip(self.policies, obs)):
            if i == agent_i:
                all_actor_acts.append(gumbel_softmax(curr_actor_policy))
            else:
                all_actor_acts.append(onehot_from_logits(actor(_obs)))
        actor_loss = -curr_agent.critic(
            torch.cat((*obs, *all_actor_acts), dim=1)
        ).mean()
        actor_loss += (curr_actor_policy**2).mean() * 1e-3
        actor_loss.backward()
        curr_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def save_model(self, agent_i):
        self.agents[agent_i].actor.save_checkpoint()
        self.agents[agent_i].critic.save_checkpoint()

    def load_model(self, agent_i):
        self.agents[agent_i].actor.load_checkpoint()
        self.agents[agent_i].critic.load_checkpoint()


class DDPG:

    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim,
                 total_dims,
                 action_bound,
                 actor_lr, 
                 critic_lr, 
                 gamma, 
                 sigma, 
                 device,
                 chkpt_file='./param_dict/'):
        """
        Parameters:
         -actor: For each drone's. input(batch_dim, state_dim)
         -sigma: Random guassian noise arg
        """
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, sigma).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, sigma).to(device)
        self.critic = QvalueNet(total_dims, hidden_dim).to(device)
        self.target_critic = QvalueNet(total_dims, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.sigma = sigma
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        action = self.actor(state)[0]
        action = action.cpu().detach().numpy()
        return action

    def soft_update(self, target_net, net, tau):
        for target_param, param in zip(target_net.parameters(),
                                       net.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)

class MADDPG:

    def __init__(self,
                 state_dims, 
                 hidden_dim, 
                 action_dims,
                 total_dims,
                 action_bounds,
                 actor_lr, 
                 critic_lr, 
                 gamma, 
                 tau, 
                 sigma, 
                 num_agents, 
                 device):
        self.agents = []
        for i in range(num_agents):
            self.agents.append(
                DDPG(state_dims[i], hidden_dim, action_dims[i], total_dims, action_bounds[i],
                     actor_lr, critic_lr, gamma, sigma, device)
            )
        self.nums = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.critic_lossfunc = torch.nn.MSELoss()

    def take_actions(self, states):
        """
        Args:
         -states: Shape of (nums_agents, state_dim)
        Return:
         -array(num_drones, action_dim)
             For gym-pybullet-drone frame need
        """
        states = [
            torch.tensor([states[i]], dtype=torch.float).to(self.device)
        for i in range(self.nums)
        ] # List of [(1, state_dim)_1, ..., (1, state_dim)_i]
        actions = [agent.take_action(state)
                   for agent, state in zip(self.agents, states)]
        actions = {i: actions[i] for i in range(self.nums)}
        return actions

    @property
    def policies(self):
        return [agent.actor for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_actor for agent in self.agents]

    def update(self, transition_dict, agent_i):
        # dict with each drone's: [int, tensor(batch_size, dims)]
        obs = transition_dict['obs']
        act = transition_dict['actions']
        rew = transition_dict['rewards']
        next_obs = transition_dict['next_obs']
        done = transition_dict['dones']
        curr_agent = self.agents[agent_i]
        # transfer dict to list for critic net
        obs = list(obs.values())
        act = list(act.values())
        next_obs = list(next_obs.values())
        rew = list(rew.values())
        
        # update critic net for each agent
        curr_agent.critic_optimizer.zero_grad()
        all_target_actor_acts = [
            target_actor(next_obs[i]) # little different from discrete MADDPG
        for i, target_actor in enumerate(self.target_policies)
        ]
        # td-target computation
        td_target = rew[agent_i].view(-1, 1) + \
                    self.gamma * \
                    curr_agent.target_critic(torch.cat((*next_obs, *all_target_actor_acts), dim=1)) * \
                    (1-done[agent_i].view(-1, 1))
        qvalue = curr_agent.target_critic(torch.cat((*obs, *act), dim=1))
        critic_loss = self.critic_lossfunc(qvalue, td_target.detach())
        critic_loss.backward()
        curr_agent.critic_optimizer.step()
        # update actor net 
        curr_agent.actor_optimizer.zero_grad()
        curr_actor_policy = curr_agent.actor(obs[agent_i])
        all_actor_acts = [
            actor(obs[i])
        for i, actor in enumerate(self.policies)
        ]
        actor_loss = -torch.mean(
            curr_agent.critic(torch.cat((*obs, *all_actor_acts), dim=1))
        )
        actor_loss += (curr_actor_policy**2).mean() * 1e-3 # Regularization
        actor_loss.backward()
        curr_agent.actor_optimizer.step()

    def update_target(self):
        for agent in self.agents:
            agent.soft_update(agent.target_actor, agent.actor, self.tau)
            agent.soft_update(agent.target_critic, agent.critic, self.tau)









