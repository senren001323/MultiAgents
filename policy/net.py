import torch
import os
import torch.nn.functional as F

class PolicyDiscrete(torch.nn.Module):
    """DDPG Discrete Policy Net
    """
    def __init__(self, 
                 state_dim,
                 hidden_dim,
                 action_dim,
                 agent_idx,
                 chkpt_file):
        super(PolicyDiscrete, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

        self.chkpt_file = os.path.join(chkpt_file + f'ddpg_agent{agent_idx}_discrete_actor.pth')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class PolicyNet(torch.nn.Module):
    """DDPG Policy Net
    """
    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim, 
                 action_bound,
                 sigma,
                 agent_idx,
                 chkpt_file):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.sigma = sigma

        self.chkpt_file = os.path.join(chkpt_file + f'ddpg_agent{agent_idx}_continuous_actor.pth')

    def forward(self, x):
        """Return: (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        noise = self.sigma * torch.randn_like(action)
        action = action + noise
        return torch.tanh(action) * self.action_bound

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class QvalueNet(torch.nn.Module):

    def __init__(self, 
                 total_dims, 
                 hidden_dim,
                 agent_idx,
                 chkpt_file):
        super(QvalueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_dims, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

        self.chkpt_file = os.path.join(chkpt_file + f'ddpg_agent{agent_idx}_cirtic.pth')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))







