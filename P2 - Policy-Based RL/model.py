import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        #reset parameters
        lim_fc1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        lim_fc2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-lim_fc1, lim_fc1)
        self.fc2.weight.data.uniform_(-lim_fc2, lim_fc2)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
        #reset parameters
        lim_fc1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        lim_fc2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-lim_fc1, lim_fc1)
        self.fc2.weight.data.uniform_(-lim_fc2, lim_fc2)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = self.relu(self.fc2(x))
        return self.fc3(x)