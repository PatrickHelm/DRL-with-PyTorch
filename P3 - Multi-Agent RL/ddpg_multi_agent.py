import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import ActorNetwork, CriticNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # every n timesteps, the network are updated
UPDATES_PER_STEP = 1    # how often to update the network (if at all)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Multi_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.actor_local_1 = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target_1 = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_optimizer_1 = optim.Adam(self.actor_local_1.parameters(), lr=LR_ACTOR)

        self.critic_local_1 = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target_1 = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.actor_local_2 = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target_2 = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_optimizer_2 = optim.Adam(self.actor_local_2.parameters(), lr=LR_ACTOR)

        self.critic_local_2 = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target_2 = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise((n_agents, action_size), random_seed)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step=0
        
        print('training on '+str(device))
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        self.t_step+=1
        if self.t_step%UPDATE_EVERY==0:
            for _ in range(UPDATES_PER_STEP):
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local_1.eval()
        self.actor_local_2.eval()
        with torch.no_grad():
            action_1 = self.actor_local_1(state[0]).cpu().data.numpy()
            action_2 = self.actor_local_2(state[1]).cpu().data.numpy()
        self.actor_local_1.train()
        self.actor_local_2.train()
        action = np.array([action_1, action_2])
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions_1 = self.actor_target_1(next_states)
        next_actions_2 = self.actor_target_2(next_states)
        next_actions = [next_actions_1, next_actions_2]
        
        next_targets_1 = self.critic_target_1(next_states, next_actions[0])
        next_targets_2 = self.critic_target_2(next_states, next_actions[1])
        next_targets = [next_targets_1, next_targets_2]
        
        # Compute Q targets for current states (y_i)
        targets_1 = rewards + (gamma * next_targets[0] * (1 - dones))
        targets_2 = rewards + (gamma * next_targets[1] * (1 - dones))
        targets = [targets_1, targets_2]
        
        # Compute critic loss
        expected_1 = self.critic_local_1(states, actions)
        expected_2 = self.critic_local_2(states, actions)
        expected = [expected_1, expected_2]
        
        critic_loss_1 = F.mse_loss(expected[0], targets[0])
        critic_loss_2 = F.mse_loss(expected[1], targets[1])
        
        # Minimize the loss
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_predicted_1 = self.actor_local_1(states)
        actions_predicted_2 = self.actor_local_2(states)
        actions_predicted = [actions_predicted_1, actions_predicted_2]
        
        actor_loss_1 = -self.critic_local_1(states, actions_predicted[0]).mean()
        actor_loss_2 = -self.critic_local_2(states, actions_predicted[1]).mean()
        # Minimize the loss
        self.actor_optimizer_1.zero_grad()
        actor_loss_1.backward()
        self.actor_optimizer_1.step()
        
        self.actor_optimizer_2.zero_grad()
        actor_loss_2.backward()
        self.actor_optimizer_2.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local_1, self.critic_target_1, TAU)
        self.soft_update(self.actor_local_1, self.actor_target_1, TAU)
        self.soft_update(self.critic_local_2, self.critic_target_2, TAU)
        self.soft_update(self.actor_local_2, self.actor_target_2, TAU)   

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)