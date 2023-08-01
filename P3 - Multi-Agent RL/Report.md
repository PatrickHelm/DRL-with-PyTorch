# Report on Collaboration and Competition Project

## Algorithm Description

I solved the Tennis environment using a Multi-Agent Deep Deterministic Policy Gradient (DDPG) algorithm.

### **Network Architecture**
In the model.py file, I defined the neural netwoks for the actor and the critic. Although DDPG is not a classic actor-critic method, it consists of two networks that are often called actor and critic. The actor takes the current state of the environment and directly selects an action to be taken. Thus, it represents a deterministic policy $\pi(s) = a$. The critic takes a state $s$ and an action $a$ and estimates the corresponding action value function $Q(s, a)$.
 
 The actor consists of two fully connected hidden layers with $128$ neurons each and a Rectified Linear Unit (ReLU) as an activation function for both layers. The size of the input layer is defined by the dimensionality of states ($24$) and the size of the output layer is defined by the dimensionality of actions ($2$). A tangens hyperbolicus activation function is applied to the output layer because the actions are continuous and range from $-1$ to $1$. 
 
 The critic consists of two fully connected hidden layers with $128$ neurons each and a Rectified Linear Unit (ReLU) as an activation function for both layers. The actions are included as an input to the first hidden layer. The size of the input layer is defined by the dimensionality of states ($24$) and the output layer consists of a single neuron. No activation function is applied to the output layer so the logit is used as an output. 

 I used a batch size of $128$ and a learning rate of $\alpha=0.001$ for both neural networks. I maintained two sets of weights for both, the actor and the critic. One set of weights is for the local networks used to select actions and predict action values. The other set of weights is for the target networks that are used to compute the loss of the critic. The actor's loss is computed by performing a forward pass with the critic to estimate the action value of the actor's predicted action in the current state. For minimization, this is multiplied by $-1$. The weights of the target networks are updated once every timestep by a soft update rule with parameter $\tau=0.001$.

### **Agent Details**

The agent itself is defined in the ddpg_multi_agent.py file.  Since this is a Multi-Agent environment, both agents receive different observations of the environment in each timestep. Therefore, I seperated the agents by assigning each of them a local/target actor and a local/target critic network. So they are only two network architectures for actor and critic but each has 4 different sets of weights.

Both agents' experience tuples (state, action, reward, next_state, done) are stored in a shared replay buffer of size $100,000$ for retrieval when learning. Future rewards whitin an episode are discounted using a factor of $\gamma=0.99$. The training is conducted in the Tennid.ipynb file. To enforce some exploration, I added Ornstein-Uhlenbeck noise with parameters $\mu=0,\theta=0.15, \sigma=0.2$ to the actions taken by both agents. This is line with the original [paper](https://arxiv.org/pdf/1509.02971.pdf) proposing DDPG from DeepMind. Training is done for at most $1000$ timesteps per episode and at most $3000$ episodes. I experimented with the training device and found out that training on the CPU was at least as fast as training on the GPU so I preferred the former.

## Results

Below, I inserted a plot of the development of the agents' scores (i.e. the maximum returns over both agents) during training. In the first $1400$ episodes, the scores are flat at $0$ with only few exceptions. Then the agents suddenly start learning something but rather slow in the beginning. After episode $2100$, the agents are ramping up their scores and finally manage to solve the environment in $2258$ epsiodes by achieving an average score of $0.51$ in the consecutive $100$ episodes. I saved the network weights (state_dict) of the actor and critic corresponding to this run in the files checkpoint_actor_1.pth and checkpoint_critic_1.pth as well as checkpoint_actor_2.pth and checkpoint_critic_2.pth for the first and second agent, respectively.

[image1]: rewards.png "Rewards"

![Rewards][image1]

## Future Work

 Although the DDPG agents are solving the environment in $2258$ episodes, I think it could be done in a much faster and more stable way. In particular, the agents seem to learn close to nothing in the first $1400$ episodes and take until episode $2100$ before they really start picking up. I have several suggestions how the agents' learning could be improved:

Since DDPG is closely related to Deep Q-Networks (DQN), it can also benefit from Prioritized Experience Replay. This means preferably sampling experiences from which the agent can still learn a lot from the replay buffer. 

Furthermore, DeepMind developed [Distributed Distributional Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1804.08617.pdf) (D4PG) which constitutes an improvement of DDPG on several levels. Most notably, D4PG includes a distributional critic update and uses distributed parallel actors. This was shown to outperform DDPG on a variety of tasks.