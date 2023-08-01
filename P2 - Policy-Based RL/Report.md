# Report on Navigation Project

## Algorithm Description

I solved the Reacher environment with 20 agents using a Deep Deterministic Policy Gradient (DDPG) agent.

### **Network Architecture**
In the model.py file, I defined the neural netwoks for the actor and the critic. Although DDPG is not a classic actor-critic method, it consists of two networks that are often called actor and critic. The actor takes the current state of the environment and directly selects an action to be taken. Thus, it represents a deterministic policy $\pi(s) = a$. The critic takes a state $s$ and an action $a$ and estimates the corresponding action value function $Q(s, a)$.
 
 The actor consists of two fully connected hidden layers with $128$ neurons each and a Rectified Linear Unit (ReLU) as an activation function for both layers. The size of the input layer is defined by the dimensionality of states ($33$) and the size of the output layer is defined by the dimensionality of actions ($4$). A tangens hyperbolicus activation function is applied to the output layer because the actions are continuous and range from $-1$ to $1$. 
 
 The critic consists of two fully connected hidden layers with $128$ neurons each and a Rectified Linear Unit (ReLU) as an activation function for both layers. The actions are included as an input to the first hidden layer. The size of the input layer is defined by the dimensionality of states ($33$) and the output layer consists of a single neuron. No activation function is applied to the output layer so the logit is used as an output. 

 I used a batch size of $128$ and a learning rate of $\alpha=0.001$ for both neural networks. I maintained two sets of weights for both, the actor and the critic. One set of weights is for the local networks used to select actions and predict action values. The other set of weights is for the target networks that are used to compute the loss of the critic. The actor's loss is computed by performing a forward pass with the critic to estimate the action value of the actor's predicted action in the current state. For minimization, this is multiplied by $-1$. The weights of the target networks are updated once every timestep by a soft update rule with parameter $\tau=0.001$. 

### **Agent Details**

The agent itself is defined in the ddpg_agent.py file. The agent's experience tuples (state, action, reward, next_state, done) are stored in a replay buffer of size $100,000$ for retrieval when learning. Future rewards whitin an episode are discounted using a factor of $\gamma=0.99$. The training is conducted in the Continuous_Control.ipynb file. To enforce some exploration, I added Ornstein-Uhlenbeck noise with parameters $\mu=0,\theta=0.15, \sigma=0.2$ to the actions taken by the agent. This is line with the original [paper](https://arxiv.org/pdf/1509.02971.pdf) proposing DDPG from DeepMind. Training is done for at most $1000$ timesteps per episode and at most $2000$ episodes. I experimented with the training device and found out that training on the CPU was at least as fast as training on the GPU so I preferred the former.

## Results

Below, I inserted a plot of the development of the agent's scores (i.e. the sum of rewards within an episode, averaged over all agents) during training. The scores are increasing considerably over the first 50 epsiodes peaking around 38. Afterwards, the scores stagnate and even decline slightly. The agent managed to solve the environment in $17$ epsiodes by achieving an average score of $30.19$ in the consecutive $100$ episodes. I saved the network weights (state_dict) corresponding to this run in the files checkpoint_actor.pth and checkpoint_critic.pth for the actor and the critic, respectively.

[image1]: rewards.png "Rewards"

![Rewards][image1]

## Future Work

I think the DDPG agent is already quite successful, solving the environment in only $17$ episodes. Still, further improvements are always possible so I will provide some options which might enhance the agent's ability to learn this continuous control task.

Since DDPG is closely related to Deep Q-Networks (DQN), it can also benefit from Prioritized Experience Replay. This means preferably sampling experiences from which the agent can still learn a lot from the replay buffer. 

Furthermore, DeepMind developed [Distributed Distributional Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1804.08617.pdf) (D4PG) which constitutes an improvement of DDPG on several levels. Most notably, D4PG includes a distributional critic update and uses distributed parallel actors. This was shown to outperform DDPG on various continuous control tasks.