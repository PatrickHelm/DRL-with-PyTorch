# Report on Navigation Project

## Algorithm Description

I solved the Banana environment using a Deep Q-Network (DQN) agent.

### Network Architecture

In the model.py file, I defined the neural netwok to estimate the action value function given a state of the environment. It consists of two hidden layers with $64$ neurons each and a Rectified Linear Unit (ReLU) as an activation function for both layers. The size of the input layer is defined by the dimensionality of states ($37$) and the size of the output layer is defined by the number of actions ($4$). No activation function is applied to the output layer so the logits are used as outputs. I used a batch size of $64$ for the neural network and a learning rate of $\alpha=0.0005$. I maintained two sets of weights, one for the local network used to select actions and one for the target network used to compute the Temporal Difference (TD) error. The weights of the target network are updated every $4$ steps by a soft update rule with parameter $\tau=0.001$. 

### Q-Learning Details

The agent itself is defined in the dqn_agent.py file. The agent's experience tuples (state, action, reward, next_state, done) are stored in a replay buffer of size $100,000$ for retrieval when learning. Future rewards whitin an episode are discounted using a factor of $\gamma=0.99$. The training is conducted in the Navigation.ipynb file. During training, the agent follows an $\epsilon$-greedy policy i.e. with a probability of $\epsilon$, it chose a random action and otherwise the greedy action (with the highest action value). The value of $\epsilon$ is initially set to $1$ and decays at a rate of $0.995$ to the minimum value of  $0.01$. Training is done for at most $1000$ timesteps per episode and at most 2000 episodes. 

## Results

Below, I inserted a plot of the development of the agent's scores (i.e. sum of rewards within an episode) during training. There is considerable variability in the scores but, on average, they are steadily increasing. The agent managed to solve the environment in $421$ epsiodes by achieving an average score of $13$ in the consecutive $100$ episodes. I saved the model's weights (state_dict) corresponding to this run in the file checkpoint.pth.

[image1]: rewards.png "Rewards"

![Rewards][image1]

## Future Work

I think the DQN agent is already quite successful as it was suggested to solve the environment in less then $1800$ episodes and the agent does it in less than one quarter. Still, further improvements are always possible so I will provide some options which might enhance the agent's ability to learn to navigate this environment.

Several promising enhancements of the classic DQN algorithms have been proposed in the literature. Among those are Double DQN, Dueling DQN and Prioritized Experience Replay. All of those serve to mitigate a particular weakness of classic Deep Q-Learning. For example, Double DQN deal with the fact that a traditional DQN tends to overestimate action values.

To make the agent's performance more comparable to that of a human, it is imperative to provide equal prerequisites. This means the agent should be able to learn from the raw pixels instead of the vector-encoded states. The would be arguably harder and also require a different network architecture, in particular da Convolutional Neural Network (CNN).