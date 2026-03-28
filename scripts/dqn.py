import random
from collections import deque

import torch
import numpy as np
from environment import CartPoleEnv



class MLP_DQN:
    """ 
    Generates DQN agent with a simple MLP architecture. No buffer or TN.

    PARAMS:
    lr: learning rate
    epsilon_max: epsilon upper limit
    epsilon_min: epsilon lower limit

    FUNCTIONS:
    e_greedy(q_vals,n_actions,epsilon): simulates e-greedy policy 
    select_action(state,policy, epsilon): selects action based on policy and epsilon

    """
    def __init__(self,lr,epsilon_max,epsilon_min, replay_buffer_size=-1, batch_size=-1, network_size="medium"):
        self.state_dim = 4
        self.action_dim = 2
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min

        if network_size == "small":
            self.q_network = torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, self.action_dim)
            )
        elif network_size == "medium":
            self.q_network = torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.action_dim)
            )
        elif network_size == "large":
            self.q_network = torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.action_dim)
            )


        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        if replay_buffer_size > 0 and batch_size > 0:
            self.replay_buffer = deque(maxlen=replay_buffer_size)
        else:
            self.replay_buffer = None

    # action selection using e-greedy/softmax

    def e_greedy(self,q_vals,n_actions, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q_vals.numpy())

    def sample_replay_buffer(self):
        if self.replay_buffer is None or len(self.replay_buffer) < self.batch_size:
            return None
        else:
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        if self.replay_buffer is not None:
            self.replay_buffer.append((state, action, reward, next_state, done))
        

    def select_action(self, state, policy = "e-greedy", epsilon=1):
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
        if policy == "greedy":
            return np.argmax(q_values.numpy())
        elif policy == "e-greedy":
            return self.e_greedy(q_values, n_actions=self.action_dim, epsilon=epsilon)


    # update function for DQN, transforms the input to tensors
    # when we dont have a replay buffer we have to wrap the input in a list to make it compatible with the tensor transformation
    # i think that is a little cleaner than a if else statement for the replay buffer, but it is a bit hacky, so we can change it if you think of a better way to do it
    def update(self, states, actions, rewards, states_n, dones, gamma=0.99):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        states_n = torch.tensor(np.array(states_n), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Q(s, a) for each action taken
        q_curr = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_network(states_n).max(dim=1).values
            y_i = rewards + gamma * q_next * (1 - dones)

        loss_val = self.loss(q_curr, y_i)

        self.optimizer.zero_grad()

        loss_val.backward()

        self.optimizer.step()

