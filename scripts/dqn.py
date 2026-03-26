
import torch 
import numpy as np
from environment import CartPoleEnv



class MLP_DQN:
    def __init__(self,lr):
        self.state_dim = 4
        self.action_dim = 2
        
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
    # action selection using e-greedy/softmax

    def select_action(self, state, policy = "e-greedy", epsilon=1):
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
        if policy == "greedy":
            return np.argmax(q_values.numpy())
        elif policy == "e-greedy":
            return e_greedy(q_values, n_actions=self.action_dim, epsilon=epsilon)

def e_greedy(q_vals,n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(q_vals.numpy())
    

class ERBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = torch.randperm(len(self.buffer))[:batch_size]
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)