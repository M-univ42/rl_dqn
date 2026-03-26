
import torch 
import numpy as np
from scripts.environment import CartPoleEnv



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
    def __init__(self,lr,epsilon_max,epsilon_min):
        self.state_dim = 4
        self.action_dim = 2
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dim)
        )


        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()

    # action selection using e-greedy/softmax

    def e_greedy(self,q_vals,n_actions, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q_vals.numpy())
        

    def select_action(self, state, policy = "e-greedy", epsilon=1):
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
        if policy == "greedy":
            return np.argmax(q_values.numpy())
        elif policy == "e-greedy":
            return self.e_greedy(q_values, n_actions=self.action_dim, epsilon=epsilon)
        

    def update(self,state,action,reward, state_n,done,  gamma=0.99):
        """Updates learning steps.
        PARAMS:

        """
        state = torch.tensor(state)
        state_n = torch.tensor(state_n)

        q_curr = self.q_network(state)[action]

        # no_grad to stop gradient flow

        with torch.no_grad():
            q_n = np.max(self.q_network(state_n))
            if done:
                y_i = reward
            else:
                y_i = reward+gamma*q_n

        loss_val = self.loss(q_curr,y_i)


        # reset gradients
        self.optimizer.zero_grad()

        loss_val.backward()

        self.optimizer.step()