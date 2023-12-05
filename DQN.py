import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import random
from collections import namedtuple
from main import ConnectFourEnv

class HiddenLayer(nn.Module):
    def __init__(self, M1, M2, activation=nn.Tanh, use_bias=True):
        super(HiddenLayer, self).__init__()
        self.use_bias = use_bias
        self.linear = nn.Linear(M1, M2, bias=use_bias)
        self.activation = activation()

    def forward(self, X):
        if self.use_bias:
            return self.activation(self.linear(X))
        else:
            return self.activation(X @ self.linear.weight)

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        M1 = input_size
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # Final layer without activation
        self.layers.append(nn.Linear(M1, output_size))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

from main import ConnectFourEnv, ConnectFour

class CompeteEnv(ConnectFourEnv):
    def __init__(self, dqn_agent):
        super().__init__()
        self.dqn_agent = dqn_agent

    def choose_action(self):
        if self.game.turn == self.game.AI:
            # DQN Agent's turn
            with torch.no_grad():
                state = torch.tensor(self.game.board.flatten(), dtype=torch.float32)
                q_values = self.dqn_agent(state)
                action = q_values.argmax().item()
        else:
            # Minimax's turn
            action, _ = self.game.minimax(self.game.board, 5, -float("inf"), float("inf"), True)
        
        return action

def main():
    # Load DQN Agent
    input_size = 42  # 6 rows * 7 columns
    output_size = 7  # 7 possible actions (columns)
    sizes = [512,256,128,64,64]
    dqn_agent = DQN(input_size, output_size, sizes)
    dqn_agent.load("connect_four_dqn.pth", device=torch.device("cpu"))

    # Initialize the environment
    env = CompeteEnv(dqn_agent)
    num_episodes = 100  # Set the number of episodes
    dqn_wins = 0
    minimax_wins = 0
    draws = 0

    for episode in range(num_episodes):
        env.reset()
        done = False
        while not done:
            action = env.choose_action()
            obs, reward, done, info = env.step(action)
            if done:
                if reward == 1:
                    dqn_wins += 1
                elif reward == -1:
                    minimax_wins += 1
                else:
                    draws += 1
                break

        print(f"Episode: {episode}")

    print(f"Results after {num_episodes} episodes:")
    print(f"DQN Wins: {dqn_wins}")
    print(f"Minimax Wins: {minimax_wins}")
    print(f"Draws: {draws}")

if __name__ == '__main__':
    main()
