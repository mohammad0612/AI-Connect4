import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import random
from collections import namedtuple
from main import ConnectFourEnv, ConnectFour
from DQN import DQN, ReplayBuffer
from blocking_bot import bot_move 
import tkinter as tk
import numpy as np

class ConnectFourGUI:
    def __init__(self, rows=6, columns=7, cell_size=50):
        self.rows = rows
        self.columns = columns
        self.cell_size = cell_size
        self.window = tk.Tk()
        self.window.title("Connect Four DQN Training")
        self.canvas = tk.Canvas(self.window, width=self.columns*self.cell_size, height=self.rows*self.cell_size)
        self.canvas.pack()

    def draw_board(self, board):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.columns):
                color = "white"
                if board[r, c] == 1:
                    color = "blue"
                elif board[r, c] == 2:
                    color = "red"
                self.draw_cell(r, c, color)

    def draw_cell(self, row, col, color):
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill=color, outline="black")

    def update(self):
        self.window.update_idletasks()
        self.window.update()

    def close(self):
        self.window.destroy()

def train_model(dqn, target_dqn, replay_buffer, optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    # Sample a batch of experiences from the replay buffer
    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Extract states, actions, rewards, next_states, dones from the batch
    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)
    dones = torch.cat(batch.done)

    # Compute Q(s_t, a)
    current_q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute V(s_{t+1}) for all next states, using the target network
    next_state_values = target_dqn(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    # Compute the target q values
    target_q_values = rewards + gamma * next_state_values

    # Compute loss
    loss = F.mse_loss(current_q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def epsilon_greedy_action(dqn, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(dqn.output_size)
    else:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = dqn(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    
# Assuming the following classes are already defined:
# DQN, ReplayBuffer, epsilon_greedy_action

# Setup for training
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Initialize environment
env = ConnectFourEnv()

# Setup for DQN agents
input_size = 42  # Assuming a flattened state space
output_size = 7  # Number of possible actions (columns)
layer_sizes = [512, 256, 128, 64, 64]

# Initialize DQN and Target DQN for both agents
dqn = DQN(input_size, output_size, layer_sizes).to(device)
target_dqn = DQN(input_size, output_size, layer_sizes).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)

# Replay Buffers
replay_buffer = ReplayBuffer(capacity=100000)

batch_size = 32
gamma = 0.99  # Discount factor

# Epsilon values for ε-greedy strategy
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9999
epsilon = epsilon_start

def optimize_model(dqn, target_dqn, replay_buffer, optimizer):
    if len(replay_buffer) < batch_size:
        return

    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # Convert batch components from NumPy arrays to PyTorch tensors
    states = torch.cat([torch.tensor([s], dtype=torch.float32) for s in batch.state]).to(device)
    actions = torch.cat([torch.tensor([a], dtype=torch.int64) for a in batch.action]).to(device)
    rewards = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward]).to(device)
    next_states = torch.cat([torch.tensor([ns], dtype=torch.float32) for ns in batch.next_state]).to(device)
    dones = torch.cat([torch.tensor([d], dtype=torch.uint8) for d in batch.done]).to(device)

    # Convert 'dones' to boolean tensor
    dones = dones.type(torch.bool)

    # Compute Q(s_t, a)
    current_q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute V(s_{t+1}) for all next states, using the target network
    next_state_values = target_dqn(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    # Compute the target q values
    target_q_values = rewards + gamma * next_state_values

    # Compute loss
    loss = F.mse_loss(current_q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Set up environment, DQN, target DQN, optimizer, replay buffer, etc.

num_episodes = 100000  # Total number of episodes for training
copy_steps = 50     # Number of episodes to update the target network

gui = ConnectFourGUI()

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        gui.draw_board(np.reshape(state, (6, 7)))
        gui.update()

        # DQN Agent's turn
        if env.current_player == env.game.USER:
            action = epsilon_greedy_action(dqn, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            optimize_model(dqn, target_dqn, replay_buffer, optimizer)

        # Minimax Algorithm's turn
        else:
            board = np.reshape(state, (6, 7))
            action, _ = env.game.minimax(board, 5, -np.inf, np.inf, True)  # Depth set to 5
            next_state, reward, done, _ = env.step(action)

        state = next_state

        if episode % copy_steps == 0:
            target_dqn.load_state_dict(dqn.state_dict())

    epsilon = max(epsilon_end, epsilon_decay * epsilon)  # Decay epsilon
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
    if episode % 100 == 0:
        print("Saving model")
        dqn.save("connect_four_dqn.pth")

gui.close()