import torch
from main import ConnectFourEnv, ConnectFour
import numpy as np
from DQN import DQN, ReplayBuffer

# Include your DQN and ReplayBuffer classes here
# ...

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
            action, _ = self.game.minimax(self.game.board, 5, -np.inf, np.inf, True)
        
        return action

def main():
    # Load DQN Agent
    input_size = 42  # 6 rows * 7 columns
    output_size = 7  # 7 possible actions (columns)
    dqn_agent = DQN(input_size, output_size)
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

    print(f"Results after {num_episodes} episodes:")
    print(f"DQN Wins: {dqn_wins}")
    print(f"Minimax Wins: {minimax_wins}")
    print(f"Draws: {draws}")

if __name__ == '__main__':
    main()
