from DQN import DQNAgent
from main import ConnectFourEnv
import numpy as np 

# Initialize environment and DQN agent
env = ConnectFourEnv()
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Hyperparameters
episodes = 1000
batch_size = 32

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):  # Set maximum time steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Optionally save the model
    if e % 10 == 0:
        agent.save("connect_four_dqn.h5")
