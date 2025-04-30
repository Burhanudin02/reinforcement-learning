import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # 4 values
action_dim = env.action_space.n             # 2 actions (left or right)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

gamma = 0.99
lr = 1e-3
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer()

#training process
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):  # max steps per episode
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Training
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            current_q = q_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].detach()
            target_q = rewards + gamma * next_q * (1 - dones)

            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Update ε and target network
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if ep % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

#visualizing performance after training

# Create a new env with render_mode='human'
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(5):  # play 5 episodes
    state, _ = env.reset()
    total_reward = 0

    for _ in range(500):
        env.render()  # show the environment
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(q_net(state_tensor)).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    print(f"Test Episode {episode+1}: Total Reward = {total_reward}")
    time.sleep(1)

env.close()
