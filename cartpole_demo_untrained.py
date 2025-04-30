#visualize performance before training
import gymnasium as gym
import time

# Create the environment with human rendering
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(3):  # Try a few episodes
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):  # Max steps per episode
        env.render()

        # Take random actions (this is a baseline)
        action = env.action_space.sample()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Untrained Episode {episode+1}: Total Reward = {total_reward}")
    time.sleep(1)

env.close()
