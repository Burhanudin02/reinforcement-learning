import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Custom Callback: Rekam reward + simpan data
class RewardCallback(BaseCallback):
    def __init__(self):
        super(RewardCallback, self).__init__()
        self.rewards = []
        self.timesteps = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        self.episode_reward += reward
        if done:
            self.rewards.append(self.episode_reward)
            self.timesteps.append(self.num_timesteps)
            self.episode_reward = 0
        return True

# Moving Average Function
def moving_average(values, window=10):
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')

# 1) Buat environment tanpa render untuk training
vec_env = DummyVecEnv([lambda: gym.make('CarRacing-v3', render_mode=None)])

# 2) Model
model = PPO('CnnPolicy', vec_env, verbose=1, tensorboard_log="./ppo_car_racing_tb/")

# 3) Setup callback
reward_callback = RewardCallback()

# 4) Mulai training
total_timesteps = 50_000
start_time = time.time()

model.learn(total_timesteps=total_timesteps, callback=reward_callback)

training_time = time.time() - start_time
# model.save("ppo_car_racing_vec")
vec_env.close()

# 5) Save hasil ke CSV
save_dir = 'csv'
base_filename = 'ppo_reward_vs_timesteps'
extension = '.csv'

# Pastikan folder csv ada
os.makedirs(save_dir, exist_ok=True)

# Cari nama file berikutnya
counter = 0
while True:
    if counter == 0:
        filename = f"{base_filename}{extension}"
    else:
        filename = f"{base_filename}_{counter}{extension}"
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        break
    counter += 1

# Simpan file
df = pd.DataFrame({
    'Timesteps': reward_callback.timesteps,
    'Episode_Reward': reward_callback.rewards
})
df.to_csv(filepath, index=False)

print(f"\n✅ Data reward vs timesteps disimpan di {filepath}")

# 6) Plot reward vs timesteps + moving average
plt.figure(figsize=(10,6))
plt.plot(df['Timesteps'], df['Episode_Reward'], label='Reward per Episode', alpha=0.3)
# plt.plot(df['Timesteps'][len(df)-len(moving_average(df['Episode_Reward'])):], 
#          moving_average(df['Episode_Reward']), label='Moving Average (window=10)', color='red')
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('PPO on CarRacing-v3: Reward vs Timesteps')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 7) Hitung metrik evaluasi
avg_reward = np.mean(df['Episode_Reward'])
sample_efficiency = avg_reward / (total_timesteps / 1000)  # reward per 1000 timesteps
stability = np.std(df['Episode_Reward'])

# 8) Cetak hasil
print("\n==== Training Summary ====")
print(f"Average Episodic Reward : {avg_reward:.2f}")
print(f"Sample Efficiency       : {sample_efficiency:.2f} reward per 1000 timesteps")
print(f"Stability (Std Dev)      : {stability:.2f}")
print(f"Training Time            : {training_time:.2f} seconds")
print("==========================")

# Menyimpan metrik performa ke CSV
metrics_df = pd.DataFrame({
    'Average Episodic Reward': [avg_reward],
    'Sample Efficiency (timesteps/sec)': [sample_efficiency],
    'Stability (reward variance / avg reward)': [stability],
    'Training Time (sec)': [training_time]
})

# Membuat nama file CSV yang otomatis untuk metrik performa
metrics_save_path = 'csv/ppo_metrics_performance.csv'
if os.path.exists(metrics_save_path):
    base, ext = os.path.splitext(metrics_save_path)
    i = 1
    while os.path.exists(f'{base}_{i}{ext}'):
        i += 1
    metrics_save_path = f'{base}_{i}{ext}'

metrics_df.to_csv(metrics_save_path, index=False)
print(f"✅ Data metrik performa disimpan di {metrics_save_path}")

# 9) Save model PPO
model_save_dir = 'models'
model_base_name = 'ppo_car_racing_vec'
os.makedirs(model_save_dir, exist_ok=True)

# Cari nama file model otomatis
model_counter = 0
while True:
    if model_counter == 0:
        model_filename = f"{model_base_name}.zip"
    else:
        model_filename = f"{model_base_name}_{model_counter}.zip"
    model_filepath = os.path.join(model_save_dir, model_filename)
    if not os.path.exists(model_filepath):
        break
    model_counter += 1

model.save(model_filepath)
print(f"\n✅ Model PPO disimpan di {model_filepath}")


# 10) Render setelah training (load ulang model)
print("\n🎬 Mulai rendering dari model terlatih...")
env_display = gym.make('CarRacing-v3', render_mode="human")
model_loaded = PPO.load("ppo_car_racing_vec")

obs, _ = env_display.reset()
for _ in range(1000):
    action, _states = model_loaded.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_display.step(action)
    if terminated or truncated:
        obs, _ = env_display.reset()

env_display.close()
