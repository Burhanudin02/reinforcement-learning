import gymnasium as gym
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time

# Callback untuk merekam reward dan timesteps
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.timesteps = []
        self.rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Menyimpan reward dan timesteps
        self.timesteps.append(self.num_timesteps)
        self.rewards.append(np.sum(self.locals['rewards']))
        return True

    def _on_training_end(self) -> None:
        # Menyimpan data ke CSV
        os.makedirs('csv', exist_ok=True)

        df = pd.DataFrame({
            'Timesteps': self.timesteps,
            'Episode_Reward': self.rewards
        })
        # Membuat nama file CSV yang otomatis
        save_path = 'csv/a2c_reward_vs_timesteps.csv'
        if os.path.exists(save_path):
            base, ext = os.path.splitext(save_path)
            i = 1
            while os.path.exists(f'{base}_{i}{ext}'):
                i += 1
            save_path = f'{base}_{i}{ext}'
        df.to_csv(save_path, index=False)
        print(f"\n✅ Data reward vs timesteps disimpan di {save_path}")

        # Plot reward vs timesteps + moving average

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

# 1) Bungkus env jadi VecEnv
vec_env = DummyVecEnv([lambda: gym.make('CarRacing-v3', render_mode=None)])  # Render None selama training

# 2) Inisialisasi dan train model
model = A2C('CnnPolicy', vec_env, verbose=1, tensorboard_log="./a2c_car_racing_tb/")
reward_callback = RewardCallback()

model.learn(total_timesteps=50_000, callback=reward_callback)

# 3) Simpan model setelah training
# model.save("a2c_car_racing_trained")

model_save_dir = 'models'
model_base_name = 'a2c_car_racing_vec'
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
print(f"\n✅ Model A2C disimpan di {model_filepath}")

# 4) Evaluasi & Render setelah training (gunakan model yang sudah terlatih)
# Memuat model yang telah dilatih
model = A2C.load(model_filepath)

# 5) Display hasil training dengan model terlatih
vec_env = DummyVecEnv([lambda: gym.make('CarRacing-v3', render_mode="human")])  # Render mode "human" untuk evaluasi
obs = vec_env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    vec_env.render()  # Render environment
    if dones[0]:
        obs = vec_env.reset()

vec_env.close()

# 6) Print hasil metrik performa
end_time = time.time()
elapsed_time = end_time - reward_callback.start_time
average_reward = np.mean(reward_callback.rewards)
sample_efficiency = reward_callback.timesteps[-1] / elapsed_time if elapsed_time > 0 else 0
stability = np.std(reward_callback.rewards) / average_reward if average_reward > 0 else 0

print("\n💡 Metrik Performa:")
print(f"1. Average Episodic Reward: {average_reward:.2f}")
print(f"2. Sample Efficiency (timesteps per second): {sample_efficiency:.2f}")
print(f"3. Stability (reward variance / average reward): {stability:.4f}")
print(f"4. Training Time: {elapsed_time:.2f} detik")

# Menyimpan metrik performa ke CSV
metrics_df = pd.DataFrame({
    'Average Episodic Reward': [average_reward],
    'Sample Efficiency (timesteps/sec)': [sample_efficiency],
    'Stability (reward variance / avg reward)': [stability],
    'Training Time (sec)': [elapsed_time]
})

# Membuat nama file CSV yang otomatis untuk metrik performa
metrics_save_path = 'csv/a2c_metrics_performance.csv'
if os.path.exists(metrics_save_path):
    base, ext = os.path.splitext(metrics_save_path)
    i = 1
    while os.path.exists(f'{base}_{i}{ext}'):
        i += 1
    metrics_save_path = f'{base}_{i}{ext}'

metrics_df.to_csv(metrics_save_path, index=False)
print(f"✅ Data metrik performa disimpan di {metrics_save_path}")