import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C, TD3

print("\n🎬 Mulai rendering dari model terlatih...")
env_display = gym.make('CarRacing-v3', render_mode="human")
# model_loaded = PPO.load("./algorithms/PPO/models/ppo_car_racing_vec")
model_loaded = SAC.load("./algorithms/SAC/models/sac_car_racing_vec_1")
# model_loaded = A2C.load("./algorithms/A2C/models/a2c_car_racing_vec")
# model_loaded = TD3.load("./algorithms/TD3/models/td3_car_racing_vec")

obs, _ = env_display.reset()

for _ in range(1000):
    action, _states = model_loaded.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_display.step(action)

    # Render manual kalau environment tidak auto-render
    if hasattr(env_display, "render") and callable(getattr(env_display, "render", None)):
        env_display.render()

    # Cek apakah episode selesai
    if terminated or truncated:
        obs, _ = env_display.reset()

env_display.close()

# PPO
# obs, _ = env_display.reset()
# for _ in range(1000):
#     action, _states = model_loaded.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env_display.step(action)
#     if terminated or truncated:
#         obs, _ = env_display.reset()

# env_display.close()

#SAC
# obs, _ = env_display.reset()
# for _ in range(1000):
#     action, _states = model_loaded.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env_display.step(action)
#     if terminated or truncated:
#         obs, _ = env_display.reset()

# env_display.close()

#A2C
# obs = env_display.reset()

# for _ in range(1000):
#     action, _states = model_loaded.predict(obs, deterministic=True)
#     obs, rewards, dones, infos = env_display.step(action)
#     env_display.render()  # Render environment
#     if dones[0]:
#         obs = env_display.reset()

# env_display.close()

#TD3
# obs = env_display.reset()

# for _ in range(1000):
#     action, _states = model_loaded.predict(obs, deterministic=True)
#     obs, rewards, dones, infos = env_display.step(action)
#     env_display.render()  # Render environment
#     if dones[0]:
#         obs = env_display.reset()

# env_display.close()