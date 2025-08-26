# test.py
import os
import time
from stable_baselines3 import PPO
from environment.robotic_arm_env import RoboticArmEnv

# Workaround for OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env = RoboticArmEnv()

model = PPO.load("models/ppo_robotic_arm_4dof")

obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        break
    time.sleep(0.1)

env.close()
