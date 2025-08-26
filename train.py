# train.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.robotic_arm_env import RoboticArmEnv

def main():
    # TensorBoard logs
    log_dir = "./logs/ppo_robotic_arm_4dof/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = RoboticArmEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir
    )

    # Train
    model.learn(total_timesteps=50000)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_robotic_arm_4dof")
    print("Training complete! Model saved as ppo_robotic_arm_4dof.zip")

    env.close()

if __name__ == "__main__":
    main()
