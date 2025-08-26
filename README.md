# 4 DOF Robotic Arm Reinforcement Learning Model 

This project implements reinforcement learning for controlling a robotic arm using the PPO algorithm. It uses [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and [PyBullet](https://github.com/bulletphysics/bullet3) for simulation and training.

## Project Structure

- `environment/robotic_arm_env.py`: Custom Gymnasium environment for the robotic arm.
- `train.py`: Script to train the PPO agent.
- `test.py`: Script to test the trained agent.
- `models/`: Saved trained models.
- `logs/`: TensorBoard logs for training runs.

## Installation

Install dependencies using pip:

```sh
pip install -r requirements.txt
```

## Usage

### Training

```sh
python train.py
```

### Testing

```sh
python test.py
```

## Monitoring Training

TensorBoard logs are saved in the `logs/` directory. To visualize:

```sh
tensorboard --logdir logs/
```

## Requirements

See [`requirements.txt`](requirements.txt) for required Python packages.
