# env/robotic_arm_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RoboticArmEnv(gym.Env):
    def __init__(self):
        super(RoboticArmEnv, self).__init__()

        self.num_joints = 4
        self.link_lengths = np.array([0.5, 0.5, 0.3, 0.2])

        # Action: delta angles for 4 joints
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.num_joints,), dtype=np.float32)

        # Observation: joint angles + target (x, y)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints + 2,), dtype=np.float32)

        # Obstacles: list of (x, y, radius)
        self.obstacles = [
            (0.5, 0.5, 0.15),
            (0.8, -0.3, 0.1),
            (-0.4, 0.7, 0.1)
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        self.joint_angles = np.random.uniform(-0.1, 0.1, size=self.num_joints)

        # Random target placement avoiding direct obstacle overlap
        while True:
            self.target_position = np.random.uniform(low=-1.2, high=1.2, size=(2,))
            collision = any(np.linalg.norm(self.target_position - np.array([ox, oy])) < r + 0.1
                            for ox, oy, r in self.obstacles)
            if not collision:
                break

        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        self.joint_angles += action
        self.joint_angles = np.clip(self.joint_angles, -np.pi, np.pi)

        # Get end-effector position
        ee_pos = self._get_ee_position()
        distance_to_target = np.linalg.norm(ee_pos - self.target_position)

        # Attraction to target (normalized)
        max_reach = np.sum(self.link_lengths)
        reward = -distance_to_target / max_reach

        # Penalize large joint movements (smooth motion)
        reward -= 0.05 * np.sum(np.square(action))

        # Obstacle repulsion (smooth potential field)
        for ox, oy, r in self.obstacles:
            for link_pos in self._get_link_positions():
                dist = np.linalg.norm(np.array([ox, oy]) - link_pos)
                buffer = r + 0.2  # safety buffer
                if dist < buffer:
                    # Gradual penalty: stronger when closer
                    reward -= 0.5 * (buffer - dist)

        # Done conditions
        done_target = distance_to_target < 0.05
        done = done_target

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.joint_angles, self.target_position])

    def _get_link_positions(self):
        positions = []
        x, y = 0, 0
        angle_sum = 0
        for i, l in enumerate(self.link_lengths):
            angle_sum += self.joint_angles[i]
            x += l * np.cos(angle_sum)
            y += l * np.sin(angle_sum)
            positions.append(np.array([x, y]))
        return positions

    def _get_ee_position(self):
        return self._get_link_positions()[-1]

    def render(self):
        plt.clf()
        # Draw links
        x0, y0 = 0, 0
        positions = self._get_link_positions()
        xs = [x0] + [p[0] for p in positions]
        ys = [y0] + [p[1] for p in positions]
        plt.plot(xs, ys, marker='o', lw=4, color='blue')

        # Draw obstacles
        for ox, oy, r in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='gray')
            plt.gca().add_patch(circle)

        # Draw target
        plt.plot(self.target_position[0], self.target_position[1], marker='*', color='red', markersize=15)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title("4-DOF Robotic Arm with Obstacles (Potential Field Rewards)")
        plt.pause(0.05)
