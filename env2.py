import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class FlattenedActionEnv(gym.Env):
    def __init__(self):
        super(FlattenedActionEnv, self).__init__()

        # Define the original action space
        self.original_action_space = spaces.Dict({
            'change_bus': spaces.MultiBinary(57),
            'change_line_status': spaces.MultiBinary(20),
            'curtail': spaces.Box(low=np.array([-1., -1., 0., 0., 0., -1.]), high=np.array([1., 1., 1., 1., 1., -1.]), dtype=np.float32),
            'redispatch': spaces.Box(low=np.array([-5., -10., 0., 0., 0., -15.]), high=np.array([5., 10., 0., 0., 0., 15.]), dtype=np.float32),
            'set_bus': spaces.Box(low=-1, high=2, shape=(57,), dtype=np.int32),
            'set_line_status': spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int32)
        })

        # Flattened action space
        self.action_space = self.flatten_action_space(self.original_action_space)

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)

    def flatten_action_space(self, action_space):
        action_low = []
        action_high = []

        for space in action_space.spaces.values():
            if isinstance(space, spaces.MultiBinary):
                action_low.append(np.zeros(space.n))
                action_high.append(np.ones(space.n))
            else:
                action_low.append(space.low.flatten())
                action_high.append(space.high.flatten())

        return spaces.Box(low=np.concatenate(action_low), high=np.concatenate(action_high), dtype=np.float32)

    def step(self, action):
        idx = 0
        change_bus = action[idx:idx + 57].astype(np.int32)
        idx += 57

        change_line_status = action[idx:idx + 20].astype(np.int32)
        idx += 20

        curtail = action[idx:idx + 6]
        idx += 6

        redispatch = action[idx:idx + 6]
        idx += 6

        set_bus = action[idx:idx + 57].astype(np.int32)
        idx += 57

        set_line_status = action[idx:idx + 20].astype(np.int32)

        # Implement environment logic...
        reward = 0  # Adjust this based on your logic
        done = False  # Set True when done
        truncated = False  # Set True if the episode is truncated
        info = {}  # Additional info

        return self._get_obs(), reward, done, truncated, info


    def reset(self, seed=None):
        initial_obs = self._get_obs()
        print(f"Initial Observation: {initial_obs}")  # Debugging print statement
        return initial_obs, {}  # Return the observation and an empty info dictionary

    def _get_obs(self):
        return np.zeros(self.observation_space.shape)

# Create and wrap the environment
env = FlattenedActionEnv()
env = Monitor(env)
env = DummyVecEnv([lambda: env])

print(type(env.action_space))

# Train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
