# # # # # # # # # # # # # # #
# Reece Lazarus:    2345362 #
# Kaylyn Karuppen:  2465081 #
# # # # # # # # # # # # # # #

import gymnasium as gym
from gym.spaces import Box, Dict, MultiBinary

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
import numpy as np
from gymnasium import spaces


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

        self.action_space = self._gym_env.action_space
        self.observation_space = self._gym_env.observation_space


    def setup_observations(self):
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        print("WARNING: setup_observations is not doing anything. Implement your own code in this method.")


    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started

        print("ORIGINAL Action Space:")
        print(self._gym_env.action_space)

        # Ignore specific attributes like 'set_bus' and 'set_line_status'
        self._gym_env.action_space = self._gym_env.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")

        # Retrieve action components from the Grid2Op action space
        action_components = self._g2op_env.action_space

        # Check if redispatch and curtail exist and discretize them
        # Redispatch
        if hasattr(action_components, 'redispatch'):
            redispatch_low = action_components.redispatch_space.low
            redispatch_high = action_components.redispatch_space.high
            redispatch_bins = np.linspace(redispatch_low, redispatch_high, 30)  # 30 discrete bins for redispatch

            # You can replace the redispatch component in the action space here if necessary

        # Curtail
        if hasattr(action_components, 'curtail'):
            curtail_low = action_components.curtail_space.low
            curtail_high = action_components.curtail_space.high
            curtail_bins = np.linspace(curtail_low, curtail_high, 3)  # 3 discrete bins for curtail

            # You can replace the curtail component in the action space here if necessary

        # Use DiscreteActSpace for your modified action space
        self._gym_env.action_space = gym_compat.DiscreteActSpace(self._g2op_env.action_space)

        # After modifying, print the new action space
        print("MODIFIED Action Space:")
        print(self._gym_env.action_space)

        '''
        print("ORIGINAL Action Space:")
        print(self._gym_env.action_space)

        self._gym_env.action_space =  self._gym_env.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")

        self._gym_env.action_space =  self._gym_env.action_space.reencode_space("redispatch", gym_compat.ContinuousToDiscreteConverter(nb_bins=30))

        self._gym_env.action_space =  self._gym_env.action_space.reencode_space("curtail", gym_compat.ContinuousToDiscreteConverter(nb_bins=3))

        self._gym_env.action_space = gym_compat.DiscreteActSpace(self._gym_env.action_space)

        print("MODIFIED Action Space:")
        print(self._gym_env.action_space)
        '''

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():

    env = Gym2OpEnv()

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    # PPO Algorithm #

    # Initialize the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1)
    #model = PPO("MlpPolicy", env, verbose=1)

    print("Training")
    # Train the model
    model.learn(total_timesteps=2)

    #env = model.get_env()
    # Test the trained model
    obs = env.reset()
    for i in range(2):
        print("This loop iteration: ", i)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    # Random agent interacting in environment #
    '''
    max_steps = 100

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
        print(f"\t obs = {obs}")
        print(f"\t reward = {reward}")
        print(f"\t terminated = {terminated}")
        print(f"\t truncated = {truncated}")
        print(f"\t info = {info}")

        # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
        # Invalid actions are replaced with 'do nothing' action
        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info['exception']}")
        print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")
    '''

if __name__ == "__main__":
    main()