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

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        # TODO: Your code to specify & modify the observation space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started
        print("WARNING: setup_observations is not doing anything. Implement your own code in this method.")

    def flatten_space_custom(self, space):
        if isinstance(space, MultiBinary):
            return np.zeros(space.n, dtype=int)  # Initialize a zero array for MultiBinary
        elif isinstance(space, Box):
            return np.zeros(space.shape, dtype=space.dtype)  # Initialize a zero array for Box
        elif isinstance(space, Dict):
            # For Dict, flatten each component and return as a concatenated array
            flat = []
            for key, subspace in space.spaces.items():
                flat.append(self.flatten_space_custom(subspace))  # Recursively flatten subspaces
            return np.concatenate(flat)
        else:
            raise NotImplementedError(f"Unknown space type: {type(space)}")


    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started

        # Flatten each space inside the Dict action space
         # Prepare to flatten the action space
        # Prepare to flatten the action space
        flat_action_spaces = []
    
        # Iterate through each space in the action space
        for key, space in self._gym_env.action_space.spaces.items():
            flat_action_space = self.flatten_space_custom(space)
            flat_action_spaces.append(flat_action_space)

        # Create a single Box action space that is the combination of all flattened spaces
        low = np.concatenate([space.low.flatten() for space in flat_action_spaces if isinstance(space, Box)])
        high = np.concatenate([space.high.flatten() for space in flat_action_spaces if isinstance(space, Box)])

        # For MultiBinary, concatenate zeros and ones as needed
        for space in flat_action_spaces:
            if isinstance(space, MultiBinary):
                low = np.concatenate([low, np.zeros(space.n, dtype=int)])
                high = np.concatenate([high, np.ones(space.n, dtype=int)])

        # Create the Box action space with combined bounds
        self.action_space = Box(low=low, high=high)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        unflattened_action = self._gym_env.action_space.unflatten(action)
        return self._gym_env.step(unflattened_action)
        #return self._gym_env.step(action)

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

    flat_action_space = self.flatten_space_custom(env.action_space)
    print("flattened space: ", flat_action_space)

    # PPO Algorithm

    # Initialize the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Test the trained model
    obs = env.reset()
    for _ in range(1000):
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