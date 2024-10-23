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

from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import numpy as np
from gymnasium import spaces

import tensorflow as tf

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
        # None required. All observation space formatting handled through saving of trained model and reloading.
        return
        

    def setup_actions(self):
        # TODO: Your code to specify & modify the action space goes here
        # See Grid2Op 'getting started' notebooks for guidance
        #  - Notebooks: https://github.com/rte-france/Grid2Op/tree/master/getting_started

        # Ignore actions 'set_bus' and 'set_line_status' as they can be manipulated by the 'change' action variables
        self._gym_env.action_space = self._gym_env.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")

        action_components = self._g2op_env.action_space

        # For the continuous variable actions 'redispatch' and 'curtail', discretise them into bins for use by stable_baselines3 training algorithms
        if hasattr(action_components, 'redispatch'):
            redispatch_low = action_components.redispatch_space.low
            redispatch_high = action_components.redispatch_space.high
            redispatch_bins = np.linspace(redispatch_low, redispatch_high, 30)# Discretise into 30 bins

        if hasattr(action_components, 'curtail'):
            curtail_low = action_components.curtail_space.low
            curtail_high = action_components.curtail_space.high
            curtail_bins = np.linspace(curtail_low, curtail_high, 3)# Discretise into 3 bins

        self._gym_env.action_space = gym_compat.DiscreteActSpace(self._g2op_env.action_space)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._gym_env.step(action)

        # Get variables from observation space for reward shaping
        load_p = obs['load_p']
        load_q = obs['load_q']
        load_theta = obs['load_theta']
        load_v = obs['load_v']

        load = np.concatenate([load_p, load_q, load_theta, load_v], axis=0)

        gen_p = obs['gen_p']
        gen_q = obs['gen_q']
        gen_theta = obs['gen_theta']
        gen_v = obs['gen_v']

        gen = np.concatenate([gen_p, gen_q, gen_theta, gen_v], axis=0)

        max_gen_output = 85  # Maximum generator output before punishing
        gen_penalty = 0
        for gen_output in gen.flatten():
            if gen_output < 0:
                gen_penalty -= 0.1 # Negative power output penalty
            elif gen_output < 10:  
                gen_penalty -= 0.05  # Low power penalty
            elif gen_output > 70:
                gen_penalty -= 0.1  # High power penalty

        max_load_diff = 145  # Maximum difference between load and generated power before punishing
        load_balance_reward = 0
        load_diff = np.abs(load.sum() - gen.sum())
        if load_diff < 200:
            load_balance_reward += 0.4 # Reward if generation close to load
        else:
            load_balance_reward -= 0.1

        overflow_penalty = 0
        timestep_overflow = obs['timestep_overflow']
        for overflow in timestep_overflow.flatten():
            if overflow > 0:
                overflow_penalty -= 0.1  # Punish for each timestep a generator is in overflow
            else:
                overflow_penalty += 0.4 

        p_or = obs['p_or']
        q_or = obs['q_or']
        v_or = obs['v_or'] 
        a_or = obs['a_or']
        theta_or = obs['theta_or']

        origin = np.concatenate([p_or, q_or, v_or, a_or, theta_or], axis=0)

        p_ex = obs['p_ex']
        q_ex = obs['q_ex']
        v_ex = obs['v_ex'] 
        a_ex = obs['a_ex']
        theta_ex = obs['theta_ex']

        extremity = np.concatenate([p_ex, q_ex, v_ex, a_ex, theta_ex], axis=0)

        or_ex_balance_reward = 0
        or_ex_diff = np.abs(origin.sum() - extremity.sum())
        if or_ex_diff < 10:
            or_ex_balance_reward += 0.4 # Reward if the balance between origin and extremeties of each power line is maintained
        else:
            or_ex_balance_reward -= 0.1

        # Sum the original reward with the additional rewards for shaping
        shaped_reward = reward + gen_penalty + load_balance_reward + overflow_penalty + or_ex_balance_reward

        return obs, shaped_reward, terminated, truncated, info

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def main():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

    # DQN Algorithm #

    # Initialize the DQN model
    buffer_size = 10000

    model = DQN("MultiInputPolicy", env, verbose=0, buffer_size=buffer_size)

    obs = env.reset()
    
    total_timesteps = 300 # Train for this number of timesteps

    print("Training")
    # Train the model
    with tf.device('/device:GPU:1'):
        model.learn(total_timesteps=total_timesteps)

    model.save("dqn_model")    
    
    episodes = 100
    rewards_per_episode = []
    total_rewards = 0

    for i in range(episodes):
        model = DQN.load("dqn_model", env=env)
        env = model.get_env()

        curr_step = 0
        curr_return = 0
        is_done = False

        # Evaluate the trained model
        obs = env.reset()
        while not is_done:
            action, _states = model.predict(obs) # Using trained model, get next action using current observation
            obs, reward, is_done, info = env.step(action) # Take action selected and get reward
            curr_step += 1 # Increment number of steps taken throughout all episodes
            curr_return += reward # Add reward for current action to return for this episode
            total_rewards += reward # Update total reward

            if is_done:
                rewards_per_episode.append(curr_return)
                if(i % 10 == 0):
                    print("Current Episode: ", i)
                break

            #env.render()

    average_return  = total_rewards / episodes

    # Plotting the average return per episode
    plt.plot(range(episodes), rewards_per_episode, label='Average Return')  # Correcting plot arguments
    plt.title('DQN Average Return per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid(True)  
    plt.savefig('dqn_Itr1_return.png')
    plt.show()  

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print(f"average return = {average_return}")
    print("###########")

if __name__ == "__main__":
    main()