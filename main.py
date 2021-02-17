from random import Random
from typing import List

import numpy
import torch
import SholoGutiUnityWrapper
import matplotlib.pyplot as plt
from experience import Buffer
import random

experiences: Buffer = []

cumulative_rewards: List[float] = []

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 2
# The number of experiences to collect per training step
NUM_NEW_EXP = 1000
# The maximum size of the Buffer
BUFFER_SIZE = 10000


# Create a new Neural Network here

def main():
    # Launch and store connection to environment
    env = SholoGutiUnityWrapper.SholoGutiUnityWrapper(path='C:/Users/samin/OneDrive/Desktop/KamlaGutiBuild/KamlaGuti',
                                                      time_out_wait=120, no_graphics=True)
    print("Environment Created")
    c = 0
    for i in range(0, NUM_TRAINING_STEPS):
        run_agent(env)
        # env.reset_env()
    env.close_env()



def run_agent(env: SholoGutiUnityWrapper):
    episode_done = False
    while(not episode_done):
        obs = env.get_observation()
        # get evaluation for observation from NN here
        value = random.uniform(0, 1)
        action = [value, -value]
        env.set_actions(action)
        episode_done, step_done, reward, next_obs = env.env_step_partial()
        print("is current Step done? ", step_done)
        print("is Episode Done? ", episode_done)
        if(step_done):
            print(f"Load experience tuple including max_obs as prev_obs here: \nprev_obs {obs} \nnext_obs {next_obs} ")
            print("reset Max_state too")
        if(episode_done):
            print("Add cumulative reward")

def train_agent():
    # plt.plot(range(NUM_TRAINING_STEPS), cumulative_rewards)
    raise NotImplemented


main()

