from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi
from agent import Agent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true",
                    help="Run in headless mode")
args = parser.parse_args()




def plot(observation):
    plt.imshow(observation/255)
    plt.show()


def calculate_epsilon(a, episode_num):
    epsilon = a/(a + episode_num)


env = Pong(headless=args.headless)
episodes = 10
epsilon = 1

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = Agent(env, player_id)

env.set_names(player.get_name(), opponent.get_name())

for i in range(0, episodes):
    reward_sum, timesteps = 0, 0
    done = False
    # Reset the environment and observe initial states for both players
    (ob1, ob2) = env.reset()
    while not done:

        # Get actions for both agents
        action1, aprob = player.get_action(ob1, epsilon)
        action2 = opponent.get_action()
        # Save previous observation for our player
        prev_ob1 = ob1

        # Make an action with both players
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
       

        # Store action's outcome (so that the agent can improve its policy)
        player.store_outcome(prev_ob1, aprob, action1, rew1)
        

        if not args.headless:
            env.render()
        if done:
            # ob1.tofile('observation.txt', ';')
            #observation = env.reset()
            #plot(ob1) # plot the reset observation
            print("episode {} over".format(i))

    epsilon = calculate_epsilon(epsilon, episodes)

    # Store total episode reward
    reward_sum += reward
    timesteps +=1

# Needs to be called in the end to shut down pygame
env.end()
