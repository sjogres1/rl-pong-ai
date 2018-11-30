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
    done = False
    (ob1, ob2) = env.reset()
    while not done:
        action1 = player.get_action(ob1, epsilon)
        action2 = opponent.get_action()
        (next_ob1, next_ob2), (rew1, rew2), done, info = env.step((action1, action2))

        
        # print(ob1[0][5])
        # print(rew1)



        if not args.headless:
            env.render()
        if done:
            # ob1.tofile('observation.txt', ';')
            observation = env.reset()
            #plot(ob1) # plot the reset observation
            print("episode {} over".format(i))

    epsilon = calculate_epsilon(epsilon, episode_num)

# Needs to be called in the end to shut down pygame
env.end()
