import os
from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi
from agent_high_batch_size_small_lr import Agent
import argparse
import torch
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true",
                    help="Run in headless mode")
args = parser.parse_args()


def plot(observation):
    plt.imshow(observation/255)
    plt.show()


def save_model_final(player):
    model_file = "Pong_params_high_batch_size_small_lr.mdl"
    i = 1
    while os.path.isfile(model_file):
        model_file = "Pong_params%s.mdl" % i
        i += 1
    torch.save(player.policy.state_dict(), model_file)
    print("Model saved to: ", model_file)

start = datetime.now()

# Create environment
env = Pong(headless=args.headless)
episodes = 400000

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = Agent(env, player_id)

env.set_names(player.get_name(), opponent.get_name())
# Initialize lists
reward_history, timestep_history = [], []
average_reward_history = []

try:
    for episode_num in range(0, episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe initial states for both players
        (ob1, ob2) = env.reset()
        while not done:

            # Get actions for both agents
            action1, aprob = player.get_action(ob1, episode_num)
            action2 = opponent.get_action()
            # Save previous observation for our player
            prev_ob1 = ob1

            # Make an action with both players
            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        

            # Store action's outcome (so that the agent can improve its policy)
            player.store_outcome(prev_ob1, aprob, action1, rew1)
            
            # Store total episode reward
            timesteps += 1
            reward_sum += rew1

            if not args.headless:
                env.render()
            if done: #and episode_num % 50 == 0:
                # ob1.tofile('observation.txt', ';')
                #observation = env.reset()
                #plot(ob1) # plot the reset observation
                print("episode {} over, reward: {} \t({} timesteps)".format(episode_num, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_num > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Run the episode finished protocol, update policy
        player.episode_finished(episode_num)

    # Save model after training
    save_model_final(player)
    # Training is finished - plot rewards
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.legend(["Reward", "100-episode average"])
    # plt.title("Reward history (sig=%f, net 18)" % agent.policy.sigma.item())
    plt.show()
    print("Training finished.")
    print("Total time used: {}".format(datetime.now() - start))

except KeyboardInterrupt:
    print("Interrupted!")
    save_model_final(player)
    # Training is finished - plot rewards
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.legend(["Reward", "100-episode average"])
    # plt.title("Reward history (sig=%f, net 18)" % agent.policy.sigma.item())
    plt.show()
    print("Training finished.")

# Needs to be called in the end to shut down pygame
env.end()
