import os
from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi
from agent import Agent
import argparse
import torch
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true",
                    help="Run in headless mode")
parser.add_argument("--model", "-m", type=str, default=None,
                    help="Model to be tested")
parser.add_argument("--params", "-p", type=str, default=None,
                    help="Model to be tested")
args = parser.parse_args()


def plot(observation):
    plt.imshow(observation/255)
    plt.show()


def save_model_final(player):
    param_file = "Pong_params.mdl"
    model_file = "Pong_model.obj"
    i = 1
    while os.path.isfile(param_file):
        param_file = "Pong_params%s.mdl" % i
        i += 1
    i = 1
    while os.path.isfile(model_file):
        model_file = "Pong_model%s.obj" % i
        i += 1
    torch.save(player.policy.state_dict(), param_file)
    torch.save(player, model_file)
    print("Params saved to: ", param_file)
    print("Model saved to: ", model_file)

start = datetime.now()

# Create environment
env = Pong(headless=args.headless)
episodes = 10

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)

if args.model:
    player = torch.load(args.model)
elif args.params:
    player = Agent(player_id)
    state_dict = torch.load(args.model)
    player.policy.load_state_dict(state_dict)
else:
    player = Agent(player_id)

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
except:
    print("Exception thrown!")
    save_model_final(player)
    raise

finally:
    # Needs to be called in the end to shut down pygame
    env.end()

# Needs to be called in the end to shut down pygame
env.end()
