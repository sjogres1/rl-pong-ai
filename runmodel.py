import argparse
import torch
from pong import Pong
from agent import Agent
from policy import Policy
from simple_ai import PongAi


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default=None,
                    help="Model to be tested")
parser.add_argument("--params", "-p", type=str, default=None,
                    help="Params to be tested")
args = parser.parse_args()

if args.model:
    player = torch.load(args.model)
elif args.params:
    player = Agent(player_id)
    state_dict = torch.load(args.model)
    player.policy.load_state_dict(state_dict)
else:
    player = Agent(player_id)


env = Pong(headless=False)
episodes = 10

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)

env.set_names(player.get_name(), opponent.get_name())

# Function to test a trained policy
def test(episodes):
    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        reward_sum, timesteps = 0, 0
        (ob1, ob2) = env.reset()
        while not done:
            action1, _ = player.get_action(ob1, ep, evaluation=True)
            action2 = opponent.get_action()
            
            (ob1, ob2), (rew1, _), done, info = env.step((action1, action2))
            
            # New reward function
            env.render()

            test_reward += rew1
            test_len += 1
            timesteps += 1
            reward_sum += rew1

            if done: #and episode_num % 50 == 0:
                # ob1.tofile('observation.txt', ';')
                #observation = env.reset()
                #plot(ob1) # plot the reset observation
                print("episode {} over, reward: {} \t({} timesteps)".format(ep, reward_sum, timesteps))

    print("Average test reward:", test_reward/episodes,
          "episode length:", test_len/episodes)


print("Testing...")
try:
    test(25)
except KeyboardInterrupt:
    print("Testing interrupted.")
