parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default=None,
                    help="Model to be tested")
args = parser.parse_args()


state_dict = torch.load(args.model)
policy.load_state_dict(state_dict)


env = Pong(headless=args.headless)
episodes = 10

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = Agent(env, player_id)

env.set_names(player.get_name(), opponent.get_name())

# Function to test a trained policy
def test(episodes, agent):
    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        (ob1, ob2) = env.reset()
        
        while not done:
            action1, _ = player.get_action(ob1, ep, evaluation=True)
            action2 = opponent.get_action()
            
            (ob1, ob2), (rew1, _), done, info = env.step((action1, action2))
            
            # New reward function
            env.render()

            test_reward += rew1
            test_len += 1
    print("Average test reward:", test_reward/episodes,
          "episode length:", test_len/episodes)


print("Testing...")
try:
    test(25, agent)
except KeyboardInterrupt:
    print("Testing interrupted.")
