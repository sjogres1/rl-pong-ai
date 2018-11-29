""" Our Agent that needs to beat the SImpleAI """
from pong import Pong
from policy import Policy
from torch.distributions import Categorical

policy = Policy(3)


class Agent(object):
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id
        self.action_space = [self.env.MOVE_UP, self.env.MOVE_DOWN, self.env.STAY]
        self.action_space_dim = len(self.action_space)
        self.name = "uber_AI"
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = self.torch.optim.RMSprop(policy.parameters(),lr=5e-3)
        # self.batch_size = 3
        self.gamma = 0.99
        self.observations = []
        self.actions = []
        self.rewards = []


    def get_name(self):
        """ Returns name of the agent """
        return self.name

    def get_action(self, ob=None):
        """ Returns the next action of the agent"""
        #player = self.env.player1 if self.player_id == 1 else self.env.player2
        #action = self.env.MOVE_UP


        return action

    def reset(self):
        """ Resets the agent to inital state """"
        raise NotImplementedError("Implementoi tämä, vitun perse")
        # return


    def episode_finished(self, episode_finished):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        all_actions = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        loss.backward()

        self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()



