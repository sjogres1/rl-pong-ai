""" Our Agent that needs to beat the SImpleAI """
from pong import Pong
from policy import Policy


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
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        action = self.env.MOVE_UP


        return action

    def reset(self):
        """ Resets the agent to inital state """"
        raise NotImplementedError("Implementoi tämä, vitun perse")
        # return



