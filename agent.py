""" Our Agent that needs to beat the SImpleAI """
from pong import Pong


class Agent(object):
    def __init__(self, env, player_id=1):
        self.env = env
        self.player_id = player_id
        self.bpe = 4
        self.name = "uber_AI"


    def get_name(self):
        return self.name

    def get_action(self, ob=None):
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        action = self.env.MOVE_UP


        return action

    def reset(self):
        
        return