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


""" Links that could be possible solutions to solve this problem:

https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0

http://www.cs.huji.ac.il/~ai/projects/2014/Pong/files/report.pdf

https://codeincomplete.com/posts/javascript-pong/part5/

https://github.com/topics/pong-game

http://nifty.stanford.edu/2018/guerzhoy-pong-ai-tournament/

https://www.quora.com/How-do-I-improve-my-paddle-pong-AI

http://www.pygame.org/project-py-pong-2040-.html

https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55

https://arxiv.org/abs/1807.08452

https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

https://www.youtube.com/watch?v=g9svjZxpug0

https://www.youtube.com/watch?v=YFe0YbaRIi8

https://www.reddit.com/r/MachineLearning/comments/3y16pa/i_trained_a_deep_q_network_built_in_tensorflow_to/

https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

https://ray.readthedocs.io/en/latest/example-rl-pong.html

"""