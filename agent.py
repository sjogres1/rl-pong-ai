""" Our Agent that needs to beat the SImpleAI """
import os
from pong import Pong
from policy import Policy
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


    # Discounted reward calculation
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Agent(object):
    def __init__(self, env, player_id=1):
        policy = Policy(3)
        self.env = env
        self.player_id = player_id
        self.name = "uber_AI"
        self.model_file = self.init_run_model_file_name()
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        # What should the learning rate lr be?
        self.optimizer = torch.optim.RMSprop(policy.parameters(),lr=5e-3)
        # Should this be one or maybe 2-5?
        self.batch_size = 1 
        # What should the gamma be?
        self.gamma = 0.99
        self.epsilon = 1.0
        self.a = 1
        self.observations = []
        self.actions = []
        self.rewards = []
        self.grid_count = 0
        self.grid_action = None


    def get_name(self):
        """ Returns name of the agent """
        return self.name

    def update_epsilon(self, episode_num):
        # Update epsilon. Is the epsilon decreasing too fast?
        epsilon = self.a/(self.a + (episode_num/100))
        if epsilon < 0.01:
            epsilon = 0.01
        
        self.epsilon = epsilon

    def get_action(self, observation, episodes, evaluation=False):
        """ Returns the next action of the agent """

        # Is this correct?
        x = torch.from_numpy(self.preprocess(observation)).float().to(self.train_device)
        aprob = self.policy.forward(x)

        # Printing action probalities that softmax returns
        print(aprob)
        # Stochastic exploration, we can try this at some point
        #m = Categorical(aprob)
        #action = m.sample().item()
        
        # Greedy exploration (Jagusta & Zaguero magic)
        # if there is exploration, explores on the same direction 5 steps

        if self.grid_count == 5:
            self.grid_action = None
            self.grid_count = 0
        
        if self.grid_action in [0, 1, 2]:
            self.grid_count += 1
            return self.grid_action, aprob

        # Epsilon_greedy exploration
        if np.random.random() <= self.epsilon and not evaluation:
            action = int(np.random.random()*3)
            self.grid_action = action
            #print(action)
        else:
            action = torch.argmax(aprob).item()
            #print(action)
        
        return action, aprob

    def reset(self):
        """ Resets the agent to inital state """
        return self.env.reset()


    def episode_finished(self, episode_num):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # Calculate Policy Gradient values
        weighted_probs = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        # use backpropagation to feed information backward to neural network
        loss.backward()

        # Update epsilon value
        self.update_epsilon(episode_num)

        # Updates policy. In default batch_size update is done after each episode
        if (episode_num+1) % self.batch_size == 0: 
            
            self.update_policy()

        if episode_num % 4000 == 0:
            self.save_model_run()

    def update_policy(self):
        # Should these be somewhere else or first zero_grad and then step?
        # source : https://gist.github.com/nailo2c/09c3fd3a92fe212dea8f97ac5c7a1043
        # line 99
        self.optimizer.step()
        self.optimizer.zero_grad()


    def store_outcome(self, observation, action_dist, action_taken, reward):
        dist = torch.distributions.Categorical(action_dist)
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        # Action probalilities
        log_action_prob = -dist.log_prob(action_taken)

        # Saving observations, actions and rewards to a list
        self.observations.append(observation)
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))

    # Reshape preprosessed image so it goes nicely to neural network 
    def to_tensor(self, x):
        x = np.array(x)
        x = x.reshape(-1, 1, 105, 100)
        # Are width and height wrong way around?
        return x


    def preprocess(self, image):
        # Ball 5x5px, paddle 20x5px
        # Remove colors, convert to black and white (0,1)
        image = image[:,:,0] + image[:,:,1] + image[:,:,2]
        image[image !=0 ] = 1
        # Downsample by 2 
        image = image[::2,::2]
        #plt.imshow(image)
        #plt.show()
        return self.to_tensor(image)

    def init_run_model_file_name(self,):
        model_file = "Pong_params_run.mdl"
        i = 1
        while os.path.isfile(model_file):
            model_file = "Pong_params_run%s.mdl" % i
            i += 1
        return model_file

    def save_model_run(self):
        torch.save(self.policy.state_dict(), self.model_file)
        print("Model saved to: ", self.model_file)

