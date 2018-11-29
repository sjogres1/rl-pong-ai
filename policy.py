import numpy as np
import torch
import torch.nn.functional as F
#from utils import discount_rewards, softmax_sample


class Policy(torch.nn.Module):
    def __init__(self, action_space):
        # Create neural network
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin1 = torch.nn.Linear(3136, 784)
        self.lin2 = torch.nn.Linear(784, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight, -1e-3,1e-3)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        x = self.lin2(x)
        return F.softmax(x, dim=-1)


    def preprocess(self, image):
        # Ball 5x5px, paddle 20x5px
        # Remove colors
        image = image[:,:,0] + image[:,:,1] + image[:,:,2]
        image[image !=0 ] = 1
        # Downsample
        # should we compress image twice more?
        image = image[::2,::2]
        self.observation = image






