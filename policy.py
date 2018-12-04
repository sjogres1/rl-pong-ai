import numpy as np
import torch
import torch.nn.functional as F
#from utils import discount_rewards, softmax_sample


class Policy(torch.nn.Module):
    def __init__(self, action_space):
       
        super().__init__()
        # Create convolutional neural network
        # stride could be betweeen 2-3
        self.conv_1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv_2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Create linear neural networks
        self.lin1 = torch.nn.Linear(5184, 700) # Default was 784. Jagusta run one with 200 and one with 784
        # Maybe put one more layer here, hint from Oliver if still does not work
        self.lin2 = torch.nn.Linear(700, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight, -1e-3,1e-3)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Run preprosecced image trought convolutional nn and activate with relu functions
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
       
        # Relu layers return Tensor size of [1, 64, 9, 9]
        # Reshape the Tensor to match fit linear nn
        x = x.view(-1, 5184)
          # Going through linear neural network layer with relu function
        x = F.relu(self.lin1(x))

        #Output/activation of the last neural network that gives action (3 actions in total)
        x = self.lin2(x)

        # Softmax returns a probality of each action.

        # If still does not learn, we can try to normalize the x values before feeding them to softmax
         x_max = torch.max(x)
        #x = x - torch.mean(x)
         x = torch.div(x,x_max)
        
        return F.softmax(x, dim=-1) # should this be 1?







