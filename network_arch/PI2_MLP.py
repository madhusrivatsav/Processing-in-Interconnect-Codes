import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

current_dir = "" ##path of your current directory
layer_dir = os.path.join(current_dir, 'PI2_Layers')
sys.path.append(layer_dir)
print(layer_dir)
from TEMP_FC_in import MPLayer_in_K

#conventional MLP network with relu activation
class MLP_Network(torch.nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list):
            super(MLP_Network, self).__init__()
            self.in_features = in_features
            self.num_layers = num_layers
            self.num_hidden_list = num_hidden_list

            layers = []
            layers.append(nn.Linear(in_features, num_hidden_list[0],bias=False))
            for idx in range(num_layers - 1):
                layers.append(nn.Linear(num_hidden_list[idx], num_hidden_list[idx+1],bias=False))
            self.linear = torch.nn.Sequential(*layers)

    def forward(self, inputs):
            inputs = inputs.flatten(start_dim=1)
            linear= self.linear[0](inputs)
            linear = F.relu(linear)
            linear = self.linear[1](linear)
            return linear

#PI^2_K MLP network         
class TEMP_Network_hybrid_nobn(torch.nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list,gamma=[]):
            super(TEMP_Network_hybrid_nobn, self).__init__()
            self.in_features = in_features
            self.num_layers = num_layers
            self.num_hidden_list = num_hidden_list
            self.gamma = gamma
            if(len(gamma)> 0):
                self.fc1 = MPLayer_in_K(in_features, num_hidden_list[0], gamma[0],diff=0)
            else:
                self.fc1 = nn.Linear(in_features, num_hidden_list[0],bias=False)
            if(len(gamma) > 1):
                self.fc2 = MPLayer_in_K(num_hidden_list[0], num_hidden_list[1], gamma[1],diff=0)
            else:
                self.fc2 = nn.Linear(num_hidden_list[0], num_hidden_list[1],bias=False)

    def forward(self, inputs):
            inputs = inputs.flatten(start_dim=1)
            if(len(self.gamma) > 0):
                out = F.relu(-10*self.fc1(inputs))
            else:
                out = F.relu(self.fc1(inputs))
            if(len(self.gamma) > 1):
                out = (-10*self.fc2(out))
            else:
                out = self.fc2(out)
            return out
