import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

import copy

class ReLU_FC_net(torch.nn.Module):
    def __init__(self,input_length,num_layers,width,output_width, weight_std=0.5, bias_std=0.5, depth_scaling=None):

        super(ReLU_FC_net, self).__init__()

        assert(num_layers >= 1)
        self.num_layers = num_layers
        self.width = width

        linear_list = []
        linear_list.append(nn.Linear(input_length, width,bias=True)) # One d x w
        for i in range(num_layers-1):
            # One w x w
            linear_list.append(nn.Linear(width, width,bias=True))
            
        linear_list.append(nn.Linear(width, output_width, bias=True)) # One w x d_{out}

        self.linears = nn.ModuleList(linear_list)


        # Initialize according to width MUP
        for i, l in enumerate(self.linears):
            if i == 0: # d x w
                bwi = 0.5
                bbi = 0.5
            elif i == len(self.linears) - 1: # w x d_{out}
                bwi = 0.5
                bbi = 0
            else:
                bwi = 0.5
                bbi = 0.5

            torch.nn.init.normal_(l.weight, std=weight_std * math.pow(width, -bwi))
            torch.nn.init.normal_(l.bias, std=bias_std * math.pow(width, -bbi))




    def forward(self, x, store_preact=None):

        x = torch.reshape(x, (x.shape[0], -1))
        
        linear_len = len(self.linears)

        # 1)
        curr_linear = 0
        x = F.linear(x, math.pow(self.width, 0.5) * self.linears[curr_linear].weight, math.pow(self.width,0.5) * self.linears[curr_linear].bias)
        x = F.relu(x)

        # 2)
        for layer_num in range(1,linear_len-1):

            curr_linear = layer_num
            x = F.linear(x, self.linears[curr_linear].weight, math.pow(self.width, 0.5) * self.linears[curr_linear].bias)
            x = F.relu(x)
            
        if store_preact is not None:
            store_preact['last_layer'] = copy.deepcopy(x)

        # 3)
        curr_linear = linear_len-1
        x = F.linear(x, math.pow(self.width, -0.5) * self.linears[curr_linear].weight, self.linears[curr_linear].bias)

        return x