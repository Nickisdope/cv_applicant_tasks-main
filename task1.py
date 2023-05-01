"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# params = list(grouped_layer.parameters())
# print(params)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, w_torch=None, b_torch=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.grouped_conv2d = nn.ModuleList()
        for i in range(groups):
            temp_conv2d = nn.Conv2d(in_channels//groups, out_channels//groups, kernel_size, stride=stride, padding=padding, bias=bias)
            if w_torch is not None:
                temp_conv2d.weight.data = w_torch[(out_channels//groups)*i:(out_channels//groups)*(i+1) , :, :, :]
            if b_torch is not None:
                temp_conv2d.bias.data = b_torch[(out_channels//groups)*i:(out_channels//groups)*(i+1)]
            self.grouped_conv2d.append(temp_conv2d)


    def forward(self, x):
        output_list = []
        for i in range(self.groups): 
            output_list.append(self.grouped_conv2d[i](x[:, (self.in_channels//self.groups) * i: (self.in_channels//self.groups) * (i+1), :, :]))
        out = torch.cat(output_list, dim=1)

        return out

# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
custom_grouped_layer = CustomGroupedConv2D(64, 128, 3, stride=1, padding=1, groups=16, bias=True, w_torch=w_torch, b_torch=b_torch)

y_custom = custom_grouped_layer(x)
assert torch.allclose(y, y_custom, atol=1e-06)



        
