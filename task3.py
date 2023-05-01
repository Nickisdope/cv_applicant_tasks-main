"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!


# write your code here ...
# model = onnx.load("model/model.onnx")
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))

# define block
# define the residual block
class Residual(nn.Module):
  def __init__(self, input_channels, num_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.bn1 = nn.BatchNorm2d(num_channels)
  
  def forward(self, X):
    Y = self.bn1(self.conv1(X))
    act_Y = F.sigmoid(Y)

    return Y * act_Y

# define the network
class Network(nn.Module):
  def __init__(self, in_features):
    super().__init__()
    
    self.b1 = nn.Sequential(Residual(in_features, 32), Residual(32, 64, stride=2), Residual(64, 64), Residual(64, 128, stride=2))
    
    self.b1_branch_left = Residual(128, 64, kernel_size=1, padding=0)
    self.b1_branch_right_1 = Residual(128, 64, kernel_size=1, padding=0)

    self.b1_branch_right_1_branch_right = nn.Sequential(Residual(64, 64, 3), Residual(64, 64, 3))
    
    self.b1_branch_right_1_branch_right_branch_right =  nn.Sequential(Residual(64, 64, 3), Residual(64, 64, 3))
    
    self.after_concat = Residual(256, 256, kernel_size=1, padding=0)

    self.after_concat_left = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0), Residual(256, 128, kernel_size=1, padding=0))
    self.after_concat_right = nn.Sequential(Residual(256, 128, kernel_size=1, padding=0), Residual(128, 128, kernel_size=3, stride=2))

    self.linear = nn.Linear(256, 256, bias=True)

  def forward(self, x):
      x = self.b1(x)
      concat_1 = self.b1_branch_left(x)
      concat_2 = self.b1_branch_right_1(x)
      concat_3 = self.b1_branch_right_1_branch_right(concat_2)
      concat_4 = self.b1_branch_right_1_branch_right_branch_right(concat_3)
      concat = torch.concatenate((concat_1, concat_2, concat_3, concat_4), axis=1)

      concat = self.after_concat(concat)

      concat_left = self.after_concat_left(concat)

      concat_right = self.after_concat_right(concat)

      concat = torch.concatenate((concat_left, concat_right), axis=1)

      concat = torch.permute(concat, (0, 2, 3, 1))

      # concat_shape = concat.shape

      # concat_shape  = concat.reshape((concat_shape[0], -1))

      concat_after_linear = self.linear(concat)

      concat_after_linear = torch.permute(concat_after_linear, (0, 3, 1, 2))

      out = F.sigmoid(concat_after_linear)


      return out

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(0)

if __name__ == '__main__':
  x = torch.randn((1,3,160,320))
  Net = Network(3)
  Net.apply(init_weights)
  out = Net(x)
