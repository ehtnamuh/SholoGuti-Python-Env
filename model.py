import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
  def __init__(self, input_size: int, encoding_size: int, output_size: int):
    """
    Creates a neural network with two static layers of encoding_size size
    """
    super(NN, self).__init__()
    self.dense1 = nn.Linear(input_size, encoding_size)
    self.dense2 = nn.Linear(encoding_size, output_size)

  def forward(self, x):
    hidden = self.dense1(x)
    hidden = F.relu(hidden)
    hidden = self.dense2(hidden)
    return hidden
