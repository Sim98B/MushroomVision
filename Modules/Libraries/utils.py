"""
Utility tools
"""

import random
import torch
import torch.nn as nn

def delta_time(start_time: float,
               end_time: float):
  
  """
  Print time difference. Useful to track models' training and testing time.
  [NOTE] You need to import -> 'from timeit import default_timer as timer' to use 'timer()' to track time:

  Args:
    start_time (float): starting time
    end_time (float): ending time

  Returns:
    Time difference formatted in hh:mm:ss

  Example:
    start_time = timer()
    ...your_code...
    end_time = timer()

    delta_time(start_time = strat_time, end_time = end_time)
  """
  total_diff = end_time - start_time
  hours = total_diff // 3600
  min = hours // 60
  sec = total_diff % 60

  print(f"Total time: {hours:.0f}:{min:.0f}:{sec:.0f}")

def linear_baseline(input_height: int,
                    input_width: int,
                    color_channels: int,
                    output_shape: int,
                    num_layers: int = 3,
                    num_neurons: int = 10):
  
  """
  Creates a multi layer linear torch neural network to be used as baseline.
  First layer is a nn.Flatten() to prepare the imges to pass through the network
  There are 'num_layers' hidden layers all with 'num_neurons' units paired with a nn.ReLU() function
  Last layer has 'output_shape' output neurons

  Args:
    input_height (int): height of images in pixels
    input_width (int): width of images in pixels
    color_channels (int): number of color channels [1 -> B/W, 3 -> RGB]
    output_shape (int): number of target's classes
    num_layers (int): number of hidden layers
    num_neurons (int): num of neurons in each hidden layer

  Returns:
   A model in the form:
    Flatten -> Hidden layers stack -> output layer
  """

  class Baseline(nn.Module):
    def __init__(self, input_height, input_width, output_shape, num_layers, num_neurons):
      super(Baseline, self).__init__()
      flatten_neurons = input_height * input_width * color_channels
      layers = [nn.Flatten()]
      layers.append(nn.Linear(flatten_neurons, num_neurons))
      layers.append(nn.ReLU())
      for _ in range(num_layers - 1):
        layers.append(nn.Linear(num_neurons, num_neurons))
        layers.append(nn.ReLU())
      layers.append(nn.Linear(num_neurons, output_shape))
      self.model = nn.Sequential(*layers)

    def forward(self, x):
      return self.model(x)

  modello = Baseline(input_height, input_width, output_shape, num_layers, num_neurons)
  return modello

def set_seed(seed: int = 42):
    """
    Set a random state seed for torch, cuda and random.
    
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
