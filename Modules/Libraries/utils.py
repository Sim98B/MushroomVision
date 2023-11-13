"""
Utility tools
"""

from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None):

    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp(yyyy/mm/dd)/experiment_name/model_name/extra.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """
                    
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def delta_time(start_time: float,
               end_time: float,
               desc: str = "Total time"):
  
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
  min = (total_diff % 3600) // 60
  sec = total_diff % 60

  print(f"{desc}: {hours:02.0f}:{min:02.0f}:{sec:02.0f}")
                 
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

import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true: list,
                          y_pred: list,
                          norm: str = None,
                          title: str = "Confusion Matrix",
                          class_names: list = None,
                          color_map: str = "viridis",
                          figsize: int = 7,):
  
  """
  Plot a confusion matrix to evaluate classifiers' performance.

  Args:
    y_true (list): a list containing real labels
    y_pred (list): a list containing predicted labels
    norm (str): ['true', 'pred', 'all'] whether to normalize or not the data
    class_names (list): a list containing classes' names to be used for ticks
    color_map (str, matplotlib.pyplot.cmap): colormap for the confusion matrix
    figsize (int): a height and width of matrix

  Returns:
    A squared heatmap containg confusion matrix data
  """

  fig = plt.figure(figsize = (figsize, figsize))
  cm = confusion_matrix(y_true = y_true, y_pred = y_pred, normalize = norm)

  if norm != None:
    hm = sns.heatmap(cm, annot = True, robust = True, annot_kws = {"weight":"bold"}, cmap = color_map, cbar = False, fmt = ".2%")
  else:
    hm = sns.heatmap(cm, annot = True, robust = True, annot_kws = {"weight":"bold"}, cmap = color_map, cbar = False)

  if class_names:
    hm.set_xticklabels(class_names)
    hm.set_yticklabels(class_names)

  hm.set_xlabel("Predicted", weight = "bold")
  hm.set_ylabel("True", weight = "bold")
  hm.set_title(title, fontsize = 20, weight = "bold")

  plt.tight_layout();

def plot_model_performance(results_dict: dict,
                           fig_size: tuple = (15,5),
                           metric_name: str = "Metric",
                           title: str = " "):
  
  """
  Makes subplots of both loss and a metric comapring training and validation

  Args:
    results_dict (dict): a dict in the form {train_loss: [...],
                                             train_metric: [...],
                                             test_loss: [...],
                                             test_metric: [...]}
    metric_name (str): metric name to display as y axis' label

  Returns:
    Subplots of lineplot to comapre performance on train and validation set
  """

  ax, fig = plt.subplots(figsize = fig_size)

  ax1 = plt.subplot(1, 2, 1)
  plt.plot(results_dict["train_loss"], label = "Train")
  plt.plot(results_dict["test_loss"], label = "Validation")
  plt.xlabel("Epochs", weight = "bold")
  plt.title("Loss", weight = "bold")
  plt.legend(frameon = False)

  ax2 = plt.subplot(1, 2, 2)
  plt.plot(results_dict["train_metric"], label = "Train")
  plt.plot(results_dict["test_metric"], label = "Validation")
  plt.xlabel("Epochs", weight = "bold")
  plt.title(f"{metric_name}", weight = "bold")
  plt.legend(frameon = False)

  plt.suptitle(f"{title}", weight = "bold", fontsize = 20)

import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
  """

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"),
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj = model.state_dict(), f = model_save_path)

def set_seed(seed: int = 42):
    """
    Set a random state seed for torch, cuda and random.
    
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
