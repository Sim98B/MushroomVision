import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

def eval_step(model,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              metric: str,
              device: torch.device):
  
  """
  Performs a single evaluation step on a given model.

  Args:
    model (torch.nn.Module): model to train
    dataloader (torch.utils.data.DataLoader): a dataloader containing test data
    loss_function (torch.nn.Module): a loss function used to track model's performance
    metric (str): 'accuracy' or 'f1' -> sklearn.metrics is used
    device (torch.device): 'cpu' or 'cuda', where to train the model

  Returns:
    Test loss and test matric values
  """
  metric_dict = {"accuracy":accuracy_score,
                 "f1":f1_score}

  model.to(device)
  model.eval()

  test_loss, test_metric = 0, 0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_pred_logits = model(X)
      loss = loss_function(y_pred_logits, y)
      test_loss += loss.item() 
      y_pred_labels = y_pred_logits.argmax(dim=1)
      if metric == "f1" and len(np.unique(y)):
        test_metric += metric_dict[metric](y.detach().numpy(), y_pred_labels.detach().numpy(), average = "weighted")
      else:
        test_metric += metric_dict[metric](y.detach().numpy(), y_pred_labels.detach().numpy())

  test_loss = test_loss / len(dataloader)
  test_metric = test_metric / len(dataloader)

  return test_loss, test_metric

def make_predictions(model: nn.Module,
                     test_dataloader: torch.utils.data.DataLoader,
                     device: torch.device = "cpu"):
  """
  Makes predictions using a torch.nn.Module trained model

  Args:
    model (nn.Module): a trained model
    test_dataloader (torch.utils.data.DataLoader): dataloader containing data to make predictions

  Returns:
    A list containing predictions
  """
  predictions = list()

  model.to(device)
  model.eval()

  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)
      pred_logits = model(X)
      pred_labels = torch.argmax(pred_logits, dim = 1)
      predictions.extend(pred_labels.tolist())


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module, 
          optimizer: torch.optim.Optimizer,
          metric: str,
          epochs: int,
          device: torch.device,
          verbose: int = 1,
          seed: int = 42):
  
  """
  Trains and evaluates a model for a given number of epochs.

  Args:
    model: torch.nn.Module, 
    train_dataloader (torch.utils.data.DataLoader): a dataloader containing train data 
    test_dataloader (torch.utils.data.DataLoader): a dataloader containing test data
    loss_function (torch.nn.Module): a loss function used to track model's performance in both training and testing steps
    optimizer (torch.optim.Optimizer): an optimizer to adjust model's parameters
    metric (str): 'accuracy' or 'f1' -> sklearn.metrics is used
    epochs (int): number of steps, training and evaluation, to be performed
    device (torch.device): 'cpu' or 'cuda', where to train the model
    verbose (int): printing verbosity: '1' -> update progress bar's postfix; '2' -> print losses and metrics for each epoch
    seed (int): a number for reporducibility

  Returns:
    A dictionary of training and testing loss as well as training and
    testing metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_metric: [...],
                  test_loss: [...],
                  test_metric: [...]}
  """

  results = {"train_loss": [],
             "train_metric": [],
             "test_loss": [],
             "test_metric": []
             }
  
  progress_bar = tqdm(range(epochs), desc = "Training")

  for epoch in range(epochs):
    train_loss, train_metric = train_step(model = model,
                                                    dataloader = train_dataloader,
                                                    loss_function = loss_function,
                                                    optimizer = optimizer,
                                                    metric = metric,
                                                    device = device,
                                                    seed = seed)
    
    test_loss, test_metric = eval_step(model = model,
                                                 dataloader = test_dataloader,
                                                 loss_function = loss_function,
                                                 metric = metric,
                                                 device = device)
    
    results["train_loss"].append(train_loss)
    results["train_metric"].append(train_metric)
    results["test_loss"].append(test_loss)
    results["test_metric"].append(test_metric)
    
    if verbose == 1:
      if metric == "accuracy":
        progress_bar.set_postfix({"Train loss": f"{train_loss:.3f}",
                                  f"Train {metric.capitalize()}": f"{train_metric:.2%}",
                                  "Test loss": f"{test_loss:.3f}",
                                  f"Test {metric.capitalize()}": f"{test_metric:.2%}"})
      else:
        progress_bar.set_postfix({"Train loss": f"{train_loss:.3f}",
                                  f"Train {metric.capitalize()}": f"{train_metric:.2f}",
                                  "Test loss": f"{test_loss:.3f}",
                                  f"Test {metric.capitalize()}": f"{test_metric:.2f}"})
    else:
      if metric == "accuracy":
        print(f"Epoch {epoch:02.0f}: Train Loss: {train_loss:.3f} | Train {metric.capitalize()}: {train_metric:.2%} | Test Loss: {test_loss:.3f} | Test {metric.capitalize()}: {test_metric:.2%}")
      else:
        print(f"Epoch {epoch:02.0f}: Train Loss: {train_loss:.3f} | Train {metric.capitalize()}: {train_metric:.2f} | Test Loss: {test_loss:.3f} | Test {metric.capitalize()}: {test_metric:.2f}")
    progress_bar.update(1)
  progress_bar.close()

  return results

def train_step(model,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metric: str,
               device: torch.device,
               seed: int = 42):
  
  """
  Performs a single training step on a given model, including loss backward and parameters optimization.

  Args:
    model (torch.nn.Module): model to train
    dataloader (torch.utils.data.DataLoader): a dataloader containing train data
    loss_function (torch.nn.Module): a loss function used to track model's performance
    optimizer (torch.optim.Optimizer): an optimizer to adjust model's parameters
    metric (str): 'accuracy' or 'f1' -> sklearn.metrics is used
    device (torch.device): 'cpu' or 'cuda', where to train the model
    seed (int): a number for reporducibility

  Returns:
    Train loss and train matric values
  """

  metric_dict = {"accuracy":accuracy_score,
                 "f1":f1_score}

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  model.to(device)
  model.train()

  train_loss, train_metric = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    train_loss += loss.item() 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    if metric == "f1" and len(np.unique(y)):
      train_metric += metric_dict[metric](y.detach().numpy(), y_pred_class.detach().numpy(), average = "weighted")
    else:
      train_metric += metric_dict[metric](y.detach().numpy(), y_pred_class.detach().numpy())

  train_loss = train_loss / len(dataloader)
  train_metric = train_metric / len(dataloader)

  return train_loss, train_metric
