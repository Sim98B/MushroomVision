import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

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

  for batch, (X, y) in enumerate(tqdm(dataloader, desc = "Training: ")):
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
