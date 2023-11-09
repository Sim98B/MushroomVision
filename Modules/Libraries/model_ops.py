import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_step(model,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               metric: str,
               device: torch.device):

  metric_dict = {"accuracy":accuracy_score,
                 "f1":f1_score}

  utils.set_seed()

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
