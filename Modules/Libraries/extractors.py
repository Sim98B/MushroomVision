from typing import Literal
import torch
from torch import nn
import torchvision
from torchvision.models import alexnet, efficientnet_b1, efficientnet_v2_l, vgg16, densenet121, densenet161, densenet169, densenet201, resnet50
from torchvision.models import AlexNet_Weights, EfficientNet_B1_Weights, EfficientNet_V2_L_Weights, VGG16_Weights, DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights, ResNet50_Weights

def create_model(model_name: Literal["alexnet", "densenet121", "densenet161", "densenet169", "densenet201", "efficientnet_b1", "efficientnet_v2l", "resnet50", "vgg16"],
                 output_shape: int,
                 seed: int = 42):

  """
  Creates a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' transform to preprocess data
  Available models are: AlexNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201, EfficientNet_2V_L, ResnNet50, VGG16

  Args:
    model_name (str): one architecture among those available
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: correpsonding model with 'output_shape' output neurons
    model_transformer: corresponding transformer
  """
  
  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  models_dict = {"alexnet" : torchvision.models.alexnet,
                 "densenet121" : torchvision.models.densenet121,
                 "densenet161" : torchvision.models.densenet161,
                 "densenet169" : torchvision.models.densenet169,
                 "densenet201" : torchvision.models.densenet201,
                 "efficientnet_b0" : torchvision.models.efficientnet_b1,
                 "efficientnet_v2l" : torchvision.models.efficientnet_v2_l,
                 "resnet50": torchvision.models.resnet50,
                 "vgg16": torchvision.models.vgg16}

  weights_dict = {"alexnet" : torchvision.models.AlexNet_Weights,
                 "densenet121" : torchvision.models.DenseNet121_Weights,
                 "densenet161" : torchvision.models.DenseNet161_Weights,
                 "densenet169" : torchvision.models.DenseNet169_Weights,
                 "densenet201" : torchvision.models.DenseNet201_Weights,
                "efficientnet_b0" : torchvision.models.EfficientNet_B1_Weights,
                 "efficientnet_v2l" : torchvision.models.EfficientNet_V2_L_Weights,
                 "resnet50": torchvision.models.ResNet50_Weights,
                 "vgg16": torchvision.models.VGG16_Weights}

  weights = weights_dict[model_name]

  model = models_dict[model_name](weights = weights)
  model_transform = weights.DEFAULT.transforms()

  if model_name == "alexnet":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier[6] = nn.Linear(in_features = 4096, out_features = output_shape, bias=True)

  elif model_name == "densenet121":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier = nn.Linear(in_features = 1024, out_features = output_shape, bias=True)

  elif model_name == "densenet161":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier = nn.Linear(in_features = 2208, out_features = output_shape, bias=True)

  elif model_name == "densenet169":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier = nn.Linear(in_features = 1664, out_features = output_shape, bias=True)

  elif model_name == "densenet201":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier = nn.Linear(in_features = 1920, out_features = output_shape, bias=True)

  elif model_name == "efficientnet_b1":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier[1] = nn.Linear(in_features=1280, out_features = output_shape, bias=True)

  elif model_name == "efficientnet_v2l":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier[1] = nn.Linear(in_features=1280, out_features = output_shape, bias=True)

  elif model_name == "resnet50":
    for param in model.parameters():
      param.requires_grad = False

    model.fc = nn.Linear(in_features = 2048, out_features = output_shape, bias=True)

  elif model_name == "vgg16":
    for param in model.parameters():
      param.requires_grad = False

    model.classifier[6] = nn.Linear(in_features = 4096, out_features = output_shape, bias=True)

  return model, model_transform
