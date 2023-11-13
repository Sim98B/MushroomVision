import torch
from torch import nn
import torchvision
from torchvision.models import alexnet, vgg16, densenet121, resnet50
from torchvision.models import AlexNet_Weights, VGG16_Weights, DenseNet121_Weights, ResNet50_Weights

def create_alexnet(output_shape: int):

  """
  Creates an AlexNet as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes

  Returns:
    model: AlexNet with 'output_shape' output neurons
    model_transformer: AlexNet transformer
  """
  
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  model = alexnet(weights = AlexNet_Weights.DEFAULT)
  model_transformer = AlexNet_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier[6] = nn.Linear(in_features=4096, out_features=output_shape, bias=True)

  return model, model_transformer

def create_densenet(output_shape: int):

  """
  Creates an DenseNet121 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes

  Returns:
    model: DenseNet121 with 'output_shape' output neurons
    model_transformer: DenseNet121 transformer
  """

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  model = densenet121(weights = DenseNet121_Weights.DEFAULT)
  model_transformer = DenseNet121_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Linear(in_features=1024, out_features=4, bias=True)

  return model, model_transformer

def create_resnet(output_shape: int):

  """
  Creates an ResNet50 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes

  Returns:
    model: ResNet50 with 'output_shape' output neurons
    model_transformer: ResNet50 transformer
  """

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  model = resnet50(weights = ResNet50_Weights.DEFAULT)
  model_transformer = ResNet50_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Linear(in_features=2048, out_features=4, bias=True)

  return model, model_transformer

def create_vgg(output_shape: int):

  """
  Creates an VGG16 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes

  Returns:
    model: VGG16 with 'output_shape' output neurons
    model_transformer: VGG16 transformer
  """

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  model = vgg16(weights = VGG16_Weights.DEFAULT)
  model_transformer = VGG16_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier[6] = nn.Linear(in_features=4096, out_features=4, bias=True)

  return model, model_transformer
