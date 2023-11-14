import torch
from torch import nn
import torchvision
from torchvision.models import alexnet, vgg16, densenet121, densenet161, densenet169, densenet201, resnet50
from torchvision.models import AlexNet_Weights, VGG16_Weights, DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights, ResNet50_Weights

def create_alexnet(output_shape: int,
                   seed: int = 42):

  """
  Creates an AlexNet as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: AlexNet with 'output_shape' output neurons
    model_transformer: AlexNet transformer
  """
  
  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = alexnet(weights = AlexNet_Weights.DEFAULT)
  model_transformer = AlexNet_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier[6] = nn.Linear(in_features = 4096, out_features = output_shape, bias=True)

  return model, model_transformer

def create_densenet121(output_shape: int,
                       seed: int = 42):

  """
  Creates an DenseNet121 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: DenseNet121 with 'output_shape' output neurons
    model_transformer: DenseNet121 transformer
  """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = densenet121(weights = DenseNet121_Weights.DEFAULT)
  model_transformer = DenseNet121_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Linear(in_features = 1024, out_features = output_shape, bias=True)

  return model, model_transformer

def create_densenet161(output_shape: int,
                       seed: int = 42):

  """
  Creates an DenseNet161 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: DenseNet161 with 'output_shape' output neurons
    model_transformer: DenseNet161 transformer
  """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = densenet161(weights = DenseNet161_Weights.DEFAULT)
  model_transformer = DenseNet161_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Linear(in_features = 2208, out_features = output_shape, bias=True)

  return model, model_transformer

def create_densenet169(output_shape: int,
                       seed: int = 42):

  """
  Creates an DenseNet169 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: DenseNet169 with 'output_shape' output neurons
    model_transformer: DenseNet169 transformer
  """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = densenet169(weights = DenseNet169_Weights.DEFAULT)
  model_transformer = DenseNet169_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Linear(in_features = 1664, out_features = output_shape, bias=True)

  return model, model_transformer

def create_densenet201(output_shape: int,
                       seed: int = 42):

  """
  Creates an DenseNet201 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: DenseNet201 with 'output_shape' output neurons
    model_transformer: DenseNet201 transformer
  """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = densenet201(weights = DenseNet201_Weights.DEFAULT)
  model_transformer = DenseNet201_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Linear(in_features = 1920, out_features = output_shape, bias=True)

  return model, model_transformer

def create_resnet50(output_shape: int,
                    seed: int = 42):

  """
  Creates an ResNet50 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: ResNet50 with 'output_shape' output neurons
    model_transformer: ResNet50 transformer
  """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = resnet50(weights = ResNet50_Weights.DEFAULT)
  model_transformer = ResNet50_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Linear(in_features = 2048, out_features = output_shape, bias=True)

  return model, model_transformer

def create_vgg16(output_shape: int,
                 seed: int = 42):

  """
  Creates an VGG16 as a feature extractor setting all parameters as not trainable 
  and changing the shape of the classifier layer equals to 'output_shape'.
  Also creates a corresponding 'torchivision.transforms' trasnformer to preprocess data

  Args:
    output_shape (int): number of classes
    seed (int): sedd for reproducibility

  Returns:
    model: VGG16 with 'output_shape' output neurons
    model_transformer: VGG16 transformer
  """

  if seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  model = vgg16(weights = VGG16_Weights.DEFAULT)
  model_transformer = VGG16_Weights.DEFAULT.transforms()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier[6] = nn.Linear(in_features = 4096, out_features = output_shape, bias=True)

  return model, model_transformer
