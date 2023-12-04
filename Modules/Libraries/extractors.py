from typing import Literal
import torch
from torch import nn
import torchvision
from torch vision import transforms
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
                 "efficientnet_b1" : torchvision.models.efficientnet_b1,
                 "efficientnet_v2l" : torchvision.models.efficientnet_v2_l,
                 "resnet50": torchvision.models.resnet50,
                 "vgg16": torchvision.models.vgg16}

  weights_dict = {"alexnet" : torchvision.models.AlexNet_Weights,
                 "densenet121" : torchvision.models.DenseNet121_Weights,
                 "densenet161" : torchvision.models.DenseNet161_Weights,
                 "densenet169" : torchvision.models.DenseNet169_Weights,
                 "densenet201" : torchvision.models.DenseNet201_Weights,
                 "efficientnet_b1" : torchvision.models.EfficientNet_B1_Weights,
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

def resnet9(in_channels: int,
            num_classes: int):
  
  """
  Creates a ResNet9 with his tensor for a 75% accuracy on CIFAR100

  Args:
    in_channels (int): number of input channels of a convolutional block
    num_classes (int): number of classes in the output layer

  Returns:
    model: ResNet9 model
    transform: transforms to be applied to images
  """
  def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

  transform = transforms.Compose([transforms.Resize(size = (32, 32)),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                    std=[0.26733428587941854, 0.25643846292120615, 0.2761504713263903])])

  class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.Linear(1028, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out

  model = ResNet9(in_channels, num_classes)

  return model, transform
