"""
Tools to inspect and manipulate data
"""
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import requests
import random
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import zipfile

def dataloader(data_dir: str,
               transformer: torchvision.transforms = torchvision.transforms.ToTensor(),
               target_transformer: torchvision.transforms = None,
               bs: int = 32,
               shuffle: bool = True,
               workers: int = os.cpu_count(),
               memory: bool =True):
  
  """
  Create a torch DataLoader via ImageFolder and a list containing class names

  Args:
    data_dir (str or Path): path to a data directory to pass to create the ImageFolder
    transformer (torchvision.transforms): transformer/s through which the images pass
    target_transformer (torchvision.transforms): transformer for the target
    bs (int): batch size for DataLoader
    shuffle (bool): whether to shuffle or not the data
    workers (int): passed as DataLoader's num_workers
    memory (bool): whether to pin memory

  Returns:
    A torch DataLoader and a list with classes'name.
  """
  
  data_image_folder = datasets.ImageFolder(root = data_dir, 
                                           transform = transformer, 
                                           target_transform = target_transformer)
  
  class_names = data_image_folder.classes

  loader = DataLoader(dataset = data_image_folder,
                      batch_size = bs,
                      shuffle = shuffle,
                      num_workers = workers,
                      pin_memory = memory)
  
  return loader, class_names
    
def inspect_dir(dir_path: Path):
    """
    Navigate through a directory listing and enumerating folders and files.
    
    Args:
        dir_path (str or pathlib.Path): target directory

    Returns:
        A print of:
        1. Number of directories in each parent folder
        2. Number of files in each directory
    """
    for path, dirname, filename in os.walk(dir_path):
        print(f"There are {len(dirname)} directory/ies and {len(filename)} file/s in {path}")


def plot_dataloader_img(dataloader: torch.utils.data.DataLoader,
                        n_images: int = 3,
                        classes: list = None,
                        seed: bool = False,
                        num_seed: int = 42):

  """
  Plot images from a dataloader; useful to take a look at transformed images.

  Args:
    dataloader(torch.utils.data.DataLoader): an iterable containing images
    n_images (int): how many images to plot
    classes (list): optional list containing class name to use as title for images
  
  Returns:
    A row of images
  """

  if seed:
    utils.set_seed(num_seed)

  iterator = iter(dataloader)
  imgs, labels = next(iterator)

  rnd_idx = random.sample(range(len(imgs)), n_images)

  fig, axs = plt.subplots(nrows=1, ncols=n_images, figsize=(n_images * 2, 5))

  for i, indice in enumerate(rnd_idx):
    img = imgs[indice]
    label = labels[indice]
        
    ax = axs[i] if n_images > 1 else axs  # Gestisci il caso in cui n_images sia 1
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')

    if classes:
      ax.set_title(f"{classes[label].replace('_', ' ')}")

  plt.tight_layout()  # Organizza gli assi in modo appropriato
  plt.show()


def plot_random_images(data_path: str,
                       target_folder: str,
                       seed: bool = False,
                       seed_num: int = 42,
                       img_per_classe: int = 3,
                       img_size: tuple = (224, 224),
                       plot_size: tuple = (16, 9)):


  """
  Show subplots with a set number of random images per class from a folder.

  Args:
    data_path (str or Path): path to data directory
    target_folder (str or Path): path to target folder
    seed (bool): whether to set a seed for reproducibility
    seed_num (int): the random state number to pass at set_seed() function
    img_per class (int > 2): number of images to pick from each class
    img_size (tuple): tuple of height x width to resize every image
    plot_size (tuple): matplotlib figsize height x width

  Returns:
    A subplot img_per_class x number of classes
  """

  if seed:
    utils.set_seed(seed_num)

  if img_per_classe < 2:
    raise Exception("img_per_class should be >= 2")

  classes = [folder.name for folder in data_path.joinpath(target_folder).glob('*')]
  fig, axs = plt.subplots(img_per_classe, len(classes), figsize=(16, 9))

  for i, class_name in enumerate(classes):
    img_folder = data_path.joinpath(target_folder, class_name)
    images = list(img_folder.glob('*.jpg'))
    random_images = random.sample(images, img_per_classe)

    for j, image_path in enumerate(random_images):
      img = Image.open(image_path)
      if img_size:
        img = img.resize(img_size)
      axs[j, i].imshow(img)
      axs[j, i].axis('off')
      if j == 0:
        axs[j, i].set_title(class_name.replace("_", " "), weight="bold")
    
    plt.suptitle(f"Showing {img_per_classe} images per class from {target_folder} folder", fontsize = 20)

  plt.tight_layout();
