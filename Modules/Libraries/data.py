"""
Contains functions to inspect and manipulate data
"""
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import requests
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import zipfile

def download_data(source: str,
                  remove_source: bool = True) -> Path:
  """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://link/to/file.zip")
    """
  data_path = Path("data/")

  if data_path.is_dir():
    print(f"[INFO] {data_path} directory exists.")
  else:
    print(f"[INFO] Did not find {data_path} directory, creating one...")
    data_path.mkdir(parents=True, exist_ok=True)

  target_file = Path(source).name
  with open(data_path / target_file, "wb") as f:
    request = requests.get(source)
    print(f"[INFO] Downloading {target_file} from {source}...")
    f.write(request.content)

  with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
    print(f"[INFO] Unzipping {target_file} data...") 
    zip_ref.extractall(data_path)

  if remove_source:
    os.remove(data_path / target_file)

  data_path = data_path.joinpath(os.listdir(data_path)[0])
  
  return data_path
    
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
