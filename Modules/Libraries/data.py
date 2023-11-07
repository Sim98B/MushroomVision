"""
Contains functions to inspect and manipulate data
"""
import os
import zipfile
from pathlib import Path
import requests

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
