"""
Downloads a number of images via search terms.
"""

from fastbook import *
import os
from tqdm.auto import tqdm

"""
Args:
    terms: a list of search terms for which to download images
    path: a path to a parent folder that will contain folders with images divided by search term
    total_images: number images to download for each term
    raname: whether or not to rename the images using the name parameter
    name: regex used to rename all images as "regex_index"
    
Return:
    Save the set number of images in the parent folder which contains a number of subfolder 
    equal to the number of search terms.
    
Example of usage:
    download_data(terms = ["cat", "dog"],
                  path = "Dog/Cat_images",
                  total_images = 10
                  rename = True,
                  name = img)
    Dog/Cat_images -> cat -> img_1.jpg, img_2.jpg ...
    Dog/Cat_images -> dog -> img_1.jpg, img_2.jpg ...
"""

def download_data(terms: list,
                  path: str,
                  total_images: int = None,
                  rename: bool = False,
                  name: str = "img"):

    terms = terms
    path = Path(path)
    
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    for term in tqdm(terms, desc = f"Downloading {total_images} images of each term"):
        print(f"{term}")
        destination_folder = path / term
        destination_folder.mkdir(exist_ok=True)
        results = search_images_ddg(f'{term} photo')
        download_images(destination_folder, urls=results[:total_images])
        
        if rename == True:
            listed_file = os.listdir(destination_folder)
            for idx, file_name in enumerate(tqdm(listed_file, desc = f"Renaming file as {name} + index")):
                new_name = f"{name}_{idx+1}.jpg"
                old_path = os.path.join(destination_folder, file_name)
                new_path = os.path.join(destination_folder, new_name)

                os.rename(old_path, new_path)
