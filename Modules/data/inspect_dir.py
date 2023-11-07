def inspect_dir(dir_path: Path):
  """
  Navigate through a directory listing and enumerating folders and files.

  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print of:
    1. 
  """
    import os
    try:
        from pathlib import Path
    except:
        pip install pathlib
    for path, dirname, filename in os.walk(dir_path):
    print(f"There are {len(dirname)} directory/ies and {len(filename)} file/s in {path}")
