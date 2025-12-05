
import os
import yaml
import json
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from .mask_vis import *

def open_image(path: str) -> PILImage:
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')

def list_paths(
    root_dir: str,
    exts: list[str] = None,
    follow_links: bool = True
):
    '''
        Lists all files in a directory with a given extension.

        Arguments:
            root_dir (str): Directory to search.
            exts (list[str]): List of file extensions to consider.

        Returns: List of paths.
    '''
    exts = set(exts) if exts else None
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_links):
        for filename in filenames:
            path = os.path.join(dirpath, filename)

            if not exts or os.path.splitext(path)[1].lower() in exts:
                paths.append(path)

    paths = sorted(paths)

    return paths

def load_yaml(path: str, encoding='utf-8'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            graph_yaml = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"[ {path} ] does not exist in the file system.")
    return graph_yaml


def load_json(path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            json_dict = json.load(f)
    else:
        raise FileNotFoundError(f"[ {path} ] does not exist in the file system.")
    return json_dict