from PIL import Image, ImageOps
from PIL.Image import Image as PILImage

def open_image(path: str) -> PILImage:
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')