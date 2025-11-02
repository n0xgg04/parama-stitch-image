"""
Image I/O utilities using PIL (Pillow)
No OpenCV dependencies.
"""

import numpy as np
from PIL import Image


def read_image(filepath):
    """
    Read image from file.
    
    Args:
        filepath: Path to image file
        
    Returns:
        Image as numpy array (H x W x C) for color or (H x W) for grayscale
    """
    try:
        img = Image.open(filepath)
        
        # Convert to RGB if needed
        if img.mode != 'RGB' and img.mode != 'L':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        return img_array
    
    except Exception as e:
        raise IOError(f"Failed to read image from {filepath}: {str(e)}")


def write_image(filepath, image):
    """
    Write image to file.
    
    Args:
        filepath: Path to save image
        image: Image as numpy array
    """
    try:
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(image.shape) == 2:
            # Grayscale
            img = Image.fromarray(image, mode='L')
        else:
            # RGB
            img = Image.fromarray(image, mode='RGB')
        
        # Save
        img.save(filepath)
        
    except Exception as e:
        raise IOError(f"Failed to write image to {filepath}: {str(e)}")


def read_images(filepaths):
    """
    Read multiple images.
    
    Args:
        filepaths: List of image file paths
        
    Returns:
        List of images as numpy arrays
    """
    images = []
    
    for filepath in filepaths:
        img = read_image(filepath)
        images.append(img)
    
    return images


def resize_image(image, scale=1.0, width=None, height=None):
    """
    Resize image.
    
    Args:
        image: Input image
        scale: Scale factor (if width and height not specified)
        width: Target width (optional)
        height: Target height (optional)
        
    Returns:
        Resized image
    """
    if width is not None and height is not None:
        new_size = (width, height)
    else:
        h, w = image.shape[:2]
        new_size = (int(w * scale), int(h * scale))
    
    # Convert to PIL
    if len(image.shape) == 2:
        img = Image.fromarray(image, mode='L')
    else:
        img = Image.fromarray(image, mode='RGB')
    
    # Resize
    img_resized = img.resize(new_size, Image.LANCZOS)
    
    return np.array(img_resized)

