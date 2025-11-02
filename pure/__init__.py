"""
Pure implementation of panorama image stitching without OpenCV.

This package provides a complete implementation of panorama stitching
using only NumPy, SciPy, and Pillow (no OpenCV).

Main components:
- SIFT: Scale-Invariant Feature Transform
- Feature Matching: Brute-force matcher with Lowe's ratio test
- Homography: RANSAC-based homography estimation
- Blending: Weighted blending with Gaussian smoothing

Example usage:
    from pure.image_io import read_images, write_image
    from pure.panorama_stitcher import PanoramaStitcher
    
    images = read_images(['img1.jpg', 'img2.jpg'])
    stitcher = PanoramaStitcher()
    panorama = stitcher.stitch_multiple(images)
    write_image('output.jpg', panorama)
"""

__version__ = '1.0.0'
__author__ = 'Pure Panorama Team'

from .sift import SIFT
from .matcher import FeatureMatcher
from .homography import HomographyEstimator, warp_perspective
from .blending import ImageBlender
from .panorama_stitcher import PanoramaStitcher
from .image_io import read_image, write_image, read_images

__all__ = [
    'SIFT',
    'FeatureMatcher',
    'HomographyEstimator',
    'warp_perspective',
    'ImageBlender',
    'PanoramaStitcher',
    'read_image',
    'write_image',
    'read_images',
]

