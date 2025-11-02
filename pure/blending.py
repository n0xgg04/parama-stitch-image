"""
Pure implementation of image blending for panorama stitching
using only NumPy - no OpenCV dependencies.
"""

import numpy as np


class ImageBlender:
    """
    Image blending for seamless panorama stitching.
    
    Uses weighted blending with Gaussian smoothing to create
    smooth transitions between overlapping regions.
    """
    
    def __init__(self, smoothing_window_percent=0.10):
        """
        Initialize Image Blender.
        
        Args:
            smoothing_window_percent: Percentage of overlap width for smoothing (0-1)
        """
        self.smoothing_window_percent = smoothing_window_percent
    
    def blend_images(self, img1, img2, H):
        """
        Blend two images using homography and weighted blending.
        
        Args:
            img1: First image (left/query image)
            img2: Second image (right/train image)
            H: Homography matrix to warp img2 to img1's coordinate system
            
        Returns:
            blended: Blended panorama image
            canvas_size: Size of the canvas (height, width)
        """
        # Determine canvas size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find corners of img2 in img1's coordinate system
        corners_img2 = np.array([
            [0, 0],
            [w2, 0],
            [w2, h2],
            [0, h2]
        ], dtype=np.float64)
        
        # Transform corners
        corners_img2_homogeneous = np.hstack([corners_img2, np.ones((4, 1))])
        corners_transformed = (H @ corners_img2_homogeneous.T).T
        corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:3]
        
        # Find bounding box
        all_corners = np.vstack([
            [[0, 0], [w1, 0], [w1, h1], [0, h1]],
            corners_transformed
        ])
        
        x_min = int(np.floor(np.min(all_corners[:, 0])))
        x_max = int(np.ceil(np.max(all_corners[:, 0])))
        y_min = int(np.floor(np.min(all_corners[:, 1])))
        y_max = int(np.ceil(np.max(all_corners[:, 1])))
        
        # Translation to make all coordinates positive
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Canvas size
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        
        # Warp img2
        from .homography import warp_perspective
        H_translated = translation @ H
        img2_warped = warp_perspective(img2, H_translated, (canvas_height, canvas_width))
        
        # Place img1 on canvas
        x_offset = -x_min
        y_offset = -y_min
        
        # Handle grayscale and color
        if len(img1.shape) == 2:
            img1 = img1[:, :, np.newaxis]
            img2_warped = img2_warped[:, :, np.newaxis]
        
        channels = img1.shape[2]
        
        # Create canvas
        canvas1 = np.zeros((canvas_height, canvas_width, channels), dtype=np.float32)
        canvas2 = np.zeros((canvas_height, canvas_width, channels), dtype=np.float32)
        
        # Place img1
        canvas1[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1.astype(np.float32)
        
        # Place img2_warped
        canvas2 = img2_warped.astype(np.float32)
        
        # Create masks
        mask1 = self._create_mask(canvas1, canvas2, img1, x_offset, version='left')
        mask2 = self._create_mask(canvas1, canvas2, img1, x_offset, version='right')
        
        # Blend
        blended = canvas1 * mask1 + canvas2 * mask2
        
        # Crop black borders
        blended = self._crop_black_borders(blended)
        
        # Convert back to uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        if channels == 1:
            blended = blended.squeeze()
        
        return blended, (canvas_height, canvas_width)
    
    def _create_mask(self, canvas1, canvas2, img1, x_offset, version='left'):
        """
        Create blending mask with Gaussian smoothing.
        
        Args:
            canvas1: First canvas
            canvas2: Second canvas (warped)
            img1: Original first image
            x_offset: X offset of img1 on canvas
            version: 'left' or 'right'
            
        Returns:
            mask: Blending mask
        """
        h, w = canvas1.shape[:2]
        channels = canvas1.shape[2]
        
        # Create binary masks for valid regions
        mask1_valid = np.any(canvas1 > 0, axis=2)
        mask2_valid = np.any(canvas2 > 0, axis=2)
        
        # Find overlap region
        overlap = mask1_valid & mask2_valid
        
        if not np.any(overlap):
            # No overlap, return binary masks
            if version == 'left':
                return (mask1_valid[:, :, np.newaxis]).astype(np.float32)
            else:
                return (mask2_valid[:, :, np.newaxis]).astype(np.float32)
        
        # Find overlap boundaries
        overlap_cols = np.where(np.any(overlap, axis=0))[0]
        if len(overlap_cols) == 0:
            if version == 'left':
                return (mask1_valid[:, :, np.newaxis]).astype(np.float32)
            else:
                return (mask2_valid[:, :, np.newaxis]).astype(np.float32)
        
        overlap_start = overlap_cols[0]
        overlap_end = overlap_cols[-1]
        overlap_width = overlap_end - overlap_start
        
        # Calculate smoothing window size
        smoothing_window = int(overlap_width * self.smoothing_window_percent)
        smoothing_window = max(10, min(smoothing_window, overlap_width // 2))
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.float32)
        
        if version == 'left':
            # Left image: fade out in overlap region
            mask[mask1_valid] = 1.0
            
            # Create linear gradient in overlap
            for col in range(overlap_start, overlap_end + 1):
                alpha = 1.0 - (col - overlap_start) / max(overlap_width, 1)
                alpha = np.clip(alpha, 0, 1)
                mask[overlap[:, col], col] = alpha
                
        else:
            # Right image: fade in in overlap region
            mask[mask2_valid] = 1.0
            
            # Create linear gradient in overlap
            for col in range(overlap_start, overlap_end + 1):
                alpha = (col - overlap_start) / max(overlap_width, 1)
                alpha = np.clip(alpha, 0, 1)
                mask[overlap[:, col], col] = alpha
        
        # Apply Gaussian smoothing to mask
        mask = self._gaussian_smooth(mask, sigma=smoothing_window / 6.0)
        
        # Expand to all channels
        mask = np.stack([mask] * channels, axis=2)
        
        return mask
    
    def _gaussian_smooth(self, image, sigma=1.0):
        """
        Apply Gaussian smoothing to image.
        
        Args:
            image: Input image
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            Smoothed image
        """
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=sigma, mode='nearest')
    
    def _crop_black_borders(self, image):
        """
        Crop black borders from image.
        
        Args:
            image: Input image
            
        Returns:
            Cropped image
        """
        if len(image.shape) == 3:
            # Color image
            mask = np.any(image > 0, axis=2)
        else:
            # Grayscale
            mask = image > 0
        
        if not np.any(mask):
            return image
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return image
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return image[y_min:y_max+1, x_min:x_max+1]


class MultiImageBlender:
    """
    Blender for multiple images (panorama from 3+ images).
    """
    
    def __init__(self, smoothing_window_percent=0.10):
        """
        Initialize Multi-Image Blender.
        
        Args:
            smoothing_window_percent: Percentage of overlap width for smoothing
        """
        self.blender = ImageBlender(smoothing_window_percent)
    
    def blend_multiple(self, images, homographies):
        """
        Blend multiple images into panorama.
        
        Args:
            images: List of images
            homographies: List of homography matrices
            
        Returns:
            Blended panorama
        """
        if len(images) == 0:
            return None
        
        if len(images) == 1:
            return images[0]
        
        # Start with first two images
        result, _ = self.blender.blend_images(images[0], images[1], homographies[0])
        
        # Progressively blend remaining images
        for i in range(2, len(images)):
            result, _ = self.blender.blend_images(result, images[i], homographies[i-1])
        
        return result

