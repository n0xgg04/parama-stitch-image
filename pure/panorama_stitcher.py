"""
Pure implementation of Panorama Stitching pipeline
using only NumPy and mathematical libraries - no OpenCV dependencies.
"""

import numpy as np
from .sift import SIFT
from .matcher import FeatureMatcher
from .homography import HomographyEstimator
from .blending import ImageBlender


class PanoramaStitcher:
    """
    Complete panorama stitching pipeline.
    
    This class coordinates all components:
    1. SIFT feature detection
    2. Feature matching
    3. Homography estimation with RANSAC
    4. Image warping and blending
    """
    
    def __init__(self, 
                 sift_params=None,
                 matcher_params=None,
                 ransac_params=None,
                 blending_params=None):
        """
        Initialize Panorama Stitcher.
        
        Args:
            sift_params: Parameters for SIFT detector
            matcher_params: Parameters for feature matcher
            ransac_params: Parameters for RANSAC
            blending_params: Parameters for blending
        """
        # Initialize SIFT
        sift_params = sift_params or {}
        self.sift = SIFT(**sift_params)
        
        # Initialize matcher
        matcher_params = matcher_params or {}
        self.matcher = FeatureMatcher(**matcher_params)
        
        # Initialize homography estimator
        ransac_params = ransac_params or {}
        self.homography_estimator = HomographyEstimator(**ransac_params)
        
        # Initialize blender
        blending_params = blending_params or {}
        self.blender = ImageBlender(**blending_params)
    
    def stitch_pair(self, img1, img2, return_debug_info=False):
        """
        Stitch two images together.
        
        Args:
            img1: First image (left/query)
            img2: Second image (right/train)
            return_debug_info: If True, return additional debug information
            
        Returns:
            result: Stitched panorama image
            debug_info: (Optional) Dictionary with debug information
        """
        print("  Detecting features in image 1...")
        # Convert to grayscale if needed
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)
        
        # Detect features
        kp1, desc1 = self.sift.detect_and_compute(gray1)
        print(f"    Found {len(kp1)} keypoints in image 1")
        
        print("  Detecting features in image 2...")
        kp2, desc2 = self.sift.detect_and_compute(gray2)
        print(f"    Found {len(kp2)} keypoints in image 2")
        
        if len(kp1) < 4 or len(kp2) < 4:
            raise ValueError("Not enough keypoints detected in images")
        
        # Match features
        print("  Matching features...")
        matches = self.matcher.match(desc1, desc2)
        print(f"    Found {len(matches)} matches")
        
        if len(matches) < 4:
            raise ValueError("Not enough matches found between images")
        
        # Extract matched points
        src_pts = np.array([kp2[m['trainIdx']]['x'] for m in matches])
        src_pts_y = np.array([kp2[m['trainIdx']]['y'] for m in matches])
        src_pts = np.column_stack([src_pts, src_pts_y])
        
        dst_pts = np.array([kp1[m['queryIdx']]['x'] for m in matches])
        dst_pts_y = np.array([kp1[m['queryIdx']]['y'] for m in matches])
        dst_pts = np.column_stack([dst_pts, dst_pts_y])
        
        # Compute homography with RANSAC
        print("  Computing homography with RANSAC...")
        H, inliers = self.homography_estimator.find_homography(src_pts, dst_pts)
        
        if H is None:
            raise ValueError("Failed to compute homography")
        
        num_inliers = np.sum(inliers) if inliers is not None else 0
        print(f"    Found {num_inliers} inliers out of {len(matches)} matches")
        
        # Blend images
        print("  Blending images...")
        result, canvas_size = self.blender.blend_images(img1, img2, H)
        print("  Done!")
        
        if return_debug_info:
            debug_info = {
                'keypoints1': kp1,
                'keypoints2': kp2,
                'descriptors1': desc1,
                'descriptors2': desc2,
                'matches': matches,
                'homography': H,
                'inliers': inliers,
                'num_inliers': num_inliers,
                'canvas_size': canvas_size
            }
            return result, debug_info
        
        return result
    
    def stitch_multiple(self, images, return_debug_info=False):
        """
        Stitch multiple images into panorama.
        
        Args:
            images: List of images (ordered left to right)
            return_debug_info: If True, return debug information
            
        Returns:
            result: Stitched panorama
            debug_info: (Optional) List of debug info for each pair
        """
        if len(images) == 0:
            raise ValueError("No images provided")
        
        if len(images) == 1:
            return images[0]
        
        print(f"\nStitching {len(images)} images...")
        
        # Stitch recursively from right to left
        debug_infos = []
        
        result = images[-1]
        
        for i in range(len(images) - 2, -1, -1):
            print(f"\nStitching image {i+1} with current panorama...")
            
            if return_debug_info:
                result, debug = self.stitch_pair(images[i], result, return_debug_info=True)
                debug_infos.append(debug)
            else:
                result = self.stitch_pair(images[i], result)
        
        if return_debug_info:
            return result, debug_infos
        
        return result
    
    def _to_grayscale(self, image):
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            # RGB to grayscale using standard weights
            return np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return image
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, max_matches=100):
        """
        Create visualization of feature matches.
        
        Args:
            img1: First image
            img2: Second image
            kp1: Keypoints in first image
            kp2: Keypoints in second image
            matches: List of matches
            max_matches: Maximum number of matches to draw
            
        Returns:
            Visualization image with matches drawn
        """
        # Limit number of matches
        matches_to_draw = matches[:max_matches]
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create side-by-side image
        h = max(h1, h2)
        w = w1 + w2
        
        if len(img1.shape) == 3:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1+w2] = img2
        else:
            vis = np.zeros((h, w), dtype=np.uint8)
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1+w2] = img2
            # Convert to RGB for colored lines
            vis = np.stack([vis, vis, vis], axis=2)
        
        # Draw matches
        for match in matches_to_draw:
            pt1 = (int(kp1[match['queryIdx']]['x']), 
                   int(kp1[match['queryIdx']]['y']))
            pt2 = (int(kp2[match['trainIdx']]['x']) + w1, 
                   int(kp2[match['trainIdx']]['y']))
            
            # Draw line
            color = (0, 255, 0)  # Green
            self._draw_line(vis, pt1, pt2, color)
            
            # Draw circles at keypoints
            self._draw_circle(vis, pt1, 3, color)
            self._draw_circle(vis, pt2, 3, color)
        
        return vis
    
    def _draw_line(self, image, pt1, pt2, color, thickness=1):
        """Draw line on image using Bresenham's algorithm."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        steep = dy > dx
        
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        dx = x2 - x1
        dy = abs(y2 - y1)
        
        error = dx / 2
        ystep = 1 if y1 < y2 else -1
        y = y1
        
        for x in range(int(x1), int(x2) + 1):
            if steep:
                if 0 <= y < image.shape[1] and 0 <= x < image.shape[0]:
                    image[x, y] = color
            else:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    image[y, x] = color
            
            error -= dy
            if error < 0:
                y += ystep
                error += dx
    
    def _draw_circle(self, image, center, radius, color):
        """Draw circle on image."""
        cx, cy = center
        
        for y in range(max(0, cy - radius), min(image.shape[0], cy + radius + 1)):
            for x in range(max(0, cx - radius), min(image.shape[1], cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    image[y, x] = color

