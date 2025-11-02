"""
Pure implementation of Homography computation and RANSAC algorithm
using only NumPy - no OpenCV dependencies.
"""

import numpy as np


class HomographyEstimator:
    """
    Homography matrix estimation using RANSAC algorithm.
    
    A homography is a 3x3 matrix that describes the projective transformation
    between two planes (images).
    """
    
    def __init__(self, ransac_reproj_threshold=4.0, max_iters=2000, 
                 confidence=0.995, min_inliers=10):
        """
        Initialize Homography Estimator.
        
        Args:
            ransac_reproj_threshold: Maximum reprojection error to be considered inlier
            max_iters: Maximum number of RANSAC iterations
            confidence: Desired confidence level for RANSAC
            min_inliers: Minimum number of inliers required
        """
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.min_inliers = min_inliers
    
    def find_homography(self, src_points, dst_points):
        """
        Find homography matrix using RANSAC.
        
        Args:
            src_points: Source points (N x 2)
            dst_points: Destination points (N x 2)
            
        Returns:
            H: Homography matrix (3 x 3)
            mask: Inlier mask (N,)
        """
        src_points = np.array(src_points, dtype=np.float64)
        dst_points = np.array(dst_points, dtype=np.float64)
        
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("Need at least 4 point correspondences")
        
        if len(src_points) != len(dst_points):
            raise ValueError("Source and destination points must have same length")
        
        # Run RANSAC
        best_H = None
        best_inliers = None
        best_num_inliers = 0
        
        n_points = len(src_points)
        
        for iteration in range(self.max_iters):
            # Randomly sample 4 points
            indices = np.random.choice(n_points, 4, replace=False)
            src_sample = src_points[indices]
            dst_sample = dst_points[indices]
            
            # Compute homography from 4 points
            H = self._compute_homography_4pts(src_sample, dst_sample)
            
            if H is None:
                continue
            
            # Compute reprojection error for all points
            inliers = self._get_inliers(src_points, dst_points, H)
            num_inliers = np.sum(inliers)
            
            # Update best model
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_H = H
                
                # Adaptive termination
                inlier_ratio = num_inliers / n_points
                if inlier_ratio > 0.01:  # Avoid log(0)
                    n_iters_needed = np.log(1 - self.confidence) / np.log(1 - inlier_ratio**4)
                    if iteration > n_iters_needed:
                        break
        
        # Refine homography using all inliers
        if best_H is not None and best_num_inliers >= self.min_inliers:
            inlier_src = src_points[best_inliers]
            inlier_dst = dst_points[best_inliers]
            best_H = self._compute_homography_dlt(inlier_src, inlier_dst)
            
            # Recompute inliers with refined H
            best_inliers = self._get_inliers(src_points, dst_points, best_H)
        
        if best_H is None:
            return None, None
        
        return best_H, best_inliers
    
    def _compute_homography_4pts(self, src_pts, dst_pts):
        """
        Compute homography from exactly 4 point correspondences.
        
        Uses Direct Linear Transform (DLT) algorithm.
        """
        return self._compute_homography_dlt(src_pts, dst_pts)
    
    def _compute_homography_dlt(self, src_pts, dst_pts):
        """
        Compute homography using Direct Linear Transform.
        
        For each point correspondence (x, y) -> (x', y'), we have:
        x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
        y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
        
        This gives us 2 equations per point correspondence.
        We need at least 4 points (8 equations) to solve for 8 unknowns.
        """
        n = len(src_pts)
        
        if n < 4:
            return None
        
        # Normalize points for better numerical stability
        src_pts_norm, T_src = self._normalize_points(src_pts)
        dst_pts_norm, T_dst = self._normalize_points(dst_pts)
        
        # Build matrix A for homogeneous linear system
        A = []
        for i in range(n):
            x, y = src_pts_norm[i]
            x_prime, y_prime = dst_pts_norm[i]
            
            # Two rows per correspondence
            A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
        
        A = np.array(A)
        
        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            
            # Denormalize
            H = np.linalg.inv(T_dst) @ H @ T_src
            
            # Normalize so that H[2, 2] = 1
            H = H / H[2, 2]
            
            return H
        except np.linalg.LinAlgError:
            return None
    
    def _normalize_points(self, points):
        """
        Normalize points for better numerical stability.
        
        Translates points so centroid is at origin and scales so
        average distance from origin is sqrt(2).
        """
        points = np.array(points, dtype=np.float64)
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Translate points
        points_centered = points - centroid
        
        # Compute average distance from origin
        avg_dist = np.mean(np.sqrt(np.sum(points_centered**2, axis=1)))
        
        if avg_dist < 1e-10:
            avg_dist = 1.0
        
        # Scale factor
        scale = np.sqrt(2) / avg_dist
        
        # Transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # Apply transformation
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        points_normalized = (T @ points_homogeneous.T).T
        points_normalized = points_normalized[:, :2] / points_normalized[:, 2:3]
        
        return points_normalized, T
    
    def _get_inliers(self, src_pts, dst_pts, H):
        """
        Get inlier mask based on reprojection error.
        
        Args:
            src_pts: Source points (N x 2)
            dst_pts: Destination points (N x 2)
            H: Homography matrix (3 x 3)
            
        Returns:
            mask: Boolean mask indicating inliers
        """
        # Transform source points
        src_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        dst_projected = (H @ src_homogeneous.T).T
        
        # Convert from homogeneous coordinates
        dst_projected = dst_projected[:, :2] / dst_projected[:, 2:3]
        
        # Compute reprojection error
        errors = np.sqrt(np.sum((dst_pts - dst_projected)**2, axis=1))
        
        # Inliers are points with error below threshold
        inliers = errors < self.ransac_reproj_threshold
        
        return inliers
    
    def apply_homography(self, points, H):
        """
        Apply homography transformation to points.
        
        Args:
            points: Points to transform (N x 2)
            H: Homography matrix (3 x 3)
            
        Returns:
            Transformed points (N x 2)
        """
        points = np.array(points, dtype=np.float64)
        
        # Convert to homogeneous coordinates
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        
        # Apply transformation
        transformed = (H @ points_homogeneous.T).T
        
        # Convert back to Cartesian coordinates
        transformed = transformed[:, :2] / transformed[:, 2:3]
        
        return transformed


def warp_perspective(image, H, output_shape):
    """
    Warp image using homography matrix.
    
    Args:
        image: Input image (H x W x C) or (H x W)
        H: Homography matrix (3 x 3)
        output_shape: Output image shape (height, width)
        
    Returns:
        Warped image
    """
    h, w = output_shape
    
    # Handle grayscale and color images
    if len(image.shape) == 2:
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        channels = image.shape[2]
    
    # Create output image
    output = np.zeros((h, w, channels), dtype=image.dtype)
    
    # Inverse homography for backward warping
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return output.squeeze() if channels == 1 else output
    
    # Create coordinate grid for output image
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones(h * w)], axis=1)
    
    # Apply inverse homography
    src_coords = (H_inv @ coords.T).T
    src_coords = src_coords[:, :2] / src_coords[:, 2:3]
    
    # Reshape
    src_x = src_coords[:, 0].reshape(h, w)
    src_y = src_coords[:, 1].reshape(h, w)
    
    # Bilinear interpolation
    output = bilinear_interpolate(image, src_x, src_y)
    
    return output.squeeze() if channels == 1 else output


def bilinear_interpolate(image, x, y):
    """
    Bilinear interpolation for image warping.
    
    Args:
        image: Input image (H x W x C)
        x: X coordinates (H x W)
        y: Y coordinates (H x W)
        
    Returns:
        Interpolated values (H x W x C)
    """
    h, w = image.shape[:2]
    
    # Get integer coordinates
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    
    # Clip to image boundaries
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    # Get fractional parts
    fx = x - x0
    fy = y - y0
    
    # Handle out of bounds
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    
    # Bilinear interpolation
    if len(image.shape) == 3:
        channels = image.shape[2]
        output = np.zeros((y.shape[0], y.shape[1], channels), dtype=image.dtype)
        
        for c in range(channels):
            I00 = image[y0, x0, c]
            I01 = image[y1, x0, c]
            I10 = image[y0, x1, c]
            I11 = image[y1, x1, c]
            
            # Interpolate
            w00 = (1 - fx) * (1 - fy)
            w01 = (1 - fx) * fy
            w10 = fx * (1 - fy)
            w11 = fx * fy
            
            output[:, :, c] = (w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11) * mask
    else:
        I00 = image[y0, x0]
        I01 = image[y1, x0]
        I10 = image[y0, x1]
        I11 = image[y1, x1]
        
        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy
        
        output = (w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11) * mask
    
    return output

