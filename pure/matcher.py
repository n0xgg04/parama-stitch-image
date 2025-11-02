"""
Feature matching using L2 distance (Euclidean distance).
Pure implementation without OpenCV.
"""

import numpy as np


class FeatureMatcher:
    """
    Feature matcher using L2 (Euclidean) distance.
    Implements brute-force matching with cross-check and ratio test.
    """
    
    def __init__(self, cross_check=True, ratio_threshold=0.75):
        """
        Initialize feature matcher.
        
        Args:
            cross_check: Whether to perform cross-check for matches
            ratio_threshold: Lowe's ratio test threshold (0.75 recommended)
        """
        self.cross_check = cross_check
        self.ratio_threshold = ratio_threshold
    
    def match(self, descriptors1, descriptors2):
        """
        Match features between two sets of descriptors.
        
        Args:
            descriptors1: Descriptors from first image (N x 128)
            descriptors2: Descriptors from second image (M x 128)
            
        Returns:
            matches: List of DMatch objects with queryIdx, trainIdx, and distance
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # Convert to float for computation
        desc1 = descriptors1.astype(np.float32)
        desc2 = descriptors2.astype(np.float32)
        
        # Compute distance matrix
        distances = self._compute_distance_matrix(desc1, desc2)
        
        # Find best matches from desc1 to desc2
        matches_1to2 = self._find_best_matches(distances)
        
        if self.cross_check:
            # Find best matches from desc2 to desc1
            matches_2to1 = self._find_best_matches(distances.T)
            
            # Cross-check: keep only mutual best matches
            matches = self._cross_check_matches(matches_1to2, matches_2to1)
        else:
            matches = matches_1to2
        
        # Sort by distance (ascending)
        matches.sort(key=lambda x: x['distance'])
        
        return matches
    
    def _compute_distance_matrix(self, desc1, desc2):
        """
        Compute L2 distance matrix between two sets of descriptors.
        
        Args:
            desc1: N x D array
            desc2: M x D array
            
        Returns:
            distances: N x M matrix where distances[i, j] is L2 distance 
                      between desc1[i] and desc2[j]
        """
        # Efficient computation using broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        
        sq_norms1 = np.sum(desc1**2, axis=1, keepdims=True)  # N x 1
        sq_norms2 = np.sum(desc2**2, axis=1, keepdims=True)  # M x 1
        
        # Compute dot products
        dot_products = np.dot(desc1, desc2.T)  # N x M
        
        # Compute squared distances
        sq_distances = sq_norms1 + sq_norms2.T - 2 * dot_products
        
        # Ensure non-negative (numerical stability)
        sq_distances = np.maximum(sq_distances, 0)
        
        # Take square root to get L2 distances
        distances = np.sqrt(sq_distances)
        
        return distances
    
    def _find_best_matches(self, distances):
        """
        Find best matches using Lowe's ratio test.
        
        Args:
            distances: N x M distance matrix
            
        Returns:
            matches: List of match dictionaries
        """
        matches = []
        
        for i in range(distances.shape[0]):
            # Get distances for this query descriptor
            dists = distances[i]
            
            # Find two nearest neighbors
            if len(dists) < 2:
                continue
            
            # Get indices of two smallest distances
            sorted_indices = np.argsort(dists)
            nearest_idx = sorted_indices[0]
            second_nearest_idx = sorted_indices[1]
            
            nearest_dist = dists[nearest_idx]
            second_nearest_dist = dists[second_nearest_idx]
            
            # Lowe's ratio test
            if second_nearest_dist > 0 and nearest_dist / second_nearest_dist < self.ratio_threshold:
                match = {
                    'queryIdx': i,
                    'trainIdx': nearest_idx,
                    'distance': nearest_dist
                }
                matches.append(match)
        
        return matches
    
    def _cross_check_matches(self, matches_1to2, matches_2to1):
        """
        Perform cross-check: keep only mutual best matches.
        
        Args:
            matches_1to2: Matches from image 1 to image 2
            matches_2to1: Matches from image 2 to image 1
            
        Returns:
            cross_checked_matches: Mutually consistent matches
        """
        # Create dictionary for fast lookup
        matches_2to1_dict = {m['queryIdx']: m['trainIdx'] for m in matches_2to1}
        
        cross_checked = []
        for match in matches_1to2:
            query_idx = match['queryIdx']
            train_idx = match['trainIdx']
            
            # Check if reverse match exists and is consistent
            if train_idx in matches_2to1_dict:
                if matches_2to1_dict[train_idx] == query_idx:
                    cross_checked.append(match)
        
        return cross_checked


class Keypoint:
    """Simple keypoint class to store keypoint information."""
    
    def __init__(self, x, y, size=1.0, angle=0.0, response=0.0, octave=0):
        self.pt = (x, y)  # (x, y) coordinates
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
    
    def __repr__(self):
        return f"Keypoint(pt={self.pt}, size={self.size:.2f}, angle={self.angle:.2f})"


def convert_to_keypoints(keypoint_dicts):
    """
    Convert list of keypoint dictionaries to Keypoint objects.
    
    Args:
        keypoint_dicts: List of dictionaries with keypoint information
        
    Returns:
        List of Keypoint objects
    """
    keypoints = []
    for kp in keypoint_dicts:
        keypoint = Keypoint(
            x=kp['x'],
            y=kp['y'],
            size=kp.get('sigma', 1.0),
            angle=kp.get('orientation', 0.0),
            response=kp.get('response', 0.0),
            octave=kp.get('octave', 0)
        )
        keypoints.append(keypoint)
    return keypoints

