"""
Pure SIFT (Scale-Invariant Feature Transform) implementation
using only NumPy - no OpenCV dependencies.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


class SIFT:
    """
    Scale-Invariant Feature Transform (SIFT) implementation.
    
    This class implements the full SIFT pipeline:
    1. Scale-space extrema detection
    2. Keypoint localization
    3. Orientation assignment
    4. Keypoint descriptor
    """
    
    def __init__(self, num_octaves=4, num_scales=5, sigma=1.6, 
                 contrast_threshold=0.04, edge_threshold=10, 
                 border_width=5):
        """
        Initialize SIFT detector.
        
        Args:
            num_octaves: Number of octaves in the scale space
            num_scales: Number of scales per octave
            sigma: Base sigma for Gaussian blur
            contrast_threshold: Threshold for low-contrast keypoint removal
            edge_threshold: Threshold for edge response removal
            border_width: Border width to ignore keypoints
        """
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.border_width = border_width
        self.k = 2 ** (1.0 / num_scales)  # Scale multiplication factor
        
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Grayscale image (2D numpy array)
            
        Returns:
            keypoints: List of keypoint objects
            descriptors: Array of descriptors (N x 128)
        """
        # Ensure image is grayscale and float
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        image = image.astype(np.float32) / 255.0
        
        # Build Gaussian pyramid
        gaussian_pyramid = self._build_gaussian_pyramid(image)
        
        # Build Difference of Gaussians (DoG) pyramid
        dog_pyramid = self._build_dog_pyramid(gaussian_pyramid)
        
        # Find keypoints in scale space
        keypoints = self._find_scale_space_extrema(gaussian_pyramid, dog_pyramid)
        
        # Assign orientations to keypoints
        keypoints = self._assign_orientations(gaussian_pyramid, keypoints)
        
        # Compute descriptors
        keypoints, descriptors = self._generate_descriptors(gaussian_pyramid, keypoints)
        
        return keypoints, descriptors
    
    def _build_gaussian_pyramid(self, image):
        """Build Gaussian pyramid."""
        pyramid = []
        
        # Initial sigma for the first octave
        sigma_0 = self.sigma
        
        for octave in range(self.num_octaves):
            octave_pyramid = []
            
            # Resize image for this octave
            if octave == 0:
                base_image = image.copy()
            else:
                # Downsample by factor of 2
                h, w = pyramid[octave - 1][-3].shape
                base_image = self._downsample(pyramid[octave - 1][-3])
            
            # Generate scales within octave
            for scale in range(self.num_scales + 3):
                sigma = sigma_0 * (self.k ** scale)
                blurred = gaussian_filter(base_image, sigma)
                octave_pyramid.append(blurred)
            
            pyramid.append(octave_pyramid)
        
        return pyramid
    
    def _build_dog_pyramid(self, gaussian_pyramid):
        """Build Difference of Gaussians pyramid."""
        dog_pyramid = []
        
        for octave_pyramid in gaussian_pyramid:
            octave_dog = []
            for i in range(len(octave_pyramid) - 1):
                dog = octave_pyramid[i + 1] - octave_pyramid[i]
                octave_dog.append(dog)
            dog_pyramid.append(octave_dog)
        
        return dog_pyramid
    
    def _find_scale_space_extrema(self, gaussian_pyramid, dog_pyramid):
        """Find local extrema in DoG scale space."""
        keypoints = []
        
        for octave_idx, octave_dog in enumerate(dog_pyramid):
            for scale_idx in range(1, len(octave_dog) - 1):
                # Get three consecutive scales
                prev_dog = octave_dog[scale_idx - 1]
                curr_dog = octave_dog[scale_idx]
                next_dog = octave_dog[scale_idx + 1]
                
                # Find local extrema
                extrema = self._find_local_extrema(prev_dog, curr_dog, next_dog)
                
                # Refine keypoints and filter
                for y, x in extrema:
                    if self._is_on_border(curr_dog, y, x):
                        continue
                    
                    # Sub-pixel refinement
                    refined = self._refine_keypoint(
                        octave_dog, scale_idx, y, x
                    )
                    
                    if refined is not None:
                        y_refined, x_refined, scale_refined = refined
                        
                        # Calculate sigma for this keypoint
                        sigma = self.sigma * (self.k ** (scale_idx + scale_refined))
                        sigma *= 2 ** octave_idx  # Account for octave
                        
                        # Create keypoint
                        keypoint = {
                            'octave': octave_idx,
                            'scale': scale_idx + scale_refined,
                            'y': (y_refined + y) * (2 ** octave_idx),
                            'x': (x_refined + x) * (2 ** octave_idx),
                            'sigma': sigma,
                            'response': abs(curr_dog[y, x])
                        }
                        keypoints.append(keypoint)
        
        return keypoints
    
    def _find_local_extrema(self, prev_dog, curr_dog, next_dog):
        """Find local maxima and minima in DoG scale space."""
        extrema = []
        
        # Find maxima
        max_filter = maximum_filter(curr_dog, size=3)
        is_max = (curr_dog == max_filter)
        
        # Check with adjacent scales
        is_max &= (curr_dog > prev_dog)
        is_max &= (curr_dog > next_dog)
        
        # Threshold
        is_max &= (abs(curr_dog) > self.contrast_threshold)
        
        # Find minima
        min_filter = minimum_filter(curr_dog, size=3)
        is_min = (curr_dog == min_filter)
        
        # Check with adjacent scales
        is_min &= (curr_dog < prev_dog)
        is_min &= (curr_dog < next_dog)
        
        # Threshold
        is_min &= (abs(curr_dog) > self.contrast_threshold)
        
        # Combine maxima and minima
        is_extrema = is_max | is_min
        
        extrema_coords = np.argwhere(is_extrema)
        return extrema_coords
    
    def _is_on_border(self, image, y, x):
        """Check if point is on image border."""
        h, w = image.shape
        border = self.border_width
        return (x < border or x >= w - border or 
                y < border or y >= h - border)
    
    def _refine_keypoint(self, octave_dog, scale_idx, y, x):
        """
        Refine keypoint location using quadratic interpolation.
        Also removes edge responses.
        """
        prev_dog = octave_dog[scale_idx - 1]
        curr_dog = octave_dog[scale_idx]
        next_dog = octave_dog[scale_idx + 1]
        
        # Compute gradient
        dx = (curr_dog[y, x + 1] - curr_dog[y, x - 1]) / 2.0
        dy = (curr_dog[y + 1, x] - curr_dog[y - 1, x]) / 2.0
        ds = (next_dog[y, x] - prev_dog[y, x]) / 2.0
        
        # Compute Hessian
        dxx = curr_dog[y, x + 1] + curr_dog[y, x - 1] - 2 * curr_dog[y, x]
        dyy = curr_dog[y + 1, x] + curr_dog[y - 1, x] - 2 * curr_dog[y, x]
        dss = next_dog[y, x] + prev_dog[y, x] - 2 * curr_dog[y, x]
        
        dxy = ((curr_dog[y + 1, x + 1] - curr_dog[y + 1, x - 1]) -
               (curr_dog[y - 1, x + 1] - curr_dog[y - 1, x - 1])) / 4.0
        dxs = ((next_dog[y, x + 1] - next_dog[y, x - 1]) -
               (prev_dog[y, x + 1] - prev_dog[y, x - 1])) / 4.0
        dys = ((next_dog[y + 1, x] - next_dog[y - 1, x]) -
               (prev_dog[y + 1, x] - prev_dog[y - 1, x])) / 4.0
        
        # Hessian matrix
        H = np.array([[dxx, dxy, dxs],
                      [dxy, dyy, dys],
                      [dxs, dys, dss]])
        
        gradient = np.array([dx, dy, ds])
        
        # Solve for offset
        try:
            offset = -np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            return None
        
        # Check if offset is too large
        if abs(offset[0]) > 1.5 or abs(offset[1]) > 1.5 or abs(offset[2]) > 1.5:
            return None
        
        # Check contrast
        value = curr_dog[y, x] + 0.5 * np.dot(gradient, offset)
        if abs(value) < self.contrast_threshold:
            return None
        
        # Eliminate edge responses
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        
        if det <= 0:
            return None
        
        ratio = trace * trace / det
        threshold_ratio = ((self.edge_threshold + 1) ** 2) / self.edge_threshold
        
        if ratio > threshold_ratio:
            return None
        
        return offset[1], offset[0], offset[2]
    
    def _assign_orientations(self, gaussian_pyramid, keypoints):
        """Assign orientations to keypoints."""
        keypoints_with_orientation = []
        
        for kp in keypoints:
            octave_idx = kp['octave']
            scale_idx = int(round(kp['scale']))
            
            # Get image at this scale
            if scale_idx < 0 or scale_idx >= len(gaussian_pyramid[octave_idx]):
                continue
            
            image = gaussian_pyramid[octave_idx][scale_idx]
            
            # Convert to octave coordinates
            y = int(round(kp['y'] / (2 ** octave_idx)))
            x = int(round(kp['x'] / (2 ** octave_idx)))
            
            # Check bounds
            if (x < self.border_width or x >= image.shape[1] - self.border_width or
                y < self.border_width or y >= image.shape[0] - self.border_width):
                continue
            
            # Compute gradient orientations in region
            orientations = self._compute_keypoint_orientations(image, y, x, kp['sigma'])
            
            # Create a keypoint for each dominant orientation
            for orientation in orientations:
                kp_oriented = kp.copy()
                kp_oriented['orientation'] = orientation
                keypoints_with_orientation.append(kp_oriented)
        
        return keypoints_with_orientation
    
    def _compute_keypoint_orientations(self, image, y, x, sigma):
        """Compute dominant orientations for a keypoint."""
        # Compute gradients
        dy = image[y + 1:y + 2, x - 1:x + 2] - image[y - 1:y, x - 1:x + 2]
        dx = image[y - 1:y + 2, x + 1:x + 2] - image[y - 1:y + 2, x - 1:x]
        
        # Get gradient magnitude and orientation in larger window
        window_size = int(round(3 * sigma * 1.5))
        window_size = max(1, min(window_size, 8))
        
        y_start = max(0, y - window_size)
        y_end = min(image.shape[0] - 1, y + window_size + 1)
        x_start = max(0, x - window_size)
        x_end = min(image.shape[1] - 1, x + window_size + 1)
        
        # Compute gradients in window
        region = image[y_start:y_end, x_start:x_end]
        
        if region.shape[0] < 3 or region.shape[1] < 3:
            return [0.0]
        
        gy = np.zeros_like(region)
        gx = np.zeros_like(region)
        
        gy[1:-1, :] = region[2:, :] - region[:-2, :]
        gx[:, 1:-1] = region[:, 2:] - region[:, :-2]
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        
        # Create orientation histogram (36 bins)
        num_bins = 36
        hist = np.zeros(num_bins)
        
        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                # Gaussian weighting
                dy_weight = i - (y - y_start)
                dx_weight = j - (x - x_start)
                weight = magnitude[i, j] * np.exp(-(dx_weight**2 + dy_weight**2) / (2 * sigma**2))
                
                # Add to histogram
                angle = orientation[i, j]
                angle_deg = np.degrees(angle) % 360
                bin_idx = int(angle_deg * num_bins / 360) % num_bins
                hist[bin_idx] += weight
        
        # Smooth histogram
        hist = np.convolve(hist, [1/3, 1/3, 1/3], mode='same')
        
        # Find peaks (orientations > 80% of max)
        max_val = np.max(hist)
        threshold = 0.8 * max_val
        
        orientations = []
        for i in range(num_bins):
            if hist[i] > threshold:
                # Parabolic interpolation for sub-bin accuracy
                prev_val = hist[(i - 1) % num_bins]
                next_val = hist[(i + 1) % num_bins]
                
                if hist[i] >= prev_val and hist[i] >= next_val:
                    interp = 0.5 * (prev_val - next_val) / (prev_val - 2 * hist[i] + next_val)
                    angle = ((i + interp) * 360.0 / num_bins) % 360
                    orientations.append(np.radians(angle))
        
        if len(orientations) == 0:
            orientations = [0.0]
        
        return orientations
    
    def _generate_descriptors(self, gaussian_pyramid, keypoints):
        """Generate SIFT descriptors for keypoints."""
        descriptors = []
        valid_keypoints = []
        
        for kp in keypoints:
            octave_idx = kp['octave']
            scale_idx = int(round(kp['scale']))
            
            if scale_idx < 0 or scale_idx >= len(gaussian_pyramid[octave_idx]):
                continue
            
            image = gaussian_pyramid[octave_idx][scale_idx]
            
            # Convert to octave coordinates
            y = int(round(kp['y'] / (2 ** octave_idx)))
            x = int(round(kp['x'] / (2 ** octave_idx)))
            
            # Generate descriptor
            descriptor = self._compute_descriptor(image, y, x, 
                                                  kp['orientation'], 
                                                  kp['sigma'])
            
            if descriptor is not None:
                descriptors.append(descriptor)
                valid_keypoints.append(kp)
        
        if len(descriptors) > 0:
            descriptors = np.array(descriptors)
        else:
            descriptors = np.array([]).reshape(0, 128)
        
        return valid_keypoints, descriptors
    
    def _compute_descriptor(self, image, y, x, orientation, sigma):
        """
        Compute 128-dimensional SIFT descriptor.
        Uses 4x4 grid of histograms with 8 orientation bins each.
        """
        # Descriptor parameters
        d = 4  # 4x4 grid
        n = 8  # 8 orientation bins
        
        # Window size
        window_size = int(round(3 * sigma * d))
        
        # Check bounds
        if (x - window_size < 0 or x + window_size >= image.shape[1] or
            y - window_size < 0 or y + window_size >= image.shape[0]):
            return None
        
        # Compute gradients
        region = image[y - window_size:y + window_size + 1, 
                      x - window_size:x + window_size + 1]
        
        if region.shape[0] < 3 or region.shape[1] < 3:
            return None
        
        gy = np.zeros_like(region)
        gx = np.zeros_like(region)
        
        gy[1:-1, :] = region[2:, :] - region[:-2, :]
        gx[:, 1:-1] = region[:, 2:] - region[:, :-2]
        
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) - orientation
        
        # Normalize angles to [0, 2π)
        angle = angle % (2 * np.pi)
        
        # Initialize descriptor
        descriptor = np.zeros((d, d, n))
        
        # Divide into 4x4 subregions
        patch_size = 2 * window_size / d
        
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                # Relative position
                y_rel = i - window_size
                x_rel = j - window_size
                
                # Rotate coordinates according to keypoint orientation
                cos_o = np.cos(-orientation)
                sin_o = np.sin(-orientation)
                x_rot = cos_o * x_rel - sin_o * y_rel
                y_rot = sin_o * x_rel + cos_o * y_rel
                
                # Which bin?
                y_bin = (y_rot / patch_size) + d / 2.0
                x_bin = (x_rot / patch_size) + d / 2.0
                
                if y_bin < 0 or y_bin >= d or x_bin < 0 or x_bin >= d:
                    continue
                
                # Orientation bin
                angle_bin = (angle[i, j] / (2 * np.pi)) * n
                
                # Trilinear interpolation
                weight = magnitude[i, j]
                
                # Spatial bins
                y_bin_floor = int(np.floor(y_bin))
                x_bin_floor = int(np.floor(x_bin))
                angle_bin_floor = int(np.floor(angle_bin)) % n
                
                # Interpolation weights
                dy = y_bin - y_bin_floor
                dx = x_bin - x_bin_floor
                dangle = angle_bin - angle_bin_floor
                
                # Add contributions
                for dy_i in range(2):
                    y_idx = y_bin_floor + dy_i
                    if y_idx < 0 or y_idx >= d:
                        continue
                    wy = (1 - dy) if dy_i == 0 else dy
                    
                    for dx_i in range(2):
                        x_idx = x_bin_floor + dx_i
                        if x_idx < 0 or x_idx >= d:
                            continue
                        wx = (1 - dx) if dx_i == 0 else dx
                        
                        for dangle_i in range(2):
                            angle_idx = (angle_bin_floor + dangle_i) % n
                            wangle = (1 - dangle) if dangle_i == 0 else dangle
                            
                            descriptor[y_idx, x_idx, angle_idx] += weight * wy * wx * wangle
        
        # Flatten to 128-dimensional vector
        descriptor = descriptor.flatten()
        
        # Normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Clip values to 0.2 and renormalize (illumination invariance)
        descriptor = np.clip(descriptor, 0, 0.2)
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Convert to uint8 equivalent (0-255 range)
        descriptor = np.clip(descriptor * 512, 0, 255).astype(np.uint8)
        
        return descriptor
    
    def _downsample(self, image):
        """Downsample image by factor of 2."""
        return image[::2, ::2]

