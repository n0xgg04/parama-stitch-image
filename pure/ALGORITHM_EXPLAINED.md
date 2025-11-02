# Giải Thích Chi Tiết Thuật Toán Panorama Stitching

Tài liệu này giải thích chi tiết về cơ sở lý thuyết và triển khai từng bước của thuật toán ghép ảnh panorama.

---

## 📚 Mục Lục

1. [Tổng Quan Pipeline](#1-tổng-quan-pipeline)
2. [SIFT - Scale-Invariant Feature Transform](#2-sift---scale-invariant-feature-transform)
3. [Feature Matching](#3-feature-matching)
4. [Homography Estimation & RANSAC](#4-homography-estimation--ransac)
5. [Image Warping](#5-image-warping)
6. [Weighted Blending](#6-weighted-blending)
7. [Tham Khảo](#7-tham-khảo)

---

## 1. Tổng Quan Pipeline

### 1.1. Quy Trình Tổng Thể

```
Input Images
     ↓
[SIFT Feature Detection]
     ↓
Keypoints + Descriptors
     ↓
[Feature Matching]
     ↓
Match Pairs
     ↓
[RANSAC + Homography]
     ↓
Homography Matrix H
     ↓
[Image Warping]
     ↓
Warped Images
     ↓
[Weighted Blending]
     ↓
Panorama Output
```

### 1.2. File Cấu Trúc

- **`sift.py`**: Triển khai SIFT detector và descriptor
- **`matcher.py`**: Matching descriptors giữa hai ảnh
- **`homography.py`**: Ước lượng homography và RANSAC
- **`blending.py`**: Blend ảnh với weighted masks
- **`panorama_stitcher.py`**: Kết hợp tất cả thành pipeline hoàn chỉnh

---

## 2. SIFT - Scale-Invariant Feature Transform

### 2.1. Cơ Sở Lý Thuyết

SIFT (David Lowe, 1999) là thuật toán phát hiện và mô tả đặc trưng bất biến với:

- **Scale invariance**: Bất biến với tỷ lệ (zoom in/out)
- **Rotation invariance**: Bất biến với xoay
- **Illumination invariance**: Bất biến với thay đổi ánh sáng

### 2.2. Bước 1: Xây Dựng Scale Space

#### 2.2.1. Gaussian Pyramid

Ảnh được làm mờ với Gaussian filters ở nhiều scales khác nhau:

```
L(x, y, σ) = G(x, y, σ) * I(x, y)
```

Trong đó:

- `L(x, y, σ)`: Scale space representation
- `G(x, y, σ)`: Gaussian kernel với standard deviation σ
- `I(x, y)`: Ảnh gốc
- `*`: Phép convolution

**Gaussian Kernel:**

```
G(x, y, σ) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)
```

**Triển Khai:**

```python
def _build_gaussian_pyramid(self, image):
    pyramid = []

    for octave in range(self.num_octaves):
        octave_pyramid = []

        # Downsample cho mỗi octave
        if octave == 0:
            base_image = image.copy()
        else:
            base_image = self._downsample(pyramid[octave - 1][-3])

        # Tạo scales trong octave
        for scale in range(self.num_scales + 3):
            sigma = sigma_0 * (k ** scale)  # k = 2^(1/num_scales)
            blurred = gaussian_filter(base_image, sigma)
            octave_pyramid.append(blurred)

        pyramid.append(octave_pyramid)

    return pyramid
```

**Tại sao cần nhiều octaves?**

- Mỗi octave xử lý ảnh ở resolution khác nhau
- Octave 0: Ảnh gốc
- Octave 1: Ảnh downsampled 2x
- Octave 2: Ảnh downsampled 4x
- ...

### 2.3. Bước 2: Difference of Gaussian (DoG)

DoG là xấp xỉ của Laplacian of Gaussian, dùng để phát hiện blob:

```
D(x, y, σ) = L(x, y, kσ) - L(x, y, σ)
```

**Tại sao DoG?**

- Laplacian of Gaussian (LoG) tốn kém tính toán
- DoG xấp xỉ LoG nhưng nhanh hơn nhiều:

```
G(x, y, kσ) - G(x, y, σ) ≈ (k - 1)σ² ∇²G
```

**Triển Khai:**

```python
def _build_dog_pyramid(self, gaussian_pyramid):
    dog_pyramid = []

    for octave_pyramid in gaussian_pyramid:
        octave_dog = []
        for i in range(len(octave_pyramid) - 1):
            # Difference of Gaussians
            dog = octave_pyramid[i + 1] - octave_pyramid[i]
            octave_dog.append(dog)
        dog_pyramid.append(octave_dog)

    return dog_pyramid
```

### 2.4. Bước 3: Phát Hiện Extrema

Tìm local maxima và minima trong DoG scale space (3D: x, y, scale).

Một điểm là extrema nếu:

- Lớn hơn/nhỏ hơn 26 điểm láng giềng (8 trong cùng scale + 9 ở scale trên + 9 ở scale dưới)

```python
def _find_local_extrema(self, prev_dog, curr_dog, next_dog):
    # Tìm maxima
    max_filter = maximum_filter(curr_dog, size=3)
    is_max = (curr_dog == max_filter)
    is_max &= (curr_dog > prev_dog)  # So với scale dưới
    is_max &= (curr_dog > next_dog)  # So với scale trên
    is_max &= (abs(curr_dog) > self.contrast_threshold)

    # Tìm minima
    min_filter = minimum_filter(curr_dog, size=3)
    is_min = (curr_dog == min_filter)
    is_min &= (curr_dog < prev_dog)
    is_min &= (curr_dog < next_dog)
    is_min &= (abs(curr_dog) > self.contrast_threshold)

    # Kết hợp
    is_extrema = is_max | is_min
    return np.argwhere(is_extrema)
```

### 2.5. Bước 4: Keypoint Refinement

#### 2.5.1. Sub-pixel Localization

Sử dụng Taylor expansion để tìm vị trí chính xác hơn:

```
D(x) ≈ D + (∂D/∂x)ᵀ·x + (1/2)·xᵀ·(∂²D/∂x²)·x
```

Đạo hàm và tìm cực trị:

```
x̂ = -(∂²D/∂x²)⁻¹ · (∂D/∂x)
```

Trong đó `x = [x, y, σ]ᵀ`

**Triển Khai:**

```python
def _refine_keypoint(self, octave_dog, scale_idx, y, x):
    # Tính gradient (đạo hàm bậc 1)
    dx = (curr_dog[y, x + 1] - curr_dog[y, x - 1]) / 2.0
    dy = (curr_dog[y + 1, x] - curr_dog[y - 1, x]) / 2.0
    ds = (next_dog[y, x] - prev_dog[y, x]) / 2.0

    # Tính Hessian (đạo hàm bậc 2)
    dxx = curr_dog[y, x + 1] + curr_dog[y, x - 1] - 2 * curr_dog[y, x]
    dyy = curr_dog[y + 1, x] + curr_dog[y - 1, x] - 2 * curr_dog[y, x]
    dss = next_dog[y, x] + prev_dog[y, x] - 2 * curr_dog[y, x]

    dxy = ((curr_dog[y + 1, x + 1] - curr_dog[y + 1, x - 1]) -
           (curr_dog[y - 1, x + 1] - curr_dog[y - 1, x - 1])) / 4.0
    # ... dxs, dys tương tự

    # Hessian matrix
    H = [[dxx, dxy, dxs],
         [dxy, dyy, dys],
         [dxs, dys, dss]]

    gradient = [dx, dy, ds]

    # Giải hệ: H·offset = -gradient
    offset = -np.linalg.solve(H, gradient)

    return offset[1], offset[0], offset[2]  # dy, dx, ds
```

#### 2.5.2. Edge Response Elimination

Loại bỏ keypoints nằm trên edges (không stable):

Sử dụng Harris corner detector principle. Tính tỷ số eigenvalues của Hessian:

```
Tr(H)² / Det(H) < threshold
```

Trong đó:

- `Tr(H) = dxx + dyy` (trace)
- `Det(H) = dxx·dyy - dxy²` (determinant)

```python
# Edge elimination
trace = dxx + dyy
det = dxx * dyy - dxy * dxy

if det <= 0:
    return None

ratio = trace * trace / det
threshold_ratio = ((edge_threshold + 1) ** 2) / edge_threshold

if ratio > threshold_ratio:
    return None  # Edge, loại bỏ
```

**Tại sao?**

- Trên edge: 1 eigenvalue lớn, 1 eigenvalue nhỏ → ratio lớn
- Ở corner: 2 eigenvalues tương đương → ratio nhỏ

### 2.6. Bước 5: Orientation Assignment

Gán hướng dominant cho keypoint để đạt rotation invariance.

#### 2.6.1. Tính Gradient Orientation

Trong vùng lân cận keypoint (window size = 3σ):

```
magnitude(x, y) = √((L(x+1,y) - L(x-1,y))² + (L(x,y+1) - L(x,y-1))²)
orientation(x, y) = atan2(L(x,y+1) - L(x,y-1), L(x+1,y) - L(x-1,y))
```

#### 2.6.2. Orientation Histogram

Tạo histogram 36 bins (mỗi bin = 10°):

```python
def _compute_keypoint_orientations(self, image, y, x, sigma):
    # Compute gradients trong window
    window_size = int(round(3 * sigma * 1.5))

    # Gradient magnitude và orientation
    gy = image[2:, :] - image[:-2, :]
    gx = image[:, 2:] - image[:, :-2]

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    # Tạo histogram 36 bins
    num_bins = 36
    hist = np.zeros(num_bins)

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            # Gaussian weighting
            weight = magnitude[i, j] * exp(-(dx²+dy²) / (2σ²))

            # Thêm vào bin tương ứng
            angle_deg = np.degrees(orientation[i, j]) % 360
            bin_idx = int(angle_deg * num_bins / 360) % num_bins
            hist[bin_idx] += weight

    # Smooth histogram
    hist = np.convolve(hist, [1/3, 1/3, 1/3], mode='same')

    # Tìm peaks (> 80% max)
    orientations = find_peaks(hist, threshold=0.8 * max(hist))

    return orientations
```

**Tại sao có thể có nhiều orientations?**

- Một keypoint có thể có nhiều dominant directions
- Mỗi orientation tạo một keypoint riêng biệt
- Tăng số lượng keypoints, tăng matching robustness

### 2.7. Bước 6: Descriptor Generation

Tạo descriptor 128-chiều mô tả vùng xung quanh keypoint.

#### 2.7.1. Cấu Trúc Descriptor

- Chia vùng 16×16 pixels thành lưới 4×4 = 16 cells
- Mỗi cell có histogram 8 bins (8 orientations)
- Tổng: 4 × 4 × 8 = 128 dimensions

```
┌─────────────────┐
│ 4x4 │ 4x4 │ ... │  ← 4×4 grid
│ cell│ cell│     │
├─────────────────┤
│ 8-bin histogram │  ← Mỗi cell
│ per cell        │
└─────────────────┘
     ↓
[128-dim vector]
```

#### 2.7.2. Triển Khai

```python
def _compute_descriptor(self, image, y, x, orientation, sigma):
    d = 4  # 4x4 grid
    n = 8  # 8 orientation bins

    window_size = int(round(3 * sigma * d))

    # Compute gradients
    region = image[y-window_size:y+window_size+1,
                  x-window_size:x+window_size+1]

    gy = region[2:, :] - region[:-2, :]
    gx = region[:, 2:] - region[:, :-2]

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) - orientation  # Rotate theo keypoint
    angle = angle % (2 * np.pi)

    # Initialize descriptor
    descriptor = np.zeros((d, d, n))

    patch_size = 2 * window_size / d

    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            # Rotate coordinates
            x_rot = cos(orientation) * x_rel - sin(orientation) * y_rel
            y_rot = sin(orientation) * x_rel + cos(orientation) * y_rel

            # Which bin?
            y_bin = (y_rot / patch_size) + d / 2.0
            x_bin = (x_rot / patch_size) + d / 2.0
            angle_bin = (angle[i, j] / (2π)) * n

            # Trilinear interpolation
            # Distribute gradient magnitude to neighboring bins
            # ... (weighted by distance to bin centers)

    # Flatten to 128-D
    descriptor = descriptor.flatten()

    # Normalize
    descriptor = descriptor / np.linalg.norm(descriptor)

    # Clip to 0.2 and renormalize (illumination invariance)
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor = descriptor / np.linalg.norm(descriptor)

    return descriptor
```

#### 2.7.3. Illumination Invariance

**Normalization**: Loại bỏ ảnh hưởng của brightness changes

```
descriptor = descriptor / ||descriptor||
```

**Clipping & Re-normalization**: Giảm ảnh hưởng của saturation

```
descriptor = clip(descriptor, 0, 0.2)
descriptor = descriptor / ||descriptor||
```

---

## 3. Feature Matching

### 3.1. Cơ Sở Lý Thuyết

Matching tìm correspondences giữa descriptors của hai ảnh.

**Mục tiêu**: Tìm descriptor trong ảnh 2 giống nhất với mỗi descriptor trong ảnh 1.

### 3.2. L2 Distance (Euclidean Distance)

Độ tương đồng giữa hai descriptors:

```
d(v₁, v₂) = ||v₁ - v₂|| = √(Σᵢ(v₁ᵢ - v₂ᵢ)²)
```

**Tối ưu hóa tính toán:**

```
||a - b||² = ||a||² + ||b||² - 2·a·b
```

```python
def _compute_distance_matrix(self, desc1, desc2):
    # desc1: N x D
    # desc2: M x D

    # Efficient computation: ||a-b||² = ||a||² + ||b||² - 2(a·b)
    sq_norms1 = np.sum(desc1**2, axis=1, keepdims=True)  # N x 1
    sq_norms2 = np.sum(desc2**2, axis=1, keepdims=True)  # M x 1
    dot_products = np.dot(desc1, desc2.T)  # N x M

    sq_distances = sq_norms1 + sq_norms2.T - 2 * dot_products
    sq_distances = np.maximum(sq_distances, 0)  # Numerical stability

    distances = np.sqrt(sq_distances)

    return distances  # N x M matrix
```

### 3.3. Lowe's Ratio Test

**Vấn đề**: Làm sao biết một match là good match?

**Giải pháp**: So sánh với second-best match:

```
ratio = distance_to_nearest / distance_to_second_nearest
```

**Nếu ratio < 0.75**: Match tốt (nearest rõ ràng tốt hơn second-nearest)
**Nếu ratio ≥ 0.75**: Ambiguous, loại bỏ

```python
def _find_best_matches(self, distances):
    matches = []

    for i in range(distances.shape[0]):
        dists = distances[i]

        # Find two nearest neighbors
        sorted_indices = np.argsort(dists)
        nearest_idx = sorted_indices[0]
        second_nearest_idx = sorted_indices[1]

        nearest_dist = dists[nearest_idx]
        second_nearest_dist = dists[second_nearest_idx]

        # Lowe's ratio test
        if nearest_dist / second_nearest_dist < self.ratio_threshold:
            match = {
                'queryIdx': i,
                'trainIdx': nearest_idx,
                'distance': nearest_dist
            }
            matches.append(match)

    return matches
```

**Tại sao ratio test hiệu quả?**

- Loại bỏ matches ambiguous (nhiều descriptors giống nhau)
- Chỉ giữ lại matches distinctive
- Giảm false positives

### 3.4. Cross-Check

**Thêm một lớp validation**: Match phải bidirectional

```
Match(i→j) là valid nếu:
- descriptor[i] trong ảnh 1 match với descriptor[j] trong ảnh 2
- descriptor[j] trong ảnh 2 cũng match với descriptor[i] trong ảnh 1
```

```python
def _cross_check_matches(self, matches_1to2, matches_2to1):
    # Tạo mapping từ train → query
    train_to_query = {m['trainIdx']: m['queryIdx'] for m in matches_2to1}

    cross_checked = []
    for match in matches_1to2:
        query_idx = match['queryIdx']
        train_idx = match['trainIdx']

        # Check if reverse match exists and consistent
        if train_idx in train_to_query:
            if train_to_query[train_idx] == query_idx:
                cross_checked.append(match)

    return cross_checked
```

---

## 4. Homography Estimation & RANSAC

### 4.1. Homography Matrix

Homography mô tả phép biến đổi projective giữa hai planes.

#### 4.1.1. Định Nghĩa

```
    [x']       [h₁₁  h₁₂  h₁₃]   [x]
    [y']   =   [h₂₁  h₂₂  h₂₃] · [y]
    [w']       [h₃₁  h₃₂  h₃₃]   [1]
```

Sau đó normalize:

```
x'_actual = x' / w'
y'_actual = y' / w'
```

#### 4.1.2. Degrees of Freedom

Homography có 8 DOF (9 parameters - 1 scale factor):

- **4 point correspondences** cần thiết để giải (4 points × 2 equations = 8 equations)

### 4.2. Direct Linear Transform (DLT)

#### 4.2.1. Cơ Sở Toán Học

Từ phương trình homography:

```
x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)
```

Cross-multiply để loại bỏ denominator:

```
x'(h₃₁x + h₃₂y + h₃₃) = h₁₁x + h₁₂y + h₁₃
y'(h₃₁x + h₃₂y + h₃₃) = h₂₁x + h₂₂y + h₂₃
```

Rearrange thành linear equations:

```
-x  -y  -1   0   0   0   xx'  yx'  x'     [h₁₁]
 0   0   0  -x  -y  -1   xy'  yy'  y'  ·  [h₁₂]  = 0
                                           [...]
                                           [h₃₃]
```

#### 4.2.2. Triển Khai

```python
def _compute_homography_dlt(self, src_pts, dst_pts):
    n = len(src_pts)

    # Normalize points (numerical stability)
    src_norm, T_src = self._normalize_points(src_pts)
    dst_norm, T_dst = self._normalize_points(dst_pts)

    # Build matrix A
    A = []
    for i in range(n):
        x, y = src_norm[i]
        x_p, y_p = dst_norm[i]

        # Two rows per correspondence
        A.append([-x, -y, -1,  0,  0,  0,  x*x_p,  y*x_p,  x_p])
        A.append([ 0,  0,  0, -x, -y, -1,  x*y_p,  y*y_p,  y_p])

    A = np.array(A)

    # Solve using SVD: A·h = 0
    # Solution is last column of V (smallest singular value)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T_dst) @ H @ T_src

    # Normalize so H[2,2] = 1
    H = H / H[2, 2]

    return H
```

#### 4.2.3. Point Normalization

**Tại sao cần normalize?**

- Coordinates có thể rất lớn (e.g., 1920×1080)
- Gây ill-conditioned matrix
- SVD không stable

**Cách normalize:**

1. Translate sao cho centroid ở origin
2. Scale sao cho average distance = √2

```python
def _normalize_points(self, points):
    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Center points
    points_centered = points - centroid

    # Compute average distance
    avg_dist = np.mean(np.sqrt(np.sum(points_centered**2, axis=1)))

    # Scale factor
    scale = np.sqrt(2) / avg_dist

    # Transformation matrix
    T = [[scale,     0,  -scale * centroid[0]],
         [    0, scale,  -scale * centroid[1]],
         [    0,     0,                      1]]

    return points_normalized, T
```

### 4.3. RANSAC Algorithm

**RANdom SAmple Consensus** - Robust estimation với outliers.

#### 4.3.1. Vấn Đề

Matches có thể chứa outliers (false matches):

- Wrong correspondences
- Moving objects
- Repetitive patterns

→ Cần thuật toán robust!

#### 4.3.2. Ý Tưởng RANSAC

```
Repeat N times:
    1. Sample 4 random points
    2. Compute homography H from 4 points
    3. Count inliers (points with reprojection error < threshold)
    4. Keep H with most inliers

Refine H using all inliers
```

#### 4.3.3. Triển Khai

```python
def find_homography(self, src_points, dst_points):
    best_H = None
    best_inliers = None
    best_num_inliers = 0

    n_points = len(src_points)

    for iteration in range(self.max_iters):
        # 1. Random sample 4 points
        indices = np.random.choice(n_points, 4, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        # 2. Compute homography
        H = self._compute_homography_4pts(src_sample, dst_sample)

        if H is None:
            continue

        # 3. Count inliers
        inliers = self._get_inliers(src_points, dst_points, H)
        num_inliers = np.sum(inliers)

        # 4. Update best
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_H = H

            # Adaptive termination
            inlier_ratio = num_inliers / n_points
            n_iters_needed = log(1 - confidence) / log(1 - inlier_ratio^4)
            if iteration > n_iters_needed:
                break

    # Refine với all inliers
    if best_H is not None:
        inlier_src = src_points[best_inliers]
        inlier_dst = dst_points[best_inliers]
        best_H = self._compute_homography_dlt(inlier_src, inlier_dst)

    return best_H, best_inliers
```

#### 4.3.4. Inlier Detection

Reprojection error:

```
error = ||dst_point - H·src_point||
```

```python
def _get_inliers(self, src_pts, dst_pts, H):
    # Transform source points
    src_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    dst_projected = (H @ src_homogeneous.T).T

    # Convert from homogeneous
    dst_projected = dst_projected[:, :2] / dst_projected[:, 2:3]

    # Compute errors
    errors = np.sqrt(np.sum((dst_pts - dst_projected)**2, axis=1))

    # Inliers: error < threshold
    inliers = errors < self.ransac_reproj_threshold

    return inliers
```

#### 4.3.5. Số Lượng Iterations Cần Thiết

Probability of success:

```
P(success) = 1 - (1 - p^s)^N
```

Trong đó:

- `p`: Inlier ratio (e.g., 0.5)
- `s`: Sample size (4 for homography)
- `N`: Number of iterations

Giải cho N với confidence = 0.995:

```
N = log(1 - confidence) / log(1 - p^s)
```

Ví dụ với p=0.5, s=4, confidence=0.995:

```
N = log(0.005) / log(1 - 0.5^4) ≈ 35 iterations
```

---

## 5. Image Warping

### 5.1. Perspective Transformation

Apply homography H để warp image.

#### 5.1.1. Forward vs Backward Warping

**Forward Warping** (không dùng):

```
For each pixel (x,y) in source:
    Compute (x',y') = H · (x,y)
    Set output[x',y'] = input[x,y]
```

❌ Problem: Holes trong output (không phủ kín)

**Backward Warping** (dùng):

```
For each pixel (x',y') in output:
    Compute (x,y) = H⁻¹ · (x',y')
    Set output[x',y'] = interpolate(input, x, y)
```

✅ No holes, mỗi output pixel có giá trị

#### 5.1.2. Triển Khai

```python
def warp_perspective(image, H, output_shape):
    h, w = output_shape

    # Inverse homography
    H_inv = np.linalg.inv(H)

    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([x_coords.flatten(),
                       y_coords.flatten(),
                       np.ones(h * w)], axis=1)

    # Apply inverse homography
    src_coords = (H_inv @ coords.T).T
    src_coords = src_coords[:, :2] / src_coords[:, 2:3]

    # Reshape
    src_x = src_coords[:, 0].reshape(h, w)
    src_y = src_coords[:, 1].reshape(h, w)

    # Bilinear interpolation
    output = bilinear_interpolate(image, src_x, src_y)

    return output
```

### 5.2. Bilinear Interpolation

Khi (x, y) không phải integer, interpolate từ 4 neighboring pixels.

#### 5.2.1. Công Thức

Cho pixel tại (x, y) với x = x₀ + fx, y = y₀ + fy:

```
I(x, y) = (1-fx)(1-fy)·I(x₀,y₀) + fx(1-fy)·I(x₁,y₀) +
          (1-fx)fy·I(x₀,y₁) + fx·fy·I(x₁,y₁)
```

Trong đó:

- `(x₀, y₀)`: Top-left pixel
- `fx, fy`: Fractional parts

```
    (x₀,y₀)────────(x₁,y₀)
        │             │
        │    (x,y)    │
        │      ·      │
        │             │
    (x₀,y₁)────────(x₁,y₁)
```

#### 5.2.2. Triển Khai

```python
def bilinear_interpolate(image, x, y):
    h, w = image.shape[:2]

    # Integer coordinates
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip to boundaries
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    # Fractional parts
    fx = x - x0
    fy = y - y0

    # Bounds mask
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    # Interpolate for each channel
    if len(image.shape) == 3:
        output = np.zeros((y.shape[0], y.shape[1], channels))
        for c in range(channels):
            I00 = image[y0, x0, c]
            I01 = image[y1, x0, c]
            I10 = image[y0, x1, c]
            I11 = image[y1, x1, c]

            w00 = (1 - fx) * (1 - fy)
            w01 = (1 - fx) * fy
            w10 = fx * (1 - fy)
            w11 = fx * fy

            output[:, :, c] = (w00*I00 + w01*I01 + w10*I10 + w11*I11) * mask

    return output
```

---

## 6. Weighted Blending

### 6.1. Vấn Đề Cần Giải Quyết

Khi ghép ảnh, vùng overlap có thể có:

- **Seams rõ ràng** (hard edges)
- **Differences về exposure/illumination**
- **Vignetting** (darkening ở góc ảnh)

→ Cần blending mượt mà!

### 6.2. Alpha Blending với Linear Gradient

#### 6.2.1. Ý Tưởng

Trong vùng overlap, blend hai ảnh với weights thay đổi dần:

```
Result = α·Image1 + (1-α)·Image2
```

Trong đó α thay đổi từ 1 → 0 trong vùng overlap.

#### 6.2.2. Mask Creation

```python
def _create_mask(self, canvas1, canvas2, img1, x_offset, version='left'):
    h, w = canvas1.shape[:2]

    # Find valid regions
    mask1_valid = np.any(canvas1 > 0, axis=2)
    mask2_valid = np.any(canvas2 > 0, axis=2)

    # Overlap region
    overlap = mask1_valid & mask2_valid

    # Find overlap boundaries
    overlap_cols = np.where(np.any(overlap, axis=0))[0]
    overlap_start = overlap_cols[0]
    overlap_end = overlap_cols[-1]
    overlap_width = overlap_end - overlap_start

    # Create mask
    mask = np.zeros((h, w))

    if version == 'left':
        # Left image: 1.0 → 0.0
        mask[mask1_valid] = 1.0

        for col in range(overlap_start, overlap_end + 1):
            alpha = 1.0 - (col - overlap_start) / overlap_width
            mask[overlap[:, col], col] = alpha

    else:
        # Right image: 0.0 → 1.0
        mask[mask2_valid] = 1.0

        for col in range(overlap_start, overlap_end + 1):
            alpha = (col - overlap_start) / overlap_width
            mask[overlap[:, col], col] = alpha

    return mask
```

### 6.3. Gaussian Smoothing

Làm mượt mask với Gaussian filter để tránh banding artifacts:

```python
mask = gaussian_filter(mask, sigma=smoothing_window/6.0)
```

### 6.4. Final Blending

```python
def blend_images(self, img1, img2, H):
    # Warp img2
    img2_warped = warp_perspective(img2, H_translated, canvas_size)

    # Create masks
    mask1 = self._create_mask(..., version='left')
    mask2 = self._create_mask(..., version='right')

    # Blend
    blended = canvas1 * mask1 + img2_warped * mask2

    # Crop black borders
    blended = self._crop_black_borders(blended)

    return blended
```

### 6.5. Visualization của Blending Process

```
Image 1 Mask:          Image 2 Mask:
│ 1.0                  │ 0.0
│ 1.0                  │ 0.0
│ 1.0→0.5→0.0         │ 0.0→0.5→1.0
│     overlap          │     overlap
│                      │                1.0
│                      │                1.0

Combined:
│ Image1 only
│ Image1 only
│ Smooth blend region
│             Image2 only
│             Image2 only
```

---

## 7. Tham Khảo

### Papers

1. **Lowe, D. G. (2004)**  
   _"Distinctive Image Features from Scale-Invariant Keypoints"_  
   International Journal of Computer Vision, 60(2), 91-110  
   [Link](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

2. **Brown, M., & Lowe, D. G. (2007)**  
   _"Automatic Panoramic Image Stitching using Invariant Features"_  
   International Journal of Computer Vision, 74(1), 59-73

3. **Hartley, R., & Zisserman, A. (2004)**  
   _"Multiple View Geometry in Computer Vision"_  
   Cambridge University Press (Chapter 4: Homography Estimation)

4. **Fischler, M. A., & Bolles, R. C. (1981)**  
   _"Random Sample Consensus: A Paradigm for Model Fitting"_  
   Communications of the ACM, 24(6), 381-395

### Courses

1. **First Principles of Computer Vision - Shree K. Nayar**  
   Columbia University  
   [https://fpcv.cs.columbia.edu/](https://fpcv.cs.columbia.edu/)

2. **Computer Vision - Andrew Ng**  
   Stanford University CS231n

### Tutorials

1. **OpenCV SIFT Tutorial**  
   [https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)

2. **Image Stitching with OpenCV**  
   [https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)

---

## 📞 Liên Hệ & Đóng Góp

Nếu bạn có câu hỏi hoặc muốn đóng góp cải thiện tài liệu này, vui lòng tạo issue hoặc pull request!

**Happy Learning!** 🎓✨
