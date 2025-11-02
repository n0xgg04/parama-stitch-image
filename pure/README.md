# Pure Panorama Image Stitching

Triển khai thuần túy (pure implementation) của thuật toán ghép ảnh panorama **không sử dụng OpenCV**. Chỉ dùng các thư viện toán học cơ bản như NumPy và SciPy.

## 🌟 Tính năng

- ✅ **Triển khai SIFT từ đầu**: Scale-space pyramid, DoG, keypoint detection, orientation assignment, và descriptor generation
- ✅ **Feature Matching**: Brute-force matcher với L2 distance và Lowe's ratio test
- ✅ **RANSAC Algorithm**: Ước lượng homography matrix với outlier rejection
- ✅ **Image Warping**: Perspective transformation với bilinear interpolation
- ✅ **Weighted Blending**: Kết hợp mượt mà với Gaussian smoothing

## 📦 Cài đặt

```bash
# Tạo virtual environment
python -m venv venv_pure
source venv_pure/bin/activate  # On Windows: venv_pure\Scripts\activate

# Cài đặt dependencies
pip install -r pure/requirements.txt
```

## 🚀 Sử dụng

### Cách 1: Sử dụng script CLI

```bash
# Ghép 2 ảnh
python -m pure.panorama_cli inputs/front/front_01.jpeg inputs/front/front_02.jpeg

# Ghép 3 ảnh
python -m pure.panorama_cli inputs/back/back_01.jpeg inputs/back/back_02.jpeg inputs/back/back_03.jpeg

# Với tùy chọn nâng cao
python -m pure.panorama_cli \
    inputs/front/front_01.jpeg \
    inputs/front/front_02.jpeg \
    --output pure_outputs/my_panorama.jpg \
    --smoothing 0.15 \
    --ransac-threshold 5.0 \
    --visualize
```

### Cách 2: Sử dụng trong Python

```python
from pure.image_io import read_images, write_image
from pure.panorama_stitcher import PanoramaStitcher

# Đọc ảnh
images = read_images([
    'inputs/front/front_01.jpeg',
    'inputs/front/front_02.jpeg',
    'inputs/front/front_03.jpeg'
])

# Tạo stitcher
stitcher = PanoramaStitcher(
    blending_params={'smoothing_window_percent': 0.10}
)

# Ghép ảnh
panorama = stitcher.stitch_multiple(images)

# Lưu kết quả
write_image('pure_outputs/panorama.jpg', panorama)
```

## 📐 Cấu trúc dự án

```
pure/
├── __init__.py              # Package initialization
├── sift.py                  # SIFT feature detector & descriptor
├── matcher.py               # Feature matching
├── homography.py            # Homography estimation & RANSAC
├── blending.py              # Image blending
├── panorama_stitcher.py     # Main stitching pipeline
├── image_io.py              # Image I/O utilities
├── panorama_cli.py          # Command-line interface
├── requirements.txt         # Dependencies
└── README.md               # Tài liệu này
```

## 🔬 Chi tiết thuật toán

### 1. SIFT (Scale-Invariant Feature Transform)

**File:** `sift.py`

- **Scale-space construction**: Xây dựng Gaussian pyramid với nhiều octaves và scales
- **DoG pyramid**: Tính Difference of Gaussian để phát hiện extrema
- **Keypoint localization**: Refine vị trí keypoint bằng quadratic interpolation
- **Orientation assignment**: Gán hướng dominant cho mỗi keypoint
- **Descriptor generation**: Tạo descriptor 128-chiều với histogram of gradients

### 2. Feature Matching

**File:** `matcher.py`

- **Brute-force matching**: Tính L2 distance giữa tất cả descriptor pairs
- **Lowe's ratio test**: Lọc matches bằng cách so sánh best match với second-best match
- **Cross-check**: Đảm bảo matches là bidirectional

### 3. Homography & RANSAC

**File:** `homography.py`

- **Direct Linear Transform (DLT)**: Tính homography matrix từ point correspondences
- **RANSAC algorithm**: Tìm homography tốt nhất với inlier maximization
- **Point normalization**: Chuẩn hóa điểm để tăng numerical stability
- **Perspective warping**: Biến đổi ảnh với bilinear interpolation

### 4. Image Blending

**File:** `blending.py`

- **Weighted blending**: Tạo masks với linear gradient trong vùng overlap
- **Gaussian smoothing**: Làm mượt masks để tránh seams
- **Canvas computation**: Tính toán kích thước và vị trí để chứa toàn bộ panorama
- **Black border cropping**: Tự động cắt viền đen

## ⚙️ Tham số

### SIFT Parameters

- `num_octaves` (default: 4): Số lượng octaves trong scale space
- `num_scales` (default: 5): Số scales mỗi octave
- `sigma` (default: 1.6): Base sigma cho Gaussian blur
- `contrast_threshold` (default: 0.04): Ngưỡng loại bỏ low-contrast keypoints
- `edge_threshold` (default: 10): Ngưỡng loại bỏ edge responses

### Matcher Parameters

- `ratio_threshold` (default: 0.75): Lowe's ratio test threshold
- `cross_check` (default: True): Enable cross-checking

### RANSAC Parameters

- `ransac_reproj_threshold` (default: 4.0): Maximum reprojection error (pixels)
- `max_iters` (default: 2000): Maximum RANSAC iterations
- `confidence` (default: 0.995): Desired confidence level

### Blending Parameters

- `smoothing_window_percent` (default: 0.10): Tỷ lệ vùng overlap dùng cho smoothing

## 📊 So sánh với OpenCV version

| Tính năng        | OpenCV Version          | Pure Version                       |
| ---------------- | ----------------------- | ---------------------------------- |
| SIFT             | `cv2.SIFT_create()`     | Custom implementation              |
| Matching         | `cv2.BFMatcher()`       | Custom brute-force                 |
| Homography       | `cv2.findHomography()`  | Custom RANSAC + DLT                |
| Warping          | `cv2.warpPerspective()` | Custom with bilinear interpolation |
| Blending         | `cv2.merge()` + masking | Custom weighted blending           |
| **Dependencies** | opencv-python           | numpy, scipy, Pillow               |
| **Speed**        | ⚡⚡⚡ Rất nhanh (C++)  | ⚡⚡ Nhanh (Python + NumPy)        |
| **Tùy chỉnh**    | 🔒 Black-box            | ✅ Hoàn toàn kiểm soát             |

## 🎯 Ví dụ kết quả

```bash
# Chạy với ảnh mẫu
python -m pure.panorama_cli \
    inputs/back/back_01.jpeg \
    inputs/back/back_02.jpeg \
    inputs/back/back_03.jpeg \
    --output pure_outputs/panorama.jpg \
    --visualize
```

Output:

```
____
|  _ \ __ _ _ __   ___  _ __ __ _ _ __ ___   __ _
| |_) / _` | '_ \ / _ \| '__/ _` | '_ ` _ \ / _` |
|  __/ (_| | | | | (_) | | | (_| | | | | | | (_| |
|_|   \__,_|_| |_|\___/|_|  \__,_|_| |_| |_|\__,_|

Pure Implementation (No OpenCV)

Initializing...
Input images: 3

Reading images...
  Loaded 3 images

Stitching 3 images...
  Detecting features in image 1...
    Found 2847 keypoints
  ...

✓ Success!
  Panorama saved to: pure_outputs/panorama.jpg
  Processing time: 45.23 seconds
```

## 🔧 Troubleshooting

### Không đủ keypoints

Nếu gặp lỗi "Not enough keypoints detected":

- Giảm `contrast_threshold` trong SIFT parameters
- Tăng `num_octaves` hoặc `num_scales`
- Kiểm tra ảnh có đủ texture/features không

### Homography thất bại

Nếu "Failed to compute homography":

- Tăng `ransac_reproj_threshold`
- Tăng `max_iters`
- Kiểm tra ảnh có đủ overlap không (ít nhất 30-40%)

### Seams hiện rõ trong panorama

- Tăng `smoothing_window_percent` (thử 0.15-0.20)
- Đảm bảo ảnh có exposure tương đương nhau

### Chậm quá

- Giảm resolution của ảnh input
- Giảm `num_octaves` xuống 3
- Giảm `num_scales` xuống 3-4

## 📚 Tài liệu tham khảo

- [SIFT Paper - David Lowe](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [Homography Estimation](https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-21-homography-estimation.pptx.pdf)

## 📝 License

Cùng license với project chính.

## 👨‍💻 Tác giả

Pure implementation được tạo để hiểu sâu về thuật toán panorama stitching mà không phụ thuộc vào OpenCV.
