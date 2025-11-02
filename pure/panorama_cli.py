#!/usr/bin/env python3
"""
Pure Panorama Stitching CLI
Command-line interface for panorama stitching without OpenCV.

Usage:
    python pure/panorama_cli.py image1.jpg image2.jpg image3.jpg [options]
"""

import sys
import os
import argparse
import time
from .image_io import read_images, write_image
from .panorama_stitcher import PanoramaStitcher


def print_banner():
    """Print ASCII art banner."""
    banner = """
____                                             
|  _ \ __ _ _ __   ___  _ __ __ _ _ __ ___   __ _ 
| |_) / _` | '_ \ / _ \| '__/ _` | '_ ` _ \ / _` |
|  __/ (_| | | | | (_) | | | (_| | | | | | | (_| |
|_|   \__,_|_| |_|\___/|_|  \__,_|_| |_| |_|\__,_|

Pure Implementation (No OpenCV)
    """
    print(banner)


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Stitch images into panorama using pure Python implementation'
    )
    
    parser.add_argument(
        'images',
        nargs='+',
        help='Input images (left to right order)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='pure_outputs/panorama_image.jpg',
        help='Output panorama image path (default: pure_outputs/panorama_image.jpg)'
    )
    
    parser.add_argument(
        '--matched-output',
        default='pure_outputs/matched_features.jpg',
        help='Output matched features visualization (default: pure_outputs/matched_features.jpg)'
    )
    
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.10,
        help='Smoothing window percentage (0.0-1.0, default: 0.10)'
    )
    
    parser.add_argument(
        '--sift-octaves',
        type=int,
        default=4,
        help='Number of SIFT octaves (default: 4)'
    )
    
    parser.add_argument(
        '--sift-scales',
        type=int,
        default=5,
        help='Number of scales per octave (default: 5)'
    )
    
    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=4.0,
        help='RANSAC reprojection threshold (default: 4.0)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate matched features visualization'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check input files
    if len(args.images) < 2:
        print("Error: Need at least 2 images to stitch")
        return 1
    
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            return 1
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nInitializing...")
    print(f"Input images: {len(args.images)}")
    
    # Read images
    print("\nReading images...")
    try:
        images = read_images(args.images)
        print(f"  Loaded {len(images)} images")
        for i, img in enumerate(images):
            print(f"  Image {i+1}: {img.shape}")
    except Exception as e:
        print(f"Error reading images: {str(e)}")
        return 1
    
    # Initialize stitcher
    print("\nInitializing stitcher...")
    stitcher = PanoramaStitcher(
        sift_params={
            'num_octaves': args.sift_octaves,
            'num_scales': args.sift_scales,
            'contrast_threshold': 0.01,  # Giảm threshold để detect nhiều keypoints hơn
            'edge_threshold': 20,  # Tăng để accept nhiều edges hơn
            'border_width': 3,  # Giảm border để có nhiều keypoints hơn
        },
        matcher_params={
            'ratio_threshold': 0.8,  # Tăng để accept nhiều matches hơn (default 0.75)
            'cross_check': True,
        },
        ransac_params={
            'ransac_reproj_threshold': args.ransac_threshold,
            'max_iters': 3000,  # Tăng iterations
            'min_inliers': 8,  # Giảm min inliers
        },
        blending_params={
            'smoothing_window_percent': args.smoothing,
        }
    )
    
    # Stitch images
    start_time = time.time()
    
    try:
        if len(images) == 2 and args.visualize:
            # For pair, generate visualization
            result, debug_info = stitcher.stitch_pair(
                images[0], images[1], return_debug_info=True
            )
            
            # Create visualization
            print("\nGenerating matched features visualization...")
            vis = stitcher.visualize_matches(
                images[0], images[1],
                debug_info['keypoints1'],
                debug_info['keypoints2'],
                debug_info['matches'],
                max_matches=100
            )
            
            matched_dir = os.path.dirname(args.matched_output)
            if matched_dir and not os.path.exists(matched_dir):
                os.makedirs(matched_dir)
            
            write_image(args.matched_output, vis)
            print(f"  Matched features saved to: {args.matched_output}")
        
        else:
            # Stitch multiple images
            result = stitcher.stitch_multiple(images)
        
        elapsed_time = time.time() - start_time
        
        # Save result
        print(f"\nSaving panorama...")
        write_image(args.output, result)
        
        print(f"\n✓ Success!")
        print(f"  Panorama saved to: {args.output}")
        print(f"  Final size: {result.shape}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        print(f"\nError during stitching: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

