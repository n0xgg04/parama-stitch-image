#!/usr/bin/env python3
"""
Test script for pure panorama stitching implementation.

This script tests the pure implementation on sample images.
"""

import os
import sys
from image_io import read_images, write_image
from panorama_stitcher import PanoramaStitcher


def test_two_images():
    """Test stitching two images."""
    print("\n" + "="*60)
    print("TEST 1: Stitching 2 images")
    print("="*60)
    
    # Input images
    img_paths = [
        '../inputs/front/front_01.jpeg',
        '../inputs/front/front_02.jpeg',
    ]
    
    # Check if images exist
    for path in img_paths:
        if not os.path.exists(path):
            print(f"❌ Image not found: {path}")
            return False
    
    try:
        # Read images
        print("Reading images...")
        images = read_images(img_paths)
        print(f"  ✓ Loaded {len(images)} images")
        
        # Create stitcher
        print("Creating stitcher...")
        stitcher = PanoramaStitcher(
            blending_params={'smoothing_window_percent': 0.10}
        )
        
        # Stitch
        print("Stitching...")
        result, debug = stitcher.stitch_pair(
            images[0], images[1], return_debug_info=True
        )
        
        # Save result
        output_path = '../pure_outputs/test_2_images.jpg'
        write_image(output_path, result)
        print(f"  ✓ Result saved to: {output_path}")
        print(f"  ✓ Result shape: {result.shape}")
        print(f"  ✓ Keypoints: {len(debug['keypoints1'])} + {len(debug['keypoints2'])}")
        print(f"  ✓ Matches: {len(debug['matches'])}")
        print(f"  ✓ Inliers: {debug['num_inliers']}")
        
        # Save visualization
        print("Creating visualization...")
        vis = stitcher.visualize_matches(
            images[0], images[1],
            debug['keypoints1'],
            debug['keypoints2'],
            debug['matches'][:50]
        )
        vis_path = '../pure_outputs/test_2_images_matches.jpg'
        write_image(vis_path, vis)
        print(f"  ✓ Visualization saved to: {vis_path}")
        
        print("✅ TEST 1 PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_three_images():
    """Test stitching three images."""
    print("\n" + "="*60)
    print("TEST 2: Stitching 3 images")
    print("="*60)
    
    # Input images
    img_paths = [
        '../inputs/back/back_01.jpeg',
        '../inputs/back/back_02.jpeg',
        '../inputs/back/back_03.jpeg',
    ]
    
    # Check if images exist
    for path in img_paths:
        if not os.path.exists(path):
            print(f"❌ Image not found: {path}")
            return False
    
    try:
        # Read images
        print("Reading images...")
        images = read_images(img_paths)
        print(f"  ✓ Loaded {len(images)} images")
        
        # Create stitcher
        print("Creating stitcher...")
        stitcher = PanoramaStitcher(
            blending_params={'smoothing_window_percent': 0.10}
        )
        
        # Stitch
        print("Stitching...")
        result = stitcher.stitch_multiple(images)
        
        # Save result
        output_path = '../pure_outputs/test_3_images.jpg'
        write_image(output_path, result)
        print(f"  ✓ Result saved to: {output_path}")
        print(f"  ✓ Result shape: {result.shape}")
        
        print("✅ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_room_images():
    """Test stitching room images."""
    print("\n" + "="*60)
    print("TEST 3: Stitching room images")
    print("="*60)
    
    # Input images
    img_paths = [
        '../inputs/room/room01.jpeg',
        '../inputs/room/room02.jpeg',
    ]
    
    # Check if images exist
    for path in img_paths:
        if not os.path.exists(path):
            print(f"❌ Image not found: {path}")
            print("  (Skipping - images may not exist)")
            return True  # Don't fail if optional images missing
    
    try:
        # Read images
        print("Reading images...")
        images = read_images(img_paths)
        print(f"  ✓ Loaded {len(images)} images")
        
        # Create stitcher
        print("Creating stitcher...")
        stitcher = PanoramaStitcher(
            blending_params={'smoothing_window_percent': 0.12}
        )
        
        # Stitch
        print("Stitching...")
        result = stitcher.stitch_multiple(images)
        
        # Save result
        output_path = '../pure_outputs/test_room.jpg'
        write_image(output_path, result)
        print(f"  ✓ Result saved to: {output_path}")
        print(f"  ✓ Result shape: {result.shape}")
        
        print("✅ TEST 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  PURE PANORAMA STITCHING - TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Create output directory
    os.makedirs('../pure_outputs', exist_ok=True)
    
    # Run tests
    results = []
    results.append(("Test 1: Two images", test_two_images()))
    results.append(("Test 2: Three images", test_three_images()))
    results.append(("Test 3: Room images", test_room_images()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

