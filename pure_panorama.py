#!/usr/bin/env python3
"""
Wrapper script for pure panorama stitching.
Makes it easier to run without the -m flag.

Usage:
    python pure_panorama.py image1.jpg image2.jpg image3.jpg
"""

import sys
from pure.panorama_cli import main

if __name__ == '__main__':
    sys.exit(main())

