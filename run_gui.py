#!/usr/bin/env python3
"""
Launcher script cho GUI application.
Chạy: python run_gui.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

try:
    from pure.gui_app import main
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

if __name__ == '__main__':
    main()

