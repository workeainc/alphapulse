#!/usr/bin/env python3
"""Simple import test"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

print("Testing import...")

try:
    import config.prod_config
    print("✅ Simple import successful!")
except Exception as e:
    print(f"❌ Simple import failed: {e}")

try:
    import config.production
    print("✅ Production import successful!")
except Exception as e:
    print(f"❌ Production import failed: {e}")

print("Test completed!")
