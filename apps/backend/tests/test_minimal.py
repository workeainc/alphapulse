#!/usr/bin/env python3
"""Minimal test for production config import"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    print("Attempting to import production config...")
    import config.production
    print("✅ Import successful!")
    
    # Test accessing the config
    from config.production import production_config
    print(f"Database host: {production_config.DATABASE_CONFIG['host']}")
    print(f"Redis host: {production_config.REDIS_CONFIG['host']}")
    print("✅ Configuration access successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ All tests passed!")
