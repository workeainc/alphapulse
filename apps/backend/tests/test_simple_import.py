#!/usr/bin/env python3
"""Simple test for production config import"""

import sys
import traceback

def test_import():
    """Test importing production configuration"""
    try:
        print("Testing import of production config...")
        from config.production import production_config
        print("✅ Import successful!")
        print(f"Database host: {production_config.database.host}")
        print(f"Redis host: {production_config.redis.host}")
        print(f"Server port: {production_config.server.port}")
        print(f"Environment: {production_config.ENVIRONMENT}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
