"""
Simple Production Configuration Test
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.production import production_config
    print("✅ Production config imported successfully")
    print(f"Environment: {production_config.ENVIRONMENT}")
    print(f"API Host: {production_config.API_HOST}")
    print(f"API Port: {production_config.API_PORT}")
    print("✅ Simple production test passed")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
