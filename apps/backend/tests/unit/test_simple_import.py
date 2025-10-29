#!/usr/bin/env python3
"""
Very simple test to isolate UTF-8 encoding issue
"""

import sys
import asyncio

# Add backend to path
sys.path.append('.')

async def simple_test():
    """Simple test function"""
    try:
        print("Testing production monitoring import...")
        from ..src.ai.production_monitoring import production_monitoring
        print("✅ Production monitoring imported successfully")
        
        print("Testing error recovery import...")
        from ..src.ai.error_recovery_system import error_recovery_system
        print("✅ Error recovery imported successfully")
        
        print("Testing orchestrator import...")
        from ..src.ai.retraining import retraining_orchestrator
        print("✅ Orchestrator imported successfully")
        
        print("All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(simple_test())
    if result:
        print("🎉 Test passed!")
    else:
        print("💥 Test failed!")
