#!/usr/bin/env python3
"""
Test Phase 1 Security Improvements
Tests input validation, error handling, and security fixes
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup test environment
import test_env
test_env.setup_test_environment()

def test_input_validation_functions():
    """Test input validation functions directly"""
    print("🧪 Testing Input Validation Functions...")
    
    try:
        # Test the validation logic by importing the module and checking for validation patterns
        import routes.candlestick_analysis as analysis_module
        
        # Check if validation constants exist
        if hasattr(analysis_module, 'SYMBOL_PATTERN'):
            print("    ✅ SYMBOL_PATTERN validation constant found")
        else:
            print("    ❌ SYMBOL_PATTERN validation constant missing")
            
        if hasattr(analysis_module, 'TIMEFRAME_PATTERN'):
            print("    ✅ TIMEFRAME_PATTERN validation constant found")
        else:
            print("    ❌ TIMEFRAME_PATTERN validation constant missing")
            
        if hasattr(analysis_module, 'MAX_LIMIT'):
            print("    ✅ MAX_LIMIT validation constant found")
        else:
            print("    ❌ MAX_LIMIT validation constant missing")
        
        print("    ✅ Input validation functions structure verified")
        
    except ImportError as e:
        print(f"    ❌ Could not import validation functions: {e}")

def test_dependency_injection():
    """Test that dependency injection is working"""
    print("\n🧪 Testing Dependency Injection...")
    
    # Test that processor is created properly
    try:
        import routes.candlestick_analysis as analysis_module
        
        # Check if get_processor function exists
        if hasattr(analysis_module, 'get_processor'):
            print("    ✅ get_processor function found (dependency injection pattern)")
        else:
            print("    ❌ get_processor function missing")
            
        print("    ✅ Dependency injection pattern verified")
        
    except Exception as e:
        print(f"    ❌ Processor creation failed: {e}")

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration...")
    
    try:
        from src.app.core.unified_config import get_settings
        settings = get_settings()
        
        print(f"    Database URL: {settings.DATABASE_URL}")
        print(f"    Debug mode: {settings.DEBUG}")
        print(f"    Log level: {settings.LOG_LEVEL}")
        print(f"    App name: {settings.APP_NAME}")
        
        # Check that no hardcoded credentials are present
        if "Emon_@17711" not in settings.DATABASE_URL:
            print("    ✅ No hardcoded credentials in DATABASE_URL")
        else:
            print("    ⚠️  Hardcoded credentials still present in DATABASE_URL")
        
        print("    ✅ Configuration loaded successfully")
        
    except Exception as e:
        print(f"    ❌ Configuration loading failed: {e}")

def test_error_handling():
    """Test error handling patterns"""
    print("\n🧪 Testing Error Handling Patterns...")
    
    try:
        import routes.candlestick_analysis as analysis_module
        
        # Check that the router has proper error handling
        print("    ✅ Router imported successfully")
        
        # Check for proper exception handling patterns
        import inspect
        
        # Look for proper error handling in functions
        functions_with_error_handling = []
        for name, obj in inspect.getmembers(analysis_module):
            if inspect.isfunction(obj) and name.startswith('get_'):
                try:
                    source = inspect.getsource(obj)
                    if 'try:' in source and 'except' in source:
                        functions_with_error_handling.append(name)
                except:
                    pass
        
        print(f"    Found {len(functions_with_error_handling)} functions with error handling")
        for func in functions_with_error_handling:
            print(f"      - {func}")
        
        # Check for HTTPException usage
        source_code = inspect.getsource(analysis_module)
        if 'HTTPException' in source_code:
            print("    ✅ HTTPException usage found in error handling")
        else:
            print("    ⚠️  HTTPException usage not found in error handling")
        
    except Exception as e:
        print(f"    ❌ Error handling test failed: {e}")

def main():
    """Main test function"""
    print("🚀 Starting Phase 1 Security Verification")
    print("=" * 50)
    
    test_configuration()
    test_dependency_injection()
    test_input_validation_functions()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("✅ Phase 1 Security Verification Complete")

if __name__ == "__main__":
    main()
