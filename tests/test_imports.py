#!/usr/bin/env python3
"""
Test imports step by step to identify blocking issues
"""
import sys
import time

def test_import(module_name, description):
    """Test importing a module and report success/failure"""
    print(f"Testing import: {description}")
    start_time = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start_time
        print(f"✅ {description} imported successfully in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Imports Step by Step ===")
    
    # Test basic imports
    test_import("fastapi", "FastAPI")
    test_import("uvicorn", "Uvicorn")
    test_import("asyncpg", "AsyncPG")
    test_import("ccxt", "CCXT")
    test_import("pandas", "Pandas")
    
    # Test our app imports
    print("\n=== Testing App Imports ===")
    test_import("app", "App package")
    test_import("app.main_intelligent", "Main intelligent app")
    
    print("\n=== Testing Component Imports ===")
    test_import("app.data_collection", "Data collection")
    test_import("app.analysis", "Analysis")
    test_import("app.signals", "Signals")
    test_import("app.services", "Services")
    
    print("\n=== Testing AI Imports ===")
    test_import("ai", "AI package")
    test_import("ai.sde_framework", "SDE Framework")
    
    print("\n=== Testing Strategy Imports ===")
    test_import("app.strategies", "Strategies")
    
    print("\n=== Testing Core Imports ===")
    test_import("app.core", "Core")
    
    print("\n=== Import Test Complete ===")
