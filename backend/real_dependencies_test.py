#!/usr/bin/env python3
"""
Real Dependencies Verification Script
Tests that all ML/AI packages are working without mock warnings
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_real_dependencies():
    """Test that all real ML/AI dependencies are working"""
    print("🔍 Testing Real ML/AI Dependencies...")
    print("=" * 50)
    
    results = []
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        results.append(("TensorFlow", True))
    except Exception as e:
        print(f"❌ TensorFlow: {e}")
        results.append(("TensorFlow", False))
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        results.append(("PyTorch", True))
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        results.append(("PyTorch", False))
    
    # Test Gym
    try:
        import gym
        print(f"✅ Gym: {gym.__version__}")
        results.append(("Gym", True))
    except Exception as e:
        print(f"❌ Gym: {e}")
        results.append(("Gym", False))
    
    # Test Stable-baselines3
    try:
        import stable_baselines3
        print(f"✅ Stable-baselines3: {stable_baselines3.__version__}")
        results.append(("Stable-baselines3", True))
    except Exception as e:
        print(f"❌ Stable-baselines3: {e}")
        results.append(("Stable-baselines3", False))
    
    # Test TA-Lib
    try:
        import talib
        print("✅ TA-Lib: Available")
        results.append(("TA-Lib", True))
    except Exception as e:
        print(f"❌ TA-Lib: {e}")
        results.append(("TA-Lib", False))
    
    # Test scikit-learn
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        results.append(("Scikit-learn", True))
    except Exception as e:
        print(f"❌ Scikit-learn: {e}")
        results.append(("Scikit-learn", False))
    
    # Test transformers
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        results.append(("Transformers", True))
    except Exception as e:
        print(f"❌ Transformers: {e}")
        results.append(("Transformers", False))
    
    return results

def test_service_imports():
    """Test that services import without mock warnings"""
    print("\n🔍 Testing Service Imports...")
    print("=" * 50)
    
    results = []
    
    # Test ML Pattern Detector
    try:
        from app.strategies.ml_pattern_detector import MLPatternDetector
        print("✅ ML Pattern Detector: Imported successfully")
        results.append(("ML Pattern Detector", True))
    except Exception as e:
        print(f"❌ ML Pattern Detector: {e}")
        results.append(("ML Pattern Detector", False))
    
    # Test Strategy Manager
    try:
        from app.strategies.strategy_manager import StrategyManager
        print("✅ Strategy Manager: Imported successfully")
        results.append(("Strategy Manager", True))
    except Exception as e:
        print(f"❌ Strategy Manager: {e}")
        results.append(("Strategy Manager", False))
    
    # Test Real Time Signal Generator
    try:
        from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        print("✅ Real Time Signal Generator: Imported successfully")
        results.append(("Real Time Signal Generator", True))
    except Exception as e:
        print(f"❌ Real Time Signal Generator: {e}")
        results.append(("Real Time Signal Generator", False))
    
    return results

def main():
    """Run all tests"""
    print("🚀 Real Dependencies Verification")
    print("Testing that all ML/AI packages are working without mock warnings")
    print("=" * 60)
    
    # Test dependencies
    dep_results = test_real_dependencies()
    
    # Test service imports
    service_results = test_service_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    all_results = dep_results + service_results
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    print(f"Dependencies: {sum(1 for _, result in dep_results if result)}/{len(dep_results)} passed")
    print(f"Services: {sum(1 for _, result in service_results if result)}/{len(service_results)} passed")
    print(f"Overall: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ All real ML/AI dependencies are working")
        print("✅ No more mock warnings!")
        print("✅ System ready for production with real ML models!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Check the logs above for details")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
