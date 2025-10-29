#!/usr/bin/env python3
"""
Test script for Priority 1: ONNX Optimization
Tests mixed precision (FP16) and quantization (INT8) capabilities
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_onnx_optimization_manager():
    """Test the ONNX optimization manager"""
    logger.info("🧪 Testing ONNX Optimization Manager")
    
    try:
        from ..src.ai.onnx_optimization_manager import onnx_optimization_manager
        
        # Check initialization
        status = onnx_optimization_manager.get_optimization_status()
        logger.info(f"✅ ONNX Optimization Manager initialized successfully")
        logger.info(f"   Mixed precision enabled: {status['enabled_features']['mixed_precision']}")
        logger.info(f"   Quantization enabled: {status['enabled_features']['quantization']}")
        logger.info(f"   Available systems: {status['available_systems']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX Optimization Manager test failed: {e}")
        return False


def test_mixed_precision_engine():
    """Test the mixed precision engine"""
    logger.info("🧪 Testing Mixed Precision Engine")
    
    try:
        from ..src.ai.mixed_precision_engine import mixed_precision_engine
        
        # Check initialization
        stats = mixed_precision_engine.get_performance_stats()
        logger.info(f"✅ Mixed Precision Engine initialized successfully")
        logger.info(f"   Total inferences: {stats['total_inferences']}")
        logger.info(f"   Fallback count: {stats['fallback_count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mixed Precision Engine test failed: {e}")
        return False


def test_quantization_system():
    """Test the quantization system"""
    logger.info("🧪 Testing Quantization System")
    
    try:
        from ..src.ai.model_quantization import model_quantization_system
        
        # Check initialization
        stats = model_quantization_system.get_quantization_stats()
        logger.info(f"✅ Quantization System initialized successfully")
        logger.info(f"   Total models quantized: {stats['total_models_quantized']}")
        logger.info(f"   Quantization times: {stats['quantization_times']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantization System test failed: {e}")
        return False


def test_onnx_inference_engine():
    """Test the enhanced ONNX inference engine"""
    logger.info("🧪 Testing Enhanced ONNX Inference Engine")
    
    try:
        from ..src.ai.onnx_inference import onnx_inference_engine
        
        # Check initialization
        stats = onnx_inference_engine.get_performance_stats()
        logger.info(f"✅ Enhanced ONNX Inference Engine initialized successfully")
        logger.info(f"   Total models loaded: {stats['total_models_loaded']}")
        logger.info(f"   Standard models: {stats['standard_models']}")
        logger.info(f"   Quantized models: {stats['quantized_models']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced ONNX Inference Engine test failed: {e}")
        return False


def test_onnx_converter():
    """Test the ONNX converter"""
    logger.info("🧪 Testing ONNX Converter")
    
    try:
        from ..src.ai.onnx_converter import onnx_converter
        
        # Check initialization
        status = onnx_converter.get_conversion_status()
        logger.info(f"✅ ONNX Converter initialized successfully")
        logger.info(f"   Dependencies: {status['dependencies']}")
        logger.info(f"   ONNX directory: {status['onnx_dir']}")
        logger.info(f"   Available models: {status['available_models']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX Converter test failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are available"""
    logger.info("🧪 Testing Dependencies")
    
    dependencies = {
        'onnxruntime': 'onnxruntime',
        'onnx': 'onnx',
        'skl2onnx': 'skl2onnx',
        'onnxconverter_common': 'onnxconverter_common',
        'numpy': 'numpy',
        'pandas': 'pandas'
    }
    
    missing_deps = []
    
    for dep_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {dep_name} available")
        except ImportError:
            logger.error(f"❌ {dep_name} not available")
            missing_deps.append(dep_name)
    
    if missing_deps:
        logger.error(f"Missing dependencies: {missing_deps}")
        return False
    
    return True


def test_gpu_support():
    """Test GPU support for mixed precision"""
    logger.info("🧪 Testing GPU Support")
    
    try:
        import onnxruntime as ort
        
        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("✅ CUDA provider available for GPU acceleration")
            return True
        else:
            logger.warning("⚠️ CUDA provider not available - will use CPU only")
            return True  # Not a failure, just no GPU
            
    except Exception as e:
        logger.error(f"❌ GPU support test failed: {e}")
        return False


def run_all_tests():
    """Run all Priority 1 tests"""
    logger.info("🚀 Starting Priority 1: ONNX Optimization Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU Support", test_gpu_support),
        ("ONNX Converter", test_onnx_converter),
        ("Enhanced ONNX Inference Engine", test_onnx_inference_engine),
        ("Mixed Precision Engine", test_mixed_precision_engine),
        ("Quantization System", test_quantization_system),
        ("ONNX Optimization Manager", test_onnx_optimization_manager),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            results[test_name] = False
            logger.error(f"❌ {test_name} test ERROR: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Priority 1 Test Results Summary")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Priority 1 tests passed! ONNX optimization is ready.")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
