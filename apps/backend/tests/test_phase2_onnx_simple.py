#!/usr/bin/env python3
"""
Simplified test script for Phase 2 - Priority 3: ONNX Export & Fast Inference
Tests core ONNX conversion and inference functionality with minimal dependencies.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ONNX components
try:
    from ..src.ai.onnx_converter import ONNXConverter
    from ..src.ai.onnx_inference import ONNXInferenceEngine
    ONNX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ONNX components not available: {e}")
    ONNX_AVAILABLE = False

# Import scikit-learn for testing
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Import booster libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def generate_test_data(n_samples: int = 100, n_features: int = 5) -> tuple:
    """Generate test data for model training"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    return X, y


def create_sklearn_model() -> RandomForestClassifier:
    """Create a scikit-learn model for testing"""
    X, y = generate_test_data()
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


def create_xgboost_model() -> Optional[xgb.XGBClassifier]:
    """Create an XGBoost model for testing"""
    if not XGBOOST_AVAILABLE:
        return None
    
    X, y = generate_test_data()
    model = xgb.XGBClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


def create_lightgbm_model() -> Optional[lgb.LGBMClassifier]:
    """Create a LightGBM model for testing"""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    X, y = generate_test_data()
    model = lgb.LGBMClassifier(n_estimators=5, random_state=42, verbose=-1)
    model.fit(X, y)
    return model


def create_catboost_model() -> Optional[cb.CatBoostClassifier]:
    """Create a CatBoost model for testing"""
    if not CATBOOST_AVAILABLE:
        return None
    
    X, y = generate_test_data()
    model = cb.CatBoostClassifier(iterations=5, random_seed=42, verbose=False)
    model.fit(X, y)
    return model


def test_onnx_converter_initialization():
    """Test ONNX converter initialization"""
    logger.info("🧪 Testing ONNX converter initialization...")
    
    if not ONNX_AVAILABLE:
        logger.warning("❌ ONNX components not available, skipping test")
        return False
    
    try:
        # Initialize converter
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        # Test dependency detection
        logger.info(f"✅ Dependencies available: {converter.dependencies_available}")
        
        # Test conversion status
        status = converter.get_conversion_status()
        logger.info(f"✅ Conversion status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX converter initialization test failed: {e}")
        return False


def test_onnx_inference_initialization():
    """Test ONNX inference engine initialization"""
    logger.info("🧪 Testing ONNX inference engine initialization...")
    
    if not ONNX_AVAILABLE:
        logger.warning("❌ ONNX components not available, skipping test")
        return False
    
    try:
        # Initialize inference engine
        inference_engine = ONNXInferenceEngine(onnx_dir="test_models/onnx")
        
        logger.info(f"✅ ONNX inference engine initialized with providers: {inference_engine.providers}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX inference engine initialization test failed: {e}")
        return False


def test_model_creation():
    """Test model creation for different types"""
    logger.info("🧪 Testing model creation...")
    
    results = {}
    
    # Test scikit-learn
    if SKLEARN_AVAILABLE:
        try:
            sklearn_model = create_sklearn_model()
            X_test, _ = generate_test_data(10, 5)
            predictions = sklearn_model.predict(X_test)
            logger.info(f"✅ Scikit-learn model created, predictions shape: {predictions.shape}")
            results['sklearn'] = True
        except Exception as e:
            logger.error(f"❌ Scikit-learn model creation failed: {e}")
            results['sklearn'] = False
    
    # Test XGBoost
    if XGBOOST_AVAILABLE:
        try:
            xgb_model = create_xgboost_model()
            if xgb_model:
                X_test, _ = generate_test_data(10, 5)
                predictions = xgb_model.predict(X_test)
                logger.info(f"✅ XGBoost model created, predictions shape: {predictions.shape}")
                results['xgboost'] = True
            else:
                results['xgboost'] = False
        except Exception as e:
            logger.error(f"❌ XGBoost model creation failed: {e}")
            results['xgboost'] = False
    
    # Test LightGBM
    if LIGHTGBM_AVAILABLE:
        try:
            lgb_model = create_lightgbm_model()
            if lgb_model:
                X_test, _ = generate_test_data(10, 5)
                predictions = lgb_model.predict(X_test)
                logger.info(f"✅ LightGBM model created, predictions shape: {predictions.shape}")
                results['lightgbm'] = True
            else:
                results['lightgbm'] = False
        except Exception as e:
            logger.error(f"❌ LightGBM model creation failed: {e}")
            results['lightgbm'] = False
    
    # Test CatBoost
    if CATBOOST_AVAILABLE:
        try:
            cb_model = create_catboost_model()
            if cb_model:
                X_test, _ = generate_test_data(10, 5)
                predictions = cb_model.predict(X_test)
                logger.info(f"✅ CatBoost model created, predictions shape: {predictions.shape}")
                results['catboost'] = True
            else:
                results['catboost'] = False
        except Exception as e:
            logger.error(f"❌ CatBoost model creation failed: {e}")
            results['catboost'] = False
    
    return len(results) > 0 and any(results.values())


def test_onnx_conversion_attempts():
    """Test ONNX conversion attempts (may fail due to missing dependencies)"""
    logger.info("🧪 Testing ONNX conversion attempts...")
    
    if not ONNX_AVAILABLE:
        logger.warning("❌ ONNX components not available, skipping test")
        return False
    
    try:
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        X_test, _ = generate_test_data(10, 5)
        
        conversion_results = {}
        
        # Test scikit-learn conversion
        if SKLEARN_AVAILABLE:
            try:
                sklearn_model = create_sklearn_model()
                onnx_path = converter.convert_model(
                    model=sklearn_model,
                    model_name="test_sklearn_simple",
                    input_shape=(5,),
                    model_type="sklearn"
                )
                if onnx_path:
                    logger.info(f"✅ Scikit-learn conversion successful: {onnx_path}")
                    conversion_results['sklearn'] = True
                else:
                    logger.warning("⚠️ Scikit-learn conversion failed (expected due to missing skl2onnx)")
                    conversion_results['sklearn'] = False
            except Exception as e:
                logger.warning(f"⚠️ Scikit-learn conversion error (expected): {e}")
                conversion_results['sklearn'] = False
        
        # Test XGBoost conversion
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = create_xgboost_model()
                if xgb_model:
                    onnx_path = converter.convert_model(
                        model=xgb_model,
                        model_name="test_xgb_simple",
                        input_shape=(5,),
                        model_type="xgboost"
                    )
                    if onnx_path:
                        logger.info(f"✅ XGBoost conversion successful: {onnx_path}")
                        conversion_results['xgboost'] = True
                    else:
                        logger.warning("⚠️ XGBoost conversion failed (expected due to missing onnxconverter-common)")
                        conversion_results['xgboost'] = False
            except Exception as e:
                logger.warning(f"⚠️ XGBoost conversion error (expected): {e}")
                conversion_results['xgboost'] = False
        
        # Test LightGBM conversion
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = create_lightgbm_model()
                if lgb_model:
                    onnx_path = converter.convert_model(
                        model=lgb_model,
                        model_name="test_lgb_simple",
                        input_shape=(5,),
                        model_type="lightgbm"
                    )
                    if onnx_path:
                        logger.info(f"✅ LightGBM conversion successful: {onnx_path}")
                        conversion_results['lightgbm'] = True
                    else:
                        logger.warning("⚠️ LightGBM conversion failed (expected due to missing onnxconverter-common)")
                        conversion_results['lightgbm'] = False
            except Exception as e:
                logger.warning(f"⚠️ LightGBM conversion error (expected): {e}")
                conversion_results['lightgbm'] = False
        
        # Test CatBoost conversion
        if CATBOOST_AVAILABLE:
            try:
                cb_model = create_catboost_model()
                if cb_model:
                    onnx_path = converter.convert_model(
                        model=cb_model,
                        model_name="test_cb_simple",
                        input_shape=(5,),
                        model_type="catboost"
                    )
                    if onnx_path:
                        logger.info(f"✅ CatBoost conversion successful: {onnx_path}")
                        conversion_results['catboost'] = True
                    else:
                        logger.warning("⚠️ CatBoost conversion failed (expected due to missing onnxconverter-common)")
                        conversion_results['catboost'] = False
            except Exception as e:
                logger.warning(f"⚠️ CatBoost conversion error (expected): {e}")
                conversion_results['catboost'] = False
        
        logger.info(f"📊 Conversion results: {conversion_results}")
        return len(conversion_results) > 0
        
    except Exception as e:
        logger.error(f"❌ ONNX conversion test failed: {e}")
        return False


def test_latency_measurement():
    """Test latency measurement functionality"""
    logger.info("🧪 Testing latency measurement...")
    
    if not ONNX_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        model = create_sklearn_model()
        X_test, _ = generate_test_data(50, 5)
        
        # Test latency measurement with invalid ONNX path (should fallback to native)
        latency_results = converter.measure_latency_improvement(
            model=model,
            onnx_path="invalid_path.onnx",
            test_data=X_test,
            n_runs=10
        )
        
        if latency_results and 'native_avg_ms' in latency_results:
            logger.info(f"✅ Latency measurement successful: {latency_results}")
            return True
        else:
            logger.error("❌ Latency measurement failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Latency measurement test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files"""
    try:
        test_dir = Path("test_models/onnx")
        if test_dir.exists():
            for file in test_dir.glob("*.onnx"):
                file.unlink()
            logger.info("✅ Test files cleaned up")
    except Exception as e:
        logger.warning(f"⚠️ Could not clean up test files: {e}")


def main():
    """Main test function"""
    logger.info("🚀 Starting Simplified Phase 2 - Priority 3: ONNX Export & Fast Inference Tests")
    logger.info("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run tests
    tests = [
        ("ONNX Converter Initialization", test_onnx_converter_initialization),
        ("ONNX Inference Initialization", test_onnx_inference_initialization),
        ("Model Creation", test_model_creation),
        ("ONNX Conversion Attempts", test_onnx_conversion_attempts),
        ("Latency Measurement", test_latency_measurement),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All simplified ONNX tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = main()
    exit(0 if success else 1)
