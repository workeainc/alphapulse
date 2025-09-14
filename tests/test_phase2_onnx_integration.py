#!/usr/bin/env python3
"""
Test script for Phase 2 - Priority 3: ONNX Export & Fast Inference
Tests conversion of trained boosters (XGBoost, LightGBM, CatBoost) to ONNX,
latency measurement, and fallback mechanisms.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ONNX components
try:
    from ..ai.onnx_converter import ONNXConverter
    from ..ai.onnx_inference import ONNXInferenceEngine
    ONNX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ONNX components not available: {e}")
    ONNX_AVAILABLE = False

# Import ML models for testing
try:
    from ..ai.ml_models.trainer import MLModelTrainer, ModelType
    from ..ai.ml_models.online_learner import OnlineLearner
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML models not available: {e}")
    ML_MODELS_AVAILABLE = False

# Import scikit-learn for comparison
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
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


class MockModelRegistry:
    """Mock model registry for testing"""
    
    def __init__(self):
        self.models = {}
        self.model_versions = {}
    
    def register_model(self, model_id: str, model: Any, version: str = "1.0"):
        """Register a model"""
        self.models[model_id] = model
        self.model_versions[model_id] = version
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a model by ID"""
        return self.models.get(model_id)
    
    def get_model_version(self, model_id: str) -> Optional[str]:
        """Get model version"""
        return self.model_versions.get(model_id)


def generate_test_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """Generate test data for model training"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y


def create_sklearn_model() -> RandomForestClassifier:
    """Create a scikit-learn model for testing"""
    X, y = generate_test_data()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def create_xgboost_model() -> Optional[xgb.XGBClassifier]:
    """Create an XGBoost model for testing"""
    if not XGBOOST_AVAILABLE:
        return None
    
    X, y = generate_test_data()
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def create_lightgbm_model() -> Optional[lgb.LGBMClassifier]:
    """Create a LightGBM model for testing"""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    X, y = generate_test_data()
    model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def create_catboost_model() -> Optional[cb.CatBoostClassifier]:
    """Create a CatBoost model for testing"""
    if not CATBOOST_AVAILABLE:
        return None
    
    X, y = generate_test_data()
    model = cb.CatBoostClassifier(iterations=10, random_seed=42, verbose=False)
    model.fit(X, y)
    return model


def test_onnx_converter_basic():
    """Test basic ONNX converter functionality"""
    logger.info("🧪 Testing basic ONNX converter functionality...")
    
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
        logger.error(f"❌ Basic ONNX converter test failed: {e}")
        return False


def test_sklearn_conversion():
    """Test scikit-learn model conversion"""
    logger.info("🧪 Testing scikit-learn model conversion...")
    
    if not ONNX_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create test model
        model = create_sklearn_model()
        X_test, _ = generate_test_data(100, 10)
        
        # Initialize converter
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        # Convert to ONNX
        onnx_path = converter.convert_model(
            model=model,
            model_name="test_sklearn_rf",
            input_shape=(10,),
            model_type="sklearn"
        )
        
        if onnx_path:
            logger.info(f"✅ Scikit-learn model converted: {onnx_path}")
            
            # Test latency improvement
            latency_results = converter.measure_latency_improvement(
                model=model,
                onnx_path=onnx_path,
                test_data=X_test,
                n_runs=50
            )
            
            logger.info(f"✅ Latency results: {latency_results}")
            return True
        else:
            logger.error("❌ Scikit-learn model conversion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Scikit-learn conversion test failed: {e}")
        return False


def test_xgboost_conversion():
    """Test XGBoost model conversion"""
    logger.info("🧪 Testing XGBoost model conversion...")
    
    if not ONNX_AVAILABLE or not XGBOOST_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create test model
        model = create_xgboost_model()
        if model is None:
            logger.warning("❌ Could not create XGBoost model")
            return False
        
        X_test, _ = generate_test_data(100, 10)
        
        # Initialize converter
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        # Convert to ONNX
        onnx_path = converter.convert_model(
            model=model,
            model_name="test_xgboost",
            input_shape=(10,),
            model_type="xgboost"
        )
        
        if onnx_path:
            logger.info(f"✅ XGBoost model converted: {onnx_path}")
            
            # Test latency improvement
            latency_results = converter.measure_latency_improvement(
                model=model,
                onnx_path=onnx_path,
                test_data=X_test,
                n_runs=50
            )
            
            logger.info(f"✅ XGBoost latency results: {latency_results}")
            return True
        else:
            logger.error("❌ XGBoost model conversion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ XGBoost conversion test failed: {e}")
        return False


def test_lightgbm_conversion():
    """Test LightGBM model conversion"""
    logger.info("🧪 Testing LightGBM model conversion...")
    
    if not ONNX_AVAILABLE or not LIGHTGBM_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create test model
        model = create_lightgbm_model()
        if model is None:
            logger.warning("❌ Could not create LightGBM model")
            return False
        
        X_test, _ = generate_test_data(100, 10)
        
        # Initialize converter
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        # Convert to ONNX
        onnx_path = converter.convert_model(
            model=model,
            model_name="test_lightgbm",
            input_shape=(10,),
            model_type="lightgbm"
        )
        
        if onnx_path:
            logger.info(f"✅ LightGBM model converted: {onnx_path}")
            
            # Test latency improvement
            latency_results = converter.measure_latency_improvement(
                model=model,
                onnx_path=onnx_path,
                test_data=X_test,
                n_runs=50
            )
            
            logger.info(f"✅ LightGBM latency results: {latency_results}")
            return True
        else:
            logger.error("❌ LightGBM model conversion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ LightGBM conversion test failed: {e}")
        return False


def test_catboost_conversion():
    """Test CatBoost model conversion"""
    logger.info("🧪 Testing CatBoost model conversion...")
    
    if not ONNX_AVAILABLE or not CATBOOST_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create test model
        model = create_catboost_model()
        if model is None:
            logger.warning("❌ Could not create CatBoost model")
            return False
        
        X_test, _ = generate_test_data(100, 10)
        
        # Initialize converter
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        # Convert to ONNX
        onnx_path = converter.convert_model(
            model=model,
            model_name="test_catboost",
            input_shape=(10,),
            model_type="catboost"
        )
        
        if onnx_path:
            logger.info(f"✅ CatBoost model converted: {onnx_path}")
            
            # Test latency improvement
            latency_results = converter.measure_latency_improvement(
                model=model,
                onnx_path=onnx_path,
                test_data=X_test,
                n_runs=50
            )
            
            logger.info(f"✅ CatBoost latency results: {latency_results}")
            return True
        else:
            logger.error("❌ CatBoost model conversion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ CatBoost conversion test failed: {e}")
        return False


def test_onnx_inference():
    """Test ONNX inference functionality"""
    logger.info("🧪 Testing ONNX inference functionality...")
    
    if not ONNX_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create test model and convert to ONNX
        model = create_sklearn_model()
        X_test, _ = generate_test_data(100, 10)
        
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        onnx_path = converter.convert_model(
            model=model,
            model_name="test_inference",
            input_shape=(10,),
            model_type="sklearn"
        )
        
        if not onnx_path:
            logger.error("❌ Could not create ONNX model for inference test")
            return False
        
        # Test ONNX inference
        onnx_inference = ONNXInferenceEngine(onnx_dir="test_models/onnx")
        
        # Load model
        model_loaded = onnx_inference.load_model("test_inference")
        if not model_loaded:
            logger.error("❌ Could not load ONNX model")
            return False
        
        # Test prediction
        predictions = onnx_inference.predict("test_inference", X_test)
        if predictions is not None:
            logger.info(f"✅ ONNX inference successful, predictions shape: {predictions.shape}")
            
            # Compare with original model
            original_predictions = model.predict(X_test)
            accuracy = np.mean(predictions == original_predictions)
            logger.info(f"✅ Prediction accuracy: {accuracy:.4f}")
            
            return accuracy > 0.95  # Should be very close
        else:
            logger.error("❌ ONNX inference failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ ONNX inference test failed: {e}")
        return False


def test_fallback_mechanism():
    """Test fallback mechanism when ONNX fails"""
    logger.info("🧪 Testing fallback mechanism...")
    
    if not ONNX_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create test model
        model = create_sklearn_model()
        X_test, _ = generate_test_data(100, 10)
        
        # Test with invalid ONNX path (should fallback to native)
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        # Simulate ONNX failure by using invalid path
        latency_results = converter.measure_latency_improvement(
            model=model,
            onnx_path="invalid_path.onnx",
            test_data=X_test,
            n_runs=10
        )
        
        # Should still return results with fallback
        if latency_results and 'native_latency_ms' in latency_results:
            logger.info(f"✅ Fallback mechanism working: {latency_results}")
            return True
        else:
            logger.error("❌ Fallback mechanism failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Fallback mechanism test failed: {e}")
        return False


def test_ml_models_integration():
    """Test integration with ML models package"""
    logger.info("🧪 Testing ML models integration...")
    
    if not ONNX_AVAILABLE or not ML_MODELS_AVAILABLE:
        logger.warning("❌ Required dependencies not available, skipping test")
        return False
    
    try:
        # Create trainer
        trainer = MLModelTrainer()
        
        # Generate test data
        X, y = generate_test_data(500, 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        if XGBOOST_AVAILABLE:
            xgb_model = trainer.train_model(
                X_train, y_train,
                model_type=ModelType.XGBOOST,
                config={
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            )
            
            if xgb_model:
                # Convert to ONNX
                converter = ONNXConverter(onnx_dir="test_models/onnx")
                onnx_path = converter.convert_model(
                    model=xgb_model,
                    model_name="ml_trainer_xgb",
                    input_shape=(10,),
                    model_type="xgboost"
                )
                
                if onnx_path:
                    logger.info(f"✅ ML trainer XGBoost converted: {onnx_path}")
                    
                    # Test latency improvement
                    latency_results = converter.measure_latency_improvement(
                        model=xgb_model,
                        onnx_path=onnx_path,
                        test_data=X_test,
                        n_runs=30
                    )
                    
                    logger.info(f"✅ ML trainer latency results: {latency_results}")
                    return True
        
        logger.warning("❌ XGBoost not available for ML models integration test")
        return False
        
    except Exception as e:
        logger.error(f"❌ ML models integration test failed: {e}")
        return False


def test_performance_comparison():
    """Test performance comparison between different model types"""
    logger.info("🧪 Testing performance comparison...")
    
    if not ONNX_AVAILABLE:
        logger.warning("❌ ONNX not available, skipping test")
        return False
    
    try:
        X_test, _ = generate_test_data(200, 10)
        converter = ONNXConverter(onnx_dir="test_models/onnx")
        
        results = {}
        
        # Test scikit-learn
        if SKLEARN_AVAILABLE:
            sklearn_model = create_sklearn_model()
            sklearn_onnx = converter.convert_model(
                sklearn_model, "perf_sklearn", (10,), "sklearn"
            )
            if sklearn_onnx:
                sklearn_latency = converter.measure_latency_improvement(
                    sklearn_model, sklearn_onnx, X_test, 50
                )
                results['sklearn'] = sklearn_latency
        
        # Test XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = create_xgboost_model()
            if xgb_model:
                xgb_onnx = converter.convert_model(
                    xgb_model, "perf_xgb", (10,), "xgboost"
                )
                if xgb_onnx:
                    xgb_latency = converter.measure_latency_improvement(
                        xgb_model, xgb_onnx, X_test, 50
                    )
                    results['xgboost'] = xgb_latency
        
        # Test LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_model = create_lightgbm_model()
            if lgb_model:
                lgb_onnx = converter.convert_model(
                    lgb_model, "perf_lgb", (10,), "lightgbm"
                )
                if lgb_onnx:
                    lgb_latency = converter.measure_latency_improvement(
                        lgb_model, lgb_onnx, X_test, 50
                    )
                    results['lightgbm'] = lgb_latency
        
        # Test CatBoost
        if CATBOOST_AVAILABLE:
            cb_model = create_catboost_model()
            if cb_model:
                cb_onnx = converter.convert_model(
                    cb_model, "perf_cb", (10,), "catboost"
                )
                if cb_onnx:
                    cb_latency = converter.measure_latency_improvement(
                        cb_model, cb_onnx, X_test, 50
                    )
                    results['catboost'] = cb_latency
        
        # Log results
        logger.info("📊 Performance Comparison Results:")
        for model_type, latency_data in results.items():
            if latency_data:
                improvement = latency_data.get('improvement_pct', 0)
                logger.info(f"  {model_type}: {improvement:.2f}% improvement")
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
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


async def main():
    """Main test function"""
    logger.info("🚀 Starting Phase 2 - Priority 3: ONNX Export & Fast Inference Tests")
    logger.info("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run tests
    tests = [
        ("Basic ONNX Converter", test_onnx_converter_basic),
        ("Scikit-learn Conversion", test_sklearn_conversion),
        ("XGBoost Conversion", test_xgboost_conversion),
        ("LightGBM Conversion", test_lightgbm_conversion),
        ("CatBoost Conversion", test_catboost_conversion),
        ("ONNX Inference", test_onnx_inference),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("ML Models Integration", test_ml_models_integration),
        ("Performance Comparison", test_performance_comparison),
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
        logger.info("🎉 All ONNX Export & Fast Inference tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
