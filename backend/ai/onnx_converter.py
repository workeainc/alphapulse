"""
ONNX Model Converter for AlphaPulse
Converts scikit-learn models and trained boosters to ONNX format for faster inference

Phase 2 Enhancements:
- Convert trained boosters to ONNX (XGBoost, LightGBM, CatBoost)
- Use onnxmltools and onnxconverter-common
- Measure latency improvement
- Fallback to native booster if ONNX fails
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import pickle
import onnx
import onnxruntime as ort
from datetime import datetime
import time

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# ONNX conversion imports
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False
    logging.warning("skl2onnx not available. Install with: pip install skl2onnx")

# ONNX converter common imports
try:
    import onnxconverter_common
    ONNXCONVERTER_COMMON_AVAILABLE = True
except ImportError:
    ONNXCONVERTER_COMMON_AVAILABLE = False
    logging.warning("onnxconverter-common not available. Install with: pip install onnxconverter-common")

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

# LightGBM imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

# CatBoost imports
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available")

logger = logging.getLogger(__name__)


class ONNXConverter:
    """
    Converts scikit-learn models and trained boosters to ONNX format for optimized inference.
    Supports RandomForest, LogisticRegression, Pipeline, XGBoost, LightGBM, and CatBoost models.
    
    Phase 2: Enhanced with booster conversion and performance measurement
    """
    
    def __init__(self, onnx_dir: str = "models/onnx"):
        self.onnx_dir = Path(onnx_dir)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        
        # Check dependencies
        self.dependencies_available = {
            'skl2onnx': SKL2ONNX_AVAILABLE,
            'onnxconverter_common': ONNXCONVERTER_COMMON_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE
        }
        
        # Log dependency status
        for dep, available in self.dependencies_available.items():
            if available:
                logger.info(f"✅ {dep} available")
            else:
                logger.warning(f"⚠️ {dep} not available")
        
        logger.info(f"ONNXConverter initialized with directory: {self.onnx_dir}")
    
    def convert_model(self, model: Any, model_name: str, 
                     input_shape: Tuple[int, ...] = None,
                     model_type: str = "auto") -> Optional[str]:
        """
        Convert a model to ONNX format.
        
        Args:
            model: Model to convert (scikit-learn, XGBoost, LightGBM, CatBoost)
            model_name: Name for the ONNX model file
            input_shape: Expected input shape (n_samples, n_features)
            model_type: Type of model ("auto", "sklearn", "xgboost", "lightgbm", "catboost")
            
        Returns:
            Path to the saved ONNX model file, or None if conversion failed
        """
        try:
            logger.info(f"Converting model {model_name} to ONNX (type: {model_type})...")
            
            # Auto-detect model type if not specified
            if model_type == "auto":
                model_type = self._detect_model_type(model)
            
            # Convert based on model type
            if model_type == "sklearn":
                return self._convert_sklearn_model(model, model_name, input_shape)
            elif model_type == "xgboost":
                return self._convert_xgboost_model(model, model_name, input_shape)
            elif model_type == "lightgbm":
                return self._convert_lightgbm_model(model, model_name, input_shape)
            elif model_type == "catboost":
                return self._convert_catboost_model(model, model_name, input_shape)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to convert {model_name} to ONNX: {e}")
            return None
    
    def _detect_model_type(self, model: Any) -> str:
        """Auto-detect the type of model"""
        if hasattr(model, 'get_booster'):  # XGBoost
            return "xgboost"
        elif hasattr(model, 'booster_'):  # LightGBM
            return "lightgbm"
        elif hasattr(model, 'get_evals_result'):  # CatBoost
            return "catboost"
        else:  # Assume scikit-learn
            return "sklearn"
    
    def _convert_sklearn_model(self, model: Any, model_name: str, 
                             input_shape: Tuple[int, ...] = None) -> Optional[str]:
        """Convert scikit-learn model to ONNX"""
        if not SKL2ONNX_AVAILABLE:
            logger.warning(f"skl2onnx not available, skipping ONNX conversion for {model_name}")
            return None
            
        try:
            # Determine input shape if not provided
            if input_shape is None:
                input_shape = self._infer_input_shape(model)
            
            # Create ONNX model
            onnx_model = self._create_sklearn_onnx_model(model, input_shape)
            
            # Save ONNX model
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            onnx.save(onnx_model, str(onnx_path))
            
            # Validate ONNX model
            self._validate_onnx_model(onnx_path)
            
            logger.info(f"✅ Successfully converted sklearn model {model_name} to ONNX: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert sklearn model {model_name} to ONNX: {e}")
            return None
    
    def _convert_xgboost_model(self, model: Any, model_name: str, 
                             input_shape: Tuple[int, ...] = None) -> Optional[str]:
        """Convert XGBoost model to ONNX"""
        if not XGBOOST_AVAILABLE:
            logger.warning(f"XGBoost not available, skipping ONNX conversion for {model_name}")
            return None
        
        if not ONNXCONVERTER_COMMON_AVAILABLE:
            logger.warning(f"onnxconverter-common not available, skipping ONNX conversion for {model_name}")
            return None
            
        try:
            # Get the booster from XGBoost model
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
            else:
                booster = model
            
            # Determine input shape if not provided
            if input_shape is None:
                input_shape = self._infer_xgboost_input_shape(booster)
            
            # Convert to ONNX using onnxconverter-common
            onnx_model = self._create_xgboost_onnx_model(booster, input_shape)
            
            # Save ONNX model
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            onnx.save(onnx_model, str(onnx_path))
            
            # Validate ONNX model
            self._validate_onnx_model(onnx_path)
            
            logger.info(f"✅ Successfully converted XGBoost model {model_name} to ONNX: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert XGBoost model {model_name} to ONNX: {e}")
            return None
    
    def _convert_lightgbm_model(self, model: Any, model_name: str, 
                              input_shape: Tuple[int, ...] = None) -> Optional[str]:
        """Convert LightGBM model to ONNX"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning(f"LightGBM not available, skipping ONNX conversion for {model_name}")
            return None
        
        if not ONNXCONVERTER_COMMON_AVAILABLE:
            logger.warning(f"onnxconverter-common not available, skipping ONNX conversion for {model_name}")
            return None
            
        try:
            # Get the booster from LightGBM model
            if hasattr(model, 'booster_'):
                booster = model.booster_
            else:
                booster = model
            
            # Determine input shape if not provided
            if input_shape is None:
                input_shape = self._infer_lightgbm_input_shape(booster)
            
            # Convert to ONNX using onnxconverter-common
            onnx_model = self._create_lightgbm_onnx_model(booster, input_shape)
            
            # Save ONNX model
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            onnx.save(onnx_model, str(onnx_path))
            
            # Validate ONNX model
            self._validate_onnx_model(onnx_path)
            
            logger.info(f"✅ Successfully converted LightGBM model {model_name} to ONNX: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert LightGBM model {model_name} to ONNX: {e}")
            return None
    
    def _convert_catboost_model(self, model: Any, model_name: str, 
                              input_shape: Tuple[int, ...] = None) -> Optional[str]:
        """Convert CatBoost model to ONNX"""
        if not CATBOOST_AVAILABLE:
            logger.warning(f"CatBoost not available, skipping ONNX conversion for {model_name}")
            return None
        
        if not ONNXCONVERTER_COMMON_AVAILABLE:
            logger.warning(f"onnxconverter-common not available, skipping ONNX conversion for {model_name}")
            return None
            
        try:
            # Determine input shape if not provided
            if input_shape is None:
                input_shape = self._infer_catboost_input_shape(model)
            
            # Convert to ONNX using onnxconverter-common
            onnx_model = self._create_catboost_onnx_model(model, input_shape)
            
            # Save ONNX model
            onnx_path = self.onnx_dir / f"{model_name}.onnx"
            onnx.save(onnx_model, str(onnx_path))
            
            # Validate ONNX model
            self._validate_onnx_model(onnx_path)
            
            logger.info(f"✅ Successfully converted CatBoost model {model_name} to ONNX: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert CatBoost model {model_name} to ONNX: {e}")
            return None
    
    def _infer_input_shape(self, model: Any) -> Tuple[int, ...]:
        """Infer the expected input shape from the model"""
        if hasattr(model, 'n_features_in_'):
            return (None, model.n_features_in_)
        elif hasattr(model, 'n_features_'):
            return (None, model.n_features_)
        else:
            # Default shape for unknown models
            return (None, 100)  # Assume 100 features
    
    def _infer_xgboost_input_shape(self, booster: Any) -> Tuple[int, ...]:
        """Infer input shape from XGBoost booster"""
        try:
            # Try to get feature count from booster
            if hasattr(booster, 'num_features'):
                return (None, booster.num_features)
            elif hasattr(booster, 'feature_names'):
                return (None, len(booster.feature_names))
            else:
                return (None, 100)  # Default
        except:
            return (None, 100)  # Default
    
    def _infer_lightgbm_input_shape(self, booster: Any) -> Tuple[int, ...]:
        """Infer input shape from LightGBM booster"""
        try:
            # Try to get feature count from booster
            if hasattr(booster, 'num_feature'):
                return (None, booster.num_feature)
            elif hasattr(booster, 'feature_name'):
                return (None, len(booster.feature_name))
            else:
                return (None, 100)  # Default
        except:
            return (None, 100)  # Default
    
    def _infer_catboost_input_shape(self, model: Any) -> Tuple[int, ...]:
        """Infer input shape from CatBoost model"""
        try:
            # Try to get feature count from model
            if hasattr(model, 'feature_names_'):
                return (None, len(model.feature_names_))
            elif hasattr(model, 'feature_importances_'):
                return (None, len(model.feature_importances_))
            else:
                return (None, 100)  # Default
        except:
            return (None, 100)  # Default
    
    def _create_sklearn_onnx_model(self, model: Any, input_shape: Tuple[int, ...]) -> onnx.ModelProto:
        """Create ONNX model from scikit-learn model"""
        # Define input type
        input_type = FloatTensorType(input_shape)
        
        # Convert model
        onnx_model = convert_sklearn(model, initial_types=[('input', input_type)])
        return onnx_model
    
    def _create_xgboost_onnx_model(self, booster: Any, input_shape: Tuple[int, ...]) -> onnx.ModelProto:
        """Create ONNX model from XGBoost booster"""
        try:
            # Use onnxconverter-common for XGBoost conversion
            from onnxconverter_common import convert_xgboost
            
            # Convert booster to ONNX
            onnx_model = convert_xgboost(booster, initial_types=[('input', FloatTensorType(input_shape))])
            return onnx_model
            
        except ImportError:
            raise ImportError("onnxconverter-common required for XGBoost conversion")
    
    def _create_lightgbm_onnx_model(self, booster: Any, input_shape: Tuple[int, ...]) -> onnx.ModelProto:
        """Create ONNX model from LightGBM booster"""
        try:
            # Use onnxconverter-common for LightGBM conversion
            from onnxconverter_common import convert_lightgbm
            
            # Convert booster to ONNX
            onnx_model = convert_lightgbm(booster, initial_types=[('input', FloatTensorType(input_shape))])
            return onnx_model
            
        except ImportError:
            raise ImportError("onnxconverter-common required for LightGBM conversion")
    
    def _create_catboost_onnx_model(self, model: Any, input_shape: Tuple[int, ...]) -> onnx.ModelProto:
        """Create ONNX model from CatBoost model"""
        try:
            # Use onnxconverter-common for CatBoost conversion
            from onnxconverter_common import convert_catboost
            
            # Convert model to ONNX
            onnx_model = convert_catboost(model, initial_types=[('input', FloatTensorType(input_shape))])
            return onnx_model
            
        except ImportError:
            raise ImportError("onnxconverter-common required for CatBoost conversion")
    
    def _validate_onnx_model(self, onnx_path: Path):
        """Validate ONNX model"""
        try:
            # Load and validate ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"✅ ONNX model validation passed: {onnx_path}")
        except Exception as e:
            logger.error(f"❌ ONNX model validation failed: {e}")
            raise
    
    def measure_latency_improvement(self, model: Any, onnx_path: str, 
                                  test_data: np.ndarray, 
                                  n_runs: int = 100) -> Dict[str, float]:
        """
        Measure latency improvement between native model and ONNX model
        
        Args:
            model: Native model
            onnx_path: Path to ONNX model
            test_data: Test data for inference
            n_runs: Number of runs for averaging
            
        Returns:
            Dictionary with latency measurements
        """
        try:
            logger.info(f"Measuring latency improvement for {onnx_path}")
            
            # Test native model latency
            native_times = []
            for _ in range(n_runs):
                start_time = time.time()
                if hasattr(model, 'predict_proba'):
                    _ = model.predict_proba(test_data)
                else:
                    _ = model.predict(test_data)
                native_times.append(time.time() - start_time)
            
            native_avg = np.mean(native_times) * 1000  # Convert to ms
            native_std = np.std(native_times) * 1000
            
            # Test ONNX model latency
            onnx_times = []
            try:
                # Load ONNX model
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                
                for _ in range(n_runs):
                    start_time = time.time()
                    _ = session.run(None, {input_name: test_data.astype(np.float32)})
                    onnx_times.append(time.time() - start_time)
                
                onnx_avg = np.mean(onnx_times) * 1000  # Convert to ms
                onnx_std = np.std(onnx_times) * 1000
                
                # Calculate improvement
                improvement = ((native_avg - onnx_avg) / native_avg) * 100
                
                results = {
                    'native_avg_ms': native_avg,
                    'native_std_ms': native_std,
                    'onnx_avg_ms': onnx_avg,
                    'onnx_std_ms': onnx_std,
                    'improvement_pct': improvement,
                    'speedup_factor': native_avg / onnx_avg if onnx_avg > 0 else 1.0,
                    'onnx_success': True
                }
                
                logger.info(f"✅ Latency measurement completed:")
                logger.info(f"   - Native: {native_avg:.2f}ms ± {native_std:.2f}ms")
                logger.info(f"   - ONNX: {onnx_avg:.2f}ms ± {onnx_std:.2f}ms")
                logger.info(f"   - Improvement: {improvement:.1f}%")
                logger.info(f"   - Speedup: {results['speedup_factor']:.1f}x")
                
                return results
                
            except Exception as e:
                logger.warning(f"⚠️ ONNX inference failed, falling back to native: {e}")
                return {
                    'native_avg_ms': native_avg,
                    'native_std_ms': native_std,
                    'onnx_avg_ms': float('inf'),
                    'onnx_std_ms': 0.0,
                    'improvement_pct': 0.0,
                    'speedup_factor': 1.0,
                    'onnx_success': False,
                    'fallback_reason': str(e)
                }
                
        except Exception as e:
            logger.error(f"❌ Latency measurement failed: {e}")
            return {
                'error': str(e),
                'onnx_success': False
            }
    
    def get_conversion_status(self) -> Dict[str, Any]:
        """Get status of ONNX conversion capabilities"""
        return {
            'dependencies': self.dependencies_available,
            'onnx_dir': str(self.onnx_dir),
            'available_models': self._get_available_onnx_models()
        }
    
    def _get_available_onnx_models(self) -> List[str]:
        """Get list of available ONNX models"""
        try:
            onnx_files = list(self.onnx_dir.glob("*.onnx"))
            return [f.stem for f in onnx_files]
        except Exception as e:
            logger.error(f"Error getting available ONNX models: {e}")
            return []

# Global ONNX converter instance
onnx_converter = ONNXConverter()

# Export for use in other modules
__all__ = [
    'ONNXConverter',
    'onnx_converter'
]
