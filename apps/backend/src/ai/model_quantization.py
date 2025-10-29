"""
Model Quantization System for AlphaPulse
INT8 quantization for further inference speedup
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import pickle
import time
from datetime import datetime
import gc

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# ONNX imports
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available for quantization")

logger = logging.getLogger(__name__)


class ModelQuantizationSystem:
    """
    Model quantization system for INT8 optimization.
    Provides quantization-aware training and post-training quantization.
    """
    
    def __init__(self, quantization_dir: str = "models/quantized",
                 enable_int8: bool = True,
                 enable_dynamic_quantization: bool = True):
        """
        Initialize model quantization system.
        
        Args:
            quantization_dir: Directory for quantized models
            enable_int8: Enable INT8 quantization
            enable_dynamic_quantization: Enable dynamic quantization
        """
        self.quantization_dir = Path(quantization_dir)
        self.quantization_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_int8 = enable_int8
        self.enable_dynamic_quantization = enable_dynamic_quantization
        
        # Quantization tracking
        self.quantized_models: Dict[str, str] = {}
        self.quantization_stats: Dict[str, Dict] = {}
        
        # Performance tracking
        self.quantization_times: List[float] = []
        self.accuracy_comparisons: Dict[str, Dict] = {}
        
        logger.info(f"ModelQuantizationSystem initialized with directory: {self.quantization_dir}")
        logger.info(f"INT8 enabled: {self.enable_int8}, Dynamic quantization: {self.enable_dynamic_quantization}")
    
    def quantize_model(self, model: Any, model_name: str,
                      calibration_data: np.ndarray = None,
                      quantization_type: str = "dynamic") -> Optional[str]:
        """
        Quantize a model to INT8.
        
        Args:
            model: Model to quantize
            model_name: Name for the quantized model
            calibration_data: Data for calibration (optional)
            quantization_type: Type of quantization ("dynamic" or "static")
            
        Returns:
            Path to quantized model file, or None if quantization failed
        """
        if not self.enable_int8:
            logger.warning("INT8 quantization is disabled")
            return None
        
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available for quantization")
            return None
        
        try:
            logger.info(f"Quantizing model {model_name} to INT8...")
            start_time = time.time()
            
            # First convert to ONNX if not already
            onnx_path = self._convert_to_onnx_if_needed(model, model_name)
            if not onnx_path:
                logger.error(f"Failed to convert {model_name} to ONNX for quantization")
                return None
            
            # Perform quantization
            quantized_path = self._perform_quantization(onnx_path, model_name, quantization_type)
            
            if quantized_path:
                quantization_time = time.time() - start_time
                self.quantization_times.append(quantization_time)
                
                # Store quantization info
                self.quantized_models[model_name] = quantized_path
                self.quantization_stats[model_name] = {
                    'quantization_type': quantization_type,
                    'quantization_time': quantization_time,
                    'original_size_mb': self._get_file_size_mb(onnx_path),
                    'quantized_size_mb': self._get_file_size_mb(quantized_path),
                    'compression_ratio': self._calculate_compression_ratio(onnx_path, quantized_path)
                }
                
                logger.info(f"✅ Successfully quantized {model_name} to INT8: {quantized_path}")
                logger.info(f"   Compression ratio: {self.quantization_stats[model_name]['compression_ratio']:.2f}x")
                
                return quantized_path
            else:
                logger.error(f"Failed to quantize {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error quantizing {model_name}: {e}")
            return None
    
    def _convert_to_onnx_if_needed(self, model: Any, model_name: str) -> Optional[str]:
        """Convert model to ONNX if needed for quantization"""
        # Check if ONNX model already exists
        onnx_path = self.quantization_dir / f"{model_name}_original.onnx"
        
        if onnx_path.exists():
            return str(onnx_path)
        
        # Try to convert using existing ONNX converter
        try:
            from .onnx_converter import onnx_converter
            converted_path = onnx_converter.convert_model(model, f"{model_name}_original")
            return converted_path
        except Exception as e:
            logger.warning(f"Could not convert {model_name} to ONNX: {e}")
            return None
    
    def _perform_quantization(self, onnx_path: str, model_name: str,
                            quantization_type: str) -> Optional[str]:
        """Perform the actual quantization"""
        try:
            quantized_path = self.quantization_dir / f"{model_name}_int8.onnx"
            
            if quantization_type == "dynamic":
                # Dynamic quantization (no calibration data needed)
                quantize_dynamic(
                    model_input=onnx_path,
                    model_output=str(quantized_path),
                    weight_type=QuantType.QInt8,
                    optimize_model=True
                )
            else:
                # Static quantization (requires calibration data)
                logger.warning("Static quantization not implemented yet")
                return None
            
            return str(quantized_path)
            
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            return None
    
    def quantize_model_registry(self, model_registry: Any) -> Dict[str, str]:
        """
        Quantize all models in a model registry.
        
        Args:
            model_registry: ModelRegistry instance
            
        Returns:
            Dictionary mapping model names to quantized file paths
        """
        quantized_models = {}
        
        try:
            logger.info("🔄 Quantizing all models in registry...")
            
            # Quantize pattern models
            for model_name, model in model_registry.models.items():
                if 'pattern_model' in model_name or 'regime_model' in model_name:
                    quantized_path = self.quantize_model(model, model_name)
                    if quantized_path:
                        quantized_models[model_name] = quantized_path
            
            # Quantize ensemble models
            for ensemble_name, ensemble in model_registry.ensembles.items():
                quantized_path = self.quantize_model(ensemble, ensemble_name)
                if quantized_path:
                    quantized_models[ensemble_name] = quantized_path
            
            logger.info(f"✅ Quantized {len(quantized_models)} models to INT8")
            return quantized_models
            
        except Exception as e:
            logger.error(f"❌ Failed to quantize model registry: {e}")
            return {}
    
    def compare_accuracy(self, original_model: Any, quantized_model_path: str,
                        test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Compare accuracy between original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model_path: Path to quantized model
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            
        Returns:
            Dictionary with accuracy comparison results
        """
        try:
            # Test original model
            original_start = time.time()
            original_predictions = original_model.predict(test_data)
            original_time = time.time() - original_start
            
            # Test quantized model
            if not ONNX_AVAILABLE:
                return {'error': 'ONNX not available for quantized model testing'}
            
            quantized_start = time.time()
            quantized_predictions = self._predict_with_quantized_model(quantized_model_path, test_data)
            quantized_time = time.time() - quantized_start
            
            if quantized_predictions is None:
                return {'error': 'Failed to get predictions from quantized model'}
            
            # Calculate accuracies
            from sklearn.metrics import accuracy_score
            original_accuracy = accuracy_score(test_labels, original_predictions)
            quantized_accuracy = accuracy_score(test_labels, quantized_predictions)
            
            # Calculate speedup
            speedup = original_time / quantized_time if quantized_time > 0 else 0
            
            comparison = {
                'original_accuracy': original_accuracy,
                'quantized_accuracy': quantized_accuracy,
                'accuracy_difference': quantized_accuracy - original_accuracy,
                'original_inference_time': original_time,
                'quantized_inference_time': quantized_time,
                'speedup_factor': speedup,
                'test_samples': len(test_data)
            }
            
            logger.info(f"Accuracy comparison: Original={original_accuracy:.4f}, "
                       f"Quantized={quantized_accuracy:.4f}, Speedup={speedup:.2f}x")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing accuracy: {e}")
            return {'error': str(e)}
    
    def _predict_with_quantized_model(self, model_path: str, test_data: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using quantized model"""
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Ensure input is float32
            if test_data.dtype != np.float32:
                test_data = test_data.astype(np.float32)
            
            # Run inference
            predictions = session.run(None, {input_name: test_data})
            
            # Return predictions
            if isinstance(predictions, list) and len(predictions) > 0:
                return predictions[0]
            else:
                return predictions
                
        except Exception as e:
            logger.error(f"Error predicting with quantized model: {e}")
            return None
    
    def benchmark_quantization(self, model_registry: Any, 
                             test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark quantization performance across all models.
        
        Args:
            model_registry: ModelRegistry instance
            test_data: Dictionary mapping model names to test data
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            'total_models': 0,
            'successfully_quantized': 0,
            'average_compression_ratio': 0.0,
            'average_speedup': 0.0,
            'model_results': {}
        }
        
        try:
            logger.info("🔄 Starting quantization benchmark...")
            
            total_compression_ratios = []
            total_speedups = []
            
            # Benchmark each model
            for model_name, model in model_registry.models.items():
                if model_name not in test_data:
                    continue
                
                logger.info(f"Benchmarking {model_name}...")
                
                # Quantize model
                quantized_path = self.quantize_model(model, model_name)
                
                if quantized_path:
                    benchmark_results['successfully_quantized'] += 1
                    
                    # Get compression ratio
                    compression_ratio = self.quantization_stats[model_name]['compression_ratio']
                    total_compression_ratios.append(compression_ratio)
                    
                    # Test speedup if test data available
                    test_data_for_model = test_data[model_name]
                    if len(test_data_for_model) > 0:
                        # Generate dummy labels for testing
                        dummy_labels = np.random.randint(0, 2, len(test_data_for_model))
                        
                        # Compare accuracy and speed
                        comparison = self.compare_accuracy(model, quantized_path, 
                                                         test_data_for_model, dummy_labels)
                        
                        if 'speedup_factor' in comparison:
                            total_speedups.append(comparison['speedup_factor'])
                        
                        benchmark_results['model_results'][model_name] = {
                            'compression_ratio': compression_ratio,
                            'comparison': comparison
                        }
                
                benchmark_results['total_models'] += 1
            
            # Calculate averages
            if total_compression_ratios:
                benchmark_results['average_compression_ratio'] = np.mean(total_compression_ratios)
            
            if total_speedups:
                benchmark_results['average_speedup'] = np.mean(total_speedups)
            
            logger.info(f"✅ Benchmark completed: {benchmark_results['successfully_quantized']}/"
                       f"{benchmark_results['total_models']} models quantized")
            logger.info(f"   Average compression: {benchmark_results['average_compression_ratio']:.2f}x")
            logger.info(f"   Average speedup: {benchmark_results['average_speedup']:.2f}x")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return {'error': str(e)}
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            return Path(file_path).stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def _calculate_compression_ratio(self, original_path: str, quantized_path: str) -> float:
        """Calculate compression ratio"""
        original_size = self._get_file_size_mb(original_path)
        quantized_size = self._get_file_size_mb(quantized_path)
        
        if quantized_size > 0:
            return original_size / quantized_size
        else:
            return 1.0
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get comprehensive quantization statistics"""
        stats = {
            'total_models_quantized': len(self.quantized_models),
            'quantization_times': {
                'total_quantizations': len(self.quantization_times),
                'average_time': np.mean(self.quantization_times) if self.quantization_times else 0,
                'min_time': np.min(self.quantization_times) if self.quantization_times else 0,
                'max_time': np.max(self.quantization_times) if self.quantization_times else 0
            },
            'compression_stats': {},
            'model_details': {}
        }
        
        # Calculate compression statistics
        compression_ratios = []
        for model_name, model_stats in self.quantization_stats.items():
            compression_ratios.append(model_stats['compression_ratio'])
            stats['model_details'][model_name] = model_stats
        
        if compression_ratios:
            stats['compression_stats'] = {
                'average_compression_ratio': np.mean(compression_ratios),
                'min_compression_ratio': np.min(compression_ratios),
                'max_compression_ratio': np.max(compression_ratios),
                'total_compression_ratios': compression_ratios
            }
        
        return stats
    
    def list_quantized_models(self) -> Dict[str, Any]:
        """List all quantized models"""
        models = {}
        
        for quantized_file in self.quantization_dir.glob("*_int8.onnx"):
            try:
                model_name = quantized_file.stem.replace("_int8", "")
                model_path = str(quantized_file)
                
                models[model_name] = {
                    'path': model_path,
                    'size_mb': self._get_file_size_mb(model_path),
                    'created_at': datetime.fromtimestamp(quantized_file.stat().st_mtime),
                    'quantization_stats': self.quantization_stats.get(model_name, {})
                }
                
            except Exception as e:
                logger.warning(f"Failed to get quantized model info for {quantized_file}: {e}")
        
        return models
    
    def cleanup_quantization_cache(self):
        """Clean up quantization cache and temporary files"""
        try:
            # Remove temporary ONNX files
            for temp_file in self.quantization_dir.glob("*_original.onnx"):
                temp_file.unlink()
            
            # Clear quantization tracking
            self.quantization_times.clear()
            self.accuracy_comparisons.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("🧹 Quantization cache cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up quantization cache: {e}")


# Global model quantization system instance
model_quantization_system = ModelQuantizationSystem()
