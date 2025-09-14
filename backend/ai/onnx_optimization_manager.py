"""
ONNX Optimization Manager for AlphaPulse
Unified system for managing ONNX model optimization with mixed precision and quantization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
from datetime import datetime
import gc

logger = logging.getLogger(__name__)


class ONNXOptimizationManager:
    """
    Unified manager for ONNX model optimization including mixed precision and quantization.
    Integrates with existing ONNX inference engine, mixed precision engine, and quantization system.
    """
    
    def __init__(self, 
                 enable_mixed_precision: bool = True,
                 enable_quantization: bool = True,
                 auto_optimize: bool = True):
        """
        Initialize ONNX optimization manager.
        
        Args:
            enable_mixed_precision: Enable FP16 mixed precision optimization
            enable_quantization: Enable INT8 quantization optimization
            auto_optimize: Automatically optimize models when loaded
        """
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_quantization = enable_quantization
        self.auto_optimize = auto_optimize
        
        # Import existing systems
        try:
            from .onnx_inference import onnx_inference_engine
            self.inference_engine = onnx_inference_engine
        except ImportError:
            self.inference_engine = None
            logger.warning("ONNX inference engine not available")
        
        try:
            from .mixed_precision_engine import mixed_precision_engine
            self.mixed_precision_engine = mixed_precision_engine
        except ImportError:
            self.mixed_precision_engine = None
            logger.warning("Mixed precision engine not available")
        
        try:
            from .model_quantization import model_quantization_system
            self.quantization_system = model_quantization_system
        except ImportError:
            self.quantization_system = None
            logger.warning("Model quantization system not available")
        
        # Optimization tracking
        self.optimization_stats: Dict[str, Dict] = {}
        self.optimized_models: Dict[str, Dict] = {}
        self.performance_comparisons: Dict[str, Dict] = {}
        
        logger.info(f"ONNXOptimizationManager initialized")
        logger.info(f"Mixed precision: {self.enable_mixed_precision}")
        logger.info(f"Quantization: {self.enable_quantization}")
        logger.info(f"Auto optimize: {self.auto_optimize}")
    
    def optimize_model(self, model_name: str, model: Any = None,
                      test_data: np.ndarray = None,
                      optimization_level: str = "balanced") -> Dict[str, Any]:
        """
        Optimize a model with mixed precision and quantization.
        
        Args:
            model_name: Name of the model to optimize
            model: Model object (optional, will try to load from registry)
            test_data: Test data for performance comparison
            optimization_level: "speed", "balanced", or "accuracy"
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(f"🔄 Optimizing model: {model_name}")
            start_time = time.time()
            
            optimization_results = {
                'model_name': model_name,
                'optimization_level': optimization_level,
                'optimizations_applied': [],
                'performance_improvements': {},
                'errors': []
            }
            
            # Load model if not provided
            if model is None:
                model = self._load_model_from_registry(model_name)
                if model is None:
                    optimization_results['errors'].append("Could not load model from registry")
                    return optimization_results
            
            # Apply mixed precision optimization
            if self.enable_mixed_precision and self.mixed_precision_engine:
                fp16_results = self._apply_mixed_precision_optimization(model_name, model, test_data)
                if fp16_results:
                    optimization_results['optimizations_applied'].append('mixed_precision')
                    optimization_results['performance_improvements']['fp16'] = fp16_results
            
            # Apply quantization optimization
            if self.enable_quantization and self.quantization_system:
                int8_results = self._apply_quantization_optimization(model_name, model, test_data)
                if int8_results:
                    optimization_results['optimizations_applied'].append('quantization')
                    optimization_results['performance_improvements']['int8'] = int8_results
            
            # Load optimized models into inference engine
            self._load_optimized_models(model_name)
            
            # Store optimization results
            optimization_time = time.time() - start_time
            optimization_results['optimization_time'] = optimization_time
            optimization_results['total_optimizations'] = len(optimization_results['optimizations_applied'])
            
            self.optimization_stats[model_name] = optimization_results
            self.optimized_models[model_name] = {
                'optimized_at': datetime.now(),
                'optimizations': optimization_results['optimizations_applied']
            }
            
            logger.info(f"✅ Model {model_name} optimized in {optimization_time:.2f}s")
            logger.info(f"   Applied optimizations: {optimization_results['optimizations_applied']}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Error optimizing model {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'optimizations_applied': []
            }
    
    def _apply_mixed_precision_optimization(self, model_name: str, model: Any, 
                                          test_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Apply mixed precision optimization"""
        try:
            logger.info(f"   Applying mixed precision optimization to {model_name}")
            
            # Convert model to ONNX if needed
            onnx_path = self._ensure_onnx_model(model_name, model)
            if not onnx_path:
                return None
            
            # Load with mixed precision
            session = self.mixed_precision_engine.load_model_with_mixed_precision(onnx_path, model_name)
            if not session:
                return None
            
            # Test performance if test data available
            if test_data is not None:
                performance = self._test_mixed_precision_performance(session, test_data)
                return performance
            
            return {'status': 'applied', 'session_loaded': True}
            
        except Exception as e:
            logger.error(f"   Mixed precision optimization failed for {model_name}: {e}")
            return None
    
    def _apply_quantization_optimization(self, model_name: str, model: Any,
                                       test_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Apply quantization optimization"""
        try:
            logger.info(f"   Applying quantization optimization to {model_name}")
            
            # Quantize model
            quantized_path = self.quantization_system.quantize_model(model, model_name)
            if not quantized_path:
                return None
            
            # Test performance if test data available
            if test_data is not None:
                performance = self._test_quantization_performance(model, quantized_path, test_data)
                return performance
            
            return {'status': 'applied', 'quantized_path': quantized_path}
            
        except Exception as e:
            logger.error(f"   Quantization optimization failed for {model_name}: {e}")
            return None
    
    def _ensure_onnx_model(self, model_name: str, model: Any) -> Optional[str]:
        """Ensure model is available in ONNX format"""
        try:
            # Check if ONNX model already exists
            onnx_path = Path(f"models/onnx/{model_name}.onnx")
            if onnx_path.exists():
                return str(onnx_path)
            
            # Convert to ONNX if needed
            if hasattr(self, 'onnx_converter'):
                from .onnx_converter import onnx_converter
                converted_path = onnx_converter.convert_model(model, model_name)
                return converted_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error ensuring ONNX model for {model_name}: {e}")
            return None
    
    def _load_model_from_registry(self, model_name: str) -> Optional[Any]:
        """Load model from model registry"""
        try:
            from .model_registry import model_registry
            if model_name in model_registry.models:
                return model_registry.models[model_name]
            elif model_name in model_registry.ensembles:
                return model_registry.ensembles[model_name]
            else:
                logger.warning(f"Model {model_name} not found in registry")
                return None
        except ImportError:
            logger.warning("Model registry not available")
            return None
    
    def _load_optimized_models(self, model_name: str):
        """Load optimized models into inference engine"""
        if not self.inference_engine:
            return
        
        try:
            # Load standard model
            self.inference_engine.load_model(model_name, use_quantized=False)
            
            # Load quantized model if available
            if self.enable_quantization:
                self.inference_engine.load_model(model_name, use_quantized=True)
                
        except Exception as e:
            logger.error(f"Error loading optimized models for {model_name}: {e}")
    
    def _test_mixed_precision_performance(self, session: Any, test_data: np.ndarray) -> Dict[str, Any]:
        """Test mixed precision performance"""
        try:
            # Test FP32 performance
            fp32_start = time.time()
            fp32_predictions = self.mixed_precision_engine.predict_with_mixed_precision(
                session, test_data.astype(np.float32)
            )
            fp32_time = (time.time() - fp32_start) * 1000
            
            # Test FP16 performance
            fp16_start = time.time()
            fp16_predictions = self.mixed_precision_engine.predict_with_mixed_precision(
                session, test_data.astype(np.float16)
            )
            fp16_time = (time.time() - fp16_start) * 1000
            
            # Calculate speedup
            speedup = fp32_time / fp16_time if fp16_time > 0 else 1.0
            
            return {
                'fp32_time_ms': fp32_time,
                'fp16_time_ms': fp16_time,
                'speedup_factor': speedup,
                'memory_savings': 0.5,  # FP16 uses half the memory
                'test_samples': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Error testing mixed precision performance: {e}")
            return {'error': str(e)}
    
    def _test_quantization_performance(self, original_model: Any, quantized_path: str,
                                     test_data: np.ndarray) -> Dict[str, Any]:
        """Test quantization performance"""
        try:
            # Test original model performance
            original_start = time.time()
            original_predictions = original_model.predict(test_data)
            original_time = (time.time() - original_start) * 1000
            
            # Test quantized model performance
            quantized_start = time.time()
            quantized_predictions = self.quantization_system._predict_with_quantized_model(
                quantized_path, test_data
            )
            quantized_time = (time.time() - quantized_start) * 1000
            
            # Calculate speedup
            speedup = original_time / quantized_time if quantized_time > 0 else 1.0
            
            return {
                'original_time_ms': original_time,
                'quantized_time_ms': quantized_time,
                'speedup_factor': speedup,
                'test_samples': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Error testing quantization performance: {e}")
            return {'error': str(e)}
    
    def predict_optimized(self, model_name: str, input_data: np.ndarray,
                         optimization_preference: str = "auto") -> Optional[np.ndarray]:
        """
        Make predictions using optimized models.
        
        Args:
            model_name: Name of the model
            input_data: Input data
            optimization_preference: "auto", "speed", "accuracy", "mixed_precision", "quantized"
            
        Returns:
            Model predictions
        """
        if not self.inference_engine:
            logger.error("ONNX inference engine not available")
            return None
        
        try:
            # Determine which optimization to use
            if optimization_preference == "auto":
                # Auto-select based on data size and available optimizations
                if len(input_data) > 1000 and self.enable_quantization:
                    use_quantized = True
                    use_mixed_precision = False
                else:
                    use_quantized = False
                    use_mixed_precision = self.enable_mixed_precision
            elif optimization_preference == "speed":
                use_quantized = self.enable_quantization
                use_mixed_precision = self.enable_mixed_precision
            elif optimization_preference == "accuracy":
                use_quantized = False
                use_mixed_precision = False
            elif optimization_preference == "mixed_precision":
                use_quantized = False
                use_mixed_precision = True
            elif optimization_preference == "quantized":
                use_quantized = True
                use_mixed_precision = False
            else:
                use_quantized = False
                use_mixed_precision = False
            
            # Make prediction with selected optimization
            predictions = self.inference_engine.predict(
                model_name, input_data,
                use_quantized=use_quantized,
                use_mixed_precision=use_mixed_precision
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making optimized prediction for {model_name}: {e}")
            return None
    
    def benchmark_optimizations(self, model_names: List[str], 
                              test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark all optimization methods for given models.
        
        Args:
            model_names: List of model names to benchmark
            test_data: Dictionary mapping model names to test data
            
        Returns:
            Benchmark results
        """
        benchmark_results = {
            'total_models': len(model_names),
            'benchmark_time': datetime.now(),
            'model_results': {},
            'summary': {
                'average_fp16_speedup': 0.0,
                'average_int8_speedup': 0.0,
                'best_optimization': 'none'
            }
        }
        
        try:
            logger.info(f"🔄 Starting optimization benchmark for {len(model_names)} models")
            
            fp16_speedups = []
            int8_speedups = []
            
            for model_name in model_names:
                if model_name not in test_data:
                    continue
                
                logger.info(f"Benchmarking {model_name}...")
                
                # Optimize model
                optimization_results = self.optimize_model(
                    model_name, test_data=test_data[model_name]
                )
                
                # Store results
                benchmark_results['model_results'][model_name] = optimization_results
                
                # Collect speedup metrics
                if 'performance_improvements' in optimization_results:
                    improvements = optimization_results['performance_improvements']
                    
                    if 'fp16' in improvements and 'speedup_factor' in improvements['fp16']:
                        fp16_speedups.append(improvements['fp16']['speedup_factor'])
                    
                    if 'int8' in improvements and 'speedup_factor' in improvements['int8']:
                        int8_speedups.append(improvements['int8']['speedup_factor'])
            
            # Calculate summary statistics
            if fp16_speedups:
                benchmark_results['summary']['average_fp16_speedup'] = np.mean(fp16_speedups)
            
            if int8_speedups:
                benchmark_results['summary']['average_int8_speedup'] = np.mean(int8_speedups)
            
            # Determine best optimization
            fp16_avg = benchmark_results['summary']['average_fp16_speedup']
            int8_avg = benchmark_results['summary']['average_int8_speedup']
            
            if fp16_avg > int8_avg and fp16_avg > 1.0:
                benchmark_results['summary']['best_optimization'] = 'mixed_precision'
            elif int8_avg > fp16_avg and int8_avg > 1.0:
                benchmark_results['summary']['best_optimization'] = 'quantization'
            else:
                benchmark_results['summary']['best_optimization'] = 'none'
            
            logger.info(f"✅ Benchmark completed")
            logger.info(f"   Average FP16 speedup: {fp16_avg:.2f}x")
            logger.info(f"   Average INT8 speedup: {int8_avg:.2f}x")
            logger.info(f"   Best optimization: {benchmark_results['summary']['best_optimization']}")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return {'error': str(e)}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        status = {
            'enabled_features': {
                'mixed_precision': self.enable_mixed_precision,
                'quantization': self.enable_quantization,
                'auto_optimize': self.auto_optimize
            },
            'available_systems': {
                'inference_engine': self.inference_engine is not None,
                'mixed_precision_engine': self.mixed_precision_engine is not None,
                'quantization_system': self.quantization_system is not None
            },
            'optimized_models': len(self.optimized_models),
            'optimization_stats': self.optimization_stats,
            'performance_summary': self._get_performance_summary()
        }
        
        return status
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all optimizations"""
        summary = {
            'total_optimizations': len(self.optimization_stats),
            'average_optimization_time': 0.0,
            'optimization_success_rate': 0.0,
            'best_speedup': 0.0
        }
        
        if self.optimization_stats:
            optimization_times = []
            successful_optimizations = 0
            speedups = []
            
            for model_name, stats in self.optimization_stats.items():
                if 'optimization_time' in stats:
                    optimization_times.append(stats['optimization_time'])
                
                if 'optimizations_applied' in stats and stats['optimizations_applied']:
                    successful_optimizations += 1
                
                if 'performance_improvements' in stats:
                    improvements = stats['performance_improvements']
                    for opt_type, perf in improvements.items():
                        if 'speedup_factor' in perf:
                            speedups.append(perf['speedup_factor'])
            
            if optimization_times:
                summary['average_optimization_time'] = np.mean(optimization_times)
            
            summary['optimization_success_rate'] = successful_optimizations / len(self.optimization_stats)
            
            if speedups:
                summary['best_speedup'] = np.max(speedups)
        
        return summary
    
    def cleanup_optimization_cache(self):
        """Clean up optimization cache and temporary files"""
        try:
            # Clear optimization tracking
            self.optimization_stats.clear()
            self.performance_comparisons.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("🧹 Optimization cache cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up optimization cache: {e}")


# Global ONNX optimization manager instance
onnx_optimization_manager = ONNXOptimizationManager()
