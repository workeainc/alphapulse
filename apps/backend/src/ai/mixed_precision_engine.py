"""
Mixed Precision Inference Engine for AlphaPulse
Enables FP16 inference for faster GPU processing with automatic fallback
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import onnxruntime as ort
from datetime import datetime
import time
import gc

logger = logging.getLogger(__name__)


class MixedPrecisionEngine:
    """
    Mixed precision inference engine that automatically switches between FP16 and FP32
    based on hardware capabilities and performance requirements.
    """
    
    def __init__(self, enable_fp16: bool = True, 
                 enable_auto_fallback: bool = True,
                 memory_pool_size: int = 1024 * 1024 * 1024):  # 1GB default
        """
        Initialize mixed precision inference engine.
        
        Args:
            enable_fp16: Enable FP16 inference
            enable_auto_fallback: Automatically fallback to FP32 if FP16 fails
            memory_pool_size: GPU memory pool size in bytes
        """
        self.enable_fp16 = enable_fp16
        self.enable_auto_fallback = enable_auto_fallback
        self.memory_pool_size = memory_pool_size
        
        # Performance tracking
        self.fp16_inference_times: List[float] = []
        self.fp32_inference_times: List[float] = []
        self.fallback_count = 0
        self.total_inferences = 0
        
        # Memory management
        self.gpu_memory_usage = 0
        self.memory_pool = {}
        
        # Initialize providers
        self.providers = self._setup_providers()
        
        logger.info(f"MixedPrecisionEngine initialized with providers: {self.providers}")
        logger.info(f"FP16 enabled: {self.enable_fp16}, Auto fallback: {self.enable_auto_fallback}")
    
    def _setup_providers(self) -> List[Dict[str, Any]]:
        """Setup ONNX Runtime providers with mixed precision support"""
        providers = []
        
        # Try CUDA with FP16 support first
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            cuda_provider_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': self.memory_pool_size,
                'cudnn_conv_use_max_workspace': '1',
                'do_copy_in_default_stream': '1',
            }
            
            # Add FP16 support if enabled
            if self.enable_fp16:
                cuda_provider_options['enable_fp16'] = '1'
                cuda_provider_options['enable_mixed_precision'] = '1'
            
            providers.append(('CUDAExecutionProvider', cuda_provider_options))
            logger.info("✅ CUDA provider with mixed precision support available")
        
        # Add CPU provider as fallback
        providers.append(('CPUExecutionProvider', {}))
        
        return providers
    
    def create_session_options(self, enable_graph_optimization: bool = True,
                              enable_memory_pattern: bool = True) -> ort.SessionOptions:
        """Create optimized session options for mixed precision"""
        session_options = ort.SessionOptions()
        
        # Enable graph optimization
        if enable_graph_optimization:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization
        if enable_memory_pattern:
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
        
        # Set execution mode for better performance
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Enable memory reuse
        session_options.enable_mem_reuse = True
        
        return session_options
    
    def load_model_with_mixed_precision(self, model_path: str, 
                                       model_name: str = None) -> Optional[ort.InferenceSession]:
        """
        Load ONNX model with mixed precision support.
        
        Args:
            model_path: Path to ONNX model file
            model_name: Name for the model (optional)
            
        Returns:
            ONNX Runtime inference session with mixed precision support
        """
        try:
            if model_name is None:
                model_name = Path(model_path).stem
            
            # Create session options
            session_options = self.create_session_options()
            
            # Try to load with mixed precision
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=self.providers
            )
            
            # Check if FP16 is being used
            current_providers = session.get_providers()
            if 'CUDAExecutionProvider' in current_providers:
                logger.info(f"✅ Loaded {model_name} with CUDA provider (FP16: {self.enable_fp16})")
            else:
                logger.info(f"✅ Loaded {model_name} with CPU provider")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name} with mixed precision: {e}")
            return None
    
    def predict_with_mixed_precision(self, session: ort.InferenceSession,
                                   input_data: np.ndarray,
                                   output_names: List[str] = None) -> Optional[np.ndarray]:
        """
        Make prediction using mixed precision inference.
        
        Args:
            session: ONNX Runtime inference session
            input_data: Input data as numpy array
            output_names: Names of output tensors (optional)
            
        Returns:
            Model predictions as numpy array
        """
        if session is None:
            return None
        
        try:
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Determine optimal precision based on data type and hardware
            optimal_dtype = self._determine_optimal_precision(input_data)
            
            # Convert input to optimal precision
            if input_data.dtype != optimal_dtype:
                input_data = input_data.astype(optimal_dtype)
            
            # Time the inference
            start_time = time.time()
            
            # Run inference
            if output_names:
                outputs = session.run(output_names, {input_name: input_data})
            else:
                outputs = session.run(None, {input_name: input_data})
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Track performance based on precision used
            if optimal_dtype == np.float16:
                self.fp16_inference_times.append(inference_time)
            else:
                self.fp32_inference_times.append(inference_time)
            
            self.total_inferences += 1
            
            # Return first output if multiple outputs
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]
            else:
                return outputs
                
        except Exception as e:
            logger.error(f"❌ Mixed precision inference failed: {e}")
            
            # Try fallback to FP32 if enabled
            if self.enable_auto_fallback and self.enable_fp16:
                logger.info("🔄 Attempting fallback to FP32...")
                return self._fallback_to_fp32(session, input_data, output_names)
            
            return None
    
    def _determine_optimal_precision(self, input_data: np.ndarray) -> np.dtype:
        """Determine optimal precision for input data"""
        # Check if FP16 is enabled and supported
        if not self.enable_fp16:
            return np.float32
        
        # Check if CUDA provider is available
        if not any('CUDAExecutionProvider' in provider[0] for provider in self.providers):
            return np.float32
        
        # Use FP16 for inference if data is already float32 or float64
        if input_data.dtype in [np.float32, np.float64]:
            return np.float16
        
        # Keep original precision for other data types
        return input_data.dtype
    
    def _fallback_to_fp32(self, session: ort.InferenceSession,
                          input_data: np.ndarray,
                          output_names: List[str] = None) -> Optional[np.ndarray]:
        """Fallback to FP32 inference"""
        try:
            input_name = session.get_inputs()[0].name
            
            # Convert to FP32
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Time the fallback inference
            start_time = time.time()
            
            if output_names:
                outputs = session.run(output_names, {input_name: input_data})
            else:
                outputs = session.run(None, {input_name: input_data})
            
            inference_time = (time.time() - start_time) * 1000
            
            # Track fallback performance
            self.fp32_inference_times.append(inference_time)
            self.fallback_count += 1
            self.total_inferences += 1
            
            logger.info(f"✅ FP32 fallback successful (time: {inference_time:.2f}ms)")
            
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]
            else:
                return outputs
                
        except Exception as e:
            logger.error(f"❌ FP32 fallback also failed: {e}")
            return None
    
    def predict_batch_mixed_precision(self, session: ort.InferenceSession,
                                    input_batch: np.ndarray,
                                    batch_size: int = 100) -> Optional[np.ndarray]:
        """
        Make batch predictions using mixed precision.
        
        Args:
            session: ONNX Runtime inference session
            input_batch: Batch input data as numpy array
            batch_size: Batch size for processing
            
        Returns:
            Batch predictions as numpy array
        """
        if session is None:
            return None
        
        try:
            input_name = session.get_inputs()[0].name
            
            # Determine optimal precision for batch
            optimal_dtype = self._determine_optimal_precision(input_batch)
            
            # Convert batch to optimal precision
            if input_batch.dtype != optimal_dtype:
                input_batch = input_batch.astype(optimal_dtype)
            
            # Process in batches
            predictions = []
            total_samples = len(input_batch)
            
            for i in range(0, total_samples, batch_size):
                batch_data = input_batch[i:i + batch_size]
                
                # Time the batch inference
                start_time = time.time()
                
                batch_outputs = session.run(None, {input_name: batch_data})
                
                inference_time = (time.time() - start_time) * 1000
                
                # Track performance
                if optimal_dtype == np.float16:
                    self.fp16_inference_times.append(inference_time)
                else:
                    self.fp32_inference_times.append(inference_time)
                
                self.total_inferences += len(batch_data)
                
                # Add batch predictions
                if isinstance(batch_outputs, list) and len(batch_outputs) > 0:
                    predictions.append(batch_outputs[0])
                else:
                    predictions.append(batch_outputs)
            
            # Concatenate all batch predictions
            return np.concatenate(predictions, axis=0)
            
        except Exception as e:
            logger.error(f"❌ Mixed precision batch inference failed: {e}")
            
            # Try fallback to FP32
            if self.enable_auto_fallback and self.enable_fp16:
                logger.info("🔄 Attempting batch fallback to FP32...")
                return self._fallback_batch_to_fp32(session, input_batch, batch_size)
            
            return None
    
    def _fallback_batch_to_fp32(self, session: ort.InferenceSession,
                               input_batch: np.ndarray,
                               batch_size: int) -> Optional[np.ndarray]:
        """Fallback batch processing to FP32"""
        try:
            input_name = session.get_inputs()[0].name
            
            # Convert to FP32
            if input_batch.dtype != np.float32:
                input_batch = input_batch.astype(np.float32)
            
            # Process in batches
            predictions = []
            total_samples = len(input_batch)
            
            for i in range(0, total_samples, batch_size):
                batch_data = input_batch[i:i + batch_size]
                
                start_time = time.time()
                batch_outputs = session.run(None, {input_name: batch_data})
                inference_time = (time.time() - start_time) * 1000
                
                self.fp32_inference_times.append(inference_time)
                self.fallback_count += 1
                self.total_inferences += len(batch_data)
                
                if isinstance(batch_outputs, list) and len(batch_outputs) > 0:
                    predictions.append(batch_outputs[0])
                else:
                    predictions.append(batch_outputs)
            
            return np.concatenate(predictions, axis=0)
            
        except Exception as e:
            logger.error(f"❌ FP32 batch fallback also failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get mixed precision performance statistics"""
        stats = {
            'total_inferences': self.total_inferences,
            'fallback_count': self.fallback_count,
            'fallback_rate': self.fallback_count / self.total_inferences if self.total_inferences > 0 else 0,
            'fp16_performance': {},
            'fp32_performance': {}
        }
        
        # FP16 performance stats
        if self.fp16_inference_times:
            stats['fp16_performance'] = {
                'total_inferences': len(self.fp16_inference_times),
                'avg_inference_time_ms': np.mean(self.fp16_inference_times),
                'min_inference_time_ms': np.min(self.fp16_inference_times),
                'max_inference_time_ms': np.max(self.fp16_inference_times),
                'throughput_inferences_per_second': 1000 / np.mean(self.fp16_inference_times) if np.mean(self.fp16_inference_times) > 0 else 0
            }
        
        # FP32 performance stats
        if self.fp32_inference_times:
            stats['fp32_performance'] = {
                'total_inferences': len(self.fp32_inference_times),
                'avg_inference_time_ms': np.mean(self.fp32_inference_times),
                'min_inference_time_ms': np.min(self.fp32_inference_times),
                'max_inference_time_ms': np.max(self.fp32_inference_times),
                'throughput_inferences_per_second': 1000 / np.mean(self.fp32_inference_times) if np.mean(self.fp32_inference_times) > 0 else 0
            }
        
        # Performance comparison
        if self.fp16_inference_times and self.fp32_inference_times:
            fp16_avg = np.mean(self.fp16_inference_times)
            fp32_avg = np.mean(self.fp32_inference_times)
            if fp16_avg > 0 and fp32_avg > 0:
                stats['speedup_factor'] = fp32_avg / fp16_avg
                stats['memory_savings'] = 0.5  # FP16 uses half the memory of FP32
        
        return stats
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Clear performance tracking arrays if they get too large
        if len(self.fp16_inference_times) > 10000:
            self.fp16_inference_times = self.fp16_inference_times[-1000:]
        
        if len(self.fp32_inference_times) > 10000:
            self.fp32_inference_times = self.fp32_inference_times[-1000:]
        
        # Force garbage collection
        gc.collect()
        
        logger.info("🧹 Memory optimization completed")


# Global mixed precision engine instance
mixed_precision_engine = MixedPrecisionEngine()
