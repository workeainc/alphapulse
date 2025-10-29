"""
ONNX Inference Engine for AlphaPulse
High-performance inference using ONNX Runtime with mixed precision and quantization
Enhanced with standardized interfaces for surgical upgrades
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import onnxruntime as ort
from datetime import datetime
import time
import asyncio
import asyncpg

logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """
    High-performance inference engine using ONNX Runtime.
    Provides optimized inference for ONNX models with batch processing support,
    mixed precision (FP16), and quantization (INT8) capabilities.
    Enhanced with standardized interfaces for surgical upgrades.
    """
    
    def __init__(self, onnx_dir: str = "models/onnx", 
                 execution_mode: str = "sequential",
                 enable_graph_optimization: bool = True,
                 enable_mixed_precision: bool = True,
                 enable_quantization: bool = True,
                 db_pool: Optional[asyncpg.Pool] = None):
        """
        Initialize ONNX inference engine.
        
        Args:
            onnx_dir: Directory containing ONNX models
            execution_mode: "sequential", "parallel", or "parallel_execution"
            enable_graph_optimization: Enable ONNX graph optimizations
            enable_mixed_precision: Enable FP16 inference where supported
            enable_quantization: Enable INT8 quantization inference
            db_pool: Database pool for interface standardization tracking
        """
        self.onnx_dir = Path(onnx_dir)
        self.execution_mode = execution_mode
        self.enable_graph_optimization = enable_graph_optimization
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_quantization = enable_quantization
        self.db_pool = db_pool
        
        # ONNX Runtime sessions
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.quantized_sessions: Dict[str, ort.InferenceSession] = {}
        
        # Performance tracking
        self.inference_times: Dict[str, List[float]] = {}
        self.fp16_inference_times: Dict[str, List[float]] = {}
        self.int8_inference_times: Dict[str, List[float]] = {}
        self.total_inferences = 0
        
        # Initialize providers with enhanced capabilities
        self.providers = self._setup_providers()
        
        # Import quantization system
        try:
            from .model_quantization import model_quantization_system
            self.quantization_system = model_quantization_system
        except ImportError:
            self.quantization_system = None
            logger.warning("Model quantization system not available")
        
        # Register interface for standardization
        self._register_interface()
        
        logger.info(f"ONNXInferenceEngine initialized with providers: {self.providers}")
        logger.info(f"Mixed precision: {self.enable_mixed_precision}, Quantization: {self.enable_quantization}")
    
    def _setup_providers(self) -> List[str]:
        """Setup ONNX Runtime providers for optimal performance with mixed precision"""
        providers = []
        
        # Try CUDA first (if available) with enhanced options
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            logger.info("✅ CUDA provider available")
        
        # Add CPU provider as fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def _create_enhanced_session_options(self) -> ort.SessionOptions:
        """Create enhanced session options with mixed precision and quantization support"""
        session_options = ort.SessionOptions()
        
        # Enable graph optimization
        if self.enable_graph_optimization:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution mode
        if self.execution_mode == "parallel":
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        elif self.execution_mode == "parallel_execution":
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL_EXECUTION
        
        # Enable memory optimizations
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True
        
        # Enable mixed precision if supported
        if self.enable_mixed_precision:
            # These options help with mixed precision performance
            session_options.enable_profiling = False  # Disable profiling for production
            session_options.log_severity_level = 2  # Reduce logging overhead
        
        return session_options

    def _register_interface(self):
        """Register this component's interface for standardization"""
        if self.db_pool:
            try:
                asyncio.create_task(self._register_interface_async())
            except Exception as e:
                logger.warning(f"Could not register interface: {e}")
    
    async def _register_interface_async(self):
        """Register interface asynchronously"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO component_interface_registry 
                    (component_name, interface_type, interface_version, method_name, input_signature, output_signature) 
                    VALUES 
                    ('ONNXInferenceEngine', 'onnx', '1.0', 'load', 
                     '{"model_name": "string"}', 
                     '{"session": "InferenceSession", "success": "boolean"}'),
                    ('ONNXInferenceEngine', 'onnx', '1.0', 'predict', 
                     '{"session": "InferenceSession", "features": "ndarray"}', 
                     '{"prediction": "float", "confidence": "float"}')
                    ON CONFLICT (component_name, interface_type, method_name) DO NOTHING;
                """)
        except Exception as e:
            logger.warning(f"Interface registration failed: {e}")
    
    async def load(self, model_name: str) -> Optional[ort.InferenceSession]:
        """
        Standardized load method for ONNX models.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            ONNX Runtime InferenceSession or None if failed
        """
        start_time = time.time()
        
        try:
            # Check if model is already loaded
            if model_name in self.sessions:
                return self.sessions[model_name]
            
            # Load model file
            model_path = self.onnx_dir / f"{model_name}.onnx"
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Create session options
            session_options = self._create_enhanced_session_options()
            
            # Load session
            session = ort.InferenceSession(
                str(model_path), 
                session_options, 
                providers=self.providers
            )
            
            # Store session
            self.sessions[model_name] = session
            
            # Track performance
            load_time = (time.time() - start_time) * 1000
            await self._track_interface_performance('ONNXInferenceEngine', 'onnx', 'load', load_time, True)
            
            logger.info(f"✅ Loaded ONNX model: {model_name} in {load_time:.2f}ms")
            return session
            
        except Exception as e:
            load_time = (time.time() - start_time) * 1000
            await self._track_interface_performance('ONNXInferenceEngine', 'onnx', 'load', load_time, False)
            logger.error(f"❌ Failed to load ONNX model {model_name}: {e}")
            return None
    
    def _find_quantized_model(self, model_name: str) -> Optional[Path]:
        """Find quantized model file"""
        if not self.quantization_system:
            return None
        
        quantized_dir = Path(self.quantization_system.quantization_dir)
        quantized_path = quantized_dir / f"{model_name}_int8.onnx"
        
        if quantized_path.exists():
            return quantized_path
        
        return None
    
    def load_all_models(self, model_paths: Dict[str, str], 
                       use_quantized: bool = False) -> Dict[str, bool]:
        """
        Load multiple ONNX models with enhanced capabilities.
        
        Args:
            model_paths: Dictionary mapping model names to file paths
            use_quantized: Whether to load quantized versions if available
            
        Returns:
            Dictionary mapping model names to load success status
        """
        results = {}
        
        for model_name, model_path in model_paths.items():
            results[model_name] = self.load_model(model_name, model_path, use_quantized)
        
        loaded_count = sum(results.values())
        model_type = "quantized" if use_quantized else "standard"
        logger.info(f"✅ Loaded {loaded_count}/{len(model_paths)} {model_type} ONNX models")
        
        return results
    
    async def predict(self, model_name: str, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict using model name (convenience method).
        
        Args:
            model_name: Name of the loaded model
            features: Input features as numpy array
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get session for model
            session = None
            if model_name in self.sessions:
                session = self.sessions[model_name]
            elif model_name in self.quantized_sessions:
                session = self.quantized_sessions[model_name]
            else:
                logger.warning(f"Model {model_name} not loaded, returning default prediction")
                return {'probability': 0.5, 'confidence': 0.5, 'model_loaded': False}
            
            # Run prediction
            probability = await self._predict_with_session(session, features)
            
            return {
                'probability': probability,
                'confidence': probability,
                'model_loaded': True,
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for model {model_name}: {e}")
            return {'probability': 0.5, 'confidence': 0.5, 'model_loaded': False, 'error': str(e)}
    
    async def _predict_with_session(self, session: ort.InferenceSession, features: np.ndarray) -> float:
        """
        Standardized predict method for ONNX models.
        
        Args:
            session: ONNX Runtime InferenceSession
            features: Input features as numpy array
            
        Returns:
            Calibrated probability (0-1)
        """
        start_time = time.time()
        
        try:
            # Ensure features are in correct format
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            outputs = session.run(None, {input_name: features.astype(np.float32)})
            
            # Extract prediction (assuming binary classification)
            prediction = float(outputs[0][0][1] if outputs[0].shape[1] > 1 else outputs[0][0][0])
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000
            await self._track_interface_performance('ONNXInferenceEngine', 'onnx', 'predict', inference_time, True)
            
            # Track inference time
            model_name = getattr(session, '_model_name', 'unknown')
            if model_name not in self.inference_times:
                self.inference_times[model_name] = []
            self.inference_times[model_name].append(inference_time)
            self.total_inferences += 1
            
            return prediction
            
        except Exception as e:
            inference_time = (time.time() - start_time) * 1000
            await self._track_interface_performance('ONNXInferenceEngine', 'onnx', 'predict', inference_time, False)
            logger.error(f"❌ ONNX prediction failed: {e}")
            return 0.5  # Default neutral prediction
    
    async def _track_interface_performance(self, component_name: str, interface_type: str, 
                                         method_name: str, execution_time_ms: float, success: bool):
        """Track interface performance for standardization"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO interface_performance_metrics 
                    (component_name, interface_type, method_name, execution_time_ms, success_rate, error_count) 
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, component_name, interface_type, method_name, execution_time_ms, 
                    1.0 if success else 0.0, 0 if success else 1)
        except Exception as e:
            logger.warning(f"Performance tracking failed: {e}")
    
    async def store_interface_result(self, signal_id: str, component_name: str, 
                                   interface_type: str, input_data: Dict, output_data: Dict,
                                   confidence_score: float, processing_time_ms: float):
        """Store standardized interface result"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO standardized_interface_results 
                    (signal_id, component_name, interface_type, input_data, output_data, 
                     confidence_score, processing_time_ms) 
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, signal_id, component_name, interface_type, input_data, output_data,
                     confidence_score, processing_time_ms)
        except Exception as e:
            logger.warning(f"Interface result storage failed: {e}")

    def predict_batch(self, model_name: str, input_batch: np.ndarray,
                     batch_size: int = 100,
                     use_quantized: bool = False,
                     use_mixed_precision: bool = True) -> Optional[np.ndarray]:
        """
        Make batch predictions using ONNX model with enhanced precision options.
        
        Args:
            model_name: Name of the loaded model
            input_batch: Batch input data as numpy array
            batch_size: Batch size for processing
            use_quantized: Whether to use quantized model if available
            use_mixed_precision: Whether to use mixed precision inference
            
        Returns:
            Batch predictions as numpy array
        """
        # Determine which session to use
        if use_quantized and model_name in self.quantized_sessions:
            session = self.quantized_sessions[model_name]
            time_tracker = self.int8_inference_times[model_name]
            precision_type = "INT8"
        elif model_name in self.sessions:
            session = self.sessions[model_name]
            time_tracker = self.fp16_inference_times[model_name] if use_mixed_precision else self.inference_times[model_name]
            precision_type = "FP16" if use_mixed_precision else "FP32"
        else:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        try:
            input_name = session.get_inputs()[0].name
            
            # Determine optimal precision for batch data
            if use_mixed_precision and not use_quantized:
                optimal_dtype = self._determine_optimal_precision(input_batch)
                if input_batch.dtype != optimal_dtype:
                    input_batch = input_batch.astype(optimal_dtype)
            else:
                # For quantized models or when mixed precision is disabled, use float32
                if input_batch.dtype != np.float32:
                    input_batch = input_batch.astype(np.float32)
            
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
                time_tracker.append(inference_time)
                self.total_inferences += len(batch_data)
                
                # Add batch predictions
                if isinstance(batch_outputs, list) and len(batch_outputs) > 0:
                    predictions.append(batch_outputs[0])
                else:
                    predictions.append(batch_outputs)
            
            # Concatenate all batch predictions
            return np.concatenate(predictions, axis=0)
            
        except Exception as e:
            logger.error(f"❌ {precision_type} batch inference failed for {model_name}: {e}")
            
            # Try fallback to standard precision if mixed precision failed
            if use_mixed_precision and not use_quantized and model_name in self.sessions:
                logger.info(f"🔄 Attempting batch fallback to FP32 for {model_name}")
                return self.predict_batch(model_name, input_batch, batch_size, use_quantized=False, use_mixed_precision=False)
            
            return None

    async def predict_proba(self, model_name: str, input_data: np.ndarray,
                     use_quantized: bool = False,
                     use_mixed_precision: bool = True) -> Optional[np.ndarray]:
        """
        Get probability predictions (for classification models) with enhanced precision.
        
        Args:
            model_name: Name of the loaded model
            input_data: Input data as numpy array
            use_quantized: Whether to use quantized model if available
            use_mixed_precision: Whether to use mixed precision inference
            
        Returns:
            Probability predictions as numpy array
        """
        try:
            # Get session for model
            session = None
            if use_quantized and model_name in self.quantized_sessions:
                session = self.quantized_sessions[model_name]
            elif model_name in self.sessions:
                session = self.sessions[model_name]
            else:
                logger.warning(f"Model {model_name} not loaded")
                return None
            
            # Run prediction
            probability = await self._predict_with_session(session, input_data)
            
            # Return as array for consistency
            return np.array([probability])
            
        except Exception as e:
            logger.error(f"❌ Probability prediction failed for {model_name}: {e}")
            return None

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model with enhanced details"""
        session = None
        time_tracker = None
        precision_type = "Unknown"
        
        # Check quantized sessions first
        if model_name in self.quantized_sessions:
            session = self.quantized_sessions[model_name]
            time_tracker = self.int8_inference_times[model_name]
            precision_type = "INT8"
        elif model_name in self.sessions:
            session = self.sessions[model_name]
            time_tracker = self.inference_times[model_name]
            precision_type = "FP32"
        
        if session is None:
            return None
        
        try:
            info = {
                'model_name': model_name,
                'precision_type': precision_type,
                'input_shape': session.get_inputs()[0].shape,
                'output_shape': session.get_outputs()[0].shape,
                'input_name': session.get_inputs()[0].name,
                'output_name': session.get_outputs()[0].name,
                'providers': session.get_providers(),
                'total_inferences': len(time_tracker) if time_tracker else 0,
                'avg_inference_time_ms': np.mean(time_tracker) if time_tracker and len(time_tracker) > 0 else 0
            }
            
            # Add mixed precision stats if available
            if model_name in self.fp16_inference_times:
                fp16_times = self.fp16_inference_times[model_name]
                if fp16_times:
                    info['fp16_avg_inference_time_ms'] = np.mean(fp16_times)
                    info['fp16_total_inferences'] = len(fp16_times)
            
            # Add quantization stats if available
            if model_name in self.int8_inference_times:
                int8_times = self.int8_inference_times[model_name]
                if int8_times:
                    info['int8_avg_inference_time_ms'] = np.mean(int8_times)
                    info['int8_total_inferences'] = len(int8_times)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics for all models"""
        stats = {
            'total_models_loaded': len(self.sessions) + len(self.quantized_sessions),
            'standard_models': len(self.sessions),
            'quantized_models': len(self.quantized_sessions),
            'total_inferences': self.total_inferences,
            'model_performance': {},
            'precision_performance': {
                'fp32': {},
                'fp16': {},
                'int8': {}
            }
        }
        
        # Standard model performance
        for model_name in self.sessions.keys():
            times = self.inference_times.get(model_name, [])
            if times:
                stats['model_performance'][model_name] = {
                    'precision': 'FP32',
                    'total_inferences': len(times),
                    'avg_inference_time_ms': np.mean(times),
                    'min_inference_time_ms': np.min(times),
                    'max_inference_time_ms': np.max(times),
                    'throughput_inferences_per_second': 1000 / np.mean(times) if np.mean(times) > 0 else 0
                }
        
        # FP16 performance
        for model_name in self.fp16_inference_times.keys():
            times = self.fp16_inference_times[model_name]
            if times:
                stats['precision_performance']['fp16'][model_name] = {
                    'total_inferences': len(times),
                    'avg_inference_time_ms': np.mean(times),
                    'throughput_inferences_per_second': 1000 / np.mean(times) if np.mean(times) > 0 else 0
                }
        
        # INT8 performance
        for model_name in self.int8_inference_times.keys():
            times = self.int8_inference_times[model_name]
            if times:
                stats['model_performance'][model_name] = {
                    'precision': 'INT8',
                    'total_inferences': len(times),
                    'avg_inference_time_ms': np.mean(times),
                    'min_inference_time_ms': np.min(times),
                    'max_inference_time_ms': np.max(times),
                    'throughput_inferences_per_second': 1000 / np.mean(times) if np.mean(times) > 0 else 0
                }
                stats['precision_performance']['int8'][model_name] = {
                    'total_inferences': len(times),
                    'avg_inference_time_ms': np.mean(times),
                    'throughput_inferences_per_second': 1000 / np.mean(times) if np.mean(times) > 0 else 0
                }
        
        return stats

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        unloaded = False
        
        if model_name in self.sessions:
            del self.sessions[model_name]
            if model_name in self.inference_times:
                del self.inference_times[model_name]
            if model_name in self.fp16_inference_times:
                del self.fp16_inference_times[model_name]
            unloaded = True
        
        if model_name in self.quantized_sessions:
            del self.quantized_sessions[model_name]
            if model_name in self.int8_inference_times:
                del self.int8_inference_times[model_name]
            unloaded = True
        
        if unloaded:
            logger.info(f"Unloaded model: {model_name}")
        
        return unloaded

    def unload_all_models(self):
        """Unload all models from memory"""
        self.sessions.clear()
        self.quantized_sessions.clear()
        self.inference_times.clear()
        self.fp16_inference_times.clear()
        self.int8_inference_times.clear()
        logger.info("Unloaded all ONNX models")

    def _determine_optimal_precision(self, input_data: np.ndarray) -> np.dtype:
        """Determine optimal precision for input data"""
        if not self.enable_mixed_precision:
            return np.float32
        
        # Check if CUDA provider is available
        if 'CUDAExecutionProvider' not in self.providers:
            return np.float32
        
        # Use FP16 for inference if data is already float32 or float64
        if input_data.dtype in [np.float32, np.float64]:
            return np.float16
        
        # Keep original precision for other data types
        return input_data.dtype


# Global ONNX inference engine instance
onnx_inference_engine = ONNXInferenceEngine()
