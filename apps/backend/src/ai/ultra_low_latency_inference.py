#!/usr/bin/env python3
"""
Ultra-Low Latency Inference Engine for AlphaPulse
Integrates knowledge distillation, feature caching, and ONNX optimization
Target: <10ms inference latency for real-time trading
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import time
import pickle
from pathlib import Path
from collections import defaultdict, deque

# ONNX imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Local imports
from .knowledge_distillation import KnowledgeDistillation, DistillationConfig
from .feature_cache_manager import FeatureCacheManager, FeatureCacheConfig
from .onnx_inference import ONNXInferenceEngine
from .model_registry import ModelRegistry
from ..advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for ultra-low latency inference"""
    target_latency_ms: float = 10.0  # Target inference latency
    enable_knowledge_distillation: bool = True
    enable_feature_caching: bool = True
    enable_onnx: bool = True
    enable_batching: bool = True
    batch_size: int = 4  # Optimal batch size for latency/throughput trade-off
    warmup_runs: int = 100  # Number of warmup runs
    enable_async: bool = True
    enable_monitoring: bool = True
    fallback_to_ensemble: bool = True  # Fallback to ensemble if student model fails

@dataclass
class InferenceResult:
    """Result from ultra-low latency inference"""
    prediction: float
    confidence: float
    latency_ms: float
    model_used: str
    features_used: Dict[str, float]
    cache_hit: bool
    timestamp: datetime
    metadata: Dict[str, Any]

class UltraLowLatencyInference:
    """
    Ultra-low latency inference engine for real-time trading
    """
    
    def __init__(self, config: InferenceConfig = None):
        """
        Initialize ultra-low latency inference engine
        
        Args:
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        
        # Initialize components
        self.knowledge_distillation = KnowledgeDistillation()
        self.feature_cache_manager = FeatureCacheManager()
        self.onnx_inference_engine = ONNXInferenceEngine() if ONNX_AVAILABLE else None
        self.model_registry = ModelRegistry()
        
        # Load models
        self.student_model = None
        self.ensemble_model = None
        self.onnx_model = None
        
        # Performance tracking
        self.inference_times: List[float] = deque(maxlen=1000)
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.error_count: int = 0
        
        # Warm-up flag
        self.is_warmed_up = False
        
        logger.info(f"✅ Ultra-low latency inference engine initialized")
        logger.info(f"   - Target latency: {self.config.target_latency_ms}ms")
        logger.info(f"   - Knowledge distillation: {self.config.enable_knowledge_distillation}")
        logger.info(f"   - Feature caching: {self.config.enable_feature_caching}")
        logger.info(f"   - ONNX: {self.config.enable_onnx and ONNX_AVAILABLE}")
    
    async def initialize(self, 
                        training_data: pd.DataFrame = None,
                        distillation_config: DistillationConfig = None):
        """
        Initialize the inference engine with models
        
        Args:
            training_data: Data for knowledge distillation
            distillation_config: Distillation configuration
        """
        try:
            logger.info("🔄 Initializing ultra-low latency inference engine...")
            
            # Step 1: Load or create distilled model
            if self.config.enable_knowledge_distillation:
                await self._initialize_distilled_model(training_data, distillation_config)
            
            # Step 2: Load ensemble model as fallback
            if self.config.fallback_to_ensemble:
                await self._initialize_ensemble_model()
            
            # Step 3: Load ONNX model if available
            if self.config.enable_onnx and ONNX_AVAILABLE:
                await self._initialize_onnx_model()
            
            # Step 4: Warm up models
            await self._warm_up_models()
            
            logger.info("✅ Ultra-low latency inference engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing inference engine: {e}")
            raise
    
    async def predict(self, 
                     symbol: str, 
                     timeframe: str,
                     candlestick_data: pd.DataFrame) -> InferenceResult:
        """
        Make ultra-low latency prediction
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            candlestick_data: OHLCV data
            
        Returns:
            InferenceResult with prediction and performance metrics
        """
        start_time = time.time()
        cache_hit = False
        
        try:
            # Step 1: Get features (cached or computed)
            features = await self._get_features(symbol, timeframe, candlestick_data)
            if not features:
                raise ValueError("No features available")
            
            # Step 2: Make prediction with fastest available model
            prediction, confidence, model_used = await self._make_prediction(features)
            
            # Step 3: Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Step 4: Track performance
            self.inference_times.append(latency_ms)
            self.model_usage[model_used] += 1
            
            # Step 5: Check if target latency is met
            if latency_ms > self.config.target_latency_ms:
                logger.warning(f"⚠️ Inference latency {latency_ms:.1f}ms exceeds target {self.config.target_latency_ms}ms")
            
            result = InferenceResult(
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                model_used=model_used,
                features_used=features,
                cache_hit=cache_hit,
                timestamp=datetime.now(),
                metadata={
                    'target_latency_ms': self.config.target_latency_ms,
                    'latency_target_met': latency_ms <= self.config.target_latency_ms,
                    'feature_count': len(features)
                }
            )
            
            logger.debug(f"✅ Prediction: {prediction:.3f} (confidence: {confidence:.3f}) in {latency_ms:.1f}ms using {model_used}")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in prediction: {e}")
            
            # Return fallback result
            return InferenceResult(
                prediction=0.5,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                model_used="fallback",
                features_used={},
                cache_hit=False,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    async def predict_batch(self, 
                           predictions: List[Tuple[str, str, pd.DataFrame]]) -> List[InferenceResult]:
        """
        Make batch predictions for multiple symbols/timeframes
        
        Args:
            predictions: List of (symbol, timeframe, candlestick_data) tuples
            
        Returns:
            List of InferenceResult objects
        """
        if not self.config.enable_batching:
            # Sequential predictions
            results = []
            for symbol, timeframe, candlestick_data in predictions:
                result = await self.predict(symbol, timeframe, candlestick_data)
                results.append(result)
            return results
        
        try:
            # Batch processing
            start_time = time.time()
            
            # Step 1: Get features for all predictions
            feature_tasks = []
            for symbol, timeframe, candlestick_data in predictions:
                task = self._get_features(symbol, timeframe, candlestick_data)
                feature_tasks.append(task)
            
            features_list = await asyncio.gather(*feature_tasks)
            
            # Step 2: Make batch predictions
            results = []
            for i, (symbol, timeframe, candlestick_data) in enumerate(predictions):
                features = features_list[i]
                if features:
                    prediction, confidence, model_used = await self._make_prediction(features)
                    
                    result = InferenceResult(
                        prediction=prediction,
                        confidence=confidence,
                        latency_ms=(time.time() - start_time) * 1000,
                        model_used=model_used,
                        features_used=features,
                        cache_hit=False,  # Batch processing doesn't track individual cache hits
                        timestamp=datetime.now(),
                        metadata={'batch_index': i, 'batch_size': len(predictions)}
                    )
                else:
                    result = InferenceResult(
                        prediction=0.5,
                        confidence=0.0,
                        latency_ms=(time.time() - start_time) * 1000,
                        model_used="fallback",
                        features_used={},
                        cache_hit=False,
                        timestamp=datetime.now(),
                        metadata={'error': 'No features available', 'batch_index': i}
                    )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in batch prediction: {e}")
            # Return fallback results
            return [
                InferenceResult(
                    prediction=0.5,
                    confidence=0.0,
                    latency_ms=0.0,
                    model_used="fallback",
                    features_used={},
                    cache_hit=False,
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                )
                for _ in predictions
            ]
    
    async def _initialize_distilled_model(self, 
                                        training_data: pd.DataFrame = None,
                                        distillation_config: DistillationConfig = None):
        """Initialize knowledge distillation model"""
        try:
            logger.info("🔄 Initializing knowledge distillation model...")
            
            # Try to load existing distilled model
            student_model_path = self._find_latest_student_model()
            if student_model_path:
                self.student_model = await self.knowledge_distillation.load_distilled_model(student_model_path)
                logger.info(f"✅ Loaded existing distilled model: {student_model_path}")
                return
            
            # Create new distilled model if training data provided
            if training_data is not None:
                logger.info("🔄 Creating new distilled model...")
                
                # Prepare training data
                X = training_data.drop(['target', 'timestamp'], axis=1, errors='ignore')
                y = training_data['target'] if 'target' in training_data.columns else pd.Series([0.5] * len(training_data))
                
                # Create distilled model
                config = distillation_config or DistillationConfig()
                result = await self.knowledge_distillation.create_distilled_ensemble(X, y, config)
                
                # Load the created model
                self.student_model = await self.knowledge_distillation.load_distilled_model(result.student_model_path)
                
                logger.info(f"✅ Created new distilled model: {result.student_model_path}")
                logger.info(f"   - Accuracy preservation: {result.accuracy_preservation:.1f}%")
                logger.info(f"   - Latency improvement: {result.latency_improvement:.1f}x")
            else:
                logger.warning("⚠️ No training data provided for knowledge distillation")
                
        except Exception as e:
            logger.error(f"❌ Error initializing distilled model: {e}")
    
    async def _initialize_ensemble_model(self):
        """Initialize ensemble model as fallback"""
        try:
            logger.info("🔄 Initializing ensemble model...")
            
            # Try to get ensemble from model registry
            self.ensemble_model = self.model_registry.get_ensemble()
            
            if self.ensemble_model:
                logger.info("✅ Loaded ensemble model from registry")
            else:
                logger.warning("⚠️ No ensemble model available in registry")
                
        except Exception as e:
            logger.error(f"❌ Error initializing ensemble model: {e}")
    
    async def _initialize_onnx_model(self):
        """Initialize ONNX model"""
        try:
            if not ONNX_AVAILABLE:
                return
            
            logger.info("🔄 Initializing ONNX model...")
            
            # Try to load ONNX model
            onnx_model_path = self._find_latest_onnx_model()
            if onnx_model_path:
                success = self.onnx_inference_engine.load_model("student", str(onnx_model_path))
                if success:
                    self.onnx_model = onnx_model_path
                    logger.info(f"✅ Loaded ONNX model: {onnx_model_path}")
                    return
            
            logger.warning("⚠️ No ONNX model available")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ONNX model: {e}")
    
    async def _warm_up_models(self):
        """Warm up models for optimal performance"""
        try:
            logger.info(f"🔄 Warming up models ({self.config.warmup_runs} runs)...")
            
            # Create dummy features for warm-up
            dummy_features = {
                'rsi': 50.0, 'macd': 0.0, 'bb_position': 0.5,
                'sma_20': 100.0, 'volume_ratio': 1.0, 'atr': 1.0,
                'close': 100.0, 'open': 100.0, 'high': 101.0, 'low': 99.0,
                'volume': 1000.0, 'price_change': 0.0, 'price_change_pct': 0.0
            }
            
            # Warm up each model
            for _ in range(self.config.warmup_runs):
                if self.student_model:
                    await self._predict_with_student_model(dummy_features)
                
                if self.ensemble_model:
                    await self._predict_with_ensemble_model(dummy_features)
                
                if self.onnx_model:
                    await self._predict_with_onnx_model(dummy_features)
            
            self.is_warmed_up = True
            logger.info("✅ Models warmed up successfully")
            
        except Exception as e:
            logger.error(f"❌ Error warming up models: {e}")
    
    async def _get_features(self, symbol: str, timeframe: str, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Get features with caching"""
        try:
            if self.config.enable_feature_caching:
                features = await self.feature_cache_manager.get_features(symbol, timeframe, candlestick_data)
            else:
                # Direct computation without caching
                features = await self.feature_cache_manager._compute_features(candlestick_data)
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            return {}
    
    async def _make_prediction(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """Make prediction using fastest available model"""
        try:
            # Try ONNX model first (fastest)
            if self.onnx_model and self.config.enable_onnx:
                try:
                    prediction, confidence = await self._predict_with_onnx_model(features)
                    return prediction, confidence, "onnx"
                except Exception as e:
                    logger.debug(f"ONNX prediction failed: {e}")
            
            # Try student model (knowledge distillation)
            if self.student_model and self.config.enable_knowledge_distillation:
                try:
                    prediction, confidence = await self._predict_with_student_model(features)
                    return prediction, confidence, "student"
                except Exception as e:
                    logger.debug(f"Student model prediction failed: {e}")
            
            # Fallback to ensemble model
            if self.ensemble_model and self.config.fallback_to_ensemble:
                try:
                    prediction, confidence = await self._predict_with_ensemble_model(features)
                    return prediction, confidence, "ensemble"
                except Exception as e:
                    logger.debug(f"Ensemble prediction failed: {e}")
            
            # Final fallback
            logger.warning("⚠️ All models failed, using fallback prediction")
            return 0.5, 0.0, "fallback"
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5, 0.0, "error"
    
    async def _predict_with_student_model(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction with student model"""
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = await self.knowledge_distillation.predict_with_distilled_model(
                self.student_model, features_df
            )
            
            prediction_value = float(prediction[0])
            confidence = abs(prediction_value - 0.5) * 2  # Distance from 0.5
            
            return prediction_value, confidence
            
        except Exception as e:
            logger.error(f"Error with student model prediction: {e}")
            raise
    
    async def _predict_with_ensemble_model(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction with ensemble model"""
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Make prediction
            prediction_proba = self.ensemble_model.predict_proba(features_df)
            prediction_value = float(prediction_proba[0][1])  # Probability of positive class
            confidence = abs(prediction_value - 0.5) * 2
            
            return prediction_value, confidence
            
        except Exception as e:
            logger.error(f"Error with ensemble model prediction: {e}")
            raise
    
    async def _predict_with_onnx_model(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction with ONNX model"""
        try:
            if not self.onnx_inference_engine:
                raise ValueError("ONNX inference engine not available")
            
            # Convert features to numpy array
            feature_names = sorted(features.keys())
            feature_values = np.array([[features[name] for name in feature_names]], dtype=np.float32)
            
            # Make prediction
            prediction = self.onnx_inference_engine.predict("student", feature_values)
            prediction_value = float(prediction[0])
            confidence = abs(prediction_value - 0.5) * 2
            
            return prediction_value, confidence
            
        except Exception as e:
            logger.error(f"Error with ONNX model prediction: {e}")
            raise
    
    def _find_latest_student_model(self) -> Optional[str]:
        """Find the latest student model file"""
        try:
            student_dir = Path("models/student")
            if not student_dir.exists():
                return None
            
            # Find all student model files
            model_files = list(student_dir.glob("student_*.pkl"))
            if not model_files:
                return None
            
            # Return the most recent one
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            return str(latest_model)
            
        except Exception as e:
            logger.error(f"Error finding latest student model: {e}")
            return None
    
    def _find_latest_onnx_model(self) -> Optional[str]:
        """Find the latest ONNX model file"""
        try:
            student_dir = Path("models/student")
            if not student_dir.exists():
                return None
            
            # Find all ONNX model files
            onnx_files = list(student_dir.glob("student_*.onnx"))
            if not onnx_files:
                return None
            
            # Return the most recent one
            latest_onnx = max(onnx_files, key=lambda x: x.stat().st_mtime)
            return str(latest_onnx)
            
        except Exception as e:
            logger.error(f"Error finding latest ONNX model: {e}")
            return None
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            inference_times = list(self.inference_times)
            
            stats = {
                'total_predictions': len(inference_times),
                'avg_latency_ms': np.mean(inference_times) if inference_times else 0,
                'p50_latency_ms': np.percentile(inference_times, 50) if inference_times else 0,
                'p90_latency_ms': np.percentile(inference_times, 90) if inference_times else 0,
                'p99_latency_ms': np.percentile(inference_times, 99) if inference_times else 0,
                'max_latency_ms': np.max(inference_times) if inference_times else 0,
                'min_latency_ms': np.min(inference_times) if inference_times else 0,
                'target_latency_ms': self.config.target_latency_ms,
                'target_met_pct': (
                    sum(1 for t in inference_times if t <= self.config.target_latency_ms) / len(inference_times) * 100
                    if inference_times else 0
                ),
                'model_usage': dict(self.model_usage),
                'error_count': self.error_count,
                'cache_stats': await self.feature_cache_manager.get_cache_stats(),
                'is_warmed_up': self.is_warmed_up,
                'models_available': {
                    'student_model': self.student_model is not None,
                    'ensemble_model': self.ensemble_model is not None,
                    'onnx_model': self.onnx_model is not None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    async def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times.clear()
        self.model_usage.clear()
        self.error_count = 0
        logger.info("✅ Performance statistics reset")

# Global instance
ultra_low_latency_inference = UltraLowLatencyInference()
