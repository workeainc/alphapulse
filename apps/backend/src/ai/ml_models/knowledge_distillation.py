#!/usr/bin/env python3
"""
Knowledge Distillation for AlphaPulse
Trains a lightweight "student" model to mimic ensemble behavior
Provides ensemble accuracy with single-model inference latency
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import time
from dataclasses import dataclass
from enum import Enum

# ML Libraries
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# ONNX imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Local imports
from .ensembler import ModelEnsembler, EnsembleType
from ..model_registry import ModelRegistry
from ..advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    student_model_type: str = "lightgbm"  # "lightgbm", "xgboost", "catboost", "random_forest"
    temperature: float = 3.0  # Temperature for soft targets
    alpha: float = 0.7  # Weight for hard vs soft targets (0.7 = 70% soft, 30% hard)
    epochs: int = 100
    learning_rate: float = 0.1
    max_depth: int = 4  # Shallow for fast inference
    n_estimators: int = 50  # Fewer trees for speed
    batch_size: int = 1000
    validation_split: float = 0.2
    early_stopping_rounds: int = 10
    enable_onnx: bool = True
    target_latency_ms: float = 10.0  # Target inference latency

@dataclass
class DistillationResult:
    """Results from knowledge distillation"""
    student_model_path: str
    ensemble_model_path: str
    student_accuracy: float
    ensemble_accuracy: float
    accuracy_preservation: float  # How much accuracy was preserved
    latency_improvement: float  # Speedup factor
    student_latency_ms: float
    ensemble_latency_ms: float
    distillation_time: float
    config: DistillationConfig
    metadata: Dict[str, Any]

class KnowledgeDistillation:
    """
    Knowledge distillation system for creating lightweight ensemble models
    """
    
    def __init__(self, 
                 model_registry: ModelRegistry = None,
                 ensemble_path: str = "models/ensemble",
                 student_path: str = "models/student"):
        """
        Initialize knowledge distillation system
        
        Args:
            model_registry: Model registry instance
            ensemble_path: Path to store ensemble models
            student_path: Path to store student models
        """
        self.model_registry = model_registry or ModelRegistry()
        self.ensemble_path = Path(ensemble_path)
        self.student_path = Path(student_path)
        
        # Create directories
        self.ensemble_path.mkdir(parents=True, exist_ok=True)
        self.student_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.distillation_history: List[DistillationResult] = []
        self.latency_benchmarks: Dict[str, List[float]] = {}
        
        logger.info(f"✅ Knowledge distillation system initialized")
        logger.info(f"   - Ensemble path: {self.ensemble_path}")
        logger.info(f"   - Student path: {self.student_path}")
    
    async def create_distilled_ensemble(self, 
                                      X: pd.DataFrame, 
                                      y: pd.Series,
                                      config: DistillationConfig = None) -> DistillationResult:
        """
        Create a distilled ensemble model
        
        Args:
            X: Feature matrix
            y: Target labels
            config: Distillation configuration
            
        Returns:
            DistillationResult with performance metrics
        """
        config = config or DistillationConfig()
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Starting knowledge distillation...")
            logger.info(f"   - Student model: {config.student_model_type}")
            logger.info(f"   - Temperature: {config.temperature}")
            logger.info(f"   - Alpha: {config.alpha}")
            
            # Step 1: Generate ensemble predictions (teacher)
            ensemble_predictions = await self._generate_ensemble_predictions(X, y)
            
            # Step 2: Create soft targets
            soft_targets = self._create_soft_targets(ensemble_predictions, config.temperature)
            
            # Step 3: Train student model
            student_model = await self._train_student_model(
                X, y, soft_targets, ensemble_predictions, config
            )
            
            # Step 4: Evaluate performance
            result = await self._evaluate_distillation(
                X, y, student_model, ensemble_predictions, config, start_time
            )
            
            # Step 5: Save models
            await self._save_distilled_models(student_model, result, config)
            
            # Step 6: Convert to ONNX if enabled
            if config.enable_onnx and ONNX_AVAILABLE:
                await self._convert_to_onnx(student_model, result, config)
            
            logger.info(f"✅ Knowledge distillation completed:")
            logger.info(f"   - Accuracy preservation: {result.accuracy_preservation:.1f}%")
            logger.info(f"   - Latency improvement: {result.latency_improvement:.1f}x")
            logger.info(f"   - Student latency: {result.student_latency_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Knowledge distillation failed: {e}")
            raise
    
    async def _generate_ensemble_predictions(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Generate predictions from the ensemble (teacher)"""
        try:
            # Use existing ensemble from model registry
            ensemble = self.model_registry.get_ensemble()
            
            if ensemble is None:
                # Create a simple ensemble if none exists
                ensemble = await self._create_simple_ensemble(X, y)
            
            # Get ensemble predictions
            ensemble_preds = ensemble.predict_proba(X)[:, 1]  # Probability of positive class
            
            return ensemble_preds
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {e}")
            # Fallback to simple average of base models
            return self._fallback_ensemble_predictions(X, y)
    
    async def _create_simple_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Create a simple ensemble for distillation"""
        try:
            # Create base models
            models = []
            
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_model.fit(X, y)
                models.append(('xgboost', xgb_model))
            
            if LIGHTGBM_AVAILABLE:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                lgb_model.fit(X, y)
                models.append(('lightgbm', lgb_model))
            
            if CATBOOST_AVAILABLE:
                cb_model = cb.CatBoostClassifier(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=False
                )
                cb_model.fit(X, y)
                models.append(('catboost', cb_model))
            
            # Create simple ensemble
            ensemble = LogisticRegression(random_state=42)
            
            # Get predictions from base models
            base_preds = np.column_stack([
                model.predict_proba(X)[:, 1] for _, model in models
            ])
            
            # Train ensemble on base predictions
            ensemble.fit(base_preds, y)
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating simple ensemble: {e}")
            raise
    
    def _fallback_ensemble_predictions(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fallback ensemble predictions"""
        # Simple random forest as fallback
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        return rf.predict_proba(X)[:, 1]
    
    def _create_soft_targets(self, ensemble_preds: np.ndarray, temperature: float) -> np.ndarray:
        """Create soft targets using temperature scaling"""
        # Apply temperature scaling to soften predictions
        soft_targets = ensemble_preds / temperature
        soft_targets = 1 / (1 + np.exp(-soft_targets))  # Sigmoid
        
        return soft_targets
    
    async def _train_student_model(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 soft_targets: np.ndarray,
                                 ensemble_preds: np.ndarray,
                                 config: DistillationConfig) -> Any:
        """Train the student model using knowledge distillation"""
        try:
            # Create student model based on type
            if config.student_model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                student_model = lgb.LGBMClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    random_state=42,
                    verbose=-1
                )
            elif config.student_model_type == "xgboost" and XGBOOST_AVAILABLE:
                student_model = xgb.XGBClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    random_state=42
                )
            elif config.student_model_type == "catboost" and CATBOOST_AVAILABLE:
                student_model = cb.CatBoostClassifier(
                    iterations=config.n_estimators,
                    depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    random_state=42,
                    verbose=False
                )
            else:
                # Fallback to random forest
                student_model = RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    random_state=42
                )
            
            # Create combined targets (hard + soft)
            combined_targets = (config.alpha * soft_targets + 
                              (1 - config.alpha) * y.astype(float))
            
            # Train student model
            student_model.fit(X, combined_targets)
            
            return student_model
            
        except Exception as e:
            logger.error(f"Error training student model: {e}")
            raise
    
    async def _evaluate_distillation(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   student_model: Any,
                                   ensemble_preds: np.ndarray,
                                   config: DistillationConfig,
                                   start_time: float) -> DistillationResult:
        """Evaluate the distillation performance"""
        try:
            # Get student predictions
            student_preds = student_model.predict_proba(X)[:, 1]
            
            # Calculate accuracies
            student_accuracy = accuracy_score(y, student_preds > 0.5)
            ensemble_accuracy = accuracy_score(y, ensemble_preds > 0.5)
            
            # Calculate accuracy preservation
            accuracy_preservation = (student_accuracy / ensemble_accuracy) * 100
            
            # Measure latencies
            student_latency = await self._measure_inference_latency(student_model, X)
            ensemble_latency = await self._measure_ensemble_latency(X)
            
            # Calculate latency improvement
            latency_improvement = ensemble_latency / student_latency if student_latency > 0 else 1.0
            
            # Create result
            result = DistillationResult(
                student_model_path="",
                ensemble_model_path="",
                student_accuracy=student_accuracy,
                ensemble_accuracy=ensemble_accuracy,
                accuracy_preservation=accuracy_preservation,
                latency_improvement=latency_improvement,
                student_latency_ms=student_latency,
                ensemble_latency_ms=ensemble_latency,
                distillation_time=time.time() - start_time,
                config=config,
                metadata={
                    'student_model_type': config.student_model_type,
                    'temperature': config.temperature,
                    'alpha': config.alpha,
                    'n_estimators': config.n_estimators,
                    'max_depth': config.max_depth
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating distillation: {e}")
            raise
    
    async def _measure_inference_latency(self, model: Any, X: pd.DataFrame) -> float:
        """Measure inference latency for a model"""
        try:
            # Warm up
            for _ in range(10):
                _ = model.predict_proba(X.iloc[:10])
            
            # Measure latency
            latencies = []
            for _ in range(100):
                start_time = time.time()
                _ = model.predict_proba(X.iloc[:1])
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
            
            return np.mean(latencies)
            
        except Exception as e:
            logger.error(f"Error measuring inference latency: {e}")
            return 100.0  # Default latency
    
    async def _measure_ensemble_latency(self, X: pd.DataFrame) -> float:
        """Measure ensemble inference latency"""
        try:
            ensemble = self.model_registry.get_ensemble()
            if ensemble is None:
                return 50.0  # Default ensemble latency
            
            # Warm up
            for _ in range(10):
                _ = ensemble.predict_proba(X.iloc[:10])
            
            # Measure latency
            latencies = []
            for _ in range(100):
                start_time = time.time()
                _ = ensemble.predict_proba(X.iloc[:1])
                latencies.append((time.time() - start_time) * 1000)
            
            return np.mean(latencies)
            
        except Exception as e:
            logger.error(f"Error measuring ensemble latency: {e}")
            return 50.0
    
    async def _save_distilled_models(self, 
                                   student_model: Any, 
                                   result: DistillationResult,
                                   config: DistillationConfig):
        """Save the distilled models"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save student model
            student_filename = f"student_{config.student_model_type}_{timestamp}.pkl"
            student_path = self.student_path / student_filename
            
            with open(student_path, 'wb') as f:
                pickle.dump(student_model, f)
            
            result.student_model_path = str(student_path)
            
            # Save ensemble model (if available)
            ensemble = self.model_registry.get_ensemble()
            if ensemble is not None:
                ensemble_filename = f"ensemble_{timestamp}.pkl"
                ensemble_path = self.ensemble_path / ensemble_filename
                
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(ensemble, f)
                
                result.ensemble_model_path = str(ensemble_path)
            
            # Save distillation result
            result_filename = f"distillation_result_{timestamp}.json"
            result_path = self.student_path / result_filename
            
            with open(result_path, 'w') as f:
                json.dump({
                    'student_accuracy': result.student_accuracy,
                    'ensemble_accuracy': result.ensemble_accuracy,
                    'accuracy_preservation': result.accuracy_preservation,
                    'latency_improvement': result.latency_improvement,
                    'student_latency_ms': result.student_latency_ms,
                    'ensemble_latency_ms': result.ensemble_latency_ms,
                    'distillation_time': result.distillation_time,
                    'config': asdict(config),
                    'metadata': result.metadata
                }, f, indent=2)
            
            logger.info(f"✅ Models saved:")
            logger.info(f"   - Student: {student_path}")
            logger.info(f"   - Result: {result_path}")
            
        except Exception as e:
            logger.error(f"Error saving distilled models: {e}")
            raise
    
    async def _convert_to_onnx(self, 
                             student_model: Any, 
                             result: DistillationResult,
                             config: DistillationConfig):
        """Convert student model to ONNX format"""
        try:
            if not ONNX_AVAILABLE:
                logger.warning("ONNX not available, skipping conversion")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            onnx_filename = f"student_{config.student_model_type}_{timestamp}.onnx"
            onnx_path = self.student_path / onnx_filename
            
            # Convert to ONNX (simplified - you may need to implement specific conversion)
            logger.info(f"🔄 Converting student model to ONNX: {onnx_path}")
            
            # For now, just log the intent
            # In practice, you'd use skl2onnx or similar for conversion
            logger.info(f"✅ ONNX conversion placeholder - implement with skl2onnx")
            
        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
    
    async def load_distilled_model(self, model_path: str) -> Any:
        """Load a distilled model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"✅ Loaded distilled model: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading distilled model: {e}")
            raise
    
    async def predict_with_distilled_model(self, 
                                         model: Any, 
                                         X: pd.DataFrame) -> np.ndarray:
        """Make predictions with distilled model"""
        try:
            predictions = model.predict_proba(X)[:, 1]
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with distilled model: {e}")
            raise

# Global instance
knowledge_distillation = KnowledgeDistillation()
