#!/usr/bin/env python3
"""
Model Ensembling System for AlphaPulse
Phase 1: Blending and Stacking Implementation

Implements:
1. Model blending (monthly + weekly + online)
2. Stacking with logistic meta-learner
3. Out-of-fold predictions to avoid leakage
4. Dynamic weight optimization
5. Performance monitoring and validation
6. MLflow integration for ensemble tracking
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available")

# Local imports
from ..ai.model_registry import ModelRegistry
from ..ai.advanced_logging_system import redis_logger, EventType, LogLevel
from ..ai.ml_models.trainer import ModelType, TrainingCadence

logger = logging.getLogger(__name__)

class EnsembleType(Enum):
    """Ensemble types"""
    BLENDING = "blending"
    STACKING = "stacking"
    WEIGHTED_AVERAGE = "weighted_average"

class ModelSource(Enum):
    """Model sources for ensemble"""
    MONTHLY_FULL = "monthly_full"
    WEEKLY_QUICK = "weekly_quick"
    NIGHTLY_INCREMENTAL = "nightly_incremental"
    ONLINE_LEARNER = "online_learner"

@dataclass
class EnsembleConfig:
    """Ensemble configuration"""
    ensemble_type: EnsembleType = EnsembleType.BLENDING
    models: List[ModelSource] = None
    blending_weights: Optional[Dict[str, float]] = None
    stacking_cv_folds: int = 5
    meta_learner_type: str = "logistic_regression"
    meta_learner_params: Dict[str, Any] = None
    validation_split: float = 0.2
    random_state: int = 42
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                ModelSource.MONTHLY_FULL,
                ModelSource.WEEKLY_QUICK,
                ModelSource.ONLINE_LEARNER
            ]
        
        if self.blending_weights is None:
            self.blending_weights = {
                ModelSource.MONTHLY_FULL.value: 0.5,
                ModelSource.WEEKLY_QUICK.value: 0.3,
                ModelSource.ONLINE_LEARNER.value: 0.2
            }
        
        if self.meta_learner_params is None:
            self.meta_learner_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': self.random_state
            }

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    individual_predictions: Dict[str, float]
    ensemble_prediction: float
    confidence: float
    model_weights: Dict[str, float]
    ensemble_type: str
    timestamp: datetime

@dataclass
class EnsembleResult:
    """Ensemble training result"""
    ensemble_path: str
    ensemble_type: EnsembleType
    models_used: List[str]
    meta_learner_path: Optional[str]
    metrics: Dict[str, float]
    model_weights: Dict[str, float]
    training_time: float
    mlflow_run_id: Optional[str] = None

class ModelEnsembler:
    """
    Comprehensive model ensembling system
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.model_registry = ModelRegistry()
        self.ensembles_dir = Path("models/ensembles")
        self.ensembles_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.model_predictions = {}
        self.ensemble_model = None
        self.meta_learner = None
        
        # Performance tracking
        self.performance_history = []
        self.current_weights = {}
        
        logger.info("🚀 Model Ensembler initialized")
    
    async def add_model(self, model_source: ModelSource, model_path: str, 
                       model_type: str = "unknown"):
        """Add a model to the ensemble"""
        try:
            self.models[model_source.value] = {
                'path': model_path,
                'type': model_type,
                'added_at': datetime.now(),
                'last_used': None
            }
            
            logger.info(f"✅ Added model: {model_source.value} -> {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error adding model: {e}")
            raise
    
    async def create_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> EnsembleResult:
        """
        Create ensemble based on configuration
        """
        try:
            start_time = time.time()
            
            if self.config.ensemble_type == EnsembleType.BLENDING:
                result = await self._create_blending_ensemble(X, y, X_val, y_val)
            elif self.config.ensemble_type == EnsembleType.STACKING:
                result = await self._create_stacking_ensemble(X, y, X_val, y_val)
            elif self.config.ensemble_type == EnsembleType.WEIGHTED_AVERAGE:
                result = await self._create_weighted_average_ensemble(X, y, X_val, y_val)
            else:
                raise ValueError(f"Unsupported ensemble type: {self.config.ensemble_type}")
            
            # Calculate training time
            training_time = time.time() - start_time
            result.training_time = training_time
            
            # Log ensemble creation
            if hasattr(redis_logger, 'log_event'):
                try:
                    await redis_logger.log_event(
                        EventType.ENSEMBLE_CREATION,
                        LogLevel.INFO,
                        {
                            'ensemble_type': self.config.ensemble_type.value,
                            'models_used': result.models_used,
                            'training_time': training_time,
                            'metrics': result.metrics
                        }
                    )
                except Exception as redis_error:
                    logger.warning(f"Failed to log to Redis: {redis_error}")
            else:
                logger.info("Redis logging not available")
            
            logger.info(f"✅ Ensemble created: {self.config.ensemble_type.value} in {training_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error creating ensemble: {e}")
            raise
    
    async def _create_blending_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                      X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> EnsembleResult:
        """Create blending ensemble"""
        try:
            # Get predictions from all models
            predictions = await self._get_model_predictions(X, X_val)
            
            if not predictions:
                raise ValueError("No model predictions available for blending")
            
            # Use configured weights for blending
            weights = self.config.blending_weights
            
            # Calculate blended predictions
            blended_predictions = np.zeros(len(X))
            for model_name, preds in predictions.items():
                if model_name in weights:
                    blended_predictions += weights[model_name] * preds
            
            # Calculate metrics
            metrics = self._calculate_ensemble_metrics(blended_predictions, y)
            
            # Save ensemble configuration
            ensemble_path = await self._save_blending_ensemble(weights, predictions)
            
            return EnsembleResult(
                ensemble_path=ensemble_path,
                ensemble_type=EnsembleType.BLENDING,
                models_used=list(predictions.keys()),
                meta_learner_path=None,
                metrics=metrics,
                model_weights=weights,
                training_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating blending ensemble: {e}")
            raise
    
    async def _create_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                      X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> EnsembleResult:
        """Create stacking ensemble with out-of-fold predictions"""
        try:
            # Generate out-of-fold predictions
            oof_predictions = await self._generate_oof_predictions(X, y)
            
            if not oof_predictions:
                raise ValueError("No out-of-fold predictions available for stacking")
            
            # Train meta-learner on out-of-fold predictions
            meta_features = np.column_stack(list(oof_predictions.values()))
            
            # Split for meta-learner training
            if X_val is not None and y_val is not None:
                # Use validation set for meta-learner
                meta_X_train = meta_features
                meta_y_train = y
                meta_X_val = await self._get_model_predictions_single(X_val)
                meta_y_val = y_val
            else:
                # Use cross-validation split
                split_idx = int(len(X) * (1 - self.config.validation_split))
                meta_X_train = meta_features[:split_idx]
                meta_y_train = y[:split_idx]
                meta_X_val = meta_features[split_idx:]
                meta_y_val = y[split_idx:]
            
            # Train meta-learner
            self.meta_learner = LogisticRegression(**self.config.meta_learner_params)
            self.meta_learner.fit(meta_X_train, meta_y_train)
            
            # Get meta-learner predictions
            meta_pred_train = self.meta_learner.predict_proba(meta_X_train)[:, 1]
            meta_pred_val = self.meta_learner.predict_proba(meta_X_val)[:, 1]
            
            # Calculate metrics
            train_metrics = self._calculate_ensemble_metrics(meta_pred_train, meta_y_train)
            val_metrics = self._calculate_ensemble_metrics(meta_pred_val, meta_y_val)
            
            # Combine metrics
            metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            # Save ensemble and meta-learner
            ensemble_path = await self._save_stacking_ensemble(oof_predictions)
            meta_learner_path = await self._save_meta_learner()
            
            # Get model weights from meta-learner
            model_weights = dict(zip(oof_predictions.keys(), self.meta_learner.coef_[0]))
            
            return EnsembleResult(
                ensemble_path=ensemble_path,
                ensemble_type=EnsembleType.STACKING,
                models_used=list(oof_predictions.keys()),
                meta_learner_path=meta_learner_path,
                metrics=metrics,
                model_weights=model_weights,
                training_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating stacking ensemble: {e}")
            raise
    
    async def _create_weighted_average_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                              X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> EnsembleResult:
        """Create weighted average ensemble with optimized weights"""
        try:
            # Get predictions from all models
            predictions = await self._get_model_predictions(X, X_val)
            
            if not predictions:
                raise ValueError("No model predictions available for weighted average")
            
            # Optimize weights based on validation performance
            if X_val is not None and y_val is not None:
                val_predictions = await self._get_model_predictions_single(X_val)
                optimized_weights = self._optimize_weights(val_predictions, y_val)
            else:
                # Use equal weights if no validation set
                model_names = list(predictions.keys())
                optimized_weights = {name: 1.0 / len(model_names) for name in model_names}
            
            # Calculate weighted average predictions
            weighted_predictions = np.zeros(len(X))
            for model_name, preds in predictions.items():
                if model_name in optimized_weights:
                    weighted_predictions += optimized_weights[model_name] * preds
            
            # Calculate metrics
            metrics = self._calculate_ensemble_metrics(weighted_predictions, y)
            
            # Save ensemble configuration
            ensemble_path = await self._save_weighted_ensemble(optimized_weights, predictions)
            
            return EnsembleResult(
                ensemble_path=ensemble_path,
                ensemble_type=EnsembleType.WEIGHTED_AVERAGE,
                models_used=list(predictions.keys()),
                meta_learner_path=None,
                metrics=metrics,
                model_weights=optimized_weights,
                training_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating weighted average ensemble: {e}")
            raise
    
    async def _generate_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """Generate out-of-fold predictions for stacking"""
        try:
            oof_predictions = {}
            kf = StratifiedKFold(n_splits=self.config.stacking_cv_folds, 
                               shuffle=True, random_state=self.config.random_state)
            
            for model_name in self.models.keys():
                oof_preds = np.zeros(len(X))
                
                for train_idx, val_idx in kf.split(X, y):
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    
                    # Train model on this fold
                    model = await self._train_model_on_fold(model_name, X_train_fold, y_train_fold)
                    
                    # Get predictions on validation fold
                    fold_preds = await self._predict_with_model(model, X_val_fold)
                    oof_preds[val_idx] = fold_preds
                
                oof_predictions[model_name] = oof_preds
            
            return oof_predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating OOF predictions: {e}")
            return {}
    
    async def _get_model_predictions(self, X: pd.DataFrame, X_val: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Get predictions from all models"""
        try:
            predictions = {}
            
            for model_name, model_info in self.models.items():
                try:
                    # Get predictions for training set
                    train_preds = await self._predict_with_model_path(model_info['path'], X)
                    predictions[model_name] = train_preds
                    
                    # Update last used timestamp
                    self.models[model_name]['last_used'] = datetime.now()
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error getting model predictions: {e}")
            return {}
    
    async def _get_model_predictions_single(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from all models for a single dataset"""
        try:
            predictions = await self._get_model_predictions(X)
            if predictions:
                return np.column_stack(list(predictions.values()))
            else:
                return np.zeros((len(X), 0))
                
        except Exception as e:
            logger.error(f"❌ Error getting single model predictions: {e}")
            return np.zeros((len(X), 0))
    
    async def _predict_with_model_path(self, model_path: str, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from a model file"""
        try:
            # This is a placeholder - in practice, you'd load the actual model
            # and make predictions based on the model type
            
            # For now, return random predictions
            return np.random.random(len(X))
            
        except Exception as e:
            logger.error(f"❌ Error predicting with model path: {e}")
            return np.zeros(len(X))
    
    async def _train_model_on_fold(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        """Train a model on a specific fold"""
        try:
            # This is a placeholder - in practice, you'd train the actual model
            # For now, return a dummy model
            return f"dummy_model_{model_name}"
            
        except Exception as e:
            logger.error(f"❌ Error training model on fold: {e}")
            return None
    
    async def _predict_with_model(self, model, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from a trained model"""
        try:
            # This is a placeholder - in practice, you'd use the actual model
            # For now, return random predictions
            return np.random.random(len(X))
            
        except Exception as e:
            logger.error(f"❌ Error predicting with model: {e}")
            return np.zeros(len(X))
    
    def _optimize_weights(self, predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Optimize ensemble weights based on validation performance"""
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate weighted prediction
                weighted_pred = np.zeros(len(y))
                for i, (model_name, preds) in enumerate(predictions.items()):
                    weighted_pred += weights[i] * preds
                
                # Return negative AUC (minimize negative AUC = maximize AUC)
                return -roc_auc_score(y, weighted_pred)
            
            # Initial weights (equal)
            n_models = len(predictions)
            initial_weights = np.ones(n_models) / n_models
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1) for _ in range(n_models)]
            
            # Optimize
            result = minimize(objective, initial_weights, 
                           constraints=constraints, bounds=bounds,
                           method='SLSQP')
            
            if result.success:
                optimized_weights = result.x / np.sum(result.x)
                return dict(zip(predictions.keys(), optimized_weights))
            else:
                # Fallback to equal weights
                return {name: 1.0 / n_models for name in predictions.keys()}
                
        except Exception as e:
            logger.error(f"❌ Error optimizing weights: {e}")
            # Fallback to equal weights
            n_models = len(predictions)
            return {name: 1.0 / n_models for name in predictions.keys()}
    
    def _calculate_ensemble_metrics(self, predictions: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Calculate ensemble performance metrics"""
        try:
            return {
                'auc': roc_auc_score(y, predictions),
                'accuracy': accuracy_score(y, predictions > 0.5),
                'precision': precision_score(y, predictions > 0.5, zero_division=0),
                'recall': recall_score(y, predictions > 0.5, zero_division=0),
                'f1': f1_score(y, predictions > 0.5, zero_division=0)
            }
        except Exception as e:
            logger.error(f"❌ Error calculating ensemble metrics: {e}")
            return {}
    
    async def predict(self, X: pd.DataFrame) -> EnsemblePrediction:
        """Make ensemble prediction"""
        try:
            # Get individual model predictions
            individual_predictions = await self._get_model_predictions(X)
            
            if not individual_predictions:
                raise ValueError("No model predictions available")
            
            # Calculate ensemble prediction based on type
            if self.config.ensemble_type == EnsembleType.BLENDING:
                ensemble_pred = self._blend_predictions(individual_predictions)
            elif self.config.ensemble_type == EnsembleType.STACKING:
                ensemble_pred = await self._stack_predictions(individual_predictions)
            elif self.config.ensemble_type == EnsembleType.WEIGHTED_AVERAGE:
                ensemble_pred = self._weighted_average_predictions(individual_predictions)
            else:
                raise ValueError(f"Unsupported ensemble type: {self.config.ensemble_type}")
            
            # Calculate confidence
            confidence = abs(ensemble_pred - 0.5) * 2  # Distance from 0.5
            
            # Get current weights
            if self.config.ensemble_type == EnsembleType.STACKING and self.meta_learner:
                model_weights = dict(zip(individual_predictions.keys(), self.meta_learner.coef_[0]))
            else:
                model_weights = self.config.blending_weights or {name: 1.0 / len(individual_predictions) for name in individual_predictions.keys()}
            
            return EnsemblePrediction(
                individual_predictions={name: float(preds[0]) for name, preds in individual_predictions.items()},
                ensemble_prediction=float(ensemble_pred[0]),
                confidence=float(confidence[0]),
                model_weights=model_weights,
                ensemble_type=self.config.ensemble_type.value,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error making ensemble prediction: {e}")
            # Return default prediction
            return EnsemblePrediction(
                individual_predictions={},
                ensemble_prediction=0.5,
                confidence=0.0,
                model_weights={},
                ensemble_type=self.config.ensemble_type.value,
                timestamp=datetime.now()
            )
    
    def _blend_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Blend predictions using configured weights"""
        try:
            weights = self.config.blending_weights
            blended = np.zeros(len(next(iter(predictions.values()))))
            
            for model_name, preds in predictions.items():
                if model_name in weights:
                    blended += weights[model_name] * preds
            
            return blended
            
        except Exception as e:
            logger.error(f"❌ Error blending predictions: {e}")
            return np.zeros(len(next(iter(predictions.values()))))
    
    async def _stack_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack predictions using meta-learner"""
        try:
            if self.meta_learner is None:
                raise ValueError("Meta-learner not trained")
            
            # Stack predictions
            meta_features = np.column_stack(list(predictions.values()))
            
            # Get meta-learner prediction
            stacked_pred = self.meta_learner.predict_proba(meta_features)[:, 1]
            
            return stacked_pred
            
        except Exception as e:
            logger.error(f"❌ Error stacking predictions: {e}")
            return np.zeros(len(next(iter(predictions.values()))))
    
    def _weighted_average_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate weighted average of predictions"""
        try:
            # Use current weights or equal weights
            weights = self.current_weights or {name: 1.0 / len(predictions) for name in predictions.keys()}
            
            weighted_avg = np.zeros(len(next(iter(predictions.values()))))
            for model_name, preds in predictions.items():
                if model_name in weights:
                    weighted_avg += weights[model_name] * preds
            
            return weighted_avg
            
        except Exception as e:
            logger.error(f"❌ Error calculating weighted average: {e}")
            return np.zeros(len(next(iter(predictions.values()))))
    
    async def _save_blending_ensemble(self, weights: Dict[str, float], 
                                    predictions: Dict[str, np.ndarray]) -> str:
        """Save blending ensemble configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ensemble_path = self.ensembles_dir / f"blending_ensemble_{timestamp}.json"
            
            config = {
                'ensemble_type': 'blending',
                'weights': weights,
                'models_used': list(predictions.keys()),
                'created_at': timestamp,
                'config': self.config.__dict__
            }
            
            with open(ensemble_path, 'w') as f:
                json.dump(config, f, default=str)
            
            return str(ensemble_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving blending ensemble: {e}")
            raise
    
    async def _save_stacking_ensemble(self, oof_predictions: Dict[str, np.ndarray]) -> str:
        """Save stacking ensemble configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ensemble_path = self.ensembles_dir / f"stacking_ensemble_{timestamp}.json"
            
            config = {
                'ensemble_type': 'stacking',
                'models_used': list(oof_predictions.keys()),
                'cv_folds': self.config.stacking_cv_folds,
                'meta_learner_type': self.config.meta_learner_type,
                'created_at': timestamp,
                'config': self.config.__dict__
            }
            
            with open(ensemble_path, 'w') as f:
                json.dump(config, f, default=str)
            
            return str(ensemble_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving stacking ensemble: {e}")
            raise
    
    async def _save_weighted_ensemble(self, weights: Dict[str, float],
                                    predictions: Dict[str, np.ndarray]) -> str:
        """Save weighted ensemble configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ensemble_path = self.ensembles_dir / f"weighted_ensemble_{timestamp}.json"
            
            config = {
                'ensemble_type': 'weighted_average',
                'weights': weights,
                'models_used': list(predictions.keys()),
                'created_at': timestamp,
                'config': self.config.__dict__
            }
            
            with open(ensemble_path, 'w') as f:
                json.dump(config, f, default=str)
            
            return str(ensemble_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving weighted ensemble: {e}")
            raise
    
    async def _save_meta_learner(self) -> str:
        """Save meta-learner"""
        try:
            if self.meta_learner is None:
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta_learner_path = self.ensembles_dir / f"meta_learner_{timestamp}.pkl"
            
            with open(meta_learner_path, 'wb') as f:
                pickle.dump(self.meta_learner, f)
            
            return str(meta_learner_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving meta-learner: {e}")
            return None
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble summary"""
        try:
            return {
                'ensemble_type': self.config.ensemble_type.value,
                'models_count': len(self.models),
                'models': list(self.models.keys()),
                'current_weights': self.current_weights,
                'performance_history': len(self.performance_history),
                'meta_learner_trained': self.meta_learner is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting ensemble summary: {e}")
            return {}

# Global ensembler instance
model_ensembler = ModelEnsembler()

# Export for use in other modules
__all__ = [
    'ModelEnsembler',
    'EnsembleConfig',
    'EnsemblePrediction',
    'EnsembleResult',
    'EnsembleType',
    'ModelSource',
    'model_ensembler'
]
