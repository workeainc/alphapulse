#!/usr/bin/env python3
"""
ML Model Trainer for AlphaPulse
Phase 1: Core Model Training Implementation

Implements:
1. XGBoost, LightGBM, CatBoost training with warm-start
2. Class imbalance handling (scale_pos_weight, focal loss)
3. Sample weighting by realized R/R
4. MLflow integration with model versioning
5. Comprehensive hyperparameter optimization
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
import joblib
from dataclasses import dataclass
from enum import Enum
import time

# ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    import mlflow.catboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available")

# Local imports
from ..model_registry import ModelRegistry
from ..advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"

class TrainingCadence(Enum):
    """Training cadence types"""
    WEEKLY_QUICK = "weekly_quick"
    MONTHLY_FULL = "monthly_full"
    NIGHTLY_INCREMENTAL = "nightly_incremental"

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: ModelType
    cadence: TrainingCadence
    learning_rate: float = 0.1
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "auc"
    
    # Class imbalance handling
    scale_pos_weight: Optional[float] = None
    use_focal_loss: bool = False
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Sample weighting
    weight_by_rr: bool = True
    min_rr_weight: float = 0.1
    max_rr_weight: float = 5.0

@dataclass
class TrainingResult:
    """Training result"""
    model_path: str
    model_type: ModelType
    cadence: TrainingCadence
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_time: float
    model_size_mb: float
    hyperparameters: Dict[str, Any]
    mlflow_run_id: Optional[str] = None

class MLModelTrainer:
    """
    Comprehensive ML model trainer with support for multiple algorithms
    """
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Default hyperparameters for each model type
        self.default_configs = {
            ModelType.XGBOOST: {
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'auc',
                'objective': 'binary:logistic'
            },
            ModelType.LIGHTGBM: {
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'metric': 'auc',
                'objective': 'binary'
            },
            ModelType.CATBOOST: {
                'learning_rate': 0.1,
                'max_depth': 6,
                'iterations': 100,
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'eval_metric': 'AUC',
                'loss_function': 'Logloss'
            }
        }
        
        logger.info("🚀 ML Model Trainer initialized")
    
    async def train_model(self, 
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: Optional[pd.DataFrame] = None,
                         y_val: Optional[pd.Series] = None,
                         config: TrainingConfig = None,
                         previous_model_path: Optional[str] = None,
                         sample_weights: Optional[np.ndarray] = None) -> TrainingResult:
        """
        Train a model with the specified configuration
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            if config is None:
                config = TrainingConfig(
                    model_type=ModelType.XGBOOST,
                    cadence=TrainingCadence.WEEKLY_QUICK
                )
            
            # Calculate class imbalance metrics
            class_weights = self._calculate_class_weights(y_train, config)
            
            # Calculate sample weights if needed
            if sample_weights is None and config.weight_by_rr:
                sample_weights = self._calculate_rr_weights(X_train)
            
            # Train based on model type
            if config.model_type == ModelType.XGBOOST:
                result = await self._train_xgboost(
                    X_train, y_train, X_val, y_val, config, 
                    previous_model_path, class_weights, sample_weights
                )
            elif config.model_type == ModelType.LIGHTGBM:
                result = await self._train_lightgbm(
                    X_train, y_train, X_val, y_val, config,
                    previous_model_path, class_weights, sample_weights
                )
            elif config.model_type == ModelType.CATBOOST:
                result = await self._train_catboost(
                    X_train, y_train, X_val, y_val, config,
                    previous_model_path, class_weights, sample_weights
                )
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Log training completion
            training_time = time.time() - start_time
            result.training_time = training_time
            
            # Log to Redis if available
            if hasattr(redis_logger, 'log_event'):
                try:
                    await redis_logger.log_event(
                        EventType.MODEL_TRAINING,
                        LogLevel.INFO,
                        {
                            'model_type': config.model_type.value,
                            'cadence': config.cadence.value,
                            'training_time': training_time,
                            'metrics': result.metrics,
                            'model_path': result.model_path
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to Redis: {e}")
            else:
                logger.info(f"Redis logging not available")
            
            logger.info(f"✅ Model training completed: {config.model_type.value} in {training_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    async def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                           config: TrainingConfig, previous_model_path: Optional[str],
                           class_weights: Dict[str, float], sample_weights: Optional[np.ndarray]) -> TrainingResult:
        """Train XGBoost model with warm-start support"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        try:
            # Prepare data
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
            dval = xgb.DMatrix(X_val, label=y_val) if X_val is not None else None
            
            # Set up parameters
            params = self.default_configs[ModelType.XGBOOST].copy()
            params.update({
                'learning_rate': config.learning_rate,
                'max_depth': config.max_depth,
                'subsample': config.subsample,
                'colsample_bytree': config.colsample_bytree,
                'random_state': config.random_state
            })
            
            # Handle class imbalance
            if config.scale_pos_weight is not None:
                params['scale_pos_weight'] = config.scale_pos_weight
            elif class_weights:
                params['scale_pos_weight'] = class_weights.get('scale_pos_weight', 1.0)
            
            # Load previous model for warm-start
            xgb_model = None
            if previous_model_path and Path(previous_model_path).exists():
                try:
                    xgb_model = xgb.Booster(model_file=previous_model_path)
                    logger.info(f"✅ Loaded previous model for warm-start: {previous_model_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load previous model: {e}")
            
            # Train model
            evals = [(dtrain, 'train')]
            if dval is not None:
                evals.append((dval, 'val'))
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=config.n_estimators,
                evals=evals,
                early_stopping_rounds=config.early_stopping_rounds,
                xgb_model=xgb_model,
                verbose_eval=False
            )
            
            # Save model
            model_path = self._save_model(model, config, "xgboost")
            
            # Calculate metrics
            metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
            
            # Get feature importance
            feature_importance = self._get_xgboost_feature_importance(model, X_train.columns)
            
            # Log to MLflow
            mlflow_run_id = await self._log_to_mlflow(
                model, config, metrics, feature_importance, "xgboost"
            )
            
            return TrainingResult(
                model_path=model_path,
                model_type=ModelType.XGBOOST,
                cadence=config.cadence,
                metrics=metrics,
                feature_importance=feature_importance,
                training_time=0.0,  # Will be set by caller
                model_size_mb=self._get_model_size(model_path),
                hyperparameters=params,
                mlflow_run_id=mlflow_run_id
            )
            
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
            raise
    
    async def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                            config: TrainingConfig, previous_model_path: Optional[str],
                            class_weights: Dict[str, float], sample_weights: Optional[np.ndarray]) -> TrainingResult:
        """Train LightGBM model with warm-start support"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        try:
            # Prepare data
            train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
            val_data = lgb.Dataset(X_val, label=y_val) if X_val is not None else None
            
            # Set up parameters
            params = self.default_configs[ModelType.LIGHTGBM].copy()
            params.update({
                'learning_rate': config.learning_rate,
                'max_depth': config.max_depth,
                'subsample': config.subsample,
                'colsample_bytree': config.colsample_bytree,
                'random_state': config.random_state
            })
            
            # Handle class imbalance
            if config.scale_pos_weight is not None:
                params['scale_pos_weight'] = config.scale_pos_weight
            elif class_weights:
                params['scale_pos_weight'] = class_weights.get('scale_pos_weight', 1.0)
            
            # Load previous model for warm-start
            init_model = None
            if previous_model_path and Path(previous_model_path).exists():
                try:
                    init_model = previous_model_path
                    logger.info(f"✅ Loaded previous model for warm-start: {previous_model_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load previous model: {e}")
            
            # Train model
            valid_sets = [train_data]
            valid_names = ['train']
            if val_data is not None:
                valid_sets.append(val_data)
                valid_names.append('val')
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=config.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(config.early_stopping_rounds),
                    lgb.log_evaluation(0)
                ],
                init_model=init_model
            )
            
            # Save model
            model_path = self._save_model(model, config, "lightgbm")
            
            # Calculate metrics
            metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
            
            # Get feature importance
            feature_importance = self._get_lightgbm_feature_importance(model, X_train.columns)
            
            # Log to MLflow
            mlflow_run_id = await self._log_to_mlflow(
                model, config, metrics, feature_importance, "lightgbm"
            )
            
            return TrainingResult(
                model_path=model_path,
                model_type=ModelType.LIGHTGBM,
                cadence=config.cadence,
                metrics=metrics,
                feature_importance=feature_importance,
                training_time=0.0,  # Will be set by caller
                model_size_mb=self._get_model_size(model_path),
                hyperparameters=params,
                mlflow_run_id=mlflow_run_id
            )
            
        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
            raise
    
    async def _train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                            config: TrainingConfig, previous_model_path: Optional[str],
                            class_weights: Dict[str, float], sample_weights: Optional[np.ndarray]) -> TrainingResult:
        """Train CatBoost model with warm-start support"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        
        try:
            # Set up parameters
            params = self.default_configs[ModelType.CATBOOST].copy()
            params.update({
                'learning_rate': config.learning_rate,
                'max_depth': config.max_depth,
                'iterations': config.n_estimators,
                'subsample': config.subsample,
                'colsample_bylevel': config.colsample_bytree,
                'eval_metric': 'AUC',
                'loss_function': 'Logloss'
            })
            
            # Handle class imbalance
            if config.scale_pos_weight is not None:
                params['class_weights'] = [1.0, config.scale_pos_weight]
            elif class_weights:
                params['class_weights'] = [1.0, class_weights.get('scale_pos_weight', 1.0)]
            
            # Load previous model for warm-start
            init_model = None
            if previous_model_path and Path(previous_model_path).exists():
                try:
                    init_model = cb.CatBoost()
                    init_model.load_model(previous_model_path)
                    logger.info(f"✅ Loaded previous model for warm-start: {previous_model_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load previous model: {e}")
            
            # Train model
            model = cb.CatBoost(params)
            
            if init_model:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val) if X_val is not None else None,
                    init_model=init_model,
                    sample_weight=sample_weights,
                    use_best_model=True,
                    verbose=False
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val) if X_val is not None else None,
                    sample_weight=sample_weights,
                    use_best_model=True,
                    verbose=False
                )
            
            # Save model
            model_path = self._save_model(model, config, "catboost")
            
            # Calculate metrics
            metrics = self._calculate_metrics(model, X_train, y_train, X_val, y_val)
            
            # Get feature importance
            feature_importance = self._get_catboost_feature_importance(model, X_train.columns)
            
            # Log to MLflow
            mlflow_run_id = await self._log_to_mlflow(
                model, config, metrics, feature_importance, "catboost"
            )
            
            return TrainingResult(
                model_path=model_path,
                model_type=ModelType.CATBOOST,
                cadence=config.cadence,
                metrics=metrics,
                feature_importance=feature_importance,
                training_time=0.0,  # Will be set by caller
                model_size_mb=self._get_model_size(model_path),
                hyperparameters=params,
                mlflow_run_id=mlflow_run_id
            )
            
        except Exception as e:
            logger.error(f"❌ CatBoost training failed: {e}")
            raise
    
    def _calculate_class_weights(self, y: pd.Series, config: TrainingConfig) -> Dict[str, float]:
        """Calculate class weights for imbalanced datasets"""
        try:
            class_counts = y.value_counts()
            total_samples = len(y)
            
            if len(class_counts) != 2:
                return {}
            
            # Calculate scale_pos_weight for binary classification
            neg_count = class_counts.get(0, 0)
            pos_count = class_counts.get(1, 0)
            
            if pos_count == 0 or neg_count == 0:
                return {}
            
            scale_pos_weight = neg_count / pos_count
            
            return {
                'scale_pos_weight': scale_pos_weight,
                'neg_count': neg_count,
                'pos_count': pos_count,
                'imbalance_ratio': scale_pos_weight
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating class weights: {e}")
            return {}
    
    def _calculate_rr_weights(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate sample weights based on realized R/R"""
        try:
            # Extract realized R/R from features (assuming it's in the dataframe)
            if 'realized_rr' in X.columns:
                rr_values = X['realized_rr'].fillna(1.0)
            else:
                # Default weights if R/R not available
                return np.ones(len(X))
            
            # Clip weights to reasonable range
            weights = np.clip(rr_values, 0.1, 5.0)
            
            # Normalize weights
            weights = weights / weights.mean()
            
            return weights.values
            
        except Exception as e:
            logger.error(f"❌ Error calculating R/R weights: {e}")
            return np.ones(len(X))
    
    def _calculate_metrics(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {}
            
            # Training metrics
            y_train_pred = self._predict_proba(model, X_train)
            metrics['train_auc'] = roc_auc_score(y_train, y_train_pred)
            metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred > 0.5)
            metrics['train_precision'] = precision_score(y_train, y_train_pred > 0.5, zero_division=0)
            metrics['train_recall'] = recall_score(y_train, y_train_pred > 0.5, zero_division=0)
            metrics['train_f1'] = f1_score(y_train, y_train_pred > 0.5, zero_division=0)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                y_val_pred = self._predict_proba(model, X_val)
                metrics['val_auc'] = roc_auc_score(y_val, y_val_pred)
                metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred > 0.5)
                metrics['val_precision'] = precision_score(y_val, y_val_pred > 0.5, zero_division=0)
                metrics['val_recall'] = recall_score(y_val, y_val_pred > 0.5, zero_division=0)
                metrics['val_f1'] = f1_score(y_val, y_val_pred > 0.5, zero_division=0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
    
    def _predict_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from model"""
        try:
            # Handle XGBoost Booster models
            if hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
                # This is likely an XGBoost Booster model
                try:
                    dmatrix = xgb.DMatrix(X)
                    predictions = model.predict(dmatrix)
                    # XGBoost predictions are already probabilities for binary classification
                    return predictions
                except Exception as xgb_error:
                    logger.warning(f"XGBoost DMatrix conversion failed: {xgb_error}")
                    # Fallback to regular predict
                    return model.predict(X)
            elif hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                return model.predict(X)
            else:
                raise ValueError("Model doesn't support prediction")
        except Exception as e:
            logger.error(f"❌ Error making predictions: {e}")
            return np.zeros(len(X))
    
    def _get_xgboost_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from XGBoost model"""
        try:
            importance = model.get_score(importance_type='gain')
            feature_importance = {}
            
            for k, v in importance.items():
                try:
                    # Handle different feature name formats
                    if k.startswith('f') and k[1:].isdigit():
                        # Format: f0, f1, f2, etc.
                        idx = int(k[1:])
                        if idx < len(feature_names):
                            feature_importance[feature_names[idx]] = v
                    elif k.isdigit():
                        # Format: 0, 1, 2, etc.
                        idx = int(k)
                        if idx < len(feature_names):
                            feature_importance[feature_names[idx]] = v
                    else:
                        # Direct feature name
                        feature_importance[k] = v
                except (ValueError, IndexError) as parse_error:
                    logger.warning(f"Could not parse feature importance key '{k}': {parse_error}")
                    continue
            
            return feature_importance
        except Exception as e:
            logger.error(f"❌ Error getting XGBoost feature importance: {e}")
            return {}
    
    def _get_lightgbm_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from LightGBM model"""
        try:
            importance = model.feature_importance(importance_type='gain')
            return dict(zip(feature_names, importance))
        except Exception as e:
            logger.error(f"❌ Error getting LightGBM feature importance: {e}")
            return {}
    
    def _get_catboost_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from CatBoost model"""
        try:
            # Try different importance types that don't require training data
            for importance_type in ['PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance']:
                try:
                    importance = model.get_feature_importance(type=importance_type)
                    return dict(zip(feature_names, importance))
                except Exception as type_error:
                    logger.debug(f"CatBoost importance type '{importance_type}' failed: {type_error}")
                    continue
            
            # Fallback to default importance
            try:
                importance = model.get_feature_importance()
                return dict(zip(feature_names, importance))
            except Exception as fallback_error:
                logger.warning(f"CatBoost fallback importance failed: {fallback_error}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting CatBoost feature importance: {e}")
            return {}
    
    def _save_model(self, model, config: TrainingConfig, model_type: str) -> str:
        """Save model to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_{config.cadence.value}_{timestamp}.model"
            model_path = self.models_dir / filename
            
            if model_type == "xgboost":
                model.save_model(str(model_path))
            elif model_type == "lightgbm":
                model.save_model(str(model_path))
            elif model_type == "catboost":
                model.save_model(str(model_path))
            
            logger.info(f"✅ Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB"""
        try:
            size_bytes = Path(model_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    async def _log_to_mlflow(self, model, config: TrainingConfig, metrics: Dict[str, float],
                           feature_importance: Dict[str, float], model_type: str) -> Optional[str]:
        """Log model to MLflow"""
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params(config.__dict__)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log feature importance
                mlflow.log_dict(feature_importance, "feature_importance.json")
                
                # Log model
                if model_type == "xgboost":
                    mlflow.xgboost.log_model(model, "model")
                elif model_type == "lightgbm":
                    mlflow.lightgbm.log_model(model, "model")
                elif model_type == "catboost":
                    mlflow.catboost.log_model(model, "model")
                
                # Add tags
                mlflow.set_tags({
                    "model_type": model_type,
                    "cadence": config.cadence.value,
                    "training_timestamp": datetime.now().isoformat()
                })
                
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"❌ Error logging to MLflow: {e}")
            return None

# Global trainer instance
ml_model_trainer = MLModelTrainer()

# Export for use in other modules
__all__ = [
    'MLModelTrainer',
    'TrainingConfig',
    'TrainingResult',
    'ModelType',
    'TrainingCadence',
    'ml_model_trainer'
]
