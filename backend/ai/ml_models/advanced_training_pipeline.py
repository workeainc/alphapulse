"""
Advanced ML Training Pipeline for AlphaPlus
Handles model training, validation, and hyperparameter optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import asyncio
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import catboost as cb
import xgboost as xgb
import lightgbm as lgb

# Import our enhanced components
try:
    from ..advanced_feature_engineering import AdvancedFeatureEngineering
    from .model_registry import ModelRegistry, ModelType, ModelStatus
except ImportError:
    # Fallback for testing
    AdvancedFeatureEngineering = None
    ModelRegistry = None
    ModelType = None
    ModelStatus = None

logger = logging.getLogger(__name__)

class TrainingStatus(Enum):
    """Training status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    test_size: float
    validation_size: float
    random_state: int
    hyperparameter_grid: Dict[str, List[Any]]
    cv_folds: int
    scoring_metric: str
    early_stopping_rounds: int
    max_training_time: int  # seconds

@dataclass
class TrainingResult:
    """Training result"""
    model_id: str
    training_status: TrainingStatus
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    best_hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_time: float
    model_size: float
    created_at: datetime

class AdvancedTrainingPipeline:
    """Advanced ML training pipeline with hyperparameter optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Training configuration
        self.default_test_size = self.config.get('default_test_size', 0.2)
        self.default_validation_size = self.config.get('default_validation_size', 0.1)
        self.default_cv_folds = self.config.get('default_cv_folds', 5)
        self.default_scoring = self.config.get('default_scoring', 'neg_mean_squared_error')
        
        # Component references
        self.feature_engineering = None
        self.model_registry = None
        
        # Training state
        self.active_trainings: Dict[str, asyncio.Task] = {}
        self.training_results: Dict[str, TrainingResult] = {}
        
        # Performance tracking
        self.stats = {
            'total_trainings': 0,
            'successful_trainings': 0,
            'failed_trainings': 0,
            'total_training_time': 0.0
        }
        
    async def initialize(self):
        """Initialize the training pipeline"""
        try:
            self.logger.info("Initializing Advanced ML Training Pipeline...")
            
            # Initialize feature engineering
            if AdvancedFeatureEngineering:
                self.feature_engineering = AdvancedFeatureEngineering(
                    self.config.get('feature_engineering_config', {})
                )
                await self.feature_engineering.initialize()
            
            # Initialize model registry
            if ModelRegistry:
                self.model_registry = ModelRegistry(
                    self.config.get('model_registry_config', {})
                )
                await self.model_registry.initialize()
            
            self.logger.info("Advanced ML Training Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Training Pipeline: {e}")
            raise
    
    async def train_model(self, training_config: TrainingConfig, 
                         training_data: pd.DataFrame) -> str:
        """Start model training"""
        try:
            # Validate training data
            if training_data.empty:
                raise ValueError("Training data is empty")
            
            if training_config.target_column not in training_data.columns:
                raise ValueError(f"Target column '{training_config.target_column}' not found in data")
            
            # Register model in registry
            model_id = await self.model_registry.register_model(
                name=training_config.model_name,
                model_type=training_config.model_type,
                description=f"Auto-trained {training_config.model_type.value} model",
                author="training_pipeline",
                tags=["auto_trained", training_config.model_type.value],
                hyperparameters=training_config.hyperparameter_grid
            )
            
            # Start training task
            training_task = asyncio.create_task(
                self._train_model_task(model_id, training_config, training_data)
            )
            
            # Track active training
            self.active_trainings[model_id] = training_task
            
            self.logger.info(f"Started training for model: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error starting training: {e}")
            raise
    
    async def _train_model_task(self, model_id: str, training_config: TrainingConfig, 
                               training_data: pd.DataFrame):
        """Main training task"""
        try:
            start_time = datetime.now()
            
            # Update model status
            await self.model_registry.update_model_metadata(
                model_id, {'status': ModelStatus.TRAINING}
            )
            
            # Prepare data
            X, y = await self._prepare_training_data(training_data, training_config)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = await self._split_data(
                X, y, training_config
            )
            
            # Create and train model
            model = await self._create_and_train_model(
                training_config, X_train, y_train, X_val, y_val
            )
            
            # Evaluate model
            training_metrics = await self._evaluate_model(model, X_train, y_train)
            validation_metrics = await self._evaluate_model(model, X_val, y_val)
            test_metrics = await self._evaluate_model(model, X_test, y_test)
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(
                model, training_config.feature_columns
            )
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save model to registry
            await self.model_registry.save_model(
                model_id=model_id,
                model_object=model,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                feature_importance=feature_importance
            )
            
            # Create training result
            training_result = TrainingResult(
                model_id=model_id,
                training_status=TrainingStatus.COMPLETED,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                best_hyperparameters=model.get_params() if hasattr(model, 'get_params') else {},
                feature_importance=feature_importance,
                training_time=training_time,
                model_size=await self._get_model_size(model),
                created_at=datetime.now(timezone.utc)
            )
            
            # Store result
            self.training_results[model_id] = training_result
            
            # Update statistics
            self.stats['total_trainings'] += 1
            self.stats['successful_trainings'] += 1
            self.stats['total_training_time'] += training_time
            
            # Remove from active trainings
            if model_id in self.active_trainings:
                del self.active_trainings[model_id]
            
            self.logger.info(f"Training completed successfully for model: {model_id}")
            
        except Exception as e:
            self.logger.error(f"Training failed for model {model_id}: {e}")
            
            # Update model status
            await self.model_registry.update_model_metadata(
                model_id, {'status': ModelStatus.FAILED}
            )
            
            # Store failed result
            training_result = TrainingResult(
                model_id=model_id,
                training_status=TrainingStatus.FAILED,
                training_metrics={},
                validation_metrics={},
                best_hyperparameters={},
                feature_importance={},
                training_time=0.0,
                model_size=0.0,
                created_at=datetime.now(timezone.utc)
            )
            
            self.training_results[model_id] = training_result
            
            # Update statistics
            self.stats['total_trainings'] += 1
            self.stats['failed_trainings'] += 1
            
            # Remove from active trainings
            if model_id in self.active_trainings:
                del self.active_trainings[model_id]
    
    async def _prepare_training_data(self, data: pd.DataFrame, 
                                   training_config: TrainingConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with feature engineering"""
        try:
            # Create features if feature engineering is available
            if self.feature_engineering:
                # Get all available features
                available_features = list(self.feature_engineering.feature_definitions.keys())
                
                # Create features
                enhanced_data = await self.feature_engineering.create_features(
                    data, available_features
                )
                
                # Select feature columns
                feature_cols = [col for col in enhanced_data.columns 
                              if col in training_config.feature_columns or 
                              col in available_features]
                
                # Remove duplicates and ensure target column is not in features
                feature_cols = list(set(feature_cols) - {training_config.target_column})
                
                X = enhanced_data[feature_cols]
                y = enhanced_data[training_config.target_column]
            else:
                # Use original data without feature engineering
                feature_cols = [col for col in data.columns 
                              if col in training_config.feature_columns and 
                              col != training_config.target_column]
                
                X = data[feature_cols]
                y = data[training_config.target_column]
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            y = y.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove rows with any remaining NaN values
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    async def _split_data(self, X: pd.DataFrame, y: pd.Series, 
                          training_config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                 pd.DataFrame, pd.Series, 
                                                                 pd.Series, pd.Series]:
        """Split data into train, validation, and test sets"""
        try:
            total_size = len(X)
            test_size = int(total_size * training_config.test_size)
            val_size = int(total_size * training_config.validation_size)
            train_size = total_size - test_size - val_size
            
            # Time series split (chronological order)
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            
            X_val = X.iloc[train_size:train_size + val_size]
            y_val = y.iloc[train_size:train_size + val_size]
            
            X_test = X.iloc[train_size + val_size:]
            y_test = y.iloc[train_size + val_size:]
            
            self.logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise
    
    async def _create_and_train_model(self, training_config: TrainingConfig, 
                                    X_train: pd.DataFrame, y_train: pd.Series,
                                    X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """Create and train the model with hyperparameter optimization"""
        try:
            # Create base model
            base_model = self._create_base_model(training_config.model_type)
            
            # Hyperparameter optimization
            if training_config.hyperparameter_grid:
                best_model = await self._optimize_hyperparameters(
                    base_model, training_config, X_train, y_train, X_val, y_val
                )
            else:
                # Train with default parameters
                best_model = await self._train_single_model(
                    base_model, X_train, y_train, X_val, y_val
                )
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error creating and training model: {e}")
            raise
    
    def _create_base_model(self, model_type: ModelType) -> Any:
        """Create base model based on type"""
        try:
            if model_type == ModelType.CATBOOST:
                return cb.CatBoostRegressor(
                    verbose=False,
                    random_state=42,
                    loss_function='RMSE'
                )
            elif model_type == ModelType.XGBOOST:
                return xgb.XGBRegressor(
                    random_state=42,
                    objective='reg:squarederror',
                    eval_metric='rmse'
                )
            elif model_type == ModelType.LIGHTGBM:
                return lgb.LGBMRegressor(
                    random_state=42,
                    objective='regression',
                    metric='rmse',
                    verbose=-1
                )
            elif model_type == ModelType.ENSEMBLE:
                return RandomForestRegressor(
                    random_state=42,
                    n_estimators=100
                )
            else:
                # Default to Random Forest
                return RandomForestRegressor(
                    random_state=42,
                    n_estimators=100
                )
                
        except Exception as e:
            self.logger.error(f"Error creating base model: {e}")
            raise
    
    async def _optimize_hyperparameters(self, base_model: Any, training_config: TrainingConfig,
                                       X_train: pd.DataFrame, y_train: pd.Series,
                                       X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """Optimize hyperparameters using cross-validation"""
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=training_config.cv_folds)
            
            # Grid search or random search
            if len(training_config.hyperparameter_grid) <= 20:
                # Use grid search for small parameter spaces
                search = GridSearchCV(
                    base_model,
                    training_config.hyperparameter_grid,
                    cv=tscv,
                    scoring=training_config.scoring_metric,
                    n_jobs=-1,
                    verbose=0
                )
            else:
                # Use random search for large parameter spaces
                search = RandomizedSearchCV(
                    base_model,
                    training_config.hyperparameter_grid,
                    n_iter=20,  # Number of parameter combinations to try
                    cv=tscv,
                    scoring=training_config.scoring_metric,
                    n_jobs=-1,
                    verbose=0,
                    random_state=42
                )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Get best model
            best_model = search.best_estimator_
            
            # Final training with best parameters
            best_model = await self._train_single_model(
                best_model, X_train, y_train, X_val, y_val
            )
            
            self.logger.info(f"Hyperparameter optimization completed. Best score: {search.best_score_:.4f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error optimizing hyperparameters: {e}")
            raise
    
    async def _train_single_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """Train a single model with early stopping if supported"""
        try:
            # Check if model supports early stopping
            if hasattr(model, 'fit') and hasattr(model, 'set_params'):
                # Set validation data for early stopping
                if hasattr(model, 'eval_set'):
                    # Models like XGBoost, LightGBM support eval_set
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    # Standard fit
                    model.fit(X_train, y_train)
            else:
                # Fallback for models without fit method
                model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training single model: {e}")
            raise
    
    async def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {}
    
    async def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        try:
            feature_importance = {}
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for i, feature_name in enumerate(feature_names):
                    if i < len(importances):
                        feature_importance[feature_name] = float(importances[i])
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = model.coef_
                for i, feature_name in enumerate(feature_names):
                    if i < len(coefficients):
                        feature_importance[feature_name] = float(abs(coefficients[i]))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    async def _get_model_size(self, model: Any) -> float:
        """Get model size in MB"""
        try:
            # Serialize model to get size
            model_bytes = joblib.dumps(model)
            size_mb = len(model_bytes) / (1024 * 1024)
            return round(size_mb, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating model size: {e}")
            return 0.0
    
    async def cancel_training(self, model_id: str) -> bool:
        """Cancel an active training"""
        try:
            if model_id not in self.active_trainings:
                return False
            
            # Cancel the training task
            training_task = self.active_trainings[model_id]
            training_task.cancel()
            
            # Update model status
            await self.model_registry.update_model_metadata(
                model_id, {'status': ModelStatus.FAILED}
            )
            
            # Remove from active trainings
            del self.active_trainings[model_id]
            
            self.logger.info(f"Training cancelled for model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling training: {e}")
            return False
    
    async def get_training_status(self, model_id: str) -> Optional[TrainingStatus]:
        """Get training status for a model"""
        try:
            if model_id in self.active_trainings:
                return TrainingStatus.IN_PROGRESS
            elif model_id in self.training_results:
                return self.training_results[model_id].training_status
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return None
    
    async def get_training_results(self, model_id: str) -> Optional[TrainingResult]:
        """Get training results for a model"""
        try:
            return self.training_results.get(model_id)
            
        except Exception as e:
            self.logger.error(f"Error getting training results: {e}")
            return None
    
    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get training pipeline statistics"""
        try:
            stats = self.stats.copy()
            
            # Add active training info
            stats['active_trainings'] = len(self.active_trainings)
            stats['completed_trainings'] = len(self.training_results)
            
            # Add average training time
            if stats['successful_trainings'] > 0:
                stats['avg_training_time'] = stats['total_training_time'] / stats['successful_trainings']
            else:
                stats['avg_training_time'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for training pipeline"""
        try:
            health_status = {
                'status': 'healthy',
                'active_trainings': len(self.active_trainings),
                'feature_engineering_available': self.feature_engineering is not None,
                'model_registry_available': self.model_registry is not None
            }
            
            # Check feature engineering health
            if self.feature_engineering:
                try:
                    fe_health = await self.feature_engineering.health_check()
                    health_status['feature_engineering_health'] = fe_health
                    
                    if fe_health.get('status') != 'healthy':
                        health_status['status'] = 'degraded'
                        health_status['warnings'] = ['Feature engineering issues']
                except Exception as e:
                    health_status['feature_engineering_health'] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'
            
            # Check model registry health
            if self.model_registry:
                try:
                    mr_health = await self.model_registry.health_check()
                    health_status['model_registry_health'] = mr_health
                    
                    if mr_health.get('status') != 'healthy':
                        health_status['status'] = 'degraded'
                        if 'warnings' not in health_status:
                            health_status['warnings'] = []
                        health_status['warnings'].append('Model registry issues')
                except Exception as e:
                    health_status['model_registry_health'] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Close training pipeline"""
        try:
            # Cancel all active trainings
            for model_id in list(self.active_trainings.keys()):
                await self.cancel_training(model_id)
            
            if self.feature_engineering:
                await self.feature_engineering.close()
            
            if self.model_registry:
                await self.model_registry.close()
            
            self.logger.info("Training pipeline closed")
            
        except Exception as e:
            self.logger.error(f"Error closing training pipeline: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
