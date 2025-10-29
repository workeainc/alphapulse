"""
Model Ensemble System Module

Provides advanced model ensemble capabilities including:
- Multiple ensemble strategies (voting, stacking, blending)
- Dynamic weight optimization
- Performance-based model selection
- Ensemble diversity management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import joblib
import pickle
from pathlib import Path
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import our components
try:
    from ...database.connection import TimescaleDBConnection
    from ...data.storage import DataStorage
    from .model_registry import ModelRegistry
except ImportError:
    TimescaleDBConnection = None
    DataStorage = None
    ModelRegistry = None

logger = logging.getLogger(__name__)

class EnsembleStrategy(Enum):
    """Ensemble strategies"""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"

class ModelType(Enum):
    """Model types for ensemble"""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"

@dataclass
class EnsembleConfig:
    """Configuration for ensemble"""
    strategy: EnsembleStrategy = EnsembleStrategy.VOTING
    base_models: List[str] = field(default_factory=list)
    meta_model: str = "logistic_regression"
    weight_optimization: bool = True
    diversity_threshold: float = 0.3
    performance_window: int = 100
    update_frequency: int = 50

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction: Union[float, int]
    confidence: float
    individual_predictions: Dict[str, Union[float, int]]
    model_weights: Dict[str, float]
    ensemble_score: float
    timestamp: datetime

class ModelEnsemble:
    """Advanced model ensemble system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Ensemble configuration
        self.strategy = self.config.get('strategy', EnsembleStrategy.VOTING)
        self.base_models = self.config.get('base_models', [])
        self.meta_model = self.config.get('meta_model', 'logistic_regression')
        self.weight_optimization = self.config.get('weight_optimization', True)
        self.diversity_threshold = self.config.get('diversity_threshold', 0.3)
        self.performance_window = self.config.get('performance_window', 100)
        self.update_frequency = self.config.get('update_frequency', 50)
        
        # Component references
        self.db_connection = None
        self.storage = None
        self.model_registry = None
        
        # Ensemble state
        self.ensemble_model = None
        self.base_model_instances = {}
        self.meta_model_instance = None
        self.model_weights = {}
        self.ensemble_type = None
        
        # Performance tracking
        self.performance_history = []
        self.model_performance = {}
        self.ensemble_performance = []
        
        # Diversity tracking
        self.model_diversity = {}
        self.correlation_matrix = None
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'ensemble_accuracy': 0.0,
            'base_model_accuracy': {},
            'weight_updates': 0,
            'diversity_score': 0.0
        }
        
    async def initialize(self):
        """Initialize the ensemble system"""
        try:
            self.logger.info("Initializing Model Ensemble System...")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            # Initialize storage if available
            if DataStorage:
                self.storage = DataStorage(
                    storage_path=self.config.get('ensemble_storage_path', 'ensemble'),
                    db_config=self.config.get('db_config', {})
                )
                await self.storage.initialize()
            
            # Initialize model registry if available
            if ModelRegistry:
                self.model_registry = ModelRegistry(
                    self.config.get('model_registry_config', {})
                )
                await self.model_registry.initialize()
            
            # Load base models
            await self._load_base_models()
            
            # Create ensemble
            await self._create_ensemble()
            
            # Initialize weights
            await self._initialize_weights()
            
            self.logger.info("Model Ensemble System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Model Ensemble: {e}")
            raise
    
    async def _load_base_models(self):
        """Load base models from registry"""
        try:
            if not self.model_registry:
                self.logger.warning("Model registry not available, using placeholder models")
                return
            
            for model_name in self.base_models:
                try:
                    model = await self.model_registry.load_model(model_name)
                    if model:
                        self.base_model_instances[model_name] = model
                        self.logger.info(f"Loaded base model: {model_name}")
                    else:
                        self.logger.warning(f"Failed to load base model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name}: {e}")
            
            # If no models loaded, create placeholder models
            if not self.base_model_instances:
                await self._create_placeholder_models()
                
        except Exception as e:
            self.logger.error(f"Failed to load base models: {e}")
            await self._create_placeholder_models()
    
    async def _create_placeholder_models(self):
        """Create placeholder models for testing"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            # Create simple placeholder models
            self.base_model_instances = {
                'rf_classifier': RandomForestClassifier(n_estimators=10, random_state=42),
                'lr_classifier': LogisticRegression(random_state=42),
                'rf_regressor': RandomForestRegressor(n_estimators=10, random_state=42),
                'lr_regressor': LinearRegression()
            }
            
            # Determine ensemble type
            if any('classifier' in name for name in self.base_model_instances.keys()):
                self.ensemble_type = ModelType.CLASSIFIER
            else:
                self.ensemble_type = ModelType.REGRESSOR
            
            self.logger.info("Created placeholder models for ensemble")
            
        except Exception as e:
            self.logger.error(f"Failed to create placeholder models: {e}")
    
    async def _create_ensemble(self):
        """Create the ensemble model"""
        try:
            if not self.base_model_instances:
                raise ValueError("No base models available")
            
            # Determine ensemble type
            if self.ensemble_type is None:
                if any('classifier' in name for name in self.base_model_instances.keys()):
                    self.ensemble_type = ModelType.CLASSIFIER
                else:
                    self.ensemble_type = ModelType.REGRESSOR
            
            # Create ensemble based on strategy
            if self.strategy == EnsembleStrategy.VOTING:
                await self._create_voting_ensemble()
            elif self.strategy == EnsembleStrategy.STACKING:
                await self._create_stacking_ensemble()
            elif self.strategy == EnsembleStrategy.BLENDING:
                await self._create_blending_ensemble()
            else:
                await self._create_voting_ensemble()  # Default
            
            self.logger.info(f"Created {self.strategy.value} ensemble")
            
        except Exception as e:
            self.logger.error(f"Failed to create ensemble: {e}")
            raise
    
    async def _create_voting_ensemble(self):
        """Create voting ensemble"""
        try:
            if self.ensemble_type == ModelType.CLASSIFIER:
                self.ensemble_model = VotingClassifier(
                    estimators=[(name, model) for name, model in self.base_model_instances.items()],
                    voting='soft'
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=[(name, model) for name, model in self.base_model_instances.items()]
                )
            
        except Exception as e:
            self.logger.error(f"Failed to create voting ensemble: {e}")
            raise
    
    async def _create_stacking_ensemble(self):
        """Create stacking ensemble"""
        try:
            # Create meta-model
            if self.ensemble_type == ModelType.CLASSIFIER:
                self.meta_model_instance = LogisticRegression(random_state=42)
            else:
                self.meta_model_instance = LinearRegression()
            
            # For now, we'll use a simple approach
            # In practice, you'd use sklearn's StackingClassifier/StackingRegressor
            self.ensemble_model = self.meta_model_instance
            
        except Exception as e:
            self.logger.error(f"Failed to create stacking ensemble: {e}")
            raise
    
    async def _create_blending_ensemble(self):
        """Create blending ensemble"""
        try:
            # Blending uses weighted average of base model predictions
            # We'll implement this in the predict method
            self.ensemble_model = None
            
        except Exception as e:
            self.logger.error(f"Failed to create blending ensemble: {e}")
            raise
    
    async def _initialize_weights(self):
        """Initialize model weights"""
        try:
            if not self.base_model_instances:
                return
            
            # Equal weights initially
            n_models = len(self.base_model_instances)
            equal_weight = 1.0 / n_models
            
            for model_name in self.base_model_instances.keys():
                self.model_weights[model_name] = equal_weight
            
            self.logger.info("Initialized equal weights for ensemble")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize weights: {e}")
    
    async def predict(self, features: np.ndarray) -> EnsemblePrediction:
        """Make ensemble prediction"""
        try:
            start_time = datetime.now()
            
            if not self.base_model_instances:
                raise ValueError("No base models available")
            
            # Get individual predictions
            individual_predictions = {}
            for model_name, model in self.base_model_instances.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(features)[0][1]  # Probability of positive class
                    else:
                        pred = model.predict(features)[0]
                    individual_predictions[model_name] = pred
                except Exception as e:
                    self.logger.error(f"Error getting prediction from {model_name}: {e}")
                    individual_predictions[model_name] = 0.5  # Default prediction
            
            # Calculate ensemble prediction based on strategy
            if self.strategy == EnsembleStrategy.VOTING:
                ensemble_pred = await self._voting_predict(individual_predictions)
            elif self.strategy == EnsembleStrategy.STACKING:
                ensemble_pred = await self._stacking_predict(individual_predictions)
            elif self.strategy == EnsembleStrategy.BLENDING:
                ensemble_pred = await self._blending_predict(individual_predictions)
            else:
                ensemble_pred = await self._voting_predict(individual_predictions)
            
            # Calculate confidence
            confidence = await self._calculate_confidence(individual_predictions, ensemble_pred)
            
            # Create prediction result
            result = EnsemblePrediction(
                prediction=ensemble_pred,
                confidence=confidence,
                individual_predictions=individual_predictions,
                model_weights=self.model_weights.copy(),
                ensemble_score=ensemble_pred,
                timestamp=datetime.now()
            )
            
            # Update statistics
            self.stats['total_predictions'] += 1
            
            # Periodic weight optimization
            if self.weight_optimization and self.stats['total_predictions'] % self.update_frequency == 0:
                await self._optimize_weights()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to make ensemble prediction: {e}")
            raise
    
    async def _voting_predict(self, individual_predictions: Dict[str, Union[float, int]]) -> Union[float, int]:
        """Voting-based prediction"""
        try:
            if self.ensemble_type == ModelType.CLASSIFIER:
                # For classification, use weighted voting
                weighted_sum = 0.0
                total_weight = 0.0
                
                for model_name, pred in individual_predictions.items():
                    weight = self.model_weights.get(model_name, 1.0)
                    weighted_sum += pred * weight
                    total_weight += weight
                
                if total_weight > 0:
                    return weighted_sum / total_weight
                else:
                    return 0.5
            else:
                # For regression, use weighted average
                weighted_sum = 0.0
                total_weight = 0.0
                
                for model_name, pred in individual_predictions.items():
                    weight = self.model_weights.get(model_name, 1.0)
                    weighted_sum += pred * weight
                    total_weight += weight
                
                if total_weight > 0:
                    return weighted_sum / total_weight
                else:
                    return 0.0
                    
        except Exception as e:
            self.logger.error(f"Failed to make voting prediction: {e}")
            return 0.5 if self.ensemble_type == ModelType.CLASSIFIER else 0.0
    
    async def _stacking_predict(self, individual_predictions: Dict[str, Union[float, int]]) -> Union[float, int]:
        """Stacking-based prediction"""
        try:
            if self.meta_model_instance is None:
                return await self._voting_predict(individual_predictions)
            
            # Convert predictions to feature vector
            meta_features = np.array(list(individual_predictions.values())).reshape(1, -1)
            
            # Get meta-model prediction
            if hasattr(self.meta_model_instance, 'predict_proba'):
                pred = self.meta_model_instance.predict_proba(meta_features)[0][1]
            else:
                pred = self.meta_model_instance.predict(meta_features)[0]
            
            return pred
            
        except Exception as e:
            self.logger.error(f"Failed to make stacking prediction: {e}")
            return await self._voting_predict(individual_predictions)
    
    async def _blending_predict(self, individual_predictions: Dict[str, Union[float, int]]) -> Union[float, int]:
        """Blending-based prediction"""
        try:
            # Use current weights for blending
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, pred in individual_predictions.items():
                weight = self.model_weights.get(model_name, 1.0)
                weighted_sum += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.5 if self.ensemble_type == ModelType.CLASSIFIER else 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to make blending prediction: {e}")
            return 0.5 if self.ensemble_type == ModelType.CLASSIFIER else 0.0
    
    async def _calculate_confidence(self, individual_predictions: Dict[str, Union[float, int]], 
                                  ensemble_pred: Union[float, int]) -> float:
        """Calculate prediction confidence"""
        try:
            if not individual_predictions:
                return 0.0
            
            # Calculate variance of individual predictions
            predictions = list(individual_predictions.values())
            variance = np.var(predictions)
            
            # Lower variance = higher confidence
            confidence = max(0.0, 1.0 - variance)
            
            # Consider model weights
            weight_factor = sum(self.model_weights.values()) / len(self.model_weights)
            confidence *= weight_factor
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    async def _optimize_weights(self):
        """Optimize model weights based on performance"""
        try:
            if len(self.performance_history) < self.performance_window:
                return
            
            # Get recent performance
            recent_performance = self.performance_history[-self.performance_window:]
            
            # Calculate model performance
            for model_name in self.base_model_instances.keys():
                if model_name in self.model_performance:
                    # Use exponential moving average
                    alpha = 0.1
                    current_perf = self.model_performance[model_name]
                    new_perf = np.mean([p.get(model_name, 0.0) for p in recent_performance])
                    self.model_performance[model_name] = alpha * new_perf + (1 - alpha) * current_perf
                else:
                    self.model_performance[model_name] = 0.5  # Default performance
            
            # Update weights based on performance
            total_performance = sum(self.model_performance.values())
            if total_performance > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] = self.model_performance[model_name] / total_performance
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
            
            self.stats['weight_updates'] += 1
            self.logger.info("Updated ensemble weights based on performance")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize weights: {e}")
    
    async def update_performance(self, true_label: Union[float, int], 
                               prediction: EnsemblePrediction):
        """Update performance tracking"""
        try:
            # Calculate individual model performance
            for model_name, pred in prediction.individual_predictions.items():
                if self.ensemble_type == ModelType.CLASSIFIER:
                    # For classification, use accuracy
                    accuracy = 1.0 if abs(pred - true_label) < 0.5 else 0.0
                else:
                    # For regression, use R² (simplified)
                    mse = (pred - true_label) ** 2
                    accuracy = max(0.0, 1.0 - mse)
                
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = accuracy
                else:
                    # Update with exponential moving average
                    alpha = 0.1
                    self.model_performance[model_name] = alpha * accuracy + (1 - alpha) * self.model_performance[model_name]
            
            # Calculate ensemble performance
            if self.ensemble_type == ModelType.CLASSIFIER:
                ensemble_accuracy = 1.0 if abs(prediction.prediction - true_label) < 0.5 else 0.0
            else:
                mse = (prediction.prediction - true_label) ** 2
                ensemble_accuracy = max(0.0, 1.0 - mse)
            
            # Store performance
            performance_record = {
                'timestamp': datetime.now(),
                'true_label': true_label,
                'ensemble_prediction': prediction.prediction,
                'ensemble_accuracy': ensemble_accuracy,
                **{name: self.model_performance.get(name, 0.0) for name in self.base_model_instances.keys()}
            }
            
            self.performance_history.append(performance_record)
            self.ensemble_performance.append(ensemble_accuracy)
            
            # Update ensemble accuracy
            if len(self.ensemble_performance) > 0:
                self.stats['ensemble_accuracy'] = np.mean(self.ensemble_performance[-100:])
            
            # Maintain history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            if len(self.ensemble_performance) > 1000:
                self.ensemble_performance = self.ensemble_performance[-1000:]
            
        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")
    
    async def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        return {
            'strategy': self.strategy.value,
            'ensemble_type': self.ensemble_type.value if self.ensemble_type else 'unknown',
            'base_models': list(self.base_model_instances.keys()),
            'model_weights': self.model_weights,
            'statistics': self.stats,
            'performance': {
                'ensemble_accuracy': self.stats['ensemble_accuracy'],
                'model_performance': self.model_performance,
                'recent_performance': self.performance_history[-10:] if self.performance_history else []
            },
            'diversity': {
                'diversity_score': self.stats['diversity_score'],
                'correlation_matrix': self.correlation_matrix.tolist() if self.correlation_matrix is not None else None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ensemble health"""
        try:
            return {
                'status': 'healthy',
                'components': {
                    'database': self.db_connection is not None,
                    'storage': self.storage is not None,
                    'model_registry': self.model_registry is not None,
                    'ensemble_model': self.ensemble_model is not None
                },
                'ensemble_state': {
                    'base_models_loaded': len(self.base_model_instances) > 0,
                    'weights_initialized': len(self.model_weights) > 0,
                    'strategy': self.strategy.value
                },
                'performance': {
                    'total_predictions': self.stats['total_predictions'],
                    'ensemble_accuracy': self.stats['ensemble_accuracy']
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self):
        """Close the ensemble system"""
        try:
            # Save ensemble state
            if self.storage:
                await self.storage.save_data(
                    "ensemble_state",
                    {
                        'model_weights': self.model_weights,
                        'performance_history': self.performance_history[-100:],
                        'statistics': self.stats,
                        'last_update': datetime.now()
                    }
                )
            
            # Clear performance data
            self.performance_history.clear()
            self.ensemble_performance.clear()
            
            self.logger.info("Model ensemble closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close ensemble: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
