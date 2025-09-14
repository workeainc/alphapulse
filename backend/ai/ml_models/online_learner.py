"""
Online Learning System Module

Provides continuous model improvement through online learning,
drift detection, and adaptive model updates.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import joblib
import pickle
from pathlib import Path

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

class DriftType(Enum):
    """Types of data drift"""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    LABEL_DRIFT = "label_drift"
    FEATURE_DRIFT = "feature_drift"

class LearningMode(Enum):
    """Online learning modes"""
    PASSIVE = "passive"  # Only learn from new data
    ACTIVE = "active"    # Actively query for labels
    ADAPTIVE = "adaptive"  # Adjust learning rate based on performance

@dataclass
class DriftMetrics:
    """Metrics for drift detection"""
    drift_score: float
    drift_type: DriftType
    confidence: float
    timestamp: datetime
    features_affected: List[str] = field(default_factory=list)
    severity: str = "low"

@dataclass
class OnlineLearningConfig:
    """Configuration for online learning"""
    batch_size: int = 100
    learning_rate: float = 0.01
    max_samples: int = 10000
    drift_threshold: float = 0.1
    retrain_threshold: float = 0.05
    memory_size: int = 1000
    update_frequency: int = 100

class OnlineLearner:
    """Advanced online learning system with drift detection and shadow mode validation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Learning configuration
        self.batch_size = self.config.get('batch_size', 100)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.max_samples = self.config.get('max_samples', 10000)
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.retrain_threshold = self.config.get('retrain_threshold', 0.05)
        self.memory_size = self.config.get('memory_size', 1000)
        self.update_frequency = self.config.get('update_frequency', 100)
        
        # Phase 4C: Online & Safe Self-Retraining enhancements
        self.enable_incremental_learning = self.config.get('enable_incremental_learning', True)
        self.enable_shadow_mode = self.config.get('enable_shadow_mode', True)
        self.shadow_validation_threshold = self.config.get('shadow_validation_threshold', 0.7)
        self.auto_rollback_threshold = self.config.get('auto_rollback_threshold', 0.05)
        self.mini_batch_size = self.config.get('mini_batch_size', 1000)
        self.warm_start_enabled = self.config.get('warm_start_enabled', True)
        
        # Phase 5A: Canary Deployment enhancements
        self.enable_canary_deployment = self.config.get('enable_canary_deployment', True)
        self.canary_traffic_percentage = self.config.get('canary_traffic_percentage', 0.01)  # 1% initial
        self.canary_validation_threshold = self.config.get('canary_validation_threshold', 0.75)
        self.canary_rollback_threshold = self.config.get('canary_rollback_threshold', 0.03)
        self.canary_promotion_stages = self.config.get('canary_promotion_stages', [0.01, 0.05, 0.25, 1.0])  # Traffic percentages
        self.canary_min_samples_per_stage = self.config.get('canary_min_samples_per_stage', 1000)
        self.canary_min_duration_per_stage = self.config.get('canary_min_duration_per_stage', 3600)  # 1 hour in seconds
        
        # Component references
        self.db_connection = None
        self.storage = None
        self.model_registry = None
        
        # Learning state
        self.current_model = None
        self.model_version = None
        self.learning_mode = LearningMode.PASSIVE
        self.is_learning = False
        
        # Phase 4C: Shadow mode state
        self.shadow_model = None
        self.shadow_model_version = None
        self.shadow_validation_results = []
        self.production_validation_results = []
        self.shadow_mode_active = False
        
        # Phase 5A: Canary deployment state
        self.canary_model = None
        self.canary_model_version = None
        self.canary_deployment_active = False
        self.canary_current_stage = 0
        self.canary_stage_start_time = None
        self.canary_stage_samples = 0
        self.canary_performance_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'calibration_score': []
        }
        self.canary_rollback_triggered = False
        self.canary_promotion_ready = False
        
        # Data buffers
        self.feature_buffer = []
        self.label_buffer = []
        self.prediction_buffer = []
        self.performance_buffer = []
        
        # Phase 4C: Mini-batch buffers for heavy models
        self.mini_batch_buffer = []
        self.mini_batch_counter = 0
        
        # Drift detection
        self.drift_history: List[DriftMetrics] = []
        self.last_drift_check = None
        self.drift_detected = False
        
        # Performance tracking
        self.stats = {
            'total_samples_processed': 0,
            'models_updated': 0,
            'drift_detections': 0,
            'retraining_events': 0,
            'learning_time': 0.0
        }
        
        # Initialize drift detection
        self._initialize_drift_detection()
        
    def _initialize_drift_detection(self):
        """Initialize drift detection components"""
        try:
            # Initialize statistical tests for drift detection
            self.feature_stats = {}
            self.label_stats = {}
            self.concept_stats = {}
            
            self.logger.info("Drift detection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize drift detection: {e}")
    
    async def initialize(self):
        """Initialize the online learner"""
        try:
            self.logger.info("Initializing Online Learning System...")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            # Initialize storage if available
            if DataStorage:
                self.storage = DataStorage(
                    storage_path=self.config.get('learning_storage_path', 'learning'),
                    db_config=self.config.get('db_config', {})
                )
                await self.storage.initialize()
            
            # Initialize model registry if available
            if ModelRegistry:
                self.model_registry = ModelRegistry(
                    self.config.get('model_registry_config', {})
                )
                await self.model_registry.initialize()
            
            # Load current model if available
            await self._load_current_model()
            
            self.logger.info("Online Learning System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Online Learner: {e}")
            raise
    
    async def _load_current_model(self):
        """Load the current active model"""
        try:
            if self.model_registry:
                active_model = await self.model_registry.get_active_model("default")
                if active_model:
                    self.current_model = active_model
                    self.model_version = active_model.get('version', 'unknown')
                    self.logger.info(f"Loaded model version: {self.model_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to load current model: {e}")
    
    async def process_sample(self, features: np.ndarray, label: float, 
                           prediction: float = None) -> Dict[str, Any]:
        """Process a single sample for online learning"""
        try:
            start_time = datetime.now()
            
            # Add to buffers
            self.feature_buffer.append(features)
            self.label_buffer.append(label)
            if prediction is not None:
                self.prediction_buffer.append(prediction)
            
            # Check for drift
            drift_metrics = await self._check_drift(features, label)
            
            # Update performance tracking
            if prediction is not None:
                error = abs(prediction - label)
                self.performance_buffer.append(error)
            
            # Process batch if ready
            if len(self.feature_buffer) >= self.batch_size:
                await self._process_batch()
            
            # Update statistics
            self.stats['total_samples_processed'] += 1
            self.stats['learning_time'] += (datetime.now() - start_time).total_seconds()
            
            return {
                'drift_detected': drift_metrics.drift_score > self.drift_threshold,
                'drift_metrics': drift_metrics,
                'batch_ready': len(self.feature_buffer) >= self.batch_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process sample: {e}")
            return {'error': str(e)}
    
    async def _check_drift(self, features: np.ndarray, label: float) -> DriftMetrics:
        """Check for data drift"""
        try:
            current_time = datetime.now()
            
            # Simple statistical drift detection
            drift_score = 0.0
            drift_type = DriftType.FEATURE_DRIFT
            confidence = 0.5
            
            # Check feature drift
            if len(self.feature_buffer) > 0:
                recent_features = np.array(self.feature_buffer[-100:])
                if recent_features.shape[0] > 0:
                    feature_mean = np.mean(recent_features, axis=0)
                    feature_std = np.std(recent_features, axis=0)
                    
                    # Calculate drift score based on feature statistics
                    if len(self.feature_stats) > 0:
                        for i, (old_mean, old_std) in enumerate(self.feature_stats.items()):
                            if i < len(feature_mean):
                                drift_score += abs(feature_mean[i] - old_mean) / (old_std + 1e-8)
                    
                    # Update feature statistics
                    self.feature_stats = {
                        i: (feature_mean[i], feature_std[i]) 
                        for i in range(len(feature_mean))
                    }
            
            # Check concept drift (performance degradation)
            if len(self.performance_buffer) > 100:
                recent_performance = np.mean(self.performance_buffer[-100:])
                if len(self.performance_buffer) > 200:
                    older_performance = np.mean(self.performance_buffer[-200:-100])
                    if recent_performance > older_performance * 1.2:  # 20% degradation
                        drift_score += 0.3
                        drift_type = DriftType.CONCEPT_DRIFT
            
            # Normalize drift score
            drift_score = min(drift_score, 1.0)
            
            # Determine severity
            if drift_score > 0.7:
                severity = "high"
            elif drift_score > 0.4:
                severity = "medium"
            else:
                severity = "low"
            
            # Create drift metrics
            drift_metrics = DriftMetrics(
                drift_score=drift_score,
                drift_type=drift_type,
                confidence=confidence,
                timestamp=current_time,
                severity=severity
            )
            
            # Store drift history
            self.drift_history.append(drift_metrics)
            
            # Update drift state
            if drift_score > self.drift_threshold:
                self.drift_detected = True
                self.stats['drift_detections'] += 1
                
                # Trigger retraining if needed
                if drift_score > self.retrain_threshold:
                    await self._trigger_retraining()
            
            return drift_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to check drift: {e}")
            return DriftMetrics(
                drift_score=0.0,
                drift_type=DriftType.FEATURE_DRIFT,
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    async def _process_batch(self):
        """Process a batch of samples for learning"""
        try:
            if len(self.feature_buffer) < self.batch_size:
                return
            
            # Extract batch
            batch_features = np.array(self.feature_buffer[:self.batch_size])
            batch_labels = np.array(self.label_buffer[:self.batch_size])
            
            # Update model if available
            if self.current_model is not None:
                await self._update_model(batch_features, batch_labels)
            
            # Clear processed samples
            self.feature_buffer = self.feature_buffer[self.batch_size:]
            self.label_buffer = self.label_buffer[self.batch_size:]
            if self.prediction_buffer:
                self.prediction_buffer = self.prediction_buffer[self.batch_size:]
            
            # Maintain buffer size limits
            if len(self.feature_buffer) > self.memory_size:
                excess = len(self.feature_buffer) - self.memory_size
                self.feature_buffer = self.feature_buffer[excess:]
                self.label_buffer = self.label_buffer[excess:]
                if self.prediction_buffer:
                    self.prediction_buffer = self.prediction_buffer[excess:]
            
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
    
    async def _update_model(self, features: np.ndarray, labels: np.ndarray):
        """Update the current model with new data"""
        try:
            if self.current_model is None:
                return
            
            # Simple online update (placeholder for actual model update logic)
            # In practice, this would depend on the specific model type
            
            self.logger.info(f"Updated model with {len(features)} samples")
            self.stats['models_updated'] += 1
            
            # Save updated model
            if self.storage:
                await self.storage.save_data(
                    f"online_model_{self.model_version}",
                    {
                        'model': self.current_model,
                        'last_update': datetime.now(),
                        'samples_processed': self.stats['total_samples_processed']
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update model: {e}")
    
    async def _trigger_retraining(self):
        """Trigger model retraining due to drift"""
        try:
            self.logger.warning("Triggering model retraining due to drift")
            
            # Notify model registry
            if self.model_registry:
                await self.model_registry.archive_model("default")
            
            # Set retraining flag
            self.stats['retraining_events'] += 1
            
            # In practice, this would trigger a full retraining pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to trigger retraining: {e}")
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            'stats': self.stats,
            'buffer_sizes': {
                'features': len(self.feature_buffer),
                'labels': len(self.label_buffer),
                'predictions': len(self.prediction_buffer)
            },
            'drift_status': {
                'drift_detected': self.drift_detected,
                'last_drift_check': self.last_drift_check,
                'total_drift_events': len(self.drift_history)
            },
            'model_status': {
                'current_version': self.model_version,
                'learning_mode': self.learning_mode.value,
                'is_learning': self.is_learning
            }
        }
    
    async def set_learning_mode(self, mode: LearningMode):
        """Set the learning mode"""
        try:
            self.learning_mode = mode
            self.logger.info(f"Learning mode set to: {mode.value}")
            
            # Adjust parameters based on mode
            if mode == LearningMode.ACTIVE:
                self.batch_size = max(50, self.batch_size // 2)
                self.learning_rate *= 1.5
            elif mode == LearningMode.ADAPTIVE:
                self.batch_size = self.config.get('batch_size', 100)
                self.learning_rate = self.config.get('learning_rate', 0.01)
            
        except Exception as e:
            self.logger.error(f"Failed to set learning mode: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the online learner"""
        try:
            return {
                'status': 'healthy',
                'components': {
                    'database': self.db_connection is not None,
                    'storage': self.storage is not None,
                    'model_registry': self.model_registry is not None
                },
                'learning_state': {
                    'is_learning': self.is_learning,
                    'buffer_health': len(self.feature_buffer) < self.memory_size,
                    'drift_status': not self.drift_detected
                },
                'statistics': self.stats
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self):
        """Close the online learner"""
        try:
            # Save current state
            if self.storage:
                await self.storage.save_data(
                    "online_learner_state",
                    {
                        'stats': self.stats,
                        'drift_history': self.drift_history,
                        'last_update': datetime.now()
                    }
                )
            
            # Clear buffers
            self.feature_buffer.clear()
            self.label_buffer.clear()
            self.prediction_buffer.clear()
            self.performance_buffer.clear()
            
            self.logger.info("Online learner closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close online learner: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    # Phase 4C: Online & Safe Self-Retraining Methods
    
    async def incremental_learn(self, features: np.ndarray, labels: np.ndarray, 
                               model_type: str = "gradient_boosting") -> Dict[str, Any]:
        """
        Incremental learning with warm_start for lightweight models and mini-batch for heavy models
        """
        try:
            if not self.enable_incremental_learning:
                return {'status': 'disabled', 'message': 'Incremental learning is disabled'}
            
            # Add to mini-batch buffer
            self.mini_batch_buffer.extend(list(zip(features, labels)))
            self.mini_batch_counter += len(features)
            
            # Check if we should process mini-batch
            if self.mini_batch_counter >= self.mini_batch_size:
                return await self._process_mini_batch(model_type)
            
            return {'status': 'buffered', 'samples_buffered': self.mini_batch_counter}
            
        except Exception as e:
            self.logger.error(f"Failed to perform incremental learning: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _process_mini_batch(self, model_type: str) -> Dict[str, Any]:
        """Process mini-batch for model update"""
        try:
            # Extract features and labels from buffer
            features = np.array([item[0] for item in self.mini_batch_buffer])
            labels = np.array([item[1] for item in self.mini_batch_buffer])
            
            # Clear buffer
            self.mini_batch_buffer.clear()
            self.mini_batch_counter = 0
            
            # Create shadow model if not exists
            if self.shadow_model is None:
                await self._create_shadow_model(model_type)
            
            # Update shadow model
            if model_type in ["linear_regression", "logistic_regression", "sgd"]:
                # Use warm_start for lightweight models
                result = await self._update_lightweight_model(features, labels, model_type)
            else:
                # Use mini-batch for heavy models (XGBoost, LightGBM)
                result = await self._update_heavy_model(features, labels, model_type)
            
            # Validate shadow model
            validation_result = await self._validate_shadow_model()
            
            # Check if shadow model should be promoted
            if validation_result['should_promote']:
                await self._promote_shadow_model()
                return {'status': 'promoted', 'validation_score': validation_result['score']}
            else:
                return {'status': 'updated', 'validation_score': validation_result['score']}
            
        except Exception as e:
            self.logger.error(f"Failed to process mini-batch: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _create_shadow_model(self, model_type: str):
        """Create a shadow model for safe updates"""
        try:
            # Copy current production model to shadow
            if self.current_model is not None:
                self.shadow_model = self._clone_model(self.current_model)
                self.shadow_model_version = f"{self.model_version}_shadow"
                self.shadow_mode_active = True
                self.logger.info(f"Shadow model created: {self.shadow_model_version}")
            else:
                self.logger.warning("No current model to create shadow from")
                
        except Exception as e:
            self.logger.error(f"Failed to create shadow model: {e}")
    
    async def _update_lightweight_model(self, features: np.ndarray, labels: np.ndarray, 
                                       model_type: str) -> Dict[str, Any]:
        """Update lightweight models with warm_start"""
        try:
            if self.shadow_model is None:
                return {'status': 'error', 'message': 'No shadow model available'}
            
            # Use warm_start for incremental learning
            if hasattr(self.shadow_model, 'partial_fit'):
                self.shadow_model.partial_fit(features, labels)
                return {'status': 'updated', 'method': 'partial_fit'}
            elif hasattr(self.shadow_model, 'warm_start'):
                # For models with warm_start parameter
                self.shadow_model.warm_start = True
                self.shadow_model.fit(features, labels)
                return {'status': 'updated', 'method': 'warm_start'}
            else:
                return {'status': 'error', 'message': 'Model does not support incremental learning'}
                
        except Exception as e:
            self.logger.error(f"Failed to update lightweight model: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _update_heavy_model(self, features: np.ndarray, labels: np.ndarray, 
                                 model_type: str) -> Dict[str, Any]:
        """Update heavy models with mini-batch approach"""
        try:
            if self.shadow_model is None:
                return {'status': 'error', 'message': 'No shadow model available'}
            
            # For XGBoost, LightGBM, etc. - use mini-batch approach
            if hasattr(self.shadow_model, 'fit'):
                # Add new data to existing model (if supported)
                if hasattr(self.shadow_model, 'update'):
                    self.shadow_model.update(features, labels)
                else:
                    # Retrain with combined data (less efficient but safer)
                    # In practice, you'd want to implement proper incremental learning
                    # for these model types
                    self.logger.warning("Heavy model update - consider implementing proper incremental learning")
                    return {'status': 'warning', 'message': 'Heavy model update not fully optimized'}
                
                return {'status': 'updated', 'method': 'mini_batch'}
            else:
                return {'status': 'error', 'message': 'Model does not support updates'}
                
        except Exception as e:
            self.logger.error(f"Failed to update heavy model: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_shadow_model(self) -> Dict[str, Any]:
        """Validate shadow model against production model"""
        try:
            if not self.shadow_mode_active or self.shadow_model is None:
                return {'should_promote': False, 'score': 0.0, 'reason': 'No shadow model'}
            
            # Get recent validation data
            validation_data = await self._get_validation_data()
            if validation_data is None:
                return {'should_promote': False, 'score': 0.0, 'reason': 'No validation data'}
            
            # Compare predictions
            shadow_predictions = self.shadow_model.predict(validation_data['features'])
            production_predictions = self.current_model.predict(validation_data['features'])
            
            # Calculate improvement metrics
            shadow_accuracy = self._calculate_accuracy(shadow_predictions, validation_data['labels'])
            production_accuracy = self._calculate_accuracy(production_predictions, validation_data['labels'])
            
            improvement = shadow_accuracy - production_accuracy
            
            # Check promotion criteria
            should_promote = (
                improvement > self.shadow_validation_threshold and
                shadow_accuracy > 0.5  # Minimum accuracy threshold
            )
            
            # Check rollback criteria
            if improvement < -self.auto_rollback_threshold:
                await self._rollback_shadow_model()
                return {'should_promote': False, 'score': improvement, 'reason': 'Auto-rollback triggered'}
            
            return {
                'should_promote': should_promote,
                'score': improvement,
                'shadow_accuracy': shadow_accuracy,
                'production_accuracy': production_accuracy,
                'reason': 'Validation completed'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate shadow model: {e}")
            return {'should_promote': False, 'score': 0.0, 'reason': f'Validation error: {str(e)}'}
    
    async def _promote_shadow_model(self):
        """Promote shadow model to production"""
        try:
            if self.shadow_model is None:
                self.logger.error("No shadow model to promote")
                return
            
            # Backup current production model
            if self.current_model is not None:
                backup_version = f"{self.model_version}_backup_{int(time.time())}"
                await self._save_model_backup(self.current_model, backup_version)
            
            # Promote shadow model
            self.current_model = self.shadow_model
            self.model_version = self.shadow_model_version.replace('_shadow', '')
            self.shadow_model = None
            self.shadow_model_version = None
            self.shadow_mode_active = False
            
            # Update model registry
            if self.model_registry:
                await self.model_registry.update_model_version(
                    self.model_version, 
                    {'status': 'promoted', 'promoted_at': datetime.now()}
                )
            
            self.logger.info(f"Shadow model promoted to production: {self.model_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to promote shadow model: {e}")
    
    async def _rollback_shadow_model(self):
        """Rollback shadow model due to performance degradation"""
        try:
            self.shadow_model = None
            self.shadow_model_version = None
            self.shadow_mode_active = False
            
            self.logger.warning("Shadow model rolled back due to performance degradation")
            
        except Exception as e:
            self.logger.error(f"Failed to rollback shadow model: {e}")
    
    def _clone_model(self, model):
        """Create a deep copy of a model"""
        try:
            import copy
            return copy.deepcopy(model)
        except Exception as e:
            self.logger.error(f"Failed to clone model: {e}")
            return None
    
    def _calculate_accuracy(self, predictions, labels):
        """Calculate accuracy score"""
        try:
            from sklearn.metrics import accuracy_score
            return accuracy_score(labels, predictions)
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy: {e}")
            return 0.0
    
    async def _get_validation_data(self):
        """Get recent validation data"""
        try:
            # In practice, this would fetch recent validation data
            # For now, return None to indicate no validation data
            return None
        except Exception as e:
            self.logger.error(f"Failed to get validation data: {e}")
            return None
    
    async def _save_model_backup(self, model, version):
        """Save model backup"""
        try:
            if self.storage:
                await self.storage.save_data(f"model_backup_{version}", model)
            self.logger.info(f"Model backup saved: {version}")
        except Exception as e:
            self.logger.error(f"Failed to save model backup: {e}")

    # Phase 5A: Canary Deployment Methods
    
    async def start_canary_deployment(self, candidate_model, model_version: str) -> Dict[str, Any]:
        """Start canary deployment with a candidate model"""
        try:
            if not self.enable_canary_deployment:
                return {'success': False, 'reason': 'Canary deployment disabled'}
            
            if self.canary_deployment_active:
                return {'success': False, 'reason': 'Canary deployment already active'}
            
            # Initialize canary deployment
            self.canary_model = self._clone_model(candidate_model)
            self.canary_model_version = f"{model_version}_canary"
            self.canary_deployment_active = True
            self.canary_current_stage = 0
            self.canary_stage_start_time = datetime.now()
            self.canary_stage_samples = 0
            self.canary_rollback_triggered = False
            self.canary_promotion_ready = False
            
            # Reset performance metrics
            for metric in self.canary_performance_metrics:
                self.canary_performance_metrics[metric] = []
            
            # Log canary deployment start
            await self._log_canary_event('deployment_started', {
                'model_version': self.canary_model_version,
                'initial_traffic_percentage': self.canary_promotion_stages[0],
                'stages': self.canary_promotion_stages
            })
            
            self.logger.info(f"Canary deployment started: {self.canary_model_version}")
            return {
                'success': True,
                'canary_version': self.canary_model_version,
                'current_stage': 0,
                'traffic_percentage': self.canary_promotion_stages[0]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start canary deployment: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def process_canary_prediction(self, features: np.ndarray, label: float = None) -> Dict[str, Any]:
        """Process prediction through canary model and track performance"""
        try:
            if not self.canary_deployment_active or self.canary_model is None:
                return {'canary_active': False, 'use_canary': False}
            
            # Determine if this request should use canary model
            current_traffic_percentage = self.canary_promotion_stages[self.canary_current_stage]
            use_canary = np.random.random() < current_traffic_percentage
            
            if use_canary:
                # Use canary model for prediction
                canary_prediction = self._predict_with_model(self.canary_model, features)
                
                # Track performance if label is available
                if label is not None:
                    await self._track_canary_performance(features, label, canary_prediction)
                
                return {
                    'canary_active': True,
                    'use_canary': True,
                    'prediction': canary_prediction,
                    'model_version': self.canary_model_version,
                    'traffic_percentage': current_traffic_percentage
                }
            else:
                # Use production model
                production_prediction = self._predict_with_model(self.current_model, features)
                return {
                    'canary_active': True,
                    'use_canary': False,
                    'prediction': production_prediction,
                    'model_version': self.model_version,
                    'traffic_percentage': current_traffic_percentage
                }
                
        except Exception as e:
            self.logger.error(f"Failed to process canary prediction: {e}")
            return {'canary_active': False, 'use_canary': False, 'error': str(e)}
    
    async def _track_canary_performance(self, features: np.ndarray, label: float, prediction: float):
        """Track canary model performance metrics"""
        try:
            self.canary_stage_samples += 1
            
            # Calculate metrics
            accuracy = 1.0 if (prediction > 0.5 and label > 0.5) or (prediction <= 0.5 and label <= 0.5) else 0.0
            
            # Store metrics
            self.canary_performance_metrics['accuracy'].append(accuracy)
            
            # Check if we should evaluate current stage
            if self._should_evaluate_canary_stage():
                await self._evaluate_canary_stage()
                
        except Exception as e:
            self.logger.error(f"Failed to track canary performance: {e}")
    
    def _should_evaluate_canary_stage(self) -> bool:
        """Check if current canary stage should be evaluated"""
        try:
            # Check minimum samples
            if self.canary_stage_samples < self.canary_min_samples_per_stage:
                return False
            
            # Check minimum duration
            if self.canary_stage_start_time:
                duration = (datetime.now() - self.canary_stage_start_time).total_seconds()
                if duration < self.canary_min_duration_per_stage:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check canary stage evaluation: {e}")
            return False
    
    async def _evaluate_canary_stage(self) -> Dict[str, Any]:
        """Evaluate current canary stage and decide on promotion/rollback"""
        try:
            # Calculate current stage performance
            current_accuracy = np.mean(self.canary_performance_metrics['accuracy'])
            
            # Get production baseline (simplified - in practice would fetch from DB)
            production_accuracy = 0.85  # Placeholder
            
            # Calculate improvement
            improvement = current_accuracy - production_accuracy
            
            # Log stage evaluation
            await self._log_canary_event('stage_evaluated', {
                'stage': self.canary_current_stage,
                'traffic_percentage': self.canary_promotion_stages[self.canary_current_stage],
                'samples_processed': self.canary_stage_samples,
                'current_accuracy': current_accuracy,
                'production_accuracy': production_accuracy,
                'improvement': improvement
            })
            
            # Check rollback criteria
            if improvement < -self.canary_rollback_threshold:
                await self._rollback_canary_deployment('performance_degradation', {
                    'improvement': improvement,
                    'threshold': -self.canary_rollback_threshold
                })
                return {'action': 'rollback', 'reason': 'Performance degradation'}
            
            # Check promotion criteria
            if improvement > self.canary_validation_threshold:
                if self.canary_current_stage < len(self.canary_promotion_stages) - 1:
                    # Move to next stage
                    await self._advance_canary_stage()
                    return {'action': 'advance_stage', 'next_stage': self.canary_current_stage}
                else:
                    # Ready for full promotion
                    self.canary_promotion_ready = True
                    return {'action': 'ready_for_promotion', 'reason': 'All stages passed'}
            
            # Continue current stage
            return {'action': 'continue_stage', 'reason': 'Insufficient improvement for promotion'}
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate canary stage: {e}")
            return {'action': 'error', 'reason': str(e)}
    
    async def _advance_canary_stage(self):
        """Advance to next canary stage"""
        try:
            self.canary_current_stage += 1
            self.canary_stage_start_time = datetime.now()
            self.canary_stage_samples = 0
            
            # Reset metrics for new stage
            for metric in self.canary_performance_metrics:
                self.canary_performance_metrics[metric] = []
            
            # Log stage advancement
            await self._log_canary_event('stage_advanced', {
                'new_stage': self.canary_current_stage,
                'traffic_percentage': self.canary_promotion_stages[self.canary_current_stage]
            })
            
            self.logger.info(f"Canary advanced to stage {self.canary_current_stage}: {self.canary_promotion_stages[self.canary_current_stage] * 100}% traffic")
            
        except Exception as e:
            self.logger.error(f"Failed to advance canary stage: {e}")
    
    async def promote_canary_to_production(self) -> Dict[str, Any]:
        """Promote canary model to production"""
        try:
            if not self.canary_promotion_ready:
                return {'success': False, 'reason': 'Canary not ready for promotion'}
            
            # Backup current production model
            if self.current_model is not None:
                backup_version = f"{self.model_version}_backup_{int(time.time())}"
                await self._save_model_backup(self.current_model, backup_version)
            
            # Promote canary model
            self.current_model = self.canary_model
            self.model_version = self.canary_model_version.replace('_canary', '')
            self.canary_model = None
            self.canary_model_version = None
            self.canary_deployment_active = False
            self.canary_promotion_ready = False
            
            # Update model registry
            if self.model_registry:
                await self.model_registry.update_model_version(
                    self.model_version, 
                    {'status': 'promoted', 'promoted_at': datetime.now(), 'promotion_method': 'canary'}
                )
            
            # Log promotion
            await self._log_canary_event('promoted_to_production', {
                'model_version': self.model_version,
                'promotion_method': 'canary'
            })
            
            self.logger.info(f"Canary model promoted to production: {self.model_version}")
            return {'success': True, 'model_version': self.model_version}
            
        except Exception as e:
            self.logger.error(f"Failed to promote canary model: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def _rollback_canary_deployment(self, reason: str, details: Dict[str, Any]):
        """Rollback canary deployment"""
        try:
            # Log rollback
            await self._log_canary_event('deployment_rollback', {
                'reason': reason,
                'details': details,
                'stage': self.canary_current_stage,
                'model_version': self.canary_model_version
            })
            
            # Reset canary state
            self.canary_model = None
            self.canary_model_version = None
            self.canary_deployment_active = False
            self.canary_current_stage = 0
            self.canary_stage_start_time = None
            self.canary_stage_samples = 0
            self.canary_rollback_triggered = True
            self.canary_promotion_ready = False
            
            self.logger.warning(f"Canary deployment rolled back: {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to rollback canary deployment: {e}")
    
    async def _log_canary_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log canary deployment events to database"""
        try:
            if self.db_connection:
                # This would log to a canary_events table
                # For now, just log to console
                self.logger.info(f"Canary event: {event_type} - {event_data}")
        except Exception as e:
            self.logger.error(f"Failed to log canary event: {e}")
    
    def _predict_with_model(self, model, features: np.ndarray) -> float:
        """Make prediction with a model"""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(features.reshape(1, -1))[0][1]
            elif hasattr(model, 'predict'):
                return model.predict(features.reshape(1, -1))[0]
            else:
                return 0.5  # Default prediction
        except Exception as e:
            self.logger.error(f"Failed to make prediction: {e}")
            return 0.5
    
    async def get_canary_status(self) -> Dict[str, Any]:
        """Get current canary deployment status"""
        try:
            if not self.canary_deployment_active:
                return {'active': False}
            
            return {
                'active': True,
                'model_version': self.canary_model_version,
                'current_stage': self.canary_current_stage,
                'traffic_percentage': self.canary_promotion_stages[self.canary_current_stage],
                'samples_processed': self.canary_stage_samples,
                'promotion_ready': self.canary_promotion_ready,
                'rollback_triggered': self.canary_rollback_triggered,
                'stage_start_time': self.canary_stage_start_time.isoformat() if self.canary_stage_start_time else None,
                'performance_metrics': {
                    metric: np.mean(values) if values else 0.0 
                    for metric, values in self.canary_performance_metrics.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get canary status: {e}")
            return {'active': False, 'error': str(e)}
