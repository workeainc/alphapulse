"""
Advanced Machine Learning Models for News Impact Analysis
Implements LightGBM, XGBoost, and Random Forest models for news prediction
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import json
import os

# ML Libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
    from sklearn.feature_selection import SelectKBest, f_regression, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR, SVC
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.ensemble import VotingRegressor, VotingClassifier
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.covariance import EllipticEnvelope
    from sklearn.manifold import TSNE
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("⚠️ ML libraries not available. Install: pip install lightgbm xgboost scikit-learn catboost tensorflow torch")

# Phase 4B: Enhanced ML Models with Self-Training Integration
# Add imports for new ML feature integration
try:
    import river
    from river import linear_model, optim, preprocessing, metrics
    from river.stream import iter_pandas
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("⚠️ River library not available for online learning")

# Phase 5: Automated Retraining Foundation
try:
    import schedule
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logging.warning("⚠️ Schedule library not available for automated retraining")

# Phase 6: Advanced ML Features
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import shap
    import lime
    import lime.lime_tabular
    import eli5
    from eli5.lime import TextExplainer
    import mlflow
    import wandb
    from transformers import AutoTokenizer, AutoModel, pipeline
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset
    import evaluate
    from peft import LoraConfig, get_peft_model
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("⚠️ Advanced ML libraries not available. Install: pip install optuna shap lime eli5 mlflow wandb transformers datasets evaluate peft")

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Data class for model predictions"""
    prediction: float
    confidence: float
    model_type: str
    features_used: List[str]
    feature_importance: Dict[str, float]
    timestamp: datetime
    model_version: str

@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    mse: float
    mae: float
    model_type: str
    timestamp: datetime
    dataset_size: int

@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    # Scheduling
    retraining_schedule: str = "0 2 * * *"  # Daily at 2 AM UTC
    performance_check_interval: int = 60  # minutes
    drift_check_interval: int = 30  # minutes
    
    # Triggers
    performance_threshold: float = 0.7  # Minimum accuracy
    drift_threshold: float = 0.1  # KS-test threshold
    data_threshold: int = 1000  # Minimum new samples
    
    # Model management
    max_model_versions: int = 3  # Keep last 3 versions
    rollback_threshold: float = 0.05  # Performance drop threshold
    
    # Resource management
    max_training_time: int = 3600  # seconds
    memory_limit: str = "4Gi"
    cpu_limit: str = "2"

@dataclass
class RetrainingLog:
    """Log entry for retraining events"""
    timestamp: datetime
    event_type: str  # 'scheduled', 'performance_triggered', 'drift_triggered', 'manual'
    model_type: str
    trigger_reason: str
    training_samples: int
    validation_samples: int
    old_performance: float
    new_performance: float
    improvement: float
    training_duration: float
    status: str  # 'success', 'failed', 'in_progress'
    error_message: Optional[str] = None
    model_version: str = ""
    feature_drift_score: Optional[float] = None
    data_quality_score: Optional[float] = None

@dataclass
class NeuralNetworkConfig:
    """Configuration for neural network models"""
    layers: List[int]
    activation: str
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    validation_split: float
    early_stopping_patience: int
    reduce_lr_patience: int
    model_type: str  # 'tensorflow' or 'pytorch'

@dataclass
class EnsembleConfig:
    """Configuration for ensemble models"""
    base_models: List[str]
    voting_method: str  # 'hard', 'soft', 'weighted'
    weights: List[float]
    meta_learner: str
    cross_validation_folds: int

@dataclass
class FeatureEngineeringConfig:
    """Configuration for advanced feature engineering"""
    pca_components: int
    feature_selection_method: str
    polynomial_degree: int
    interaction_features: bool
    temporal_features: bool
    lag_features: List[int]
    rolling_features: List[int]
    fourier_features: bool
    wavelet_features: bool

@dataclass
class RealTimeTrainingConfig:
    """Configuration for real-time model training"""
    online_learning: bool
    incremental_batch_size: int
    retraining_frequency: int  # minutes
    performance_threshold: float
    drift_detection: bool
    concept_drift_threshold: float
    model_versioning: bool

@dataclass
class SelfTrainingConfig:
    """Configuration for self-training ML pipeline"""
    # Training schedule
    batch_training_interval_hours: int = 24  # Nightly batch training
    online_learning_interval_minutes: int = 15  # Real-time updates every 15 minutes
    
    # Retraining triggers
    drift_threshold: float = 0.1  # KS-test threshold for feature drift

# Phase 6: Advanced ML Features Dataclasses
@dataclass
class HyperparameterOptimizationConfig:
    """Configuration for hyperparameter optimization"""
    optimizer: str = 'optuna'  # 'optuna', 'hyperopt', 'bayesian'
    n_trials: int = 100
    timeout: int = 3600  # seconds
    objective: str = 'minimize_rmse'  # 'minimize_rmse', 'maximize_accuracy', 'minimize_loss'
    sampler: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    pruner: str = 'median'  # 'median', 'hyperband', 'none'
    parallel_jobs: int = 1
    early_stopping_patience: int = 10
    study_name: str = 'alphapulse_optimization'
    storage: Optional[str] = None  # SQLite database path

@dataclass
class ModelInterpretabilityConfig:
    """Configuration for model interpretability"""
    shap_enabled: bool = True
    lime_enabled: bool = True
    eli5_enabled: bool = True
    background_samples: int = 100
    max_features: int = 20
    explainer_type: str = 'tree'  # 'tree', 'linear', 'kernel'
    sample_size: int = 1000
    feature_names: Optional[List[str]] = None

@dataclass
class TransformerConfig:
    """Configuration for transformer models"""
    model_type: str = 'lstm'  # 'lstm', 'gru', 'transformer', 'bert', 'gpt'
    model_name: str = 'default'
    sequence_length: int = 100
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    use_attention: bool = True
    bidirectional: bool = True
    fine_tuning: bool = False
    base_model: Optional[str] = None

@dataclass
class AdvancedEnsembleConfig:
    """Configuration for advanced ensemble models"""
    ensemble_type: str = 'voting'  # 'voting', 'stacking', 'blending', 'bagging'
    base_models: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost', 'catboost'])
    weighting_strategy: str = 'performance'  # 'equal', 'performance', 'dynamic'
    voting_method: str = 'soft'  # 'soft', 'hard'
    meta_learner: Optional[str] = None
    cross_validation_folds: int = 5
    use_probabilities: bool = True
    feature_importance_weighting: bool = True

@dataclass
class ExperimentTrackingConfig:
    """Configuration for ML experiment tracking"""
    tracking_backend: str = 'mlflow'  # 'mlflow', 'wandb', 'local'
    experiment_name: str = 'alphapulse_ml'
    log_metrics: bool = True
    log_artifacts: bool = True
    log_models: bool = True
    auto_log: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    run_name: Optional[str] = None
    performance_threshold: float = 0.7  # Minimum accuracy threshold
    data_threshold: int = 1000  # Minimum new data samples
    
    # Model management
    max_model_versions: int = 3  # Keep last 3 versions
    rollback_threshold: float = 0.05  # Performance drop threshold for rollback
    
    # Resource management
    max_training_time: int = 3600  # Maximum training time in seconds
    memory_limit: str = "4Gi"  # Memory limit for training
    cpu_limit: str = "2"  # CPU limit for training
    
    # Advanced features
    ensemble_learning: bool = True  # Use ensemble methods
    hyperparameter_optimization: bool = True  # Auto-tune hyperparameters
    feature_selection: bool = True  # Automatic feature selection
    cross_validation: bool = True  # Use cross-validation
    early_stopping: bool = True  # Early stopping for training
    
    # Monitoring
    performance_tracking: bool = True  # Track model performance over time
    drift_detection: bool = True  # Monitor feature and concept drift
    alerting: bool = True  # Send alerts for significant events
    
    # Integration
    timescaledb_integration: bool = True  # Store data in TimescaleDB
    kubernetes_integration: bool = True  # Use Kubernetes for scaling
    webhook_notifications: bool = True  # Send webhook notifications
    
    # Prediction targets
    prediction_targets: List[str] = None  # ['regime_change', 'sector_rotation', 'price_direction']
    
    def __post_init__(self):
        if self.prediction_targets is None:
            self.prediction_targets = ['regime_change', 'sector_rotation', 'price_direction']

class NewsMLModels:
    """Advanced Machine Learning Models for News Impact Analysis"""
    
    def __init__(self, config: Dict[str, Any], db_pool=None):
        self.config = config
        self.db_pool = db_pool
        self.ml_config = config.get('machine_learning', {})
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        
        # Performance tracking
        self.performance_history = []
        
        # Model paths
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Phase 3: Advanced Analytics
        self.neural_networks = {}
        self.ensemble_models = {}
        self.feature_pipelines = {}
        self.online_learners = {}
        self.drift_detectors = {}
        self.model_versions = {}
        
        # Real-time training
        self.last_retraining = {}
        self.performance_monitors = {}
        self.concept_drift_history = {}
        
        # Phase 5: Automated Retraining Foundation
        self.retraining_config = RetrainingConfig()
        self.retraining_scheduler = None
        self.retraining_lock = Lock()
        self.retraining_logs = []
        self.performance_thresholds = {}
        self.drift_monitors = {}
        self.scheduled_tasks = {}
        
        # Kubernetes integration
        self.k8s_enabled = config.get('kubernetes', {}).get('enabled', False)
        self.k8s_namespace = config.get('kubernetes', {}).get('namespace', 'alphapulse')
        
        # Phase 6: Advanced ML Features
        self.hyperopt_config = HyperparameterOptimizationConfig()
        self.interpretability_config = ModelInterpretabilityConfig()
        self.transformer_config = TransformerConfig()
        self.advanced_ensemble_config = AdvancedEnsembleConfig()
        self.experiment_tracking_config = ExperimentTrackingConfig()
        
        # Advanced ML components
        self.hyperopt_studies = {}
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.transformer_models = {}
        self.advanced_ensembles = {}
        self.experiment_runs = {}
        
        # Model interpretability storage
        self.feature_importance_cache = {}
        self.shap_values_cache = {}
        self.lime_explanations_cache = {}
        
        # Experiment tracking
        self.mlflow_client = None
        self.wandb_runs = {}
        
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available. Models will not function.")
            return
        
        logger.info("🤖 NewsMLModels initialized successfully")
    
    async def initialize_models(self):
        """Initialize or load existing models"""
        try:
            if not ML_AVAILABLE:
                return
            
            # Initialize models for each prediction type
            prediction_models = self.ml_config.get('prediction_models', {})
            
            for model_name, model_config in prediction_models.items():
                if model_config.get('enabled', False):
                    await self._initialize_model(model_name, model_config)
            
            logger.info(f"✅ Initialized {len(self.models)} ML models")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ML models: {e}")
    
    async def _initialize_model(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize a specific model"""
        try:
            model_type = model_config.get('model_type', 'lightgbm')
            model_path = os.path.join(self.model_dir, f"{model_name}_{model_type}.pkl")
            
            # Try to load existing model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"✅ Loaded existing {model_name} model")
            else:
                # Create new model
                self.models[model_name] = self._create_model(model_type, model_config)
                logger.info(f"✅ Created new {model_name} model")
            
            # Initialize scaler and encoders
            self.scalers[model_name] = StandardScaler()
            self.label_encoders[model_name] = LabelEncoder()
            self.feature_selectors[model_name] = SelectKBest(score_func=f_regression, k=10)
            
        except Exception as e:
            logger.error(f"❌ Error initializing {model_name} model: {e}")
    
    def _create_model(self, model_type: str, model_config: Dict[str, Any]):
        """Create a new ML model based on type"""
        hyperparams = model_config.get('hyperparameters', {})
        
        if model_type == 'lightgbm':
            return lgb.LGBMRegressor(**hyperparams, random_state=42, verbose=-1)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(**hyperparams, random_state=42)
        elif model_type == 'random_forest':
            return RandomForestRegressor(**hyperparams, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    async def predict_news_impact(self, features: Dict[str, float]) -> ModelPrediction:
        """Predict news impact using LightGBM model"""
        try:
            if not ML_AVAILABLE or 'impact_prediction' not in self.models:
                return self._fallback_prediction('impact_prediction', features)
            
            # Prepare features
            feature_vector = self._prepare_features(features, 'impact_prediction')
            
            # Make prediction
            prediction = self.models['impact_prediction'].predict(feature_vector)[0]
            
            # Calculate confidence (using model's prediction variance if available)
            confidence = self._calculate_confidence('impact_prediction', feature_vector)
            
            # Get feature importance
            feature_importance = self._get_feature_importance('impact_prediction', features)
            
            return ModelPrediction(
                prediction=float(prediction),
                confidence=confidence,
                model_type='lightgbm',
                features_used=list(features.keys()),
                feature_importance=feature_importance,
                timestamp=datetime.utcnow(),
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Error predicting news impact: {e}")
            return self._fallback_prediction('impact_prediction', features)
    
    async def enhance_sentiment_prediction(self, features: Dict[str, float]) -> ModelPrediction:
        """Enhance sentiment prediction using XGBoost model"""
        try:
            if not ML_AVAILABLE or 'sentiment_enhancement' not in self.models:
                return self._fallback_prediction('sentiment_enhancement', features)
            
            # Prepare features
            feature_vector = self._prepare_features(features, 'sentiment_enhancement')
            
            # Make prediction
            prediction = self.models['sentiment_enhancement'].predict(feature_vector)[0]
            
            # Calculate confidence
            confidence = self._calculate_confidence('sentiment_enhancement', feature_vector)
            
            # Get feature importance
            feature_importance = self._get_feature_importance('sentiment_enhancement', features)
            
            return ModelPrediction(
                prediction=float(prediction),
                confidence=confidence,
                model_type='xgboost',
                features_used=list(features.keys()),
                feature_importance=feature_importance,
                timestamp=datetime.utcnow(),
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Error enhancing sentiment prediction: {e}")
            return self._fallback_prediction('sentiment_enhancement', features)
    
    async def optimize_timing_prediction(self, features: Dict[str, float]) -> ModelPrediction:
        """Optimize timing prediction using Random Forest model"""
        try:
            if not ML_AVAILABLE or 'timing_optimization' not in self.models:
                return self._fallback_prediction('timing_optimization', features)
            
            # Prepare features
            feature_vector = self._prepare_features(features, 'timing_optimization')
            
            # Make prediction
            prediction = self.models['timing_optimization'].predict(feature_vector)[0]
            
            # Calculate confidence
            confidence = self._calculate_confidence('timing_optimization', feature_vector)
            
            # Get feature importance
            feature_importance = self._get_feature_importance('timing_optimization', features)
            
            return ModelPrediction(
                prediction=float(prediction),
                confidence=confidence,
                model_type='random_forest',
                features_used=list(features.keys()),
                feature_importance=feature_importance,
                timestamp=datetime.utcnow(),
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Error optimizing timing prediction: {e}")
            return self._fallback_prediction('timing_optimization', features)
    
    def _prepare_features(self, features: Dict[str, float], model_name: str) -> np.ndarray:
        """Prepare features for model prediction"""
        try:
            # Get the expected feature list from config
            expected_features = self.ml_config.get('prediction_models', {}).get(model_name, {}).get('features', [])
            
            if expected_features:
                # Use expected feature order
                feature_values = []
                for feature_name in expected_features:
                    feature_values.append(features.get(feature_name, 0.0))
                feature_vector = np.array(feature_values).reshape(1, -1)
            else:
                # Fallback to all features
                feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Apply feature selection if available and fitted
            if (model_name in self.feature_selectors and 
                hasattr(self.feature_selectors[model_name], 'get_support') and
                hasattr(self.feature_selectors[model_name], 'transform')):
                try:
                    feature_vector = self.feature_selectors[model_name].transform(feature_vector)
                except Exception as selector_e:
                    logger.warning(f"⚠️ Feature selector failed for {model_name}: {selector_e}")
                    # If feature selector fails, try to match the expected number of features
                    if hasattr(self.models[model_name], 'n_features_in_'):
                        expected_n_features = self.models[model_name].n_features_in_
                        if feature_vector.shape[1] > expected_n_features:
                            # Take first n features to match model expectation
                            feature_vector = feature_vector[:, :expected_n_features]
            
            # Scale features if scaler is fitted
            if (model_name in self.scalers and 
                hasattr(self.scalers[model_name], 'mean_') and
                hasattr(self.scalers[model_name], 'transform')):
                try:
                    feature_vector = self.scalers[model_name].transform(feature_vector)
                except Exception as scaler_e:
                    logger.warning(f"⚠️ Feature scaler failed for {model_name}: {scaler_e}")
                    # Continue without scaling
            
            # Final check: ensure feature count matches model expectation
            if model_name in self.models and hasattr(self.models[model_name], 'n_features_in_'):
                expected_n_features = self.models[model_name].n_features_in_
                if feature_vector.shape[1] != expected_n_features:
                    logger.warning(f"⚠️ Feature count mismatch for {model_name}: got {feature_vector.shape[1]}, expected {expected_n_features}")
                    if feature_vector.shape[1] > expected_n_features:
                        # Truncate to expected size
                        feature_vector = feature_vector[:, :expected_n_features]
                    else:
                        # Pad with zeros to expected size
                        padding = np.zeros((1, expected_n_features - feature_vector.shape[1]))
                        feature_vector = np.hstack([feature_vector, padding])
            
            # Ensure 2D array
            if feature_vector.ndim == 1:
                feature_vector = feature_vector.reshape(1, -1)
            elif feature_vector.ndim > 2:
                feature_vector = feature_vector.reshape(1, -1)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            # Return appropriate shape based on model expectation
            if model_name in self.models and hasattr(self.models[model_name], 'n_features_in_'):
                return np.zeros((1, self.models[model_name].n_features_in_))
            else:
                return np.zeros((1, 10))  # Default fallback
    
    def _calculate_confidence(self, model_name: str, feature_vector: np.ndarray) -> float:
        """Calculate prediction confidence"""
        try:
            # For ensemble models, use prediction variance
            if hasattr(self.models[model_name], 'estimators_'):
                predictions = []
                for estimator in self.models[model_name].estimators_:
                    pred = estimator.predict(feature_vector)[0]
                    predictions.append(pred)
                
                # Calculate confidence based on prediction variance
                variance = np.var(predictions)
                confidence = max(0.1, 1.0 - variance)
                return min(1.0, confidence)
            
            # For single models, use a default confidence
            return 0.7
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence: {e}")
            return 0.5
    
    def _get_feature_importance(self, model_name: str, features: Dict[str, float]) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(self.models[model_name], 'feature_importances_'):
                importance = self.models[model_name].feature_importances_
                feature_names = list(features.keys())
                
                # Normalize importance scores
                importance = importance / np.sum(importance)
                
                return dict(zip(feature_names, importance))
            
            return {feature: 1.0/len(features) for feature in features.keys()}
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return {feature: 1.0/len(features) for feature in features.keys()}
    
    def _fallback_prediction(self, model_name: str, features: Dict[str, float]) -> ModelPrediction:
        """Fallback prediction when ML models are not available"""
        # Simple rule-based prediction
        if model_name == 'impact_prediction':
            # High sentiment + high volume = high impact
            sentiment = features.get('sentiment_score', 0.0)
            volume = features.get('social_volume', 0.0)
            prediction = (sentiment * 0.6 + volume * 0.4) * 0.5
        elif model_name == 'sentiment_enhancement':
            # Enhance sentiment based on context
            base_sentiment = features.get('sentiment_score', 0.0)
            context_boost = features.get('market_regime', 0.0) * 0.1
            prediction = base_sentiment + context_boost
        else:  # timing_optimization
            # Optimize timing based on market hours and volatility
            hour = features.get('hour_of_day', 12)
            volatility = features.get('market_volatility', 0.0)
            prediction = (hour / 24.0) * 0.5 + volatility * 0.5
        
        return ModelPrediction(
            prediction=float(prediction),
            confidence=0.5,
            model_type='rule_based',
            features_used=list(features.keys()),
            feature_importance={feature: 1.0/len(features) for feature in features.keys()},
            timestamp=datetime.utcnow(),
            model_version='fallback'
        )
    
    async def train_models(self, training_data: List[Dict[str, Any]]):
        """Train all models with provided data"""
        try:
            if not ML_AVAILABLE or not training_data:
                logger.warning("⚠️ No training data available or ML libraries not available")
                return
            
            logger.info(f"🤖 Starting model training with {len(training_data)} samples")
            
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                if model_config.get('enabled', False):
                    await self._train_model(model_name, model_config, training_data)
            
            logger.info("✅ Model training completed")
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
    
    async def train_models_with_training_data(self, training_data_collector):
        """Train models using TrainingDataCollector"""
        try:
            if not ML_AVAILABLE:
                logger.warning("⚠️ ML libraries not available")
                return
            
            # Collect training data
            data_result = await training_data_collector.collect_training_data()
            
            if data_result['total_samples'] == 0:
                logger.warning("⚠️ No training data available")
                return
            
            logger.info(f"🤖 Starting model training with {data_result['total_samples']} samples")
            
            # Train each model
            training_results = {}
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                if model_config.get('enabled', False):
                    result = await self._train_model_with_data(
                        model_name, model_config, 
                        training_data_collector.training_data,
                        training_data_collector.validation_data
                    )
                    training_results[model_name] = result
            
            # Save training results
            await self._save_training_results(training_results)
            
            logger.info("✅ Model training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error training models with training data: {e}")
            return {}
    
    async def _train_model_with_data(
        self, 
        model_name: str, 
        model_config: Dict[str, Any],
        training_data: List,
        validation_data: List
    ) -> Dict[str, Any]:
        """Train a specific model with training data"""
        try:
            logger.info(f"🔄 Training {model_name} model...")
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data_from_points(model_name, training_data)
            X_val, y_val = self._prepare_training_data_from_points(model_name, validation_data)
            
            if len(X_train) < 100:
                logger.warning(f"⚠️ Insufficient training data for {model_name}: {len(X_train)} samples")
                return {'error': 'Insufficient training data'}
            
            # Scale features
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_val_scaled = self.scalers[model_name].transform(X_val)
            
            # Feature selection
            X_train_selected = self.feature_selectors[model_name].fit_transform(X_train_scaled, y_train)
            X_val_selected = self.feature_selectors[model_name].transform(X_val_scaled)
            
            # Train model
            self.models[model_name].fit(X_train_selected, y_train)
            
            # Evaluate model
            y_pred_train = self.models[model_name].predict(X_train_selected)
            y_pred_val = self.models[model_name].predict(X_val_selected)
            
            # Calculate performance metrics
            train_performance = self._evaluate_model(y_train, y_pred_train, model_name)
            val_performance = self._evaluate_model(y_val, y_pred_val, model_name)
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}_{model_config['model_type']}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            
            # Save scaler and feature selector
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[model_name], f)
            
            selector_path = os.path.join(self.model_dir, f"{model_name}_selector.pkl")
            with open(selector_path, 'wb') as f:
                pickle.dump(self.feature_selectors[model_name], f)
            
            result = {
                'model_name': model_name,
                'model_type': model_config['model_type'],
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'training_performance': {
                    'accuracy': train_performance.accuracy,
                    'precision': train_performance.precision,
                    'recall': train_performance.recall,
                    'f1_score': train_performance.f1_score,
                    'mse': train_performance.mse,
                    'mae': train_performance.mae
                },
                'validation_performance': {
                    'accuracy': val_performance.accuracy,
                    'precision': val_performance.precision,
                    'recall': val_performance.recall,
                    'f1_score': val_performance.f1_score,
                    'mse': val_performance.mse,
                    'mae': val_performance.mae
                },
                'feature_importance': self._get_feature_importance_dict(model_name),
                'model_path': model_path,
                'scaler_path': scaler_path,
                'selector_path': selector_path,
                'training_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ {model_name} model trained successfully - Validation F1: {val_performance.f1_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name} model: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data_from_points(self, model_name: str, data_points: List) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from TrainingDataPoint objects"""
        try:
            features_list = []
            targets = []
            
            for data_point in data_points:
                # Extract features
                features = list(data_point.features.values())
                features_list.append(features)
                
                # Extract target based on model type
                if model_name == 'impact_prediction':
                    target = data_point.labels['price_impact_24h']
                elif model_name == 'sentiment_enhancement':
                    target = data_point.labels['enhanced_sentiment']
                else:  # timing_optimization
                    target = data_point.labels['optimal_timing_score']
                
                targets.append(target)
            
            return np.array(features_list), np.array(targets)
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data from points: {e}")
            return np.array([]), np.array([])
    
    def _get_feature_importance_dict(self, model_name: str) -> Dict[str, float]:
        """Get feature importance as dictionary"""
        try:
            if hasattr(self.models[model_name], 'feature_importances_'):
                importance = self.models[model_name].feature_importances_
                feature_names = [
                    'title_length', 'content_length', 'entity_count',
                    'sentiment_score', 'normalized_sentiment', 'sentiment_confidence',
                    'market_regime_score', 'btc_dominance', 'market_volatility',
                    'correlation_30m', 'correlation_2h', 'correlation_24h',
                    'hour_of_day', 'day_of_week', 'is_market_hours',
                    'social_volume', 'cross_source_validation', 'feed_credibility'
                ]
                
                # Handle NaN values and normalize importance scores
                importance = np.nan_to_num(importance, nan=0.0)
                if np.sum(importance) > 0:
                    importance = importance / np.sum(importance)
                else:
                    # If all values are zero, use uniform distribution
                    importance = np.ones_like(importance) / len(importance)
                
                return dict(zip(feature_names, importance))
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance dict: {e}")
            return {}
    
    async def _save_training_results(self, training_results: Dict[str, Any]):
        """Save training results to file"""
        try:
            results_path = os.path.join(self.model_dir, f"training_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Convert to serializable format
            serializable_results = {}
            for model_name, result in training_results.items():
                if 'error' not in result:
                    serializable_results[model_name] = {
                        'model_name': result['model_name'],
                        'model_type': result['model_type'],
                        'training_samples': result['training_samples'],
                        'validation_samples': result['validation_samples'],
                        'training_performance': result['training_performance'],
                        'validation_performance': result['validation_performance'],
                        'feature_importance': result['feature_importance'],
                        'training_timestamp': result['training_timestamp']
                    }
                else:
                    serializable_results[model_name] = {'error': result['error']}
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'dtype') else x)
            
            logger.info(f"✅ Training results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving training results: {e}")
    
    async def load_trained_models(self):
        """Load trained models from disk"""
        try:
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                model_type = model_config.get('model_type', 'lightgbm')
                model_path = os.path.join(self.model_dir, f"{model_name}_{model_type}.pkl")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
                selector_path = os.path.join(self.model_dir, f"{model_name}_selector.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(selector_path):
                    # Load model
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    
                    # Load scaler
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                    
                    # Load feature selector
                    with open(selector_path, 'rb') as f:
                        self.feature_selectors[model_name] = pickle.load(f)
                    
                    logger.info(f"✅ Loaded trained {model_name} model ({model_type})")
                else:
                    logger.warning(f"⚠️ Trained {model_name} model not found (expected: {model_path})")
            
        except Exception as e:
            logger.error(f"❌ Error loading trained models: {e}")
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        try:
            status = {}
            
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                model_type = model_config.get('model_type', 'lightgbm')
                model_path = os.path.join(self.model_dir, f"{model_name}_{model_type}.pkl")
                
                if os.path.exists(model_path):
                    # Get model info
                    model = self.models.get(model_name)
                    if model:
                        status[model_name] = {
                            'status': 'trained',
                            'model_type': type(model).__name__,
                            'has_feature_importance': hasattr(model, 'feature_importances_'),
                            'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                        }
                    else:
                        status[model_name] = {'status': 'not_loaded'}
                else:
                    status[model_name] = {'status': 'not_trained'}
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting model status: {e}")
            return {}
    
    async def _train_model(self, model_name: str, model_config: Dict[str, Any], training_data: List[Dict[str, Any]]):
        """Train a specific model"""
        try:
            logger.info(f"🔄 Training {model_name} model...")
            
            # Prepare training data
            X, y = self._prepare_training_data(model_name, training_data)
            
            if len(X) < 100:
                logger.warning(f"⚠️ Insufficient training data for {model_name}: {len(X)} samples")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_test_scaled = self.scalers[model_name].transform(X_test)
            
            # Feature selection
            X_train_selected = self.feature_selectors[model_name].fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selectors[model_name].transform(X_test_scaled)
            
            # Train model
            self.models[model_name].fit(X_train_selected, y_train)
            
            # Evaluate model
            y_pred = self.models[model_name].predict(X_test_selected)
            performance = self._evaluate_model(y_test, y_pred, model_name)
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}_{model_config['model_type']}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            
            logger.info(f"✅ {model_name} model trained successfully - Accuracy: {performance.accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name} model: {e}")
    
    def _prepare_training_data(self, model_name: str, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model"""
        try:
            features_list = []
            targets = []
            
            for data_point in training_data:
                # Extract features
                features = self._extract_training_features(data_point)
                features_list.append(list(features.values()))
                
                # Extract target based on model type
                if model_name == 'impact_prediction':
                    target = data_point.get('price_impact_24h', 0.0)
                elif model_name == 'sentiment_enhancement':
                    target = data_point.get('enhanced_sentiment', data_point.get('sentiment_score', 0.0))
                else:  # timing_optimization
                    target = data_point.get('optimal_timing_score', 0.5)
                
                targets.append(target)
            
            return np.array(features_list), np.array(targets)
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def _extract_training_features(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from training data point"""
        features = {}
        
        # Text features
        features['title_length'] = len(data_point.get('title', ''))
        features['content_length'] = len(data_point.get('content', ''))
        features['sentiment_score'] = data_point.get('sentiment_score', 0.0)
        features['entity_count'] = len(data_point.get('entities', []))
        
        # Market features
        features['market_regime'] = data_point.get('market_regime_score', 0.0)
        features['btc_dominance'] = data_point.get('btc_dominance', 50.0)
        features['market_volatility'] = data_point.get('market_volatility', 0.02)
        features['correlation_30m'] = data_point.get('correlation_30m', 0.0)
        
        # Temporal features
        features['hour_of_day'] = data_point.get('hour_of_day', 12)
        features['day_of_week'] = data_point.get('day_of_week', 3)
        features['is_market_hours'] = data_point.get('is_market_hours', 1.0)
        
        # Social features
        features['social_volume'] = data_point.get('social_volume', 0.0)
        features['cross_source_validation'] = data_point.get('cross_source_validation', 0.0)
        features['feed_credibility'] = data_point.get('feed_credibility', 0.5)
        
        return features
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            # For regression models, use regression metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate R² score for regression
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            
            # For regression, we'll use R² as accuracy equivalent
            accuracy = max(0.0, min(1.0, r2))  # Clamp between 0 and 1
            
            # Use MSE-based precision and recall for regression
            precision = max(0.0, 1.0 - mse)  # Higher precision for lower MSE
            recall = max(0.0, 1.0 - mae)     # Higher recall for lower MAE
            
            # Calculate F1 score for regression
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # For regression, AUC-ROC doesn't apply, use R² instead
            auc_roc = max(0.0, r2)
            
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                mse=mse,
                mae=mae,
                model_type=model_name,
                timestamp=datetime.utcnow(),
                dataset_size=len(y_true)
            )
            
            self.performance_history.append(performance)
            return performance
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model: {e}")
            return ModelPerformance(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                auc_roc=0.0, mse=0.0, mae=0.0, model_type=model_name,
                timestamp=datetime.utcnow(), dataset_size=0
            )
    
    async def get_model_performance(self, model_name: str = None) -> List[ModelPerformance]:
        """Get model performance history"""
        if model_name:
            return [p for p in self.performance_history if p.model_type == model_name]
        return self.performance_history
    
    async def save_models(self):
        """Save all models to disk"""
        try:
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info("✅ All models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    async def load_models(self):
        """Load all models from disk"""
        try:
            for model_name in self.ml_config.get('prediction_models', {}).keys():
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")

    # ==================== PHASE 3: ADVANCED ANALYTICS ====================
    
    async def create_neural_network(self, model_name: str, config: NeuralNetworkConfig, input_dim: int):
        """Create and train neural network model"""
        try:
            if config.model_type == 'tensorflow':
                model = self._create_tensorflow_nn(config, input_dim)
            elif config.model_type == 'pytorch':
                model = self._create_pytorch_nn(config, input_dim)
            else:
                model = self._create_sklearn_nn(config, input_dim)
            
            self.neural_networks[model_name] = model
            logger.info(f"✅ Neural network {model_name} created successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error creating neural network: {e}")
            return None
    
    def _create_tensorflow_nn(self, config: NeuralNetworkConfig, input_dim: int):
        """Create TensorFlow neural network"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(config.layers[0], input_dim=input_dim, activation=config.activation))
        model.add(layers.Dropout(config.dropout_rate))
        
        # Hidden layers
        for units in config.layers[1:]:
            model.add(layers.Dense(units, activation=config.activation))
            model.add(layers.Dropout(config.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_pytorch_nn(self, config: NeuralNetworkConfig, input_dim: int):
        """Create PyTorch neural network"""
        class NeuralNetwork(nn.Module):
            def __init__(self, input_dim, layers, activation='relu', dropout_rate=0.2):
                super(NeuralNetwork, self).__init__()
                self.layers_list = nn.ModuleList()
                
                # Input layer
                self.layers_list.append(nn.Linear(input_dim, layers[0]))
                self.layers_list.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
                self.layers_list.append(nn.Dropout(dropout_rate))
                
                # Hidden layers
                for i in range(len(layers) - 1):
                    self.layers_list.append(nn.Linear(layers[i], layers[i + 1]))
                    self.layers_list.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
                    self.layers_list.append(nn.Dropout(dropout_rate))
                
                # Output layer
                self.layers_list.append(nn.Linear(layers[-1], 1))
            
            def forward(self, x):
                for layer in self.layers_list:
                    x = layer(x)
                return x
        
        return NeuralNetwork(input_dim, config.layers, config.activation, config.dropout_rate)
    
    def _create_sklearn_nn(self, config: NeuralNetworkConfig, input_dim: int):
        """Create scikit-learn neural network"""
        return MLPRegressor(
            hidden_layer_sizes=tuple(config.layers),
            activation=config.activation,
            learning_rate_init=config.learning_rate,
            max_iter=config.epochs,
            early_stopping=True,
            validation_fraction=config.validation_split
        )
    
    async def create_ensemble_model(self, model_name: str, config: EnsembleConfig):
        """Create ensemble model with multiple base models"""
        try:
            base_models = []
            
            for base_model_name in config.base_models:
                if base_model_name in self.models:
                    base_models.append((base_model_name, self.models[base_model_name]))
                elif base_model_name in self.neural_networks:
                    base_models.append((base_model_name, self.neural_networks[base_model_name]))
            
            if config.voting_method == 'weighted':
                ensemble = VotingRegressor(
                    estimators=base_models,
                    weights=config.weights
                )
            else:
                ensemble = VotingRegressor(
                    estimators=base_models,
                    voting='soft' if config.voting_method == 'soft' else 'hard'
                )
            
            self.ensemble_models[model_name] = ensemble
            logger.info(f"✅ Ensemble model {model_name} created successfully")
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error creating ensemble model: {e}")
            return None
    
    async def create_feature_pipeline(self, pipeline_name: str, config: FeatureEngineeringConfig):
        """Create advanced feature engineering pipeline"""
        try:
            steps = []
            
            # PCA for dimensionality reduction
            if config.pca_components > 0:
                steps.append(('pca', PCA(n_components=config.pca_components)))
            
            # Feature selection
            if config.feature_selection_method:
                if config.feature_selection_method == 'kbest':
                    steps.append(('feature_selection', SelectKBest(f_regression, k=20)))
            
            # Polynomial features
            if config.polynomial_degree > 1:
                from sklearn.preprocessing import PolynomialFeatures
                steps.append(('poly', PolynomialFeatures(degree=config.polynomial_degree)))
            
            # Standard scaling
            steps.append(('scaler', StandardScaler()))
            
            pipeline = Pipeline(steps)
            self.feature_pipelines[pipeline_name] = pipeline
            
            logger.info(f"✅ Feature pipeline {pipeline_name} created successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Error creating feature pipeline: {e}")
            return None
    
    async def detect_concept_drift(self, model_name: str, recent_data: np.ndarray, recent_labels: np.ndarray) -> bool:
        """Detect concept drift in model performance"""
        try:
            if model_name not in self.models:
                return False
            
            # Get recent predictions
            model = self.models[model_name]
            recent_predictions = model.predict(recent_data)
            
            # Calculate recent performance
            recent_mse = mean_squared_error(recent_labels, recent_predictions)
            
            # Get historical performance
            if model_name in self.performance_monitors:
                historical_mse = self.performance_monitors[model_name].get('historical_mse', recent_mse)
                
                # Calculate drift
                drift_ratio = abs(recent_mse - historical_mse) / historical_mse
                
                # Update monitor
                self.performance_monitors[model_name] = {
                    'historical_mse': historical_mse,
                    'recent_mse': recent_mse,
                    'drift_ratio': drift_ratio,
                    'last_check': datetime.utcnow()
                }
                
                # Check if drift exceeds threshold
                if drift_ratio > 0.2:  # 20% performance degradation
                    logger.warning(f"⚠️ Concept drift detected for {model_name}: {drift_ratio:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error detecting concept drift: {e}")
            return False
    
    async def retrain_model_online(self, model_name: str, new_data: np.ndarray, new_labels: np.ndarray):
        """Retrain model with new data (online learning)"""
        try:
            if model_name not in self.models:
                return False
            
            model = self.models[model_name]
            
            # Check if model supports partial_fit
            if hasattr(model, 'partial_fit'):
                model.partial_fit(new_data, new_labels)
                logger.info(f"✅ Model {model_name} updated with new data")
                return True
            else:
                # For models that don't support partial_fit, retrain from scratch
                model.fit(new_data, new_labels)
                logger.info(f"✅ Model {model_name} retrained with new data")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error retraining model: {e}")
            return False
    
    async def get_model_ensemble_prediction(self, model_names: List[str], data: np.ndarray) -> Dict[str, Any]:
        """Get ensemble prediction from multiple models"""
        try:
            predictions = {}
            weights = {}
            
            for model_name in model_names:
                if model_name in self.models:
                    model = self.models[model_name]
                    pred = model.predict(data)
                    predictions[model_name] = pred
                    weights[model_name] = 1.0  # Equal weights for now
                elif model_name in self.neural_networks:
                    model = self.neural_networks[model_name]
                    if hasattr(model, 'predict'):
                        pred = model.predict(data)
                        predictions[model_name] = pred
                        weights[model_name] = 1.0
            
            # Calculate weighted ensemble prediction
            if predictions:
                total_weight = sum(weights.values())
                ensemble_pred = np.zeros_like(list(predictions.values())[0])
                
                for model_name, pred in predictions.items():
                    weight = weights[model_name] / total_weight
                    ensemble_pred += weight * pred
                
                return {
                    'ensemble_prediction': ensemble_pred,
                    'individual_predictions': predictions,
                    'weights': weights,
                    'confidence': self._calculate_ensemble_confidence(predictions)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error getting ensemble prediction: {e}")
            return {}
    
    def _calculate_ensemble_confidence(self, predictions: Dict[str, np.ndarray]) -> float:
        """Calculate confidence based on prediction agreement"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            # Calculate standard deviation of predictions
            pred_array = np.array(list(predictions.values()))
            std_dev = np.std(pred_array, axis=0)
            
            # Convert to confidence (lower std = higher confidence)
            confidence = 1.0 / (1.0 + np.mean(std_dev))
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"❌ Error calculating ensemble confidence: {e}")
            return 0.5
    
    async def get_model_analytics(self) -> Dict[str, Any]:
        """Get comprehensive model analytics"""
        try:
            analytics = {
                'total_models': len(self.models) + len(self.neural_networks) + len(self.ensemble_models),
                'model_types': {
                    'traditional': len(self.models),
                    'neural_networks': len(self.neural_networks),
                    'ensembles': len(self.ensemble_models)
                },
                'performance_summary': {},
                'drift_detection': {},
                'feature_pipelines': len(self.feature_pipelines),
                'online_learners': len(self.online_learners)
            }
            
            # Performance summary
            for model_name in self.models.keys():
                if model_name in self.performance_monitors:
                    analytics['performance_summary'][model_name] = self.performance_monitors[model_name]
            
            # Drift detection summary
            for model_name in self.models.keys():
                if model_name in self.concept_drift_history:
                    analytics['drift_detection'][model_name] = self.concept_drift_history[model_name]
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error getting model analytics: {e}")
            return {}

# Phase 4B: Enhanced ML Models with Self-Training Integration
# Add imports for new ML feature integration
try:
    import river
    from river import linear_model, optim, preprocessing, metrics
    from river.stream import iter_pandas
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("⚠️ River library not available for online learning")

class EnhancedMLModels:
    """Enhanced ML Models with self-training and online learning capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config
        self.db_pool = config.get('db_pool') # Assuming db_pool is in config
        self.ml_config = config.get('machine_learning', {})
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        
        # Performance tracking
        self.performance_history = []
        
        # Model paths
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Phase 3: Advanced Analytics
        self.neural_networks = {}
        self.ensemble_models = {}
        self.feature_pipelines = {}
        self.online_learners = {}
        self.drift_detectors = {}
        self.model_versions = {}
        
        # Real-time training
        self.last_retraining = {}
        self.performance_monitors = {}
        self.concept_drift_history = {}
        
        if not ML_AVAILABLE:
            logger.error("❌ ML libraries not available. Models will not function.")
            return
        
        logger.info("🤖 NewsMLModels initialized successfully")
        
        # Phase 4B: Self-training configuration
        self.self_training_config = SelfTrainingConfig()
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.online_learners: Dict[str, Any] = {}
        self.drift_detectors: Dict[str, Any] = {}
        
        # Phase 4B: ML Feature integration
        self.ml_feature_tables = [
            'ml_features_ohlcv',
            'ml_features_sentiment', 
            'ml_labels',
            'ml_models_metadata',
            'ml_predictions'
        ]
        
        # Initialize online learning if available
        if RIVER_AVAILABLE:
            self._initialize_online_learners()
    
    def _initialize_online_learners(self):
        """Initialize online learning models for each target"""
        try:
            for target in ['regime_change', 'sector_rotation', 'price_direction']:
                # Create online learning pipeline
                scaler = preprocessing.StandardScaler()
                model = linear_model.LogisticRegression()
                self.online_learners[target] = river.pipeline.Pipeline([
                    ('scaler', scaler),
                    ('model', model)
                ])
                
                # Initialize drift detector
                self.drift_detectors[target] = {
                    'feature_drift': [],
                    'concept_drift': [],
                    'last_check': datetime.utcnow()
                }
                
            logger.info(f"✅ Initialized online learners for {len(self.online_learners)} targets")
            
        except Exception as e:
            logger.error(f"❌ Error initializing online learners: {e}")
    
    async def collect_ml_training_data(self, 
                                     symbols: List[str] = None,
                                     start_time: datetime = None,
                                     end_time: datetime = None) -> Dict[str, Any]:
        """Collect ML training data from feature tables"""
        try:
            if not self.db_pool:
                logger.error("❌ Database connection not available")
                return {}
            
            symbols = symbols or ['BTC/USDT', 'ETH/USDT']
            start_time = start_time or (datetime.utcnow() - timedelta(days=self.self_training_config.feature_lookback_days))
            end_time = end_time or datetime.utcnow()
            
            logger.info(f"📊 Collecting ML training data for {len(symbols)} symbols from {start_time} to {end_time}")
            
            training_data = {
                'features': {},
                'labels': {},
                'metadata': {}
            }
            
            # Collect OHLCV features
            ohlcv_query = """
            SELECT symbol, timestamp, timeframe, 
                   vwap, atr, rsi, macd, macd_signal, macd_histogram,
                   bollinger_upper, bollinger_middle, bollinger_lower,
                   stoch_k, stoch_d, williams_r, cci, adx, obv, mfi
            FROM ml_features_ohlcv 
            WHERE symbol = ANY($1) AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp
            """
            
            # Collect sentiment features
            sentiment_query = """
            SELECT symbol, timestamp,
                   fear_greed_index, social_sentiment_score, news_sentiment_score,
                   weighted_coin_sentiment, whale_sentiment_proxy, sentiment_divergence_score,
                   sentiment_momentum, sentiment_volatility, sentiment_trend_strength
            FROM ml_features_sentiment 
            WHERE symbol = ANY($1) AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp
            """
            
            # Collect labels
            labels_query = """
            SELECT symbol, timestamp, label_type, label_value, label_confidence,
                   future_timestamp, realized_value, is_realized
            FROM ml_labels 
            WHERE symbol = ANY($1) AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp
            """
            
            async with self.db_pool.acquire() as conn:
                # Execute queries
                ohlcv_data = await conn.fetch(ohlcv_query, symbols, start_time, end_time)
                sentiment_data = await conn.fetch(sentiment_query, symbols, start_time, end_time)
                labels_data = await conn.fetch(labels_query, symbols, start_time, end_time)
                
                # Process OHLCV features
                for row in ohlcv_data:
                    symbol = row['symbol']
                    if symbol not in training_data['features']:
                        training_data['features'][symbol] = []
                    
                    features = {
                        'timestamp': row['timestamp'],
                        'timeframe': row['timeframe'],
                        'vwap': float(row['vwap']) if row['vwap'] else 0.0,
                        'atr': float(row['atr']) if row['atr'] else 0.0,
                        'rsi': float(row['rsi']) if row['rsi'] else 50.0,
                        'macd': float(row['macd']) if row['macd'] else 0.0,
                        'macd_signal': float(row['macd_signal']) if row['macd_signal'] else 0.0,
                        'macd_histogram': float(row['macd_histogram']) if row['macd_histogram'] else 0.0,
                        'bollinger_upper': float(row['bollinger_upper']) if row['bollinger_upper'] else 0.0,
                        'bollinger_middle': float(row['bollinger_middle']) if row['bollinger_middle'] else 0.0,
                        'bollinger_lower': float(row['bollinger_lower']) if row['bollinger_lower'] else 0.0,
                        'stoch_k': float(row['stoch_k']) if row['stoch_k'] else 50.0,
                        'stoch_d': float(row['stoch_d']) if row['stoch_d'] else 50.0,
                        'williams_r': float(row['williams_r']) if row['williams_r'] else -50.0,
                        'cci': float(row['cci']) if row['cci'] else 0.0,
                        'adx': float(row['adx']) if row['adx'] else 25.0,
                        'obv': float(row['obv']) if row['obv'] else 0.0,
                        'mfi': float(row['mfi']) if row['mfi'] else 50.0
                    }
                    training_data['features'][symbol].append(features)
                
                # Process sentiment features
                sentiment_by_symbol = {}
                for row in sentiment_data:
                    symbol = row['symbol']
                    if symbol not in sentiment_by_symbol:
                        sentiment_by_symbol[symbol] = []
                    
                    sentiment = {
                        'timestamp': row['timestamp'],
                        'fear_greed_index': int(row['fear_greed_index']) if row['fear_greed_index'] else 50,
                        'social_sentiment_score': float(row['social_sentiment_score']) if row['social_sentiment_score'] else 0.0,
                        'news_sentiment_score': float(row['news_sentiment_score']) if row['news_sentiment_score'] else 0.0,
                        'weighted_coin_sentiment': float(row['weighted_coin_sentiment']) if row['weighted_coin_sentiment'] else 0.0,
                        'whale_sentiment_proxy': float(row['whale_sentiment_proxy']) if row['whale_sentiment_proxy'] else 0.0,
                        'sentiment_divergence_score': float(row['sentiment_divergence_score']) if row['sentiment_divergence_score'] else 0.0,
                        'sentiment_momentum': float(row['sentiment_momentum']) if row['sentiment_momentum'] else 0.0,
                        'sentiment_volatility': float(row['sentiment_volatility']) if row['sentiment_volatility'] else 0.0,
                        'sentiment_trend_strength': float(row['sentiment_trend_strength']) if row['sentiment_trend_strength'] else 0.0
                    }
                    sentiment_by_symbol[symbol].append(sentiment)
                
                # Merge features with sentiment
                for symbol in training_data['features']:
                    if symbol in sentiment_by_symbol:
                        # Create a mapping of timestamp to sentiment
                        sentiment_map = {s['timestamp']: s for s in sentiment_by_symbol[symbol]}
                        
                        # Merge sentiment into features
                        for feature in training_data['features'][symbol]:
                            if feature['timestamp'] in sentiment_map:
                                feature.update(sentiment_map[feature['timestamp']])
                
                # Process labels
                for row in labels_data:
                    symbol = row['symbol']
                    if symbol not in training_data['labels']:
                        training_data['labels'][symbol] = []
                    
                    label = {
                        'timestamp': row['timestamp'],
                        'label_type': row['label_type'],
                        'label_value': row['label_value'],
                        'label_confidence': float(row['label_confidence']) if row['label_confidence'] else 0.0,
                        'future_timestamp': row['future_timestamp'],
                        'realized_value': row['realized_value'],
                        'is_realized': bool(row['is_realized'])
                    }
                    training_data['labels'][symbol].append(label)
            
            # Calculate summary statistics
            total_features = sum(len(features) for features in training_data['features'].values())
            total_labels = sum(len(labels) for labels in training_data['labels'].values())
            
            training_data['metadata'] = {
                'total_symbols': len(symbols),
                'total_features': total_features,
                'total_labels': total_labels,
                'start_time': start_time,
                'end_time': end_time,
                'collection_timestamp': datetime.utcnow()
            }
            
            logger.info(f"✅ Collected {total_features} feature samples and {total_labels} labels")
            return training_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting ML training data: {e}")
            return {}
    
    async def prepare_training_datasets(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare training datasets from collected data"""
        try:
            logger.info("🔄 Preparing training datasets...")
            
            datasets = {
                'regime_change': {'X': [], 'y': [], 'symbols': []},
                'sector_rotation': {'X': [], 'y': [], 'symbols': []},
                'price_direction': {'X': [], 'y': [], 'symbols': []}
            }
            
            for symbol, features in training_data['features'].items():
                if symbol not in training_data['labels']:
                    continue
                
                # Create feature matrix for this symbol
                feature_matrix = []
                for feature in features:
                    # Extract numerical features
                    feature_vector = [
                        feature.get('vwap', 0.0),
                        feature.get('atr', 0.0),
                        feature.get('rsi', 50.0),
                        feature.get('macd', 0.0),
                        feature.get('macd_signal', 0.0),
                        feature.get('macd_histogram', 0.0),
                        feature.get('bollinger_upper', 0.0),
                        feature.get('bollinger_middle', 0.0),
                        feature.get('bollinger_lower', 0.0),
                        feature.get('stoch_k', 50.0),
                        feature.get('stoch_d', 50.0),
                        feature.get('williams_r', -50.0),
                        feature.get('cci', 0.0),
                        feature.get('adx', 25.0),
                        feature.get('obv', 0.0),
                        feature.get('mfi', 50.0),
                        feature.get('fear_greed_index', 50),
                        feature.get('social_sentiment_score', 0.0),
                        feature.get('news_sentiment_score', 0.0),
                        feature.get('weighted_coin_sentiment', 0.0),
                        feature.get('whale_sentiment_proxy', 0.0),
                        feature.get('sentiment_divergence_score', 0.0),
                        feature.get('sentiment_momentum', 0.0),
                        feature.get('sentiment_volatility', 0.0),
                        feature.get('sentiment_trend_strength', 0.0)
                    ]
                    feature_matrix.append(feature_vector)
                
                if not feature_matrix:
                    continue
                
                # Match features with labels
                for label in training_data['labels'][symbol]:
                    # Find closest feature timestamp
                    label_time = label['timestamp']
                    closest_feature_idx = None
                    min_time_diff = float('inf')
                    
                    for i, feature in enumerate(features):
                        time_diff = abs((feature['timestamp'] - label_time).total_seconds())
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_feature_idx = i
                    
                    # Only use if within reasonable time window (1 hour)
                    if closest_feature_idx is not None and min_time_diff <= 3600:
                        feature_vector = feature_matrix[closest_feature_idx]
                        label_type = label['label_type']
                        
                        if label_type in datasets:
                            # Convert label value to numerical
                            label_value = self._encode_label(label['label_value'], label_type)
                            
                            if label_value is not None:
                                datasets[label_type]['X'].append(feature_vector)
                                datasets[label_type]['y'].append(label_value)
                                datasets[label_type]['symbols'].append(symbol)
            
            # Convert to numpy arrays
            for target in datasets:
                if datasets[target]['X']:
                    datasets[target]['X'] = np.array(datasets[target]['X'])
                    datasets[target]['y'] = np.array(datasets[target]['y'])
                    datasets[target]['symbols'] = np.array(datasets[target]['symbols'])
                else:
                    datasets[target]['X'] = np.array([])
                    datasets[target]['y'] = np.array([])
                    datasets[target]['symbols'] = np.array([])
            
            # Log dataset statistics
            for target, data in datasets.items():
                if len(data['X']) > 0:
                    logger.info(f"📊 {target}: {len(data['X'])} samples, {len(data['X'][0])} features")
                else:
                    logger.warning(f"⚠️ {target}: No training data available")
            
            return datasets
            
        except Exception as e:
            logger.error(f"❌ Error preparing training datasets: {e}")
            return {}
    
    def _encode_label(self, label_value: str, label_type: str) -> Optional[int]:
        """Encode label values to numerical format"""
        try:
            if label_type == 'regime_change':
                encoding = {
                    'bullish': 0, 'bearish': 1, 'sideways': 2, 'volatile': 3
                }
            elif label_type == 'sector_rotation':
                encoding = {
                    'btc_dominance': 0, 'altcoin_rotation': 1, 'stable': 2
                }
            elif label_type == 'price_direction':
                encoding = {
                    'bullish': 0, 'bearish': 1, 'neutral': 2
                }
            else:
                return None
            
            return encoding.get(label_value, None)
            
        except Exception as e:
            logger.error(f"❌ Error encoding label {label_value} for {label_type}: {e}")
            return None
    
    async def train_self_training_models(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Train models using self-training pipeline"""
        try:
            logger.info("🤖 Starting self-training model training...")
            
            training_results = {}
            
            for target, data in datasets.items():
                if len(data['X']) < 100:
                    logger.warning(f"⚠️ Insufficient data for {target}: {len(data['X'])} samples")
                    continue
                
                logger.info(f"🔄 Training {target} model with {len(data['X'])} samples...")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    data['X'], data['y'], 
                    test_size=0.2, 
                    random_state=42,
                    stratify=data['y'] if len(np.unique(data['y'])) > 1 else None
                )
                
                # Train multiple models
                models = {}
                
                # XGBoost
                try:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    xgb_model.fit(X_train, y_train)
                    models['xgboost'] = xgb_model
                except Exception as e:
                    logger.error(f"❌ XGBoost training failed for {target}: {e}")
                
                # LightGBM
                try:
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    lgb_model.fit(X_train, y_train)
                    models['lightgbm'] = lgb_model
                except Exception as e:
                    logger.error(f"❌ LightGBM training failed for {target}: {e}")
                
                # Random Forest
                try:
                    rf_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                    rf_model.fit(X_train, y_train)
                    models['random_forest'] = rf_model
                except Exception as e:
                    logger.error(f"❌ Random Forest training failed for {target}: {e}")
                
                # Evaluate models
                best_model = None
                best_score = 0.0
                model_metrics = {}
                
                for model_name, model in models.items():
                    try:
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        model_metrics[model_name] = {
                            'accuracy': accuracy,
                            'f1_score': f1,
                            'model': model
                        }
                        
                        if f1 > best_score:
                            best_score = f1
                            best_model = model_name
                            
                    except Exception as e:
                        logger.error(f"❌ Evaluation failed for {model_name} on {target}: {e}")
                
                # Store results
                training_results[target] = {
                    'models': model_metrics,
                    'best_model': best_model,
                    'best_score': best_score,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': X_train.shape[1] if len(X_train) > 0 else 0
                }
                
                logger.info(f"✅ {target} training completed. Best model: {best_model} (F1: {best_score:.3f})")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error in self-training: {e}")
            return {}
    
    async def store_model_versions(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Store trained models and metadata"""
        try:
            logger.info("💾 Storing model versions...")
            
            stored_models = {}
            timestamp = datetime.utcnow()
            
            for target, result in training_results.items():
                if 'best_model' not in result or not result['best_model']:
                    continue
                
                best_model_name = result['best_model']
                best_model = result['models'][best_model_name]['model']
                
                # Generate version
                version = f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                # Save model
                model_path = f"models/{target}_{best_model_name}_{version}.pkl"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                
                # Store metadata
                model_version = ModelVersion(
                    model_id=f"{target}_{best_model_name}",
                    version=version,
                    model_type=best_model_name,
                    training_timestamp=timestamp,
                    performance_metrics={
                        'accuracy': result['models'][best_model_name]['accuracy'],
                        'f1_score': result['models'][best_model_name]['f1_score']
                    },
                    feature_importance={},  # Will be populated if available
                    hyperparameters={},  # Will be populated if available
                    is_active=True,
                    model_path=model_path
                )
                
                # Store in database
                if self.db_pool:
                    await self._store_model_metadata(model_version)
                
                stored_models[target] = model_version
                
                logger.info(f"✅ Stored {target} model: {model_path}")
            
            return stored_models
            
        except Exception as e:
            logger.error(f"❌ Error storing model versions: {e}")
            return {}
    
    async def _store_model_metadata(self, model_version: ModelVersion):
        """Store model metadata in database"""
        try:
            query = """
            INSERT INTO ml_models_metadata (
                model_name, model_version, model_type, model_path,
                training_timestamp, is_active, performance_metrics,
                feature_importance, hyperparameters, training_data_size,
                validation_accuracy, test_accuracy
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    model_version.model_id,
                    model_version.version,
                    model_version.model_type,
                    model_version.model_path,
                    model_version.training_timestamp,
                    model_version.is_active,
                    json.dumps(model_version.performance_metrics),
                    json.dumps(model_version.feature_importance),
                    json.dumps(model_version.hyperparameters),
                    0,  # training_data_size - will be updated
                    model_version.performance_metrics.get('accuracy', 0.0),
                    model_version.performance_metrics.get('f1_score', 0.0)
                )
                
        except Exception as e:
            logger.error(f"❌ Error storing model metadata: {e}")
    
    async def run_online_learning_update(self, new_features: Dict[str, Any], new_labels: Dict[str, Any]):
        """Update online learning models with new data"""
        try:
            if not RIVER_AVAILABLE:
                logger.warning("⚠️ River library not available for online learning")
                return
            
            logger.info("🔄 Updating online learning models...")
            
            for target in self.online_learners:
                if target not in new_labels:
                    continue
                
                # Prepare new data
                X_new = []
                y_new = []
                
                for symbol, features in new_features.items():
                    if symbol not in new_labels[target]:
                        continue
                    
                    # Extract feature vector
                    feature_vector = self._extract_feature_vector(features)
                    if feature_vector is None:
                        continue
                    
                    # Get label
                    label_value = new_labels[target][symbol]
                    encoded_label = self._encode_label(label_value, target)
                    if encoded_label is None:
                        continue
                    
                    X_new.append(feature_vector)
                    y_new.append(encoded_label)
                
                if not X_new:
                    continue
                
                # Update online learner
                try:
                    for x, y in zip(X_new, y_new):
                        self.online_learners[target].learn_one(x, y)
                    
                    logger.info(f"✅ Updated {target} online learner with {len(X_new)} samples")
                    
                except Exception as e:
                    logger.error(f"❌ Error updating {target} online learner: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error in online learning update: {e}")
    
    def _extract_feature_vector(self, features: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from features dictionary"""
        try:
            feature_vector = [
                features.get('vwap', 0.0),
                features.get('atr', 0.0),
                features.get('rsi', 50.0),
                features.get('macd', 0.0),
                features.get('macd_signal', 0.0),
                features.get('macd_histogram', 0.0),
                features.get('bollinger_upper', 0.0),
                features.get('bollinger_middle', 0.0),
                features.get('bollinger_lower', 0.0),
                features.get('stoch_k', 50.0),
                features.get('stoch_d', 50.0),
                features.get('williams_r', -50.0),
                features.get('cci', 0.0),
                features.get('adx', 25.0),
                features.get('obv', 0.0),
                features.get('mfi', 50.0),
                features.get('fear_greed_index', 50),
                features.get('social_sentiment_score', 0.0),
                features.get('news_sentiment_score', 0.0),
                features.get('weighted_coin_sentiment', 0.0),
                features.get('whale_sentiment_proxy', 0.0),
                features.get('sentiment_divergence_score', 0.0),
                features.get('sentiment_momentum', 0.0),
                features.get('sentiment_volatility', 0.0),
                features.get('sentiment_trend_strength', 0.0)
            ]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"❌ Error extracting feature vector: {e}")
            return None
    
    async def detect_feature_drift(self, current_features: np.ndarray, reference_features: np.ndarray) -> Dict[str, Any]:
        """Detect feature drift using KS-test"""
        try:
            from scipy.stats import ks_2samp
            
            drift_results = {}
            
            for i in range(current_features.shape[1]):
                try:
                    # Perform KS-test
                    statistic, p_value = ks_2samp(
                        reference_features[:, i], 
                        current_features[:, i]
                    )
                    
                    # Determine drift severity
                    if p_value < 0.01:
                        severity = 'high'
                    elif p_value < 0.05:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    drift_results[f'feature_{i}'] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'severity': severity,
                        'drift_detected': p_value < self.self_training_config.drift_threshold
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error in KS-test for feature {i}: {e}")
                    continue
            
            # Overall drift assessment
            drift_detected = any(
                result['drift_detected'] for result in drift_results.values()
            )
            
            return {
                'drift_detected': drift_detected,
                'feature_drift': drift_results,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in drift detection: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    async def run_complete_self_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete self-training pipeline"""
        try:
            logger.info("🚀 Starting complete self-training pipeline...")
            
            # Step 1: Collect training data
            training_data = await self.collect_ml_training_data()
            if not training_data['features']:
                return {'status': 'no_data', 'message': 'No training data available'}
            
            # Step 2: Prepare datasets
            datasets = await self.prepare_training_datasets(training_data)
            if not datasets:
                return {'status': 'no_datasets', 'message': 'No datasets prepared'}
            
            # Step 3: Train models
            training_results = await self.train_self_training_models(datasets)
            if not training_results:
                return {'status': 'training_failed', 'message': 'Model training failed'}
            
            # Step 4: Store models
            stored_models = await self.store_model_versions(training_results)
            
            # Step 5: Generate summary
            summary = {
                'status': 'success',
                'timestamp': datetime.utcnow(),
                'training_data_summary': training_data['metadata'],
                'training_results': training_results,
                'stored_models': len(stored_models),
                'total_samples': sum(len(data['X']) for data in datasets.values() if len(data['X']) > 0)
            }
            
            logger.info(f"✅ Self-training pipeline completed successfully. Stored {len(stored_models)} models.")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error in self-training pipeline: {e}")
            return {'status': 'error', 'message': str(e)}

    # Phase 5: Automated Retraining Foundation Methods
    
    async def initialize_automated_retraining(self):
        """Initialize automated retraining system"""
        try:
            if not SCHEDULER_AVAILABLE:
                logger.warning("⚠️ Schedule library not available. Automated retraining disabled.")
                return False
            
            logger.info("🔄 Initializing automated retraining system...")
            
            # Initialize scheduler
            self.retraining_scheduler = schedule.Scheduler()
            
            # Set up scheduled retraining
            self._setup_scheduled_retraining()
            
            # Set up performance monitoring
            self._setup_performance_monitoring()
            
            # Set up drift monitoring
            self._setup_drift_monitoring()
            
            # Start scheduler in background thread
            self._start_scheduler()
            
            logger.info("✅ Automated retraining system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing automated retraining: {e}")
            return False
    
    def _setup_scheduled_retraining(self):
        """Set up scheduled retraining jobs"""
        try:
            # Daily retraining at 2 AM UTC
            self.retraining_scheduler.every().day.at("02:00").do(
                self._scheduled_retraining_job, "daily_scheduled"
            )
            
            # Weekly retraining on Sunday at 3 AM UTC
            self.retraining_scheduler.every().sunday.at("03:00").do(
                self._scheduled_retraining_job, "weekly_scheduled"
            )
            
            logger.info("✅ Scheduled retraining jobs configured")
            
        except Exception as e:
            logger.error(f"❌ Error setting up scheduled retraining: {e}")
    
    def _setup_performance_monitoring(self):
        """Set up performance monitoring for retraining triggers"""
        try:
            # Check performance every hour
            self.retraining_scheduler.every().hour.do(
                self._performance_check_job
            )
            
            logger.info("✅ Performance monitoring configured")
            
        except Exception as e:
            logger.error(f"❌ Error setting up performance monitoring: {e}")
    
    def _setup_drift_monitoring(self):
        """Set up drift monitoring for retraining triggers"""
        try:
            # Check drift every 30 minutes
            self.retraining_scheduler.every(30).minutes.do(
                self._drift_check_job
            )
            
            logger.info("✅ Drift monitoring configured")
            
        except Exception as e:
            logger.error(f"❌ Error setting up drift monitoring: {e}")
    
    def _start_scheduler(self):
        """Start the scheduler in a background thread"""
        try:
            def run_scheduler():
                while True:
                    self.retraining_scheduler.run_pending()
                    time.sleep(60)  # Check every minute
            
            import threading
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            
            logger.info("✅ Scheduler started in background thread")
            
        except Exception as e:
            logger.error(f"❌ Error starting scheduler: {e}")
    
    async def _scheduled_retraining_job(self, trigger_type: str):
        """Scheduled retraining job"""
        try:
            with self.retraining_lock:
                logger.info(f"🔄 Starting scheduled retraining (trigger: {trigger_type})")
                
                # Check if we have enough new data
                new_data_count = await self._get_new_data_count()
                if new_data_count < self.retraining_config.data_threshold:
                    logger.info(f"⚠️ Insufficient new data ({new_data_count} < {self.retraining_config.data_threshold})")
                    return
                
                # Run retraining
                retraining_result = await self._run_retraining(trigger_type)
                
                # Log retraining event
                await self._log_retraining_event(retraining_result)
                
                logger.info(f"✅ Scheduled retraining completed: {retraining_result['status']}")
                
        except Exception as e:
            logger.error(f"❌ Error in scheduled retraining job: {e}")
            await self._log_retraining_event({
                'status': 'failed',
                'error_message': str(e),
                'trigger_type': trigger_type
            })
    
    async def _performance_check_job(self):
        """Performance check job for retraining triggers"""
        try:
            logger.info("📊 Checking model performance...")
            
            # Get current model performance
            current_performance = await self._get_current_model_performance()
            
            # Check if performance is below threshold
            for model_type, performance in current_performance.items():
                if performance < self.retraining_config.performance_threshold:
                    logger.warning(f"⚠️ Performance below threshold for {model_type}: {performance:.3f} < {self.retraining_config.performance_threshold}")
                    
                    # Trigger retraining
                    await self._run_retraining("performance_triggered", model_type)
            
        except Exception as e:
            logger.error(f"❌ Error in performance check job: {e}")
    
    async def _drift_check_job(self):
        """Drift check job for retraining triggers"""
        try:
            logger.info("🔍 Checking for feature drift...")
            
            # Get recent features for drift detection
            recent_features = await self._get_recent_features()
            reference_features = await self._get_reference_features()
            
            if recent_features is not None and reference_features is not None:
                # Detect drift
                drift_result = await self.detect_feature_drift(recent_features, reference_features)
                
                if drift_result.get('drift_detected', False):
                    logger.warning("⚠️ Feature drift detected, triggering retraining")
                    
                    # Trigger retraining
                    await self._run_retraining("drift_triggered")
            
        except Exception as e:
            logger.error(f"❌ Error in drift check job: {e}")
    
    async def _run_retraining(self, trigger_type: str, model_type: str = None) -> Dict[str, Any]:
        """Run retraining process"""
        try:
            start_time = time.time()
            
            # Run the complete self-training pipeline
            result = await self.run_complete_self_training_pipeline()
            
            training_duration = time.time() - start_time
            
            retraining_result = {
                'timestamp': datetime.utcnow(),
                'trigger_type': trigger_type,
                'model_type': model_type,
                'status': result.get('status', 'unknown'),
                'training_duration': training_duration,
                'total_samples': result.get('total_samples', 0),
                'stored_models': result.get('stored_models', 0)
            }
            
            if result.get('status') == 'success':
                logger.info(f"✅ Retraining completed successfully in {training_duration:.2f}s")
            else:
                logger.error(f"❌ Retraining failed: {result.get('message', 'Unknown error')}")
                retraining_result['error_message'] = result.get('message')
            
            return retraining_result
            
        except Exception as e:
            logger.error(f"❌ Error in retraining process: {e}")
            return {
                'timestamp': datetime.utcnow(),
                'trigger_type': trigger_type,
                'model_type': model_type,
                'status': 'failed',
                'error_message': str(e),
                'training_duration': 0
            }
    
    async def _get_new_data_count(self) -> int:
        """Get count of new data samples since last retraining"""
        try:
            if not self.db_pool:
                return 0
            
            async with self.db_pool.acquire() as conn:
                # Count new samples in ML feature tables
                query = """
                SELECT COUNT(*) as count
                FROM ml_features_ohlcv
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                """
                
                result = await conn.fetchrow(query)
                return result['count'] if result else 0
                
        except Exception as e:
            logger.error(f"❌ Error getting new data count: {e}")
            return 0
    
    async def _get_current_model_performance(self) -> Dict[str, float]:
        """Get current model performance metrics"""
        try:
            if not self.db_pool:
                return {}
            
            async with self.db_pool.acquire() as conn:
                # Get recent performance metrics
                query = """
                SELECT model_type, AVG(accuracy_score) as avg_accuracy
                FROM ml_predictions
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY model_type
                """
                
                results = await conn.fetch(query)
                performance = {}
                
                for row in results:
                    performance[row['model_type']] = float(row['avg_accuracy'] or 0.0)
                
                return performance
                
        except Exception as e:
            logger.error(f"❌ Error getting current performance: {e}")
            return {}
    
    async def _get_recent_features(self) -> Optional[np.ndarray]:
        """Get recent features for drift detection"""
        try:
            if not self.db_pool:
                return None
            
            async with self.db_pool.acquire() as conn:
                # Get recent OHLCV features
                query = """
                SELECT rsi, macd, vwap, atr, bollinger_upper, bollinger_lower
                FROM ml_features_ohlcv
                WHERE timestamp > NOW() - INTERVAL '6 hours'
                ORDER BY timestamp DESC
                LIMIT 100
                """
                
                results = await conn.fetch(query)
                if not results:
                    return None
                
                # Convert to numpy array
                features = []
                for row in results:
                    features.append([
                        float(row['rsi'] or 0),
                        float(row['macd'] or 0),
                        float(row['vwap'] or 0),
                        float(row['atr'] or 0),
                        float(row['bollinger_upper'] or 0),
                        float(row['bollinger_lower'] or 0)
                    ])
                
                return np.array(features)
                
        except Exception as e:
            logger.error(f"❌ Error getting recent features: {e}")
            return None
    
    async def _get_reference_features(self) -> Optional[np.ndarray]:
        """Get reference features for drift detection"""
        try:
            if not self.db_pool:
                return None
            
            async with self.db_pool.acquire() as conn:
                # Get reference features from 7 days ago
                query = """
                SELECT rsi, macd, vwap, atr, bollinger_upper, bollinger_lower
                FROM ml_features_ohlcv
                WHERE timestamp BETWEEN NOW() - INTERVAL '7 days' AND NOW() - INTERVAL '6 days'
                ORDER BY timestamp DESC
                LIMIT 100
                """
                
                results = await conn.fetch(query)
                if not results:
                    return None
                
                # Convert to numpy array
                features = []
                for row in results:
                    features.append([
                        float(row['rsi'] or 0),
                        float(row['macd'] or 0),
                        float(row['vwap'] or 0),
                        float(row['atr'] or 0),
                        float(row['bollinger_upper'] or 0),
                        float(row['bollinger_lower'] or 0)
                    ])
                
                return np.array(features)
                
        except Exception as e:
            logger.error(f"❌ Error getting reference features: {e}")
            return None
    
    async def _log_retraining_event(self, retraining_result: Dict[str, Any]):
        """Log retraining event to database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                # Insert retraining log
                query = """
                INSERT INTO ml_models_metadata (
                    model_name, model_version, model_type, training_timestamp,
                    is_active, performance_metrics, hyperparameters,
                    training_data_size, validation_accuracy, test_accuracy
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                await conn.execute(
                    query,
                    f"retraining_{retraining_result['trigger_type']}",
                    f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    retraining_result.get('model_type', 'ensemble'),
                    retraining_result['timestamp'],
                    retraining_result['status'] == 'success',
                    json.dumps({
                        'trigger_type': retraining_result['trigger_type'],
                        'training_duration': retraining_result['training_duration'],
                        'total_samples': retraining_result['total_samples'],
                        'stored_models': retraining_result['stored_models']
                    }),
                    json.dumps({'automated_retraining': True}),
                    retraining_result.get('total_samples', 0),
                    0.0,  # validation_accuracy (would need to be calculated)
                    0.0   # test_accuracy (would need to be calculated)
                )
                
                logger.info("✅ Retraining event logged to database")
                
        except Exception as e:
            logger.error(f"❌ Error logging retraining event: {e}")
    
    async def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining system status"""
        try:
            status = {
                'automated_retraining_enabled': SCHEDULER_AVAILABLE and self.retraining_scheduler is not None,
                'scheduler_running': self.retraining_scheduler is not None,
                'last_retraining': {},
                'next_scheduled_retraining': {},
                'performance_thresholds': self.retraining_config.performance_threshold,
                'drift_threshold': self.retraining_config.drift_threshold,
                'data_threshold': self.retraining_config.data_threshold,
                'kubernetes_integration': self.k8s_enabled
            }
            
            # Get last retraining info
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    query = """
                    SELECT model_name, model_version, training_timestamp, is_active
                    FROM ml_models_metadata
                    WHERE model_name LIKE 'retraining_%'
                    ORDER BY training_timestamp DESC
                    LIMIT 5
                    """
                    
                    results = await conn.fetch(query)
                    for row in results:
                        status['last_retraining'][row['model_name']] = {
                            'version': row['model_version'],
                            'timestamp': row['training_timestamp'].isoformat(),
                            'active': row['is_active']
                        }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting retraining status: {e}")
            return {'error': str(e)}
    
    async def trigger_manual_retraining(self, model_type: str = None) -> Dict[str, Any]:
        """Trigger manual retraining"""
        try:
            logger.info(f"🔄 Triggering manual retraining for {model_type or 'all models'}")
            
            result = await self._run_retraining("manual", model_type)
            
            logger.info(f"✅ Manual retraining completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in manual retraining: {e}")
            return {'status': 'failed', 'error_message': str(e)}

    # Phase 6: Advanced ML Features Methods
    
    async def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                                     X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        try:
            if not ADVANCED_ML_AVAILABLE:
                logger.warning("⚠️ Advanced ML libraries not available for hyperparameter optimization")
                return {'status': 'failed', 'error': 'Advanced ML libraries not available'}
            
            logger.info(f"🔧 Starting hyperparameter optimization for {model_name}")
            
            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(),
                pruner=MedianPruner()
            )
            
            def objective(trial):
                # Define hyperparameter search space
                if model_name == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                    }
                    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
                elif model_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                    }
                    model = xgb.XGBRegressor(**params, random_state=42)
                else:
                    return float('inf')
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                
                return mse
            
            # Run optimization
            study.optimize(objective, n_trials=self.hyperopt_config.n_trials, timeout=self.hyperopt_config.timeout)
            
            # Store results
            best_params = study.best_params
            best_value = study.best_value
            
            # Log to database
            await self._log_hyperopt_results(model_name, study)
            
            logger.info(f"✅ Hyperparameter optimization completed for {model_name}")
            return {
                'status': 'success',
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hyperparameter optimization: {e}")
            return {'status': 'failed', 'error_message': str(e)}
    
    async def _log_hyperopt_results(self, model_name: str, study: optuna.Study):
        """Log hyperparameter optimization results to database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                for trial in study.trials:
                    query = """
                    INSERT INTO hyperparameter_optimization (
                        model_name, optimization_id, trial_number, hyperparameters,
                        objective_value, objective_name, optimization_status,
                        training_duration_seconds, validation_metrics
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """
                    
                    await conn.execute(
                        query,
                        model_name,
                        study.study_name,
                        trial.number,
                        json.dumps(trial.params),
                        trial.value,
                        'minimize_rmse',
                        trial.state.name,
                        trial.duration.total_seconds() if trial.duration else None,
                        json.dumps({'mse': trial.value})
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error logging hyperopt results: {e}")
    
    async def explain_model_prediction(self, model_name: str, X_sample: np.ndarray, 
                                     feature_names: List[str] = None) -> Dict[str, Any]:
        """Explain model prediction using SHAP, LIME, and ELI5"""
        try:
            if not ADVANCED_ML_AVAILABLE:
                logger.warning("⚠️ Advanced ML libraries not available for model interpretability")
                return {'status': 'failed', 'error': 'Advanced ML libraries not available'}
            
            if model_name not in self.models:
                return {'status': 'failed', 'error': f'Model {model_name} not found'}
            
            model = self.models[model_name]
            explanations = {}
            
            # SHAP explanations
            if self.interpretability_config.shap_enabled:
                try:
                    if model_name not in self.shap_explainers:
                        # Create SHAP explainer
                        if hasattr(model, 'feature_importances_'):
                            self.shap_explainers[model_name] = shap.TreeExplainer(model)
                        else:
                            self.shap_explainers[model_name] = shap.KernelExplainer(model.predict, X_sample[:100])
                    
                    explainer = self.shap_explainers[model_name]
                    shap_values = explainer.shap_values(X_sample)
                    
                    explanations['shap'] = {
                        'values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                        'feature_names': feature_names or [f'feature_{i}' for i in range(X_sample.shape[1])]
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ SHAP explanation failed: {e}")
            
            # LIME explanations
            if self.interpretability_config.lime_enabled:
                try:
                    if model_name not in self.lime_explainers:
                        self.lime_explainers[model_name] = lime.lime_tabular.LimeTabularExplainer(
                            X_sample,
                            feature_names=feature_names or [f'feature_{i}' for i in range(X_sample.shape[1])],
                            class_names=['prediction'],
                            mode='regression'
                        )
                    
                    explainer = self.lime_explainers[model_name]
                    lime_exp = explainer.explain_instance(
                        X_sample[0], 
                        model.predict, 
                        num_features=min(self.interpretability_config.max_features, X_sample.shape[1])
                    )
                    
                    explanations['lime'] = {
                        'feature_weights': dict(lime_exp.as_list()),
                        'score': lime_exp.score
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ LIME explanation failed: {e}")
            
            # ELI5 explanations
            if self.interpretability_config.eli5_enabled:
                try:
                    eli5_explanation = eli5.explain_prediction(
                        model, 
                        X_sample[0],
                        feature_names=feature_names or [f'feature_{i}' for i in range(X_sample.shape[1])]
                    )
                    
                    explanations['eli5'] = {
                        'explanation': str(eli5_explanation),
                        'feature_weights': eli5_explanation.feature_weights
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ ELI5 explanation failed: {e}")
            
            # Store explanations in database
            await self._log_model_explanations(model_name, X_sample[0], explanations)
            
            return {
                'status': 'success',
                'explanations': explanations,
                'model_name': model_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in model explanation: {e}")
            return {'status': 'failed', 'error_message': str(e)}
    
    async def _log_model_explanations(self, model_name: str, X_sample: np.ndarray, explanations: Dict[str, Any]):
        """Log model explanations to database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                prediction_id = f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
                
                for method, explanation in explanations.items():
                    if method == 'shap':
                        shap_values = explanation['values']
                        feature_names = explanation['feature_names']
                        
                        for i, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values)):
                            query = """
                            INSERT INTO model_interpretability (
                                model_name, model_version, prediction_id, feature_name,
                                feature_value, shap_value, interpretation_type
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """
                            
                            await conn.execute(
                                query,
                                model_name,
                                'v1.0',  # Default version
                                prediction_id,
                                feature_name,
                                float(X_sample[i]),
                                float(shap_value),
                                'shap'
                            )
                    
                    elif method == 'lime':
                        feature_weights = explanation['feature_weights']
                        
                        for feature_name, weight in feature_weights:
                            query = """
                            INSERT INTO model_interpretability (
                                model_name, model_version, prediction_id, feature_name,
                                feature_value, lime_value, interpretation_type
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """
                            
                            await conn.execute(
                                query,
                                model_name,
                                'v1.0',
                                prediction_id,
                                feature_name,
                                0.0,  # LIME doesn't provide feature values
                                float(weight),
                                'lime'
                            )
                    
                    elif method == 'eli5':
                        feature_weights = explanation['feature_weights']
                        
                        for feature_name, weight in feature_weights:
                            query = """
                            INSERT INTO model_interpretability (
                                model_name, model_version, prediction_id, feature_name,
                                feature_value, eli5_value, interpretation_type
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """
                            
                            await conn.execute(
                                query,
                                model_name,
                                'v1.0',
                                prediction_id,
                                feature_name,
                                0.0,
                                float(weight),
                                'eli5'
                            )
                            
        except Exception as e:
            logger.error(f"❌ Error logging model explanations: {e}")
    
    async def create_transformer_model(self, model_type: str = 'lstm') -> Dict[str, Any]:
        """Create and train a transformer model (LSTM, GRU, Transformer)"""
        try:
            if not ADVANCED_ML_AVAILABLE:
                logger.warning("⚠️ Advanced ML libraries not available for transformer models")
                return {'status': 'failed', 'error': 'Advanced ML libraries not available'}
            
            logger.info(f"🧠 Creating {model_type} transformer model")
            
            # Create model based on type
            if model_type == 'lstm':
                model = self._create_lstm_model()
            elif model_type == 'gru':
                model = self._create_gru_model()
            elif model_type == 'transformer':
                model = self._create_transformer_model_arch()
            else:
                return {'status': 'failed', 'error': f'Unsupported model type: {model_type}'}
            
            # Store model
            self.transformer_models[model_type] = model
            
            # Log to database
            await self._log_transformer_model(model_type, model)
            
            return {
                'status': 'success',
                'model_type': model_type,
                'model_config': self.transformer_config.__dict__,
                'model_summary': str(model)
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating transformer model: {e}")
            return {'status': 'failed', 'error_message': str(e)}
    
    def _create_lstm_model(self):
        """Create LSTM model using TensorFlow/Keras"""
        model = keras.Sequential([
            keras.layers.LSTM(
                self.transformer_config.hidden_dim,
                return_sequences=True,
                input_shape=(self.transformer_config.sequence_length, self.transformer_config.embedding_dim)
            ),
            keras.layers.Dropout(self.transformer_config.dropout),
            keras.layers.LSTM(self.transformer_config.hidden_dim // 2),
            keras.layers.Dropout(self.transformer_config.dropout),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.transformer_config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_gru_model(self):
        """Create GRU model using TensorFlow/Keras"""
        model = keras.Sequential([
            keras.layers.GRU(
                self.transformer_config.hidden_dim,
                return_sequences=True,
                input_shape=(self.transformer_config.sequence_length, self.transformer_config.embedding_dim)
            ),
            keras.layers.Dropout(self.transformer_config.dropout),
            keras.layers.GRU(self.transformer_config.hidden_dim // 2),
            keras.layers.Dropout(self.transformer_config.dropout),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.transformer_config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_transformer_model_arch(self):
        """Create Transformer model using TensorFlow/Keras"""
        inputs = keras.layers.Input(shape=(self.transformer_config.sequence_length, self.transformer_config.embedding_dim))
        
        # Multi-head attention
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=self.transformer_config.num_heads,
            key_dim=self.transformer_config.embedding_dim
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = keras.layers.Add()([inputs, attention_output])
        attention_output = keras.layers.LayerNormalization()(attention_output)
        
        # Feed forward
        ffn_output = keras.layers.Dense(self.transformer_config.hidden_dim, activation='relu')(attention_output)
        ffn_output = keras.layers.Dense(self.transformer_config.embedding_dim)(ffn_output)
        
        # Add & Norm
        ffn_output = keras.layers.Add()([attention_output, ffn_output])
        ffn_output = keras.layers.LayerNormalization()(ffn_output)
        
        # Global average pooling and output
        pooled_output = keras.layers.GlobalAveragePooling1D()(ffn_output)
        outputs = keras.layers.Dense(1)(pooled_output)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.transformer_config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def _log_transformer_model(self, model_type: str, model):
        """Log transformer model to database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                query = """
                INSERT INTO transformer_models (
                    model_name, model_type, model_config, parameters_count,
                    embedding_dimension, num_layers, num_heads, dropout_rate,
                    learning_rate, batch_size, epochs_trained, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """
                
                # Count parameters
                param_count = model.count_params()
                
                await conn.execute(
                    query,
                    f"{model_type}_model",
                    model_type,
                    json.dumps(self.transformer_config.__dict__),
                    param_count,
                    self.transformer_config.embedding_dim,
                    self.transformer_config.num_layers,
                    self.transformer_config.num_heads,
                    self.transformer_config.dropout,
                    self.transformer_config.learning_rate,
                    self.transformer_config.batch_size,
                    0,  # epochs_trained
                    True
                )
                
        except Exception as e:
            logger.error(f"❌ Error logging transformer model: {e}")
    
    async def create_advanced_ensemble(self, ensemble_type: str = 'voting') -> Dict[str, Any]:
        """Create advanced ensemble model"""
        try:
            logger.info(f"🎯 Creating {ensemble_type} ensemble model")
            
            base_models = []
            for model_name in self.advanced_ensemble_config.base_models:
                if model_name in self.models:
                    base_models.append((model_name, self.models[model_name]))
            
            if not base_models:
                return {'status': 'failed', 'error': 'No base models available'}
            
            # Create ensemble
            if ensemble_type == 'voting':
                if self.advanced_ensemble_config.voting_method == 'soft':
                    ensemble = VotingRegressor(
                        estimators=base_models,
                        weights=self.advanced_ensemble_config.base_model_weights
                    )
                else:
                    ensemble = VotingClassifier(
                        estimators=base_models,
                        voting='hard'
                    )
            
            elif ensemble_type == 'stacking':
                # Create meta-learner
                meta_learner = LinearRegression()
                ensemble = StackingRegressor(
                    estimators=base_models,
                    final_estimator=meta_learner,
                    cv=self.advanced_ensemble_config.cross_validation_folds
                )
            
            else:
                return {'status': 'failed', 'error': f'Unsupported ensemble type: {ensemble_type}'}
            
            # Store ensemble
            self.advanced_ensembles[ensemble_type] = ensemble
            
            # Log to database
            await self._log_ensemble_model(ensemble_type, ensemble, base_models)
            
            return {
                'status': 'success',
                'ensemble_type': ensemble_type,
                'base_models': [name for name, _ in base_models],
                'ensemble_config': self.advanced_ensemble_config.__dict__
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating advanced ensemble: {e}")
            return {'status': 'failed', 'error_message': str(e)}
    
    async def _log_ensemble_model(self, ensemble_type: str, ensemble, base_models: List[Tuple[str, Any]]):
        """Log ensemble model to database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                query = """
                INSERT INTO ensemble_models (
                    ensemble_name, ensemble_type, base_models, ensemble_config,
                    weighting_strategy, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """
                
                await conn.execute(
                    query,
                    f"{ensemble_type}_ensemble",
                    ensemble_type,
                    json.dumps([name for name, _ in base_models]),
                    json.dumps(self.advanced_ensemble_config.__dict__),
                    self.advanced_ensemble_config.weighting_strategy,
                    True
                )
                
        except Exception as e:
            logger.error(f"❌ Error logging ensemble model: {e}")
    
    async def start_experiment_tracking(self, experiment_name: str = None) -> Dict[str, Any]:
        """Start ML experiment tracking"""
        try:
            if not ADVANCED_ML_AVAILABLE:
                logger.warning("⚠️ Advanced ML libraries not available for experiment tracking")
                return {'status': 'failed', 'error': 'Advanced ML libraries not available'}
            
            experiment_name = experiment_name or self.experiment_tracking_config.experiment_name
            
            if self.experiment_tracking_config.tracking_backend == 'mlflow':
                mlflow.set_experiment(experiment_name)
                run = mlflow.start_run()
                self.experiment_runs[experiment_name] = run
                
                # Log configuration
                mlflow.log_params(self.experiment_tracking_config.__dict__)
                
            elif self.experiment_tracking_config.tracking_backend == 'wandb':
                wandb.init(
                    project=experiment_name,
                    config=self.experiment_tracking_config.__dict__
                )
                self.wandb_runs[experiment_name] = wandb.run
            
            # Log to database
            await self._log_experiment_start(experiment_name)
            
            return {
                'status': 'success',
                'experiment_name': experiment_name,
                'tracking_backend': self.experiment_tracking_config.tracking_backend
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting experiment tracking: {e}")
            return {'status': 'failed', 'error_message': str(e)}
    
    async def _log_experiment_start(self, experiment_name: str):
        """Log experiment start to database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                query = """
                INSERT INTO ml_experiments (
                    experiment_id, experiment_name, experiment_type, status, config, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """
                
                await conn.execute(
                    query,
                    f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    experiment_name,
                    'ml_training',
                    'running',
                    json.dumps(self.experiment_tracking_config.__dict__),
                    'system'
                )
                
        except Exception as e:
            logger.error(f"❌ Error logging experiment start: {e}")
    
    async def get_phase6_advanced_features_summary(self) -> Dict[str, Any]:
        """Get summary of Phase 6 advanced ML features"""
        try:
            summary = {
                'phase': 'Phase 6: Advanced ML Features',
                'status': 'implemented',
                'features': {
                    'hyperparameter_optimization': {
                        'enabled': ADVANCED_ML_AVAILABLE,
                        'optimizer': self.hyperopt_config.optimizer,
                        'n_trials': self.hyperopt_config.n_trials,
                        'studies_count': len(self.hyperopt_studies)
                    },
                    'model_interpretability': {
                        'enabled': ADVANCED_ML_AVAILABLE,
                        'shap_enabled': self.interpretability_config.shap_enabled,
                        'lime_enabled': self.interpretability_config.lime_enabled,
                        'eli5_enabled': self.interpretability_config.eli5_enabled,
                        'explainers_count': len(self.shap_explainers) + len(self.lime_explainers)
                    },
                    'transformer_models': {
                        'enabled': ADVANCED_ML_AVAILABLE,
                        'supported_types': ['lstm', 'gru', 'transformer'],
                        'models_count': len(self.transformer_models),
                        'config': self.transformer_config.__dict__
                    },
                    'advanced_ensembles': {
                        'enabled': True,
                        'ensemble_types': ['voting', 'stacking', 'blending', 'bagging'],
                        'ensembles_count': len(self.advanced_ensembles),
                        'config': self.advanced_ensemble_config.__dict__
                    },
                    'experiment_tracking': {
                        'enabled': ADVANCED_ML_AVAILABLE,
                        'tracking_backend': self.experiment_tracking_config.tracking_backend,
                        'active_runs': len(self.experiment_runs) + len(self.wandb_runs)
                    }
                },
                'database_tables': [
                    'hyperparameter_optimization',
                    'model_interpretability', 
                    'ml_experiments',
                    'advanced_feature_engineering',
                    'transformer_models',
                    'ensemble_models',
                    'feature_selection_history',
                    'model_performance_comparison'
                ],
                'libraries_available': ADVANCED_ML_AVAILABLE,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting Phase 6 summary: {e}")
            return {'error': str(e)}
