"""
Machine Learning Strategy Enhancement for AlphaPulse
Advanced ML-powered trading strategies and ensemble methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .feature_engineering import FeatureExtractor
from .model_registry import ModelRegistry
from .risk_management import RiskManager, risk_manager
from .market_regime_detection import MarketRegimeDetector, market_regime_detector

logger = logging.getLogger(__name__)

class MLStrategyType(Enum):
    """Machine learning strategy types"""
    ENSEMBLE_VOTING = "ensemble_voting"
    STACKING = "stacking"
    NEURAL_NETWORK = "neural_network"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"

class ModelType(Enum):
    """Individual model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    prediction_time: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class StrategySignal:
    """ML strategy trading signal"""
    strategy_name: str
    strategy_type: MLStrategyType
    prediction: str  # 'buy', 'sell', 'hold'
    confidence: float
    probability: Dict[str, float]
    model_weights: Dict[str, float]
    ensemble_score: float
    market_regime: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleConfig:
    """Ensemble configuration"""
    strategy_type: MLStrategyType
    base_models: List[ModelType]
    voting_method: str = "soft"
    weights: Optional[List[float]] = None
    meta_learner: Optional[ModelType] = None
    adaptive_weights: bool = True

class BaseMLModel:
    """Base class for machine learning models"""
    
    def __init__(self, model_type: ModelType, **kwargs):
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.performance = None
        
        logger.info(f"BaseMLModel initialized with {model_type.value}")
    
    def _create_model(self, **kwargs):
        """Create the specific model based on type"""
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        elif self.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif self.model_type == ModelType.SVM:
            return SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
        elif self.model_type == ModelType.NEURAL_NETWORK:
            return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the model"""
        try:
            start_time = datetime.now()
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1] if len(self.model.classes_) > 1 else None
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            self.performance = ModelPerformance(
                model_name=f"{self.model_type.value}_model",
                model_type=self.model_type,
                accuracy=accuracy_score(y, y_pred),
                precision=precision_score(y, y_pred, average='weighted'),
                recall=recall_score(y, y_pred, average='weighted'),
                f1_score=f1_score(y, y_pred, average='weighted'),
                roc_auc=roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else 0.0,
                training_time=training_time,
                prediction_time=0.0
            )
            
            self.is_trained = True
            logger.info(f"Model {self.model_type.value} trained successfully")
            return self.performance
            
        except Exception as e:
            logger.error(f"Training error for {self.model_type.value}: {e}")
            return None
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)
            
            return y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Prediction error for {self.model_type.value}: {e}")
            return np.array([]), np.array([])

class EnsembleStrategy:
    """Advanced Ensemble machine learning strategy with multi-model voting"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: Dict[str, BaseMLModel] = {}
        self.ensemble_model = None
        self.model_weights: Dict[str, float] = {}
        self.is_trained = False
        self.performance_history: List[ModelPerformance] = []
        
        # Phase 2.3: Enhanced ensemble tracking
        self.ensemble_analysis = {
            'voting_method': self.config.voting_method,
            'model_count': 0,
            'diversity_score': 0.0,
            'agreement_ratio': 0.0,
            'individual_predictions': {},
            'model_weights': {},
            'performance_metrics': {}
        }
        
        # Initialize base models
        for model_type in config.base_models:
            model_name = f"{model_type.value}_model"
            self.models[model_name] = BaseMLModel(model_type)
        
        self.ensemble_analysis['model_count'] = len(self.models)
        
        self._create_ensemble()
        logger.info(f"Advanced EnsembleStrategy initialized with {config.strategy_type.value} and {len(self.models)} models")
    
    def _create_ensemble(self):
        """Create advanced ensemble model"""
        try:
            estimators = [(name, model.model) for name, model in self.models.items()]
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting=self.config.voting_method,
                weights=self.config.weights
            )
            
            # Initialize equal weights
            for name in self.models.keys():
                self.model_weights[name] = 1.0 / len(self.models)
            
            self.ensemble_analysis['model_weights'] = self.model_weights.copy()
            
        except Exception as e:
            logger.error(f"Ensemble creation error: {e}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the advanced ensemble strategy"""
        try:
            start_time = datetime.now()
            
            # Train individual models
            model_performances = {}
            for name, model in self.models.items():
                performance = model.train(X, y)
                if performance:
                    model_performances[name] = performance
                    self.performance_history.append(performance)
                    
                    # Store performance metrics
                    self.ensemble_analysis['performance_metrics'][name] = {
                        'accuracy': performance.accuracy,
                        'f1_score': performance.f1_score,
                        'roc_auc': performance.roc_auc
                    }
            
            # Calculate adaptive weights
            if self.config.adaptive_weights:
                self._calculate_adaptive_weights(model_performances)
            
            # Train ensemble model
            if self.ensemble_model:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.ensemble_model.fit(X_scaled, y)
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            
            # Calculate ensemble diversity
            self._calculate_ensemble_diversity(X, y)
            
            logger.info(f"Advanced ensemble strategy trained successfully. Diversity: {self.ensemble_analysis['diversity_score']:.4f}")
            return self.performance_history[-1] if self.performance_history else None
            
        except Exception as e:
            logger.error(f"Ensemble training error: {e}")
            return None
    
    def predict(self, X: np.ndarray) -> StrategySignal:
        """Make advanced ensemble prediction with multi-model voting"""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble not trained")
            
            # Get individual model predictions
            model_predictions = {}
            model_probabilities = {}
            
            for name, model in self.models.items():
                if model.is_trained:
                    y_pred, y_proba = model.predict(X)
                    if len(y_pred) > 0:
                        model_predictions[name] = y_pred
                        model_probabilities[name] = y_proba
                        
                        # Store individual predictions for analysis
                        self.ensemble_analysis['individual_predictions'][name] = {
                            'prediction': y_pred[0] if len(y_pred) > 0 else 'hold',
                            'confidence': np.max(y_proba[0]) if len(y_proba) > 0 else 0.0,
                            'probabilities': y_proba[0].tolist() if len(y_proba) > 0 else [0.33, 0.33, 0.34]
                        }
            
            # Calculate weighted prediction
            weighted_prediction = self._calculate_weighted_prediction(model_predictions, model_probabilities)
            
            # Calculate ensemble agreement ratio
            self._calculate_agreement_ratio(model_predictions)
            
            # Create strategy signal
            signal = StrategySignal(
                strategy_name=f"advanced_{self.config.strategy_type.value}_ensemble",
                strategy_type=self.config.strategy_type,
                prediction=weighted_prediction['prediction'],
                confidence=weighted_prediction['confidence'],
                probability=weighted_prediction['probability'],
                model_weights=self.model_weights,
                ensemble_score=weighted_prediction['ensemble_score'],
                market_regime="unknown"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return None
    
    def _calculate_adaptive_weights(self, model_performances: Dict[str, ModelPerformance]):
        """Calculate adaptive weights based on model performance"""
        try:
            total_score = 0.0
            for name, performance in model_performances.items():
                score = performance.f1_score
                self.model_weights[name] = score
                total_score += score
            
            # Normalize weights
            if total_score > 0:
                for name in self.model_weights:
                    self.model_weights[name] /= total_score
            else:
                equal_weight = 1.0 / len(self.model_weights)
                for name in self.model_weights:
                    self.model_weights[name] = equal_weight
                    
        except Exception as e:
            logger.error(f"Adaptive weights calculation error: {e}")
    
    def _calculate_ensemble_diversity(self, X: np.ndarray, y: np.ndarray):
        """Calculate ensemble diversity score"""
        try:
            predictions = []
            for name, model in self.models.items():
                if model.is_trained:
                    pred, _ = model.predict(X)
                    if len(pred) > 0:
                        predictions.append(pred)
            
            # Calculate pairwise diversity
            diversity_scores = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    # Calculate disagreement ratio
                    disagreement = np.mean(predictions[i] != predictions[j])
                    diversity_scores.append(disagreement)
            
            if diversity_scores:
                self.ensemble_analysis['diversity_score'] = np.mean(diversity_scores)
            else:
                self.ensemble_analysis['diversity_score'] = 0.0
                
        except Exception as e:
            logger.error(f"Diversity calculation error: {e}")
            self.ensemble_analysis['diversity_score'] = 0.0
    
    def _calculate_agreement_ratio(self, model_predictions: Dict[str, np.ndarray]):
        """Calculate agreement ratio among models"""
        try:
            if not model_predictions:
                self.ensemble_analysis['agreement_ratio'] = 0.0
                return
            
            # Get predictions for the latest data point
            latest_predictions = []
            for name, preds in model_predictions.items():
                if len(preds) > 0:
                    latest_predictions.append(preds[0])
            
            if not latest_predictions:
                self.ensemble_analysis['agreement_ratio'] = 0.0
                return
            
            # Calculate agreement ratio
            unique_predictions = set(latest_predictions)
            if len(unique_predictions) == 1:
                self.ensemble_analysis['agreement_ratio'] = 1.0  # Perfect agreement
            else:
                # Calculate ratio of most common prediction
                from collections import Counter
                prediction_counts = Counter(latest_predictions)
                most_common_count = prediction_counts.most_common(1)[0][1]
                self.ensemble_analysis['agreement_ratio'] = most_common_count / len(latest_predictions)
                
        except Exception as e:
            logger.error(f"Agreement ratio calculation error: {e}")
            self.ensemble_analysis['agreement_ratio'] = 0.0
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """Get comprehensive ensemble analysis"""
        return {
            'voting_method': self.ensemble_analysis['voting_method'],
            'model_count': self.ensemble_analysis['model_count'],
            'diversity_score': self.ensemble_analysis['diversity_score'],
            'agreement_ratio': self.ensemble_analysis['agreement_ratio'],
            'individual_predictions': self.ensemble_analysis['individual_predictions'],
            'model_weights': self.model_weights,
            'performance_metrics': self.ensemble_analysis['performance_metrics'],
            'is_trained': self.is_trained
        }
    
    def _calculate_weighted_prediction(self, model_predictions: Dict[str, np.ndarray], 
                                     model_probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate weighted prediction from individual models"""
        try:
            if not model_predictions:
                return {
                    'prediction': 'hold',
                    'confidence': 0.0,
                    'probability': {'buy': 0.33, 'sell': 0.33, 'hold': 0.34},
                    'ensemble_score': 0.0
                }
            
            # Calculate weighted probabilities
            weighted_proba = np.zeros(3)  # buy, sell, hold
            total_weight = 0.0
            
            for name, proba in model_probabilities.items():
                weight = self.model_weights.get(name, 1.0 / len(model_predictions))
                
                if proba.shape[1] == 3:  # buy, sell, hold
                    weighted_proba += weight * proba[0]
                elif proba.shape[1] == 2:  # binary classification
                    weighted_proba[0] += weight * proba[0, 1] * 0.5  # buy
                    weighted_proba[1] += weight * proba[0, 1] * 0.5  # sell
                    weighted_proba[2] += weight * proba[0, 0]  # hold
                
                total_weight += weight
            
            if total_weight > 0:
                weighted_proba /= total_weight
            
            # Determine prediction
            prediction_map = {0: 'buy', 1: 'sell', 2: 'hold'}
            prediction_idx = np.argmax(weighted_proba)
            prediction = prediction_map[prediction_idx]
            
            # Calculate confidence and ensemble score
            confidence = weighted_proba[prediction_idx]
            ensemble_score = np.mean([perf.f1_score for perf in self.performance_history[-len(self.models):]])
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probability': {
                    'buy': weighted_proba[0],
                    'sell': weighted_proba[1],
                    'hold': weighted_proba[2]
                },
                'ensemble_score': ensemble_score
            }
            
        except Exception as e:
            logger.error(f"Weighted prediction calculation error: {e}")
            return {
                'prediction': 'hold',
                'confidence': 0.0,
                'probability': {'buy': 0.33, 'sell': 0.33, 'hold': 0.34},
                'ensemble_score': 0.0
            }

class MLStrategyEnhancement:
    """Machine Learning Strategy Enhancement system"""
    
    def __init__(self,
                 feature_extractor: FeatureExtractor = None,
                 model_registry: ModelRegistry = None,
                 risk_manager: RiskManager = None,
                 market_regime_detector: MarketRegimeDetector = None,
                 enable_adaptive_learning: bool = True,
                 max_strategies: int = 10):
        
        # Dependencies
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.model_registry = model_registry or ModelRegistry()
        self.risk_manager = risk_manager or risk_manager
        self.market_regime_detector = market_regime_detector or market_regime_detector
        
        # Configuration
        self.enable_adaptive_learning = enable_adaptive_learning
        self.max_strategies = max_strategies
        
        # Strategy management
        self.strategies: Dict[str, EnsembleStrategy] = {}
        self.performance_tracker: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        logger.info("MLStrategyEnhancement initialized")
    
    def _initialize_default_strategies(self):
        """Initialize default ML strategies"""
        try:
            # Ensemble Voting Strategy
            voting_config = EnsembleConfig(
                strategy_type=MLStrategyType.ENSEMBLE_VOTING,
                base_models=[
                    ModelType.RANDOM_FOREST,
                    ModelType.GRADIENT_BOOSTING,
                    ModelType.LOGISTIC_REGRESSION
                ],
                voting_method="soft",
                adaptive_weights=True
            )
            self.strategies["ensemble_voting"] = EnsembleStrategy(voting_config)
            
            # Neural Network Strategy
            nn_config = EnsembleConfig(
                strategy_type=MLStrategyType.NEURAL_NETWORK,
                base_models=[
                    ModelType.NEURAL_NETWORK,
                    ModelType.SVM
                ],
                voting_method="soft",
                adaptive_weights=True
            )
            self.strategies["neural_network"] = EnsembleStrategy(nn_config)
            
            logger.info(f"Initialized {len(self.strategies)} default strategies")
            
        except Exception as e:
            logger.error(f"Default strategy initialization error: {e}")
    
    def add_strategy(self, name: str, config: EnsembleConfig) -> bool:
        """Add a new ML strategy"""
        try:
            if len(self.strategies) >= self.max_strategies:
                logger.warning(f"Maximum strategies reached ({self.max_strategies})")
                return False
            
            if name in self.strategies:
                logger.warning(f"Strategy {name} already exists")
                return False
            
            strategy = EnsembleStrategy(config)
            self.strategies[name] = strategy
            
            logger.info(f"Added strategy: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Strategy addition error: {e}")
            return False
    
    def train_strategy(self, strategy_name: str, X: np.ndarray, y: np.ndarray) -> Optional[ModelPerformance]:
        """Train a specific strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Strategy {strategy_name} not found")
                return None
            
            strategy = self.strategies[strategy_name]
            performance = strategy.train(X, y)
            
            if performance:
                logger.info(f"Strategy {strategy_name} trained successfully")
                return performance
            else:
                logger.error(f"Strategy {strategy_name} training failed")
                return None
                
        except Exception as e:
            logger.error(f"Strategy training error: {e}")
            return None
    
    def train_all_strategies(self, X: np.ndarray, y: np.ndarray) -> Dict[str, ModelPerformance]:
        """Train all strategies"""
        try:
            results = {}
            
            for name, strategy in self.strategies.items():
                logger.info(f"Training strategy: {name}")
                performance = self.train_strategy(name, X, y)
                if performance:
                    results[name] = performance
            
            logger.info(f"Trained {len(results)} strategies successfully")
            return results
            
        except Exception as e:
            logger.error(f"All strategies training error: {e}")
            return {}
    
    def predict(self, strategy_name: str, features: np.ndarray, market_regime: str = "unknown") -> Optional[StrategySignal]:
        """Make prediction using a specific strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Strategy {strategy_name} not found")
                return None
            
            strategy = self.strategies[strategy_name]
            signal = strategy.predict(features)
            
            if signal:
                signal.market_regime = market_regime
                self.performance_tracker[strategy_name].append(signal.confidence)
            
            return signal
            
        except Exception as e:
            logger.error(f"Strategy prediction error: {e}")
            return None
    
    def predict_best_strategy(self, features: np.ndarray, market_regime: str = "unknown") -> Optional[StrategySignal]:
        """Make prediction using the best performing strategy"""
        try:
            # Find strategy with highest average confidence
            best_signal = None
            best_confidence = 0.0
            
            for name in self.strategies:
                signal = self.predict(name, features, market_regime)
                if signal and signal.confidence > best_confidence:
                    best_confidence = signal.confidence
                    best_signal = signal
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Best strategy prediction error: {e}")
            return None
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a strategy"""
        try:
            if strategy_name not in self.strategies:
                return None
            
            strategy = self.strategies[strategy_name]
            # Return the last performance from history if available
            if strategy.performance_history:
                return strategy.performance_history[-1]
            return None
            
        except Exception as e:
            logger.error(f"Strategy performance error: {e}")
            return None
    
    def get_all_performances(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all strategies"""
        try:
            performances = {}
            
            for name, strategy in self.strategies.items():
                if strategy.performance_history:
                    performances[name] = strategy.performance_history[-1]
            
            return performances
            
        except Exception as e:
            logger.error(f"All performances error: {e}")
            return {}

# Global ML strategy enhancement instance
ml_strategy_enhancement = MLStrategyEnhancement(
    feature_extractor=FeatureExtractor(),
    model_registry=ModelRegistry(),
    risk_manager=risk_manager,
    market_regime_detector=market_regime_detector,
    enable_adaptive_learning=True
)
