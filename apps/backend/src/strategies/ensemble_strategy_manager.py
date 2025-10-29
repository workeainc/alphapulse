"""
Ensemble Strategy Manager for AlphaPlus
Learns which strategies perform best under each market regime using ML
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import time
import json
import pickle
from pathlib import Path

# ML imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ML libraries not available. Ensemble learning will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    symbol: str
    timeframe: str
    market_regime: str
    timestamp: datetime
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    avg_profit: float
    sharpe_ratio: float
    success: bool = True

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    strategy_name: str
    confidence: float
    predicted_performance: float
    market_regime: str
    features: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EnsembleStrategyManager:
    """
    Ensemble learning system that learns which strategies perform best under each market regime
    Uses logistic regression and LightGBM to predict strategy performance
    """
    
    def __init__(self, 
                 model_save_path: str = "models/ensemble",
                 min_training_samples: int = 100,
                 retrain_interval_hours: int = 24):
        
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.min_training_samples = min_training_samples
        self.retrain_interval_hours = retrain_interval_hours
        
        # Performance history
        self.performance_history: List[StrategyPerformance] = []
        self.strategy_registry: Dict[str, Dict[str, Any]] = {}
        
        # ML models
        self.regime_classifier = None
        self.strategy_selector = None
        self.performance_predictor = None
        
        # Model metadata
        self.model_metadata = {
            'last_trained': None,
            'training_samples': 0,
            'accuracy': 0.0,
            'feature_importance': {}
        }
        
        # Feature engineering
        self.feature_columns = [
            'rsi', 'macd', 'bb_position', 'atr', 'volume_ratio',
            'price_change_1h', 'price_change_4h', 'price_change_1d',
            'volatility_1h', 'volatility_4h', 'volatility_1d',
            'trend_strength', 'support_resistance_distance'
        ]
        
        # Market regime mapping
        self.market_regimes = ['trending', 'ranging', 'volatile', 'consolidation']
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'avg_prediction_confidence': 0.0,
            'model_accuracy': 0.0
        }
        
        logger.info("Ensemble Strategy Manager initialized")
    
    def register_strategy(self, name: str, metadata: Dict[str, Any] = None):
        """Register a strategy for ensemble learning"""
        self.strategy_registry[name] = metadata or {}
        logger.info(f"📝 Registered strategy for ensemble: {name}")
    
    async def record_strategy_performance(self, performance: StrategyPerformance):
        """Record strategy performance for training"""
        self.performance_history.append(performance)
        
        # Check if we have enough data for training
        if len(self.performance_history) >= self.min_training_samples:
            await self._check_and_retrain_models()
    
    async def predict_best_strategy(self, 
                                  symbol: str,
                                  timeframe: str,
                                  market_data: Dict[str, Any],
                                  current_regime: str = None) -> Optional[EnsemblePrediction]:
        """
        Predict the best strategy for current market conditions
        """
        if not ML_AVAILABLE or self.strategy_selector is None:
            return None
        
        try:
            # Extract features
            features = self._extract_features(market_data)
            
            if not features:
                return None
            
            # Predict market regime if not provided
            if current_regime is None:
                current_regime = self._predict_market_regime(features)
            
            # Predict best strategy
            strategy_predictions = self._predict_strategy_performance(
                features, current_regime
            )
            
            if not strategy_predictions:
                return None
            
            # Select best strategy
            best_strategy = max(strategy_predictions, key=lambda x: x['confidence'])
            
            # Create ensemble prediction
            prediction = EnsemblePrediction(
                strategy_name=best_strategy['strategy_name'],
                confidence=best_strategy['confidence'],
                predicted_performance=best_strategy['predicted_performance'],
                market_regime=current_regime,
                features=features,
                metadata={
                    'all_predictions': strategy_predictions,
                    'feature_importance': self.model_metadata.get('feature_importance', {})
                }
            )
            
            # Update statistics
            self.stats['total_predictions'] += 1
            self.stats['avg_prediction_confidence'] = (
                (self.stats['avg_prediction_confidence'] * (self.stats['total_predictions'] - 1) + prediction.confidence) /
                self.stats['total_predictions']
            )
            
            logger.debug(f"🎯 Ensemble prediction: {prediction.strategy_name} (confidence: {prediction.confidence:.3f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error in ensemble prediction: {e}")
            return None
    
    def _extract_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from market data"""
        features = {}
        
        try:
            # Technical indicators
            features['rsi'] = market_data.get('rsi', 50.0)
            features['macd'] = market_data.get('macd', 0.0)
            features['bb_position'] = market_data.get('bb_position', 0.5)
            features['atr'] = market_data.get('atr', 0.0)
            features['volume_ratio'] = market_data.get('volume_ratio', 1.0)
            
            # Price changes
            features['price_change_1h'] = market_data.get('price_change_1h', 0.0)
            features['price_change_4h'] = market_data.get('price_change_4h', 0.0)
            features['price_change_1d'] = market_data.get('price_change_1d', 0.0)
            
            # Volatility
            features['volatility_1h'] = market_data.get('volatility_1h', 0.0)
            features['volatility_4h'] = market_data.get('volatility_4h', 0.0)
            features['volatility_1d'] = market_data.get('volatility_1d', 0.0)
            
            # Trend strength
            features['trend_strength'] = market_data.get('trend_strength', 0.0)
            features['support_resistance_distance'] = market_data.get('support_resistance_distance', 0.0)
            
            # Normalize features
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return {}
    
    def _predict_market_regime(self, features: Dict[str, float]) -> str:
        """Predict market regime using trained classifier"""
        if self.regime_classifier is None:
            return 'trending'  # Default regime
        
        try:
            # Convert features to array
            feature_array = np.array([features.get(col, 0.0) for col in self.feature_columns])
            feature_array = feature_array.reshape(1, -1)
            
            # Predict regime
            regime_idx = self.regime_classifier.predict(feature_array)[0]
            regime = self.market_regimes[regime_idx]
            
            return regime
            
        except Exception as e:
            logger.error(f"❌ Error predicting market regime: {e}")
            return 'trending'
    
    def _predict_strategy_performance(self, 
                                    features: Dict[str, float], 
                                    market_regime: str) -> List[Dict[str, Any]]:
        """Predict performance for all strategies"""
        if self.strategy_selector is None:
            return []
        
        try:
            # Convert features to array
            feature_array = np.array([features.get(col, 0.0) for col in self.feature_columns])
            feature_array = feature_array.reshape(1, -1)
            
            predictions = []
            
            # Predict for each registered strategy
            for strategy_name in self.strategy_registry:
                # Create feature vector with strategy and regime info
                strategy_features = np.concatenate([
                    feature_array,
                    np.array([[self.market_regimes.index(market_regime)]]),
                    np.array([[list(self.strategy_registry.keys()).index(strategy_name)]])
                ], axis=1)
                
                # Predict confidence and performance
                confidence = self.strategy_selector.predict_proba(strategy_features)[0][1]
                predicted_performance = self.performance_predictor.predict(strategy_features)[0]
                
                predictions.append({
                    'strategy_name': strategy_name,
                    'confidence': confidence,
                    'predicted_performance': predicted_performance,
                    'market_regime': market_regime
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error predicting strategy performance: {e}")
            return []
    
    async def _check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        if not ML_AVAILABLE:
            return
        
        current_time = datetime.now(timezone.utc)
        
        # Check if enough time has passed since last training
        if (self.model_metadata['last_trained'] and 
            (current_time - self.model_metadata['last_trained']).total_seconds() < self.retrain_interval_hours * 3600):
            return
        
        # Check if we have enough data
        if len(self.performance_history) < self.min_training_samples:
            return
        
        logger.info("🔄 Retraining ensemble models...")
        
        try:
            await self._train_models()
            self.model_metadata['last_trained'] = current_time
            
            # Save models
            await self._save_models()
            
            logger.info("✅ Ensemble models retrained successfully")
            
        except Exception as e:
            logger.error(f"❌ Error retraining ensemble models: {e}")
    
    async def _train_models(self):
        """Train ensemble models"""
        # Prepare training data
        X, y_regime, y_strategy, y_performance = self._prepare_training_data()
        
        if len(X) < self.min_training_samples:
            logger.warning(f"Not enough training samples: {len(X)} < {self.min_training_samples}")
            return
        
        # Split data
        X_train, X_test, y_regime_train, y_regime_test, y_strategy_train, y_strategy_test, y_performance_train, y_performance_test = train_test_split(
            X, y_regime, y_strategy, y_performance, test_size=0.2, random_state=42
        )
        
        # Train market regime classifier
        self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regime_classifier.fit(X_train, y_regime_train)
        
        regime_accuracy = accuracy_score(y_regime_test, self.regime_classifier.predict(X_test))
        
        # Train strategy selector
        self.strategy_selector = LogisticRegression(random_state=42)
        self.strategy_selector.fit(X_train, y_strategy_train)
        
        strategy_accuracy = accuracy_score(y_strategy_test, self.strategy_selector.predict(X_test))
        
        # Train performance predictor
        self.performance_predictor = lgb.LGBMRegressor(random_state=42)
        self.performance_predictor.fit(X_train, y_performance_train)
        
        # Update metadata
        self.model_metadata.update({
            'training_samples': len(X),
            'accuracy': (regime_accuracy + strategy_accuracy) / 2,
            'regime_accuracy': regime_accuracy,
            'strategy_accuracy': strategy_accuracy,
            'feature_importance': dict(zip(self.feature_columns, self.regime_classifier.feature_importances_))
        })
        
        self.stats['model_accuracy'] = self.model_metadata['accuracy']
        
        logger.info(f"📊 Model training complete - Accuracy: {self.model_metadata['accuracy']:.3f}")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from performance history"""
        X = []
        y_regime = []
        y_strategy = []
        y_performance = []
        
        for performance in self.performance_history:
            # Create feature vector (simplified - in practice, you'd need historical market data)
            features = np.array([
                performance.win_rate,
                performance.profit_factor,
                performance.max_drawdown,
                performance.avg_profit,
                performance.sharpe_ratio
            ])
            
            # Pad with zeros to match feature columns
            padded_features = np.zeros(len(self.feature_columns))
            padded_features[:len(features)] = features
            
            X.append(padded_features)
            
            # Target variables
            y_regime.append(self.market_regimes.index(performance.market_regime))
            y_strategy.append(1 if performance.success else 0)
            y_performance.append(performance.win_rate)
        
        return np.array(X), np.array(y_regime), np.array(y_strategy), np.array(y_performance)
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save regime classifier
            with open(self.model_save_path / 'regime_classifier.pkl', 'wb') as f:
                pickle.dump(self.regime_classifier, f)
            
            # Save strategy selector
            with open(self.model_save_path / 'strategy_selector.pkl', 'wb') as f:
                pickle.dump(self.strategy_selector, f)
            
            # Save performance predictor
            with open(self.model_save_path / 'performance_predictor.pkl', 'wb') as f:
                pickle.dump(self.performance_predictor, f)
            
            # Save metadata
            with open(self.model_save_path / 'model_metadata.json', 'w') as f:
                json.dump({
                    'last_trained': self.model_metadata['last_trained'].isoformat() if self.model_metadata['last_trained'] else None,
                    'training_samples': self.model_metadata['training_samples'],
                    'accuracy': self.model_metadata['accuracy'],
                    'feature_importance': self.model_metadata['feature_importance']
                }, f, indent=2)
            
            logger.info("💾 Ensemble models saved to disk")
            
        except Exception as e:
            logger.error(f"❌ Error saving ensemble models: {e}")
    
    async def load_models(self):
        """Load trained models from disk"""
        try:
            # Load regime classifier
            with open(self.model_save_path / 'regime_classifier.pkl', 'rb') as f:
                self.regime_classifier = pickle.load(f)
            
            # Load strategy selector
            with open(self.model_save_path / 'strategy_selector.pkl', 'rb') as f:
                self.strategy_selector = pickle.load(f)
            
            # Load performance predictor
            with open(self.model_save_path / 'performance_predictor.pkl', 'rb') as f:
                self.performance_predictor = pickle.load(f)
            
            # Load metadata
            with open(self.model_save_path / 'model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.model_metadata.update(metadata)
                if metadata['last_trained']:
                    self.model_metadata['last_trained'] = datetime.fromisoformat(metadata['last_trained'])
            
            logger.info("📂 Ensemble models loaded from disk")
            
        except FileNotFoundError:
            logger.info("📂 No saved ensemble models found - will train when data is available")
        except Exception as e:
            logger.error(f"❌ Error loading ensemble models: {e}")
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble learning statistics"""
        return {
            'performance_history_count': len(self.performance_history),
            'registered_strategies': list(self.strategy_registry.keys()),
            'model_metadata': self.model_metadata,
            'stats': self.stats,
            'ml_available': ML_AVAILABLE
        }
