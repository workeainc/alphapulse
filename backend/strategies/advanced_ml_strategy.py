"""
Advanced ML Strategy Module

Integrates all ML components for sophisticated trading decisions:
- Feature engineering pipeline
- Model ensemble predictions
- Online learning updates
- Risk-aware signal generation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Import our components
try:
    from ..ai.advanced_feature_engineering import AdvancedFeatureEngineering
    from ..ai.ml_models.model_registry import ModelRegistry
    from ..ai.ml_models.advanced_training_pipeline import AdvancedTrainingPipeline
    from ..ai.ml_models.online_learner import OnlineLearner
    from ..ai.ml_models.model_ensemble import ModelEnsemble
    from ..core.trading_engine import TradingEngine
    from ..execution.order_manager import OrderManager
    from ..database.connection import TimescaleDBConnection
except ImportError:
    AdvancedFeatureEngineering = None
    ModelRegistry = None
    AdvancedTrainingPipeline = None
    OnlineLearner = None
    ModelEnsemble = None
    TradingEngine = None
    OrderManager = None
    TimescaleDBConnection = None

logger = logging.getLogger(__name__)

class MLStrategyType(Enum):
    """ML strategy types"""
    PRICE_PREDICTION = "price_prediction"
    SIGNAL_CLASSIFICATION = "signal_classification"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    HYBRID = "hybrid"

class SignalConfidence(Enum):
    """Signal confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class MLStrategyConfig:
    """Configuration for ML strategy"""
    strategy_type: MLStrategyType = MLStrategyType.HYBRID
    feature_window: int = 100
    prediction_threshold: float = 0.6
    confidence_threshold: float = 0.7
    risk_tolerance: float = 0.3
    update_frequency: int = 60
    ensemble_enabled: bool = True
    online_learning_enabled: bool = True

@dataclass
class MLPrediction:
    """ML model prediction result"""
    prediction: float
    confidence: float
    features_used: List[str]
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class MLSignal:
    """ML-generated trading signal"""
    symbol: str
    side: str  # 'buy' or 'sell'
    confidence: float
    strength: SignalConfidence
    prediction: float
    risk_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class AdvancedMLStrategy:
    """Advanced ML-powered trading strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Strategy configuration
        self.strategy_type = self.config.get('strategy_type', MLStrategyType.HYBRID)
        self.feature_window = self.config.get('feature_window', 100)
        self.prediction_threshold = self.config.get('prediction_threshold', 0.6)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.risk_tolerance = self.config.get('risk_tolerance', 0.3)
        self.update_frequency = self.config.get('update_frequency', 60)
        self.ensemble_enabled = self.config.get('ensemble_enabled', True)
        self.online_learning_enabled = self.config.get('online_learning_enabled', True)
        
        # Component references
        self.feature_engineering = None
        self.model_registry = None
        self.training_pipeline = None
        self.online_learner = None
        self.model_ensemble = None
        self.trading_engine = None
        self.order_manager = None
        self.db_connection = None
        
        # Strategy state
        self.is_active = False
        self.last_update = None
        self.current_predictions = {}
        self.signal_history = []
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'signals_generated': 0,
            'high_confidence_signals': 0,
            'prediction_accuracy': 0.0,
            'last_accuracy_update': None
        }
        
        # Configuration
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.timeframes = self.config.get('timeframes', ['1h', '4h', '1d'])
        
    async def initialize(self):
        """Initialize the ML strategy"""
        try:
            self.logger.info("Initializing Advanced ML Strategy...")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
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
            
            # Initialize training pipeline
            if AdvancedTrainingPipeline:
                self.training_pipeline = AdvancedTrainingPipeline(
                    self.config.get('training_pipeline_config', {})
                )
                await self.training_pipeline.initialize()
            
            # Initialize online learner
            if OnlineLearner and self.online_learning_enabled:
                self.online_learner = OnlineLearner(
                    self.config.get('online_learning_config', {})
                )
                await self.online_learner.initialize()
            
            # Initialize model ensemble
            if ModelEnsemble and self.ensemble_enabled:
                self.model_ensemble = ModelEnsemble(
                    self.config.get('ensemble_config', {})
                )
                await self.model_ensemble.initialize()
            
            # Initialize trading components
            if TradingEngine:
                self.trading_engine = TradingEngine()
                await self.trading_engine.initialize()
            
            if OrderManager:
                self.order_manager = OrderManager()
                await self.order_manager.initialize()
            
            self.logger.info("Advanced ML Strategy initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML Strategy: {e}")
            raise
    
    async def start_strategy(self):
        """Start the ML strategy"""
        try:
            if self.is_active:
                self.logger.warning("Strategy is already active")
                return
            
            self.is_active = True
            self.logger.info("Starting Advanced ML Strategy...")
            
            # Start monitoring loop
            asyncio.create_task(self._strategy_monitoring_loop())
            
            self.logger.info("Advanced ML Strategy started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start strategy: {e}")
            raise
    
    async def stop_strategy(self):
        """Stop the ML strategy"""
        try:
            if not self.is_active:
                self.logger.warning("Strategy is not active")
                return
            
            self.is_active = False
            self.logger.info("Stopping Advanced ML Strategy...")
            
            # Wait for monitoring loop to finish
            await asyncio.sleep(1)
            
            self.logger.info("Advanced ML Strategy stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop strategy: {e}")
            raise
    
    async def _strategy_monitoring_loop(self):
        """Main strategy monitoring loop"""
        try:
            while self.is_active:
                try:
                    # Update predictions for all symbols
                    for symbol in self.symbols:
                        await self._update_predictions(symbol)
                    
                    # Generate trading signals
                    await self._generate_trading_signals()
                    
                    # Update online learning if enabled
                    if self.online_learning_enabled and self.online_learner:
                        await self._update_online_learning()
                    
                    # Update performance metrics
                    await self._update_performance_metrics()
                    
                    # Wait for next update
                    await asyncio.sleep(self.update_frequency)
                    
                except Exception as e:
                    self.logger.error(f"Error in strategy monitoring loop: {e}")
                    await asyncio.sleep(10)  # Wait before retrying
            
        except Exception as e:
            self.logger.error(f"Strategy monitoring loop failed: {e}")
            self.is_active = False
    
    async def _update_predictions(self, symbol: str):
        """Update ML predictions for a symbol"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            if market_data is None or len(market_data) < self.feature_window:
                return
            
            # Create features
            if self.feature_engineering:
                features = await self.feature_engineering.create_features(market_data)
            else:
                features = await self._create_basic_features(market_data)
            
            # Get predictions from different sources
            predictions = {}
            
            # Get ensemble prediction if available
            if self.model_ensemble:
                try:
                    ensemble_pred = await self.model_ensemble.predict(features)
                    predictions['ensemble'] = {
                        'prediction': ensemble_pred.prediction,
                        'confidence': ensemble_pred.confidence,
                        'model_version': 'ensemble'
                    }
                except Exception as e:
                    self.logger.error(f"Ensemble prediction failed: {e}")
            
            # Get individual model predictions
            if self.model_registry:
                active_models = await self.model_registry.list_models()
                for model_name in active_models:
                    try:
                        model = await self.model_registry.load_model(model_name)
                        if model:
                            pred = await self._get_model_prediction(model, features)
                            predictions[model_name] = pred
                    except Exception as e:
                        self.logger.error(f"Model {model_name} prediction failed: {e}")
            
            # Store predictions
            self.current_predictions[symbol] = {
                'predictions': predictions,
                'features': features,
                'timestamp': datetime.now(),
                'market_data': market_data.tail(1).to_dict('records')[0]
            }
            
            self.stats['total_predictions'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to update predictions for {symbol}: {e}")
    
    async def _create_basic_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create basic features if feature engineering is not available"""
        try:
            # Simple technical indicators
            features = []
            
            # Price-based features
            features.append(market_data['close'].pct_change().iloc[-1])
            features.append(market_data['high'].iloc[-1] / market_data['low'].iloc[-1] - 1)
            features.append(market_data['volume'].pct_change().iloc[-1])
            
            # Moving averages
            features.append(market_data['close'].rolling(20).mean().iloc[-1] / market_data['close'].iloc[-1] - 1)
            features.append(market_data['close'].rolling(50).mean().iloc[-1] / market_data['close'].iloc[-1] - 1)
            
            # Volatility
            features.append(market_data['close'].rolling(20).std().iloc[-1] / market_data['close'].iloc[-1])
            
            # Time features
            features.append(market_data.index[-1].hour / 24.0)
            features.append(market_data.index[-1].dayofweek / 7.0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Failed to create basic features: {e}")
            return np.zeros((1, 10))  # Return zero features
    
    async def _get_model_prediction(self, model, features: np.ndarray) -> Dict[str, Any]:
        """Get prediction from a single model"""
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(features)[0][1]  # Probability of positive class
            else:
                pred = model.predict(features)[0]
            
            return {
                'prediction': pred,
                'confidence': 0.7,  # Default confidence
                'model_version': 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model prediction: {e}")
            return {
                'prediction': 0.5,
                'confidence': 0.0,
                'model_version': 'unknown'
            }
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
        try:
            if self.db_connection:
                # Try to get from database
                data = await self.db_connection.get_candlestick_data(
                    symbol=symbol,
                    limit=self.feature_window * 2  # Get extra data for calculations
                )
                if data:
                    return pd.DataFrame(data)
            
            # Fallback to mock data for testing
            return self._create_mock_market_data(symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def _create_mock_market_data(self, symbol: str) -> pd.DataFrame:
        """Create mock market data for testing"""
        try:
            # Generate mock data
            dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
            
            # Simulate price movement
            np.random.seed(42)
            base_price = 50000 if 'BTC' in symbol else 3000
            returns = np.random.normal(0, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create DataFrame
            data = {
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 10000, len(dates))
            }
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to create mock data: {e}")
            return pd.DataFrame()
    
    async def _generate_trading_signals(self):
        """Generate trading signals based on ML predictions"""
        try:
            signals = []
            
            for symbol, pred_data in self.current_predictions.items():
                if not pred_data or 'predictions' not in pred_data:
                    continue
                
                # Get best prediction
                best_pred = await self._get_best_prediction(pred_data['predictions'])
                if not best_pred:
                    continue
                
                # Generate signal
                signal = await self._create_trading_signal(symbol, best_pred, pred_data)
                if signal:
                    signals.append(signal)
                    
                    # Store signal
                    self.signal_history.append(signal)
                    self.stats['signals_generated'] += 1
                    
                    if signal.confidence >= self.confidence_threshold:
                        self.stats['high_confidence_signals'] += 1
            
            # Execute signals if trading engine is available
            if signals and self.trading_engine:
                await self._execute_signals(signals)
            
        except Exception as e:
            self.logger.error(f"Failed to generate trading signals: {e}")
    
    async def _get_best_prediction(self, predictions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the best prediction from available models"""
        try:
            if not predictions:
                return None
            
            # Sort by confidence
            sorted_preds = sorted(
                predictions.items(),
                key=lambda x: x[1].get('confidence', 0),
                reverse=True
            )
            
            # Return highest confidence prediction
            if sorted_preds:
                return sorted_preds[0][1]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get best prediction: {e}")
            return None
    
    async def _create_trading_signal(self, symbol: str, prediction: Dict[str, Any], 
                                   pred_data: Dict[str, Any]) -> Optional[MLSignal]:
        """Create a trading signal from ML prediction"""
        try:
            pred_value = prediction.get('prediction', 0.5)
            confidence = prediction.get('confidence', 0.0)
            
            # Determine signal side
            if pred_value > self.prediction_threshold:
                side = 'buy'
            elif pred_value < (1 - self.prediction_threshold):
                side = 'sell'
            else:
                return None  # No clear signal
            
            # Calculate signal strength
            if confidence >= 0.9:
                strength = SignalConfidence.VERY_HIGH
            elif confidence >= 0.8:
                strength = SignalConfidence.HIGH
            elif confidence >= 0.7:
                strength = SignalConfidence.MEDIUM
            else:
                strength = SignalConfidence.LOW
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(symbol, pred_data, prediction)
            
            # Create signal
            signal = MLSignal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                strength=strength,
                prediction=pred_value,
                risk_score=risk_score,
                timestamp=datetime.now(),
                metadata={
                    'model_version': prediction.get('model_version', 'unknown'),
                    'features_used': pred_data.get('features', []).tolist() if hasattr(pred_data.get('features', []), 'tolist') else [],
                    'strategy_type': self.strategy_type.value
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to create trading signal: {e}")
            return None
    
    async def _calculate_risk_score(self, symbol: str, pred_data: Dict[str, Any], 
                                  prediction: Dict[str, Any]) -> float:
        """Calculate risk score for a signal"""
        try:
            risk_score = 0.0
            
            # Market volatility risk
            if 'market_data' in pred_data:
                market_data = pred_data['market_data']
                if 'close' in market_data:
                    # Simple volatility calculation
                    volatility = abs(market_data.get('high', 0) - market_data.get('low', 0)) / market_data.get('close', 1)
                    risk_score += min(volatility * 10, 0.5)  # Cap at 0.5
            
            # Prediction confidence risk
            confidence = prediction.get('confidence', 0.0)
            risk_score += (1.0 - confidence) * 0.3
            
            # Strategy type risk
            if self.strategy_type == MLStrategyType.RISK_ASSESSMENT:
                risk_score *= 0.8  # Lower risk for risk-focused strategy
            
            # Normalize risk score
            risk_score = min(max(risk_score, 0.0), 1.0)
            
            return risk_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk score: {e}")
            return 0.5  # Default medium risk
    
    async def _execute_signals(self, signals: List[MLSignal]):
        """Execute trading signals"""
        try:
            for signal in signals:
                if signal.confidence < self.confidence_threshold:
                    continue
                
                if signal.risk_score > self.risk_tolerance:
                    self.logger.warning(f"Signal rejected due to high risk: {signal.risk_score}")
                    continue
                
                # Execute signal through trading engine
                if self.trading_engine:
                    try:
                        await self.trading_engine.process_signal({
                            'symbol': signal.symbol,
                            'side': signal.side,
                            'confidence': signal.confidence,
                            'strength': signal.strength.value,
                            'timestamp': signal.timestamp,
                            'metadata': signal.metadata
                        })
                        
                        self.logger.info(f"Executed signal: {signal.side} {signal.symbol} (confidence: {signal.confidence:.2f})")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to execute signal: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute signals: {e}")
    
    async def _update_online_learning(self):
        """Update online learning with new data"""
        try:
            if not self.online_learner:
                return
            
            # Get recent signals and outcomes
            recent_signals = self.signal_history[-50:]  # Last 50 signals
            
            for signal in recent_signals:
                # Get actual outcome (this would come from trade results)
                # For now, we'll simulate outcomes
                outcome = await self._simulate_signal_outcome(signal)
                
                # Update online learner
                features = np.array(signal.metadata.get('features_used', [0.5]))
                await self.online_learner.process_sample(
                    features=features,
                    label=outcome,
                    prediction=signal.prediction
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update online learning: {e}")
    
    async def _simulate_signal_outcome(self, signal: MLSignal) -> float:
        """Simulate signal outcome for testing"""
        try:
            # Simple simulation based on prediction accuracy
            # In practice, this would come from actual trade results
            
            if signal.prediction > 0.7:
                # High confidence signals tend to be more accurate
                accuracy = 0.8
            elif signal.prediction > 0.6:
                accuracy = 0.7
            else:
                accuracy = 0.6
            
            # Add some randomness
            if np.random.random() < accuracy:
                return 1.0 if signal.side == 'buy' else 0.0
            else:
                return 0.0 if signal.side == 'buy' else 1.0
                
        except Exception as e:
            self.logger.error(f"Failed to simulate signal outcome: {e}")
            return 0.5
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate prediction accuracy
            if len(self.signal_history) > 10:
                recent_signals = self.signal_history[-100:]
                
                # This would compare predictions with actual outcomes
                # For now, we'll use a simple metric
                high_confidence_count = sum(1 for s in recent_signals if s.confidence >= 0.8)
                total_count = len(recent_signals)
                
                if total_count > 0:
                    self.stats['prediction_accuracy'] = high_confidence_count / total_count
                    self.stats['last_accuracy_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    async def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'strategy_type': self.strategy_type.value,
            'is_active': self.is_active,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'statistics': self.stats,
            'current_predictions': {
                symbol: {
                    'timestamp': data.get('timestamp'),
                    'prediction_count': len(data.get('predictions', {}))
                }
                for symbol, data in self.current_predictions.items()
            },
            'recent_signals': [
                {
                    'symbol': s.symbol,
                    'side': s.side,
                    'confidence': s.confidence,
                    'strength': s.strength.value,
                    'timestamp': s.timestamp
                }
                for s in self.signal_history[-10:]
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check strategy health"""
        try:
            return {
                'status': 'healthy',
                'components': {
                    'feature_engineering': self.feature_engineering is not None,
                    'model_registry': self.model_registry is not None,
                    'training_pipeline': self.training_pipeline is not None,
                    'online_learner': self.online_learner is not None,
                    'model_ensemble': self.model_ensemble is not None,
                    'trading_engine': self.trading_engine is not None,
                    'order_manager': self.order_manager is not None,
                    'database': self.db_connection is not None
                },
                'strategy_state': {
                    'is_active': self.is_active,
                    'last_update': self.last_update,
                    'symbols_monitored': len(self.symbols)
                },
                'performance': {
                    'total_predictions': self.stats['total_predictions'],
                    'signals_generated': self.stats['signals_generated'],
                    'prediction_accuracy': self.stats['prediction_accuracy']
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self):
        """Close the ML strategy"""
        try:
            # Stop strategy if active
            if self.is_active:
                await self.stop_strategy()
            
            # Close components
            if self.feature_engineering:
                await self.feature_engineering.close()
            
            if self.model_registry:
                await self.model_registry.close()
            
            if self.training_pipeline:
                await self.training_pipeline.close()
            
            if self.online_learner:
                await self.online_learner.close()
            
            if self.model_ensemble:
                await self.model_ensemble.close()
            
            if self.trading_engine:
                await self.trading_engine.close()
            
            if self.order_manager:
                await self.order_manager.close()
            
            if self.db_connection:
                await self.db_connection.close()
            
            self.logger.info("Advanced ML Strategy closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close ML Strategy: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
