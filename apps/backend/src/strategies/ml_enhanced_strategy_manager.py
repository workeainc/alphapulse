"""
ML-Enhanced Strategy Manager for AlphaPlus
Integrates machine learning models with traditional trading strategies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

# Import our enhanced components
try:
    from ..src.ai.ml_models.model_registry import ModelRegistry, ModelType, ModelStatus
    from ..src.ai.ml_models.advanced_training_pipeline import AdvancedTrainingPipeline, TrainingConfig
    from ..src.ai.advanced_feature_engineering import AdvancedFeatureEngineering
    from ..src.core.trading_engine import TradingEngine
    from ..execution.order_manager import OrderManager
    from ..src.database.connection import TimescaleDBConnection
except ImportError:
    # Fallback for testing
    ModelRegistry = None
    ModelType = None
    ModelStatus = None
    AdvancedTrainingPipeline = None
    TrainingConfig = None
    AdvancedFeatureEngineering = None
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
    MARKET_REGIME_DETECTION = "market_regime_detection"
    HYBRID = "hybrid"

@dataclass
class MLStrategyConfig:
    """ML strategy configuration"""
    strategy_name: str
    strategy_type: MLStrategyType
    model_name: str
    confidence_threshold: float
    update_frequency: int  # minutes
    feature_columns: List[str]
    target_column: str
    prediction_horizon: int  # periods ahead
    risk_limits: Dict[str, float]
    auto_retrain: bool
    retrain_threshold: float

@dataclass
class MLPrediction:
    """ML model prediction result"""
    timestamp: datetime
    symbol: str
    prediction: float
    confidence: float
    model_id: str
    features_used: List[str]
    prediction_horizon: int
    metadata: Dict[str, Any]

class MLEnhancedStrategyManager:
    """ML-enhanced strategy manager with automated model integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Strategy configuration
        self.strategies: Dict[str, MLStrategyConfig] = {}
        self.active_predictions: Dict[str, MLPrediction] = {}
        
        # Component references
        self.model_registry = None
        self.training_pipeline = None
        self.feature_engineering = None
        self.trading_engine = None
        self.order_manager = None
        self.db_connection = None
        
        # ML model cache
        self.model_cache = {}
        self.prediction_cache = {}
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'high_confidence_predictions': 0,
            'predictions_used_for_trading': 0,
            'model_retraining_count': 0
        }
        
        # Strategy monitoring
        self.strategy_monitoring_task = None
        self.is_monitoring = False
        
    async def initialize(self):
        """Initialize the ML-enhanced strategy manager"""
        try:
            self.logger.info("Initializing ML-Enhanced Strategy Manager...")
            
            # Initialize ML components
            if ModelRegistry:
                self.model_registry = ModelRegistry(
                    self.config.get('model_registry_config', {})
                )
                await self.model_registry.initialize()
            
            if AdvancedTrainingPipeline:
                self.training_pipeline = AdvancedTrainingPipeline(
                    self.config.get('training_pipeline_config', {})
                )
                await self.training_pipeline.initialize()
            
            if AdvancedFeatureEngineering:
                self.feature_engineering = AdvancedFeatureEngineering(
                    self.config.get('feature_engineering_config', {})
                )
                await self.feature_engineering.initialize()
            
            # Initialize trading components
            if TradingEngine:
                self.trading_engine = TradingEngine()
                await self.trading_engine.initialize()
            
            if OrderManager:
                self.order_manager = OrderManager()
                await self.order_manager.initialize()
            
            # Initialize database connection
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            self.logger.info("ML-Enhanced Strategy Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML Strategy Manager: {e}")
            raise
    
    async def add_ml_strategy(self, strategy_config: MLStrategyConfig) -> bool:
        """Add a new ML strategy"""
        try:
            # Validate strategy configuration
            if not await self._validate_strategy_config(strategy_config):
                return False
            
            # Check if model exists and is deployed
            if not await self._verify_model_availability(strategy_config.model_name):
                self.logger.warning(f"Model {strategy_config.model_name} not available for strategy {strategy_config.strategy_name}")
                return False
            
            # Add strategy
            self.strategies[strategy_config.strategy_name] = strategy_config
            
            self.logger.info(f"Added ML strategy: {strategy_config.strategy_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding ML strategy: {e}")
            return False
    
    async def _validate_strategy_config(self, strategy_config: MLStrategyConfig) -> bool:
        """Validate strategy configuration"""
        try:
            # Check required fields
            if not strategy_config.strategy_name or not strategy_config.model_name:
                return False
            
            # Validate confidence threshold
            if not (0.0 <= strategy_config.confidence_threshold <= 1.0):
                return False
            
            # Validate update frequency
            if strategy_config.update_frequency <= 0:
                return False
            
            # Validate prediction horizon
            if strategy_config.prediction_horizon <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating strategy config: {e}")
            return False
    
    async def _verify_model_availability(self, model_name: str) -> bool:
        """Verify if model is available and deployed"""
        try:
            if not self.model_registry:
                return False
            
            # Get active model
            active_model_id = await self.model_registry.get_active_model(model_name)
            if not active_model_id:
                return False
            
            # Check model status
            models = await self.model_registry.list_models(name=model_name, status=ModelStatus.DEPLOYED)
            return len(models) > 0
            
        except Exception as e:
            self.logger.error(f"Error verifying model availability: {e}")
            return False
    
    async def start_strategy_monitoring(self):
        """Start monitoring and updating ML strategies"""
        try:
            if self.is_monitoring:
                self.logger.warning("Strategy monitoring already active")
                return
            
            self.is_monitoring = True
            self.strategy_monitoring_task = asyncio.create_task(self._strategy_monitoring_loop())
            
            self.logger.info("ML strategy monitoring started")
            
        except Exception as e:
            self.logger.error(f"Error starting strategy monitoring: {e}")
            raise
    
    async def stop_strategy_monitoring(self):
        """Stop strategy monitoring"""
        try:
            self.is_monitoring = False
            
            if self.strategy_monitoring_task:
                self.strategy_monitoring_task.cancel()
                try:
                    await self.strategy_monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("ML strategy monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy monitoring: {e}")
    
    async def _strategy_monitoring_loop(self):
        """Main strategy monitoring loop"""
        try:
            while self.is_monitoring:
                # Update predictions for all strategies
                for strategy_name, strategy_config in self.strategies.items():
                    try:
                        await self._update_strategy_predictions(strategy_name, strategy_config)
                    except Exception as e:
                        self.logger.error(f"Error updating predictions for strategy {strategy_name}: {e}")
                
                # Check for model retraining needs
                await self._check_retraining_needs()
                
                # Wait for next update cycle
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            self.logger.info("Strategy monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in strategy monitoring loop: {e}")
            self.is_monitoring = False
    
    async def _update_strategy_predictions(self, strategy_name: str, strategy_config: MLStrategyConfig):
        """Update predictions for a specific strategy"""
        try:
            # Get latest market data
            market_data = await self._get_latest_market_data(strategy_config)
            if market_data.empty:
                return
            
            # Create features
            features = await self._create_strategy_features(market_data, strategy_config)
            if features.empty:
                return
            
            # Get model prediction
            prediction = await self._get_model_prediction(
                strategy_config.model_name, features, strategy_config
            )
            
            if prediction:
                # Store prediction
                self.active_predictions[strategy_name] = prediction
                
                # Update statistics
                self.stats['total_predictions'] += 1
                if prediction.confidence >= strategy_config.confidence_threshold:
                    self.stats['high_confidence_predictions'] += 1
                
                # Generate trading signals if confidence is high enough
                if prediction.confidence >= strategy_config.confidence_threshold:
                    await self._generate_trading_signal(strategy_name, prediction, strategy_config)
                
                self.logger.info(f"Updated predictions for strategy {strategy_name}: "
                               f"prediction={prediction.prediction:.4f}, "
                               f"confidence={prediction.confidence:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy predictions: {e}")
    
    async def _get_latest_market_data(self, strategy_config: MLStrategyConfig) -> pd.DataFrame:
        """Get latest market data for strategy"""
        try:
            if self.db_connection:
                # Get data from database
                data = await self.db_connection.get_candlestick_data(
                    symbol='BTCUSDT',  # Default symbol
                    timeframe='1m',
                    limit=1000
                )
                
                if data:
                    return pd.DataFrame(data)
            
            # Fallback: return empty DataFrame
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    async def _create_strategy_features(self, market_data: pd.DataFrame, 
                                      strategy_config: MLStrategyConfig) -> pd.DataFrame:
        """Create features for strategy"""
        try:
            if self.feature_engineering:
                # Create features using feature engineering pipeline
                features = await self.feature_engineering.create_features(
                    market_data, strategy_config.feature_columns
                )
                
                # Select only required features
                available_features = [col for col in features.columns 
                                   if col in strategy_config.feature_columns]
                
                if available_features:
                    return features[available_features]
            
            # Fallback: use original data columns
            available_columns = [col for col in market_data.columns 
                               if col in strategy_config.feature_columns]
            
            if available_columns:
                return market_data[available_columns]
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error creating strategy features: {e}")
            return pd.DataFrame()
    
    async def _get_model_prediction(self, model_name: str, features: pd.DataFrame, 
                                   strategy_config: MLStrategyConfig) -> Optional[MLPrediction]:
        """Get prediction from ML model"""
        try:
            # Get model from cache or load from registry
            model = await self._get_cached_model(model_name)
            if not model:
                return None
            
            # Prepare features for prediction
            X = features.iloc[-1:].values  # Latest data point
            
            # Make prediction
            prediction_value = model.predict(X)[0]
            
            # Calculate confidence (simplified - in practice this would be more sophisticated)
            confidence = 0.8  # Placeholder confidence
            
            # Create prediction object
            prediction = MLPrediction(
                timestamp=datetime.now(timezone.utc),
                symbol='BTCUSDT',  # Default symbol
                prediction=prediction_value,
                confidence=confidence,
                model_id=model_name,
                features_used=list(features.columns),
                prediction_horizon=strategy_config.prediction_horizon,
                metadata={'strategy': strategy_config.strategy_name}
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error getting model prediction: {e}")
            return None
    
    async def _get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get model from cache or load from registry"""
        try:
            # Check cache first
            if model_name in self.model_cache:
                return self.model_cache[model_name]
            
            # Load from registry
            if self.model_registry:
                active_model_id = await self.model_registry.get_active_model(model_name)
                if active_model_id:
                    model = await self.model_registry.load_model(active_model_id)
                    if model:
                        # Cache model
                        self.model_cache[model_name] = model
                        return model
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached model: {e}")
            return None
    
    async def _generate_trading_signal(self, strategy_name: str, prediction: MLPrediction, 
                                     strategy_config: MLStrategyConfig):
        """Generate trading signal based on ML prediction"""
        try:
            # Determine signal direction based on prediction type
            if strategy_config.strategy_type == MLStrategyType.PRICE_PREDICTION:
                signal = await self._generate_price_prediction_signal(prediction, strategy_config)
            elif strategy_config.strategy_type == MLStrategyType.SIGNAL_CLASSIFICATION:
                signal = await self._generate_signal_classification_signal(prediction, strategy_config)
            else:
                signal = await self._generate_generic_signal(prediction, strategy_config)
            
            if signal:
                # Execute signal if trading engine is available
                if self.trading_engine:
                    await self.trading_engine.process_signal(signal)
                
                # Update statistics
                self.stats['predictions_used_for_trading'] += 1
                
                self.logger.info(f"Generated trading signal for strategy {strategy_name}: {signal}")
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
    
    async def _generate_price_prediction_signal(self, prediction: MLPrediction, 
                                              strategy_config: MLStrategyConfig) -> Optional[Dict[str, Any]]:
        """Generate signal for price prediction strategy"""
        try:
            # Simple signal generation based on prediction direction
            current_price = 50000  # Placeholder - get from market data
            
            if prediction.prediction > current_price * 1.01:  # 1% above current
                signal = {
                    'type': 'buy',
                    'symbol': prediction.symbol,
                    'confidence': prediction.confidence,
                    'strategy': strategy_config.strategy_name,
                    'prediction': prediction.prediction,
                    'timestamp': prediction.timestamp.isoformat()
                }
                return signal
            elif prediction.prediction < current_price * 0.99:  # 1% below current
                signal = {
                    'type': 'sell',
                    'symbol': prediction.symbol,
                    'confidence': prediction.confidence,
                    'strategy': strategy_config.strategy_name,
                    'prediction': prediction.prediction,
                    'timestamp': prediction.timestamp.isoformat()
                }
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating price prediction signal: {e}")
            return None
    
    async def _generate_signal_classification_signal(self, prediction: MLPrediction, 
                                                   strategy_config: MLStrategyConfig) -> Optional[Dict[str, Any]]:
        """Generate signal for signal classification strategy"""
        try:
            # For classification, prediction value represents class
            if prediction.prediction > 0.5:  # Buy signal
                signal = {
                    'type': 'buy',
                    'symbol': prediction.symbol,
                    'confidence': prediction.confidence,
                    'strategy': strategy_config.strategy_name,
                    'prediction': prediction.prediction,
                    'timestamp': prediction.timestamp.isoformat()
                }
                return signal
            else:  # Sell signal
                signal = {
                    'type': 'sell',
                    'symbol': prediction.symbol,
                    'confidence': prediction.confidence,
                    'strategy': strategy_config.strategy_name,
                    'prediction': prediction.prediction,
                    'timestamp': prediction.timestamp.isoformat()
                }
                return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal classification signal: {e}")
            return None
    
    async def _generate_generic_signal(self, prediction: MLPrediction, 
                                     strategy_config: MLStrategyConfig) -> Optional[Dict[str, Any]]:
        """Generate generic signal for other strategy types"""
        try:
            # Generic signal based on prediction value
            signal = {
                'type': 'buy' if prediction.prediction > 0 else 'sell',
                'symbol': prediction.symbol,
                'confidence': prediction.confidence,
                'strategy': strategy_config.strategy_name,
                'prediction': prediction.prediction,
                'timestamp': prediction.timestamp.isoformat()
            }
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating generic signal: {e}")
            return None
    
    async def _check_retraining_needs(self):
        """Check if models need retraining"""
        try:
            if not self.training_pipeline or not self.model_registry:
                return
            
            for strategy_name, strategy_config in self.strategies.items():
                if not strategy_config.auto_retrain:
                    continue
                
                # Check prediction accuracy (simplified)
                if strategy_name in self.active_predictions:
                    prediction = self.active_predictions[strategy_name]
                    
                    # Simple accuracy check (in practice, this would be more sophisticated)
                    if prediction.confidence < strategy_config.retrain_threshold:
                        await self._retrain_model(strategy_config)
                        
        except Exception as e:
            self.logger.error(f"Error checking retraining needs: {e}")
    
    async def _retrain_model(self, strategy_config: MLStrategyConfig):
        """Retrain a model"""
        try:
            if not self.training_pipeline:
                return
            
            self.logger.info(f"Starting model retraining for strategy: {strategy_config.strategy_name}")
            
            # Get training data
            training_data = await self._get_training_data(strategy_config)
            if training_data.empty:
                self.logger.warning(f"No training data available for retraining {strategy_config.strategy_name}")
                return
            
            # Create training configuration
            training_config = TrainingConfig(
                model_name=strategy_config.model_name,
                model_type=ModelType.CATBOOST,  # Default type
                target_column=strategy_config.target_column,
                feature_columns=strategy_config.feature_columns,
                test_size=0.2,
                validation_size=0.1,
                random_state=42,
                hyperparameter_grid={},  # Use defaults
                cv_folds=5,
                scoring_metric='neg_mean_squared_error',
                early_stopping_rounds=50,
                max_training_time=3600
            )
            
            # Start training
            model_id = await self.training_pipeline.train_model(training_config, training_data)
            
            # Update statistics
            self.stats['model_retraining_count'] += 1
            
            self.logger.info(f"Model retraining started: {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
    
    async def _get_training_data(self, strategy_config: MLStrategyConfig) -> pd.DataFrame:
        """Get training data for model retraining"""
        try:
            if self.db_connection:
                # Get historical data for training
                data = await self.db_connection.get_candlestick_data(
                    symbol='BTCUSDT',
                    timeframe='1m',
                    limit=10000  # More data for training
                )
                
                if data:
                    return pd.DataFrame(data)
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    async def get_strategy_predictions(self, strategy_name: str = None) -> Dict[str, MLPrediction]:
        """Get current strategy predictions"""
        try:
            if strategy_name:
                return {strategy_name: self.active_predictions.get(strategy_name)}
            else:
                return self.active_predictions.copy()
                
        except Exception as e:
            self.logger.error(f"Error getting strategy predictions: {e}")
            return {}
    
    async def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get strategy manager statistics"""
        try:
            stats = self.stats.copy()
            
            # Add strategy info
            stats['total_strategies'] = len(self.strategies)
            stats['active_strategies'] = len([s for s in self.strategies.values() if s.auto_retrain])
            stats['total_predictions'] = len(self.active_predictions)
            
            # Add strategy types distribution
            strategy_types = {}
            for strategy_config in self.strategies.values():
                strategy_type = strategy_config.strategy_type.value
                strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
            
            stats['strategy_types'] = strategy_types
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting strategy statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for ML strategy manager"""
        try:
            health_status = {
                'status': 'healthy',
                'monitoring_active': self.is_monitoring,
                'total_strategies': len(self.strategies),
                'active_predictions': len(self.active_predictions)
            }
            
            # Check component health
            if self.model_registry:
                try:
                    mr_health = await self.model_registry.health_check()
                    health_status['model_registry_health'] = mr_health
                    
                    if mr_health.get('status') != 'healthy':
                        health_status['status'] = 'degraded'
                        health_status['warnings'] = ['Model registry issues']
                except Exception as e:
                    health_status['model_registry_health'] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'
            
            if self.training_pipeline:
                try:
                    tp_health = await self.training_pipeline.health_check()
                    health_status['training_pipeline_health'] = tp_health
                    
                    if tp_health.get('status') != 'healthy':
                        health_status['status'] = 'degraded'
                        if 'warnings' not in health_status:
                            health_status['warnings'] = []
                        health_status['warnings'].append('Training pipeline issues')
                except Exception as e:
                    health_status['training_pipeline_health'] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Close ML strategy manager"""
        try:
            await self.stop_strategy_monitoring()
            
            # Clear caches
            self.model_cache.clear()
            self.prediction_cache.clear()
            
            if self.model_registry:
                await self.model_registry.close()
            
            if self.training_pipeline:
                await self.training_pipeline.close()
            
            if self.feature_engineering:
                await self.feature_engineering.close()
            
            if self.trading_engine:
                # Note: TradingEngine doesn't have a close method yet
                pass
            
            if self.order_manager:
                # Note: OrderManager doesn't have a close method yet
                pass
            
            if self.db_connection:
                await self.db_connection.close()
            
            self.logger.info("ML strategy manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing ML strategy manager: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
