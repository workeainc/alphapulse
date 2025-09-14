"""
Predictive Signal Service for AlphaPulse
Week 7.4: Predictive Signal Optimization

Features:
- ML-based signal prediction using XGBoost
- Intelligent data pruning for efficiency
- Feature engineering from multiple data sources
- Real-time prediction with <15ms latency

Author: AlphaPulse Team
Date: 2025
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SignalPrediction:
    """Predicted signal with confidence and PnL"""
    symbol: str
    signal_type: str
    confidence: float
    predicted_pnl: float
    features: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PruningResult:
    """Result of data pruning operation"""
    original_count: int
    pruned_count: int
    retention_rate: float
    pruned_data: pd.DataFrame
    retained_data: pd.DataFrame

class PredictiveSignal:
    """ML-based predictive signal service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # ML configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.prune_threshold = self.config.get('prune_threshold', 0.1)
        self.min_data_points = self.config.get('min_data_points', 50)
        self.prediction_window = self.config.get('prediction_window', 24)  # hours
        
        # Feature engineering parameters
        self.volatility_window = self.config.get('volatility_window', 20)
        self.correlation_window = self.config.get('correlation_window', 50)
        self.momentum_window = self.config.get('momentum_window', 10)
        
        # Model storage
        self.models = defaultdict(dict)  # symbol -> {signal_type -> model}
        self.feature_scalers = defaultdict(dict)  # symbol -> {signal_type -> scaler}
        self.prediction_cache = defaultdict(deque)  # symbol -> predictions
        
        # Position sizing models (Week 9 enhancement)
        self.position_sizing_models = defaultdict(dict)  # symbol -> {signal_type -> sizing_model}
        
        # Performance tracking
        self.stats = {
            'predictions_made': 0,
            'data_pruned': 0,
            'signals_generated': 0,
            'accuracy_score': 0.0,
            'last_update': None
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different signal types"""
        try:
            # Try to import XGBoost, fallback to simpler models if not available
            try:
                import xgboost as xgb
                self.xgb_available = True
                self.logger.info("XGBoost available for advanced predictions")
            except ImportError:
                self.xgb_available = False
                self.logger.warning("XGBoost not available, using simplified models")
            
            # Initialize models for common signal types
            signal_types = ['funding_rate', 'correlation', 'volatility', 'arbitrage']
            for signal_type in signal_types:
                if self.xgb_available:
                    self.models['default'][signal_type] = self._create_xgb_model()
                    # Initialize position sizing models (Week 9)
                    self.position_sizing_models['default'][signal_type] = self._create_position_sizing_model()
                else:
                    self.models['default'][signal_type] = self._create_simple_model()
            
            self.logger.info("Predictive models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
    
    def _create_xgb_model(self):
        """Create XGBoost model for signal prediction"""
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating XGBoost model: {e}")
            return None
    
    def _create_simple_model(self):
        """Create simple statistical model as fallback"""
        try:
            # Simple linear regression model
            class SimpleModel:
                def __init__(self):
                    self.coefficients = None
                    self.intercept = 0.0
                
                def fit(self, X, y):
                    if len(X) > 0 and len(y) > 0:
                        # Simple linear regression
                        X_array = np.array(X)
                        y_array = np.array(y)
                        
                        if X_array.shape[1] > 0:
                            # Calculate coefficients using least squares
                            X_with_intercept = np.column_stack([np.ones(X_array.shape[0]), X_array])
                            coefficients = np.linalg.lstsq(X_with_intercept, y_array, rcond=None)[0]
                            self.intercept = coefficients[0]
                            self.coefficients = coefficients[1:]
                
                def predict(self, X):
                    if self.coefficients is not None and len(X) > 0:
                        X_array = np.array(X)
                        if X_array.shape[1] == len(self.coefficients):
                            return self.intercept + np.dot(X_array, self.coefficients)
                    return np.zeros(len(X))
                
                def predict_proba(self, X):
                    # Convert regression output to probability-like scores
                    predictions = self.predict(X)
                    # Normalize to [0, 1] range
                    if len(predictions) > 0:
                        min_val = np.min(predictions)
                        max_val = np.max(predictions)
                        if max_val > min_val:
                            normalized = (predictions - min_val) / (max_val - min_val)
                        else:
                            normalized = np.ones_like(predictions) * 0.5
                        return np.column_stack([1 - normalized, normalized])
                    return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
            
            return SimpleModel()
            
        except Exception as e:
            self.logger.error(f"Error creating simple model: {e}")
            return None
    
    def _create_position_sizing_model(self):
        """Create ML model for position sizing (Week 9)"""
        try:
            if self.xgb_available:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    objective='reg:squarederror'
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
        except Exception as e:
            self.logger.error(f"Error creating position sizing model: {e}")
            return None
    
    async def train_model(self, symbol: str, signal_type: str, 
                         historical_data: pd.DataFrame, target_values: pd.Series) -> bool:
        """Train ML model on historical data"""
        try:
            if historical_data.empty or len(historical_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for training: {len(historical_data)} points")
                return False
            
            # Feature engineering
            features = self._engineer_features(historical_data)
            if features.empty:
                self.logger.warning("No features generated from historical data")
                return False
            
            # Get or create model
            if symbol not in self.models:
                self.models[symbol] = {}
            
            if signal_type not in self.models[symbol]:
                if self.xgb_available:
                    self.models[symbol][signal_type] = self._create_xgb_model()
                else:
                    self.models[symbol][signal_type] = self._create_simple_model()
            
            model = self.models[symbol][signal_type]
            if model is None:
                return False
            
            # Train model
            model.fit(features, target_values)
            
            # Train position sizing model if we have sizing targets (Week 9)
            if 'position_size_target' in historical_data.columns:
                sizing_features = self._prepare_sizing_features(features)
                sizing_targets = historical_data['position_size_target']
                
                if symbol not in self.position_sizing_models:
                    self.position_sizing_models[symbol] = {}
                
                if signal_type not in self.position_sizing_models[symbol]:
                    self.position_sizing_models[symbol][signal_type] = self._create_position_sizing_model()
                
                sizing_model = self.position_sizing_models[symbol][signal_type]
                if sizing_model is not None:
                    sizing_model.fit(sizing_features, sizing_targets)
                    self.logger.info(f"Position sizing model trained for {symbol} - {signal_type}")
            
            self.logger.info(f"Model trained for {symbol} - {signal_type} with {len(features)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        try:
            if data.empty:
                return pd.DataFrame()
            
            features = pd.DataFrame()
            
            # Price-based features
            if 'price' in data.columns:
                features['price'] = data['price']
                features['price_change'] = data['price'].pct_change()
                features['price_volatility'] = data['price'].pct_change().rolling(self.volatility_window).std()
                features['price_momentum'] = data['price'].pct_change().rolling(self.momentum_window).mean()
            
            # Volume-based features
            if 'volume' in data.columns:
                features['volume'] = data['volume']
                features['volume_change'] = data['volume'].pct_change()
                features['volume_ma'] = data['volume'].rolling(20).mean()
            
            # Funding rate features
            if 'funding_rate' in data.columns:
                features['funding_rate'] = data['funding_rate']
                features['funding_rate_change'] = data['funding_rate'].pct_change()
                features['funding_rate_volatility'] = data['funding_rate'].rolling(self.volatility_window).std()
            
            # Correlation features
            if 'correlation' in data.columns:
                features['correlation'] = data['correlation']
                features['correlation_change'] = data['correlation'].pct_change()
                features['correlation_ma'] = data['correlation'].rolling(self.correlation_window).mean()
            
            # Technical indicators
            if 'price' in data.columns:
                # RSI-like indicator
                delta = data['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                features['bb_upper'] = data['price'].rolling(20).mean() + (data['price'].rolling(20).std() * 2)
                features['bb_lower'] = data['price'].rolling(20).mean() - (data['price'].rolling(20).std() * 2)
                features['bb_position'] = (data['price'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return pd.DataFrame()
    
    async def predict_signal(self, symbol: str, signal_type: str, 
                           current_data: pd.DataFrame) -> Optional[SignalPrediction]:
        """Predict signal confidence and PnL"""
        try:
            if current_data.empty:
                return None
            
            # Get model for this symbol and signal type
            model = self.models.get(symbol, {}).get(signal_type)
            if model is None:
                # Fallback to default model
                model = self.models.get('default', {}).get(signal_type)
            
            if model is None:
                self.logger.warning(f"No model available for {symbol} - {signal_type}")
                return None
            
            # Engineer features
            features = self._engineer_features(current_data)
            if features.empty:
                return None
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                # Classification model
                confidence_proba = model.predict_proba(features.iloc[-1:])
                confidence = confidence_proba[0][1]  # Probability of positive class
            else:
                # Regression model
                confidence = 0.7  # Default confidence for regression models
            
            # Predict PnL (simplified)
            if hasattr(model, 'predict'):
                predicted_pnl = model.predict(features.iloc[-1:])[0]
            else:
                predicted_pnl = 0.0
            
            # Create prediction object
            prediction = SignalPrediction(
                symbol=symbol,
                signal_type=signal_type,
                confidence=float(confidence),
                predicted_pnl=float(predicted_pnl),
                features=features.iloc[-1].to_dict(),
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'model_type': 'xgboost' if self.xgb_available else 'simple',
                    'feature_count': len(features.columns),
                    'data_points': len(current_data)
                }
            )
            
            # Cache prediction
            if symbol not in self.prediction_cache:
                self.prediction_cache[symbol] = deque(maxlen=100)
            self.prediction_cache[symbol].append(prediction)
            
            # Update statistics
            self.stats['predictions_made'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting signal: {e}")
            return None
    
    def prune_data(self, data: pd.DataFrame, data_type: str = 'market') -> PruningResult:
        """Prune low-impact data for efficiency"""
        try:
            if data.empty:
                return PruningResult(0, 0, 0.0, pd.DataFrame(), pd.DataFrame())
            
            original_count = len(data)
            
            # Apply pruning based on data type
            if data_type == 'order_book':
                # Prune order book updates with minimal price impact
                if 'price' in data.columns:
                    data['price_impact'] = data['price'].pct_change().abs()
                    pruned_data = data[data['price_impact'] <= self.prune_threshold]
                    retained_data = data[data['price_impact'] > self.prune_threshold]
                else:
                    # No price column, keep all data
                    pruned_data = pd.DataFrame()
                    retained_data = data
            
            elif data_type == 'funding_rate':
                # Prune funding rate updates with minimal changes
                if 'funding_rate' in data.columns:
                    data['rate_change'] = data['funding_rate'].pct_change().abs()
                    pruned_data = data[data['rate_change'] <= self.prune_threshold]
                    retained_data = data[data['rate_change'] > self.prune_threshold]
                else:
                    pruned_data = pd.DataFrame()
                    retained_data = data
            
            elif data_type == 'market':
                # Prune market data with minimal volatility
                if 'price' in data.columns:
                    data['volatility'] = data['price'].pct_change().rolling(5).std()
                    pruned_data = data[data['volatility'] <= self.prune_threshold]
                    retained_data = data[data['volatility'] > self.prune_threshold]
                else:
                    pruned_data = pd.DataFrame()
                    retained_data = data
            
            else:
                # Default: no pruning
                pruned_data = pd.DataFrame()
                retained_data = data
            
            # Calculate retention rate
            retention_rate = len(retained_data) / original_count if original_count > 0 else 0.0
            
            # Update statistics
            self.stats['data_pruned'] += len(pruned_data)
            
            return PruningResult(
                original_count=original_count,
                pruned_count=len(pruned_data),
                retention_rate=retention_rate,
                pruned_data=pruned_data,
                retained_data=retained_data
            )
            
        except Exception as e:
            self.logger.error(f"Error pruning data: {e}")
            return PruningResult(0, 0, 0.0, pd.DataFrame(), data)
    
    async def should_generate_signal(self, symbol: str, signal_type: str, 
                                   current_data: pd.DataFrame) -> Tuple[bool, float]:
        """Determine if a signal should be generated based on prediction"""
        try:
            # Get prediction
            prediction = await self.predict_signal(symbol, signal_type, current_data)
            if prediction is None:
                return False, 0.0
            
            # Check confidence threshold
            should_generate = prediction.confidence >= self.confidence_threshold
            
            if should_generate:
                self.stats['signals_generated'] += 1
            
            return should_generate, prediction.confidence
            
        except Exception as e:
            self.logger.error(f"Error checking signal generation: {e}")
            return False, 0.0
    
    async def predict_position_size(self, symbol: str, signal_type: str, 
                                  features: pd.DataFrame, account_balance: float,
                                  max_risk: float = 0.02) -> Dict[str, Any]:
        """Predict optimal position size using ML (Week 9 enhancement)"""
        try:
            if features.empty:
                return self._get_default_position_size(account_balance, max_risk)
            
            # Get position sizing model
            model = self.position_sizing_models.get(symbol, {}).get(signal_type)
            if model is None:
                model = self.position_sizing_models.get('default', {}).get(signal_type)
            
            if model is None:
                return self._get_default_position_size(account_balance, max_risk)
            
            # Prepare features for position sizing
            sizing_features = self._prepare_sizing_features(features)
            
            # Predict position size ratio (0.0 to 1.0)
            predicted_ratio = model.predict(sizing_features)
            
            # Ensure prediction is within bounds
            predicted_ratio = np.clip(predicted_ratio, 0.0, max_risk)
            
            # Calculate position size
            position_value = account_balance * predicted_ratio
            position_size = position_value / features['price'].iloc[-1] if 'price' in features else 0.0
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'position_ratio': predicted_ratio,
                'confidence': self._calculate_sizing_confidence(features),
                'risk_level': 'low' if predicted_ratio < max_risk * 0.5 else 'medium' if predicted_ratio < max_risk * 0.8 else 'high'
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting position size: {e}")
            return self._get_default_position_size(account_balance, max_risk)
    
    def _prepare_sizing_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specifically for position sizing"""
        try:
            sizing_features = features.copy()
            
            # Add volatility features
            if 'price' in sizing_features.columns:
                sizing_features['volatility'] = sizing_features['price'].pct_change().rolling(20).std()
                sizing_features['price_momentum'] = sizing_features['price'].pct_change(5)
            
            # Add volume features
            if 'volume' in sizing_features.columns:
                sizing_features['volume_ratio'] = sizing_features['volume'] / sizing_features['volume'].rolling(20).mean()
            
            # Add market condition features
            if 'funding_rate' in sizing_features.columns:
                sizing_features['funding_rate_abs'] = sizing_features['funding_rate'].abs()
                sizing_features['funding_rate_change'] = sizing_features['funding_rate'].diff()
            
            # Fill NaN values
            sizing_features = sizing_features.fillna(0.0)
            
            # Select numeric columns only
            numeric_columns = sizing_features.select_dtypes(include=[np.number]).columns
            return sizing_features[numeric_columns]
            
        except Exception as e:
            self.logger.error(f"Error preparing sizing features: {e}")
            return pd.DataFrame()
    
    def _calculate_sizing_confidence(self, features: pd.DataFrame) -> float:
        """Calculate confidence level for position sizing"""
        try:
            confidence_factors = []
            
            # Volatility confidence (lower volatility = higher confidence)
            if 'volatility' in features.columns:
                vol_confidence = 1.0 - features['volatility'].iloc[-1] / 0.1  # Normalize to 10% volatility
                confidence_factors.append(np.clip(vol_confidence, 0.0, 1.0))
            
            # Volume confidence (higher volume = higher confidence)
            if 'volume_ratio' in features.columns:
                vol_ratio = features['volume_ratio'].iloc[-1]
                vol_confidence = min(vol_ratio / 2.0, 1.0)  # Normalize to 2x average volume
                confidence_factors.append(vol_confidence)
            
            # Funding rate confidence (stable funding = higher confidence)
            if 'funding_rate_change' in features.columns:
                funding_change = abs(features['funding_rate_change'].iloc[-1])
                funding_confidence = 1.0 - min(funding_change / 0.001, 1.0)  # Normalize to 0.1% change
                confidence_factors.append(funding_confidence)
            
            # Return average confidence
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating sizing confidence: {e}")
            return 0.5
    
    def _get_default_position_size(self, account_balance: float, max_risk: float) -> Dict[str, Any]:
        """Get default position size when ML prediction fails"""
        return {
            'position_size': 0.0,
            'position_value': account_balance * max_risk * 0.5,  # Conservative default
            'position_ratio': max_risk * 0.5,
            'confidence': 0.5,
            'risk_level': 'medium'
        }
    
    def get_prediction_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get prediction service summary"""
        try:
            summary = {
                'stats': self.stats,
                'models_available': {
                    sym: list(models.keys()) 
                    for sym, models in self.models.items()
                },
                'predictions_cached': {
                    sym: len(predictions) 
                    for sym, predictions in self.prediction_cache.items()
                }
            }
            
            if symbol:
                symbol_summary = {
                    'models': list(self.models.get(symbol, {}).keys()),
                    'cached_predictions': len(self.prediction_cache.get(symbol, [])),
                    'recent_predictions': [
                        {
                            'signal_type': p.signal_type,
                            'confidence': p.confidence,
                            'predicted_pnl': p.predicted_pnl,
                            'timestamp': p.timestamp.isoformat()
                        }
                        for p in list(self.prediction_cache.get(symbol, []))[-5:]
                    ]
                }
                summary['symbol_details'] = {symbol: symbol_summary}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting prediction summary: {e}")
            return {}
    
    async def close(self):
        """Close the predictive signal service"""
        try:
            # Clear caches
            self.prediction_cache.clear()
            self.logger.info("Predictive Signal service closed")
        except Exception as e:
            self.logger.error(f"Error closing predictive signal service: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
