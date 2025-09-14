#!/usr/bin/env python3
"""
ML Auto-Retraining Inference Engine
Integrates ML predictions with existing pattern detection system
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import psycopg2
from dataclasses import dataclass

def safe_float(value):
    """Convert numpy types to Python float"""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return 0.0

def safe_json_dumps(obj):
    """Safely serialize object to JSON, handling numpy types"""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    converted_obj = convert_numpy(obj)
    return json.dumps(converted_obj)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

@dataclass
class MLPrediction:
    """ML prediction result"""
    pattern_id: str
    symbol: str
    regime: str
    model_name: str
    model_version: int
    prediction_confidence: float
    prediction_probability: float
    prediction_class: int
    feature_values: Dict[str, float]
    market_conditions: Dict[str, Any]
    timestamp: datetime

class MLInferenceEngine:
    """ML Inference Engine for pattern prediction"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.model_cache = {}  # Cache loaded models
        self.feature_cache = {}  # Cache feature calculations
        
    def load_production_model(self, model_name: str, regime: str, symbol: str) -> Optional[Any]:
        """Load production model from database"""
        cache_key = f"{model_name}_{regime}_{symbol}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = """
                SELECT artifact_uri, version, metrics
                FROM ml_models
                WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'production'
                ORDER BY version DESC
                LIMIT 1
            """
            
            cursor.execute(query, (model_name, regime, symbol))
            result = cursor.fetchone()
            
            if result:
                artifact_uri, version, metrics = result
                if artifact_uri and os.path.exists(artifact_uri):
                    model = joblib.load(artifact_uri)
                    self.model_cache[cache_key] = {
                        'model': model,
                        'version': version,
                        'metrics': metrics,
                        'artifact_uri': artifact_uri
                    }
                    logger.info(f"✅ Loaded production model {model_name} v{version} for {regime} {symbol}")
                    return self.model_cache[cache_key]
                else:
                    logger.warning(f"⚠️ Model artifact not found: {artifact_uri}")
            else:
                logger.info(f"ℹ️ No production model found for {model_name} - {regime} - {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load production model: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        
        return None
    
    def create_features_from_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from OHLCV data (same as training)"""
        df = df.copy()
        
        # Price-based features
        df['returns_1m'] = df['close'].pct_change()
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        df['returns_1h'] = df['close'].pct_change(60)
        
        # Volatility features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_21'] = self._calculate_atr(df, 21)
        
        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['volume_std_20'] = df['volume'].rolling(20).std()
        df['volume_z_score'] = (df['volume'] - df['volume_ma_20']) / df['volume_std_20']
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price position features
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Support/Resistance features
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """Prepare features for ML prediction"""
        # Select feature columns (same as training)
        feature_columns = [
            'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
            'high_low_ratio', 'atr_14', 'atr_21',
            'volume_ratio', 'volume_z_score',
            'price_vs_sma20', 'price_vs_sma50', 'sma_cross',
            'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_position',
            'support_distance', 'resistance_distance'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        # Get the latest row for prediction
        latest_row = df[available_features].iloc[-1:].fillna(0)
        X = latest_row.values
        
        # Create feature values dictionary for tracking
        feature_values = {}
        for i, feature_name in enumerate(available_features):
            feature_values[feature_name] = safe_float(X[0, i])
        
        return X, feature_values
    
    def predict_pattern_success(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame, 
                              model_name: str = 'alphaplus_pattern_classifier') -> Optional[MLPrediction]:
        """Predict pattern success using ML model"""
        try:
            # Extract pattern information
            symbol = pattern_data.get('symbol', 'BTCUSDT')
            pattern_id = pattern_data.get('pattern_id', f"pattern_{datetime.now().timestamp()}")
            
            # Determine market regime (simplified - in production, use your regime classifier)
            regime = self._determine_market_regime(market_data)
            
            # Load production model for this regime and symbol
            model_info = self.load_production_model(model_name, regime, symbol)
            
            if not model_info:
                logger.warning(f"⚠️ No ML model available for {regime} {symbol}")
                return None
            
            # Create features from market data
            feature_df = self.create_features_from_ohlcv(market_data)
            
            if len(feature_df) < 50:  # Need sufficient data for features
                logger.warning("⚠️ Insufficient market data for feature creation")
                return None
            
            # Prepare features for prediction
            X, feature_values = self.prepare_features_for_prediction(feature_df)
            
            # Make prediction
            model = model_info['model']
            prediction_proba = model.predict_proba(X)[0]
            prediction_class = model.predict(X)[0]
            
            # Calculate confidence based on probability
            prediction_confidence = safe_float(max(prediction_proba))
            
            # Create market conditions dictionary
            market_conditions = {
                'current_price': safe_float(market_data['close'].iloc[-1]),
                'volume_ratio': safe_float(feature_values.get('volume_ratio', 0)),
                'rsi_14': safe_float(feature_values.get('rsi_14', 0)),
                'atr_14': safe_float(feature_values.get('atr_14', 0)),
                'bb_position': safe_float(feature_values.get('bb_position', 0)),
                'regime': regime
            }
            
            # Create prediction result
            prediction = MLPrediction(
                pattern_id=pattern_id,
                symbol=symbol,
                regime=regime,
                model_name=model_name,
                model_version=model_info['version'],
                prediction_confidence=prediction_confidence,
                prediction_probability=safe_float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]),
                prediction_class=int(prediction_class),
                feature_values=feature_values,
                market_conditions=market_conditions,
                timestamp=datetime.now()
            )
            
            # Store prediction for performance tracking
            self._store_prediction(prediction)
            
            logger.info(f"✅ ML prediction: {prediction_class} (confidence: {prediction_confidence:.3f})")
            return prediction
            
        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            return None
    
    def _determine_market_regime(self, market_data: pd.DataFrame) -> str:
        """Determine market regime from market data (simplified)"""
        try:
            if len(market_data) < 20:
                return 'unknown'
            
            # Calculate trend strength
            returns = market_data['close'].pct_change().dropna()
            trend_strength = abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
            
            # Calculate overall price direction
            price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]
            strong_trend = abs(price_change) > 0.05  # 5% overall change indicates trend
            
            # Calculate volatility
            volatility = returns.std()
            
            # Calculate volume trend
            volume_ma = market_data['volume'].rolling(20).mean()
            volume_trend = (market_data['volume'].iloc[-1] / volume_ma.iloc[-1]) if volume_ma.iloc[-1] > 0 else 1
            
            # Enhanced regime classification
            if strong_trend or (trend_strength > 0.02 and volume_trend > 1.1):
                return 'trending'
            elif volatility > 0.025:
                return 'volatile'
            elif trend_strength < 0.01:
                return 'sideways'
            else:
                return 'consolidation'
                
        except Exception as e:
            logger.error(f"❌ Market regime determination failed: {e}")
            return 'unknown'
    
    def _store_prediction(self, prediction: MLPrediction):
        """Store prediction for performance tracking"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ml_performance_tracking (
                    timestamp, model_name, model_version, regime, symbol, pattern_id,
                    prediction_confidence, market_conditions, feature_values
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
                )
            """, (
                prediction.timestamp,
                prediction.model_name,
                prediction.model_version,
                prediction.regime,
                prediction.symbol,
                prediction.pattern_id,
                safe_float(prediction.prediction_confidence),
                safe_json_dumps(prediction.market_conditions),
                safe_json_dumps(prediction.feature_values)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to store prediction: {e}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_model_performance_summary(self, model_name: str, regime: str, symbol: str, 
                                    days: int = 30) -> Dict[str, Any]:
        """Get model performance summary"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get recent predictions
            cursor.execute("""
                SELECT 
                    prediction_confidence,
                    prediction_correct,
                    actual_outcome,
                    profit_loss
                FROM ml_performance_tracking
                WHERE model_name = %s AND regime = %s AND symbol = %s 
                AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
            """, (model_name, regime, symbol, days))
            
            results = cursor.fetchall()
            
            if not results:
                return {
                    'model_name': model_name,
                    'regime': regime,
                    'symbol': symbol,
                    'total_predictions': 0,
                    'accuracy': 0.0,
                    'avg_confidence': 0.0,
                    'avg_profit_loss': 0.0
                }
            
            # Calculate metrics
            total_predictions = len(results)
            correct_predictions = sum(1 for r in results if r[2] == 'success')  # actual_outcome
            avg_confidence = sum(r[0] for r in results) / total_predictions
            avg_profit_loss = sum(r[3] for r in results if r[3] is not None) / total_predictions
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'model_name': model_name,
                'regime': regime,
                'symbol': symbol,
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'avg_profit_loss': avg_profit_loss,
                'correct_predictions': correct_predictions
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up ML inference engine...")
        self.model_cache.clear()
        self.feature_cache.clear()
        logger.info("✅ ML inference engine cleanup completed")

# Integration with existing pattern detection
class EnhancedPatternDetector:
    """Enhanced pattern detector with ML predictions"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.ml_engine = MLInferenceEngine(db_config)
        
    async def detect_patterns_with_ml(self, df: pd.DataFrame, symbol: str, 
                                    pattern_signals: List[Any]) -> List[Any]:
        """Detect patterns and enhance with ML predictions"""
        logger.info(f"🔍 Detecting patterns with ML enhancement for {symbol}")
        
        enhanced_signals = []
        
        for signal in pattern_signals:
            try:
                # Create pattern data for ML prediction
                pattern_data = {
                    'pattern_id': f"{signal.pattern_name}_{datetime.now().timestamp()}",
                    'symbol': symbol,
                    'pattern_name': signal.pattern_name,
                    'confidence': signal.confidence,
                    'timestamp': datetime.now()
                }
                
                # Get ML prediction
                ml_prediction = self.ml_engine.predict_pattern_success(pattern_data, df)
                
                if ml_prediction:
                    # Enhance signal with ML prediction
                    enhanced_signal = self._enhance_signal_with_ml(signal, ml_prediction)
                    enhanced_signals.append(enhanced_signal)
                else:
                    # Use original signal if no ML prediction available
                    enhanced_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"❌ Failed to enhance signal with ML: {e}")
                enhanced_signals.append(signal)
        
        logger.info(f"✅ Enhanced {len(enhanced_signals)} patterns with ML predictions")
        return enhanced_signals
    
    def _enhance_signal_with_ml(self, signal: Any, ml_prediction: MLPrediction) -> Any:
        """Enhance pattern signal with ML prediction"""
        # Create enhanced signal (modify based on your signal structure)
        enhanced_signal = signal
        
        # Add ML prediction information
        enhanced_signal.ml_prediction = {
            'prediction_class': ml_prediction.prediction_class,
            'prediction_confidence': ml_prediction.prediction_confidence,
            'prediction_probability': ml_prediction.prediction_probability,
            'model_version': ml_prediction.model_version,
            'regime': ml_prediction.regime
        }
        
        # Adjust confidence based on ML prediction
        if ml_prediction.prediction_class == 1:  # Positive prediction
            enhanced_signal.confidence = min(1.0, signal.confidence * 1.2)  # Boost confidence
        else:  # Negative prediction
            enhanced_signal.confidence = max(0.1, signal.confidence * 0.8)  # Reduce confidence
        
        return enhanced_signal
    
    def get_ml_performance_summary(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get ML performance summary for all regimes"""
        regimes = ['trending', 'sideways', 'volatile', 'consolidation']
        summaries = {}
        
        for regime in regimes:
            summary = self.ml_engine.get_model_performance_summary(
                'alphaplus_pattern_classifier', regime, symbol, days
            )
            summaries[regime] = summary
        
        return summaries
    
    def cleanup(self):
        """Cleanup resources"""
        self.ml_engine.cleanup()

# Example usage
if __name__ == "__main__":
    # Test ML inference engine
    engine = MLInferenceEngine(DB_CONFIG)
    
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(45000, 55000, len(dates)),
        'high': np.random.uniform(45000, 55000, len(dates)),
        'low': np.random.uniform(45000, 55000, len(dates)),
        'close': np.random.uniform(45000, 55000, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)
    
    # Sample pattern data
    pattern_data = {
        'pattern_id': 'test_pattern_001',
        'symbol': 'BTCUSDT',
        'pattern_name': 'hammer',
        'confidence': 0.8
    }
    
    # Test prediction
    prediction = engine.predict_pattern_success(pattern_data, sample_data)
    
    if prediction:
        print(f"ML Prediction: {prediction.prediction_class}")
        print(f"Confidence: {prediction.prediction_confidence:.3f}")
        print(f"Regime: {prediction.regime}")
    
    engine.cleanup()
