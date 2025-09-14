#!/usr/bin/env python3
"""
ML-Enhanced Pattern Detector for AlphaPulse
Uses machine learning to improve pattern recognition accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MLPatternSignal:
    """Enhanced pattern signal with ML confidence scores"""
    pattern: str
    index: int
    strength: float
    type: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float
    ml_confidence: float
    timestamp: Optional[str] = None
    features: Optional[Dict] = None
    market_regime: Optional[str] = None

class MLPatternDetector:
    """
    Machine Learning enhanced pattern detector
    Combines traditional pattern recognition with ML predictions
    """
    
    def __init__(self, model_path: str = "models/pattern_detector.joblib"):
        """Initialize ML pattern detector"""
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
        # Load pre-trained model if available
        self._load_model()
        
        # Pattern definitions for feature engineering
        self.pattern_definitions = {
            'hammer': {'type': 'bullish', 'reliability': 0.7},
            'shooting_star': {'type': 'bearish', 'reliability': 0.7},
            'engulfing': {'type': 'both', 'reliability': 0.8},
            'doji': {'type': 'neutral', 'reliability': 0.6},
            'morning_star': {'type': 'bullish', 'reliability': 0.8},
            'evening_star': {'type': 'bearish', 'reliability': 0.8},
            'three_white_soldiers': {'type': 'bullish', 'reliability': 0.75},
            'three_black_crows': {'type': 'bearish', 'reliability': 0.75}
        }
    
    def _load_model(self):
        """Load pre-trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"Loaded pre-trained model from {self.model_path}")
            else:
                logger.info("No pre-trained model found, will train new model")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML pattern detection
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Price-based features
        features_df['price_change'] = features_df['close'].pct_change()
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['open_close_ratio'] = features_df['open'] / features_df['close']
        features_df['body_size'] = abs(features_df['close'] - features_df['open'])
        features_df['upper_shadow'] = features_df['high'] - np.maximum(features_df['open'], features_df['close'])
        features_df['lower_shadow'] = np.minimum(features_df['open'], features_df['close']) - features_df['low']
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_ma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
            features_df['volume_price_trend'] = (features_df['volume'] * features_df['price_change']).rolling(5).sum()
        
        # Technical indicators
        features_df['sma_20'] = features_df['close'].rolling(20).mean()
        features_df['sma_50'] = features_df['close'].rolling(50).mean()
        features_df['rsi'] = self._calculate_rsi(features_df['close'], 14)
        features_df['atr'] = self._calculate_atr(features_df, 14)
        features_df['bb_upper'], features_df['bb_lower'] = self._calculate_bollinger_bands(features_df['close'], 20, 2)
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Pattern-specific features
        features_df['hammer_score'] = self._calculate_hammer_score(features_df)
        features_df['doji_score'] = self._calculate_doji_score(features_df)
        features_df['engulfing_score'] = self._calculate_engulfing_score(features_df)
        
        # Market regime features
        features_df['volatility'] = features_df['close'].rolling(20).std() / features_df['close'].rolling(20).mean()
        features_df['trend_strength'] = abs(features_df['sma_20'] - features_df['sma_50']) / features_df['sma_50']
        
        # Time-based features
        features_df['hour'] = pd.to_datetime(features_df.index).hour
        features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_hammer_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate hammer pattern score"""
        body_size = df['body_size']
        lower_shadow = df['lower_shadow']
        upper_shadow = df['upper_shadow']
        
        # Hammer criteria: small body, long lower shadow, small upper shadow
        hammer_score = np.where(
            (body_size < 0.1 * (df['high'] - df['low'])) &
            (lower_shadow > 2 * body_size) &
            (upper_shadow < 0.1 * body_size),
            1.0, 0.0
        )
        return hammer_score
    
    def _calculate_doji_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate doji pattern score"""
        body_size = df['body_size']
        total_range = df['high'] - df['low']
        
        # Doji criteria: very small body relative to total range
        doji_score = np.where(
            body_size < 0.1 * total_range,
            1.0, 0.0
        )
        return doji_score
    
    def _calculate_engulfing_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate engulfing pattern score"""
        prev_body = df['body_size'].shift(1)
        curr_body = df['body_size']
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        curr_open = df['open']
        curr_close = df['close']
        
        # Bullish engulfing: current body engulfs previous body
        bullish_engulfing = np.where(
            (curr_open < prev_close) &
            (curr_close > prev_open) &
            (curr_body > prev_body),
            1.0, 0.0
        )
        
        # Bearish engulfing: current body engulfs previous body
        bearish_engulfing = np.where(
            (curr_open > prev_close) &
            (curr_close < prev_open) &
            (curr_body > prev_body),
            -1.0, 0.0
        )
        
        return bullish_engulfing + bearish_engulfing
    
    def prepare_training_data(self, df: pd.DataFrame, pattern_labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model
        
        Args:
            df: DataFrame with OHLCV data
            pattern_labels: List of pattern labels for each row
            
        Returns:
            Tuple of features and labels
        """
        # Engineer features
        features_df = self._engineer_features(df)
        
        # Define feature columns (exclude non-numeric and target columns)
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        self.feature_columns = [col for col in features_df.columns 
                               if col not in exclude_columns and features_df[col].dtype in ['float64', 'int64']]
        
        # Prepare features and labels
        X = features_df[self.feature_columns].values
        y = np.array(pattern_labels[:len(X)])  # Ensure alignment
        
        return X, y
    
    def train_model(self, df: pd.DataFrame, pattern_labels: List[str], test_size: float = 0.2):
        """
        Train the ML pattern detection model
        
        Args:
            df: DataFrame with OHLCV data
            pattern_labels: List of pattern labels
            test_size: Fraction of data to use for testing
        """
        logger.info("🚀 Training ML pattern detection model...")
        
        # Prepare training data
        X, y = self.prepare_training_data(df, pattern_labels)
        
        if len(X) < 100:
            logger.warning("⚠️ Insufficient data for training (need at least 100 samples)")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Model training completed!")
        logger.info(f"📊 Test accuracy: {accuracy:.3f}")
        logger.info(f"📈 Classification report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save model
        self._save_model()
        self.is_trained = True
        
        return True
    
    def detect_patterns_ml(self, df: pd.DataFrame) -> List[MLPatternSignal]:
        """
        Detect patterns using ML model
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of ML-enhanced pattern signals
        """
        if not self.is_trained or self.model is None:
            logger.warning("⚠️ Model not trained, falling back to basic pattern detection")
            return self._fallback_pattern_detection(df)
        
        # Engineer features
        features_df = self._engineer_features(df)
        
        if len(features_df) == 0:
            return []
        
        # Prepare features for prediction
        X = features_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create ML pattern signals
        signals = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred != 'no_pattern':
                # Get confidence from probability
                confidence = np.max(prob)
                
                # Get pattern metadata
                pattern_meta = self.pattern_definitions.get(pred, {})
                pattern_type = pattern_meta.get('type', 'neutral')
                
                # Calculate ML confidence (weighted by traditional reliability)
                traditional_reliability = pattern_meta.get('reliability', 0.5)
                ml_confidence = (confidence + traditional_reliability) / 2
                
                # Create signal
                signal = MLPatternSignal(
                    pattern=pred,
                    index=features_df.index[i],
                    strength=confidence,
                    type=pattern_type,
                    confidence=traditional_reliability,
                    ml_confidence=ml_confidence,
                    timestamp=features_df.index[i] if hasattr(features_df.index[i], 'isoformat') else str(features_df.index[i]),
                    features=dict(zip(self.feature_columns, X[i])),
                    market_regime=self._detect_market_regime(features_df.iloc[i])
                )
                signals.append(signal)
        
        return signals
    
    def _detect_market_regime(self, row: pd.Series) -> str:
        """Detect current market regime based on features"""
        volatility = row.get('volatility', 0)
        trend_strength = row.get('trend_strength', 0)
        
        if volatility > 0.03:  # High volatility
            if trend_strength > 0.02:  # Strong trend
                return 'trending_volatile'
            else:
                return 'ranging_volatile'
        else:  # Low volatility
            if trend_strength > 0.02:  # Strong trend
                return 'trending_stable'
            else:
                return 'ranging_stable'
    
    def _fallback_pattern_detection(self, df: pd.DataFrame) -> List[MLPatternSignal]:
        """Fallback to basic pattern detection when ML model is not available"""
        logger.info("🔄 Using fallback pattern detection")
        
        signals = []
        for i in range(len(df)):
            # Simple pattern detection logic
            if i < 2:  # Need at least 3 candles for patterns
                continue
                
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            prev_prev_row = df.iloc[i-2]
            
            # Hammer detection
            body_size = abs(row['close'] - row['open'])
            lower_shadow = min(row['open'], row['close']) - row['low']
            upper_shadow = row['high'] - max(row['open'], row['close'])
            
            if (body_size < 0.1 * (row['high'] - row['low']) and 
                lower_shadow > 2 * body_size and 
                upper_shadow < 0.1 * body_size):
                
                signal = MLPatternSignal(
                    pattern='hammer',
                    index=i,
                    strength=0.7,
                    type='bullish',
                    confidence=0.7,
                    ml_confidence=0.7,
                    timestamp=str(df.index[i]) if hasattr(df.index[i], 'isoformat') else str(df.index[i])
                )
                signals.append(signal)
        
        return signals
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "model_path": self.model_path,
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
            "model_type": type(self.model).__name__
        }
    
    def update_model(self, new_data: pd.DataFrame, new_labels: List[str]):
        """Update model with new data (online learning)"""
        if not self.is_trained:
            logger.warning("⚠️ Cannot update untrained model")
            return False
        
        logger.info("🔄 Updating model with new data...")
        
        # Prepare new data
        X_new, y_new = self.prepare_training_data(new_data, new_labels)
        
        if len(X_new) < 10:
            logger.warning("⚠️ Insufficient new data for model update")
            return False
        
        # Scale new data
        X_new_scaled = self.scaler.transform(X_new)
        
        # Update model (partial fit for online learning)
        try:
            self.model.partial_fit(X_new_scaled, y_new)
            logger.info("✅ Model updated successfully")
            self._save_model()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update model: {e}")
            return False
