#!/usr/bin/env python3
"""
Hybrid ML Pattern Detector
Combines TA-Lib with XGBoost for fuzzy pattern detection
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import talib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MLPatternResult:
    """Result from ML pattern detection"""
    pattern_name: str
    confidence: float
    probability: float
    features: Dict[str, float]
    is_fuzzy: bool
    talib_confidence: Optional[float] = None
    timestamp: Optional[datetime] = None
    price_level: Optional[float] = None

class FeatureExtractor:
    """Extract features for ML pattern detection"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_doji_features(self, opens: np.ndarray, highs: np.ndarray, 
                            lows: np.ndarray, closes: np.ndarray, 
                            volumes: np.ndarray) -> np.ndarray:
        """Extract features for doji pattern detection"""
        features = []
        
        for i in range(len(opens)):
            # Basic OHLC features
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            
            # Volume features
            volume_ratio = volumes[i] / np.mean(volumes[max(0, i-10):i+1]) if i > 0 else 1.0
            
            # Price movement features
            price_change = (closes[i] - opens[i]) / opens[i] if opens[i] != 0 else 0
            high_low_ratio = (highs[i] - lows[i]) / opens[i] if opens[i] != 0 else 0
            
            # Body ratio features
            body_ratio = body_size / total_range if total_range > 0 else 0
            shadow_ratio = (upper_shadow + lower_shadow) / total_range if total_range > 0 else 0
            
            # Trend features (using previous candles)
            if i > 0:
                prev_trend = (closes[i-1] - opens[i-1]) / opens[i-1] if opens[i-1] != 0 else 0
                trend_continuity = 1 if (price_change * prev_trend) > 0 else 0
            else:
                prev_trend = 0
                trend_continuity = 0
            
            # Volatility features
            if i >= 5:
                recent_volatility = np.std(closes[i-5:i+1]) / np.mean(closes[i-5:i+1])
            else:
                recent_volatility = 0
            
            feature_vector = [
                body_ratio,           # 0: Body to total range ratio
                shadow_ratio,         # 1: Shadow to total range ratio
                volume_ratio,         # 2: Volume ratio to recent average
                price_change,         # 3: Price change percentage
                high_low_ratio,       # 4: High-low range ratio
                prev_trend,           # 5: Previous candle trend
                trend_continuity,     # 6: Trend continuity indicator
                recent_volatility,    # 7: Recent volatility
                upper_shadow / total_range if total_range > 0 else 0,  # 8: Upper shadow ratio
                lower_shadow / total_range if total_range > 0 else 0   # 9: Lower shadow ratio
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_hammer_features(self, opens: np.ndarray, highs: np.ndarray, 
                              lows: np.ndarray, closes: np.ndarray, 
                              volumes: np.ndarray) -> np.ndarray:
        """Extract features for hammer pattern detection"""
        features = []
        
        for i in range(len(opens)):
            # Basic OHLC features
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            
            # Hammer-specific features
            body_ratio = body_size / total_range if total_range > 0 else 0
            lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else 0
            upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
            
            # Volume features
            volume_ratio = volumes[i] / np.mean(volumes[max(0, i-10):i+1]) if i > 0 else 1.0
            
            # Price movement features
            price_change = (closes[i] - opens[i]) / opens[i] if opens[i] != 0 else 0
            
            # Trend features
            if i > 0:
                prev_trend = (closes[i-1] - opens[i-1]) / opens[i-1] if opens[i-1] != 0 else 0
                downtrend_strength = sum(1 for j in range(max(0, i-3), i) 
                                       if closes[j] < opens[j]) / 3 if i >= 3 else 0
            else:
                prev_trend = 0
                downtrend_strength = 0
            
            # Support/resistance features
            if i >= 5:
                recent_low = np.min(lows[i-5:i+1])
                support_level = (lows[i] - recent_low) / recent_low if recent_low > 0 else 0
            else:
                support_level = 0
            
            feature_vector = [
                body_ratio,           # 0: Body to total range ratio
                lower_shadow_ratio,   # 1: Lower shadow ratio
                upper_shadow_ratio,   # 2: Upper shadow ratio
                volume_ratio,         # 3: Volume ratio
                price_change,         # 4: Price change
                prev_trend,           # 5: Previous trend
                downtrend_strength,   # 6: Downtrend strength
                support_level,        # 7: Support level proximity
                body_size / opens[i] if opens[i] != 0 else 0,  # 8: Body size ratio
                total_range / opens[i] if opens[i] != 0 else 0  # 9: Total range ratio
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_engulfing_features(self, opens: np.ndarray, highs: np.ndarray, 
                                 lows: np.ndarray, closes: np.ndarray, 
                                 volumes: np.ndarray) -> np.ndarray:
        """Extract features for engulfing pattern detection"""
        features = []
        
        for i in range(1, len(opens)):  # Start from 1 to compare with previous
            # Current and previous candle features
            curr_body_size = abs(closes[i] - opens[i])
            prev_body_size = abs(closes[i-1] - opens[i-1])
            
            # Engulfing ratios
            body_ratio = curr_body_size / prev_body_size if prev_body_size > 0 else 0
            engulfing_ratio = curr_body_size / (curr_body_size + prev_body_size) if (curr_body_size + prev_body_size) > 0 else 0
            
            # Volume features
            volume_ratio = volumes[i] / np.mean(volumes[max(0, i-10):i+1]) if i > 0 else 1.0
            
            # Price movement features
            curr_price_change = (closes[i] - opens[i]) / opens[i] if opens[i] != 0 else 0
            prev_price_change = (closes[i-1] - opens[i-1]) / opens[i-1] if opens[i-1] != 0 else 0
            
            # Trend features
            if i > 1:
                trend_strength = sum(1 for j in range(max(0, i-3), i) 
                                   if (closes[j] - opens[j]) * prev_price_change > 0) / 3
            else:
                trend_strength = 0
            
            # Volatility features
            if i >= 5:
                recent_volatility = np.std(closes[i-5:i+1]) / np.mean(closes[i-5:i+1])
            else:
                recent_volatility = 0
            
            # Momentum features
            momentum = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
            
            feature_vector = [
                body_ratio,           # 0: Current to previous body ratio
                engulfing_ratio,      # 1: Engulfing ratio
                volume_ratio,         # 2: Volume ratio
                curr_price_change,    # 3: Current price change
                prev_price_change,    # 4: Previous price change
                trend_strength,       # 5: Trend strength
                recent_volatility,    # 6: Recent volatility
                momentum,             # 7: Price momentum
                curr_body_size / opens[i] if opens[i] != 0 else 0,  # 8: Current body ratio
                prev_body_size / opens[i-1] if opens[i-1] != 0 else 0  # 9: Previous body ratio
            ]
            
            features.append(feature_vector)
        
        return np.array(features)

class HybridMLPatternDetector:
    """Hybrid pattern detector combining TA-Lib with ML"""
    
    def __init__(self, model_dir: str = "models/ml_patterns"):
        self.model_dir = model_dir
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.pattern_configs = {
            # Core patterns with ML feature extraction
            'doji': {
                'talib_func': talib.CDLDOJI,
                'feature_func': self.feature_extractor.extract_doji_features,
                'threshold': 0.6,
                'fuzzy_threshold': 0.4
            },
            'hammer': {
                'talib_func': talib.CDLHAMMER,
                'feature_func': self.feature_extractor.extract_hammer_features,
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'engulfing': {
                'talib_func': talib.CDLENGULFING,
                'feature_func': self.feature_extractor.extract_engulfing_features,
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            
            # Additional TA-Lib patterns (using generic feature extraction)
            'shooting_star': {
                'talib_func': talib.CDLSHOOTINGSTAR,
                'feature_func': self.feature_extractor.extract_doji_features,  # Similar to doji
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'morning_star': {
                'talib_func': talib.CDLMORNINGSTAR,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'evening_star': {
                'talib_func': talib.CDLEVENINGSTAR,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'three_white_soldiers': {
                'talib_func': talib.CDL3WHITESOLDIERS,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'three_black_crows': {
                'talib_func': talib.CDL3BLACKCROWS,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'hanging_man': {
                'talib_func': talib.CDLHANGINGMAN,
                'feature_func': self.feature_extractor.extract_hammer_features,  # Similar to hammer
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'inverted_hammer': {
                'talib_func': talib.CDLINVERTEDHAMMER,
                'feature_func': self.feature_extractor.extract_hammer_features,  # Similar to hammer
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'spinning_top': {
                'talib_func': talib.CDLSPINNINGTOP,
                'feature_func': self.feature_extractor.extract_doji_features,  # Similar to doji
                'threshold': 0.6,
                'fuzzy_threshold': 0.4
            },
            'marubozu': {
                'talib_func': talib.CDLMARUBOZU,
                'feature_func': self.feature_extractor.extract_doji_features,  # Body-focused
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'tristar': {
                'talib_func': talib.CDLTRISTAR,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'three_inside': {
                'talib_func': talib.CDL3INSIDE,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'three_outside': {
                'talib_func': talib.CDL3OUTSIDE,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'breakaway': {
                'talib_func': talib.CDLBREAKAWAY,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'dark_cloud_cover': {
                'talib_func': talib.CDLDARKCLOUDCOVER,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Two-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'dragonfly_doji': {
                'talib_func': talib.CDLDRAGONFLYDOJI,
                'feature_func': self.feature_extractor.extract_doji_features,  # Doji variant
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'gravestone_doji': {
                'talib_func': talib.CDLGRAVESTONEDOJI,
                'feature_func': self.feature_extractor.extract_doji_features,  # Doji variant
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'harami': {
                'talib_func': talib.CDLHARAMI,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Two-candle
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'harami_cross': {
                'talib_func': talib.CDLHARAMICROSS,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Two-candle
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'high_wave': {
                'talib_func': talib.CDLHIGHWAVE,
                'feature_func': self.feature_extractor.extract_doji_features,  # Similar to doji
                'threshold': 0.6,
                'fuzzy_threshold': 0.4
            },
            'identical_three_crows': {
                'talib_func': talib.CDLIDENTICAL3CROWS,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'kicking': {
                'talib_func': talib.CDLKICKING,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Two-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'ladder_bottom': {
                'talib_func': talib.CDLLADDERBOTTOM,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Multi-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            },
            'long_legged_doji': {
                'talib_func': talib.CDLLONGLEGGEDDOJI,
                'feature_func': self.feature_extractor.extract_doji_features,  # Doji variant
                'threshold': 0.6,
                'fuzzy_threshold': 0.4
            },
            'long_line': {
                'talib_func': talib.CDLLONGLINE,
                'feature_func': self.feature_extractor.extract_doji_features,  # Body-focused
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'on_neck': {
                'talib_func': talib.CDLONNECK,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Two-candle
                'threshold': 0.7,
                'fuzzy_threshold': 0.5
            },
            'piercing': {
                'talib_func': talib.CDLPIERCING,
                'feature_func': self.feature_extractor.extract_engulfing_features,  # Two-candle
                'threshold': 0.8,
                'fuzzy_threshold': 0.6
            }
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models or initialize new ones
        self._load_or_initialize_models()
        
        logger.info(f"🚀 Hybrid ML Pattern Detector initialized with {len(self.models)} models")
    
    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        for pattern_name in self.pattern_configs.keys():
            model_path = os.path.join(self.model_dir, f"{pattern_name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{pattern_name}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[pattern_name] = joblib.load(model_path)
                    self.scalers[pattern_name] = joblib.load(scaler_path)
                    logger.info(f"✅ Loaded existing model for {pattern_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model for {pattern_name}: {e}")
                    self._initialize_model(pattern_name)
            else:
                self._initialize_model(pattern_name)
    
    def _initialize_model(self, pattern_name: str):
        """Initialize a new ML model for a pattern"""
        self.models[pattern_name] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scalers[pattern_name] = StandardScaler()
        logger.info(f"🆕 Initialized new model for {pattern_name}")
    
    def train_model(self, pattern_name: str, training_data: List[Dict], 
                   labels: List[int], test_size: float = 0.2):
        """Train ML model for a specific pattern"""
        if pattern_name not in self.pattern_configs:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        # Extract features
        opens = np.array([d['open'] for d in training_data])
        highs = np.array([d['high'] for d in training_data])
        lows = np.array([d['low'] for d in training_data])
        closes = np.array([d['close'] for d in training_data])
        volumes = np.array([d.get('volume', 1000) for d in training_data])
        
        feature_func = self.pattern_configs[pattern_name]['feature_func']
        features = feature_func(opens, highs, lows, closes, volumes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scalers[pattern_name].fit_transform(X_train)
        X_test_scaled = self.scalers[pattern_name].transform(X_test)
        
        # Train model
        self.models[pattern_name].fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.models[pattern_name].predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"🎯 {pattern_name} model trained - Accuracy: {accuracy:.3f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{pattern_name}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{pattern_name}_scaler.pkl")
        
        joblib.dump(self.models[pattern_name], model_path)
        joblib.dump(self.scalers[pattern_name], scaler_path)
        
        return accuracy
    
    def detect_patterns(self, opens: np.ndarray, highs: np.ndarray, 
                       lows: np.ndarray, closes: np.ndarray, 
                       volumes: np.ndarray) -> List[MLPatternResult]:
        """Detect patterns using hybrid approach"""
        results = []
        
        for pattern_name, config in self.pattern_configs.items():
            # TA-Lib detection
            talib_result = config['talib_func'](opens, highs, lows, closes)
            
            # Extract features for ML
            feature_func = config['feature_func']
            features = feature_func(opens, highs, lows, closes, volumes)
            
            # Check if scaler is fitted, if not, fit it with current data
            try:
                features_scaled = self.scalers[pattern_name].transform(features)
            except:
                # If scaler is not fitted, fit it with current data
                self.scalers[pattern_name].fit(features)
                features_scaled = self.scalers[pattern_name].transform(features)
            
            # Check if model is fitted, if not, skip ML detection
            try:
                # ML prediction
                ml_probabilities = self.models[pattern_name].predict_proba(features_scaled)
                ml_predictions = self.models[pattern_name].predict(features_scaled)
            except:
                # If model is not fitted, use default values
                ml_probabilities = np.zeros((len(features), 2))
                ml_probabilities[:, 0] = 1.0  # Default to no pattern
                ml_predictions = np.zeros(len(features))
            
            # Combine TA-Lib and ML results
            # Note: features array might be shorter than opens array for some patterns (e.g., engulfing)
            max_index = min(len(opens), len(features))
            
            for i in range(max_index):
                talib_detected = talib_result[i] != 0
                ml_confidence = ml_probabilities[i][1] if len(ml_probabilities[i]) > 1 else ml_probabilities[i][0]
                ml_detected = ml_predictions[i] == 1
                
                # Determine if this is a fuzzy pattern
                is_fuzzy = not talib_detected and ml_confidence > config['fuzzy_threshold']
                
                # Calculate combined confidence
                if talib_detected and ml_detected:
                    confidence = (1.0 + ml_confidence) / 2  # High confidence
                elif talib_detected or ml_detected:
                    confidence = max(0.5, ml_confidence)  # Medium confidence
                elif is_fuzzy:
                    confidence = ml_confidence  # Fuzzy pattern confidence
                else:
                    continue  # No pattern detected
                
                # Only include if confidence meets threshold
                if confidence >= config['threshold'] or is_fuzzy:
                    # Extract feature values for this candle
                    feature_dict = {}
                    feature_names = [
                        'body_ratio', 'shadow_ratio', 'volume_ratio', 'price_change',
                        'high_low_ratio', 'prev_trend', 'trend_continuity', 'volatility',
                        'upper_shadow_ratio', 'lower_shadow_ratio'
                    ]
                    
                    for j, name in enumerate(feature_names):
                        if j < len(features[i]):
                            feature_dict[name] = float(features[i][j])
                    
                    result = MLPatternResult(
                        pattern_name=pattern_name,
                        confidence=confidence,
                        probability=ml_confidence,
                        features=feature_dict,
                        is_fuzzy=is_fuzzy,
                        talib_confidence=1.0 if talib_detected else 0.0,
                        timestamp=datetime.now(timezone.utc),
                        price_level=float(closes[i])
                    )
                    
                    results.append(result)
        
        return results
    
    def get_model_performance(self, pattern_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        if pattern_name not in self.models:
            return {"error": f"Model not found for {pattern_name}"}
        
        model = self.models[pattern_name]
        return {
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "feature_importances": model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else []
        }
    
    def save_models(self):
        """Save all models to disk"""
        for pattern_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{pattern_name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{pattern_name}_scaler.pkl")
            
            joblib.dump(self.models[pattern_name], model_path)
            joblib.dump(self.scalers[pattern_name], scaler_path)
        
        logger.info(f"💾 Saved {len(self.models)} models to {self.model_dir}")
    
    def load_models(self):
        """Load all models from disk"""
        self._load_or_initialize_models()
        logger.info(f"📂 Loaded {len(self.models)} models from {self.model_dir}")
