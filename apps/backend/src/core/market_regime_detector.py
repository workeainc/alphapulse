#!/usr/bin/env python3
"""
Market Regime Detection Module for AlphaPulse
Multi-metric regime classification with ML integration and stability controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from collections import deque
import redis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import optuna
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_TREND_BULL = "strong_trend_bull"
    STRONG_TREND_BEAR = "strong_trend_bear"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    VOLATILE_BREAKOUT = "volatile_breakout"
    CHOPPY = "choppy"

@dataclass
class RegimeMetrics:
    """Market regime classification metrics"""
    adx: float
    ma_slope: float
    bb_width: float
    atr: float
    rsi: float
    volume_ratio: float
    breakout_strength: float
    price_momentum: float
    volatility_score: float

@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: float
    duration_candles: int
    last_change: datetime
    metrics: RegimeMetrics
    stability_score: float

class MarketRegimeDetector:
    """
    Market Regime Detection with multi-metric classification and ML integration
    Optimized for <50ms latency and high accuracy (>80%)
    """
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 redis_client: Optional[redis.Redis] = None,
                 lookback_period: int = 10,
                 min_regime_duration: int = 5,
                 hysteresis_threshold: float = 0.2,
                 enable_ml: bool = True,
                 model_path: Optional[str] = None):
        """
        Initialize Market Regime Detector
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '15m')
            redis_client: Redis client for state persistence
            lookback_period: Number of candles for smoothing
            min_regime_duration: Minimum candles before regime change
            hysteresis_threshold: Threshold for regime change (0.2 = 20%)
            enable_ml: Enable ML-based regime classification
            model_path: Path to pre-trained ML model
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.redis_client = redis_client
        self.lookback_period = lookback_period
        self.min_regime_duration = min_regime_duration
        self.hysteresis_threshold = hysteresis_threshold
        self.enable_ml = enable_ml
        
        # Regime state
        self.current_regime = MarketRegime.RANGING
        self.regime_confidence = 0.5
        self.regime_duration = 0
        self.last_regime_change = datetime.now()
        self.stability_score = 0.5
        
        # Rolling buffers for smoothing
        self.regime_scores = deque(maxlen=lookback_period)
        self.metric_history = deque(maxlen=lookback_period)
        self.price_history = deque(maxlen=50)  # For MA slope calculation
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'adx', 'ma_slope', 'bb_width', 'atr', 'rsi', 
            'volume_ratio', 'breakout_strength', 'price_momentum', 'volatility_score'
        ]
        
        # Thresholds for rule-based classification
        self.thresholds = {
            'adx_trend': 25.0,
            'adx_strong_trend': 35.0,
            'ma_slope_bull': 0.0001,  # 0.01%
            'ma_slope_bear': -0.0001,  # -0.01%
            'bb_width_volatile': 0.05,
            'bb_width_breakout': 0.07,
            'rsi_overbought': 60.0,
            'rsi_oversold': 40.0,
            'volume_ratio_high': 1.5,
            'breakout_strength_high': 70.0
        }
        
        # Multi-timeframe alignment
        self.mtf_regimes = {}
        self.mtf_weights = {'1m': 0.1, '5m': 0.2, '15m': 0.4, '1h': 0.3}
        
        # Performance tracking
        self.update_count = 0
        self.avg_latency_ms = 0.0
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Load ML model if provided
        if model_path and enable_ml:
            self.load_ml_model(model_path)
        
        logger.info(f"Market Regime Detector initialized for {symbol} {timeframe}")
    
    def load_ml_model(self, model_path: str) -> bool:
        """Load pre-trained ML model"""
        try:
            self.ml_model = joblib.load(f"{model_path}_model.pkl")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            logger.info(f"ML model loaded from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            return False
    
    def save_ml_model(self, model_path: str) -> bool:
        """Save trained ML model"""
        try:
            if self.ml_model is not None:
                joblib.dump(self.ml_model, f"{model_path}_model.pkl")
                joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
                logger.info(f"ML model saved to {model_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")
        return False
    
    def calculate_ma_slope(self, prices: List[float], period: int = 50) -> float:
        """Calculate EMA slope for trend direction"""
        if len(prices) < period:
            return 0.0
        
        # Calculate EMA
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        # Calculate slope over last 10 periods
        recent_prices = list(self.price_history)[-10:]
        if len(recent_prices) >= 2:
            slope = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
            return slope / recent_prices[0]  # Normalize by price
        return 0.0
    
    def calculate_bb_width(self, bb_upper: float, bb_lower: float, bb_middle: float) -> float:
        """Calculate Bollinger Band width"""
        if bb_middle == 0:
            return 0.0
        return (bb_upper - bb_lower) / bb_middle
    
    def calculate_breakout_strength(self, volume: float, atr: float, adx: float) -> float:
        """Calculate composite breakout strength"""
        volume_multiplier = volume / 1000000  # Normalize volume
        atr_volatility = atr / 1000  # Normalize ATR
        adx_component = 1.0 if adx > 25 else 0.5
        
        return (volume_multiplier * 0.6 + atr_volatility * 0.3 + adx_component * 0.1) * 100
    
    def classify_regime_rule_based(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """
        Rule-based regime classification
        Returns (regime, confidence)
        """
        confidence = 0.0
        regime_scores = {
            MarketRegime.STRONG_TREND_BULL: 0.0,
            MarketRegime.STRONG_TREND_BEAR: 0.0,
            MarketRegime.WEAK_TREND: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.VOLATILE_BREAKOUT: 0.0,
            MarketRegime.CHOPPY: 0.0
        }
        
        # Strong Trend Bull
        if (metrics.adx > self.thresholds['adx_strong_trend'] and 
            metrics.ma_slope > self.thresholds['ma_slope_bull'] and
            metrics.rsi > self.thresholds['rsi_overbought']):
            regime_scores[MarketRegime.STRONG_TREND_BULL] = 0.9
        
        # Strong Trend Bear
        elif (metrics.adx > self.thresholds['adx_strong_trend'] and 
              metrics.ma_slope < self.thresholds['ma_slope_bear'] and
              metrics.rsi < self.thresholds['rsi_oversold']):
            regime_scores[MarketRegime.STRONG_TREND_BEAR] = 0.9
        
        # Weak Trend
        elif (metrics.adx > self.thresholds['adx_trend'] and 
              metrics.adx <= self.thresholds['adx_strong_trend']):
            if metrics.ma_slope > 0:
                regime_scores[MarketRegime.WEAK_TREND] = 0.7
            else:
                regime_scores[MarketRegime.WEAK_TREND] = 0.7
        
        # Volatile Breakout
        elif (metrics.bb_width > self.thresholds['bb_width_breakout'] and
              metrics.breakout_strength > self.thresholds['breakout_strength_high'] and
              metrics.volume_ratio > self.thresholds['volume_ratio_high']):
            regime_scores[MarketRegime.VOLATILE_BREAKOUT] = 0.8
        
        # Ranging
        elif (metrics.adx < self.thresholds['adx_trend'] and
              metrics.bb_width < self.thresholds['bb_width_volatile'] and
              abs(metrics.ma_slope) < abs(self.thresholds['ma_slope_bull'])):
            regime_scores[MarketRegime.RANGING] = 0.8
        
        # Choppy (default fallback)
        else:
            regime_scores[MarketRegime.CHOPPY] = 0.6
        
        # Find best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        return best_regime, confidence
    
    def classify_regime_ml(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """
        ML-based regime classification
        Returns (regime, confidence)
        """
        if self.ml_model is None:
            return self.classify_regime_rule_based(metrics)
        
        try:
            # Prepare features
            features = np.array([
                metrics.adx, metrics.ma_slope, metrics.bb_width, metrics.atr,
                metrics.rsi, metrics.volume_ratio, metrics.breakout_strength,
                metrics.price_momentum, metrics.volatility_score
            ]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            
            # Convert prediction to regime
            regime_map = {
                0: MarketRegime.STRONG_TREND_BULL,
                1: MarketRegime.STRONG_TREND_BEAR,
                2: MarketRegime.WEAK_TREND,
                3: MarketRegime.RANGING,
                4: MarketRegime.VOLATILE_BREAKOUT,
                5: MarketRegime.CHOPPY
            }
            
            regime = regime_map.get(prediction, MarketRegime.CHOPPY)
            confidence = max(probabilities)
            
            return regime, confidence
            
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, falling back to rule-based")
            return self.classify_regime_rule_based(metrics)
    
    def apply_smoothing(self, regime: MarketRegime, confidence: float) -> Tuple[MarketRegime, float]:
        """Apply Kalman-like smoothing to regime classification"""
        # Store regime score
        regime_score = confidence if regime == self.current_regime else -confidence
        self.regime_scores.append(regime_score)
        
        # Calculate smoothed confidence
        if len(self.regime_scores) >= 3:
            # Weight recent values more heavily
            weights = np.linspace(0.4, 1.0, len(self.regime_scores))
            weights = weights / weights.sum()
            
            smoothed_score = np.average(list(self.regime_scores), weights=weights)
            smoothed_confidence = abs(smoothed_score)
            
            # Determine regime based on smoothed score
            if smoothed_score > 0:
                smoothed_regime = self.current_regime
            else:
                smoothed_regime = regime
        else:
            smoothed_regime = regime
            smoothed_confidence = confidence
        
        return smoothed_regime, smoothed_confidence
    
    def check_regime_change(self, new_regime: MarketRegime, new_confidence: float) -> bool:
        """Check if regime change should be allowed"""
        # Minimum duration check
        if self.regime_duration < self.min_regime_duration:
            return False
        
        # Hysteresis check
        if new_regime == self.current_regime:
            return False
        
        # Confidence threshold check
        if new_confidence < self.regime_confidence + self.hysteresis_threshold:
            return False
        
        return True
    
    def update_regime(self, 
                     indicators: Dict[str, float],
                     candlestick: Dict[str, Any]) -> RegimeState:
        """
        Update market regime classification
        Returns current regime state
        """
        start_time = datetime.now()
        
        try:
            # Extract metrics from indicators
            metrics = RegimeMetrics(
                adx=indicators.get('adx', 0.0),
                ma_slope=self.calculate_ma_slope(list(self.price_history)),
                bb_width=self.calculate_bb_width(
                    indicators.get('bb_upper', 0.0),
                    indicators.get('bb_lower', 0.0),
                    indicators.get('bb_middle', 1.0)
                ),
                atr=indicators.get('atr', 0.0),
                rsi=indicators.get('rsi', 50.0),
                volume_ratio=candlestick.get('volume', 0.0) / max(indicators.get('volume_sma', 1.0), 1.0),
                breakout_strength=self.calculate_breakout_strength(
                    candlestick.get('volume', 0.0),
                    indicators.get('atr', 0.0),
                    indicators.get('adx', 0.0)
                ),
                price_momentum=(candlestick.get('close', 0.0) - candlestick.get('open', 0.0)) / max(candlestick.get('open', 1.0), 1.0),
                volatility_score=indicators.get('atr', 0.0) / max(candlestick.get('close', 1.0), 1.0)
            )
            
            # Update price history
            self.price_history.append(candlestick.get('close', 0.0))
            
            # Classify regime
            if self.enable_ml and self.ml_model is not None:
                regime, confidence = self.classify_regime_ml(metrics)
            else:
                regime, confidence = self.classify_regime_rule_based(metrics)
            
            # Apply smoothing
            smoothed_regime, smoothed_confidence = self.apply_smoothing(regime, confidence)
            
            # Check for regime change
            if self.check_regime_change(smoothed_regime, smoothed_confidence):
                self.current_regime = smoothed_regime
                self.regime_confidence = smoothed_confidence
                self.last_regime_change = datetime.now()
                self.regime_duration = 0
                
                # Persist to Redis
                self.persist_regime_state()
                
                logger.info(f"Regime changed to {smoothed_regime.value} (confidence: {smoothed_confidence:.2f})")
            else:
                self.regime_duration += 1
                self.regime_confidence = smoothed_confidence
            
            # Calculate stability score
            self.stability_score = self.calculate_stability_score()
            
            # Store metrics history
            self.metric_history.append(metrics)
            
            # Update performance metrics
            self.update_count += 1
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.avg_latency_ms = (self.avg_latency_ms * (self.update_count - 1) + latency_ms) / self.update_count
            
            # Create regime state
            regime_state = RegimeState(
                regime=self.current_regime,
                confidence=self.regime_confidence,
                duration_candles=self.regime_duration,
                last_change=self.last_regime_change,
                metrics=metrics,
                stability_score=self.stability_score
            )
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Error updating regime: {e}")
            # Return current state on error
            return RegimeState(
                regime=self.current_regime,
                confidence=self.regime_confidence,
                duration_candles=self.regime_duration,
                last_change=self.last_regime_change,
                metrics=RegimeMetrics(0.0, 0.0, 0.0, 0.0, 50.0, 1.0, 0.0, 0.0, 0.0),
                stability_score=self.stability_score
            )
    
    def calculate_stability_score(self) -> float:
        """Calculate regime stability score (0-1)"""
        if len(self.regime_scores) < 3:
            return 0.5
        
        # Calculate variance of recent regime scores
        recent_scores = list(self.regime_scores)[-5:]
        variance = np.var(recent_scores)
        
        # Convert to stability score (lower variance = higher stability)
        stability = max(0.0, 1.0 - variance)
        
        return stability
    
    def persist_regime_state(self):
        """Persist regime state to Redis"""
        if self.redis_client is None:
            return
        
        try:
            key = f"alphapulse:regime:{self.symbol}:{self.timeframe}"
            state_data = {
                'regime': self.current_regime.value,
                'confidence': self.regime_confidence,
                'duration': self.regime_duration,
                'last_change': self.last_regime_change.isoformat(),
                'stability_score': self.stability_score,
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.setex(key, 3600, json.dumps(state_data))  # 1 hour TTL
            
        except Exception as e:
            logger.warning(f"Failed to persist regime state: {e}")
    
    def get_regime_state(self) -> RegimeState:
        """Get current regime state"""
        return RegimeState(
            regime=self.current_regime,
            confidence=self.regime_confidence,
            duration_candles=self.regime_duration,
            last_change=self.last_regime_change,
            metrics=self.metric_history[-1] if self.metric_history else RegimeMetrics(0.0, 0.0, 0.0, 0.0, 50.0, 1.0, 0.0, 0.0, 0.0),
            stability_score=self.stability_score
        )
    
    def should_filter_signal(self, signal_confidence: float) -> bool:
        """Determine if signal should be filtered based on regime"""
        if self.current_regime == MarketRegime.CHOPPY:
            return signal_confidence < 0.85  # Higher threshold for choppy markets
        
        elif self.current_regime == MarketRegime.VOLATILE_BREAKOUT:
            return signal_confidence < 0.75  # Lower threshold for volatile markets
        
        elif self.current_regime in [MarketRegime.STRONG_TREND_BULL, MarketRegime.STRONG_TREND_BEAR]:
            return signal_confidence < 0.65  # Lower threshold for strong trends
        
        else:
            return signal_confidence < 0.70  # Default threshold
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'update_count': self.update_count,
            'avg_latency_ms': self.avg_latency_ms,
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'stability_score': self.stability_score,
            'regime_duration': self.regime_duration
        }
