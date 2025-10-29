"""
Technical Indicator Aggregator for AlphaPulse
Aggregates 50+ technical indicators into weighted scores for intelligent decision making
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class IndicatorCategory(Enum):
    """Indicator categories"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"

@dataclass
class AggregatedIndicatorResult:
    """Result from indicator aggregation"""
    technical_score: float  # 0-1
    trend_score: float  # 0-1
    momentum_score: float  # 0-1
    volatility_score: float  # 0-1
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    contributing_indicators: List[str]
    indicator_signals: Dict[str, float]
    reasoning: str
    calculation_time_ms: float

class TechnicalIndicatorAggregator:
    """
    Aggregates 50+ technical indicators into single weighted score
    
    Strategy: Calculate everything, aggregate intelligently
    - Trend Indicators (40% weight): EMA, SMA, MACD, ADX, Supertrend, HMA, Aroon
    - Momentum Indicators (35% weight): RSI, Stochastic, TSI, CMO, PPO, TRIX, KST
    - Volatility Indicators (25% weight): BB, ATR, Donchian, Keltner, Mass Index
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Technical Indicator Aggregator"""
        self.config = config or {}
        
        # Category weights (total = 1.0)
        self.category_weights = {
            IndicatorCategory.TREND: 0.40,
            IndicatorCategory.MOMENTUM: 0.35,
            IndicatorCategory.VOLATILITY: 0.25
        }
        
        # Individual indicator weights within each category
        self.trend_indicator_weights = {
            'ema_cross': 0.15,  # EMA crossover
            'sma_trend': 0.10,  # SMA alignment
            'macd': 0.15,  # MACD
            'adx': 0.12,  # ADX strength
            'supertrend': 0.12,  # Supertrend
            'hma': 0.10,  # Hull MA
            'aroon': 0.10,  # Aroon
            'dema_tema': 0.08,  # DEMA/TEMA
            'ichimoku': 0.08  # Ichimoku
        }
        
        self.momentum_indicator_weights = {
            'rsi': 0.15,  # RSI
            'stochastic': 0.12,  # Stochastic
            'tsi': 0.12,  # True Strength Index
            'williams_r': 0.08,  # Williams %R
            'cci': 0.08,  # CCI
            'cmo': 0.10,  # Chande Momentum
            'ppo': 0.10,  # PPO
            'trix': 0.10,  # TRIX
            'ultimate_osc': 0.08,  # Ultimate Oscillator
            'awesome_osc': 0.07  # Awesome Oscillator
        }
        
        self.volatility_indicator_weights = {
            'bollinger': 0.25,  # Bollinger Bands
            'atr': 0.20,  # ATR
            'donchian': 0.15,  # Donchian Channels
            'keltner': 0.15,  # Keltner Channels
            'mass_index': 0.12,  # Mass Index
            'chandelier': 0.13  # Chandelier Exit
        }
        
        # Thresholds for signal generation
        self.bullish_threshold = 0.55
        self.bearish_threshold = 0.45
        self.strong_signal_threshold = 0.65
        
        # Statistics
        self.stats = {
            'total_aggregations': 0,
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'avg_confidence': 0.0,
            'avg_calculation_time_ms': 0.0
        }
        
        logger.info("✅ Technical Indicator Aggregator initialized")
    
    async def aggregate_technical_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> AggregatedIndicatorResult:
        """
        Aggregate all technical indicators into single score
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dict of pre-calculated indicators
            
        Returns:
            AggregatedIndicatorResult with aggregated scores
        """
        start_time = datetime.now()
        
        try:
            if len(df) < 50:
                return self._get_default_result(0.0, "Insufficient data")
            
            # Calculate scores for each category
            trend_score, trend_signals = await self._calculate_trend_score(df, indicators)
            momentum_score, momentum_signals = await self._calculate_momentum_score(df, indicators)
            volatility_score, volatility_signals = await self._calculate_volatility_score(df, indicators)
            
            # Aggregate into final technical score
            technical_score = (
                trend_score * self.category_weights[IndicatorCategory.TREND] +
                momentum_score * self.category_weights[IndicatorCategory.MOMENTUM] +
                volatility_score * self.category_weights[IndicatorCategory.VOLATILITY]
            )
            
            # Determine direction
            if technical_score >= self.bullish_threshold:
                direction = "bullish"
            elif technical_score <= self.bearish_threshold:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                technical_score,
                trend_score,
                momentum_score,
                volatility_score
            )
            
            # Build reasoning
            reasoning = self._build_reasoning(
                direction,
                technical_score,
                trend_score,
                momentum_score,
                volatility_score,
                trend_signals,
                momentum_signals,
                volatility_signals
            )
            
            # Get contributing indicators
            all_signals = {**trend_signals, **momentum_signals, **volatility_signals}
            contributing_indicators = [
                name for name, signal in all_signals.items()
                if abs(signal - 0.5) > 0.15  # Contributing if signal is not neutral
            ]
            
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update stats
            self._update_stats(direction, confidence, calc_time)
            
            return AggregatedIndicatorResult(
                technical_score=technical_score,
                trend_score=trend_score,
                momentum_score=momentum_score,
                volatility_score=volatility_score,
                direction=direction,
                confidence=confidence,
                contributing_indicators=contributing_indicators,
                indicator_signals=all_signals,
                reasoning=reasoning,
                calculation_time_ms=calc_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error aggregating technical signals: {e}")
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            return self._get_default_result(calc_time, f"Error: {str(e)}")
    
    async def _calculate_trend_score(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted trend score from all trend indicators"""
        signals = {}
        
        try:
            # EMA Crossover (12/26 or similar)
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                ema_12 = df['ema_12'].iloc[-1]
                ema_26 = df['ema_26'].iloc[-1]
                signals['ema_cross'] = 1.0 if ema_12 > ema_26 else 0.0
            
            # SMA Trend (20/50/200 alignment)
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                close = df['close'].iloc[-1]
                alignment_score = 0.0
                if close > sma_20 > sma_50:
                    alignment_score = 1.0
                elif close < sma_20 < sma_50:
                    alignment_score = 0.0
                else:
                    alignment_score = 0.5
                signals['sma_trend'] = alignment_score
            
            # MACD
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_hist = macd - macd_signal
                # Normalize: positive = bullish (0.5-1.0), negative = bearish (0-0.5)
                signals['macd'] = self._normalize_to_01(macd_hist, -2, 2, center=0)
            
            # ADX
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
                # ADX shows strength, use with price trend
                if 'close' in df.columns and len(df) > 1:
                    price_trend = df['close'].iloc[-1] > df['close'].iloc[-10]
                    if adx > 25:  # Strong trend
                        signals['adx'] = 1.0 if price_trend else 0.0
                    else:
                        signals['adx'] = 0.5  # Weak trend
            
            # Supertrend (now in dataframe)
            if 'supertrend' in df.columns:
                close = df['close'].iloc[-1]
                supertrend = df['supertrend'].iloc[-1]
                signals['supertrend'] = 1.0 if close > supertrend else 0.0
            
            # HMA (Hull Moving Average - now in dataframe)
            if 'hma' in df.columns or 'hma_20' in df.columns:
                close = df['close'].iloc[-1]
                hma = df['hma_20'].iloc[-1] if 'hma_20' in df.columns else df['hma'].iloc[-1]
                signals['hma'] = 1.0 if close > hma else 0.0
            
            # Aroon
            if 'aroon_up' in df.columns and 'aroon_down' in df.columns:
                aroon_up = df['aroon_up'].iloc[-1]
                aroon_down = df['aroon_down'].iloc[-1]
                # Normalize aroon oscillator
                aroon_osc = (aroon_up - aroon_down) / 100  # -1 to 1
                signals['aroon'] = (aroon_osc + 1) / 2  # 0 to 1
            
            # DEMA/TEMA
            if 'dema' in df.columns:
                close = df['close'].iloc[-1]
                dema = df['dema'].iloc[-1]
                signals['dema_tema'] = 1.0 if close > dema else 0.0
            
            # Ichimoku
            if 'ichimoku_base' in df.columns:
                close = df['close'].iloc[-1]
                base = df['ichimoku_base'].iloc[-1]
                signals['ichimoku'] = 1.0 if close > base else 0.0
            
            # Calculate weighted trend score
            trend_score = 0.0
            total_weight = 0.0
            
            for indicator, weight in self.trend_indicator_weights.items():
                if indicator in signals:
                    trend_score += signals[indicator] * weight
                    total_weight += weight
            
            # Normalize by actual weights used
            if total_weight > 0:
                trend_score = trend_score / total_weight
            else:
                trend_score = 0.5  # Neutral
            
            return trend_score, signals
            
        except Exception as e:
            logger.error(f"❌ Error calculating trend score: {e}")
            return 0.5, {}
    
    async def _calculate_momentum_score(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted momentum score from all momentum indicators"""
        signals = {}
        
        try:
            # RSI
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                # Normalize RSI to 0-1
                signals['rsi'] = self._normalize_to_01(rsi, 0, 100, center=50)
            
            # Stochastic
            if 'stoch_k' in df.columns:
                stoch = df['stoch_k'].iloc[-1]
                signals['stochastic'] = self._normalize_to_01(stoch, 0, 100, center=50)
            
            # TSI (True Strength Index - now in dataframe)
            if 'tsi' in df.columns:
                tsi = df['tsi'].iloc[-1]
                # TSI ranges from -100 to 100
                signals['tsi'] = self._normalize_to_01(tsi, -100, 100, center=0)
            
            # Williams %R
            if 'williams_r' in df.columns:
                williams = df['williams_r'].iloc[-1]
                # Williams %R ranges from -100 to 0
                signals['williams_r'] = (williams + 100) / 100  # 0 to 1
            
            # CCI
            if 'cci' in df.columns:
                cci = df['cci'].iloc[-1]
                signals['cci'] = self._normalize_to_01(cci, -200, 200, center=0)
            
            # CMO (Chande Momentum Oscillator - now in dataframe)
            if 'cmo' in df.columns:
                cmo = df['cmo'].iloc[-1]
                # CMO ranges from -100 to 100
                signals['cmo'] = self._normalize_to_01(cmo, -100, 100, center=0)
            
            # PPO (Percentage Price Oscillator - now in dataframe)
            if 'ppo' in df.columns:
                ppo = df['ppo'].iloc[-1]
                # PPO is percentage, normalize around 0
                signals['ppo'] = self._normalize_to_01(ppo, -10, 10, center=0)
            
            # TRIX (now in dataframe)
            if 'trix' in df.columns:
                trix = df['trix'].iloc[-1]
                # TRIX is small percentage
                signals['trix'] = self._normalize_to_01(trix, -2, 2, center=0)
            
            # Ultimate Oscillator (now in dataframe)
            if 'ultimate_osc' in df.columns:
                ult_osc = df['ultimate_osc'].iloc[-1]
                # Ultimate Osc ranges from 0 to 100
                signals['ultimate_osc'] = self._normalize_to_01(ult_osc, 0, 100, center=50)
            
            # Awesome Oscillator (now in dataframe)
            if 'awesome_osc' in df.columns:
                ao = df['awesome_osc'].iloc[-1]
                # AO is difference, normalize
                signals['awesome_osc'] = 1.0 if ao > 0 else 0.0
            
            # Calculate weighted momentum score
            momentum_score = 0.0
            total_weight = 0.0
            
            for indicator, weight in self.momentum_indicator_weights.items():
                if indicator in signals:
                    momentum_score += signals[indicator] * weight
                    total_weight += weight
            
            # Normalize by actual weights used
            if total_weight > 0:
                momentum_score = momentum_score / total_weight
            else:
                momentum_score = 0.5  # Neutral
            
            return momentum_score, signals
            
        except Exception as e:
            logger.error(f"❌ Error calculating momentum score: {e}")
            return 0.5, {}
    
    async def _calculate_volatility_score(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted volatility score"""
        signals = {}
        
        try:
            close = df['close'].iloc[-1]
            
            # Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                bb_middle = df['bb_middle'].iloc[-1] if 'bb_middle' in df.columns else (bb_upper + bb_lower) / 2
                
                # Position within bands (0 = lower, 0.5 = middle, 1 = upper)
                bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                signals['bollinger'] = np.clip(bb_position, 0, 1)
            
            # ATR (higher ATR = higher volatility, but we normalize to price action context)
            if 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = (atr / close) * 100 if close > 0 else 0
                # Lower volatility favors continuation, higher volatility = caution
                # We use this as a confidence modifier, not a direction signal
                signals['atr'] = 0.5  # Neutral for direction
            
            # Donchian Channels
            if 'donchian_upper' in df.columns and 'donchian_lower' in df.columns:
                dc_upper = df['donchian_upper'].iloc[-1]
                dc_lower = df['donchian_lower'].iloc[-1]
                dc_position = (close - dc_lower) / (dc_upper - dc_lower) if dc_upper != dc_lower else 0.5
                signals['donchian'] = np.clip(dc_position, 0, 1)
            
            # Keltner Channels
            if 'keltner_upper' in df.columns and 'keltner_lower' in df.columns:
                kc_upper = df['keltner_upper'].iloc[-1]
                kc_lower = df['keltner_lower'].iloc[-1]
                kc_position = (close - kc_lower) / (kc_upper - kc_lower) if kc_upper != kc_lower else 0.5
                signals['keltner'] = np.clip(kc_position, 0, 1)
            
            # Mass Index (trend reversal indicator - now in dataframe)
            if 'mass_index' in df.columns:
                mi = df['mass_index'].iloc[-1]
                # Mass Index > 27 signals reversal
                signals['mass_index'] = 1.0 if mi > 27 else 0.5 if mi > 26.5 else 0.0
            
            # Chandelier Exit (now in dataframe)
            if 'chandelier_long' in df.columns and 'chandelier_short' in df.columns:
                close = df['close'].iloc[-1]
                chandelier_long = df['chandelier_long'].iloc[-1]
                chandelier_short = df['chandelier_short'].iloc[-1]
                # If price above chandelier_long = bullish, below chandelier_short = bearish
                if close > chandelier_long:
                    signals['chandelier'] = 1.0
                elif close < chandelier_short:
                    signals['chandelier'] = 0.0
                else:
                    signals['chandelier'] = 0.5
            
            # Calculate weighted volatility score
            volatility_score = 0.0
            total_weight = 0.0
            
            for indicator, weight in self.volatility_indicator_weights.items():
                if indicator in signals:
                    volatility_score += signals[indicator] * weight
                    total_weight += weight
            
            # Normalize by actual weights used
            if total_weight > 0:
                volatility_score = volatility_score / total_weight
            else:
                volatility_score = 0.5  # Neutral
            
            return volatility_score, signals
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility score: {e}")
            return 0.5, {}
    
    def _normalize_to_01(
        self,
        value: float,
        min_val: float,
        max_val: float,
        center: float = None
    ) -> float:
        """
        Normalize value to 0-1 range
        
        If center is provided:
        - Below center = 0-0.5
        - Above center = 0.5-1.0
        """
        if center is not None:
            if value < center:
                # Map [min_val, center] to [0, 0.5]
                normalized = 0.5 * (value - min_val) / (center - min_val)
            else:
                # Map [center, max_val] to [0.5, 1.0]
                normalized = 0.5 + 0.5 * (value - center) / (max_val - center)
        else:
            # Simple linear normalization
            normalized = (value - min_val) / (max_val - min_val)
        
        return np.clip(normalized, 0, 1)
    
    def _calculate_confidence(
        self,
        technical_score: float,
        trend_score: float,
        momentum_score: float,
        volatility_score: float
    ) -> float:
        """Calculate confidence based on category agreement"""
        # Confidence is higher when all categories agree
        scores = [trend_score, momentum_score, volatility_score]
        
        # Calculate standard deviation (lower = more agreement)
        std = np.std(scores)
        
        # Calculate distance from neutral (0.5)
        distance_from_neutral = abs(technical_score - 0.5)
        
        # Combine: high agreement + strong direction = high confidence
        agreement_confidence = 1.0 - (std / 0.5)  # 0-1
        direction_confidence = distance_from_neutral * 2  # 0-1
        
        confidence = (agreement_confidence * 0.6 + direction_confidence * 0.4)
        
        return np.clip(confidence, 0, 1)
    
    def _build_reasoning(
        self,
        direction: str,
        technical_score: float,
        trend_score: float,
        momentum_score: float,
        volatility_score: float,
        trend_signals: Dict[str, float],
        momentum_signals: Dict[str, float],
        volatility_signals: Dict[str, float]
    ) -> str:
        """Build human-readable reasoning string"""
        reasons = []
        
        # Overall direction
        reasons.append(f"Overall: {direction.upper()} (score: {technical_score:.3f})")
        
        # Category scores
        reasons.append(f"Trend: {trend_score:.3f}, Momentum: {momentum_score:.3f}, Volatility: {volatility_score:.3f}")
        
        # Top contributing signals
        all_signals = {**trend_signals, **momentum_signals, **volatility_signals}
        sorted_signals = sorted(all_signals.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        
        top_signals = sorted_signals[:5]  # Top 5 most significant
        if top_signals:
            signal_str = ", ".join([f"{name}:{val:.2f}" for name, val in top_signals])
            reasons.append(f"Key indicators: {signal_str}")
        
        return "; ".join(reasons)
    
    def _update_stats(self, direction: str, confidence: float, calc_time: float):
        """Update aggregation statistics"""
        self.stats['total_aggregations'] += 1
        
        if direction == 'bullish':
            self.stats['bullish_signals'] += 1
        elif direction == 'bearish':
            self.stats['bearish_signals'] += 1
        else:
            self.stats['neutral_signals'] += 1
        
        # Running average for confidence
        n = self.stats['total_aggregations']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (n - 1) + confidence) / n
        )
        
        # Running average for calc time
        self.stats['avg_calculation_time_ms'] = (
            (self.stats['avg_calculation_time_ms'] * (n - 1) + calc_time) / n
        )
    
    def _get_default_result(self, calc_time: float, reason: str) -> AggregatedIndicatorResult:
        """Return default neutral result"""
        return AggregatedIndicatorResult(
            technical_score=0.5,
            trend_score=0.5,
            momentum_score=0.5,
            volatility_score=0.5,
            direction="neutral",
            confidence=0.0,
            contributing_indicators=[],
            indicator_signals={},
            reasoning=reason,
            calculation_time_ms=calc_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            **self.stats,
            'bullish_rate': (
                self.stats['bullish_signals'] / max(1, self.stats['total_aggregations'])
            ),
            'bearish_rate': (
                self.stats['bearish_signals'] / max(1, self.stats['total_aggregations'])
            ),
            'neutral_rate': (
                self.stats['neutral_signals'] / max(1, self.stats['total_aggregations'])
            )
        }

