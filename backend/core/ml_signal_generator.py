#!/usr/bin/env python3
"""
ML Signal Generator for AlphaPulse
Machine learning-based pattern detection and signal generation with validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from indicators_engine import IndicatorValues

logger = logging.getLogger(__name__)

class SignalDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class MarketRegime(Enum):
    TRENDING = "trending"
    CHOPPY = "choppy"
    VOLATILE = "volatile"

class PatternType(Enum):
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    RSI_DIVERGENCE = "rsi_divergence"
    MACD_CROSSOVER = "macd_crossover"
    BB_BREAKOUT = "bb_breakout"
    FIBONACCI_BOUNCE = "fibonacci_bounce"
    PIVOT_BREAKOUT = "pivot_breakout"

@dataclass
class PatternSignal:
    """Pattern-based trading signal"""
    pattern_type: PatternType
    direction: SignalDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    take_profit_4: float
    pattern_strength: float
    volume_confirmation: bool
    trend_alignment: bool
    timestamp: datetime

@dataclass
class CandlestickPattern:
    """Candlestick pattern data"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

class MLSignalGenerator:
    """
    Machine Learning Signal Generator
    Combines pattern detection with ML-based confidence scoring
    """
    
    def __init__(self, 
                 confidence_thresholds: Dict[MarketRegime, float] = None,
                 max_workers: int = 4):
        """
        Initialize ML Signal Generator
        
        Args:
            confidence_thresholds: Confidence thresholds per market regime
            max_workers: Maximum thread pool workers for ML inference
        """
        self.confidence_thresholds = confidence_thresholds or {
            MarketRegime.TRENDING: 0.65,
            MarketRegime.CHOPPY: 0.80,
            MarketRegime.VOLATILE: 0.75
        }
        
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Pattern detection parameters
        self.min_pattern_confidence = 0.6
        self.volume_confirmation_threshold = 1.5  # Volume > 1.5x SMA
        self.trend_alignment_threshold = 25.0  # ADX > 25 for trend
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_filtered = 0
        self.pattern_detections = 0
        
        # Pattern history for analysis
        self.pattern_history = []
        self.signal_outcomes = []
        
        logger.info("ML Signal Generator initialized")
    
    async def generate_signal(self, 
                            candlestick: CandlestickPattern,
                            indicators: IndicatorValues,
                            market_regime: MarketRegime,
                            symbol: str,
                            timeframe: str) -> Optional[PatternSignal]:
        """
        Generate trading signal based on patterns and indicators
        
        Args:
            candlestick: Current candlestick data
            indicators: Technical indicator values
            market_regime: Current market regime
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            PatternSignal if valid signal found, None otherwise
        """
        try:
            # Detect patterns
            patterns = await self._detect_patterns(candlestick, indicators)
            
            if not patterns:
                return None
            
            # Score patterns and select best one
            best_pattern = await self._score_patterns(patterns, indicators, market_regime)
            
            if not best_pattern:
                return None
            
            # Validate signal
            validated_signal = await self._validate_signal(best_pattern, candlestick, indicators)
            
            if validated_signal:
                self.signals_generated += 1
                self.pattern_history.append({
                    'pattern': best_pattern.pattern_type.value,
                    'confidence': best_pattern.confidence,
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'timeframe': timeframe
                })
                
                logger.info(f"🎯 Signal generated: {symbol} {timeframe} {best_pattern.direction.value} "
                           f"({best_pattern.pattern_type.value}) - Confidence: {best_pattern.confidence:.2f}")
            
            return validated_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    async def _detect_patterns(self, 
                             candlestick: CandlestickPattern,
                             indicators: IndicatorValues) -> List[PatternSignal]:
        """Detect candlestick and indicator patterns"""
        patterns = []
        
        try:
            # Candlestick patterns
            candlestick_patterns = self._detect_candlestick_patterns(candlestick)
            patterns.extend(candlestick_patterns)
            
            # Indicator patterns
            indicator_patterns = self._detect_indicator_patterns(indicators)
            patterns.extend(indicator_patterns)
            
            # Fibonacci and pivot patterns
            fib_patterns = self._detect_fibonacci_patterns(candlestick, indicators)
            patterns.extend(fib_patterns)
            
            self.pattern_detections += len(patterns)
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_candlestick_patterns(self, candlestick: CandlestickPattern) -> List[PatternSignal]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            body_size = abs(candlestick.close - candlestick.open)
            total_range = candlestick.high - candlestick.low
            upper_shadow = candlestick.high - max(candlestick.open, candlestick.close)
            lower_shadow = min(candlestick.open, candlestick.close) - candlestick.low
            
            # Bullish Engulfing
            if (candlestick.close > candlestick.open and  # Current candle is bullish
                body_size > total_range * 0.6):  # Strong body
                patterns.append(PatternSignal(
                    pattern_type=PatternType.BULLISH_ENGULFING,
                    direction=SignalDirection.BUY,
                    confidence=0.75,
                    entry_price=candlestick.close,
                    stop_loss=candlestick.low * 0.99,
                    take_profit_1=candlestick.close * 1.01,
                    take_profit_2=candlestick.close * 1.02,
                    take_profit_3=candlestick.close * 1.03,
                    take_profit_4=candlestick.close * 1.05,
                    pattern_strength=body_size / total_range,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            
            # Bearish Engulfing
            elif (candlestick.close < candlestick.open and  # Current candle is bearish
                  body_size > total_range * 0.6):  # Strong body
                patterns.append(PatternSignal(
                    pattern_type=PatternType.BEARISH_ENGULFING,
                    direction=SignalDirection.SELL,
                    confidence=0.75,
                    entry_price=candlestick.close,
                    stop_loss=candlestick.high * 1.01,
                    take_profit_1=candlestick.close * 0.99,
                    take_profit_2=candlestick.close * 0.98,
                    take_profit_3=candlestick.close * 0.97,
                    take_profit_4=candlestick.close * 0.95,
                    pattern_strength=body_size / total_range,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            
            # Hammer
            elif (lower_shadow > body_size * 2 and  # Long lower shadow
                  upper_shadow < body_size * 0.5):  # Short upper shadow
                patterns.append(PatternSignal(
                    pattern_type=PatternType.HAMMER,
                    direction=SignalDirection.BUY,
                    confidence=0.70,
                    entry_price=candlestick.close,
                    stop_loss=candlestick.low * 0.99,
                    take_profit_1=candlestick.close * 1.015,
                    take_profit_2=candlestick.close * 1.025,
                    take_profit_3=candlestick.close * 1.035,
                    take_profit_4=candlestick.close * 1.045,
                    pattern_strength=lower_shadow / total_range,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            
            # Shooting Star
            elif (upper_shadow > body_size * 2 and  # Long upper shadow
                  lower_shadow < body_size * 0.5):  # Short lower shadow
                patterns.append(PatternSignal(
                    pattern_type=PatternType.SHOOTING_STAR,
                    direction=SignalDirection.SELL,
                    confidence=0.70,
                    entry_price=candlestick.close,
                    stop_loss=candlestick.high * 1.01,
                    take_profit_1=candlestick.close * 0.985,
                    take_profit_2=candlestick.close * 0.975,
                    take_profit_3=candlestick.close * 0.965,
                    take_profit_4=candlestick.close * 0.955,
                    pattern_strength=upper_shadow / total_range,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            
            # Doji
            elif body_size < total_range * 0.1:  # Very small body
                patterns.append(PatternSignal(
                    pattern_type=PatternType.DOJI,
                    direction=SignalDirection.HOLD,
                    confidence=0.50,
                    entry_price=candlestick.close,
                    stop_loss=candlestick.low * 0.99,
                    take_profit_1=candlestick.close * 1.005,
                    take_profit_2=candlestick.close * 1.01,
                    take_profit_3=candlestick.close * 1.015,
                    take_profit_4=candlestick.close * 1.02,
                    pattern_strength=0.5,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return patterns
    
    def _detect_indicator_patterns(self, indicators: IndicatorValues) -> List[PatternSignal]:
        """Detect indicator-based patterns"""
        patterns = []
        
        try:
            # RSI Divergence
            if indicators.rsi < 30:  # Oversold
                patterns.append(PatternSignal(
                    pattern_type=PatternType.RSI_DIVERGENCE,
                    direction=SignalDirection.BUY,
                    confidence=0.65,
                    entry_price=indicators.bb_middle,  # Use BB middle as entry
                    stop_loss=indicators.bb_lower * 0.99,
                    take_profit_1=indicators.bb_middle * 1.01,
                    take_profit_2=indicators.bb_middle * 1.02,
                    take_profit_3=indicators.bb_middle * 1.03,
                    take_profit_4=indicators.bb_middle * 1.05,
                    pattern_strength=(30 - indicators.rsi) / 30,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=datetime.now()
                ))
            elif indicators.rsi > 70:  # Overbought
                patterns.append(PatternSignal(
                    pattern_type=PatternType.RSI_DIVERGENCE,
                    direction=SignalDirection.SELL,
                    confidence=0.65,
                    entry_price=indicators.bb_middle,
                    stop_loss=indicators.bb_upper * 1.01,
                    take_profit_1=indicators.bb_middle * 0.99,
                    take_profit_2=indicators.bb_middle * 0.98,
                    take_profit_3=indicators.bb_middle * 0.97,
                    take_profit_4=indicators.bb_middle * 0.95,
                    pattern_strength=(indicators.rsi - 70) / 30,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=datetime.now()
                ))
            
            # MACD Crossover
            if (indicators.macd_line > indicators.macd_signal and 
                indicators.macd_histogram > 0):
                patterns.append(PatternSignal(
                    pattern_type=PatternType.MACD_CROSSOVER,
                    direction=SignalDirection.BUY,
                    confidence=0.70,
                    entry_price=indicators.bb_middle,
                    stop_loss=indicators.bb_lower * 0.99,
                    take_profit_1=indicators.bb_middle * 1.01,
                    take_profit_2=indicators.bb_middle * 1.02,
                    take_profit_3=indicators.bb_middle * 1.03,
                    take_profit_4=indicators.bb_middle * 1.05,
                    pattern_strength=abs(indicators.macd_histogram),
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=datetime.now()
                ))
            elif (indicators.macd_line < indicators.macd_signal and 
                  indicators.macd_histogram < 0):
                patterns.append(PatternSignal(
                    pattern_type=PatternType.MACD_CROSSOVER,
                    direction=SignalDirection.SELL,
                    confidence=0.70,
                    entry_price=indicators.bb_middle,
                    stop_loss=indicators.bb_upper * 1.01,
                    take_profit_1=indicators.bb_middle * 0.99,
                    take_profit_2=indicators.bb_middle * 0.98,
                    take_profit_3=indicators.bb_middle * 0.97,
                    take_profit_4=indicators.bb_middle * 0.95,
                    pattern_strength=abs(indicators.macd_histogram),
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=datetime.now()
                ))
            
            # Bollinger Bands Breakout
            if indicators.breakout_strength > 1.5:  # Strong breakout
                patterns.append(PatternSignal(
                    pattern_type=PatternType.BB_BREAKOUT,
                    direction=SignalDirection.BUY if indicators.breakout_strength > 2.0 else SignalDirection.SELL,
                    confidence=0.75,
                    entry_price=indicators.bb_middle,
                    stop_loss=indicators.bb_lower * 0.99,
                    take_profit_1=indicators.bb_middle * 1.015,
                    take_profit_2=indicators.bb_middle * 1.025,
                    take_profit_3=indicators.bb_middle * 1.035,
                    take_profit_4=indicators.bb_middle * 1.045,
                    pattern_strength=indicators.breakout_strength / 3.0,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=datetime.now()
                ))
            
        except Exception as e:
            logger.error(f"Error detecting indicator patterns: {e}")
        
        return patterns
    
    def _detect_fibonacci_patterns(self, 
                                 candlestick: CandlestickPattern,
                                 indicators: IndicatorValues) -> List[PatternSignal]:
        """Detect Fibonacci retracement and pivot patterns"""
        patterns = []
        
        try:
            current_price = candlestick.close
            
            # Fibonacci bounce levels
            fib_levels = [
                (indicators.fib_236, 0.236),
                (indicators.fib_382, 0.382),
                (indicators.fib_500, 0.500),
                (indicators.fib_618, 0.618)
            ]
            
            for fib_price, fib_level in fib_levels:
                # Check if price is near Fibonacci level (within 0.5%)
                if abs(current_price - fib_price) / fib_price < 0.005:
                    if fib_level <= 0.5:  # Support levels
                        patterns.append(PatternSignal(
                            pattern_type=PatternType.FIBONACCI_BOUNCE,
                            direction=SignalDirection.BUY,
                            confidence=0.70,
                            entry_price=current_price,
                            stop_loss=fib_price * 0.99,
                            take_profit_1=fib_price * 1.01,
                            take_profit_2=fib_price * 1.02,
                            take_profit_3=fib_price * 1.03,
                            take_profit_4=fib_price * 1.05,
                            pattern_strength=1.0 - fib_level,  # Stronger for lower levels
                            volume_confirmation=False,
                            trend_alignment=False,
                            timestamp=candlestick.timestamp
                        ))
                    else:  # Resistance levels
                        patterns.append(PatternSignal(
                            pattern_type=PatternType.FIBONACCI_BOUNCE,
                            direction=SignalDirection.SELL,
                            confidence=0.70,
                            entry_price=current_price,
                            stop_loss=fib_price * 1.01,
                            take_profit_1=fib_price * 0.99,
                            take_profit_2=fib_price * 0.98,
                            take_profit_3=fib_price * 0.97,
                            take_profit_4=fib_price * 0.95,
                            pattern_strength=fib_level,  # Stronger for higher levels
                            volume_confirmation=False,
                            trend_alignment=False,
                            timestamp=candlestick.timestamp
                        ))
            
            # Pivot point breakout
            if current_price > indicators.r1:  # Break above R1
                patterns.append(PatternSignal(
                    pattern_type=PatternType.PIVOT_BREAKOUT,
                    direction=SignalDirection.BUY,
                    confidence=0.75,
                    entry_price=current_price,
                    stop_loss=indicators.r1 * 0.99,
                    take_profit_1=current_price * 1.01,
                    take_profit_2=current_price * 1.02,
                    take_profit_3=current_price * 1.03,
                    take_profit_4=current_price * 1.05,
                    pattern_strength=1.0,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            elif current_price < indicators.s1:  # Break below S1
                patterns.append(PatternSignal(
                    pattern_type=PatternType.PIVOT_BREAKOUT,
                    direction=SignalDirection.SELL,
                    confidence=0.75,
                    entry_price=current_price,
                    stop_loss=indicators.s1 * 1.01,
                    take_profit_1=current_price * 0.99,
                    take_profit_2=current_price * 0.98,
                    take_profit_3=current_price * 0.97,
                    take_profit_4=current_price * 0.95,
                    pattern_strength=1.0,
                    volume_confirmation=False,
                    trend_alignment=False,
                    timestamp=candlestick.timestamp
                ))
            
        except Exception as e:
            logger.error(f"Error detecting Fibonacci patterns: {e}")
        
        return patterns
    
    async def _score_patterns(self, 
                            patterns: List[PatternSignal],
                            indicators: IndicatorValues,
                            market_regime: MarketRegime) -> Optional[PatternSignal]:
        """Score patterns and select the best one"""
        if not patterns:
            return None
        
        try:
            # Score each pattern
            scored_patterns = []
            for pattern in patterns:
                score = await self._calculate_pattern_score(pattern, indicators, market_regime)
                scored_patterns.append((pattern, score))
            
            # Sort by score and select best
            scored_patterns.sort(key=lambda x: x[1], reverse=True)
            best_pattern, best_score = scored_patterns[0]
            
            # Update pattern confidence with score
            best_pattern.confidence = best_score
            
            # Check if score meets minimum threshold
            threshold = self.confidence_thresholds[market_regime]
            if best_score < threshold:
                self.signals_filtered += 1
                return None
            
            return best_pattern
            
        except Exception as e:
            logger.error(f"Error scoring patterns: {e}")
            return None
    
    async def _calculate_pattern_score(self, 
                                     pattern: PatternSignal,
                                     indicators: IndicatorValues,
                                     market_regime: MarketRegime) -> float:
        """Calculate comprehensive pattern score"""
        try:
            base_score = pattern.confidence
            
            # Pattern strength multiplier
            strength_multiplier = 1.0 + (pattern.pattern_strength * 0.3)
            
            # Volume confirmation
            volume_multiplier = 1.2 if pattern.volume_confirmation else 1.0
            
            # Trend alignment
            trend_multiplier = 1.3 if pattern.trend_alignment else 1.0
            
            # Market regime adjustment
            regime_multiplier = 1.0
            if market_regime == MarketRegime.TRENDING:
                if pattern.direction in [SignalDirection.BUY, SignalDirection.SELL]:
                    regime_multiplier = 1.2
            elif market_regime == MarketRegime.CHOPPY:
                if pattern.direction == SignalDirection.HOLD:
                    regime_multiplier = 1.1
            
            # ADX strength
            adx_multiplier = 1.0 + (indicators.adx / 100) if indicators.adx > 25 else 0.9
            
            # Breakout strength
            breakout_multiplier = 1.0 + (indicators.breakout_strength / 3.0)
            
            # Calculate final score
            final_score = (base_score * 
                          strength_multiplier * 
                          volume_multiplier * 
                          trend_multiplier * 
                          regime_multiplier * 
                          adx_multiplier * 
                          breakout_multiplier)
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {e}")
            return pattern.confidence
    
    async def _validate_signal(self, 
                             pattern: PatternSignal,
                             candlestick: CandlestickPattern,
                             indicators: IndicatorValues) -> Optional[PatternSignal]:
        """Validate trading signal with multiple filters"""
        try:
            # Volume confirmation
            volume_confirmed = candlestick.volume > (indicators.volume_sma * self.volume_confirmation_threshold)
            pattern.volume_confirmation = volume_confirmed
            
            # Trend alignment
            trend_aligned = indicators.adx > self.trend_alignment_threshold
            pattern.trend_alignment = trend_aligned
            
            # Additional validation rules
            if pattern.direction == SignalDirection.BUY:
                # Bullish signal validation
                if indicators.rsi > 80:  # Overbought
                    return None
                if indicators.macd_histogram < -0.5:  # Strong bearish momentum
                    return None
            elif pattern.direction == SignalDirection.SELL:
                # Bearish signal validation
                if indicators.rsi < 20:  # Oversold
                    return None
                if indicators.macd_histogram > 0.5:  # Strong bullish momentum
                    return None
            
            # Update confidence based on validation
            if volume_confirmed:
                pattern.confidence *= 1.1
            if trend_aligned:
                pattern.confidence *= 1.1
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return None
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'signals_generated': self.signals_generated,
            'signals_filtered': self.signals_filtered,
            'pattern_detections': self.pattern_detections,
            'filter_rate': self.signals_filtered / max(self.signals_generated + self.signals_filtered, 1)
        }
    
    def get_pattern_history(self) -> List[dict]:
        """Get pattern detection history"""
        return self.pattern_history.copy()


# Example usage
async def main():
    """Example usage of ML Signal Generator"""
    # Initialize generator
    generator = MLSignalGenerator()
    
    # Sample data
    candlestick = CandlestickPattern(
        open=50000,
        high=50200,
        low=49900,
        close=50150,
        volume=1200,
        timestamp=datetime.now()
    )
    
    indicators = IndicatorValues(
        rsi=45.0,
        macd_line=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        bb_upper=50300,
        bb_middle=50100,
        bb_lower=49900,
        pivot=50100,
        s1=49900,
        r1=50300,
        fib_236=49950,
        fib_382=50000,
        fib_500=50050,
        fib_618=50100,
        breakout_strength=1.8,
        adx=30.0,
        atr=150.0,
        volume_sma=1000.0
    )
    
    # Generate signal
    signal = await generator.generate_signal(
        candlestick=candlestick,
        indicators=indicators,
        market_regime=MarketRegime.TRENDING,
        symbol="BTC/USDT",
        timeframe="1m"
    )
    
    if signal:
        print(f"🎯 Signal Generated:")
        print(f"  Pattern: {signal.pattern_type.value}")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Entry: {signal.entry_price:.2f}")
        print(f"  Stop Loss: {signal.stop_loss:.2f}")
        print(f"  Take Profit 1: {signal.take_profit_1:.2f}")
        print(f"  Volume Confirmed: {signal.volume_confirmation}")
        print(f"  Trend Aligned: {signal.trend_alignment}")
    else:
        print("❌ No signal generated")
    
    # Print performance stats
    stats = generator.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"  Signals Generated: {stats['signals_generated']}")
    print(f"  Signals Filtered: {stats['signals_filtered']}")
    print(f"  Filter Rate: {stats['filter_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
