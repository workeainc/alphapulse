#!/usr/bin/env python3
"""
Real-Time Signal Generator for AlphaPulse
Generates high-confidence trading signals using ML-enhanced pattern detection
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import json

from .ml_pattern_detector import MLPatternDetector, MLPatternSignal
from .indicators import TechnicalIndicators
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Complete trading signal with all necessary information"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    pattern: str
    price: float
    timestamp: datetime
    timeframe: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    market_regime: Optional[str] = None
    volume_confirmation: bool = False
    trend_alignment: bool = False
    support_resistance_levels: Optional[Dict] = None
    additional_indicators: Optional[Dict] = None

class RealTimeSignalGenerator:
    """
    Real-time signal generator using ML-enhanced pattern detection
    Combines multiple data sources for high-confidence signals
    """
    
    def __init__(self, config: Dict = None):
        """Initialize signal generator"""
        self.config = config or {}
        self.ml_detector = MLPatternDetector()
        self.indicators = TechnicalIndicators()
        
        # Signal configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_strength = self.config.get('min_strength', 0.6)
        self.confirmation_required = self.config.get('confirmation_required', True)
        self.volume_confirmation = self.config.get('volume_confirmation', True)
        self.trend_confirmation = self.config.get('trend_confirmation', True)
        
        # Signal history for analysis
        self.signal_history = deque(maxlen=1000)
        self.pattern_history = deque(maxlen=500)
        
        # Market regime detection
        self.market_regimes = {}
        self.volatility_thresholds = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }
        
        # Performance tracking
        self.signal_performance = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'accuracy': 0.0
        }
        
        logger.info("Real-Time Signal Generator initialized")
    
    async def generate_signal(self, 
                            symbol: str, 
                            data: pd.DataFrame, 
                            timeframe: str,
                            market_data: Dict = None) -> Optional[TradingSignal]:
        """
        Generate real-time trading signal
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            timeframe: Timeframe of the data
            market_data: Additional market data (volume, order book, etc.)
            
        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        try:
            if len(data) < 50:  # Need sufficient data for analysis
                return None
            
            # Detect patterns using ML
            patterns = self.ml_detector.detect_patterns_ml(data)
            
            if not patterns:
                return None
            
            # Get latest pattern
            latest_pattern = patterns[-1]
            
            # Check if pattern meets minimum requirements
            if (latest_pattern.ml_confidence < self.min_confidence or 
                latest_pattern.strength < self.min_strength):
                return None
            
            # Generate signal based on pattern
            signal = await self._create_signal_from_pattern(
                symbol, latest_pattern, data, timeframe, market_data
            )
            
            if signal:
                # Validate signal with additional confirmations
                if self._validate_signal(signal, data, market_data):
                    # Store signal in history
                    self.signal_history.append(signal)
                    self.pattern_history.append(latest_pattern)
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    logger.info(f"🎯 Generated {signal.signal_type} signal for {symbol} "
                              f"(confidence: {signal.confidence:.3f}, strength: {signal.strength:.3f})")
                    
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None
    
    async def _create_signal_from_pattern(self, 
                                        symbol: str, 
                                        pattern: MLPatternSignal, 
                                        data: pd.DataFrame, 
                                        timeframe: str,
                                        market_data: Dict) -> Optional[TradingSignal]:
        """Create trading signal from detected pattern"""
        
        # Determine signal type based on pattern
        if pattern.type == 'bullish':
            signal_type = 'buy'
        elif pattern.type == 'bearish':
            signal_type = 'sell'
        else:
            signal_type = 'hold'
        
        if signal_type == 'hold':
            return None
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(
            signal_type, current_price, data, pattern
        )
        
        # Calculate risk-reward ratio
        risk_reward_ratio = None
        if stop_loss and take_profit:
            if signal_type == 'buy':
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            if risk > 0:
                risk_reward_ratio = reward / risk
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=pattern.strength,
            confidence=pattern.ml_confidence,
            pattern=pattern.pattern,
            price=current_price,
            timestamp=datetime.now(),
            timeframe=timeframe,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            market_regime=pattern.market_regime,
            volume_confirmation=self._check_volume_confirmation(data, signal_type),
            trend_alignment=self._check_trend_alignment(data, signal_type),
            support_resistance_levels=self._get_support_resistance_levels(data, current_price),
            additional_indicators=self._get_additional_indicators(data)
        )
        
        return signal
    
    def _calculate_risk_levels(self, 
                              signal_type: str, 
                              current_price: float, 
                              data: pd.DataFrame, 
                              pattern: MLPatternSignal) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        
        # Calculate ATR for dynamic levels
        atr = self.indicators.calculate_atr(data, period=14)
        if pd.isna(atr.iloc[-1]):
            atr_value = current_price * 0.02  # Default 2%
        else:
            atr_value = atr.iloc[-1]
        
        # Calculate support and resistance levels
        support_levels = self._get_support_levels(data)
        resistance_levels = self._get_resistance_levels(data)
        
        if signal_type == 'buy':
            # Stop loss: below recent support or ATR-based
            if support_levels:
                stop_loss = max(support_levels[-1], current_price - (2 * atr_value))
            else:
                stop_loss = current_price - (2 * atr_value)
            
            # Take profit: above recent resistance or ATR-based
            if resistance_levels:
                take_profit = max(resistance_levels[-1], current_price + (3 * atr_value))
            else:
                take_profit = current_price + (3 * atr_value)
                
        else:  # sell signal
            # Stop loss: above recent resistance or ATR-based
            if resistance_levels:
                stop_loss = min(resistance_levels[-1], current_price + (2 * atr_value))
            else:
                stop_loss = current_price + (2 * atr_value)
            
            # Take profit: below recent support or ATR-based
            if support_levels:
                take_profit = min(support_levels[-1], current_price - (3 * atr_value))
            else:
                take_profit = current_price - (3 * atr_value)
        
        return stop_loss, take_profit
    
    def _get_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Identify support levels from price data"""
        support_levels = []
        
        # Simple support detection using local minima
        for i in range(2, len(data) - 2):
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                
                support_levels.append(data['low'].iloc[i])
        
        return sorted(support_levels, reverse=True)
    
    def _get_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Identify resistance levels from price data"""
        resistance_levels = []
        
        # Simple resistance detection using local maxima
        for i in range(2, len(data) - 2):
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                
                resistance_levels.append(data['high'].iloc[i])
        
        return sorted(resistance_levels)
    
    def _check_volume_confirmation(self, data: pd.DataFrame, signal_type: str) -> bool:
        """Check if volume confirms the signal"""
        if not self.volume_confirmation or 'volume' not in data.columns:
            return True
        
        # Calculate volume moving average
        volume_ma = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        # Volume should be above average for confirmation
        return current_volume > avg_volume * 1.2
    
    def _check_trend_alignment(self, data: pd.DataFrame, signal_type: str) -> bool:
        """Check if signal aligns with current trend"""
        if not self.trend_confirmation:
            return True
        
        # Calculate trend using moving averages
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        if signal_type == 'buy':
            # Bullish signal should align with uptrend
            return current_sma_20 > current_sma_50
        else:
            # Bearish signal should align with downtrend
            return current_sma_20 < current_sma_50
    
    def _get_support_resistance_levels(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Get current support and resistance levels"""
        support_levels = self._get_support_levels(data)
        resistance_levels = self._get_resistance_levels(data)
        
        # Find nearest levels
        nearest_support = None
        nearest_resistance = None
        
        for level in support_levels:
            if level < current_price:
                nearest_support = level
                break
        
        for level in resistance_levels:
            if level > current_price:
                nearest_resistance = level
                break
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_levels': support_levels[:5],  # Top 5 levels
            'resistance_levels': resistance_levels[:5]  # Top 5 levels
        }
    
    def _get_additional_indicators(self, data: pd.DataFrame) -> Dict:
        """Get additional technical indicators for signal context"""
        try:
            rsi = self.indicators.calculate_rsi(data, period=14)
            macd, macd_signal, macd_hist = self.indicators.calculate_macd(data)
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(data)
            
            return {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None,
                'bb_position': self._calculate_bb_position(data, bb_upper, bb_middle, bb_lower),
                'trend_strength': self._calculate_trend_strength(data)
            }
        except Exception as e:
            logger.warning(f"⚠️ Error calculating additional indicators: {e}")
            return {}
    
    def _calculate_bb_position(self, data: pd.DataFrame, bb_upper: pd.Series, bb_middle: pd.Series, bb_lower: pd.Series) -> Optional[float]:
        """Calculate position within Bollinger Bands"""
        try:
            current_price = data['close'].iloc[-1]
            upper = bb_upper.iloc[-1]
            lower = bb_lower.iloc[-1]
            
            if pd.isna(upper) or pd.isna(lower):
                return None
            
            if upper == lower:
                return 0.5
            
            return (current_price - lower) / (upper - lower)
        except:
            return None
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate trend strength using ADX-like measure"""
        try:
            # Simple trend strength calculation
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            if pd.isna(current_sma_20) or pd.isna(current_sma_50):
                return None
            
            # Trend strength as percentage difference
            trend_strength = abs(current_sma_20 - current_sma_50) / current_sma_50 * 100
            return min(trend_strength, 100.0)  # Cap at 100%
        except:
            return None
    
    def _validate_signal(self, signal: TradingSignal, data: pd.DataFrame, market_data: Dict) -> bool:
        """Validate signal with additional criteria"""
        
        # Check minimum risk-reward ratio
        if signal.risk_reward_ratio and signal.risk_reward_ratio < 2.0:
            return False
        
        # Check if price is near support/resistance
        if signal.support_resistance_levels:
            current_price = signal.price
            nearest_support = signal.support_resistance_levels.get('nearest_support')
            nearest_resistance = signal.support_resistance_levels.get('nearest_resistance')
            
            if signal.signal_type == 'buy' and nearest_support:
                # Buy signal should be near support
                if abs(current_price - nearest_support) / current_price > 0.05:  # 5% threshold
                    return False
            
            elif signal.signal_type == 'sell' and nearest_resistance:
                # Sell signal should be near resistance
                if abs(current_price - nearest_resistance) / current_price > 0.05:  # 5% threshold
                    return False
        
        # Check market regime compatibility
        if signal.market_regime:
            if signal.market_regime == 'ranging_stable' and signal.pattern in ['hammer', 'shooting_star']:
                # Reversal patterns work well in ranging markets
                pass
            elif signal.market_regime == 'trending_stable' and signal.pattern in ['engulfing', 'morning_star', 'evening_star']:
                # Trend continuation patterns work well in trending markets
                pass
            else:
                # Additional validation for other market regimes
                pass
        
        return True
    
    def _update_performance_metrics(self):
        """Update signal performance metrics"""
        self.signal_performance['total_signals'] = len(self.signal_history)
        
        # Calculate accuracy based on historical signals
        if self.signal_performance['total_signals'] > 0:
            # This is a simplified calculation - in production you'd track actual trade outcomes
            self.signal_performance['accuracy'] = (
                self.signal_performance['successful_signals'] / 
                self.signal_performance['total_signals']
            )
    
    def get_signal_summary(self) -> Dict:
        """Get summary of generated signals"""
        if not self.signal_history:
            return {"message": "No signals generated yet"}
        
        # Group signals by type
        buy_signals = [s for s in self.signal_history if s.signal_type == 'buy']
        sell_signals = [s for s in self.signal_history if s.signal_type == 'sell']
        
        # Calculate average confidence and strength
        avg_buy_confidence = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
        avg_sell_confidence = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
        
        return {
            "total_signals": len(self.signal_history),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "avg_buy_confidence": round(avg_buy_confidence, 3),
            "avg_sell_confidence": round(avg_sell_confidence, 3),
            "performance": self.signal_performance,
            "latest_signals": [
                {
                    "symbol": s.symbol,
                    "type": s.signal_type,
                    "pattern": s.pattern,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in list(self.signal_history)[-5:]  # Last 5 signals
            ]
        }
    
    def get_pattern_analysis(self) -> Dict:
        """Get analysis of detected patterns"""
        if not self.pattern_history:
            return {"message": "No patterns detected yet"}
        
        # Group patterns by type
        pattern_counts = {}
        pattern_confidence = {}
        
        for pattern in self.pattern_history:
            if pattern.pattern not in pattern_counts:
                pattern_counts[pattern.pattern] = 0
                pattern_confidence[pattern.pattern] = []
            
            pattern_counts[pattern.pattern] += 1
            pattern_confidence[pattern.pattern].append(pattern.ml_confidence)
        
        # Calculate average confidence for each pattern
        pattern_avg_confidence = {}
        for pattern, confidences in pattern_confidence.items():
            pattern_avg_confidence[pattern] = round(np.mean(confidences), 3)
        
        return {
            "total_patterns": len(self.pattern_history),
            "pattern_counts": pattern_counts,
            "pattern_confidence": pattern_avg_confidence,
            "market_regimes": self._get_market_regime_distribution()
        }
    
    def _get_market_regime_distribution(self) -> Dict:
        """Get distribution of market regimes"""
        regime_counts = {}
        for pattern in self.pattern_history:
            if pattern.market_regime:
                regime_counts[pattern.market_regime] = regime_counts.get(pattern.market_regime, 0) + 1
        
        return regime_counts

