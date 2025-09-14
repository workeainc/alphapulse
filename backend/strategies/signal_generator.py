"""
Signal Generation Engine for AlphaPulse Trading Bot
Phase 1 Implementation - Combines patterns and indicators into trading signals
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from .pattern_detector import CandlestickPatternDetector
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: List[str]
    patterns_detected: List[str]
    indicators_confirming: List[str]
    risk_reward_ratio: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class SignalGenerator:
    """
    Generates trading signals by combining candlestick patterns and technical indicators
    Implements multi-factor analysis for signal quality assessment
    """
    
    def __init__(self, pattern_detector: CandlestickPatternDetector, 
                 technical_analyzer: TechnicalIndicators):
        self.pattern_detector = pattern_detector
        self.technical_analyzer = technical_analyzer
        self.min_confidence = 0.7
        self.min_risk_reward = 2.0
        
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate trading signals from patterns and indicators
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            symbol: Trading symbol
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            # 1. Detect candlestick patterns
            patterns = self.pattern_detector.detect_patterns_from_dataframe(df)
            
            # 2. Calculate technical indicators
            indicators = self.technical_analyzer.calculate_all_indicators(df)
            
            # 3. Generate signals for each pattern
            for pattern in patterns:
                if pattern.confidence >= self.min_confidence:
                    signal = self._create_signal(pattern, indicators, df, symbol)
                    if signal:
                        signals.append(signal)
            
            # 4. Sort signals by confidence and strength
            signals.sort(key=lambda x: (x.confidence, x.strength.value), reverse=True)
            
            logger.info(f"🎯 Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating signals for {symbol}: {e}")
            return []
    
    def _create_signal(self, pattern, indicators: Dict, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Create a trading signal from a pattern and indicators"""
        try:
            # Determine signal type based on pattern
            signal_type = self._determine_signal_type(pattern)
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(pattern, indicators)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(pattern, indicators)
            
            # Get current price data
            current_price = df['close'].iloc[-1]
            
            # Calculate entry, stop loss, and take profit
            entry_price, stop_loss, take_profit = self._calculate_price_levels(
                pattern, indicators, df, signal_type
            )
            
            # Calculate position size (placeholder for now)
            position_size = 1.0  # Will be calculated by risk manager
            
            # Generate reasoning
            reasoning = self._generate_reasoning(pattern, indicators, signal_type)
            
            # Calculate risk-reward ratio
            risk_reward = self._calculate_risk_reward(entry_price, stop_loss, take_profit)
            
            # Only create signal if risk-reward meets minimum
            if risk_reward < self.min_risk_reward:
                return None
            
            signal = Signal(
                timestamp=df.index[-1],
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                patterns_detected=[pattern.pattern_name],
                indicators_confirming=self._get_confirming_indicators(indicators, signal_type),
                risk_reward_ratio=risk_reward
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error creating signal: {e}")
            return None
    
    def _determine_signal_type(self, pattern) -> SignalType:
        """Determine if pattern suggests buy or sell"""
        bullish_patterns = [
            'Hammer', 'Inverted Hammer', 'Morning Star', 'Three White Soldiers',
            'Bullish Engulfing', 'Piercing Line', 'Three Inside Up'
        ]
        
        bearish_patterns = [
            'Shooting Star', 'Evening Star', 'Three Black Crows',
            'Bearish Engulfing', 'Dark Cloud Cover', 'Three Inside Down'
        ]
        
        if pattern.pattern_name in bullish_patterns:
            return SignalType.BUY
        elif pattern.pattern_name in bearish_patterns:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_signal_strength(self, pattern, indicators: Dict) -> SignalStrength:
        """Calculate signal strength based on pattern confidence and indicator confirmation"""
        base_strength = pattern.confidence
        
        # Add indicator confirmation bonus
        confirmation_bonus = self._calculate_indicator_confirmation(indicators, pattern)
        total_strength = base_strength + confirmation_bonus
        
        if total_strength >= 0.9:
            return SignalStrength.VERY_STRONG
        elif total_strength >= 0.8:
            return SignalStrength.STRONG
        elif total_strength >= 0.7:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    def _calculate_indicator_confirmation(self, indicators: Dict, pattern) -> float:
        """Calculate how much technical indicators confirm the pattern"""
        confirmation_score = 0.0
        max_score = 0.0
        
        # RSI confirmation
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if pattern.pattern_name in ['Hammer', 'Morning Star', 'Three White Soldiers']:
                if rsi < 30:  # Oversold
                    confirmation_score += 0.2
            elif pattern.pattern_name in ['Shooting Star', 'Evening Star', 'Three Black Crows']:
                if rsi > 70:  # Overbought
                    confirmation_score += 0.2
            max_score += 0.2
        
        # MACD confirmation
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            
            if pattern.pattern_name in ['Hammer', 'Morning Star', 'Three White Soldiers']:
                if macd > macd_signal:  # Bullish crossover
                    confirmation_score += 0.2
            elif pattern.pattern_name in ['Shooting Star', 'Evening Star', 'Three Black Crows']:
                if macd < macd_signal:  # Bearish crossover
                    confirmation_score += 0.2
            max_score += 0.2
        
        # Moving average confirmation
        if 'ema_20' in indicators and 'ema_50' in indicators:
            ema_20 = indicators['ema_20'].iloc[-1]
            ema_50 = indicators['ema_50'].iloc[-1]
            
            if pattern.pattern_name in ['Hammer', 'Morning Star', 'Three White Soldiers']:
                if ema_20 > ema_50:  # Uptrend
                    confirmation_score += 0.2
            elif pattern.pattern_name in ['Shooting Star', 'Evening Star', 'Three Black Crows']:
                if ema_20 < ema_50:  # Downtrend
                    confirmation_score += 0.2
            max_score += 0.2
        
        # Volume confirmation
        if 'volume' in indicators:
            current_volume = indicators['volume'].iloc[-1]
            avg_volume = indicators['volume'].rolling(20).mean().iloc[-1]
            
            if current_volume > avg_volume * 1.5:  # High volume
                confirmation_score += 0.2
            max_score += 0.2
        
        return confirmation_score / max_score if max_score > 0 else 0.0
    
    def _calculate_confidence(self, pattern, indicators: Dict) -> float:
        """Calculate overall confidence score"""
        pattern_confidence = pattern.confidence
        indicator_confirmation = self._calculate_indicator_confirmation(indicators, pattern)
        
        # Weight: 70% pattern, 30% indicators
        confidence = (pattern_confidence * 0.7) + (indicator_confirmation * 0.3)
        return min(confidence, 1.0)
    
    def _calculate_price_levels(self, pattern, indicators: Dict, df: pd.DataFrame, 
                               signal_type: SignalType) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = df['close'].iloc[-1]
        atr = indicators.get('atr', pd.Series([current_price * 0.02])).iloc[-1]  # Default 2% ATR
        
        if signal_type == SignalType.BUY:
            entry_price = current_price
            stop_loss = entry_price - (atr * 1.5)
            take_profit = entry_price + (atr * 3.0)  # 2:1 risk-reward
        else:  # SELL
            entry_price = current_price
            stop_loss = entry_price + (atr * 1.5)
            take_profit = entry_price - (atr * 3.0)  # 2:1 risk-reward
        
        return entry_price, stop_loss, take_profit
    
    def _generate_reasoning(self, pattern, indicators: Dict, signal_type: SignalType) -> List[str]:
        """Generate human-readable reasoning for the signal"""
        reasoning = []
        
        # Pattern reasoning
        reasoning.append(f"Detected {pattern.pattern_name} pattern with {pattern.confidence:.1%} confidence")
        
        # Technical indicator reasoning
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if rsi < 30:
                reasoning.append("RSI indicates oversold conditions")
            elif rsi > 70:
                reasoning.append("RSI indicates overbought conditions")
        
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            if macd > macd_signal:
                reasoning.append("MACD shows bullish momentum")
            else:
                reasoning.append("MACD shows bearish momentum")
        
        # Volume reasoning
        if 'volume' in indicators:
            current_volume = indicators['volume'].iloc[-1]
            avg_volume = indicators['volume'].rolling(20).mean().iloc[-1]
            if current_volume > avg_volume * 1.5:
                reasoning.append("High volume confirms pattern")
        
        return reasoning
    
    def _get_confirming_indicators(self, indicators: Dict, signal_type: SignalType) -> List[str]:
        """Get list of indicators that confirm the signal"""
        confirming = []
        
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if signal_type == SignalType.BUY and rsi < 30:
                confirming.append("RSI")
            elif signal_type == SignalType.SELL and rsi > 70:
                confirming.append("RSI")
        
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            if signal_type == SignalType.BUY and macd > macd_signal:
                confirming.append("MACD")
            elif signal_type == SignalType.SELL and macd < macd_signal:
                confirming.append("MACD")
        
        return confirming
    
    def _calculate_risk_reward(self, entry: float, stop_loss: float, take_profit: float) -> float:
        """Calculate risk-reward ratio"""
        if entry == stop_loss:
            return 0.0
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        return reward / risk if risk > 0 else 0.0
    
    def filter_signals(self, signals: List[Signal], min_confidence: float = None, 
                       min_strength: SignalStrength = None) -> List[Signal]:
        """Filter signals based on criteria"""
        filtered = signals
        
        if min_confidence is not None:
            filtered = [s for s in filtered if s.confidence >= min_confidence]
        
        if min_strength is not None:
            filtered = [s for s in filtered if s.strength.value >= min_strength.value]
        
        return filtered
    
    def get_signal_summary(self, signals: List[Signal]) -> Dict:
        """Get summary statistics for signals"""
        if not signals:
            return {"total": 0, "buy": 0, "sell": 0, "avg_confidence": 0.0}
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        return {
            "total": len(signals),
            "buy": len(buy_signals),
            "sell": len(sell_signals),
            "avg_confidence": sum(s.confidence for s in signals) / len(signals),
            "avg_risk_reward": sum(s.risk_reward_ratio for s in signals) / len(signals)
        }

# Example usage
def example_usage():
    """Example of how to use the SignalGenerator"""
    from .pattern_detector import CandlestickPatternDetector
    from .indicators import TechnicalIndicators
    
    # Initialize components
    pattern_detector = CandlestickPatternDetector()
    technical_analyzer = TechnicalIndicators()
    
    # Create signal generator
    signal_generator = SignalGenerator(pattern_detector, technical_analyzer)
    
    # Generate signals (you would pass actual data here)
    # signals = signal_generator.generate_signals(df, "BTCUSDT")
    
    print("🎯 Signal Generator initialized successfully!")

if __name__ == "__main__":
    example_usage()
