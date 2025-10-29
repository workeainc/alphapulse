import pandas as pd
import numpy as np
import ta
from typing import Dict, List
from datetime import datetime
from .base_strategy import BaseStrategy, Signal, SignalType, MarketRegime


class BreakoutDetectionStrategy(BaseStrategy):
    """
    Breakout detection strategy using support/resistance breaks, volume confirmation,
    and momentum indicators. Best suited for volatile markets with clear breakouts.
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'support_resistance_period': 20,
            'breakout_threshold': 0.02,  # 2% breakout
            'volume_multiplier': 1.5,
            'momentum_period': 14,
            'atr_period': 14,
            'min_breakout_duration': 2,  # candles
            'false_breakout_filter': True,
            'consolidation_period': 10
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Breakout Detection", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for breakout detection strategy.
        """
        df = data.copy()
        
        # Support and Resistance levels
        df['support'] = df['low_price'].rolling(
            window=self.parameters['support_resistance_period']
        ).min()
        df['resistance'] = df['high_price'].rolling(
            window=self.parameters['support_resistance_period']
        ).max()
        
        # Breakout levels using standard formula: Close > Resistance × 1.02 (2% buffer)
        df['breakout_high'] = df['resistance'] * (1 + self.parameters['breakout_threshold'])
        df['breakout_low'] = df['support'] * (1 - self.parameters['breakout_threshold'])
        
        # Volume indicators with standard formulas
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        # Volume confirmation: Volume > 1.5× SMA(Volume, 20)
        df['volume_surge'] = df['volume'] > (df['volume_sma'] * self.parameters['volume_multiplier'])
        
        # Momentum indicators
        df['momentum'] = ta.momentum.roc(
            df['close'], window=self.parameters['momentum_period']
        )
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'],
            window=self.parameters['atr_period']
        )
        
        # Price action indicators
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_range_sma'] = df['price_range'].rolling(window=20).mean()
        df['range_expansion'] = df['price_range'] > (df['price_range_sma'] * 1.5)
        
        # Consolidation detection
        df['consolidation'] = df['price_range'] < (df['price_range_sma'] * 0.7)
        
        # Breakout confirmation
        df['above_resistance'] = df['close'] > df['resistance']
        df['below_support'] = df['close'] < df['support']
        
        # Momentum confirmation: 3 consecutive closes in direction
        df['momentum_confirmation'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['close'].shift(1) > df['close'].shift(2)) &
            (df['close'].shift(2) > df['close'].shift(3))
        )
        
        # False breakout filter: Re-entry within 5 periods
        df['false_breakout_up'] = (
            (df['close'] > df['resistance']) & 
            (df['close'].shift(1) <= df['resistance'].shift(1)) &
            (df['close'].shift(2) <= df['resistance'].shift(2)) &
            (df['close'].shift(3) <= df['resistance'].shift(3)) &
            (df['close'].shift(4) <= df['resistance'].shift(4)) &
            (df['close'].shift(5) <= df['resistance'].shift(5)) &
            (df['close'] < df['resistance'].shift(1))
        )
        
        df['false_breakout_down'] = (
            (df['close'] < df['support']) & 
            (df['close'].shift(1) >= df['support'].shift(1)) &
            (df['close'].shift(2) >= df['support'].shift(2)) &
            (df['close'].shift(3) >= df['support'].shift(3)) &
            (df['close'].shift(4) >= df['support'].shift(4)) &
            (df['close'].shift(5) >= df['support'].shift(5)) &
            (df['close'] > df['support'].shift(1))
        )
        
        # Consolidation period
        df['consolidation_count'] = df['consolidation'].rolling(
            window=self.parameters['consolidation_period']
        ).sum()
        
        # Breakout strength
        df['breakout_strength'] = np.where(
            df['above_resistance'],
            (df['close'] - df['resistance']) / df['resistance'],
            np.where(
                df['below_support'],
                (df['support'] - df['close']) / df['support'],
                0
            )
        )
        
        return df
    
    def detect_reversal_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect reversal patterns using divergence analysis"""
        df = data.copy()
        
        # Calculate RSI for divergence analysis
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Calculate MACD for momentum divergence
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price vs RSI divergence
        df['price_high'] = df['high'].rolling(window=5).max()
        df['price_low'] = df['low'].rolling(window=5).min()
        df['rsi_high'] = df['rsi'].rolling(window=5).max()
        df['rsi_low'] = df['rsi'].rolling(window=5).min()
        
        # Bullish divergence: Price new low, RSI higher low
        df['bullish_divergence'] = (
            (df['price_low'] < df['price_low'].shift(5)) &
            (df['rsi_low'] > df['rsi_low'].shift(5))
        )
        
        # Bearish divergence: Price new high, RSI lower high
        df['bearish_divergence'] = (
            (df['price_high'] > df['price_high'].shift(5)) &
            (df['rsi_high'] < df['rsi_high'].shift(5))
        )
        
        # Volume divergence: Price up, Volume down
        df['volume_divergence_bullish'] = (
            (df['close'] > df['close'].shift(5)) &
            (df['volume'] < df['volume'].shift(5))
        )
        
        df['volume_divergence_bearish'] = (
            (df['close'] < df['close'].shift(5)) &
            (df['volume'] < df['volume'].shift(5))
        )
        
        # Momentum divergence: Price up, MACD down
        df['momentum_divergence_bullish'] = (
            (df['close'] > df['close'].shift(5)) &
            (df['macd_histogram'] < df['macd_histogram'].shift(5))
        )
        
        df['momentum_divergence_bearish'] = (
            (df['close'] < df['close'].shift(5)) &
            (df['macd_histogram'] > df['macd_histogram'].shift(5))
        )
        
        # Calculate reversal strength: (Divergence Magnitude × Volume) / Time Periods
        df['reversal_strength_bullish'] = np.where(
            df['bullish_divergence'],
            (abs(df['price_low'] - df['price_low'].shift(5)) * df['volume']) / 5,
            0
        )
        
        df['reversal_strength_bearish'] = np.where(
            df['bearish_divergence'],
            (abs(df['price_high'] - df['price_high'].shift(5)) * df['volume']) / 5,
            0
        )
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate breakout signals based on support/resistance breaks with volume confirmation.
        """
        if len(data) < 50:
            return []
        
        signals = []
        current_price = data['close'].iloc[-1]
        current_time = datetime.now()
        
        # Get latest values including reversal patterns
        support = data['support'].iloc[-1]
        resistance = data['resistance'].iloc[-1]
        breakout_high = data['breakout_high'].iloc[-1]
        breakout_low = data['breakout_low'].iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        volume_surge = data['volume_surge'].iloc[-1]
        momentum = data['momentum'].iloc[-1]
        atr = data['atr'].iloc[-1]
        range_expansion = data['range_expansion'].iloc[-1]
        consolidation_count = data['consolidation_count'].iloc[-1]
        breakout_strength = data['breakout_strength'].iloc[-1]
        
        # Reversal pattern detection
        bullish_divergence = data['bullish_divergence'].iloc[-1] if 'bullish_divergence' in data.columns else False
        bearish_divergence = data['bearish_divergence'].iloc[-1] if 'bearish_divergence' in data.columns else False
        volume_divergence_bullish = data['volume_divergence_bullish'].iloc[-1] if 'volume_divergence_bullish' in data.columns else False
        volume_divergence_bearish = data['volume_divergence_bearish'].iloc[-1] if 'volume_divergence_bearish' in data.columns else False
        momentum_divergence_bullish = data['momentum_divergence_bullish'].iloc[-1] if 'momentum_divergence_bullish' in data.columns else False
        momentum_divergence_bearish = data['momentum_divergence_bearish'].iloc[-1] if 'momentum_divergence_bearish' in data.columns else False
        reversal_strength_bullish = data['reversal_strength_bullish'].iloc[-1] if 'reversal_strength_bullish' in data.columns else 0
        reversal_strength_bearish = data['reversal_strength_bearish'].iloc[-1] if 'reversal_strength_bearish' in data.columns else 0
        momentum_confirmation = data['momentum_confirmation'].iloc[-1] if 'momentum_confirmation' in data.columns else False
        
        # Previous values for breakout detection
        if len(data) > 1:
            prev_close = data['close'].iloc[-2]
            prev_resistance = data['resistance'].iloc[-2]
            prev_support = data['support'].iloc[-2]
            prev_volume_ratio = data['volume_ratio'].iloc[-2]
        else:
            return []
        
        # Calculate signal confidence
        confidence = 0.0
        signal_type = SignalType.HOLD
        
        # Breakout conditions
        breakout_up = current_price > resistance and prev_close <= prev_resistance
        breakout_down = current_price < support and prev_close >= prev_support
        
        # Consolidation requirement
        sufficient_consolidation = consolidation_count >= (self.parameters['consolidation_period'] * 0.7)
        
        # Buy Signal (Bullish Breakout)
        if breakout_up and sufficient_consolidation:
            confidence = 0.5
            
            # Volume confirmation
            if volume_surge:
                confidence += 0.2
            
            # Momentum confirmation: 3 consecutive closes in direction
            if momentum_confirmation:
                confidence += 0.15
            
            signal_type = SignalType.BUY
        
        # Sell Signal (Bearish Breakout)
        elif breakout_down and sufficient_consolidation:
            confidence = 0.5
            
            # Volume confirmation: Volume > 1.5× SMA(Volume, 20)
            if volume_surge:
                confidence += 0.2
            
            # Momentum confirmation
            if momentum < 0:
                confidence += 0.1
            
            # Range expansion
            if range_expansion:
                confidence += 0.1
            
            # Breakout strength: Close > Resistance × 1.02
            if abs(breakout_strength) > 0.01:  # 1% below support
                confidence += 0.1
            
            # Volume trend
            if volume_ratio > prev_volume_ratio:
                confidence += 0.1
            
            # Momentum confirmation: 3 consecutive closes in direction
            if momentum_confirmation:
                confidence += 0.15
            
            signal_type = SignalType.SELL
        
        # Reversal Signals
        elif bullish_divergence and (volume_divergence_bullish or momentum_divergence_bullish):
            confidence = 0.6
            
            # Reversal strength: (Divergence Magnitude × Volume) / Time Periods
            if reversal_strength_bullish > 0:
                confidence += min(reversal_strength_bullish / 1000000, 0.3)
            
            signal_type = SignalType.BUY
        
        elif bearish_divergence and (volume_divergence_bearish or momentum_divergence_bearish):
            confidence = 0.6
            
            # Reversal strength: (Divergence Magnitude × Volume) / Time Periods
            if reversal_strength_bearish > 0:
                confidence += min(reversal_strength_bearish / 1000000, 0.3)
            
            signal_type = SignalType.SELL
        
        # Only generate signal if confidence is high enough
        if confidence >= 0.6 and signal_type != SignalType.HOLD:
            metadata = {
                'support': support,
                'resistance': resistance,
                'breakout_high': breakout_high,
                'breakout_low': breakout_low,
                'volume_ratio': volume_ratio,
                'volume_surge': volume_surge,
                'momentum': momentum,
                'atr': atr,
                'range_expansion': range_expansion,
                'consolidation_count': consolidation_count,
                'breakout_strength': breakout_strength,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'sufficient_consolidation': sufficient_consolidation,
                'momentum_confirmation': momentum_confirmation,
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'volume_divergence_bullish': volume_divergence_bullish,
                'volume_divergence_bearish': volume_divergence_bearish,
                'momentum_divergence_bullish': momentum_divergence_bullish,
                'momentum_divergence_bearish': momentum_divergence_bearish,
                'reversal_strength_bullish': reversal_strength_bullish,
                'reversal_strength_bearish': reversal_strength_bearish
            }
            
            signal = Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                timestamp=current_time,
                confidence=confidence,
                strategy_name=self.name,
                metadata=metadata
            )
            
            signals.append(signal)
        
        return signals
    
    def validate_signal(self, signal: Signal, market_regime: MarketRegime) -> bool:
        """
        Validate breakout signals based on market regime.
        """
        # Base validation
        if not super().validate_signal(signal, market_regime):
            return False
        
        # Breakout strategy works best in volatile markets
        if market_regime == MarketRegime.RANGING:
            # Require higher confidence in ranging markets
            if signal.confidence < 0.7:
                return False
        
        # Check for sufficient consolidation
        if 'consolidation_count' in signal.metadata:
            if signal.metadata['consolidation_count'] < 7:  # At least 7 periods of consolidation
                return False
        
        # Check for volume confirmation
        if 'volume_surge' in signal.metadata:
            if not signal.metadata['volume_surge']:
                return False
        
        # Check for range expansion
        if 'range_expansion' in signal.metadata:
            if not signal.metadata['range_expansion']:
                return False
        
        return True
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Override to add breakout-specific regime detection.
        """
        base_regime = super().get_market_regime(data)
        
        # Additional breakout-specific logic
        if len(data) >= 20:
            range_expansion = data['range_expansion'].iloc[-1] if 'range_expansion' in data.columns else False
            volume_surge = data['volume_surge'].iloc[-1] if 'volume_surge' in data.columns else False
            
            # High volatility with range expansion suggests volatile market
            if range_expansion and volume_surge:
                return MarketRegime.VOLATILE
        
        return base_regime
