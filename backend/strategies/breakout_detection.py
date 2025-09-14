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
        
        # Breakout levels
        df['breakout_high'] = df['resistance'] * (1 + self.parameters['breakout_threshold'])
        df['breakout_low'] = df['support'] * (1 - self.parameters['breakout_threshold'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
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
        
        # False breakout detection
        df['false_breakout_up'] = (
            (df['close'] > df['resistance']) & 
            (df['close'].shift(1) <= df['resistance'].shift(1)) &
            (df['close'] < df['resistance'].shift(1))
        )
        
        df['false_breakout_down'] = (
            (df['close'] < df['support']) & 
            (df['close'].shift(1) >= df['support'].shift(1)) &
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
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate breakout signals based on support/resistance breaks with volume confirmation.
        """
        if len(data) < 50:
            return []
        
        signals = []
        current_price = data['close'].iloc[-1]
        current_time = datetime.now()
        
        # Get latest values
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
            
            # Momentum confirmation
            if momentum > 0:
                confidence += 0.1
            
            # Range expansion
            if range_expansion:
                confidence += 0.1
            
            # Breakout strength
            if breakout_strength > 0.01:  # 1% above resistance
                confidence += 0.1
            
            # Volume trend
            if volume_ratio > prev_volume_ratio:
                confidence += 0.1
            
            signal_type = SignalType.BUY
        
        # Sell Signal (Bearish Breakout)
        elif breakout_down and sufficient_consolidation:
            confidence = 0.5
            
            # Volume confirmation
            if volume_surge:
                confidence += 0.2
            
            # Momentum confirmation
            if momentum < 0:
                confidence += 0.1
            
            # Range expansion
            if range_expansion:
                confidence += 0.1
            
            # Breakout strength
            if abs(breakout_strength) > 0.01:  # 1% below support
                confidence += 0.1
            
            # Volume trend
            if volume_ratio > prev_volume_ratio:
                confidence += 0.1
            
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
                'sufficient_consolidation': sufficient_consolidation
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
