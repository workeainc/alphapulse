import pandas as pd
import numpy as np
import ta
from typing import Dict, List
from datetime import datetime
from .base_strategy import BaseStrategy, Signal, SignalType, MarketRegime


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using RSI, Bollinger Bands, and support/resistance levels.
    Best suited for ranging/consolidating markets.
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2,
            'support_resistance_period': 20,
            'min_bounce_percentage': 0.5,
            'volume_confirmation': True,
            'atr_period': 14
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Mean Reversion", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for mean reversion strategy.
        """
        df = data.copy()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(
            df['close'], window=self.parameters['rsi_period']
        )
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df['close'],
            window=self.parameters['bb_period'],
            window_dev=self.parameters['bb_std']
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'],
            window=self.parameters['atr_period']
        )
        
        # Support and Resistance levels
        df['support'] = df['low'].rolling(
            window=self.parameters['support_resistance_period']
        ).min()
        df['resistance'] = df['high'].rolling(
            window=self.parameters['support_resistance_period']
        ).max()
        
        # Price relative to support/resistance
        df['price_to_support'] = (df['close'] - df['support']) / df['close']
        df['price_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(
            df['high'], df['low'], df['close']
        )
        
        # Mean reversion indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['price_deviation'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate mean reversion signals based on RSI, Bollinger Bands, and support/resistance.
        """
        if len(data) < 50:
            return []
        
        signals = []
        current_price = data['close'].iloc[-1]
        current_time = datetime.now()
        
        # Get latest values
        rsi = data['rsi'].iloc[-1]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_middle = data['bb_middle'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        bb_position = data['bb_position'].iloc[-1]
        bb_width = data['bb_width'].iloc[-1]
        support = data['support'].iloc[-1]
        resistance = data['resistance'].iloc[-1]
        price_to_support = data['price_to_support'].iloc[-1]
        price_to_resistance = data['price_to_resistance'].iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        stoch_k = data['stoch_k'].iloc[-1]
        stoch_d = data['stoch_d'].iloc[-1]
        williams_r = data['williams_r'].iloc[-1]
        price_deviation = data['price_deviation'].iloc[-1]
        
        # Previous values for crossover detection
        if len(data) > 1:
            prev_rsi = data['rsi'].iloc[-2]
            prev_stoch_k = data['stoch_k'].iloc[-2]
            prev_stoch_d = data['stoch_d'].iloc[-2]
        else:
            return []
        
        # Calculate signal confidence
        confidence = 0.0
        signal_type = SignalType.HOLD
        
        # Buy Signal Conditions (oversold conditions)
        buy_conditions = []
        
        # RSI oversold
        if rsi < self.parameters['rsi_oversold']:
            buy_conditions.append(('rsi_oversold', 0.3))
        
        # Bollinger Band oversold
        if current_price < bb_lower:
            buy_conditions.append(('bb_oversold', 0.2))
        
        # Stochastic oversold
        if stoch_k < 20 and stoch_d < 20:
            buy_conditions.append(('stoch_oversold', 0.2))
        
        # Williams %R oversold
        if williams_r < -80:
            buy_conditions.append(('williams_oversold', 0.1))
        
        # Support level bounce
        if price_to_support < 0.01:  # Within 1% of support
            buy_conditions.append(('support_bounce', 0.2))
        
        # Price below SMA (mean reversion opportunity)
        if price_deviation < -0.02:  # 2% below SMA
            buy_conditions.append(('price_below_sma', 0.1))
        
        # Sell Signal Conditions (overbought conditions)
        sell_conditions = []
        
        # RSI overbought
        if rsi > self.parameters['rsi_overbought']:
            sell_conditions.append(('rsi_overbought', 0.3))
        
        # Bollinger Band overbought
        if current_price > bb_upper:
            sell_conditions.append(('bb_overbought', 0.2))
        
        # Stochastic overbought
        if stoch_k > 80 and stoch_d > 80:
            sell_conditions.append(('stoch_overbought', 0.2))
        
        # Williams %R overbought
        if williams_r > -20:
            sell_conditions.append(('williams_overbought', 0.1))
        
        # Resistance level rejection
        if price_to_resistance < 0.01:  # Within 1% of resistance
            sell_conditions.append(('resistance_rejection', 0.2))
        
        # Price above SMA (mean reversion opportunity)
        if price_deviation > 0.02:  # 2% above SMA
            sell_conditions.append(('price_above_sma', 0.1))
        
        # Calculate confidence for buy signals
        if len(buy_conditions) >= 2:  # Require at least 2 conditions
            confidence = sum(weight for _, weight in buy_conditions)
            
            # Volume confirmation
            if volume_ratio > 1.2:
                confidence += 0.1
            
            # Bollinger Band squeeze (low volatility)
            if bb_width < 0.05:
                confidence += 0.1
            
            signal_type = SignalType.BUY
        
        # Calculate confidence for sell signals
        elif len(sell_conditions) >= 2:  # Require at least 2 conditions
            confidence = sum(weight for _, weight in sell_conditions)
            
            # Volume confirmation
            if volume_ratio > 1.2:
                confidence += 0.1
            
            # Bollinger Band squeeze (low volatility)
            if bb_width < 0.05:
                confidence += 0.1
            
            signal_type = SignalType.SELL
        
        # Only generate signal if confidence is high enough
        if confidence >= 0.5 and signal_type != SignalType.HOLD:
            metadata = {
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_position': bb_position,
                'bb_width': bb_width,
                'support': support,
                'resistance': resistance,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'williams_r': williams_r,
                'price_deviation': price_deviation,
                'volume_ratio': volume_ratio,
                'buy_conditions': [cond for cond, _ in buy_conditions] if signal_type == SignalType.BUY else [],
                'sell_conditions': [cond for cond, _ in sell_conditions] if signal_type == SignalType.SELL else []
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
        Validate mean reversion signals based on market regime.
        """
        # Base validation
        if not super().validate_signal(signal, market_regime):
            return False
        
        # Mean reversion strategy works best in ranging markets
        if market_regime == MarketRegime.TRENDING:
            # Require higher confidence in trending markets
            if signal.confidence < 0.7:
                return False
        
        # Check for extreme conditions
        if 'rsi' in signal.metadata:
            rsi = signal.metadata['rsi']
            if signal.signal_type == SignalType.BUY and rsi > 40:
                return False
            if signal.signal_type == SignalType.SELL and rsi < 60:
                return False
        
        # Check Bollinger Band position
        if 'bb_position' in signal.metadata:
            bb_pos = signal.metadata['bb_position']
            if signal.signal_type == SignalType.BUY and bb_pos > 0.3:
                return False
            if signal.signal_type == SignalType.SELL and bb_pos < 0.7:
                return False
        
        return True
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Override to add mean reversion specific regime detection.
        """
        base_regime = super().get_market_regime(data)
        
        # Additional mean reversion specific logic
        if len(data) >= 20:
            bb_width = data['bb_width'].iloc[-1] if 'bb_width' in data.columns else 0
            price_deviation = data['price_deviation'].iloc[-1] if 'price_deviation' in data.columns else 0
            
            # Low volatility and price near mean suggests ranging market
            if bb_width < 0.05 and abs(price_deviation) < 0.03:
                return MarketRegime.RANGING
        
        return base_regime
