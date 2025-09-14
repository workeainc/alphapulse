import pandas as pd
import numpy as np
import ta
from typing import Dict, List
from datetime import datetime
from .base_strategy import BaseStrategy, Signal, SignalType, MarketRegime


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-following strategy using EMA crossovers, MACD, and trend strength indicators.
    Best suited for trending markets.
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'ema_fast': 9,
            'ema_slow': 21,
            'ema_trend': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'min_trend_strength': 0.6,
            'volume_threshold': 1.2
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Trend Following", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for trend-following strategy.
        """
        df = data.copy()
        
        # EMAs
        df[f'ema_{self.parameters["ema_fast"]}'] = ta.trend.ema_indicator(
            df['close'], window=self.parameters['ema_fast']
        )
        df[f'ema_{self.parameters["ema_slow"]}'] = ta.trend.ema_indicator(
            df['close'], window=self.parameters['ema_slow']
        )
        df[f'ema_{self.parameters["ema_trend"]}'] = ta.trend.ema_indicator(
            df['close'], window=self.parameters['ema_trend']
        )
        
        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_fast=self.parameters['macd_fast'],
            window_slow=self.parameters['macd_slow'],
            window_sign=self.parameters['macd_signal']
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(
            df['close'], window=self.parameters['rsi_period']
        )
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'],
            window=self.parameters['atr_period']
        )
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Trend strength indicators
        df['price_sma_20'] = df['close'].rolling(window=20).mean()
        df['price_sma_50'] = df['close'].rolling(window=50).mean()
        df['trend_strength'] = abs(df['close'] - df['price_sma_20']) / df['price_sma_20']
        
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close']
        )
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate trend-following signals based on EMA crossovers, MACD, and trend strength.
        """
        if len(data) < 50:
            return []
        
        signals = []
        current_price = data['close'].iloc[-1]
        current_time = datetime.now()
        
        # Get latest values
        ema_fast = data[f'ema_{self.parameters["ema_fast"]}'].iloc[-1]
        ema_slow = data[f'ema_{self.parameters["ema_slow"]}'].iloc[-1]
        ema_trend = data[f'ema_{self.parameters["ema_trend"]}'].iloc[-1]
        macd = data['macd'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        macd_histogram = data['macd_histogram'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        adx = data['adx'].iloc[-1]
        di_plus = data['di_plus'].iloc[-1]
        di_minus = data['di_minus'].iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        trend_strength = data['trend_strength'].iloc[-1]
        
        # Previous values for crossover detection
        if len(data) > 1:
            prev_ema_fast = data[f'ema_{self.parameters["ema_fast"]}'].iloc[-2]
            prev_ema_slow = data[f'ema_{self.parameters["ema_slow"]}'].iloc[-2]
            prev_macd = data['macd'].iloc[-2]
            prev_macd_signal = data['macd_signal'].iloc[-2]
        else:
            return []
        
        # Calculate signal confidence
        confidence = 0.0
        signal_type = SignalType.HOLD
        
        # EMA Crossover Signal
        ema_crossover_buy = (ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow)
        ema_crossover_sell = (ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow)
        
        # MACD Signal
        macd_crossover_buy = (macd > macd_signal and prev_macd <= prev_macd_signal)
        macd_crossover_sell = (macd < macd_signal and prev_macd >= prev_macd_signal)
        
        # Trend alignment
        trend_bullish = current_price > ema_trend
        trend_bearish = current_price < ema_trend
        
        # ADX trend strength
        strong_trend = adx > 25
        
        # Volume confirmation
        volume_confirmed = volume_ratio > self.parameters['volume_threshold']
        
        # RSI filter (avoid overbought/oversold in strong trends)
        rsi_filter_buy = rsi < self.parameters['rsi_overbought']
        rsi_filter_sell = rsi > self.parameters['rsi_oversold']
        
        # Buy Signal Conditions
        if (ema_crossover_buy or macd_crossover_buy) and trend_bullish:
            confidence = 0.6
            
            # Add confidence based on additional confirmations
            if ema_crossover_buy and macd_crossover_buy:
                confidence += 0.2
            
            if strong_trend:
                confidence += 0.1
            
            if volume_confirmed:
                confidence += 0.1
            
            if rsi_filter_buy:
                confidence += 0.1
            
            if di_plus > di_minus:
                confidence += 0.1
            
            signal_type = SignalType.BUY
        
        # Sell Signal Conditions
        elif (ema_crossover_sell or macd_crossover_sell) and trend_bearish:
            confidence = 0.6
            
            # Add confidence based on additional confirmations
            if ema_crossover_sell and macd_crossover_sell:
                confidence += 0.2
            
            if strong_trend:
                confidence += 0.1
            
            if volume_confirmed:
                confidence += 0.1
            
            if rsi_filter_sell:
                confidence += 0.1
            
            if di_minus > di_plus:
                confidence += 0.1
            
            signal_type = SignalType.SELL
        
        # Only generate signal if confidence is high enough
        if confidence >= 0.6 and signal_type != SignalType.HOLD:
            metadata = {
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'ema_trend': ema_trend,
                'macd': macd,
                'macd_signal': macd_signal,
                'rsi': rsi,
                'adx': adx,
                'volume_ratio': volume_ratio,
                'trend_strength': trend_strength,
                'ema_crossover': ema_crossover_buy or ema_crossover_sell,
                'macd_crossover': macd_crossover_buy or macd_crossover_sell
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
        Validate trend-following signals based on market regime.
        """
        # Base validation
        if not super().validate_signal(signal, market_regime):
            return False
        
        # Trend-following strategy works best in trending markets
        if market_regime == MarketRegime.RANGING:
            # Require higher confidence in ranging markets
            if signal.confidence < 0.7:
                return False
        
        # Check for strong trend confirmation
        if 'adx' in signal.metadata:
            if signal.metadata['adx'] < 20:
                return False
        
        # Volume confirmation
        if 'volume_ratio' in signal.metadata:
            if signal.metadata['volume_ratio'] < 1.0:
                return False
        
        return True
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Override to add trend-specific regime detection.
        """
        base_regime = super().get_market_regime(data)
        
        # Additional trend-specific logic
        if len(data) >= 20:
            adx = data['adx'].iloc[-1] if 'adx' in data.columns else 0
            trend_strength = data['trend_strength'].iloc[-1] if 'trend_strength' in data.columns else 0
            
            # Strong trend indicators
            if adx > 25 and trend_strength > 0.02:
                return MarketRegime.TRENDING
        
        return base_regime
