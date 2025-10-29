#!/usr/bin/env python3
"""
Consolidated Feature Engineering Utilities for AlphaPulse
Combines functionality from various feature_*.py files
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Represents a set of engineered features"""
    timestamp: datetime
    symbol: str
    timeframe: str
    features: Dict[str, float]
    metadata: Dict[str, Any]

class FeatureEngineer:
    """Advanced feature engineering for trading signals"""
    
    def __init__(self, lookback_periods: List[int] = [14, 20, 50, 200]):
        self.lookback_periods = lookback_periods
        self.feature_cache = {}
    
    def engineer_features(self, candles: List[Dict]) -> FeatureSet:
        """Engineer comprehensive feature set from candle data"""
        if len(candles) < max(self.lookback_periods):
            raise ValueError(f"Insufficient data: need at least {max(self.lookback_periods)} candles")
        
        df = pd.DataFrame(candles)
        features = {}
        
        # Price-based features
        features.update(self._price_features(df))
        
        # Volume-based features
        features.update(self._volume_features(df))
        
        # Technical indicator features
        features.update(self._technical_features(df))
        
        # Volatility features
        features.update(self._volatility_features(df))
        
        # Momentum features
        features.update(self._momentum_features(df))
        
        # Pattern features
        features.update(self._pattern_features(df))
        
        # Market microstructure features
        features.update(self._microstructure_features(df))
        
        return FeatureSet(
            timestamp=datetime.now(),
            symbol=candles[-1].get('symbol', 'unknown'),
            timeframe=candles[-1].get('timeframe', 'unknown'),
            features=features,
            metadata={'lookback_periods': self.lookback_periods}
        )
    
    def _price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract price-based features"""
        features = {}
        
        # Price levels
        features['close'] = df['close'].iloc[-1]
        features['open'] = df['open'].iloc[-1]
        features['high'] = df['high'].iloc[-1]
        features['low'] = df['low'].iloc[-1]
        
        # Price changes
        for period in self.lookback_periods:
            if len(df) >= period:
                features[f'price_change_{period}'] = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
                features[f'price_volatility_{period}'] = df['close'].pct_change().rolling(period).std().iloc[-1]
        
        # Moving averages
        for period in self.lookback_periods:
            if len(df) >= period:
                features[f'sma_{period}'] = df['close'].rolling(period).mean().iloc[-1]
                features[f'ema_{period}'] = df['close'].ewm(span=period).mean().iloc[-1]
        
        # Price position relative to moving averages
        for period in self.lookback_periods:
            if len(df) >= period:
                sma = df['close'].rolling(period).mean().iloc[-1]
                features[f'price_vs_sma_{period}'] = (df['close'].iloc[-1] - sma) / sma
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-based features"""
        features = {}
        
        # Volume metrics
        features['volume'] = df['volume'].iloc[-1]
        
        for period in self.lookback_periods:
            if len(df) >= period:
                features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean().iloc[-1]
                features[f'volume_ratio_{period}'] = df['volume'].iloc[-1] / df['volume'].rolling(period).mean().iloc[-1]
                features[f'volume_volatility_{period}'] = df['volume'].pct_change().rolling(period).std().iloc[-1]
        
        # Volume-price relationship
        features['volume_price_trend'] = (df['volume'] * df['close']).pct_change().rolling(20).mean().iloc[-1]
        
        return features
    
    def _technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features"""
        features = {}
        
        # RSI
        for period in [14, 20]:
            if len(df) >= period:
                rsi = self._calculate_rsi(df['close'], period)
                features[f'rsi_{period}'] = rsi.iloc[-1]
        
        # MACD
        if len(df) >= 26:
            macd, signal = self._calculate_macd(df['close'])
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = signal.iloc[-1]
            features['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
        
        # Bollinger Bands
        for period in [20, 50]:
            if len(df) >= period:
                bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'], period)
                features[f'bb_upper_{period}'] = bb_upper.iloc[-1]
                features[f'bb_lower_{period}'] = bb_lower.iloc[-1]
                features[f'bb_width_{period}'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / df['close'].rolling(period).mean().iloc[-1]
        
        # ATR
        for period in [14, 20]:
            if len(df) >= period:
                atr = self._calculate_atr(df, period)
                features[f'atr_{period}'] = atr.iloc[-1]
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-based features"""
        features = {}
        
        # Historical volatility
        for period in self.lookback_periods:
            if len(df) >= period:
                returns = df['close'].pct_change()
                features[f'volatility_{period}'] = returns.rolling(period).std().iloc[-1]
                features[f'volatility_annualized_{period}'] = returns.rolling(period).std().iloc[-1] * np.sqrt(252)
        
        # Realized volatility
        for period in [5, 10, 20]:
            if len(df) >= period:
                features[f'realized_vol_{period}'] = np.sqrt((df['close'].pct_change() ** 2).rolling(period).sum().iloc[-1])
        
        # Parkinson volatility
        for period in [20, 50]:
            if len(df) >= period:
                features[f'parkinson_vol_{period}'] = np.sqrt(
                    (1 / (4 * np.log(2))) * 
                    ((np.log(df['high'] / df['low']) ** 2).rolling(period).mean().iloc[-1])
                )
        
        return features
    
    def _momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum-based features"""
        features = {}
        
        # Price momentum
        for period in self.lookback_periods:
            if len(df) >= period:
                features[f'momentum_{period}'] = df['close'].iloc[-1] / df['close'].iloc[-period] - 1
        
        # Rate of change
        for period in [10, 20, 50]:
            if len(df) >= period:
                features[f'roc_{period}'] = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period] * 100
        
        # Stochastic oscillator
        for period in [14, 20]:
            if len(df) >= period:
                stoch_k, stoch_d = self._calculate_stochastic(df, period)
                features[f'stoch_k_{period}'] = stoch_k.iloc[-1]
                features[f'stoch_d_{period}'] = stoch_d.iloc[-1]
        
        return features
    
    def _pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract pattern-based features"""
        features = {}
        
        # Candlestick patterns
        features['doji'] = self._is_doji(df.iloc[-1])
        features['hammer'] = self._is_hammer(df.iloc[-1])
        features['shooting_star'] = self._is_shooting_star(df.iloc[-1])
        features['engulfing'] = self._is_engulfing(df.iloc[-2:])
        
        # Support/resistance levels
        for period in [20, 50]:
            if len(df) >= period:
                support, resistance = self._find_support_resistance(df, period)
                features[f'support_{period}'] = support
                features[f'resistance_{period}'] = resistance
                features[f'price_vs_support_{period}'] = (df['close'].iloc[-1] - support) / support if support > 0 else 0
                features[f'price_vs_resistance_{period}'] = (resistance - df['close'].iloc[-1]) / df['close'].iloc[-1] if resistance > 0 else 0
        
        return features
    
    def _microstructure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}
        
        # Bid-ask spread proxy (using high-low)
        features['spread_proxy'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        
        # Price efficiency
        for period in [20, 50]:
            if len(df) >= period:
                features[f'price_efficiency_{period}'] = self._calculate_price_efficiency(df['close'], period)
        
        # Volume profile
        for period in [20, 50]:
            if len(df) >= period:
                features[f'volume_profile_{period}'] = self._calculate_volume_profile(df, period)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    def _is_doji(self, candle: pd.Series) -> float:
        """Check if candle is a doji"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        return 1.0 if body_size / total_range < 0.1 else 0.0
    
    def _is_hammer(self, candle: pd.Series) -> float:
        """Check if candle is a hammer"""
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return 1.0 if lower_shadow > 2 * body_size and upper_shadow < body_size else 0.0
    
    def _is_shooting_star(self, candle: pd.Series) -> float:
        """Check if candle is a shooting star"""
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        return 1.0 if upper_shadow > 2 * body_size and lower_shadow < body_size else 0.0
    
    def _is_engulfing(self, candles: pd.DataFrame) -> float:
        """Check if there's an engulfing pattern"""
        if len(candles) < 2:
            return 0.0
        
        prev = candles.iloc[0]
        curr = candles.iloc[1]
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        
        # Bullish engulfing
        if (curr['close'] > curr['open'] and  # Current is bullish
            prev['close'] < prev['open'] and  # Previous is bearish
            curr['open'] < prev['close'] and  # Current opens below previous close
            curr['close'] > prev['open']):    # Current closes above previous open
            return 1.0
        
        # Bearish engulfing
        elif (curr['close'] < curr['open'] and  # Current is bearish
              prev['close'] > prev['open'] and  # Previous is bullish
              curr['open'] > prev['close'] and  # Current opens above previous close
              curr['close'] < prev['open']):    # Current closes below previous open
            return -1.0
        
        return 0.0
    
    def _find_support_resistance(self, df: pd.DataFrame, period: int) -> Tuple[float, float]:
        """Find support and resistance levels"""
        recent_data = df.tail(period)
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        return support, resistance
    
    def _calculate_price_efficiency(self, prices: pd.Series, period: int) -> float:
        """Calculate price efficiency (how random the price movement is)"""
        if len(prices) < period:
            return 0.0
        
        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0
        
        # Variance ratio test
        var_1 = returns.var()
        var_k = returns.rolling(period).sum().var()
        efficiency = var_1 / var_k if var_k > 0 else 0.0
        return efficiency
    
    def _calculate_volume_profile(self, df: pd.DataFrame, period: int) -> float:
        """Calculate volume profile (concentration of volume)"""
        if len(df) < period:
            return 0.0
        
        recent_data = df.tail(period)
        total_volume = recent_data['volume'].sum()
        if total_volume == 0:
            return 0.0
        
        # Calculate volume-weighted average price
        vwap = (recent_data['volume'] * recent_data['close']).sum() / total_volume
        
        # Calculate volume concentration around VWAP
        volume_concentration = recent_data['volume'].abs().sum() / total_volume
        return volume_concentration

# Example usage
if __name__ == "__main__":
    # Example candle data
    sample_candles = [
        {'timestamp': datetime.now() - timedelta(minutes=i), 'open': 100 + i*0.1, 'high': 100.5 + i*0.1, 
         'low': 99.5 + i*0.1, 'close': 100.2 + i*0.1, 'volume': 1000 + i*10, 'symbol': 'BTC/USDT', 'timeframe': '1m'}
        for i in range(200, 0, -1)
    ]
    
    engineer = FeatureEngineer()
    feature_set = engineer.engineer_features(sample_candles)
    
    print(f"Generated {len(feature_set.features)} features")
    print("Sample features:")
    for key, value in list(feature_set.features.items())[:10]:
        print(f"  {key}: {value:.4f}")
