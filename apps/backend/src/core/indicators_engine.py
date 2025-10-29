#!/usr/bin/env python3
"""
Technical Indicators Engine for AlphaPulse
Accurate indicator calculations with incremental updates for real-time processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class IndicatorValues:
    """Technical indicator values"""
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    pivot: float
    s1: float
    r1: float
    fib_236: float
    fib_382: float
    fib_500: float
    fib_618: float
    breakout_strength: float
    adx: float
    atr: float
    volume_sma: float

class TechnicalIndicators:
    """
    Technical indicators calculator with incremental updates
    Optimized for real-time processing with <50ms latency
    """
    
    def __init__(self):
        """Initialize technical indicators calculator"""
        # RSI parameters
        self.rsi_period = 14
        self.rsi_gains = []
        self.rsi_losses = []
        self.rsi_avg_gain = 0
        self.rsi_avg_loss = 0
        
        # MACD parameters
        self.macd_fast = 8
        self.macd_slow = 24
        self.macd_signal = 9
        self.macd_ema_fast = 0
        self.macd_ema_slow = 0
        self.macd_signal_ema = 0
        
        # Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std_dev = 2
        self.bb_sma_values = []
        
        # ADX parameters
        self.adx_period = 14
        self.dm_plus_values = []
        self.dm_minus_values = []
        self.tr_values = []
        self.adx_values = []
        
        # ATR parameters
        self.atr_period = 14
        self.atr_values = []
        
        # Volume SMA
        self.volume_sma_period = 20
        self.volume_values = []
        
        # Pivot point calculations
        self.pivot_high = 0
        self.pivot_low = 0
        
        logger.info("Technical Indicators Engine initialized")
    
    def calculate_rsi(self, close_prices: List[float]) -> float:
        """
        Calculate RSI using Wilder's smoothed method
        RSI = 100 - (100 / (1 + (Avg Gain / Avg Loss)))
        """
        if len(close_prices) < 2:
            return 50.0  # Neutral value
        
        # Calculate price change
        price_change = close_prices[-1] - close_prices[-2]
        
        if len(self.rsi_gains) == 0:
            # Initialize with first values
            if price_change > 0:
                self.rsi_gains.append(price_change)
                self.rsi_losses.append(0)
            else:
                self.rsi_gains.append(0)
                self.rsi_losses.append(abs(price_change))
            
            return 50.0  # Neutral value for first calculation
        
        # Update gains and losses
        if price_change > 0:
            self.rsi_gains.append(price_change)
            self.rsi_losses.append(0)
        else:
            self.rsi_gains.append(0)
            self.rsi_losses.append(abs(price_change))
        
        # Keep only the last 'period' values
        if len(self.rsi_gains) > self.rsi_period:
            self.rsi_gains = self.rsi_gains[-self.rsi_period:]
            self.rsi_losses = self.rsi_losses[-self.rsi_period:]
        
        # Calculate smoothed averages
        if len(self.rsi_gains) == self.rsi_period:
            # First calculation - use simple average
            if self.rsi_avg_gain == 0:
                self.rsi_avg_gain = np.mean(self.rsi_gains)
                self.rsi_avg_loss = np.mean(self.rsi_losses)
            else:
                # Subsequent calculations - use Wilder's smoothing
                self.rsi_avg_gain = (self.rsi_avg_gain * (self.rsi_period - 1) + self.rsi_gains[-1]) / self.rsi_period
                self.rsi_avg_loss = (self.rsi_avg_loss * (self.rsi_period - 1) + self.rsi_losses[-1]) / self.rsi_period
        
        # Calculate RSI
        if self.rsi_avg_loss == 0:
            return 100.0
        
        rs = self.rsi_avg_gain / self.rsi_avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, close_prices: List[float]) -> Tuple[float, float, float]:
        """
        Calculate MACD (8-24-9 settings)
        MACD Line = 8-period EMA - 24-period EMA
        Signal = 9-period EMA of MACD
        """
        if len(close_prices) < self.macd_slow:
            return 0.0, 0.0, 0.0
        
        current_price = close_prices[-1]
        
        # Calculate EMAs
        if self.macd_ema_fast == 0:
            # Initialize with simple average
            self.macd_ema_fast = np.mean(close_prices[-self.macd_fast:])
            self.macd_ema_slow = np.mean(close_prices[-self.macd_slow:])
        else:
            # Update EMAs
            fast_multiplier = 2 / (self.macd_fast + 1)
            slow_multiplier = 2 / (self.macd_slow + 1)
            
            self.macd_ema_fast = (current_price * fast_multiplier) + (self.macd_ema_fast * (1 - fast_multiplier))
            self.macd_ema_slow = (current_price * slow_multiplier) + (self.macd_ema_slow * (1 - slow_multiplier))
        
        # Calculate MACD line
        macd_line = self.macd_ema_fast - self.macd_ema_slow
        
        # Calculate signal line
        if self.macd_signal_ema == 0:
            self.macd_signal_ema = macd_line
        else:
            signal_multiplier = 2 / (self.macd_signal + 1)
            self.macd_signal_ema = (macd_line * signal_multiplier) + (self.macd_signal_ema * (1 - signal_multiplier))
        
        # Calculate histogram
        macd_histogram = macd_line - self.macd_signal_ema
        
        return macd_line, self.macd_signal_ema, macd_histogram
    
    def calculate_bollinger_bands(self, close_prices: List[float]) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands (20-period SMA, 2 Std Dev)
        """
        if len(close_prices) < self.bb_period:
            current_price = close_prices[-1]
            return current_price * 1.02, current_price, current_price * 0.98
        
        # Update SMA values
        self.bb_sma_values = close_prices[-self.bb_period:]
        bb_middle = np.mean(self.bb_sma_values)
        
        # Calculate standard deviation
        bb_std = np.std(self.bb_sma_values)
        
        # Calculate bands
        bb_upper = bb_middle + (self.bb_std_dev * bb_std)
        bb_lower = bb_middle - (self.bb_std_dev * bb_std)
        
        return bb_upper, bb_middle, bb_lower
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Tuple[float, float, float]:
        """
        Calculate Pivot Points
        Pivot = (High + Low + Close) / 3
        S1 = (2 * Pivot) - High
        R1 = (2 * Pivot) - Low
        """
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        
        return pivot, s1, r1
    
    def calculate_fibonacci_levels(self, high: float, low: float) -> Tuple[float, float, float, float]:
        """
        Calculate Fibonacci Retracement levels
        Levels: 23.6%, 38.2%, 50%, 61.8%
        """
        price_range = high - low
        
        fib_236 = low + (price_range * 0.236)
        fib_382 = low + (price_range * 0.382)
        fib_500 = low + (price_range * 0.500)
        fib_618 = low + (price_range * 0.618)
        
        return fib_236, fib_382, fib_500, fib_618
    
    def calculate_atr(self, high: float, low: float, close: float) -> float:
        """
        Calculate Average True Range (ATR)
        """
        if len(self.atr_values) == 0:
            # First calculation
            tr = high - low
            self.atr_values.append(tr)
            return tr
        
        # Calculate True Range
        prev_close = self.atr_values[-1] if len(self.atr_values) > 0 else close
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = max(tr1, tr2, tr3)
        
        # Update ATR values
        self.atr_values.append(tr)
        
        # Keep only the last 'period' values
        if len(self.atr_values) > self.atr_period:
            self.atr_values = self.atr_values[-self.atr_period:]
        
        # Calculate ATR
        if len(self.atr_values) == self.atr_period:
            atr = np.mean(self.atr_values)
        else:
            atr = np.mean(self.atr_values)
        
        return atr
    
    def calculate_adx(self, high: float, low: float, close: float) -> float:
        """
        Calculate Average Directional Index (ADX)
        """
        if len(self.dm_plus_values) == 0:
            # First calculation
            self.dm_plus_values.append(0)
            self.dm_minus_values.append(0)
            self.tr_values.append(high - low)
            return 25.0  # Neutral value
        
        # Calculate True Range
        prev_high = self.dm_plus_values[-1] if len(self.dm_plus_values) > 0 else high
        prev_low = self.dm_minus_values[-1] if len(self.dm_minus_values) > 0 else low
        prev_close = self.tr_values[-1] if len(self.tr_values) > 0 else close
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = max(tr1, tr2, tr3)
        
        # Calculate Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        
        if up_move > down_move and up_move > 0:
            dm_plus = up_move
            dm_minus = 0
        elif down_move > up_move and down_move > 0:
            dm_plus = 0
            dm_minus = down_move
        else:
            dm_plus = 0
            dm_minus = 0
        
        # Update values
        self.dm_plus_values.append(dm_plus)
        self.dm_minus_values.append(dm_minus)
        self.tr_values.append(tr)
        
        # Keep only the last 'period' values
        if len(self.dm_plus_values) > self.adx_period:
            self.dm_plus_values = self.dm_plus_values[-self.adx_period:]
            self.dm_minus_values = self.dm_minus_values[-self.adx_period:]
            self.tr_values = self.tr_values[-self.adx_period:]
        
        # Calculate ADX
        if len(self.dm_plus_values) == self.adx_period:
            avg_tr = np.mean(self.tr_values)
            avg_dm_plus = np.mean(self.dm_plus_values)
            avg_dm_minus = np.mean(self.dm_minus_values)
            
            if avg_tr == 0:
                return 25.0
            
            di_plus = (avg_dm_plus / avg_tr) * 100
            di_minus = (avg_dm_minus / avg_tr) * 100
            
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
            
            self.adx_values.append(dx)
            
            if len(self.adx_values) > self.adx_period:
                self.adx_values = self.adx_values[-self.adx_period:]
            
            adx = np.mean(self.adx_values)
            return adx
        
        return 25.0  # Neutral value
    
    def calculate_volume_sma(self, volume: float) -> float:
        """
        Calculate Volume Simple Moving Average
        """
        self.volume_values.append(volume)
        
        # Keep only the last 'period' values
        if len(self.volume_values) > self.volume_sma_period:
            self.volume_values = self.volume_values[-self.volume_sma_period:]
        
        return np.mean(self.volume_values)
    
    def calculate_breakout_strength(self, volume: float, volume_sma: float, atr: float, adx: float) -> float:
        """
        Calculate Breakout Strength (Composite)
        Volume Multiplier × 0.6 + ATR Volatility × 0.3 + ADX > 25 × 0.1
        """
        # Volume multiplier (current volume vs SMA)
        volume_multiplier = volume / volume_sma if volume_sma > 0 else 1.0
        
        # ATR volatility (normalized)
        atr_volatility = min(atr / 100, 2.0)  # Normalize ATR, cap at 2.0
        
        # ADX component (1 if ADX > 25, 0 otherwise)
        adx_component = 1.0 if adx > 25 else 0.0
        
        # Calculate composite breakout strength
        breakout_strength = (
            (volume_multiplier * 0.6) +
            (atr_volatility * 0.3) +
            (adx_component * 0.1)
        )
        
        return min(breakout_strength, 3.0)  # Cap at 3.0
    
    def calculate_all_indicators(self, 
                                open_price: float,
                                high: float, 
                                low: float, 
                                close: float, 
                                volume: float,
                                close_prices: List[float]) -> IndicatorValues:
        """
        Calculate all technical indicators in one pass
        Optimized for real-time processing
        """
        try:
            # RSI
            rsi = self.calculate_rsi(close_prices)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self.calculate_macd(close_prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices)
            
            # Pivot Points
            pivot, s1, r1 = self.calculate_pivot_points(high, low, close)
            
            # Fibonacci Levels
            fib_236, fib_382, fib_500, fib_618 = self.calculate_fibonacci_levels(high, low)
            
            # ATR
            atr = self.calculate_atr(high, low, close)
            
            # ADX
            adx = self.calculate_adx(high, low, close)
            
            # Volume SMA
            volume_sma = self.calculate_volume_sma(volume)
            
            # Breakout Strength
            breakout_strength = self.calculate_breakout_strength(volume, volume_sma, atr, adx)
            
            return IndicatorValues(
                rsi=rsi,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                pivot=pivot,
                s1=s1,
                r1=r1,
                fib_236=fib_236,
                fib_382=fib_382,
                fib_500=fib_500,
                fib_618=fib_618,
                breakout_strength=breakout_strength,
                adx=adx,
                atr=atr,
                volume_sma=volume_sma
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return neutral values on error
            return IndicatorValues(
                rsi=50.0,
                macd_line=0.0,
                macd_signal=0.0,
                macd_histogram=0.0,
                bb_upper=close * 1.02,
                bb_middle=close,
                bb_lower=close * 0.98,
                pivot=(high + low + close) / 3,
                s1=low * 0.99,
                r1=high * 1.01,
                fib_236=low + (high - low) * 0.236,
                fib_382=low + (high - low) * 0.382,
                fib_500=low + (high - low) * 0.5,
                fib_618=low + (high - low) * 0.618,
                breakout_strength=0.5,
                adx=25.0,
                atr=100.0,
                volume_sma=volume
            )
    
    def detect_rsi_divergence(self, prices: List[float], rsi_values: List[float]) -> Optional[str]:
        """
        Detect RSI divergence
        Returns: 'bullish_divergence', 'bearish_divergence', or None
        """
        if len(prices) < 10 or len(rsi_values) < 10:
            return None
        
        # Look for recent swing highs and lows
        recent_prices = prices[-10:]
        recent_rsi = rsi_values[-10:]
        
        # Find local extremes
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []
        
        for i in range(1, len(recent_prices) - 1):
            # Price highs
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                price_highs.append((i, recent_prices[i]))
                rsi_highs.append((i, recent_rsi[i]))
            
            # Price lows
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                price_lows.append((i, recent_prices[i]))
                rsi_lows.append((i, recent_rsi[i]))
        
        # Check for divergence
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            # Bearish divergence: Price higher high, RSI lower high
            if (price_highs[-1][1] > price_highs[-2][1] and 
                rsi_highs[-1][1] < rsi_highs[-2][1]):
                return 'bearish_divergence'
        
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Bullish divergence: Price lower low, RSI higher low
            if (price_lows[-1][1] < price_lows[-2][1] and 
                rsi_lows[-1][1] > rsi_lows[-2][1]):
                return 'bullish_divergence'
        
        return None
    
    def get_indicator_summary(self) -> dict:
        """Get summary of current indicator values"""
        return {
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'adx_period': self.adx_period,
            'atr_period': self.atr_period,
            'volume_sma_period': self.volume_sma_period
        }


# Example usage
def main():
    """Example usage of Technical Indicators"""
    # Initialize indicators
    indicators = TechnicalIndicators()
    
    # Sample data
    sample_data = [
        {'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050, 'volume': 1000},
        {'open': 50050, 'high': 50200, 'low': 50000, 'close': 50150, 'volume': 1200},
        {'open': 50150, 'high': 50300, 'low': 50100, 'close': 50250, 'volume': 1100},
        {'open': 50250, 'high': 50400, 'low': 50200, 'close': 50350, 'volume': 1300},
        {'open': 50350, 'high': 50500, 'low': 50300, 'close': 50450, 'volume': 1400},
    ]
    
    close_prices = []
    
    print("📊 Technical Indicators Calculation Example:")
    print("=" * 50)
    
    for i, data in enumerate(sample_data):
        close_prices.append(data['close'])
        
        # Calculate all indicators
        result = indicators.calculate_all_indicators(
            open_price=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            close_prices=close_prices
        )
        
        print(f"Candle {i+1}:")
        print(f"  RSI: {result.rsi:.2f}")
        print(f"  MACD: {result.macd_line:.2f} | Signal: {result.macd_signal:.2f} | Hist: {result.macd_histogram:.2f}")
        print(f"  BB: Upper={result.bb_upper:.2f} | Middle={result.bb_middle:.2f} | Lower={result.bb_lower:.2f}")
        print(f"  Pivot: {result.pivot:.2f} | S1: {result.s1:.2f} | R1: {result.r1:.2f}")
        print(f"  Fib: 23.6%={result.fib_236:.2f} | 38.2%={result.fib_382:.2f} | 50%={result.fib_500:.2f} | 61.8%={result.fib_618:.2f}")
        print(f"  Breakout Strength: {result.breakout_strength:.2f}")
        print(f"  ADX: {result.adx:.2f} | ATR: {result.atr:.2f}")
        print(f"  Volume SMA: {result.volume_sma:.2f}")
        print("-" * 30)


if __name__ == "__main__":
    main()
