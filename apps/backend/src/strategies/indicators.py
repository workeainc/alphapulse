#!/usr/bin/env python3
"""
Technical Indicators Engine for AlphaPulse
Provides comprehensive technical analysis indicators using TA-Lib
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging

# Try to import TA-Lib, fallback to basic implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TA-Lib successfully imported")
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available, using basic implementations")

class TechnicalIndicators:
    """Technical indicators calculator using TA-Lib or fallback implementations"""
    
    def __init__(self):
        """Initialize technical indicators calculator"""
        logger.info(f"Technical indicators initialized. TA-Lib: {TALIB_AVAILABLE}")
    
    def calculate_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            closes: Array of closing prices
            period: RSI period (default: 14)
            
        Returns:
            Array of RSI values
        """
        if len(closes) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation (need {period + 1}, got {len(closes)})")
            return np.full_like(closes, np.nan)
        
        if TALIB_AVAILABLE:
            try:
                return talib.RSI(closes, timeperiod=period)
            except Exception as e:
                logger.error(f"Error calculating RSI with TA-Lib: {e}")
                return self._calculate_rsi_basic(closes, period)
        else:
            return self._calculate_rsi_basic(closes, period)
    
    def calculate_macd(self, closes: np.ndarray, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            closes: Array of closing prices
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(closes) < max(fast_period, slow_period) + signal_period:
            logger.warning(f"Insufficient data for MACD calculation")
            empty_array = np.full_like(closes, np.nan)
            return empty_array, empty_array, empty_array
        
        if TALIB_AVAILABLE:
            try:
                macd, signal, hist = talib.MACD(closes, fastperiod=fast_period, 
                                               slowperiod=slow_period, signalperiod=signal_period)
                return macd, signal, hist
            except Exception as e:
                logger.error(f"Error calculating MACD with TA-Lib: {e}")
                return self._calculate_macd_basic(closes, fast_period, slow_period, signal_period)
        else:
            return self._calculate_macd_basic(closes, fast_period, slow_period, signal_period)
    
    def calculate_bollinger_bands(self, closes: np.ndarray, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands
        
        Args:
            closes: Array of closing prices
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        if len(closes) < period:
            logger.warning(f"Insufficient data for Bollinger Bands calculation")
            empty_array = np.full_like(closes, np.nan)
            return empty_array, empty_array, empty_array
        
        if TALIB_AVAILABLE:
            try:
                upper, middle, lower = talib.BBANDS(closes, timeperiod=period, 
                                                   nbdevup=std_dev, nbdevdn=std_dev, 
                                                   matype=0)
                return upper, middle, lower
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands with TA-Lib: {e}")
                return self._calculate_bollinger_bands_basic(closes, period, std_dev)
        else:
            return self._calculate_bollinger_bands_basic(closes, period, std_dev)
    
    def calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, 
                           closes: np.ndarray, k_period: int = 14, 
                           d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Tuple of (%K line, %D line)
        """
        if len(closes) < k_period + d_period:
            logger.warning(f"Insufficient data for Stochastic calculation")
            empty_array = np.full_like(closes, np.nan)
            return empty_array, empty_array
        
        if TALIB_AVAILABLE:
            try:
                k, d = talib.STOCH(highs, lows, closes, fastk_period=k_period, 
                                  slowk_period=d_period, slowd_period=d_period)
                return k, d
            except Exception as e:
                logger.error(f"Error calculating Stochastic with TA-Lib: {e}")
                return self._calculate_stochastic_basic(highs, lows, closes, k_period, d_period)
        else:
            return self._calculate_stochastic_basic(highs, lows, closes, k_period, d_period)
    
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                     closes: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR)
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            period: ATR period (default: 14)
            
        Returns:
            Array of ATR values
        """
        if len(closes) < period + 1:
            logger.warning(f"Insufficient data for ATR calculation")
            return np.full_like(closes, np.nan)
        
        if TALIB_AVAILABLE:
            try:
                return talib.ATR(highs, lows, closes, timeperiod=period)
            except Exception as e:
                logger.error(f"Error calculating ATR with TA-Lib: {e}")
                return self._calculate_atr_basic(highs, lows, closes, period)
        else:
            return self._calculate_atr_basic(highs, lows, closes, period)
    
    def calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Array of values
            period: Moving average period
            
        Returns:
            Array of SMA values
        """
        if len(data) < period:
            logger.warning(f"Insufficient data for SMA calculation")
            return np.full_like(data, np.nan)
        
        if TALIB_AVAILABLE:
            try:
                return talib.SMA(data, timeperiod=period)
            except Exception as e:
                logger.error(f"Error calculating SMA with TA-Lib: {e}")
                return self._calculate_sma_basic(data, period)
        else:
            return self._calculate_sma_basic(data, period)
    
    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Array of values
            period: EMA period
            
        Returns:
            Array of EMA values
        """
        if len(data) < period:
            logger.warning(f"Insufficient data for EMA calculation")
            return np.full_like(data, np.nan)
        
        if TALIB_AVAILABLE:
            try:
                return talib.EMA(data, timeperiod=period)
            except Exception as e:
                logger.error(f"Error calculating EMA with TA-Lib: {e}")
                return self._calculate_ema_basic(data, period)
        else:
            return self._calculate_ema_basic(data, period)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate all technical indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with indicator names as keys and values as arrays
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return {}
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        
        indicators = {}
        
        # Calculate all indicators
        try:
            # RSI
            indicators['rsi'] = self.calculate_rsi(closes)
            
            # MACD
            macd, signal, hist = self.calculate_macd(closes)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(highs, lows, closes)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # ATR
            indicators['atr'] = self.calculate_atr(highs, lows, closes)
            
            # Moving Averages
            indicators['sma_20'] = self.calculate_sma(closes, 20)
            indicators['sma_50'] = self.calculate_sma(closes, 50)
            indicators['ema_12'] = self.calculate_ema(closes, 12)
            indicators['ema_26'] = self.calculate_ema(closes, 26)
            
            logger.info(f"Calculated {len(indicators)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added as columns
        """
        # Calculate all indicators
        indicators = self.calculate_all_indicators(df)
        
        # Add indicators to DataFrame
        for indicator_name, values in indicators.items():
            df[indicator_name] = values
        
        logger.info(f"Added {len(indicators)} indicators to DataFrame")
        return df
    
    def get_signal_strength(self, indicators: Dict[str, np.ndarray], 
                           current_index: int = -1) -> Dict[str, float]:
        """
        Calculate signal strength for current indicators
        
        Args:
            indicators: Dictionary of indicator values
            current_index: Index to calculate signals for (default: last)
            
        Returns:
            Dictionary with signal strengths
        """
        if not indicators:
            return {}
        
        signals = {}
        
        try:
            # RSI signals
            if 'rsi' in indicators:
                rsi = indicators['rsi'][current_index]
                if not np.isnan(rsi):
                    if rsi < 30:
                        signals['rsi'] = 1.0  # Oversold - bullish
                    elif rsi > 70:
                        signals['rsi'] = -1.0  # Overbought - bearish
                    else:
                        signals['rsi'] = 0.0  # Neutral
            
            # MACD signals
            if all(key in indicators for key in ['macd', 'macd_signal']):
                macd = indicators['macd'][current_index]
                signal = indicators['macd_signal'][current_index]
                if not (np.isnan(macd) or np.isnan(signal)):
                    if macd > signal:
                        signals['macd'] = 1.0  # Bullish crossover
                    else:
                        signals['macd'] = -1.0  # Bearish crossover
            
            # Bollinger Bands signals
            if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower', 'close']):
                close = indicators.get('close', [0])[current_index]
                bb_upper = indicators['bb_upper'][current_index]
                bb_lower = indicators['bb_lower'][current_index]
                if not (np.isnan(close) or np.isnan(bb_upper) or np.isnan(bb_lower)):
                    if close < bb_lower:
                        signals['bb'] = 1.0  # Below lower band - potential bounce
                    elif close > bb_upper:
                        signals['bb'] = -1.0  # Above upper band - potential reversal
                    else:
                        signals['bb'] = 0.0  # Within bands
            
            # Stochastic signals
            if all(key in indicators for key in ['stoch_k', 'stoch_d']):
                stoch_k = indicators['stoch_k'][current_index]
                stoch_d = indicators['stoch_d'][current_index]
                if not (np.isnan(stoch_k) or np.isnan(stoch_d)):
                    if stoch_k < 20 and stoch_d < 20:
                        signals['stoch'] = 1.0  # Oversold - bullish
                    elif stoch_k > 80 and stoch_d > 80:
                        signals['stoch'] = -1.0  # Overbought - bearish
                    else:
                        signals['stoch'] = 0.0  # Neutral
            
            # Moving Average signals
            if all(key in indicators for key in ['ema_12', 'ema_26']):
                ema_12 = indicators['ema_12'][current_index]
                ema_26 = indicators['ema_26'][current_index]
                if not (np.isnan(ema_12) or np.isnan(ema_26)):
                    if ema_12 > ema_26:
                        signals['ema_trend'] = 1.0  # Bullish trend
                    else:
                        signals['ema_trend'] = -1.0  # Bearish trend
            
        except Exception as e:
            logger.error(f"Error calculating signal strengths: {e}")
        
        return signals
    
    # Basic implementations (fallback when TA-Lib is not available)
    
    def _calculate_rsi_basic(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Basic RSI calculation"""
        rsi = np.full_like(closes, np.nan)
        
        if len(closes) < period + 1:
            return rsi
        
        # Calculate price changes
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gains = np.full_like(closes, np.nan)
        avg_losses = np.full_like(closes, np.nan)
        
        # First average
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Subsequent averages using exponential smoothing
        for i in range(period + 1, len(closes)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        # Calculate RSI
        for i in range(period, len(closes)):
            if avg_losses[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_basic(self, closes: np.ndarray, fast_period: int = 12, 
                             slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Basic MACD calculation"""
        if len(closes) < slow_period:
            empty_array = np.full_like(closes, np.nan)
            return empty_array, empty_array, empty_array
        
        # Calculate EMAs
        ema_fast = self._calculate_ema_basic(closes, fast_period)
        ema_slow = self._calculate_ema_basic(closes, slow_period)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = self._calculate_ema_basic(macd_line, signal_period)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands_basic(self, closes: np.ndarray, period: int = 20, 
                                       std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Basic Bollinger Bands calculation"""
        if len(closes) < period:
            empty_array = np.full_like(closes, np.nan)
            return empty_array, empty_array, empty_array
        
        # Middle band (SMA)
        middle_band = self._calculate_sma_basic(closes, period)
        
        # Calculate standard deviation
        upper_band = np.full_like(closes, np.nan)
        lower_band = np.full_like(closes, np.nan)
        
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            std = np.std(window)
            upper_band[i] = middle_band[i] + (std_dev * std)
            lower_band[i] = middle_band[i] - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_stochastic_basic(self, highs: np.ndarray, lows: np.ndarray, 
                                  closes: np.ndarray, k_period: int = 14, 
                                  d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Basic Stochastic calculation"""
        if len(closes) < k_period:
            empty_array = np.full_like(closes, np.nan)
            return empty_array, empty_array
        
        # Calculate %K
        k_line = np.full_like(closes, np.nan)
        for i in range(k_period - 1, len(closes)):
            window_high = np.max(highs[i - k_period + 1:i + 1])
            window_low = np.min(lows[i - k_period + 1:i + 1])
            if window_high != window_low:
                k_line[i] = ((closes[i] - window_low) / (window_high - window_low)) * 100
            else:
                k_line[i] = 50
        
        # Calculate %D (SMA of %K)
        d_line = self._calculate_sma_basic(k_line, d_period)
        
        return k_line, d_line
    
    def _calculate_atr_basic(self, highs: np.ndarray, lows: np.ndarray, 
                            closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Basic ATR calculation"""
        if len(closes) < period + 1:
            return np.full_like(closes, np.nan)
        
        # Calculate True Range
        tr = np.full_like(closes, np.nan)
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            tr[i] = max(high_low, high_close_prev, low_close_prev)
        
        # Calculate ATR (SMA of True Range)
        atr = self._calculate_sma_basic(tr, period)
        
        return atr
    
    def _calculate_sma_basic(self, data: np.ndarray, period: int) -> np.ndarray:
        """Basic SMA calculation"""
        sma = np.full_like(data, np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _calculate_ema_basic(self, data: np.ndarray, period: int) -> np.ndarray:
        """Basic EMA calculation"""
        ema = np.full_like(data, np.nan)
        
        if len(data) < period:
            return ema
        
        # First EMA is SMA
        ema[period - 1] = np.mean(data[:period])
        
        # Multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate subsequent EMAs
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema

# Example usage
if __name__ == "__main__":
    # Test the technical indicators
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.randint(1000, 10000, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Initialize indicators calculator
    indicators_calc = TechnicalIndicators()
    
    # Calculate all indicators
    indicators = indicators_calc.calculate_all_indicators(df)
    
    # Get signal strengths
    signals = indicators_calc.get_signal_strength(indicators)
    
    print("Technical Indicators Results:")
    print(f"Indicators calculated: {list(indicators.keys())}")
    print(f"Signal strengths: {signals}")
    
    # Test individual indicators
    rsi = indicators_calc.calculate_rsi(closes)
    print(f"\nRSI (last 5 values): {rsi[-5:]}")
    
    macd, signal, hist = indicators_calc.calculate_macd(closes)
    print(f"MACD (last 5 values): {macd[-5:]}")
    print(f"MACD Signal (last 5 values): {signal[-5:]}")
