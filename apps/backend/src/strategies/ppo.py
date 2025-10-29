"""
Percentage Price Oscillator (PPO) for AlphaPulse
Percentage version of MACD for better cross-asset comparison
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class PercentagePriceOscillator:
    """
    Percentage Price Oscillator (PPO)
    
    Similar to MACD but expressed as percentage, making it better for:
    - Comparing different assets/price levels
    - Identifying overbought/oversold consistently
    
    Formula:
    PPO = ((EMA_fast - EMA_slow) / EMA_slow) × 100
    Signal = EMA(PPO, signal_period)
    Histogram = PPO - Signal
    
    Interpretation:
    - PPO > 0: Bullish (fast MA above slow MA)
    - PPO < 0: Bearish
    - PPO crosses signal: Trend change
    - Divergences: Reversal signals
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        Initialize PPO
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(
        self,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate PPO
        
        Args:
            close: Close prices
            
        Returns:
            Tuple of (ppo, signal, histogram)
        """
        try:
            if len(close) < self.slow_period + self.signal_period:
                logger.warning("Insufficient data for PPO")
                empty = np.full_like(close, np.nan)
                return empty, empty, empty
            
            if TALIB_AVAILABLE:
                try:
                    ppo = talib.PPO(close, fastperiod=self.fast_period,
                                   slowperiod=self.slow_period, matype=0)
                    signal = pd.Series(ppo).ewm(span=self.signal_period, adjust=False).mean().values
                    histogram = ppo - signal
                    return ppo, signal, histogram
                except Exception:
                    pass
            
            # Manual calculation
            close_series = pd.Series(close)
            ema_fast = close_series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = close_series.ewm(span=self.slow_period, adjust=False).mean()
            
            # PPO as percentage
            ppo = ((ema_fast - ema_slow) / ema_slow * 100).values
            
            # Signal line
            signal = pd.Series(ppo).ewm(span=self.signal_period, adjust=False).mean().values
            
            # Histogram
            histogram = ppo - signal
            
            return ppo, signal, histogram
            
        except Exception as e:
            logger.error(f"Error calculating PPO: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty, empty
    
    def get_signals(
        self,
        ppo: np.ndarray,
        signal: np.ndarray,
        histogram: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from PPO"""
        signals = []
        
        try:
            for i in range(1, len(ppo)):
                # PPO crosses above signal (bullish)
                if ppo[i] > signal[i] and ppo[i-1] <= signal[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bullish_crossover',
                        'ppo': ppo[i],
                        'signal': signal[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # PPO crosses below signal (bearish)
                elif ppo[i] < signal[i] and ppo[i-1] >= signal[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bearish_crossover',
                        'ppo': ppo[i],
                        'signal': signal[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
                
                # Zero-line crosses
                if ppo[i] > 0 and ppo[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_up',
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                elif ppo[i] < 0 and ppo[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_down',
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating PPO signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate PPO from DataFrame"""
        try:
            ppo, signal, histogram = self.calculate(df['close'].values)
            
            df_copy = df.copy()
            df_copy['ppo'] = ppo
            df_copy['ppo_signal'] = signal
            df_copy['ppo_histogram'] = histogram
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating PPO from DataFrame: {e}")
            return df

# Convenience function
def percentage_price_oscillator(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate PPO (convenience function)"""
    indicator = PercentagePriceOscillator(fast_period, slow_period, signal_period)
    return indicator.calculate(close)

