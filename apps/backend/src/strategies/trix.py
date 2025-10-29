"""
TRIX Indicator for AlphaPulse
Triple exponential smoothed momentum indicator - excellent noise filter
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

class TRIX:
    """
    TRIX (Triple Exponential Average)
    
    Momentum oscillator showing rate of change of triple EMA.
    Excellent for filtering out market noise.
    
    Formula:
    1. EMA1 = EMA(Close, period)
    2. EMA2 = EMA(EMA1, period)
    3. EMA3 = EMA(EMA2, period)
    4. TRIX = (EMA3 - EMA3[1]) / EMA3[1] × 100
    5. Signal = EMA(TRIX, signal_period)
    
    Interpretation:
    - TRIX > 0: Bullish momentum
    - TRIX < 0: Bearish momentum
    - TRIX crosses signal: Trend change
    - Divergences: Reversal signals
    - TRIX turning points: Early trend changes
    
    Advantage: Filters out insignificant price movements
    """
    
    def __init__(self, period: int = 14, signal_period: int = 9):
        """
        Initialize TRIX
        
        Args:
            period: Smoothing period (default: 14)
            signal_period: Signal line period (default: 9)
        """
        self.period = period
        self.signal_period = signal_period
    
    def calculate(
        self,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate TRIX
        
        Args:
            close: Close prices
            
        Returns:
            Tuple of (trix, signal_line)
        """
        try:
            if len(close) < self.period * 3 + self.signal_period:
                logger.warning("Insufficient data for TRIX")
                empty = np.full_like(close, np.nan)
                return empty, empty
            
            if TALIB_AVAILABLE:
                try:
                    trix = talib.TRIX(close, timeperiod=self.period)
                    signal = pd.Series(trix).ewm(span=self.signal_period, adjust=False).mean().values
                    return trix, signal
                except Exception:
                    pass
            
            # Manual calculation
            close_series = pd.Series(close)
            
            # Triple smoothing
            ema1 = close_series.ewm(span=self.period, adjust=False).mean()
            ema2 = ema1.ewm(span=self.period, adjust=False).mean()
            ema3 = ema2.ewm(span=self.period, adjust=False).mean()
            
            # Rate of change of EMA3
            trix = (ema3.pct_change() * 100).values
            
            # Signal line
            signal = pd.Series(trix).ewm(span=self.signal_period, adjust=False).mean().values
            
            return trix, signal
            
        except Exception as e:
            logger.error(f"Error calculating TRIX: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty
    
    def get_signals(
        self,
        trix: np.ndarray,
        signal: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from TRIX"""
        signals = []
        
        try:
            for i in range(1, len(trix)):
                # TRIX crosses above signal (bullish)
                if trix[i] > signal[i] and trix[i-1] <= signal[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bullish_crossover',
                        'trix': trix[i],
                        'signal': signal[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # TRIX crosses below signal (bearish)
                elif trix[i] < signal[i] and trix[i-1] >= signal[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bearish_crossover',
                        'trix': trix[i],
                        'signal': signal[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
                
                # Zero-line crosses (trend change)
                if trix[i] > 0 and trix[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_up',
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                elif trix[i] < 0 and trix[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_down',
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating TRIX signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate TRIX from DataFrame"""
        try:
            trix, signal = self.calculate(df['close'].values)
            
            df_copy = df.copy()
            df_copy['trix'] = trix
            df_copy['trix_signal'] = signal
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating TRIX from DataFrame: {e}")
            return df

# Convenience function
def trix(
    close: np.ndarray,
    period: int = 14,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate TRIX (convenience function)"""
    indicator = TRIX(period, signal_period)
    return indicator.calculate(close)

