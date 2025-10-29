"""
Aroon Oscillator for AlphaPulse
Trend identification and strength measurement
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

class AroonOscillator:
    """
    Aroon Oscillator
    
    Identifies trend strength and potential reversals.
    
    Formula:
    Aroon Up = ((period - periods since period high) / period) × 100
    Aroon Down = ((period - periods since period low) / period) × 100
    Aroon Oscillator = Aroon Up - Aroon Down
    
    Interpretation:
    - Aroon Up > 70: Strong uptrend
    - Aroon Down > 70: Strong downtrend
    - Aroon Oscillator > 0: Bullish (Up > Down)
    - Aroon Oscillator < 0: Bearish (Down > Up)
    - Aroon Oscillator crossing zero: Trend change
    
    Range: -100 to +100
    
    Popular period: 25
    """
    
    def __init__(self, period: int = 25):
        """
        Initialize Aroon
        
        Args:
            period: Lookback period (default: 25)
        """
        self.period = period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Aroon Indicator
        
        Args:
            high, low: Price arrays
            
        Returns:
            Tuple of (aroon_up, aroon_down, aroon_oscillator)
        """
        try:
            if len(high) < self.period + 1:
                logger.warning(f"Insufficient data for Aroon (need {self.period + 1}, got {len(high)})")
                empty = np.full_like(high, np.nan)
                return empty, empty, empty
            
            if TALIB_AVAILABLE:
                try:
                    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=self.period)
                    aroon_osc = aroon_up - aroon_down
                    return aroon_up, aroon_down, aroon_osc
                except Exception:
                    pass
            
            # Manual calculation
            aroon_up = np.full_like(high, np.nan, dtype=float)
            aroon_down = np.full_like(low, np.nan, dtype=float)
            
            for i in range(self.period, len(high)):
                # Get window
                high_window = high[i - self.period:i + 1]
                low_window = low[i - self.period:i + 1]
                
                # Find periods since high and low
                high_idx = np.argmax(high_window)
                low_idx = np.argmin(low_window)
                
                periods_since_high = self.period - high_idx
                periods_since_low = self.period - low_idx
                
                # Calculate Aroon values
                aroon_up[i] = ((self.period - periods_since_high) / self.period) * 100
                aroon_down[i] = ((self.period - periods_since_low) / self.period) * 100
            
            aroon_osc = aroon_up - aroon_down
            
            return aroon_up, aroon_down, aroon_osc
            
        except Exception as e:
            logger.error(f"Error calculating Aroon: {e}")
            empty = np.full_like(high, np.nan)
            return empty, empty, empty
    
    def get_signals(
        self,
        aroon_up: np.ndarray,
        aroon_down: np.ndarray,
        aroon_osc: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from Aroon"""
        signals = []
        
        try:
            for i in range(1, len(aroon_osc)):
                # Strong uptrend
                if aroon_up[i] > 70 and aroon_down[i] < 30:
                    signals.append({
                        'index': i,
                        'type': 'strong_uptrend',
                        'aroon_up': aroon_up[i],
                        'aroon_down': aroon_down[i],
                        'direction': 'bullish',
                        'confidence': 0.80
                    })
                
                # Strong downtrend
                elif aroon_down[i] > 70 and aroon_up[i] < 30:
                    signals.append({
                        'index': i,
                        'type': 'strong_downtrend',
                        'aroon_up': aroon_up[i],
                        'aroon_down': aroon_down[i],
                        'direction': 'bearish',
                        'confidence': 0.80
                    })
                
                # Oscillator zero-line crosses
                if aroon_osc[i] > 0 and aroon_osc[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'osc_cross_up',
                        'aroon_osc': aroon_osc[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                elif aroon_osc[i] < 0 and aroon_osc[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'osc_cross_down',
                        'aroon_osc': aroon_osc[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Aroon signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Aroon from DataFrame"""
        try:
            aroon_up, aroon_down, aroon_osc = self.calculate(
                df['high'].values,
                df['low'].values
            )
            
            df_copy = df.copy()
            df_copy['aroon_up'] = aroon_up
            df_copy['aroon_down'] = aroon_down
            df_copy['aroon_oscillator'] = aroon_osc
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Aroon from DataFrame: {e}")
            return df

# Convenience functions
def aroon(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Aroon (convenience function)"""
    indicator = AroonOscillator(period)
    return indicator.calculate(high, low)

