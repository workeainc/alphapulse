"""
Chande Momentum Oscillator (CMO) for AlphaPulse
Alternative to RSI - more responsive momentum indicator
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class ChandeMomentumOscillator:
    """
    Chande Momentum Oscillator (CMO)
    
    Alternative to RSI that uses both up and down momentum.
    More responsive to price changes.
    
    Formula:
    Sum of Up Days = Sum of (Close - Close[1]) when positive
    Sum of Down Days = Sum of abs(Close - Close[1]) when negative
    CMO = ((Up - Down) / (Up + Down)) × 100
    
    Interpretation:
    - Range: -100 to +100
    - CMO > 50: Overbought
    - CMO < -50: Oversold
    - CMO > 0: Bullish momentum
    - CMO < 0: Bearish momentum
    
    Created by: Tushar Chande
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize CMO
        
        Args:
            period: Lookback period (default: 14)
        """
        self.period = period
    
    def calculate(
        self,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Chande Momentum Oscillator
        
        Args:
            close: Close prices
            
        Returns:
            CMO values
        """
        try:
            if len(close) < self.period + 1:
                logger.warning(f"Insufficient data for CMO (need {self.period + 1}, got {len(close)})")
                return np.full_like(close, np.nan)
            
            if TALIB_AVAILABLE:
                try:
                    return talib.CMO(close, timeperiod=self.period)
                except Exception:
                    pass
            
            # Manual calculation
            close_series = pd.Series(close)
            momentum = close_series.diff()
            
            # Separate up and down momentum
            up_momentum = momentum.clip(lower=0)
            down_momentum = (-momentum).clip(lower=0)
            
            # Sum over period
            sum_up = up_momentum.rolling(window=self.period).sum()
            sum_down = down_momentum.rolling(window=self.period).sum()
            
            # CMO formula
            cmo = ((sum_up - sum_down) / (sum_up + sum_down) * 100).values
            cmo = np.where(np.isnan(cmo), 0.0, cmo)
            
            return cmo
            
        except Exception as e:
            logger.error(f"Error calculating CMO: {e}")
            return np.full_like(close, np.nan)
    
    def get_signals(
        self,
        cmo: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from CMO"""
        signals = []
        
        try:
            for i in range(1, len(cmo)):
                # Overbought
                if cmo[i] > 50 and cmo[i-1] <= 50:
                    signals.append({
                        'index': i,
                        'type': 'overbought',
                        'cmo': cmo[i],
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
                
                # Oversold
                elif cmo[i] < -50 and cmo[i-1] >= -50:
                    signals.append({
                        'index': i,
                        'type': 'oversold',
                        'cmo': cmo[i],
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                # Zero crosses
                if cmo[i] > 0 and cmo[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_up',
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                elif cmo[i] < 0 and cmo[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_down',
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating CMO signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate CMO from DataFrame"""
        try:
            cmo = self.calculate(df['close'].values)
            
            df_copy = df.copy()
            df_copy[f'cmo_{self.period}'] = cmo
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating CMO from DataFrame: {e}")
            return df

# Convenience function
def chande_momentum_oscillator(
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """Calculate CMO (convenience function)"""
    indicator = ChandeMomentumOscillator(period)
    return indicator.calculate(close)

