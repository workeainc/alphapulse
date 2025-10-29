"""
Ultimate Oscillator for AlphaPulse
Combines three timeframes to reduce false divergence signals
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

class UltimateOscillator:
    """
    Ultimate Oscillator
    
    Combines momentum from three different timeframes to reduce
    false divergence signals common in single-timeframe oscillators.
    
    Formula:
    Buying Pressure = Close - min(Low, Previous Close)
    True Range = max(High, Previous Close) - min(Low, Previous Close)
    Average7 = sum(BP, 7) / sum(TR, 7)
    Average14 = sum(BP, 14) / sum(TR, 14)
    Average28 = sum(BP, 28) / sum(TR, 28)
    UO = 100 × (4×Avg7 + 2×Avg14 + Avg28) / (4 + 2 + 1)
    
    Interpretation:
    - UO > 70: Overbought
    - UO < 30: Oversold
    - Divergences: More reliable than RSI
    - Best used with 3 timeframe confirmation
    
    Created by: Larry Williams
    """
    
    def __init__(
        self,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ):
        """
        Initialize Ultimate Oscillator
        
        Args:
            period1: Short period (default: 7)
            period2: Medium period (default: 14)
            period3: Long period (default: 28)
        """
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Ultimate Oscillator
        
        Args:
            high, low, close: Price arrays
            
        Returns:
            Ultimate Oscillator values
        """
        try:
            if len(close) < self.period3:
                logger.warning(f"Insufficient data for Ultimate Oscillator")
                return np.full_like(close, np.nan)
            
            if TALIB_AVAILABLE:
                try:
                    return talib.ULTOSC(high, low, close,
                                       timeperiod1=self.period1,
                                       timeperiod2=self.period2,
                                       timeperiod3=self.period3)
                except Exception:
                    pass
            
            # Manual calculation
            uo = np.full_like(close, np.nan)
            
            for i in range(1, len(close)):
                # Buying Pressure
                bp = close[i] - min(low[i], close[i-1])
                
                # True Range
                tr = max(high[i], close[i-1]) - min(low[i], close[i-1])
                
                # Store for rolling sums
                if i == 1:
                    bp_list = [bp]
                    tr_list = [tr]
                else:
                    bp_list.append(bp)
                    tr_list.append(tr)
                
                # Calculate averages once we have enough data
                if i >= self.period3:
                    bp_array = np.array(bp_list[-(self.period3):])
                    tr_array = np.array(tr_list[-(self.period3):])
                    
                    avg7 = bp_array[-self.period1:].sum() / tr_array[-self.period1:].sum() if tr_array[-self.period1:].sum() > 0 else 0
                    avg14 = bp_array[-self.period2:].sum() / tr_array[-self.period2:].sum() if tr_array[-self.period2:].sum() > 0 else 0
                    avg28 = bp_array.sum() / tr_array.sum() if tr_array.sum() > 0 else 0
                    
                    # Ultimate Oscillator
                    uo[i] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            
            return uo
            
        except Exception as e:
            logger.error(f"Error calculating Ultimate Oscillator: {e}")
            return np.full_like(close, np.nan)
    
    def get_signals(
        self,
        uo: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from Ultimate Oscillator"""
        signals = []
        
        try:
            for i in range(1, len(uo)):
                # Overbought
                if uo[i] > 70 and uo[i-1] <= 70:
                    signals.append({
                        'index': i,
                        'type': 'overbought',
                        'uo': uo[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
                
                # Oversold
                elif uo[i] < 30 and uo[i-1] >= 30:
                    signals.append({
                        'index': i,
                        'type': 'oversold',
                        'uo': uo[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating UO signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Ultimate Oscillator from DataFrame"""
        try:
            uo = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            df_copy = df.copy()
            df_copy['ultimate_oscillator'] = uo
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating UO from DataFrame: {e}")
            return df

# Convenience function
def ultimate_oscillator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28
) -> np.ndarray:
    """Calculate Ultimate Oscillator (convenience function)"""
    indicator = UltimateOscillator(period1, period2, period3)
    return indicator.calculate(high, low, close)

