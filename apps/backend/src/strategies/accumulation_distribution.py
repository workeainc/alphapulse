"""
Accumulation/Distribution Line for AlphaPulse
Volume flow indicator showing cumulative buying/selling pressure
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

class AccumulationDistribution:
    """
    Accumulation/Distribution Line (A/D Line)
    
    Volume-based indicator showing cumulative flow of money:
    - Rising A/D = Accumulation (buying pressure)
    - Falling A/D = Distribution (selling pressure)
    - Divergences predict reversals
    
    Formula:
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = MFM × Volume
    A/D = Cumulative Sum of MFV
    
    Similar to OBV but uses price position in range, not just close vs open
    """
    
    def __init__(self):
        """Initialize A/D Line"""
        pass
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Accumulation/Distribution Line
        
        Args:
            high, low, close: Price arrays
            volume: Volume array
            
        Returns:
            A/D Line values
        """
        try:
            if len(close) < 2:
                logger.warning("Insufficient data for A/D Line")
                return np.full_like(close, np.nan)
            
            if TALIB_AVAILABLE:
                try:
                    return talib.AD(high, low, close, volume)
                except Exception:
                    pass
            
            # Manual calculation
            # Money Flow Multiplier
            mfm = np.zeros_like(close)
            for i in range(len(close)):
                if high[i] != low[i]:
                    mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                else:
                    mfm[i] = 0.0
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # Cumulative sum = A/D Line
            ad_line = np.cumsum(mfv)
            
            return ad_line
            
        except Exception as e:
            logger.error(f"Error calculating A/D Line: {e}")
            return np.full_like(close, np.nan)
    
    def detect_divergences(
        self,
        price: np.ndarray,
        ad_line: np.ndarray,
        window: int = 20
    ) -> List[Dict[str, Any]]:
        """Detect A/D Line divergences"""
        divergences = []
        
        try:
            for i in range(window, len(price) - 1):
                prev_idx = i - window
                
                # Bullish divergence
                if price[i] < price[prev_idx] and ad_line[i] > ad_line[prev_idx]:
                    divergences.append({
                        'index': i,
                        'type': 'bullish_divergence',
                        'confidence': 0.80
                    })
                
                # Bearish divergence
                if price[i] > price[prev_idx] and ad_line[i] < ad_line[prev_idx]:
                    divergences.append({
                        'index': i,
                        'type': 'bearish_divergence',
                        'confidence': 0.80
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting A/D divergences: {e}")
            return divergences
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate A/D Line from DataFrame"""
        try:
            ad_line = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                df['volume'].values
            )
            
            df_copy = df.copy()
            df_copy['ad_line'] = ad_line
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating A/D Line from DataFrame: {e}")
            return df

# Convenience function
def accumulation_distribution_line(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """Calculate A/D Line (convenience function)"""
    indicator = AccumulationDistribution()
    return indicator.calculate(high, low, close, volume)

