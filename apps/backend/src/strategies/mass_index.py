"""
Mass Index for AlphaPulse
Predicts trend reversals through range expansion analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MassIndex:
    """
    Mass Index
    
    Predicts trend reversals by identifying range expansion.
    Uses exponential moving averages of the high-low range.
    
    Formula:
    Single EMA = EMA(High - Low, 9)
    Double EMA = EMA(Single EMA, 9)
    Ratio = Single EMA / Double EMA
    Mass Index = sum(Ratio, 25)
    
    Interpretation:
    - Mass Index > 27: Reversal likely (bulge)
    - Mass Index < 26.5: Range contraction
    - Reversal setup: Bulge > 27, then drops below 26.5
    
    Best for: Reversal prediction in any trend
    
    Created by: Donald Dorsey
    """
    
    def __init__(self, ema_period: int = 9, sum_period: int = 25):
        """
        Initialize Mass Index
        
        Args:
            ema_period: EMA period (default: 9)
            sum_period: Summation period (default: 25)
        """
        self.ema_period = ema_period
        self.sum_period = sum_period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Mass Index
        
        Args:
            high, low: Price arrays
            
        Returns:
            Mass Index values
        """
        try:
            if len(high) < self.ema_period * 2 + self.sum_period:
                logger.warning(f"Insufficient data for Mass Index")
                return np.full_like(high, np.nan)
            
            # Calculate range
            range_hl = high - low
            
            # Single EMA of range
            single_ema = pd.Series(range_hl).ewm(span=self.ema_period, adjust=False).mean()
            
            # Double EMA
            double_ema = single_ema.ewm(span=self.ema_period, adjust=False).mean()
            
            # Ratio
            ratio = single_ema / double_ema
            ratio = ratio.fillna(1.0)
            
            # Sum of ratios
            mass_index = ratio.rolling(window=self.sum_period).sum().values
            
            return mass_index
            
        except Exception as e:
            logger.error(f"Error calculating Mass Index: {e}")
            return np.full_like(high, np.nan)
    
    def detect_reversals(
        self,
        mass_index: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect reversal signals from Mass Index"""
        reversals = []
        
        try:
            for i in range(1, len(mass_index)):
                # Reversal bulge: MI rises above 27
                if mass_index[i] > 27 and mass_index[i-1] <= 27:
                    reversals.append({
                        'index': i,
                        'type': 'reversal_bulge',
                        'mass_index': mass_index[i],
                        'signal': 'reversal_setup',
                        'confidence': 0.70
                    })
                
                # Reversal trigger: MI drops back below 26.5 after bulge
                elif mass_index[i] < 26.5 and mass_index[i-1] >= 26.5:
                    # Check if there was a recent bulge
                    recent_bulge = any(mass_index[max(0, i-10):i] > 27)
                    if recent_bulge:
                        reversals.append({
                            'index': i,
                            'type': 'reversal_trigger',
                            'mass_index': mass_index[i],
                            'signal': 'reversal_likely',
                            'confidence': 0.75
                        })
            
            return reversals
            
        except Exception as e:
            logger.error(f"Error detecting Mass Index reversals: {e}")
            return reversals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Mass Index from DataFrame"""
        try:
            mi = self.calculate(
                df['high'].values,
                df['low'].values
            )
            
            df_copy = df.copy()
            df_copy['mass_index'] = mi
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Mass Index from DataFrame: {e}")
            return df

# Convenience function
def mass_index(
    high: np.ndarray,
    low: np.ndarray,
    ema_period: int = 9,
    sum_period: int = 25
) -> np.ndarray:
    """Calculate Mass Index (convenience function)"""
    indicator = MassIndex(ema_period, sum_period)
    return indicator.calculate(high, low)

