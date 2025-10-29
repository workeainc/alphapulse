"""
Hull Moving Average (HMA) for AlphaPulse
Low-lag, smooth trend indicator
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HullMovingAverage:
    """
    Hull Moving Average (HMA)
    
    Developed by Alan Hull to reduce lag while maintaining smoothness.
    
    Formula:
    1. Half WMA = WMA(price, period/2)
    2. Full WMA = WMA(price, period)
    3. Raw HMA = 2 × Half WMA - Full WMA
    4. HMA = WMA(Raw HMA, sqrt(period))
    
    Advantages:
    - Much less lag than SMA/EMA
    - Smooth without whipsaws
    - Excellent for trend identification
    
    Popular periods: 9, 16, 20, 50, 200
    
    Created by: Alan Hull
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize HMA
        
        Args:
            period: HMA period (default: 20)
        """
        self.period = period
    
    def calculate(
        self,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Hull Moving Average
        
        Args:
            close: Close prices
            
        Returns:
            HMA values
        """
        try:
            if len(close) < self.period:
                logger.warning(f"Insufficient data for HMA (need {self.period}, got {len(close)})")
                return np.full_like(close, np.nan)
            
            half_period = int(self.period / 2)
            sqrt_period = int(np.sqrt(self.period))
            
            # Calculate WMAs
            wma_half = self._weighted_moving_average(close, half_period)
            wma_full = self._weighted_moving_average(close, self.period)
            
            # Raw HMA
            raw_hma = 2 * wma_half - wma_full
            
            # Final HMA (WMA of raw HMA)
            hma = self._weighted_moving_average(raw_hma, sqrt_period)
            
            return hma
            
        except Exception as e:
            logger.error(f"Error calculating HMA: {e}")
            return np.full_like(close, np.nan)
    
    def _weighted_moving_average(
        self,
        data: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Calculate Weighted Moving Average
        
        WMA gives more weight to recent data:
        weights = [1, 2, 3, ..., period]
        """
        try:
            weights = np.arange(1, period + 1)
            
            wma = np.full_like(data, np.nan, dtype=float)
            
            for i in range(period - 1, len(data)):
                window = data[i - period + 1:i + 1]
                wma[i] = np.dot(window, weights) / weights.sum()
            
            return wma
            
        except Exception as e:
            logger.error(f"Error calculating WMA: {e}")
            return np.full_like(data, np.nan)
    
    def get_signals(
        self,
        close: np.ndarray,
        hma: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from HMA"""
        signals = []
        
        try:
            for i in range(1, len(hma)):
                # Price crosses above HMA (bullish)
                if close[i] > hma[i] and close[i-1] <= hma[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'cross_above',
                        'price': close[i],
                        'hma': hma[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # Price crosses below HMA (bearish)
                elif close[i] < hma[i] and close[i-1] >= hma[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'cross_below',
                        'price': close[i],
                        'hma': hma[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
                
                # HMA slope change (trend strength)
                if i >= 2:
                    if hma[i] > hma[i-1] and hma[i-1] < hma[i-2]:
                        signals.append({
                            'index': i,
                            'type': 'slope_bullish',
                            'direction': 'bullish',
                            'confidence': 0.65
                        })
                    elif hma[i] < hma[i-1] and hma[i-1] > hma[i-2]:
                        signals.append({
                            'index': i,
                            'type': 'slope_bearish',
                            'direction': 'bearish',
                            'confidence': 0.65
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating HMA signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate HMA from DataFrame"""
        try:
            hma = self.calculate(df['close'].values)
            
            df_copy = df.copy()
            df_copy[f'hma_{self.period}'] = hma
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating HMA from DataFrame: {e}")
            return df

# Convenience function
def hull_moving_average(
    close: np.ndarray,
    period: int = 20
) -> np.ndarray:
    """Calculate Hull MA (convenience function)"""
    indicator = HullMovingAverage(period)
    return indicator.calculate(close)

