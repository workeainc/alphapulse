"""
Force Index for AlphaPulse
Combines price change and volume to measure trend strength
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ForceIndex:
    """
    Force Index
    
    Measures the force behind price movements by combining:
    - Price change
    - Volume
    
    Formula:
    Force Index(1) = (Close - Close[1]) × Volume
    Force Index(13) = EMA(Force Index(1), 13)
    
    Interpretation:
    - Positive Force = Bullish (buyers in control)
    - Negative Force = Bearish (sellers in control)
    - Rising Force = Strengthening trend
    - Divergences = Reversal signals
    
    Created by: Dr. Alexander Elder
    """
    
    def __init__(self, ema_period: int = 13):
        """
        Initialize Force Index
        
        Args:
            ema_period: EMA smoothing period (default: 13)
        """
        self.ema_period = ema_period
    
    def calculate(
        self,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Force Index
        
        Args:
            close: Close prices
            volume: Volume
            
        Returns:
            Force Index values
        """
        try:
            if len(close) < 2:
                logger.warning("Insufficient data for Force Index")
                return np.full_like(close, np.nan)
            
            # Calculate raw force
            price_change = np.diff(close, prepend=close[0])
            raw_force = price_change * volume
            
            # Smooth with EMA
            if self.ema_period > 1:
                force_index = pd.Series(raw_force).ewm(span=self.ema_period, adjust=False).mean().values
            else:
                force_index = raw_force
            
            return force_index
            
        except Exception as e:
            logger.error(f"Error calculating Force Index: {e}")
            return np.full_like(close, np.nan)
    
    def detect_divergences(
        self,
        price: np.ndarray,
        force_index: np.ndarray,
        window: int = 20
    ) -> List[Dict[str, Any]]:
        """Detect Force Index divergences"""
        divergences = []
        
        try:
            for i in range(window, len(price) - 1):
                prev_idx = i - window
                
                # Bullish divergence
                if price[i] < price[prev_idx] and force_index[i] > force_index[prev_idx]:
                    divergences.append({
                        'index': i,
                        'type': 'bullish_divergence',
                        'confidence': 0.80
                    })
                
                # Bearish divergence
                if price[i] > price[prev_idx] and force_index[i] < force_index[prev_idx]:
                    divergences.append({
                        'index': i,
                        'type': 'bearish_divergence',
                        'confidence': 0.80
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting Force Index divergences: {e}")
            return divergences
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Force Index from DataFrame"""
        try:
            force = self.calculate(
                df['close'].values,
                df['volume'].values
            )
            
            df_copy = df.copy()
            df_copy[f'force_index_{self.ema_period}'] = force
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Force Index from DataFrame: {e}")
            return df

# Convenience function
def force_index(
    close: np.ndarray,
    volume: np.ndarray,
    ema_period: int = 13
) -> np.ndarray:
    """Calculate Force Index (convenience function)"""
    indicator = ForceIndex(ema_period)
    return indicator.calculate(close, volume)

