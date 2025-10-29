"""
Ease of Movement (EMV) for AlphaPulse
Shows how easily price moves relative to volume
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EaseOfMovement:
    """
    Ease of Movement (EMV)
    
    Shows relationship between price change and volume.
    Identifies when price moves easily vs with difficulty.
    
    Formula:
    Distance Moved = ((High + Low) / 2) - ((Previous High + Previous Low) / 2)
    Box Ratio = (Volume / scale) / (High - Low)
    EMV = Distance Moved / Box Ratio
    EMV_MA = SMA(EMV, period)
    
    Interpretation:
    - EMV > 0: Price moving up easily (bullish)
    - EMV < 0: Price moving down easily (bearish)
    - High EMV: Price moving easily (low volume resistance)
    - Low EMV: Price moving with difficulty (high volume resistance)
    
    Created by: Richard W. Arms Jr.
    """
    
    def __init__(self, period: int = 14, scale: float = 100000000):
        """
        Initialize EMV
        
        Args:
            period: Smoothing period (default: 14)
            scale: Volume scaling factor (default: 100000000)
        """
        self.period = period
        self.scale = scale
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Ease of Movement
        
        Args:
            high, low: Price arrays
            volume: Volume array
            
        Returns:
            EMV values (smoothed)
        """
        try:
            if len(high) < 2:
                logger.warning("Insufficient data for EMV")
                return np.full_like(high, np.nan)
            
            # Distance moved
            midpoint = (high + low) / 2
            distance_moved = np.diff(midpoint, prepend=midpoint[0])
            
            # Box ratio
            box_ratio = (volume / self.scale) / (high - low)
            box_ratio = np.where(box_ratio == 0, np.nan, box_ratio)  # Avoid division by zero
            
            # EMV (raw)
            emv_raw = distance_moved / box_ratio
            
            # Smooth with SMA
            emv = pd.Series(emv_raw).rolling(window=self.period).mean().values
            
            return emv
            
        except Exception as e:
            logger.error(f"Error calculating EMV: {e}")
            return np.full_like(high, np.nan)
    
    def get_signals(
        self,
        emv: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from EMV"""
        signals = []
        
        try:
            for i in range(1, len(emv)):
                # EMV crosses above zero (bullish)
                if emv[i] > 0 and emv[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_up',
                        'emv': emv[i],
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                # EMV crosses below zero (bearish)
                elif emv[i] < 0 and emv[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_down',
                        'emv': emv[i],
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating EMV signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate EMV from DataFrame"""
        try:
            emv = self.calculate(
                df['high'].values,
                df['low'].values,
                df['volume'].values
            )
            
            df_copy = df.copy()
            df_copy['emv'] = emv
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating EMV from DataFrame: {e}")
            return df

# Convenience function
def ease_of_movement(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """Calculate EMV (convenience function)"""
    indicator = EaseOfMovement(period)
    return indicator.calculate(high, low, volume)

