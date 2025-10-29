"""
Vortex Indicator for AlphaPulse
Identifies trend starts and trend strength
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VortexIndicator:
    """
    Vortex Indicator (VI)
    
    Identifies the start of trends and measures trend strength.
    
    Formula:
    Positive Vortex Movement = |High - Previous Low|
    Negative Vortex Movement = |Low - Previous High|
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    
    VI+ = sum(Positive VM, period) / sum(True Range, period)
    VI- = sum(Negative VM, period) / sum(True Range, period)
    
    Interpretation:
    - VI+ > VI-: Uptrend
    - VI- > VI+: Downtrend
    - VI+ crosses above VI-: Bullish signal
    - VI- crosses above VI+: Bearish signal
    
    Created by: Etienne Botes and Douglas Siepman
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize Vortex
        
        Args:
            period: Period for calculation (default: 14)
        """
        self.period = period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Vortex Indicator
        
        Args:
            high, low, close: Price arrays
            
        Returns:
            Tuple of (vi_plus, vi_minus)
        """
        try:
            if len(close) < self.period + 1:
                logger.warning(f"Insufficient data for Vortex")
                empty = np.full_like(close, np.nan)
                return empty, empty
            
            # Calculate vortex movements
            vm_plus = np.abs(high[1:] - low[:-1])
            vm_minus = np.abs(low[1:] - high[:-1])
            
            # Prepend NaN for alignment
            vm_plus = np.concatenate([[np.nan], vm_plus])
            vm_minus = np.concatenate([[np.nan], vm_minus])
            
            # Calculate true range
            tr = np.zeros_like(close)
            for i in range(1, len(close)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            # Rolling sums
            vm_plus_sum = pd.Series(vm_plus).rolling(window=self.period).sum()
            vm_minus_sum = pd.Series(vm_minus).rolling(window=self.period).sum()
            tr_sum = pd.Series(tr).rolling(window=self.period).sum()
            
            # VI+ and VI-
            vi_plus = (vm_plus_sum / tr_sum).values
            vi_minus = (vm_minus_sum / tr_sum).values
            
            return vi_plus, vi_minus
            
        except Exception as e:
            logger.error(f"Error calculating Vortex: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty
    
    def get_signals(
        self,
        vi_plus: np.ndarray,
        vi_minus: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from Vortex"""
        signals = []
        
        try:
            for i in range(1, len(vi_plus)):
                # VI+ crosses above VI- (bullish)
                if vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bullish_crossover',
                        'vi_plus': vi_plus[i],
                        'vi_minus': vi_minus[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # VI- crosses above VI+ (bearish)
                elif vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bearish_crossover',
                        'vi_plus': vi_plus[i],
                        'vi_minus': vi_minus[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Vortex signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Vortex from DataFrame"""
        try:
            vi_plus, vi_minus = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            df_copy = df.copy()
            df_copy['vi_plus'] = vi_plus
            df_copy['vi_minus'] = vi_minus
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Vortex from DataFrame: {e}")
            return df

# Convenience function
def vortex_indicator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Vortex (convenience function)"""
    indicator = VortexIndicator(period)
    return indicator.calculate(high, low, close)

