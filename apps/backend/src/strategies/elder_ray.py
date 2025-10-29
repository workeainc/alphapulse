"""
Elder Ray Index for AlphaPulse
Bull Power and Bear Power indicators by Dr. Alexander Elder
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

class ElderRayIndex:
    """
    Elder Ray Index - Bull Power and Bear Power
    
    Measures the strength of buyers (bulls) and sellers (bears):
    - Bull Power = High - EMA(13)
    - Bear Power = Low - EMA(13)
    
    Interpretation:
    - Bull Power > 0 and rising = Bulls in control
    - Bear Power < 0 and falling = Bears in control
    - Bull Power > 0, Bear Power < 0 = Trend continuation
    - Divergences predict reversals
    
    Created by: Dr. Alexander Elder
    """
    
    def __init__(self, ema_period: int = 13):
        """
        Initialize Elder Ray
        
        Args:
            ema_period: EMA period for baseline (default: 13)
        """
        self.ema_period = ema_period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Elder Ray Index
        
        Args:
            high, low, close: Price arrays
            
        Returns:
            Tuple of (bull_power, bear_power, ema_baseline)
        """
        try:
            if len(close) < self.ema_period:
                logger.warning(f"Insufficient data for Elder Ray (need {self.ema_period}, got {len(close)})")
                empty = np.full_like(close, np.nan)
                return empty, empty, empty
            
            # Calculate EMA
            if TALIB_AVAILABLE:
                ema = talib.EMA(close, timeperiod=self.ema_period)
            else:
                ema = pd.Series(close).ewm(span=self.ema_period, adjust=False).mean().values
            
            # Calculate Bull and Bear Power
            bull_power = high - ema
            bear_power = low - ema
            
            return bull_power, bear_power, ema
            
        except Exception as e:
            logger.error(f"Error calculating Elder Ray: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty, empty
    
    def detect_divergences(
        self,
        price: np.ndarray,
        bull_power: np.ndarray,
        bear_power: np.ndarray,
        window: int = 20
    ) -> List[Dict[str, Any]]:
        """Detect divergences in Bull/Bear Power"""
        divergences = []
        
        try:
            for i in range(window, len(price) - 1):
                prev_idx = i - window
                
                # Bullish divergence: Price lower low, Bear Power higher low
                if (price[i] < price[prev_idx] and
                    bear_power[i] > bear_power[prev_idx] and
                    bear_power[i] < 0):
                    
                    divergences.append({
                        'index': i,
                        'type': 'bullish_divergence',
                        'indicator': 'bear_power',
                        'confidence': 0.80
                    })
                
                # Bearish divergence: Price higher high, Bull Power lower high
                if (price[i] > price[prev_idx] and
                    bull_power[i] < bull_power[prev_idx] and
                    bull_power[i] > 0):
                    
                    divergences.append({
                        'index': i,
                        'type': 'bearish_divergence',
                        'indicator': 'bull_power',
                        'confidence': 0.80
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting Elder Ray divergences: {e}")
            return divergences
    
    def get_signals(
        self,
        bull_power: np.ndarray,
        bear_power: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from Elder Ray"""
        signals = []
        
        try:
            for i in range(1, len(bull_power)):
                # Strong bullish: Bull Power > 0 and rising, Bear Power rising
                if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                    bear_power[i] > bear_power[i-1]):
                    
                    signals.append({
                        'index': i,
                        'type': 'strong_bullish',
                        'bull_power': bull_power[i],
                        'bear_power': bear_power[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # Strong bearish: Bear Power < 0 and falling, Bull Power falling
                elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                      bull_power[i] < bull_power[i-1]):
                    
                    signals.append({
                        'index': i,
                        'type': 'strong_bearish',
                        'bull_power': bull_power[i],
                        'bear_power': bear_power[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Elder Ray signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Elder Ray from DataFrame"""
        try:
            bull_power, bear_power, ema = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            df_copy = df.copy()
            df_copy['bull_power'] = bull_power
            df_copy['bear_power'] = bear_power
            df_copy[f'ema_{self.ema_period}'] = ema
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Elder Ray from DataFrame: {e}")
            return df

# Convenience function
def elder_ray(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Elder Ray (convenience function)"""
    indicator = ElderRayIndex(ema_period)
    bull_power, bear_power, _ = indicator.calculate(high, low, close)
    return bull_power, bear_power

