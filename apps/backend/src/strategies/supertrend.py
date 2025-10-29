"""
Supertrend Indicator for AlphaPulse
ATR-based trend following indicator - extremely popular in crypto trading
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import TA-Lib for ATR calculation
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using basic ATR implementation")

class SupertrendIndicator:
    """
    Supertrend Indicator
    
    Trend following indicator using ATR for volatility adjustment:
    - Green line below price = Bullish trend
    - Red line above price = Bearish trend
    - Trend change when price crosses the line
    
    Popular configurations:
    - 10/3: Standard (period=10, multiplier=3)
    - 7/3: Sensitive
    - 10/2: Less volatile
    - 11/2: Conservative
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Initialize Supertrend
        
        Args:
            period: ATR period (default: 10)
            multiplier: ATR multiplier for bands (default: 3.0)
        """
        self.period = period
        self.multiplier = multiplier
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Supertrend indicator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Tuple of (supertrend values, trend direction, bands)
            - supertrend: The supertrend line values
            - direction: 1 for uptrend, -1 for downtrend
            - bands: The basic bands before trend logic
        """
        try:
            if len(high) < self.period:
                logger.warning(f"Insufficient data for Supertrend (need {self.period}, got {len(high)})")
                empty = np.full_like(close, np.nan)
                return empty, empty, empty
            
            # Calculate ATR
            atr = self._calculate_atr(high, low, close, self.period)
            
            # Calculate basic bands
            hl_avg = (high + low) / 2
            upper_band = hl_avg + (self.multiplier * atr)
            lower_band = hl_avg - (self.multiplier * atr)
            
            # Initialize arrays
            supertrend = np.zeros_like(close)
            direction = np.zeros_like(close)
            
            # First value
            if close[0] <= upper_band[0]:
                supertrend[0] = upper_band[0]
                direction[0] = -1  # Downtrend
            else:
                supertrend[0] = lower_band[0]
                direction[0] = 1  # Uptrend
            
            # Calculate supertrend
            for i in range(1, len(close)):
                # Adjust bands
                if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                    final_lower_band = lower_band[i]
                else:
                    final_lower_band = lower_band[i-1]
                
                if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                    final_upper_band = upper_band[i]
                else:
                    final_upper_band = upper_band[i-1]
                
                # Determine trend
                if direction[i-1] == 1:  # Was in uptrend
                    if close[i] <= final_lower_band:
                        # Trend changes to down
                        supertrend[i] = final_upper_band
                        direction[i] = -1
                    else:
                        # Continue uptrend
                        supertrend[i] = final_lower_band
                        direction[i] = 1
                else:  # Was in downtrend
                    if close[i] >= final_upper_band:
                        # Trend changes to up
                        supertrend[i] = final_lower_band
                        direction[i] = 1
                    else:
                        # Continue downtrend
                        supertrend[i] = final_upper_band
                        direction[i] = -1
            
            return supertrend, direction, (upper_band + lower_band) / 2
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty, empty
    
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Average True Range"""
        if TALIB_AVAILABLE:
            try:
                return talib.ATR(high, low, close, timeperiod=period)
            except Exception as e:
                logger.error(f"Error calculating ATR with TA-Lib: {e}")
        
        # Fallback: Basic ATR calculation
        tr_list = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr = max(hl, hc, lc)
            tr_list.append(tr)
        
        # Simple moving average of TR
        tr_array = np.array(tr_list)
        atr = pd.Series(tr_array).rolling(window=period).mean().values
        
        return atr
    
    def get_signals(
        self,
        close: np.ndarray,
        supertrend: np.ndarray,
        direction: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate trading signals from Supertrend
        
        Args:
            close: Close prices
            supertrend: Supertrend values
            direction: Trend direction
            
        Returns:
            DataFrame with signals
        """
        signals = []
        
        for i in range(1, len(direction)):
            # Trend change signals
            if direction[i] != direction[i-1]:
                if direction[i] == 1:
                    signals.append({
                        'index': i,
                        'type': 'trend_change',
                        'direction': 'bullish',
                        'price': close[i],
                        'supertrend': supertrend[i],
                        'confidence': 0.75
                    })
                else:
                    signals.append({
                        'index': i,
                        'type': 'trend_change',
                        'direction': 'bearish',
                        'price': close[i],
                        'supertrend': supertrend[i],
                        'confidence': 0.75
                    })
        
        return pd.DataFrame(signals)
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Supertrend from DataFrame
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            DataFrame with added Supertrend columns
        """
        try:
            supertrend, direction, bands = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            df_copy = df.copy()
            df_copy['supertrend'] = supertrend
            df_copy['supertrend_direction'] = direction
            df_copy['supertrend_bands'] = bands
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend from DataFrame: {e}")
            return df

# Convenience function
def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Supertrend (convenience function)
    
    Args:
        high, low, close: Price arrays
        period: ATR period
        multiplier: ATR multiplier
        
    Returns:
        Tuple of (supertrend values, direction)
    """
    indicator = SupertrendIndicator(period, multiplier)
    st, direction, _ = indicator.calculate(high, low, close)
    return st, direction

