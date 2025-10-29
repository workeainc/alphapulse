"""
Chandelier Exit for AlphaPulse
ATR-based trailing stop for dynamic risk management
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

class ChandelierExit:
    """
    Chandelier Exit
    
    ATR-based trailing stop that "hangs" from the highest high (for longs)
    or lowest low (for shorts).
    
    Formula:
    Long Stop = Highest High(period) - ATR(period) × multiplier
    Short Stop = Lowest Low(period) + ATR(period) × multiplier
    
    Interpretation:
    - Price closes below Long Stop: Exit long position
    - Price closes above Short Stop: Exit short position
    - Rising Long Stop: Strengthening uptrend
    - Falling Short Stop: Strengthening downtrend
    
    Advantages:
    - Dynamic (adapts to volatility via ATR)
    - Follows market structure (uses highs/lows)
    - Prevents premature exits in trends
    
    Created by: Charles Le Beau
    """
    
    def __init__(self, period: int = 22, multiplier: float = 3.0):
        """
        Initialize Chandelier Exit
        
        Args:
            period: Period for ATR and high/low (default: 22)
            multiplier: ATR multiplier (default: 3.0)
        """
        self.period = period
        self.multiplier = multiplier
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Chandelier Exit
        
        Args:
            high, low, close: Price arrays
            
        Returns:
            Tuple of (long_stop, short_stop)
        """
        try:
            if len(close) < self.period:
                logger.warning(f"Insufficient data for Chandelier Exit")
                empty = np.full_like(close, np.nan)
                return empty, empty
            
            # Calculate ATR
            if TALIB_AVAILABLE:
                atr = talib.ATR(high, low, close, timeperiod=self.period)
            else:
                atr = self._calculate_atr_basic(high, low, close, self.period)
            
            # Highest high and lowest low
            highest_high = pd.Series(high).rolling(window=self.period).max().values
            lowest_low = pd.Series(low).rolling(window=self.period).min().values
            
            # Chandelier Exit levels
            long_stop = highest_high - (atr * self.multiplier)
            short_stop = lowest_low + (atr * self.multiplier)
            
            return long_stop, short_stop
            
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty
    
    def _calculate_atr_basic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Basic ATR calculation"""
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
        
        atr = pd.Series(tr_list).rolling(window=period).mean().values
        return atr
    
    def detect_exits(
        self,
        close: np.ndarray,
        long_stop: np.ndarray,
        short_stop: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect exit signals from Chandelier"""
        exits = []
        
        try:
            for i in range(1, len(close)):
                # Long exit signal
                if close[i] < long_stop[i] and close[i-1] >= long_stop[i-1]:
                    exits.append({
                        'index': i,
                        'type': 'long_exit',
                        'price': close[i],
                        'stop': long_stop[i],
                        'direction': 'exit_long',
                        'confidence': 0.80
                    })
                
                # Short exit signal
                if close[i] > short_stop[i] and close[i-1] <= short_stop[i-1]:
                    exits.append({
                        'index': i,
                        'type': 'short_exit',
                        'price': close[i],
                        'stop': short_stop[i],
                        'direction': 'exit_short',
                        'confidence': 0.80
                    })
            
            return exits
            
        except Exception as e:
            logger.error(f"Error detecting Chandelier exits: {e}")
            return exits
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Chandelier Exit from DataFrame"""
        try:
            long_stop, short_stop = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            df_copy = df.copy()
            df_copy['chandelier_long_stop'] = long_stop
            df_copy['chandelier_short_stop'] = short_stop
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Chandelier from DataFrame: {e}")
            return df

# Convenience function
def chandelier_exit(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 22,
    multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Chandelier Exit (convenience function)"""
    indicator = ChandelierExit(period, multiplier)
    return indicator.calculate(high, low, close)

