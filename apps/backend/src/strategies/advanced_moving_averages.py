"""
Advanced Moving Averages for AlphaPulse
DEMA, TEMA, ZLEMA - Low-lag moving averages for better trend following
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

class AdvancedMovingAverages:
    """
    Advanced Moving Average Suite
    
    Includes:
    - DEMA (Double Exponential MA) - Less lag than EMA
    - TEMA (Triple Exponential MA) - Even less lag
    - ZLEMA (Zero Lag EMA) - Minimal lag
    
    All designed to reduce lag while maintaining smoothness.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Advanced MAs
        
        Args:
            period: MA period (default: 20)
        """
        self.period = period
    
    def calculate_dema(
        self,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Double Exponential Moving Average (DEMA)
        
        Formula:
        DEMA = 2 × EMA - EMA(EMA)
        
        Reduces lag by about 50% compared to EMA
        """
        try:
            if len(close) < self.period * 2:
                logger.warning(f"Insufficient data for DEMA")
                return np.full_like(close, np.nan)
            
            if TALIB_AVAILABLE:
                try:
                    return talib.DEMA(close, timeperiod=self.period)
                except Exception:
                    pass
            
            # Calculate using pandas
            close_series = pd.Series(close)
            ema1 = close_series.ewm(span=self.period, adjust=False).mean()
            ema2 = ema1.ewm(span=self.period, adjust=False).mean()
            
            dema = (2 * ema1 - ema2).values
            
            return dema
            
        except Exception as e:
            logger.error(f"Error calculating DEMA: {e}")
            return np.full_like(close, np.nan)
    
    def calculate_tema(
        self,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Triple Exponential Moving Average (TEMA)
        
        Formula:
        TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
        
        Reduces lag even more than DEMA
        """
        try:
            if len(close) < self.period * 3:
                logger.warning(f"Insufficient data for TEMA")
                return np.full_like(close, np.nan)
            
            if TALIB_AVAILABLE:
                try:
                    return talib.TEMA(close, timeperiod=self.period)
                except Exception:
                    pass
            
            # Calculate using pandas
            close_series = pd.Series(close)
            ema1 = close_series.ewm(span=self.period, adjust=False).mean()
            ema2 = ema1.ewm(span=self.period, adjust=False).mean()
            ema3 = ema2.ewm(span=self.period, adjust=False).mean()
            
            tema = (3 * ema1 - 3 * ema2 + ema3).values
            
            return tema
            
        except Exception as e:
            logger.error(f"Error calculating TEMA: {e}")
            return np.full_like(close, np.nan)
    
    def calculate_zlema(
        self,
        close: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Zero Lag Exponential Moving Average (ZLEMA)
        
        Formula:
        Lag = (Period - 1) / 2
        ZLEMA = EMA(Close + (Close - Close[Lag]))
        
        Attempts to eliminate lag by using price momentum
        """
        try:
            if len(close) < self.period:
                logger.warning(f"Insufficient data for ZLEMA")
                return np.full_like(close, np.nan)
            
            lag = int((self.period - 1) / 2)
            
            # Create lag-adjusted price series
            adjusted_close = np.zeros_like(close)
            for i in range(len(close)):
                if i >= lag:
                    adjusted_close[i] = close[i] + (close[i] - close[i - lag])
                else:
                    adjusted_close[i] = close[i]
            
            # Calculate EMA on adjusted close
            zlema = pd.Series(adjusted_close).ewm(span=self.period, adjust=False).mean().values
            
            return zlema
            
        except Exception as e:
            logger.error(f"Error calculating ZLEMA: {e}")
            return np.full_like(close, np.nan)
    
    def calculate_all(
        self,
        close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all advanced MAs
        
        Returns:
            Dict with 'dema', 'tema', 'zlema' keys
        """
        return {
            'dema': self.calculate_dema(close),
            'tema': self.calculate_tema(close),
            'zlema': self.calculate_zlema(close)
        }
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate all advanced MAs from DataFrame"""
        try:
            mas = self.calculate_all(df['close'].values)
            
            df_copy = df.copy()
            df_copy[f'dema_{self.period}'] = mas['dema']
            df_copy[f'tema_{self.period}'] = mas['tema']
            df_copy[f'zlema_{self.period}'] = mas['zlema']
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating advanced MAs from DataFrame: {e}")
            return df

# Convenience functions
def dema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate DEMA"""
    ma = AdvancedMovingAverages(period)
    return ma.calculate_dema(close)

def tema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate TEMA"""
    ma = AdvancedMovingAverages(period)
    return ma.calculate_tema(close)

def zlema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate ZLEMA"""
    ma = AdvancedMovingAverages(period)
    return ma.calculate_zlema(close)

