"""
Chaikin Money Flow (CMF) for AlphaPulse
Volume-weighted accumulation/distribution indicator
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChaikinMoneyFlow:
    """
    Chaikin Money Flow (CMF)
    
    Measures buying and selling pressure over a period:
    - CMF > 0: Buying pressure (accumulation)
    - CMF < 0: Selling pressure (distribution)
    - CMF > 0.25: Strong buying pressure
    - CMF < -0.25: Strong selling pressure
    
    Formula:
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = MFM × Volume
    CMF = Sum(MFV, period) / Sum(Volume, period)
    """
    
    def __init__(self, period: int = 21):
        """
        Initialize Chaikin Money Flow
        
        Args:
            period: Period for CMF calculation (default: 21)
        """
        self.period = period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Chaikin Money Flow
        
        Args:
            high, low, close: Price arrays
            volume: Volume array
            
        Returns:
            CMF values
        """
        try:
            if len(close) < self.period:
                logger.warning(f"Insufficient data for CMF (need {self.period}, got {len(close)})")
                return np.full_like(close, np.nan)
            
            # Calculate Money Flow Multiplier
            mfm = np.zeros_like(close)
            for i in range(len(close)):
                if high[i] != low[i]:
                    mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                else:
                    mfm[i] = 0.0
            
            # Calculate Money Flow Volume
            mfv = mfm * volume
            
            # Calculate CMF (rolling sum)
            mfv_series = pd.Series(mfv)
            volume_series = pd.Series(volume)
            
            mfv_sum = mfv_series.rolling(window=self.period).sum()
            volume_sum = volume_series.rolling(window=self.period).sum()
            
            cmf = (mfv_sum / volume_sum).values
            cmf = np.where(np.isnan(cmf), 0.0, cmf)
            
            return cmf
            
        except Exception as e:
            logger.error(f"Error calculating CMF: {e}")
            return np.full_like(close, np.nan)
    
    def detect_divergences(
        self,
        price: np.ndarray,
        cmf: np.ndarray,
        window: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Detect CMF divergences from price
        
        Returns:
            List of divergence signals
        """
        divergences = []
        
        try:
            if len(price) < window * 2:
                return divergences
            
            # Find price peaks and troughs
            for i in range(window, len(price) - window):
                # Check for bullish divergence (price lower low, CMF higher low)
                if i >= window * 2:
                    prev_low_idx = i - window
                    
                    if (price[i] < price[prev_low_idx] and
                        cmf[i] > cmf[prev_low_idx] and
                        cmf[i] < 0):  # CMF in negative territory
                        
                        divergences.append({
                            'index': i,
                            'type': 'bullish_divergence',
                            'price_low_1': price[prev_low_idx],
                            'price_low_2': price[i],
                            'cmf_1': cmf[prev_low_idx],
                            'cmf_2': cmf[i],
                            'confidence': 0.80
                        })
                    
                    # Check for bearish divergence (price higher high, CMF lower high)
                    if (price[i] > price[prev_low_idx] and
                        cmf[i] < cmf[prev_low_idx] and
                        cmf[i] > 0):  # CMF in positive territory
                        
                        divergences.append({
                            'index': i,
                            'type': 'bearish_divergence',
                            'price_high_1': price[prev_low_idx],
                            'price_high_2': price[i],
                            'cmf_1': cmf[prev_low_idx],
                            'cmf_2': cmf[i],
                            'confidence': 0.80
                        })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting CMF divergences: {e}")
            return divergences
    
    def get_signals(
        self,
        cmf: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals from CMF
        
        Returns:
            List of signals
        """
        signals = []
        
        try:
            for i in range(1, len(cmf)):
                # Strong buying pressure
                if cmf[i] > 0.25 and cmf[i-1] <= 0.25:
                    signals.append({
                        'index': i,
                        'type': 'strong_buying',
                        'cmf_value': cmf[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # Strong selling pressure
                elif cmf[i] < -0.25 and cmf[i-1] >= -0.25:
                    signals.append({
                        'index': i,
                        'type': 'strong_selling',
                        'cmf_value': cmf[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
                
                # Zero-line crossovers
                elif cmf[i] > 0 and cmf[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_up',
                        'cmf_value': cmf[i],
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                elif cmf[i] < 0 and cmf[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_down',
                        'cmf_value': cmf[i],
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating CMF signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate CMF from DataFrame
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added CMF column
        """
        try:
            cmf = self.calculate(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                df['volume'].values
            )
            
            df_copy = df.copy()
            df_copy['cmf'] = cmf
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating CMF from DataFrame: {e}")
            return df

# Convenience function
def chaikin_money_flow(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 21
) -> np.ndarray:
    """
    Calculate Chaikin Money Flow (convenience function)
    
    Args:
        high, low, close: Price arrays
        volume: Volume array
        period: CMF period
        
    Returns:
        CMF values
    """
    indicator = ChaikinMoneyFlow(period)
    return indicator.calculate(high, low, close, volume)

