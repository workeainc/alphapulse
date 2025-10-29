"""
Donchian Channels for AlphaPulse
Classic breakout indicator used by Turtle Traders
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DonchianChannels:
    """
    Donchian Channels
    
    Breakout indicator showing highest high and lowest low over N periods:
    - Upper Channel: Highest high over N periods
    - Lower Channel: Lowest low over N periods
    - Middle Channel: (Upper + Lower) / 2
    
    Trading Rules:
    - Price breaks above upper channel = Bullish breakout
    - Price breaks below lower channel = Bearish breakdown
    - Middle channel acts as dynamic support/resistance
    
    Popular configurations:
    - 20-period: Standard
    - 55-period: Turtle Traders (long entry)
    - 20-period: Turtle Traders (short exit)
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Donchian Channels
        
        Args:
            period: Lookback period (default: 20)
        """
        self.period = period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Donchian Channels
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        try:
            if len(high) < self.period:
                logger.warning(f"Insufficient data for Donchian (need {self.period}, got {len(high)})")
                empty = np.full_like(high, np.nan)
                return empty, empty, empty
            
            # Calculate channels using pandas rolling
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            
            upper_channel = high_series.rolling(window=self.period).max().values
            lower_channel = low_series.rolling(window=self.period).min().values
            middle_channel = (upper_channel + lower_channel) / 2
            
            return upper_channel, middle_channel, lower_channel
            
        except Exception as e:
            logger.error(f"Error calculating Donchian Channels: {e}")
            empty = np.full_like(high, np.nan)
            return empty, empty, empty
    
    def detect_breakouts(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        upper_channel: np.ndarray,
        lower_channel: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect breakouts from Donchian Channels
        
        Returns:
            List of breakout signals
        """
        breakouts = []
        
        try:
            for i in range(1, len(close)):
                # Bullish breakout (close above upper channel)
                if close[i] > upper_channel[i-1] and close[i-1] <= upper_channel[i-1]:
                    breakouts.append({
                        'index': i,
                        'type': 'bullish_breakout',
                        'direction': 'bullish',
                        'price': close[i],
                        'channel': upper_channel[i-1],
                        'confidence': 0.80
                    })
                
                # Bearish breakdown (close below lower channel)
                elif close[i] < lower_channel[i-1] and close[i-1] >= lower_channel[i-1]:
                    breakouts.append({
                        'index': i,
                        'type': 'bearish_breakdown',
                        'direction': 'bearish',
                        'price': close[i],
                        'channel': lower_channel[i-1],
                        'confidence': 0.80
                    })
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Error detecting Donchian breakouts: {e}")
            return breakouts
    
    def calculate_channel_width(
        self,
        upper_channel: np.ndarray,
        lower_channel: np.ndarray
    ) -> np.ndarray:
        """
        Calculate channel width (volatility measure)
        
        Returns:
            Channel width as percentage
        """
        try:
            width = ((upper_channel - lower_channel) / lower_channel) * 100
            return np.where(np.isnan(width), 0.0, width)
        except Exception as e:
            logger.error(f"Error calculating channel width: {e}")
            return np.full_like(upper_channel, np.nan)
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Donchian Channels from DataFrame
        
        Args:
            df: DataFrame with 'high', 'low' columns
            
        Returns:
            DataFrame with added Donchian columns
        """
        try:
            upper, middle, lower = self.calculate(
                df['high'].values,
                df['low'].values
            )
            
            df_copy = df.copy()
            df_copy['donchian_upper'] = upper
            df_copy['donchian_middle'] = middle
            df_copy['donchian_lower'] = lower
            df_copy['donchian_width'] = self.calculate_channel_width(upper, lower)
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating Donchian from DataFrame: {e}")
            return df

# Convenience function
def donchian_channels(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Donchian Channels (convenience function)
    
    Returns:
        Tuple of (upper, middle, lower)
    """
    indicator = DonchianChannels(period)
    return indicator.calculate(high, low)

