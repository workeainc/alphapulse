"""
Awesome Oscillator for AlphaPulse
Bill Williams momentum indicator showing market momentum
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AwesomeOscillator:
    """
    Awesome Oscillator (AO)
    
    Momentum indicator created by Bill Williams.
    Shows the difference between 5-period and 34-period simple moving averages
    of the median price (High + Low) / 2.
    
    Formula:
    Median Price = (High + Low) / 2
    AO = SMA(Median, 5) - SMA(Median, 34)
    
    Interpretation:
    - AO > 0: Bullish momentum (histogram green)
    - AO < 0: Bearish momentum (histogram red)
    - AO increasing: Strengthening momentum
    - AO decreasing: Weakening momentum
    
    Patterns:
    - Twin Peaks: Two peaks with trough between (reversal)
    - Saucer: Three bars change direction (entry signal)
    - Zero-line cross: Trend change
    
    Created by: Bill Williams
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        """
        Initialize Awesome Oscillator
        
        Args:
            fast_period: Fast SMA period (default: 5)
            slow_period: Slow SMA period (default: 34)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Awesome Oscillator
        
        Args:
            high, low: Price arrays
            
        Returns:
            Awesome Oscillator values
        """
        try:
            if len(high) < self.slow_period:
                logger.warning(f"Insufficient data for AO (need {self.slow_period}, got {len(high)})")
                return np.full_like(high, np.nan)
            
            # Calculate median price
            median_price = (high + low) / 2
            
            # Calculate SMAs
            median_series = pd.Series(median_price)
            fast_sma = median_series.rolling(window=self.fast_period).mean()
            slow_sma = median_series.rolling(window=self.slow_period).mean()
            
            # AO = Fast SMA - Slow SMA
            ao = (fast_sma - slow_sma).values
            
            return ao
            
        except Exception as e:
            logger.error(f"Error calculating Awesome Oscillator: {e}")
            return np.full_like(high, np.nan)
    
    def detect_twin_peaks(
        self,
        ao: np.ndarray,
        window: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect Twin Peaks pattern
        
        Twin Peaks: Two peaks with trough between, second peak doesn't
        cross zero line but is higher than first = bullish reversal
        """
        patterns = []
        
        try:
            for i in range(window * 2, len(ao) - 1):
                # Look for two peaks pattern
                # This is simplified - full implementation would use peak detection
                
                # Bullish Twin Peaks: Both peaks below zero, second higher
                if (ao[i-window] < 0 and ao[i] < 0 and
                    ao[i] > ao[i-window] and
                    ao[i-int(window/2)] < min(ao[i-window], ao[i])):
                    
                    patterns.append({
                        'index': i,
                        'type': 'bullish_twin_peaks',
                        'direction': 'bullish',
                        'peak1': ao[i-window],
                        'peak2': ao[i],
                        'confidence': 0.80
                    })
                
                # Bearish Twin Peaks: Both peaks above zero, second lower
                elif (ao[i-window] > 0 and ao[i] > 0 and
                      ao[i] < ao[i-window] and
                      ao[i-int(window/2)] > max(ao[i-window], ao[i])):
                    
                    patterns.append({
                        'index': i,
                        'type': 'bearish_twin_peaks',
                        'direction': 'bearish',
                        'peak1': ao[i-window],
                        'peak2': ao[i],
                        'confidence': 0.80
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting twin peaks: {e}")
            return patterns
    
    def detect_saucer(
        self,
        ao: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect Saucer pattern
        
        Saucer: Three consecutive bars change direction
        Bullish saucer: All bars red, then first green bar
        Bearish saucer: All bars green, then first red bar
        """
        saucers = []
        
        try:
            for i in range(3, len(ao)):
                # Bullish saucer
                if (ao[i-3] < 0 and ao[i-2] < 0 and ao[i-1] < 0 and  # Three red bars
                    ao[i-2] < ao[i-3] and ao[i-1] < ao[i-2] and  # Decreasing (getting more negative)
                    ao[i] > ao[i-1]):  # First green bar (increasing)
                    
                    saucers.append({
                        'index': i,
                        'type': 'bullish_saucer',
                        'direction': 'bullish',
                        'ao_value': ao[i],
                        'confidence': 0.75
                    })
                
                # Bearish saucer
                elif (ao[i-3] > 0 and ao[i-2] > 0 and ao[i-1] > 0 and  # Three green bars
                      ao[i-2] > ao[i-3] and ao[i-1] > ao[i-2] and  # Increasing
                      ao[i] < ao[i-1]):  # First red bar (decreasing)
                    
                    saucers.append({
                        'index': i,
                        'type': 'bearish_saucer',
                        'direction': 'bearish',
                        'ao_value': ao[i],
                        'confidence': 0.75
                    })
            
            return saucers
            
        except Exception as e:
            logger.error(f"Error detecting saucer patterns: {e}")
            return saucers
    
    def get_signals(
        self,
        ao: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from Awesome Oscillator"""
        signals = []
        
        try:
            for i in range(1, len(ao)):
                # Zero-line crossovers
                if ao[i] > 0 and ao[i-1] <= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_up',
                        'ao_value': ao[i],
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                elif ao[i] < 0 and ao[i-1] >= 0:
                    signals.append({
                        'index': i,
                        'type': 'zero_cross_down',
                        'ao_value': ao[i],
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
                
                # Color change (momentum change)
                if ao[i] > ao[i-1] and ao[i-1] < ao[i-2]:  # Changed from red to green
                    signals.append({
                        'index': i,
                        'type': 'color_change_bullish',
                        'ao_value': ao[i],
                        'direction': 'bullish',
                        'confidence': 0.65
                    })
                
                elif ao[i] < ao[i-1] and ao[i-1] > ao[i-2]:  # Changed from green to red
                    signals.append({
                        'index': i,
                        'type': 'color_change_bearish',
                        'ao_value': ao[i],
                        'direction': 'bearish',
                        'confidence': 0.65
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating AO signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate AO from DataFrame"""
        try:
            ao = self.calculate(
                df['high'].values,
                df['low'].values
            )
            
            df_copy = df.copy()
            df_copy['awesome_oscillator'] = ao
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating AO from DataFrame: {e}")
            return df

# Convenience function
def awesome_oscillator(
    high: np.ndarray,
    low: np.ndarray,
    fast_period: int = 5,
    slow_period: int = 34
) -> np.ndarray:
    """Calculate Awesome Oscillator (convenience function)"""
    indicator = AwesomeOscillator(fast_period, slow_period)
    return indicator.calculate(high, low)

