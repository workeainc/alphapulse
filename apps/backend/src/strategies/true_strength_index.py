"""
True Strength Index (TSI) for AlphaPulse
Double-smoothed momentum indicator - superior divergence detection
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TrueStrengthIndex:
    """
    True Strength Index (TSI)
    
    Double-smoothed momentum indicator that filters noise better than RSI.
    Superior for divergence detection.
    
    Formula:
    1. Momentum = Close - Close[1]
    2. Double smooth momentum (2 EMAs)
    3. Double smooth absolute momentum (2 EMAs)
    4. TSI = 100 × (double_smoothed_momentum / double_smoothed_abs_momentum)
    5. Signal line = EMA of TSI
    
    Interpretation:
    - TSI > 0: Bullish momentum
    - TSI < 0: Bearish momentum
    - TSI crosses signal line: Trend change
    - Divergences: More reliable than RSI
    - Overbought: >+25
    - Oversold: <-25
    
    Created by: William Blau
    """
    
    def __init__(
        self,
        long_period: int = 25,
        short_period: int = 13,
        signal_period: int = 13
    ):
        """
        Initialize TSI
        
        Args:
            long_period: First smoothing period (default: 25)
            short_period: Second smoothing period (default: 13)
            signal_period: Signal line EMA period (default: 13)
        """
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period
    
    def calculate(
        self,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate True Strength Index
        
        Args:
            close: Close prices
            
        Returns:
            Tuple of (tsi, signal_line)
        """
        try:
            if len(close) < self.long_period + self.short_period:
                logger.warning(f"Insufficient data for TSI")
                empty = np.full_like(close, np.nan)
                return empty, empty
            
            # Calculate momentum
            momentum = pd.Series(close).diff()
            
            # Double smooth momentum
            smoothed1 = momentum.ewm(span=self.long_period, adjust=False).mean()
            smoothed2 = smoothed1.ewm(span=self.short_period, adjust=False).mean()
            
            # Double smooth absolute momentum
            abs_momentum = momentum.abs()
            abs_smoothed1 = abs_momentum.ewm(span=self.long_period, adjust=False).mean()
            abs_smoothed2 = abs_smoothed1.ewm(span=self.short_period, adjust=False).mean()
            
            # Calculate TSI
            tsi = 100 * (smoothed2 / abs_smoothed2)
            tsi = tsi.fillna(0).values
            
            # Calculate signal line
            signal_line = pd.Series(tsi).ewm(span=self.signal_period, adjust=False).mean().values
            
            return tsi, signal_line
            
        except Exception as e:
            logger.error(f"Error calculating TSI: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty
    
    def detect_divergences(
        self,
        price: np.ndarray,
        tsi: np.ndarray,
        window: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Detect TSI divergences (superior to RSI for this)
        
        Returns:
            List of divergence signals
        """
        divergences = []
        
        try:
            for i in range(window, len(price) - 1):
                prev_idx = i - window
                
                # Bullish divergence: Price lower low, TSI higher low
                if (price[i] < price[prev_idx] and
                    tsi[i] > tsi[prev_idx] and
                    tsi[i] < 0):  # TSI in negative territory
                    
                    divergences.append({
                        'index': i,
                        'type': 'bullish_divergence',
                        'price_low_1': price[prev_idx],
                        'price_low_2': price[i],
                        'tsi_1': tsi[prev_idx],
                        'tsi_2': tsi[i],
                        'confidence': 0.85  # TSI divergences are more reliable
                    })
                
                # Bearish divergence: Price higher high, TSI lower high
                if (price[i] > price[prev_idx] and
                    tsi[i] < tsi[prev_idx] and
                    tsi[i] > 0):  # TSI in positive territory
                    
                    divergences.append({
                        'index': i,
                        'type': 'bearish_divergence',
                        'price_high_1': price[prev_idx],
                        'price_high_2': price[i],
                        'tsi_1': tsi[prev_idx],
                        'tsi_2': tsi[i],
                        'confidence': 0.85
                    })
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting TSI divergences: {e}")
            return divergences
    
    def get_signals(
        self,
        tsi: np.ndarray,
        signal_line: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from TSI"""
        signals = []
        
        try:
            for i in range(1, len(tsi)):
                # TSI crosses above signal (bullish)
                if tsi[i] > signal_line[i] and tsi[i-1] <= signal_line[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bullish_crossover',
                        'tsi': tsi[i],
                        'signal': signal_line[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # TSI crosses below signal (bearish)
                elif tsi[i] < signal_line[i] and tsi[i-1] >= signal_line[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bearish_crossover',
                        'tsi': tsi[i],
                        'signal': signal_line[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
                
                # Overbought/oversold
                if tsi[i] > 25 and tsi[i-1] <= 25:
                    signals.append({
                        'index': i,
                        'type': 'overbought',
                        'tsi': tsi[i],
                        'direction': 'bearish',
                        'confidence': 0.65
                    })
                
                elif tsi[i] < -25 and tsi[i-1] >= -25:
                    signals.append({
                        'index': i,
                        'type': 'oversold',
                        'tsi': tsi[i],
                        'direction': 'bullish',
                        'confidence': 0.65
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating TSI signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate TSI from DataFrame"""
        try:
            tsi, signal = self.calculate(df['close'].values)
            
            df_copy = df.copy()
            df_copy['tsi'] = tsi
            df_copy['tsi_signal'] = signal
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating TSI from DataFrame: {e}")
            return df

# Convenience function
def true_strength_index(
    close: np.ndarray,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate TSI (convenience function)"""
    indicator = TrueStrengthIndex(long_period, short_period, signal_period)
    return indicator.calculate(close)

