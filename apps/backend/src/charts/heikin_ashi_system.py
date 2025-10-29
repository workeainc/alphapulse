"""
Heikin Ashi Chart System for AlphaPulse
Noise-reduced candlesticks for clearer trend visualization
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HeikinAshiSystem:
    """
    Heikin Ashi Candlesticks
    
    Japanese candlestick technique that filters noise and makes trends clearer.
    
    Formula:
    HA_Close = (Open + High + Low + Close) / 4
    HA_Open = (Previous HA_Open + Previous HA_Close) / 2
    HA_High = max(High, HA_Open, HA_Close)
    HA_Low = min(Low, HA_Open, HA_Close)
    
    Advantages:
    - Smoother price action
    - Clearer trends (less noise)
    - Better for trend following
    - Reduces whipsaws
    
    Patterns:
    - Consecutive green candles = Strong uptrend
    - Consecutive red candles = Strong downtrend
    - Small body + long wicks = Consolidation
    - First green after reds = Trend reversal
    
    Popular in: Crypto, Forex, Trend following
    """
    
    def __init__(self):
        """Initialize Heikin Ashi System"""
        pass
    
    def transform(
        self,
        open_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform standard OHLC to Heikin Ashi OHLC
        
        Args:
            open_prices, high, low, close: Standard OHLC arrays
            
        Returns:
            Tuple of (ha_open, ha_high, ha_low, ha_close)
        """
        try:
            if len(close) < 2:
                logger.warning("Insufficient data for Heikin Ashi")
                return open_prices, high, low, close
            
            ha_close = (open_prices + high + low + close) / 4
            ha_open = np.zeros_like(open_prices)
            
            # First candle
            ha_open[0] = (open_prices[0] + close[0]) / 2
            
            # Subsequent candles
            for i in range(1, len(open_prices)):
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            
            # HA High and Low
            ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
            ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
            
            return ha_open, ha_high, ha_low, ha_close
            
        except Exception as e:
            logger.error(f"Error transforming to Heikin Ashi: {e}")
            return open_prices, high, low, close
    
    def detect_trends(
        self,
        ha_open: np.ndarray,
        ha_close: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect trends from Heikin Ashi candles"""
        trends = []
        
        try:
            consecutive_green = 0
            consecutive_red = 0
            
            for i in range(len(ha_close)):
                # Green candle
                if ha_close[i] > ha_open[i]:
                    consecutive_green += 1
                    consecutive_red = 0
                    
                    # Strong uptrend: 3+ consecutive green
                    if consecutive_green == 3:
                        trends.append({
                            'index': i,
                            'type': 'strong_uptrend_start',
                            'direction': 'bullish',
                            'consecutive': consecutive_green,
                            'confidence': 0.80
                        })
                
                # Red candle
                else:
                    consecutive_red += 1
                    consecutive_green = 0
                    
                    # Strong downtrend: 3+ consecutive red
                    if consecutive_red == 3:
                        trends.append({
                            'index': i,
                            'type': 'strong_downtrend_start',
                            'direction': 'bearish',
                            'consecutive': consecutive_red,
                            'confidence': 0.80
                        })
            
            return trends
            
        except Exception as e:
            logger.error(f"Error detecting HA trends: {e}")
            return trends
    
    def detect_reversals(
        self,
        ha_open: np.ndarray,
        ha_close: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect trend reversals from Heikin Ashi"""
        reversals = []
        
        try:
            for i in range(1, len(ha_close)):
                # Bullish reversal: First green after red
                if (ha_close[i] > ha_open[i] and ha_close[i-1] < ha_open[i-1]):
                    reversals.append({
                        'index': i,
                        'type': 'bullish_reversal',
                        'direction': 'bullish',
                        'confidence': 0.70
                    })
                
                # Bearish reversal: First red after green
                elif (ha_close[i] < ha_open[i] and ha_close[i-1] > ha_open[i-1]):
                    reversals.append({
                        'index': i,
                        'type': 'bearish_reversal',
                        'direction': 'bearish',
                        'confidence': 0.70
                    })
            
            return reversals
            
        except Exception as e:
            logger.error(f"Error detecting HA reversals: {e}")
            return reversals
    
    def transform_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform DataFrame to Heikin Ashi"""
        try:
            ha_open, ha_high, ha_low, ha_close = self.transform(
                df['open'].values,
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            df_copy = df.copy()
            df_copy['ha_open'] = ha_open
            df_copy['ha_high'] = ha_high
            df_copy['ha_low'] = ha_low
            df_copy['ha_close'] = ha_close
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error transforming DataFrame to HA: {e}")
            return df

# Convenience function
def heikin_ashi(
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform to Heikin Ashi (convenience function)"""
    system = HeikinAshiSystem()
    return system.transform(open_prices, high, low, close)

