"""
Know Sure Thing (KST) Oscillator for AlphaPulse
Weighted sum of four ROC timeframes for comprehensive momentum
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class KnowSureThing:
    """
    Know Sure Thing (KST) Oscillator
    
    Momentum oscillator combining four different Rate of Change (ROC)
    calculations with different weights to identify cycle turns.
    
    Formula:
    ROC1 = ROC(Close, 10)
    ROC2 = ROC(Close, 15)
    ROC3 = ROC(Close, 20)
    ROC4 = ROC(Close, 30)
    
    RCMA1 = SMA(ROC1, 10)
    RCMA2 = SMA(ROC2, 10)
    RCMA3 = SMA(ROC3, 10)
    RCMA4 = SMA(ROC4, 15)
    
    KST = (RCMA1 × 1) + (RCMA2 × 2) + (RCMA3 × 3) + (RCMA4 × 4)
    Signal = SMA(KST, 9)
    
    Interpretation:
    - KST > Signal: Bullish
    - KST < Signal: Bearish
    - KST crosses zero: Trend change
    - Divergences: Reversal signals
    
    Created by: Martin Pring
    """
    
    def __init__(
        self,
        roc_periods: tuple = (10, 15, 20, 30),
        sma_periods: tuple = (10, 10, 10, 15),
        signal_period: int = 9
    ):
        """
        Initialize KST
        
        Args:
            roc_periods: ROC periods (default: 10, 15, 20, 30)
            sma_periods: SMA smoothing periods (default: 10, 10, 10, 15)
            signal_period: Signal line period (default: 9)
        """
        self.roc_periods = roc_periods
        self.sma_periods = sma_periods
        self.signal_period = signal_period
    
    def calculate(
        self,
        close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Know Sure Thing
        
        Args:
            close: Close prices
            
        Returns:
            Tuple of (kst, signal_line)
        """
        try:
            max_period = max(self.roc_periods) + max(self.sma_periods)
            if len(close) < max_period:
                logger.warning(f"Insufficient data for KST")
                empty = np.full_like(close, np.nan)
                return empty, empty
            
            close_series = pd.Series(close)
            
            # Calculate ROCs
            rocs = []
            for roc_period in self.roc_periods:
                roc = close_series.pct_change(periods=roc_period) * 100
                rocs.append(roc)
            
            # Smooth ROCs with SMAs
            rcmas = []
            for i, roc in enumerate(rocs):
                rcma = roc.rolling(window=self.sma_periods[i]).mean()
                rcmas.append(rcma)
            
            # Weighted sum (weights: 1, 2, 3, 4)
            kst = (rcmas[0] * 1 + rcmas[1] * 2 + rcmas[2] * 3 + rcmas[3] * 4).values
            
            # Signal line
            signal = pd.Series(kst).rolling(window=self.signal_period).mean().values
            
            return kst, signal
            
        except Exception as e:
            logger.error(f"Error calculating KST: {e}")
            empty = np.full_like(close, np.nan)
            return empty, empty
    
    def get_signals(
        self,
        kst: np.ndarray,
        signal: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate signals from KST"""
        signals = []
        
        try:
            for i in range(1, len(kst)):
                # KST crosses above signal (bullish)
                if kst[i] > signal[i] and kst[i-1] <= signal[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bullish_crossover',
                        'kst': kst[i],
                        'signal': signal[i],
                        'direction': 'bullish',
                        'confidence': 0.75
                    })
                
                # KST crosses below signal (bearish)
                elif kst[i] < signal[i] and kst[i-1] >= signal[i-1]:
                    signals.append({
                        'index': i,
                        'type': 'bearish_crossover',
                        'kst': kst[i],
                        'signal': signal[i],
                        'direction': 'bearish',
                        'confidence': 0.75
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating KST signals: {e}")
            return signals
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate KST from DataFrame"""
        try:
            kst, signal = self.calculate(df['close'].values)
            
            df_copy = df.copy()
            df_copy['kst'] = kst
            df_copy['kst_signal'] = signal
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error calculating KST from DataFrame: {e}")
            return df

# Convenience function
def know_sure_thing(
    close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate KST (convenience function)"""
    indicator = KnowSureThing()
    return indicator.calculate(close)

