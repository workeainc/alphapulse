"""
Adaptive Timeframe Selector
Dynamically selects optimal analysis/entry timeframe pairs based on market regime
"""

import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class AdaptiveTimeframeSelector:
    """
    Selects optimal timeframe pairs based on market conditions
    NOT hardcoded - fully adaptive to regime
    """
    
    def __init__(self):
        logger.info("Adaptive Timeframe Selector initialized")
    
    def select_timeframes(self, regime: Dict) -> Tuple[str, str, str]:
        """
        Select optimal (analysis_tf, entry_tf, scan_frequency)
        
        Returns:
            analysis_tf: Timeframe for SDE bias calculation
            entry_tf: Timeframe for entry scanning
            scan_freq: How often to scan for entries
        """
        
        strategy = regime.get('recommended_strategy', 'support_resistance')
        volatility = regime.get('volatility', 0.03)
        timeframe_pref = regime.get('timeframe_preference', 'medium')
        
        # === STRATEGY-BASED SELECTION ===
        
        if strategy == 'momentum_following':
            # Strong trend - ride momentum
            # Higher TF bias, hourly entries
            analysis_tf = '4h'
            entry_tf = '1h'
            scan_freq = '1h'  # Check every hour (quality over speed)
            
        elif strategy == 'support_resistance':
            # Ranging market - scalp at edges
            # Medium TF bias, frequent entries
            analysis_tf = '1h'
            entry_tf = '15m'
            scan_freq = '15m'  # Check every 15 min
            
        elif strategy == 'breakout_following':
            # Breakout - fast execution
            # Medium TF bias, fast entries
            analysis_tf = '1h'
            entry_tf = '5m'
            scan_freq = '5m'  # Check every 5 min
            
        elif strategy == 'mean_reversion':
            # Choppy - quick reversals
            # Short TF bias, very fast entries
            analysis_tf = '15m'
            entry_tf = '1m'
            scan_freq = '1m'  # Check every minute
            
        else:
            # Default balanced approach
            analysis_tf = '1h'
            entry_tf = '15m'
            scan_freq = '15m'
        
        # === VOLATILITY OVERRIDE ===
        # High volatility → shorter timeframes for control
        if volatility > 0.06:
            if analysis_tf == '4h':
                analysis_tf = '1h'
            if entry_tf == '1h':
                entry_tf = '15m'
            if scan_freq == '1h':
                scan_freq = '15m'
            
            logger.info(f"High volatility detected ({volatility:.3f}) - adjusted to shorter timeframes")
        
        # Low volatility → longer timeframes to avoid noise
        elif volatility < 0.02:
            if entry_tf == '1m':
                entry_tf = '5m'
            if entry_tf == '5m':
                entry_tf = '15m'
            if scan_freq == '1m':
                scan_freq = '5m'
            
            logger.info(f"Low volatility detected ({volatility:.3f}) - adjusted to longer timeframes")
        
        logger.debug(f"Selected TFs - Analysis: {analysis_tf}, Entry: {entry_tf}, Scan: {scan_freq}")
        
        return analysis_tf, entry_tf, scan_freq

