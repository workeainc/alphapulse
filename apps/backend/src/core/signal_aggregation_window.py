"""
Signal Aggregation Window
Prevents signal spam with cooldown periods
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalAggregationWindow:
    """
    Time-based cooldown management
    Prevents rapid-fire signal generation
    """
    
    def __init__(self):
        # Cooldown periods (in minutes)
        self.cooldown_periods = {
            'same_symbol': 60,      # 60 min between signals for same symbol
            'same_direction': 30,    # 30 min between same direction (prevent all LONG or SHORT)
            'total_system': 15       # 15 min between ANY signals
        }
        
        logger.info("Signal Aggregation Window initialized")
    
    def can_generate_signal(
        self,
        new_signal: Dict,
        recent_signals: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Check if signal can be generated based on cooldown periods
        
        Returns:
            (allowed, reason)
        """
        
        if not recent_signals:
            return True, "No recent signals - allowed"
        
        now = datetime.now()
        
        # === COOLDOWN 1: Same Symbol ===
        for sig in recent_signals:
            if sig['symbol'] == new_signal['symbol']:
                time_diff = (now - sig['created_at']).total_seconds() / 60
                
                if time_diff < self.cooldown_periods['same_symbol']:
                    logger.debug(f"{new_signal['symbol']}: Symbol cooldown - last signal {time_diff:.1f} min ago")
                    return False, f"Symbol cooldown: {time_diff:.1f} min ago (need {self.cooldown_periods['same_symbol']} min)"
                break
        
        # === COOLDOWN 2: Same Direction ===
        same_direction_signals = [s for s in recent_signals if s['direction'] == new_signal['direction']]
        
        if same_direction_signals:
            latest = same_direction_signals[0]
            time_diff = (now - latest['created_at']).total_seconds() / 60
            
            if time_diff < self.cooldown_periods['same_direction']:
                logger.debug(f"{new_signal['symbol']}: Direction cooldown - last {new_signal['direction']} signal {time_diff:.1f} min ago")
                return False, f"Direction cooldown: {time_diff:.1f} min ago"
        
        # === COOLDOWN 3: System-Wide ===
        if recent_signals:
            latest_any = recent_signals[0]
            time_diff = (now - latest_any['created_at']).total_seconds() / 60
            
            if time_diff < self.cooldown_periods['total_system']:
                logger.debug(f"System cooldown - last signal {time_diff:.1f} min ago")
                return False, f"System cooldown: {time_diff:.1f} min ago"
        
        return True, "All cooldown periods satisfied"

