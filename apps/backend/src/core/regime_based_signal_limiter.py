"""
Regime-Based Signal Limiter
Limits signal generation based on market regime
FEWER signals in unfavorable conditions
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class RegimeBasedSignalLimiter:
    """
    Adaptive signal limits based on market regime
    Reduces noise in poor trading conditions
    """
    
    def __init__(self):
        # Signal limits per regime
        self.regime_limits = {
            'TRENDING': {'max_signals': 2, 'min_confidence': 0.85},
            'RANGING': {'max_signals': 1, 'min_confidence': 0.90},
            'VOLATILE': {'max_signals': 1, 'min_confidence': 0.92},
            'BREAKOUT': {'max_signals': 3, 'min_confidence': 0.80},
            'LOW_VOLATILITY': {'max_signals': 2, 'min_confidence': 0.82}
        }
        
        logger.info("Regime-Based Signal Limiter initialized")
    
    def should_generate_signal(
        self,
        regime: Dict,
        existing_signals: List[Dict]
    ) -> Tuple[bool, float]:
        """
        Check if signal generation allowed based on regime
        
        Returns:
            (allowed, minimum_confidence_required)
        """
        
        regime_type = regime.get('regime_type', 'RANGING')
        
        # Get limits for this regime
        limits = self.regime_limits.get(regime_type, {'max_signals': 2, 'min_confidence': 0.85})
        
        max_signals = limits['max_signals']
        min_confidence = limits['min_confidence']
        
        # Check if limit reached
        if len(existing_signals) >= max_signals:
            logger.debug(f"Signal limit reached for {regime_type} regime: {len(existing_signals)}/{max_signals}")
            return False, min_confidence
        
        logger.debug(f"Signal generation allowed: {len(existing_signals)}/{max_signals} signals for {regime_type} regime")
        return True, min_confidence

