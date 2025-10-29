"""
Signal Generator for Multi-Timeframe Testing
Generate sample timeframe signals for testing the fusion system
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .multi_timeframe_fusion import (
    Timeframe, TimeframeSignal, SignalDirection, SignalStrength, multi_timeframe_fusion
)

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generate sample timeframe signals for testing multi-timeframe fusion
    """
    
    def __init__(self):
        self.pattern_types = [
            "bullish_engulfing", "bearish_engulfing", "hammer", "shooting_star",
            "morning_star", "evening_star", "doji", "spinning_top",
            "breakout", "breakdown", "support_bounce", "resistance_rejection"
        ]
        
        logger.info("SignalGenerator initialized")
    
    def generate_bullish_scenario(self, symbol: str, current_price: float) -> Dict[Timeframe, TimeframeSignal]:
        """Generate a bullish scenario across all timeframes"""
        signals = {}
        
        # 1-minute: Weak bullish (noise)
        signals[Timeframe.M1] = TimeframeSignal(
            timeframe=Timeframe.M1,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.WEAK,
            confidence=0.4,
            pattern_type="bullish_engulfing",
            price_level=current_price * 1.001,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "weak"}
        )
        
        # 5-minute: Moderate bullish
        signals[Timeframe.M5] = TimeframeSignal(
            timeframe=Timeframe.M5,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            confidence=0.6,
            pattern_type="hammer",
            price_level=current_price * 1.002,
            timestamp=datetime.now(),
            metadata={"volume": "medium", "momentum": "building"}
        )
        
        # 15-minute: Strong bullish
        signals[Timeframe.M15] = TimeframeSignal(
            timeframe=Timeframe.M15,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            pattern_type="breakout",
            price_level=current_price * 1.005,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        # 1-hour: Very strong bullish
        signals[Timeframe.H1] = TimeframeSignal(
            timeframe=Timeframe.H1,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.VERY_STRONG,
            confidence=0.9,
            pattern_type="morning_star",
            price_level=current_price * 1.008,
            timestamp=datetime.now(),
            metadata={"volume": "very_high", "momentum": "very_strong"}
        )
        
        # 4-hour: Strong bullish
        signals[Timeframe.H4] = TimeframeSignal(
            timeframe=Timeframe.H4,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            pattern_type="support_bounce",
            price_level=current_price * 1.010,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        # 1-day: Moderate bullish
        signals[Timeframe.D1] = TimeframeSignal(
            timeframe=Timeframe.D1,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            confidence=0.7,
            pattern_type="bullish_engulfing",
            price_level=current_price * 1.015,
            timestamp=datetime.now(),
            metadata={"volume": "medium", "momentum": "moderate"}
        )
        
        return signals
    
    def generate_bearish_scenario(self, symbol: str, current_price: float) -> Dict[Timeframe, TimeframeSignal]:
        """Generate a bearish scenario across all timeframes"""
        signals = {}
        
        # 1-minute: Weak bearish (noise)
        signals[Timeframe.M1] = TimeframeSignal(
            timeframe=Timeframe.M1,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.WEAK,
            confidence=0.4,
            pattern_type="bearish_engulfing",
            price_level=current_price * 0.999,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "weak"}
        )
        
        # 5-minute: Moderate bearish
        signals[Timeframe.M5] = TimeframeSignal(
            timeframe=Timeframe.M5,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            confidence=0.6,
            pattern_type="shooting_star",
            price_level=current_price * 0.998,
            timestamp=datetime.now(),
            metadata={"volume": "medium", "momentum": "building"}
        )
        
        # 15-minute: Strong bearish
        signals[Timeframe.M15] = TimeframeSignal(
            timeframe=Timeframe.M15,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            pattern_type="breakdown",
            price_level=current_price * 0.995,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        # 1-hour: Very strong bearish
        signals[Timeframe.H1] = TimeframeSignal(
            timeframe=Timeframe.H1,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.VERY_STRONG,
            confidence=0.9,
            pattern_type="evening_star",
            price_level=current_price * 0.992,
            timestamp=datetime.now(),
            metadata={"volume": "very_high", "momentum": "very_strong"}
        )
        
        # 4-hour: Strong bearish
        signals[Timeframe.H4] = TimeframeSignal(
            timeframe=Timeframe.H4,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            pattern_type="resistance_rejection",
            price_level=current_price * 0.990,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        # 1-day: Moderate bearish
        signals[Timeframe.D1] = TimeframeSignal(
            timeframe=Timeframe.D1,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            confidence=0.7,
            pattern_type="bearish_engulfing",
            price_level=current_price * 0.985,
            timestamp=datetime.now(),
            metadata={"volume": "medium", "momentum": "moderate"}
        )
        
        return signals
    
    def generate_mixed_scenario(self, symbol: str, current_price: float) -> Dict[Timeframe, TimeframeSignal]:
        """Generate a mixed scenario with conflicting signals"""
        signals = {}
        
        # Short-term bullish
        signals[Timeframe.M1] = TimeframeSignal(
            timeframe=Timeframe.M1,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            confidence=0.6,
            pattern_type="hammer",
            price_level=current_price * 1.002,
            timestamp=datetime.now(),
            metadata={"volume": "medium", "momentum": "building"}
        )
        
        signals[Timeframe.M5] = TimeframeSignal(
            timeframe=Timeframe.M5,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            pattern_type="breakout",
            price_level=current_price * 1.005,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        # Medium-term neutral
        signals[Timeframe.M15] = TimeframeSignal(
            timeframe=Timeframe.M15,
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.3,
            pattern_type="doji",
            price_level=current_price,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "none"}
        )
        
        signals[Timeframe.H1] = TimeframeSignal(
            timeframe=Timeframe.H1,
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.4,
            pattern_type="spinning_top",
            price_level=current_price,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "none"}
        )
        
        # Long-term bearish
        signals[Timeframe.H4] = TimeframeSignal(
            timeframe=Timeframe.H4,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.MODERATE,
            confidence=0.6,
            pattern_type="shooting_star",
            price_level=current_price * 0.998,
            timestamp=datetime.now(),
            metadata={"volume": "medium", "momentum": "building"}
        )
        
        signals[Timeframe.D1] = TimeframeSignal(
            timeframe=Timeframe.D1,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            pattern_type="evening_star",
            price_level=current_price * 0.995,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        return signals
    
    def generate_volatile_scenario(self, symbol: str, current_price: float) -> Dict[Timeframe, TimeframeSignal]:
        """Generate a volatile scenario with strong short-term signals"""
        signals = {}
        
        # Very strong short-term signals
        signals[Timeframe.M1] = TimeframeSignal(
            timeframe=Timeframe.M1,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.VERY_STRONG,
            confidence=0.9,
            pattern_type="breakout",
            price_level=current_price * 1.010,
            timestamp=datetime.now(),
            metadata={"volume": "very_high", "momentum": "very_strong"}
        )
        
        signals[Timeframe.M5] = TimeframeSignal(
            timeframe=Timeframe.M5,
            direction=SignalDirection.BEARISH,
            strength=SignalStrength.VERY_STRONG,
            confidence=0.9,
            pattern_type="breakdown",
            price_level=current_price * 0.990,
            timestamp=datetime.now(),
            metadata={"volume": "very_high", "momentum": "very_strong"}
        )
        
        signals[Timeframe.M15] = TimeframeSignal(
            timeframe=Timeframe.M15,
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            pattern_type="hammer",
            price_level=current_price * 1.005,
            timestamp=datetime.now(),
            metadata={"volume": "high", "momentum": "strong"}
        )
        
        # Weaker long-term signals
        signals[Timeframe.H1] = TimeframeSignal(
            timeframe=Timeframe.H1,
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.3,
            pattern_type="doji",
            price_level=current_price,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "none"}
        )
        
        signals[Timeframe.H4] = TimeframeSignal(
            timeframe=Timeframe.H4,
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.4,
            pattern_type="spinning_top",
            price_level=current_price,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "none"}
        )
        
        signals[Timeframe.D1] = TimeframeSignal(
            timeframe=Timeframe.D1,
            direction=SignalDirection.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.3,
            pattern_type="doji",
            price_level=current_price,
            timestamp=datetime.now(),
            metadata={"volume": "low", "momentum": "none"}
        )
        
        return signals
    
    def generate_random_scenario(self, symbol: str, current_price: float) -> Dict[Timeframe, TimeframeSignal]:
        """Generate a random scenario for testing"""
        signals = {}
        timeframes = list(Timeframe)
        
        for tf in timeframes:
            # Random direction
            direction = np.random.choice(list(SignalDirection))
            
            # Random strength
            strength = np.random.choice(list(SignalStrength))
            
            # Random confidence (0.3 to 0.9)
            confidence = np.random.uniform(0.3, 0.9)
            
            # Random pattern
            pattern = np.random.choice(self.pattern_types)
            
            # Random price level
            price_offset = np.random.uniform(-0.02, 0.02)  # ±2%
            price_level = current_price * (1 + price_offset)
            
            signals[tf] = TimeframeSignal(
                timeframe=tf,
                direction=direction,
                strength=strength,
                confidence=confidence,
                pattern_type=pattern,
                price_level=price_level,
                timestamp=datetime.now(),
                metadata={
                    "volume": np.random.choice(["low", "medium", "high", "very_high"]),
                    "momentum": np.random.choice(["weak", "building", "strong", "very_strong", "none"])
                }
            )
        
        return signals
    
    def generate_scenario_by_type(self, 
                                 scenario_type: str, 
                                 symbol: str, 
                                 current_price: float) -> Dict[Timeframe, TimeframeSignal]:
        """Generate scenario based on type"""
        if scenario_type == "bullish":
            return self.generate_bullish_scenario(symbol, current_price)
        elif scenario_type == "bearish":
            return self.generate_bearish_scenario(symbol, current_price)
        elif scenario_type == "mixed":
            return self.generate_mixed_scenario(symbol, current_price)
        elif scenario_type == "volatile":
            return self.generate_volatile_scenario(symbol, current_price)
        elif scenario_type == "random":
            return self.generate_random_scenario(symbol, current_price)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

# Global signal generator instance
signal_generator = SignalGenerator()
