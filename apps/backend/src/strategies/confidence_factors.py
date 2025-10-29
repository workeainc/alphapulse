import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class VolumePatternType(Enum):
    SPIKE = "spike"
    DIVERGENCE = "divergence"
    TREND_CONFIRMATION = "trend_confirmation"
    NO_CONFIRMATION = "no_confirmation"

class VolumeStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"

class TrendAlignment(Enum):
    STRONG_ALIGNMENT = "strong_alignment"
    WEAK_ALIGNMENT = "weak_alignment"
    COUNTER_TREND = "counter_trend"

@dataclass
class VolumeConfirmation:
    pattern_type: VolumePatternType
    strength: VolumeStrength
    factor: float
    description: str

@dataclass
class TrendConfirmation:
    alignment: TrendAlignment
    factor: float
    description: str

class ConfidenceFactors:
    """
    Advanced confidence factor calculations for pattern recognition
    """
    
    def __init__(self):
        # Volume confirmation factors
        self.volume_factors = {
            VolumePatternType.SPIKE: {
                VolumeStrength.STRONG: 1.20,
                VolumeStrength.MODERATE: 1.10,
                VolumeStrength.WEAK: 1.05
            },
            VolumePatternType.DIVERGENCE: {
                VolumeStrength.STRONG: 1.15,
                VolumeStrength.MODERATE: 1.08,
                VolumeStrength.WEAK: 1.03
            },
            VolumePatternType.TREND_CONFIRMATION: {
                VolumeStrength.STRONG: 1.12,
                VolumeStrength.MODERATE: 1.06,
                VolumeStrength.WEAK: 1.02
            },
            VolumePatternType.NO_CONFIRMATION: {
                VolumeStrength.STRONG: 0.85,
                VolumeStrength.MODERATE: 0.90,
                VolumeStrength.WEAK: 0.95
            }
        }
        
        # Trend alignment factors
        self.trend_factors = {
            TrendAlignment.STRONG_ALIGNMENT: 1.25,
            TrendAlignment.WEAK_ALIGNMENT: 1.10,
            TrendAlignment.COUNTER_TREND: 0.80
        }
        
        # Multi-timeframe confirmation factors
        self.timeframe_factors = {
            'higher_timeframe_confirmed': 1.15,
            'same_timeframe_confirmed': 1.00,
            'lower_timeframe_confirmed': 0.95,
            'no_multi_timeframe_confirmation': 0.90
        }
    
    def calculate_volume_confirmation(self, df: pd.DataFrame, pattern_type: str) -> VolumeConfirmation:
        """
        Calculate volume confirmation factor based on volume patterns
        
        Args:
            df: DataFrame with OHLCV data
            pattern_type: Type of pattern detected ("bullish", "bearish")
            
        Returns:
            VolumeConfirmation object with factor and description
        """
        if len(df) < 20:
            return VolumeConfirmation(
                pattern_type=VolumePatternType.NO_CONFIRMATION,
                strength=VolumeStrength.WEAK,
                factor=0.95,
                description="Insufficient data for volume analysis"
            )
        
        current_volume = df.iloc[-1]['volume']
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike detection
        if volume_ratio > 2.0:
            strength = VolumeStrength.STRONG
            pattern_type_enum = VolumePatternType.SPIKE
            description = f"Strong volume spike ({volume_ratio:.1f}x average)"
        elif volume_ratio > 1.5:
            strength = VolumeStrength.MODERATE
            pattern_type_enum = VolumePatternType.SPIKE
            description = f"Moderate volume spike ({volume_ratio:.1f}x average)"
        elif volume_ratio > 1.2:
            strength = VolumeStrength.WEAK
            pattern_type_enum = VolumePatternType.SPIKE
            description = f"Weak volume spike ({volume_ratio:.1f}x average)"
        
        # Volume divergence detection
        elif self._detect_volume_divergence(df, pattern_type):
            if volume_ratio > 1.3:
                strength = VolumeStrength.STRONG
                pattern_type_enum = VolumePatternType.DIVERGENCE
                description = f"Strong volume divergence ({volume_ratio:.1f}x average)"
            elif volume_ratio > 1.1:
                strength = VolumeStrength.MODERATE
                pattern_type_enum = VolumePatternType.DIVERGENCE
                description = f"Moderate volume divergence ({volume_ratio:.1f}x average)"
            else:
                strength = VolumeStrength.WEAK
                pattern_type_enum = VolumePatternType.DIVERGENCE
                description = f"Weak volume divergence ({volume_ratio:.1f}x average)"
        
        # Trend confirmation
        elif self._detect_trend_confirmation(df, pattern_type):
            if volume_ratio > 1.2:
                strength = VolumeStrength.STRONG
                pattern_type_enum = VolumePatternType.TREND_CONFIRMATION
                description = f"Strong trend confirmation ({volume_ratio:.1f}x average)"
            elif volume_ratio > 1.0:
                strength = VolumeStrength.MODERATE
                pattern_type_enum = VolumePatternType.TREND_CONFIRMATION
                description = f"Moderate trend confirmation ({volume_ratio:.1f}x average)"
            else:
                strength = VolumeStrength.WEAK
                pattern_type_enum = VolumePatternType.TREND_CONFIRMATION
                description = f"Weak trend confirmation ({volume_ratio:.1f}x average)"
        
        # No confirmation
        else:
            if volume_ratio < 0.7:
                strength = VolumeStrength.STRONG
                pattern_type_enum = VolumePatternType.NO_CONFIRMATION
                description = f"Strong volume absence ({volume_ratio:.1f}x average)"
            elif volume_ratio < 0.9:
                strength = VolumeStrength.MODERATE
                pattern_type_enum = VolumePatternType.NO_CONFIRMATION
                description = f"Moderate volume absence ({volume_ratio:.1f}x average)"
            else:
                strength = VolumeStrength.WEAK
                pattern_type_enum = VolumePatternType.NO_CONFIRMATION
                description = f"Weak volume absence ({volume_ratio:.1f}x average)"
        
        factor = self.volume_factors[pattern_type_enum][strength]
        
        return VolumeConfirmation(
            pattern_type=pattern_type_enum,
            strength=strength,
            factor=factor,
            description=description
        )
    
    def _detect_volume_divergence(self, df: pd.DataFrame, pattern_type: str) -> bool:
        """Detect volume divergence from price action"""
        if len(df) < 10:
            return False
        
        # Calculate price and volume trends
        recent_prices = df['close'].tail(10)
        recent_volumes = df['volume'].tail(10)
        
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        volume_trend = (recent_volumes.iloc[-1] - recent_volumes.iloc[0]) / recent_volumes.iloc[0]
        
        # Bullish pattern with decreasing volume or bearish pattern with increasing volume
        if pattern_type == "bullish":
            return price_trend > 0.01 and volume_trend < -0.1  # Price up, volume down
        else:  # bearish
            return price_trend < -0.01 and volume_trend > 0.1  # Price down, volume up
    
    def _detect_trend_confirmation(self, df: pd.DataFrame, pattern_type: str) -> bool:
        """Detect if volume confirms the trend direction"""
        if len(df) < 10:
            return False
        
        # Calculate price and volume trends
        recent_prices = df['close'].tail(10)
        recent_volumes = df['volume'].tail(10)
        
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        volume_trend = (recent_volumes.iloc[-1] - recent_volumes.iloc[0]) / recent_volumes.iloc[0]
        
        # Both price and volume moving in same direction
        if pattern_type == "bullish":
            return price_trend > 0.01 and volume_trend > 0.1  # Both up
        else:  # bearish
            return price_trend < -0.01 and volume_trend < -0.1  # Both down
    
    def calculate_trend_alignment(self, df: pd.DataFrame, pattern_type: str) -> TrendConfirmation:
        """
        Calculate trend alignment factor based on pattern direction vs market trend
        
        Args:
            df: DataFrame with OHLCV data
            pattern_type: Type of pattern detected ("bullish", "bearish")
            
        Returns:
            TrendConfirmation object with factor and description
        """
        if len(df) < 50:
            return TrendConfirmation(
                alignment=TrendAlignment.WEAK_ALIGNMENT,
                factor=1.00,
                description="Insufficient data for trend analysis"
            )
        
        # Calculate multiple trend indicators
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Calculate trend strength
        price_vs_ema20 = (current_price - ema_20) / ema_20 if ema_20 > 0 else 0
        price_vs_ema50 = (current_price - ema_50) / ema_50 if ema_50 > 0 else 0
        ema_alignment = (ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0
        
        # Determine trend direction
        if pattern_type == "bullish":
            # Bullish pattern
            if price_vs_ema20 > 0.02 and price_vs_ema50 > 0.02 and ema_alignment > 0.01:
                alignment = TrendAlignment.STRONG_ALIGNMENT
                description = "Strong bullish alignment with uptrend"
            elif price_vs_ema20 > 0.01 or price_vs_ema50 > 0.01:
                alignment = TrendAlignment.WEAK_ALIGNMENT
                description = "Weak bullish alignment with trend"
            else:
                alignment = TrendAlignment.COUNTER_TREND
                description = "Bullish pattern against downtrend"
        else:
            # Bearish pattern
            if price_vs_ema20 < -0.02 and price_vs_ema50 < -0.02 and ema_alignment < -0.01:
                alignment = TrendAlignment.STRONG_ALIGNMENT
                description = "Strong bearish alignment with downtrend"
            elif price_vs_ema20 < -0.01 or price_vs_ema50 < -0.01:
                alignment = TrendAlignment.WEAK_ALIGNMENT
                description = "Weak bearish alignment with trend"
            else:
                alignment = TrendAlignment.COUNTER_TREND
                description = "Bearish pattern against uptrend"
        
        factor = self.trend_factors[alignment]
        
        return TrendConfirmation(
            alignment=alignment,
            factor=factor,
            description=description
        )
    
    def calculate_multi_timeframe_confirmation(
        self, 
        current_timeframe: str, 
        higher_timeframe_data: Optional[pd.DataFrame] = None,
        lower_timeframe_data: Optional[pd.DataFrame] = None,
        pattern_type: str = "bullish"
    ) -> Tuple[float, str]:
        """
        Calculate multi-timeframe confirmation factor
        
        Args:
            current_timeframe: Current timeframe (e.g., "1h", "4h")
            higher_timeframe_data: Data from higher timeframe
            lower_timeframe_data: Data from lower timeframe
            pattern_type: Type of pattern detected
            
        Returns:
            Tuple of (factor, description)
        """
        confirmations = []
        
        # Check higher timeframe confirmation
        if higher_timeframe_data is not None and len(higher_timeframe_data) > 20:
            higher_trend = self._get_timeframe_trend(higher_timeframe_data, pattern_type)
            if higher_trend:
                confirmations.append("higher_timeframe_confirmed")
        
        # Check lower timeframe confirmation
        if lower_timeframe_data is not None and len(lower_timeframe_data) > 20:
            lower_trend = self._get_timeframe_trend(lower_timeframe_data, pattern_type)
            if lower_trend:
                confirmations.append("lower_timeframe_confirmed")
        
        # Determine factor
        if "higher_timeframe_confirmed" in confirmations:
            factor = self.timeframe_factors['higher_timeframe_confirmed']
            description = "Confirmed on higher timeframe"
        elif "lower_timeframe_confirmed" in confirmations:
            factor = self.timeframe_factors['lower_timeframe_confirmed']
            description = "Confirmed on lower timeframe"
        elif len(confirmations) > 0:
            factor = self.timeframe_factors['same_timeframe_confirmed']
            description = "Multi-timeframe confirmation"
        else:
            factor = self.timeframe_factors['no_multi_timeframe_confirmation']
            description = "No multi-timeframe confirmation"
        
        return factor, description
    
    def _get_timeframe_trend(self, df: pd.DataFrame, pattern_type: str) -> bool:
        """Get trend direction for a specific timeframe"""
        if len(df) < 20:
            return False
        
        # Simple trend detection using EMA
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if pattern_type == "bullish":
            return current_price > ema_20 * 1.01  # 1% above EMA
        else:
            return current_price < ema_20 * 0.99  # 1% below EMA
    
    def calculate_final_confidence(
        self,
        base_score: float,
        historical_success_factor: float,
        volume_confirmation: VolumeConfirmation,
        trend_confirmation: TrendConfirmation,
        multi_timeframe_factor: float = 1.0
    ) -> Dict:
        """
        Calculate final confidence score using multi-factor formula
        
        Args:
            base_score: Raw detection strength (0-1)
            historical_success_factor: Historical performance factor
            volume_confirmation: Volume confirmation object
            trend_confirmation: Trend confirmation object
            multi_timeframe_factor: Multi-timeframe confirmation factor
            
        Returns:
            Dictionary with final confidence and breakdown
        """
        # Multi-factor confidence formula
        final_confidence = (
            base_score *
            historical_success_factor *
            volume_confirmation.factor *
            trend_confirmation.factor *
            multi_timeframe_factor
        )
        
        # Clamp to maximum 1.0
        final_confidence = min(final_confidence, 1.0)
        
        # Calculate confidence level
        if final_confidence >= 0.8:
            confidence_level = "Very High"
        elif final_confidence >= 0.6:
            confidence_level = "High"
        elif final_confidence >= 0.4:
            confidence_level = "Medium"
        elif final_confidence >= 0.2:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        return {
            'final_confidence': final_confidence,
            'confidence_level': confidence_level,
            'breakdown': {
                'base_score': base_score,
                'historical_success_factor': historical_success_factor,
                'volume_factor': volume_confirmation.factor,
                'trend_factor': trend_confirmation.factor,
                'multi_timeframe_factor': multi_timeframe_factor
            },
            'descriptions': {
                'volume': volume_confirmation.description,
                'trend': trend_confirmation.description
            }
        }
    
    def get_confidence_summary(self, confidence_result: Dict) -> str:
        """Generate a human-readable confidence summary"""
        breakdown = confidence_result['breakdown']
        descriptions = confidence_result['descriptions']
        
        summary = f"""
🎯 Confidence Analysis Summary:
   Final Confidence: {confidence_result['final_confidence']:.3f} ({confidence_result['confidence_level']})
   
📊 Factor Breakdown:
   • Base Score: {breakdown['base_score']:.3f}
   • Historical Success: {breakdown['historical_success_factor']:.3f}
   • Volume Confirmation: {breakdown['volume_factor']:.3f} ({descriptions['volume']})
   • Trend Alignment: {breakdown['trend_factor']:.3f} ({descriptions['trend']})
   • Multi-Timeframe: {breakdown['multi_timeframe_factor']:.3f}
        """
        
        return summary.strip()
