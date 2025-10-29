#!/usr/bin/env python3
"""
Trend Alignment Enhancer for AlphaPulse
Phase 1: Multi-Timeframe Trend Confirmation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction enumeration"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    WEAKENING = "weakening"
    STRENGTHENING = "strengthening"

class TrendStrength(Enum):
    """Trend strength levels"""
    WEAK = "weak"      # ADX < 20
    MODERATE = "moderate"  # ADX 20-25
    STRONG = "strong"   # ADX > 25
    EXTREME = "extreme"  # ADX > 40

@dataclass
class TrendContext:
    """Complete trend context information"""
    direction: TrendDirection
    strength: TrendStrength
    adx_value: float
    ema_alignment: bool
    ema_20: float
    ema_50: float
    ema_200: float
    rsi: float
    macd_signal: str
    confidence: float
    description: str

@dataclass
class MTFTrendResult:
    """Multi-timeframe trend confirmation result"""
    current_tf_trend: TrendContext
    higher_tf_trend: Optional[TrendContext] = None
    lower_tf_trend: Optional[TrendContext] = None
    trend_alignment: bool = False
    confidence_multiplier: float = 1.0
    alignment_description: str = ""

class TrendAlignmentEnhancer:
    """
    Enhanced trend alignment system with multi-timeframe confirmation
    """
    
    def __init__(self):
        # Trend strength thresholds
        self.adx_thresholds = {
            TrendStrength.WEAK: 20,
            TrendStrength.MODERATE: 25,
            TrendStrength.STRONG: 30,
            TrendStrength.EXTREME: 40
        }
        
        # Confidence multipliers for trend alignment
        self.trend_multipliers = {
            'continuation_bullish': 1.2,    # Bullish pattern in bullish trend
            'continuation_bearish': 1.2,    # Bearish pattern in bearish trend
            'reversal_weakening': 1.1,      # Reversal pattern in weakening trend
            'counter_trend': 0.7,           # Pattern against trend
            'neutral_trend': 1.0            # Neutral trend
        }
        
        # Pattern classifications
        self.continuation_patterns = [
            'flag', 'pennant', 'bullish_engulfing', 'bearish_engulfing',
            'three_white_soldiers', 'three_black_crows', 'breakout'
        ]
        
        self.reversal_patterns = [
            'hammer', 'shooting_star', 'morning_star', 'evening_star',
            'doji', 'spinning_top', 'hanging_man', 'inverted_hammer'
        ]
        
        logger.info("🚀 Trend Alignment Enhancer initialized")
    
    def analyze_trend_context(
        self, 
        df: pd.DataFrame, 
        timeframe: str = "1h"
    ) -> TrendContext:
        """
        Analyze complete trend context for a timeframe
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe being analyzed
            
        Returns:
            TrendContext with complete trend information
        """
        if len(df) < 50:
            return self._get_default_trend_context()
        
        try:
            # Calculate technical indicators
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema_50
            current_price = df['close'].iloc[-1]
            
            # Calculate ADX for trend strength
            adx_value = self._calculate_adx(df)
            
            # Calculate RSI
            rsi = self._calculate_rsi(df)
            
            # Calculate MACD signal
            macd_signal = self._calculate_macd_signal(df)
            
            # Determine trend direction
            direction = self._determine_trend_direction(
                current_price, ema_20, ema_50, ema_200, adx_value
            )
            
            # Determine trend strength
            strength = self._determine_trend_strength(adx_value)
            
            # Check EMA alignment
            ema_alignment = self._check_ema_alignment(ema_20, ema_50, ema_200, direction)
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(
                direction, strength, ema_alignment, adx_value, rsi
            )
            
            # Generate description
            description = self._generate_trend_description(
                direction, strength, ema_alignment, adx_value
            )
            
            return TrendContext(
                direction=direction,
                strength=strength,
                adx_value=adx_value,
                ema_alignment=ema_alignment,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                rsi=rsi,
                macd_signal=macd_signal,
                confidence=confidence,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend context: {e}")
            return self._get_default_trend_context()
    
    def check_mtf_trend_confirmation(
        self, 
        pattern_name: str,
        pattern_direction: str,
        current_tf_trend: TrendContext,
        higher_tf_trend: Optional[TrendContext] = None,
        lower_tf_trend: Optional[TrendContext] = None
    ) -> MTFTrendResult:
        """
        Check multi-timeframe trend confirmation for a pattern
        
        Args:
            pattern_name: Name of the detected pattern
            pattern_direction: Direction of the pattern ("bullish", "bearish", "neutral")
            current_tf_trend: Trend context for current timeframe
            higher_tf_trend: Trend context for higher timeframe
            lower_tf_trend: Trend context for lower timeframe
            
        Returns:
            MTFTrendResult with confirmation details
        """
        try:
            # Determine pattern type
            pattern_type = self._classify_pattern_type(pattern_name)
            
            # Calculate confidence multiplier based on trend alignment
            confidence_multiplier, alignment_description = self._calculate_trend_multiplier(
                pattern_type, pattern_direction, current_tf_trend, higher_tf_trend
            )
            
            # Determine overall trend alignment
            trend_alignment = confidence_multiplier >= 1.0
            
            return MTFTrendResult(
                current_tf_trend=current_tf_trend,
                higher_tf_trend=higher_tf_trend,
                lower_tf_trend=lower_tf_trend,
                trend_alignment=trend_alignment,
                confidence_multiplier=confidence_multiplier,
                alignment_description=alignment_description
            )
            
        except Exception as e:
            logger.error(f"Error checking MTF trend confirmation: {e}")
            return MTFTrendResult(
                current_tf_trend=current_tf_trend,
                trend_alignment=False,
                confidence_multiplier=1.0,
                alignment_description="Error in trend analysis"
            )
    
    def _calculate_trend_multiplier(
        self,
        pattern_type: str,
        pattern_direction: str,
        current_tf_trend: TrendContext,
        higher_tf_trend: Optional[TrendContext]
    ) -> Tuple[float, str]:
        """
        Calculate confidence multiplier based on trend alignment rules
        """
        multiplier = 1.0
        description_parts = []
        
        # Rule 1: Continuation patterns get boost if trend aligns
        if pattern_type == "continuation":
            if pattern_direction == "bullish" and current_tf_trend.direction == TrendDirection.BULLISH:
                multiplier *= self.trend_multipliers['continuation_bullish']
                description_parts.append("Continuation pattern aligns with bullish trend")
            elif pattern_direction == "bearish" and current_tf_trend.direction == TrendDirection.BEARISH:
                multiplier *= self.trend_multipliers['continuation_bearish']
                description_parts.append("Continuation pattern aligns with bearish trend")
            elif pattern_direction != current_tf_trend.direction.value:
                multiplier *= self.trend_multipliers['counter_trend']
                description_parts.append("Continuation pattern against trend")
        
        # Rule 2: Reversal patterns get boost only if trend is weakening
        elif pattern_type == "reversal":
            if higher_tf_trend and higher_tf_trend.direction == TrendDirection.WEAKENING:
                multiplier *= self.trend_multipliers['reversal_weakening']
                description_parts.append("Reversal pattern in weakening higher timeframe trend")
            elif pattern_direction != current_tf_trend.direction.value:
                multiplier *= self.trend_multipliers['counter_trend']
                description_parts.append("Reversal pattern against strong trend")
            else:
                description_parts.append("Reversal pattern in neutral trend")
        
        # Rule 3: Higher timeframe trend influence
        if higher_tf_trend:
            if pattern_direction == "bullish" and higher_tf_trend.direction == TrendDirection.BULLISH:
                multiplier *= 1.1
                description_parts.append("Higher timeframe bullish confirmation")
            elif pattern_direction == "bearish" and higher_tf_trend.direction == TrendDirection.BEARISH:
                multiplier *= 1.1
                description_parts.append("Higher timeframe bearish confirmation")
            elif pattern_direction != higher_tf_trend.direction.value:
                multiplier *= 0.9
                description_parts.append("Higher timeframe trend conflict")
        
        # Rule 4: Trend strength influence
        if current_tf_trend.strength in [TrendStrength.STRONG, TrendStrength.EXTREME]:
            if pattern_type == "continuation":
                multiplier *= 1.05
                description_parts.append("Strong trend supports continuation")
            else:
                multiplier *= 0.95
                description_parts.append("Strong trend resists reversal")
        
        # Rule 5: EMA alignment bonus
        if current_tf_trend.ema_alignment:
            multiplier *= 1.05
            description_parts.append("EMA alignment confirmed")
        
        description = "; ".join(description_parts) if description_parts else "Neutral trend alignment"
        
        return multiplier, description
    
    def _classify_pattern_type(self, pattern_name: str) -> str:
        """Classify pattern as continuation or reversal"""
        pattern_lower = pattern_name.lower()
        
        if any(cont in pattern_lower for cont in self.continuation_patterns):
            return "continuation"
        elif any(rev in pattern_lower for rev in self.reversal_patterns):
            return "reversal"
        else:
            return "neutral"
    
    def _determine_trend_direction(
        self, 
        current_price: float, 
        ema_20: float, 
        ema_50: float, 
        ema_200: float, 
        adx: float
    ) -> TrendDirection:
        """Determine trend direction based on price and EMAs"""
        # Check if trend is weakening (EMA crossovers)
        ema_20_vs_50 = ema_20 > ema_50
        ema_50_vs_200 = ema_50 > ema_200
        
        # Strong bullish trend
        if current_price > ema_20 > ema_50 > ema_200 and adx > 25:
            return TrendDirection.BULLISH
        # Strong bearish trend
        elif current_price < ema_20 < ema_50 < ema_200 and adx > 25:
            return TrendDirection.BEARISH
        # Weakening bullish trend
        elif ema_20_vs_50 and not ema_50_vs_200:
            return TrendDirection.WEAKENING
        # Weakening bearish trend
        elif not ema_20_vs_50 and ema_50_vs_200:
            return TrendDirection.WEAKENING
        # Neutral trend
        else:
            return TrendDirection.NEUTRAL
    
    def _determine_trend_strength(self, adx: float) -> TrendStrength:
        """Determine trend strength based on ADX"""
        if adx >= self.adx_thresholds[TrendStrength.EXTREME]:
            return TrendStrength.EXTREME
        elif adx >= self.adx_thresholds[TrendStrength.STRONG]:
            return TrendStrength.STRONG
        elif adx >= self.adx_thresholds[TrendStrength.MODERATE]:
            return TrendStrength.MODERATE
        else:
            return TrendStrength.WEAK
    
    def _check_ema_alignment(self, ema_20: float, ema_50: float, ema_200: float, direction: TrendDirection) -> bool:
        """Check if EMAs are aligned with trend direction"""
        if direction == TrendDirection.BULLISH:
            return ema_20 > ema_50 > ema_200
        elif direction == TrendDirection.BEARISH:
            return ema_20 < ema_50 < ema_200
        else:
            return True  # Neutral alignment
    
    def _calculate_trend_confidence(
        self, 
        direction: TrendDirection, 
        strength: TrendStrength, 
        ema_alignment: bool, 
        adx: float, 
        rsi: float
    ) -> float:
        """Calculate overall trend confidence"""
        confidence = 0.5  # Base confidence
        
        # Direction confidence
        if direction in [TrendDirection.BULLISH, TrendDirection.BEARISH]:
            confidence += 0.2
        elif direction == TrendDirection.WEAKENING:
            confidence += 0.1
        
        # Strength confidence
        if strength == TrendStrength.STRONG:
            confidence += 0.15
        elif strength == TrendStrength.EXTREME:
            confidence += 0.2
        elif strength == TrendStrength.MODERATE:
            confidence += 0.1
        
        # EMA alignment bonus
        if ema_alignment:
            confidence += 0.1
        
        # ADX bonus
        if adx > 30:
            confidence += 0.05
        
        # RSI filter (avoid extreme readings)
        if 30 <= rsi <= 70:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _generate_trend_description(
        self, 
        direction: TrendDirection, 
        strength: TrendStrength, 
        ema_alignment: bool, 
        adx: float
    ) -> str:
        """Generate human-readable trend description"""
        direction_desc = direction.value.replace('_', ' ').title()
        strength_desc = strength.value.title()
        
        desc = f"{strength_desc} {direction_desc} trend (ADX: {adx:.1f})"
        
        if ema_alignment:
            desc += " with aligned EMAs"
        
        return desc
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate Directional Movement
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            # Smooth the values
            tr_smooth = pd.Series(tr).rolling(period).mean().iloc[-1]
            dm_plus_smooth = pd.Series(dm_plus).rolling(period).mean().iloc[-1]
            dm_minus_smooth = pd.Series(dm_minus).rolling(period).mean().iloc[-1]
            
            # Calculate DI+ and DI-
            di_plus = (dm_plus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            di_minus = (dm_minus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            
            # Calculate DX and ADX
            dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100 if (di_plus + di_minus) > 0 else 0
            adx = pd.Series([dx]).rolling(period).mean().iloc[-1]
            
            return adx if not np.isnan(adx) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd_signal(self, df: pd.DataFrame) -> str:
        """Calculate MACD signal"""
        try:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            if current_macd > current_signal:
                return "bullish"
            elif current_macd < current_signal:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return "neutral"
    
    def _get_default_trend_context(self) -> TrendContext:
        """Return default trend context when analysis fails"""
        return TrendContext(
            direction=TrendDirection.NEUTRAL,
            strength=TrendStrength.WEAK,
            adx_value=0.0,
            ema_alignment=False,
            ema_20=0.0,
            ema_50=0.0,
            ema_200=0.0,
            rsi=50.0,
            macd_signal="neutral",
            confidence=0.5,
            description="Insufficient data for trend analysis"
        )
