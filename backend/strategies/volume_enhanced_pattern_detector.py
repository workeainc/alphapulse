#!/usr/bin/env python3
"""
Volume-Enhanced Pattern Detector for AlphaPulse
Integrates VolumeAnalyzer directly into pattern detection for real-time volume confirmation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .pattern_detector import CandlestickPatternDetector, PatternSignal
from data.volume_analyzer import VolumeAnalyzer, VolumePattern, VolumePatternType, VolumeStrength

logger = logging.getLogger(__name__)

class VolumeConfirmationType(Enum):
    """Types of volume confirmation"""
    SPIKE = "spike"
    DIVERGENCE = "divergence"
    NO_CONFIRMATION = "no_confirmation"
    WEAK_CONFIRMATION = "weak_confirmation"
    STRONG_CONFIRMATION = "strong_confirmation"

@dataclass
class VolumeEnhancedPatternSignal:
    """Volume-enhanced pattern signal with integrated volume analysis"""
    pattern_name: str
    timestamp: datetime
    price: float
    signal_type: str  # "bullish", "bearish", "neutral"
    base_confidence: float
    volume_confirmed: bool
    volume_factor: float  # 0.0-1.0, how much volume confirms the pattern
    volume_confirmation_type: VolumeConfirmationType
    volume_pattern: Optional[VolumePattern] = None
    volume_ratio: float = 0.0
    volume_strength: str = "weak"
    pattern_metadata: Dict[str, Any] = None

class VolumeEnhancedPatternDetector:
    """
    Volume-enhanced pattern detector that integrates VolumeAnalyzer directly
    into detection functions for real-time volume confirmation
    """
    
    def __init__(self):
        self.pattern_detector = CandlestickPatternDetector()
        self.volume_analyzer = VolumeAnalyzer()
        
        # Pattern-specific volume rules configuration
        self.pattern_volume_rules = {
            'hammer': {
                'required_volume_ratio': 1.2,
                'volume_confirmation_bonus': 0.15,
                'description': 'Hammer requires above-average volume for confirmation'
            },
            'shooting_star': {
                'required_volume_ratio': 1.3,
                'volume_confirmation_bonus': 0.20,
                'description': 'Shooting star needs high volume for bearish confirmation'
            },
            'bullish_engulfing': {
                'required_volume_ratio': 1.5,
                'volume_confirmation_bonus': 0.25,
                'description': 'Bullish engulfing requires strong volume confirmation'
            },
            'bearish_engulfing': {
                'required_volume_ratio': 1.5,
                'volume_confirmation_bonus': 0.25,
                'description': 'Bearish engulfing requires strong volume confirmation'
            },
            'doji': {
                'required_volume_ratio': 1.1,
                'volume_confirmation_bonus': 0.10,
                'description': 'Doji benefits from moderate volume confirmation'
            },
            'morning_star': {
                'required_volume_ratio': 1.4,
                'volume_confirmation_bonus': 0.20,
                'description': 'Morning star pattern needs good volume on third candle'
            },
            'evening_star': {
                'required_volume_ratio': 1.4,
                'volume_confirmation_bonus': 0.20,
                'description': 'Evening star pattern needs good volume on third candle'
            },
            'breakout': {
                'required_volume_ratio': 2.0,
                'volume_confirmation_bonus': 0.30,
                'description': 'Breakout patterns require very high volume confirmation'
            }
        }
        
        # Volume strength thresholds
        self.volume_strength_thresholds = {
            'weak': 1.0,
            'moderate': 1.5,
            'strong': 2.0,
            'extreme': 3.0
        }
        
        logger.info("🚀 Volume-Enhanced Pattern Detector initialized")
    
    def detect_patterns_with_volume(
        self, 
        df: pd.DataFrame, 
        symbol: str = "UNKNOWN",
        timeframe: str = "1h"
    ) -> List[VolumeEnhancedPatternSignal]:
        """
        Detect patterns with integrated volume analysis
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for analysis
            timeframe: Timeframe for analysis
            
        Returns:
            List of VolumeEnhancedPatternSignal objects with volume confirmation
        """
        if len(df) < 20:
            logger.warning("Insufficient data for pattern detection (minimum 20 candles required)")
            return []
        
        # Ensure volume column exists
        if 'volume' not in df.columns:
            logger.warning("No volume data available, using default volume analysis")
            df['volume'] = 1000  # Default volume
        
        # Detect basic patterns
        pattern_signals = self.pattern_detector.detect_patterns_from_dataframe(df)
        
        if not pattern_signals:
            return []
        
        # Enhance patterns with volume analysis
        volume_enhanced_signals = []
        
        for signal in pattern_signals:
            try:
                enhanced_signal = self._enhance_pattern_with_volume(
                    signal, df, symbol, timeframe
                )
                
                if enhanced_signal:
                    volume_enhanced_signals.append(enhanced_signal)
                    
            except Exception as e:
                logger.error(f"Error enhancing pattern {signal.pattern} with volume: {e}")
                continue
        
        logger.info(f"🔍 Detected {len(volume_enhanced_signals)} volume-enhanced patterns")
        return volume_enhanced_signals
    
    def _enhance_pattern_with_volume(
        self, 
        signal: PatternSignal, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Optional[VolumeEnhancedPatternSignal]:
        """
        Enhance a pattern signal with volume analysis
        """
        try:
            # Get volume analysis for the pattern
            volume_analysis = self._analyze_volume_for_pattern(signal, df)
            
            # Determine volume confirmation
            volume_confirmed, volume_factor, confirmation_type = self._determine_volume_confirmation(
                signal, volume_analysis, df
            )
            
            # Get volume pattern if detected
            volume_pattern = self._detect_volume_pattern(signal, df)
            
            # Calculate volume ratio
            volume_ratio = self._calculate_volume_ratio(df, signal.index)
            
            # Determine volume strength
            volume_strength = self._determine_volume_strength(volume_ratio)
            
            # Create enhanced signal
            enhanced_signal = VolumeEnhancedPatternSignal(
                pattern_name=signal.pattern,
                timestamp=datetime.now(),
                price=df['close'].iloc[signal.index] if signal.index < len(df) else df['close'].iloc[-1],
                signal_type=signal.type,
                base_confidence=signal.confidence,
                volume_confirmed=volume_confirmed,
                volume_factor=volume_factor,
                volume_confirmation_type=confirmation_type,
                volume_pattern=volume_pattern,
                volume_ratio=volume_ratio,
                volume_strength=volume_strength,
                pattern_metadata=signal.additional_info or {}
            )
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error enhancing pattern with volume: {e}")
            return None
    
    def _analyze_volume_for_pattern(
        self, 
        signal: PatternSignal, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze volume specifically for the detected pattern
        """
        try:
            # Get pattern-specific data window
            pattern_window = self._get_pattern_window(signal, df)
            
            # Analyze volume using VolumeAnalyzer
            volume_analysis = self.volume_analyzer._analyze_volume_advanced(pattern_window)
            
            # Add pattern-specific volume metrics
            volume_analysis.update({
                'pattern_volume_ratio': self._calculate_volume_ratio(df, signal.index),
                'volume_trend_alignment': self._check_volume_trend_alignment(signal, df),
                'volume_divergence': self._detect_volume_divergence(signal, df)
            })
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volume for pattern: {e}")
            return self._get_default_volume_analysis()
    
    def _get_pattern_window(self, signal: PatternSignal, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the appropriate data window for pattern analysis
        """
        try:
            # For most patterns, use the last 20 candles
            start_idx = max(0, signal.index - 19)
            end_idx = min(len(df), signal.index + 1)
            
            return df.iloc[start_idx:end_idx].copy()
            
        except Exception as e:
            logger.error(f"Error getting pattern window: {e}")
            return df.tail(20).copy()
    
    def _determine_volume_confirmation(
        self, 
        signal: PatternSignal, 
        volume_analysis: Dict[str, Any], 
        df: pd.DataFrame
    ) -> Tuple[bool, float, VolumeConfirmationType]:
        """
        Determine if volume confirms the pattern and calculate confirmation factor
        """
        try:
            pattern_name = signal.pattern.lower()
            volume_rules = self.pattern_volume_rules.get(pattern_name, {})
            
            # Get required volume ratio for this pattern
            required_ratio = volume_rules.get('required_volume_ratio', 1.2)
            confirmation_bonus = volume_rules.get('volume_confirmation_bonus', 0.15)
            
            # Calculate current volume ratio
            current_volume_ratio = volume_analysis.get('pattern_volume_ratio', 1.0)
            
            # Check for volume spike
            volume_trend = volume_analysis.get('volume_trend', 'unknown')
            volume_consistency = volume_analysis.get('volume_consistency', 0.0)
            
            # Determine confirmation type
            if current_volume_ratio >= required_ratio:
                if current_volume_ratio >= self.volume_strength_thresholds['strong']:
                    confirmation_type = VolumeConfirmationType.STRONG_CONFIRMATION
                    volume_confirmed = True
                    volume_factor = min(1.0, confirmation_bonus * (current_volume_ratio / required_ratio))
                else:
                    confirmation_type = VolumeConfirmationType.SPIKE
                    volume_confirmed = True
                    volume_factor = min(1.0, confirmation_bonus * 0.8)
            elif current_volume_ratio >= 1.0:
                confirmation_type = VolumeConfirmationType.WEAK_CONFIRMATION
                volume_confirmed = True
                volume_factor = confirmation_bonus * 0.5
            else:
                # Check for volume divergence
                volume_divergence = volume_analysis.get('volume_divergence', {})
                divergence_strength = volume_divergence.get('strength', 0.0)
                
                if divergence_strength > 0.3:
                    confirmation_type = VolumeConfirmationType.DIVERGENCE
                    volume_confirmed = False
                    volume_factor = -0.1  # Penalty for divergence
                else:
                    confirmation_type = VolumeConfirmationType.NO_CONFIRMATION
                    volume_confirmed = False
                    volume_factor = 0.0
            
            return volume_confirmed, volume_factor, confirmation_type
            
        except Exception as e:
            logger.error(f"Error determining volume confirmation: {e}")
            return False, 0.0, VolumeConfirmationType.NO_CONFIRMATION
    
    def _detect_volume_pattern(self, signal: PatternSignal, df: pd.DataFrame) -> Optional[VolumePattern]:
        """
        Detect specific volume patterns for the signal
        """
        try:
            # Get pattern window
            pattern_window = self._get_pattern_window(signal, df)
            
            # Use VolumeAnalyzer to detect patterns
            # This would integrate with the existing VolumeAnalyzer pattern detection
            # For now, we'll create a basic volume pattern
            
            volume_ratio = self._calculate_volume_ratio(df, signal.index)
            
            if volume_ratio > 2.0:
                pattern_type = VolumePatternType.VOLUME_SPIKE
                strength = VolumeStrength.EXTREME
            elif volume_ratio > 1.5:
                pattern_type = VolumePatternType.VOLUME_BREAKOUT
                strength = VolumeStrength.STRONG
            elif volume_ratio > 1.2:
                pattern_type = VolumePatternType.VOLUME_ACCUMULATION
                strength = VolumeStrength.MEDIUM
            else:
                pattern_type = VolumePatternType.VOLUME_DRY_UP
                strength = VolumeStrength.WEAK
            
            volume_pattern = VolumePattern(
                pattern_type=pattern_type,
                symbol="UNKNOWN",
                timeframe="1h",
                timestamp=datetime.now(),
                strength=strength,
                confidence=min(1.0, volume_ratio / 2.0),
                volume_ratio=volume_ratio,
                price_change=0.0,
                pattern_data={'volume_ratio': volume_ratio},
                description=f"Volume {pattern_type.value} detected for {signal.pattern}"
            )
            
            return volume_pattern
            
        except Exception as e:
            logger.error(f"Error detecting volume pattern: {e}")
            return None
    
    def _calculate_volume_ratio(self, df: pd.DataFrame, pattern_index: int) -> float:
        """
        Calculate volume ratio at pattern location
        """
        try:
            if pattern_index >= len(df):
                pattern_index = len(df) - 1
            
            current_volume = df['volume'].iloc[pattern_index]
            
            # Calculate average volume over the last 20 periods
            lookback = min(20, pattern_index)
            avg_volume = df['volume'].iloc[pattern_index - lookback:pattern_index].mean()
            
            if avg_volume == 0:
                return 1.0
            
            return current_volume / avg_volume
            
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    def _determine_volume_strength(self, volume_ratio: float) -> str:
        """
        Determine volume strength based on ratio
        """
        if volume_ratio >= self.volume_strength_thresholds['extreme']:
            return 'extreme'
        elif volume_ratio >= self.volume_strength_thresholds['strong']:
            return 'strong'
        elif volume_ratio >= self.volume_strength_thresholds['moderate']:
            return 'moderate'
        else:
            return 'weak'
    
    def _check_volume_trend_alignment(self, signal: PatternSignal, df: pd.DataFrame) -> float:
        """
        Check if volume trend aligns with pattern direction
        """
        try:
            # Get recent volume trend
            recent_volume = df['volume'].tail(10).mean()
            historical_volume = df['volume'].tail(30).head(20).mean()
            
            if historical_volume == 0:
                return 0.0
            
            volume_trend_ratio = recent_volume / historical_volume
            
            # Check pattern direction
            pattern_direction = signal.type  # 'bullish', 'bearish', 'neutral'
            
            # For bullish patterns, we want increasing volume
            if pattern_direction == 'bullish':
                return min(1.0, volume_trend_ratio - 1.0)
            elif pattern_direction == 'bearish':
                return min(1.0, volume_trend_ratio - 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error checking volume trend alignment: {e}")
            return 0.0
    
    def _detect_volume_divergence(self, signal: PatternSignal, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect volume-price divergence
        """
        try:
            # Get recent price and volume data
            recent_data = df.tail(10)
            
            # Calculate price change
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Calculate volume change
            volume_change = (recent_data['volume'].iloc[-1] - recent_data['volume'].iloc[0]) / recent_data['volume'].iloc[0]
            
            # Detect divergence
            divergence_strength = 0.0
            divergence_type = "none"
            
            if abs(price_change) > 0.02:  # 2% price change
                if price_change > 0 and volume_change < -0.1:  # Price up, volume down
                    divergence_strength = min(1.0, abs(volume_change))
                    divergence_type = "negative"
                elif price_change < 0 and volume_change > 0.1:  # Price down, volume up
                    divergence_strength = min(1.0, volume_change)
                    divergence_type = "positive"
            
            return {
                'strength': divergence_strength,
                'type': divergence_type,
                'price_change': price_change,
                'volume_change': volume_change
            }
            
        except Exception as e:
            logger.error(f"Error detecting volume divergence: {e}")
            return {'strength': 0.0, 'type': 'none', 'price_change': 0.0, 'volume_change': 0.0}
    
    def _get_default_volume_analysis(self) -> Dict[str, Any]:
        """
        Return default volume analysis
        """
        return {
            'volume_trend': 'unknown',
            'volume_consistency': 0.0,
            'breakout_strength': 0.0,
            'pattern_volume_ratio': 1.0,
            'volume_trend_alignment': 0.0,
            'volume_divergence': {'strength': 0.0, 'type': 'none'}
        }
    
    def get_volume_confirmation_summary(
        self, 
        signals: List[VolumeEnhancedPatternSignal]
    ) -> Dict[str, Any]:
        """
        Get a summary of volume confirmation across all signals
        """
        if not signals:
            return {'message': 'No signals to analyze'}
        
        # Count confirmations
        confirmed_count = len([s for s in signals if s.volume_confirmed])
        total_count = len(signals)
        
        # Average volume factor
        avg_volume_factor = sum(s.volume_factor for s in signals) / total_count
        
        # Volume confirmation types
        confirmation_types = {}
        for signal in signals:
            conf_type = signal.volume_confirmation_type.value
            confirmation_types[conf_type] = confirmation_types.get(conf_type, 0) + 1
        
        # Volume strength distribution
        strength_distribution = {}
        for signal in signals:
            strength = signal.volume_strength
            strength_distribution[strength] = strength_distribution.get(strength, 0) + 1
        
        return {
            'total_signals': total_count,
            'volume_confirmed_signals': confirmed_count,
            'confirmation_rate': confirmed_count / total_count,
            'average_volume_factor': avg_volume_factor,
            'confirmation_types': confirmation_types,
            'strength_distribution': strength_distribution,
            'patterns_with_volume': [s.pattern_name for s in signals if s.volume_confirmed]
        }
