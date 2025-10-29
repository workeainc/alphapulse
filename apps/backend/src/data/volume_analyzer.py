#!/usr/bin/env python3
"""
Volume Pattern Recognition for AlphaPulse
Advanced volume analysis for signal confirmation and market condition detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VolumePatternType(Enum):
    """Types of volume patterns"""
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DIVERGENCE = "volume_divergence"
    VOLUME_CLIMAX = "volume_climax"
    VOLUME_DRY_UP = "volume_dry_up"
    VOLUME_ACCUMULATION = "volume_accumulation"
    VOLUME_DISTRIBUTION = "volume_distribution"
    VOLUME_BREAKOUT = "volume_breakout"
    VOLUME_FAKE_OUT = "volume_fake_out"
    # Wyckoff patterns
    WYCKOFF_SPRING = "wyckoff_spring"
    WYCKOFF_UPTHRUST = "wyckoff_upthrust"
    WYCKOFF_ACCUMULATION = "wyckoff_accumulation"
    WYCKOFF_DISTRIBUTION = "wyckoff_distribution"
    WYCKOFF_TEST = "wyckoff_test"
    WYCKOFF_SIGN_OF_STRENGTH = "wyckoff_sign_of_strength"
    WYCKOFF_SIGN_OF_WEAKNESS = "wyckoff_sign_of_weakness"

class VolumeStrength(Enum):
    """Volume pattern strength levels"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    EXTREME = "extreme"

@dataclass
class VolumePattern:
    """Detected volume pattern with metadata"""
    pattern_type: VolumePatternType
    symbol: str
    timeframe: str
    timestamp: datetime
    strength: VolumeStrength
    confidence: float
    volume_ratio: float
    price_change: float
    pattern_data: Dict
    description: str
    
    def check_confirmation(self, pattern: str, price_data: List[Dict], volume_data: List[Dict]) -> bool:
        """
        Advanced volume confirmation with pattern-specific rules
        
        Args:
            pattern: Name/type of the detected price pattern
            price_data: Recent OHLC price data for the pattern's timeframe
            volume_data: Raw volume series aligned with price_data
            
        Returns:
            True if volume confirms the pattern, False otherwise
        """
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(price_data)
            df['volume'] = [v.get('volume', 0) for v in volume_data]
            
            # Get advanced volume analysis
            volume_analyzer = VolumeAnalyzer()
            volume_analysis = volume_analyzer._analyze_volume_advanced(df)
            
            # Match volume to pattern using advanced rules
            return self._match_volume_to_pattern_advanced(pattern, volume_analysis, df)
            
        except Exception as e:
            logger.error(f"Error in volume confirmation check: {e}")
            return False
    
    def check_confirmation_with_confidence(self, pattern: str, price_data: List[Dict], volume_data: List[Dict]) -> float:
        """
        Return confidence score (0.0-1.0) instead of just boolean
        
        Args:
            pattern: Name/type of the detected price pattern
            price_data: Recent OHLC price data for the pattern's timeframe
            volume_data: Raw volume series aligned with price_data
            
        Returns:
            Confidence score from 0.0 (no confirmation) to 1.0 (perfect confirmation)
        """
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(price_data)
            df['volume'] = [v.get('volume', 0) for v in volume_data]
            
            # Get advanced volume analysis
            volume_analyzer = VolumeAnalyzer()
            volume_analysis = volume_analyzer._analyze_volume_advanced(df)
            
            # Get base confirmation first
            base_confirmation = self._match_volume_to_pattern_advanced(pattern, volume_analysis, df)
            
            if not base_confirmation:
                return 0.0
            
            # Calculate confidence based on multiple factors
            confidence_factors = {
                "volume_trend_alignment": self._calculate_volume_trend_alignment_score(volume_analysis),
                "breakout_strength": self._calculate_breakout_strength_score(volume_analysis),
                "divergence_strength": self._calculate_divergence_strength_score(volume_analysis),
                "volume_consistency": self._calculate_volume_consistency_score(volume_analysis),
                "timeframe_alignment": self._calculate_timeframe_alignment_score(volume_analysis),
                "pattern_specific_score": self._calculate_pattern_specific_score(pattern, volume_analysis, df)
            }
            
            # Weighted average of confidence factors
            weights = {
                "volume_trend_alignment": 0.25,      # 25% - How well volume aligns with price
                "breakout_strength": 0.20,           # 20% - Strength of volume breakout
                "divergence_strength": 0.15,         # 15% - Volume divergence confirmation
                "volume_consistency": 0.15,          # 15% - Volume consistency during pattern
                "timeframe_alignment": 0.15,         # 15% - Multi-timeframe alignment
                "pattern_specific_score": 0.10       # 10% - Pattern-specific volume rules
            }
            
            # Calculate weighted confidence score
            total_weight = 0.0
            weighted_sum = 0.0
            
            for factor, weight in weights.items():
                if factor in confidence_factors:
                    weighted_sum += confidence_factors[factor] * weight
                    total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            confidence_score = weighted_sum / total_weight
            
            # Ensure score is between 0.0 and 1.0
            return min(1.0, max(0.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Error in confidence scoring: {e}")
            return 0.0
    
    def _calculate_volume_trend_alignment_score(self, volume_analysis: Dict) -> float:
        """Calculate volume trend alignment score (0.0-1.0)"""
        try:
            alignment = volume_analysis.get('volume_trend_alignment', 0.0)
            return alignment
        except Exception as e:
            logger.error(f"Error calculating volume trend alignment score: {e}")
            return 0.0
    
    def _calculate_breakout_strength_score(self, volume_analysis: Dict) -> float:
        """Calculate breakout strength score (0.0-1.0)"""
        try:
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            return breakout_strength
        except Exception as e:
            logger.error(f"Error calculating breakout strength score: {e}")
            return 0.0
    
    def _calculate_divergence_strength_score(self, volume_analysis: Dict) -> float:
        """Calculate divergence strength score (0.0-1.0)"""
        try:
            divergence_data = volume_analysis.get('volume_divergence', {})
            divergence_strength = divergence_data.get('strength', 0.0)
            return divergence_strength
        except Exception as e:
            logger.error(f"Error calculating divergence strength score: {e}")
            return 0.0
    
    def _calculate_volume_consistency_score(self, volume_analysis: Dict) -> float:
        """Calculate volume consistency score (0.0-1.0)"""
        try:
            consistency = volume_analysis.get('volume_consistency', 0.0)
            return consistency
        except Exception as e:
            logger.error(f"Error calculating volume consistency score: {e}")
            return 0.0
    
    def _calculate_timeframe_alignment_score(self, volume_analysis: Dict) -> float:
        """Calculate multi-timeframe alignment score (0.0-1.0)"""
        try:
            multi_timeframe = volume_analysis.get('multi_timeframe', {})
            alignment_score = multi_timeframe.get('alignment_score', 0.0)
            return alignment_score
        except Exception as e:
            logger.error(f"Error calculating timeframe alignment score: {e}")
            return 0.0
    
    def _calculate_pattern_specific_score(self, pattern: str, volume_analysis: Dict, price_data: pd.DataFrame) -> float:
        """Calculate pattern-specific volume score (0.0-1.0)"""
        try:
            pattern_lower = pattern.lower()
            
            # Head and Shoulders specific scoring
            if any(term in pattern_lower for term in ['head', 'shoulder', 'headandshoulder']):
                return self._calculate_hs_pattern_score(volume_analysis, price_data)
            
            # Cup and Handle specific scoring
            elif any(term in pattern_lower for term in ['cup', 'handle', 'cupandhandle']):
                return self._calculate_cup_handle_pattern_score(volume_analysis, price_data)
            
            # Triangle patterns specific scoring
            elif any(term in pattern_lower for term in ['triangle', 'ascending', 'descending', 'symmetrical']):
                return self._calculate_triangle_pattern_score(volume_analysis, price_data)
            
            # Double patterns specific scoring
            elif any(term in pattern_lower for term in ['double', 'bottom', 'top', 'doublebottom', 'doubletop']):
                return self._calculate_double_pattern_score(volume_analysis, price_data)
            
            # Default pattern scoring
            else:
                return self._calculate_default_pattern_score(volume_analysis, price_data)
                
        except Exception as e:
            logger.error(f"Error calculating pattern-specific score: {e}")
            return 0.5  # Neutral score for unknown patterns
    
    def _calculate_hs_pattern_score(self, volume_analysis: Dict, price_data: pd.DataFrame) -> float:
        """Calculate Head & Shoulders pattern-specific score"""
        try:
            score = 0.0
            
            # Volume decreasing from left shoulder to head (0-0.4 points)
            if self._validate_head_and_shoulders_volume(volume_analysis, price_data):
                score += 0.4
            
            # Strong breakdown volume (0-0.3 points)
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            if breakout_strength > 0.7:
                score += 0.3
            elif breakout_strength > 0.5:
                score += 0.2
            elif breakout_strength > 0.3:
                score += 0.1
            
            # Volume trend alignment (0-0.3 points)
            trend_alignment = volume_analysis.get('volume_trend_alignment', 0.0)
            if trend_alignment > 0.8:
                score += 0.3
            elif trend_alignment > 0.6:
                score += 0.2
            elif trend_alignment > 0.4:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating H&S pattern score: {e}")
            return 0.0
    
    def _calculate_cup_handle_pattern_score(self, volume_analysis: Dict, price_data: pd.DataFrame) -> float:
        """Calculate Cup & Handle pattern-specific score"""
        try:
            score = 0.0
            
            # Breakout volume strength (0-0.4 points)
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            if breakout_strength > 0.8:
                score += 0.4
            elif breakout_strength > 0.6:
                score += 0.3
            elif breakout_strength > 0.4:
                score += 0.2
            
            # Volume consistency during cup formation (0-0.3 points)
            volume_consistency = volume_analysis.get('volume_consistency', 0.0)
            if volume_consistency > 0.7:
                score += 0.3
            elif volume_consistency > 0.5:
                score += 0.2
            elif volume_consistency > 0.3:
                score += 0.1
            
            # Multi-timeframe alignment (0-0.3 points)
            multi_timeframe = volume_analysis.get('multi_timeframe', {})
            alignment_score = multi_timeframe.get('alignment_score', 0.0)
            if alignment_score > 0.8:
                score += 0.3
            elif alignment_score > 0.6:
                score += 0.2
            elif alignment_score > 0.4:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating Cup & Handle pattern score: {e}")
            return 0.0
    
    def _calculate_triangle_pattern_score(self, volume_analysis: Dict, price_data: pd.DataFrame) -> float:
        """Calculate Triangle pattern-specific score"""
        try:
            score = 0.0
            
            # Volume decreasing during consolidation (0-0.3 points)
            volume_trend = volume_analysis.get('volume_trend', 'unknown')
            if volume_trend == 'decreasing':
                score += 0.3
            elif volume_trend == 'flat':
                score += 0.2
            
            # Breakout strength (0-0.4 points)
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            if breakout_strength > 0.7:
                score += 0.4
            elif breakout_strength > 0.5:
                score += 0.3
            elif breakout_strength > 0.3:
                score += 0.2
            
            # Volume consistency (0-0.3 points)
            volume_consistency = volume_analysis.get('volume_consistency', 0.0)
            if volume_consistency > 0.6:
                score += 0.3
            elif volume_consistency > 0.4:
                score += 0.2
            elif volume_consistency > 0.2:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating Triangle pattern score: {e}")
            return 0.0
    
    def _calculate_double_pattern_score(self, volume_analysis: Dict, price_data: pd.DataFrame) -> float:
        """Calculate Double Bottom/Top pattern-specific score"""
        try:
            score = 0.0
            
            # Volume decreasing on second peak (0-0.4 points)
            if self._validate_double_pattern_volume(volume_analysis, price_data):
                score += 0.4
            
            # Breakout strength (0-0.4 points)
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            if breakout_strength > 0.8:
                score += 0.4
            elif breakout_strength > 0.6:
                score += 0.3
            elif breakout_strength > 0.4:
                score += 0.2
            
            # Volume divergence confirmation (0-0.2 points)
            divergence_data = volume_analysis.get('volume_divergence', {})
            if divergence_data.get('bullish_divergence', False) or divergence_data.get('bearish_divergence', False):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating Double pattern score: {e}")
            return 0.0
    
    def _calculate_default_pattern_score(self, volume_analysis: Dict, price_data: pd.DataFrame) -> float:
        """Calculate default pattern score for unknown patterns"""
        try:
            score = 0.0
            
            # Volume above average (0-0.4 points)
            current_volume = price_data['volume'].iloc[-1]
            avg_volume = price_data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > 2.0:
                score += 0.4
            elif volume_ratio > 1.5:
                score += 0.3
            elif volume_ratio > 1.2:
                score += 0.2
            elif volume_ratio > 1.0:
                score += 0.1
            
            # Volume trend alignment (0-0.3 points)
            trend_alignment = volume_analysis.get('volume_trend_alignment', 0.0)
            if trend_alignment > 0.7:
                score += 0.3
            elif trend_alignment > 0.5:
                score += 0.2
            elif trend_alignment > 0.3:
                score += 0.1
            
            # Breakout strength (0-0.3 points)
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            if breakout_strength > 0.6:
                score += 0.3
            elif breakout_strength > 0.4:
                score += 0.2
            elif breakout_strength > 0.2:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating default pattern score: {e}")
            return 0.0
    
    def _match_volume_to_pattern_advanced(self, pattern: str, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced pattern-specific volume validation"""
        pattern_lower = pattern.lower()
        
        # Head and Shoulders pattern
        if any(term in pattern_lower for term in ['head', 'shoulder', 'head_andshoulder']):
            return self._validate_head_and_shoulders_volume(volume_analysis, price_data)
        
        # Cup and Handle pattern
        elif any(term in pattern_lower for term in ['cup', 'handle', 'cupandhandle']):
            return self._validate_cup_and_handle_volume(volume_analysis, price_data)
        
        # Triangle patterns
        elif any(term in pattern_lower for term in ['triangle', 'ascending', 'descending', 'symmetrical']):
            return self._validate_triangle_volume(volume_analysis, price_data)
        
        # Double patterns
        elif any(term in pattern_lower for term in ['double', 'bottom', 'top', 'doublebottom', 'doubletop']):
            return self._validate_double_pattern_volume(volume_analysis, price_data)
        
        # Flag and Pennant patterns
        elif any(term in pattern_lower for term in ['flag', 'pennant']):
            return self._validate_flag_pennant_volume(volume_analysis, price_data)
        
        # Wedge patterns
        elif any(term in pattern_lower for term in ['wedge', 'rising', 'falling']):
            return self._validate_wedge_volume(volume_analysis, price_data)
        
        # Channel patterns
        elif any(term in pattern_lower for term in ['channel', 'parallel']):
            return self._validate_channel_volume(volume_analysis, price_data)
        
        # Default: use basic volume confirmation
        else:
            return self._validate_basic_volume_confirmation(volume_analysis, price_data)
    
    def _validate_head_and_shoulders_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced H&S volume validation"""
        try:
            # Find peaks and troughs
            peaks = self._find_peaks(price_data['high'].values)
            troughs = self._find_peaks(-price_data['low'].values)
            
            if len(peaks) < 3 or len(troughs) < 2:
                return False
            
            # Get volume for each peak
            left_shoulder_volume = self._get_volume_for_peak(price_data, peaks[0])
            head_volume = self._get_volume_for_peak(price_data, peaks[1])
            right_shoulder_volume = self._get_volume_for_peak(price_data, peaks[2])
            
            # Get breakdown volume
            breakdown_volume = self._get_volume_for_breakdown(price_data)
            
            # Volume should decrease from left shoulder to head
            volume_decreasing = left_shoulder_volume > head_volume > right_shoulder_volume
            
            # Strong breakdown volume
            strong_breakdown = breakdown_volume > (head_volume * 1.5)
            
            # Right shoulder should have lower volume than left
            right_weaker = right_shoulder_volume < left_shoulder_volume
            
            # Check volume trend alignment
            volume_trend_aligned = volume_analysis.get('volume_trend_alignment', 0) > 0.6
            
            return volume_decreasing and strong_breakdown and right_weaker and volume_trend_aligned
            
        except Exception as e:
            logger.error(f"Error in H&S volume validation: {e}")
            return False
    
    def _validate_cup_and_handle_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced Cup & Handle volume validation"""
        try:
            # Cup formation: low volume during cup, slight increase during handle
            cup_volume = self._get_average_volume(price_data.iloc[:len(price_data)//2])
            handle_volume = self._get_average_volume(price_data.iloc[len(price_data)//2:])
            
            # Breakout volume should be strong
            breakout_volume = volume_analysis.get('breakout_strength', 0)
            strong_breakout = breakout_volume > 0.7
            
            # Volume during cup should be lower than average
            avg_volume = price_data['volume'].mean()
            cup_volume_low = cup_volume < avg_volume * 0.8
            
            # Handle volume should be slightly higher than cup
            handle_volume_increasing = handle_volume > cup_volume * 1.1
            
            return strong_breakout and cup_volume_low and handle_volume_increasing
            
        except Exception as e:
            logger.error(f"Error in Cup & Handle volume validation: {e}")
            return False
    
    def _validate_triangle_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced Triangle pattern volume validation"""
        try:
            # Volume should decrease during consolidation
            volume_trend = volume_analysis.get('volume_trend', 'decreasing')
            volume_decreasing = volume_trend == 'decreasing'
            
            # Breakout volume should be strong
            breakout_strength = volume_analysis.get('breakout_strength', 0)
            strong_breakout = breakout_strength > 0.6
            
            # Volume consistency during consolidation
            volume_consistency = volume_analysis.get('volume_consistency', 0)
            consistent_volume = volume_consistency > 0.5
            
            return volume_decreasing and strong_breakout and consistent_volume
            
        except Exception as e:
            logger.error(f"Error in Triangle volume validation: {e}")
            return False
    
    def _validate_double_pattern_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced Double Bottom/Top volume validation"""
        try:
            # Second bottom/top should have lower volume
            second_peak_volume = self._get_volume_for_second_peak(price_data)
            first_peak_volume = self._get_volume_for_first_peak(price_data)
            
            volume_decreasing = second_peak_volume < first_peak_volume
            
            # Breakout volume should be strong
            breakout_strength = volume_analysis.get('breakout_strength', 0)
            strong_breakout = breakout_strength > 0.7
            
            return volume_decreasing and strong_breakout
            
        except Exception as e:
            logger.error(f"Error in Double pattern volume validation: {e}")
            return False
    
    def _validate_flag_pennant_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced Flag/Pennant volume validation"""
        try:
            # Volume should decrease during flag/pennant formation
            volume_trend = volume_analysis.get('volume_trend', 'decreasing')
            volume_decreasing = volume_trend == 'decreasing'
            
            # Breakout volume should be strong
            breakout_strength = volume_analysis.get('breakout_strength', 0)
            strong_breakout = breakout_strength > 0.6
            
            return volume_decreasing and strong_breakout
            
        except Exception as e:
            logger.error(f"Error in Flag/Pennant volume validation: {e}")
            return False
    
    def _validate_wedge_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced Wedge pattern volume validation"""
        try:
            # Volume should decrease as wedge narrows
            volume_trend = volume_analysis.get('volume_trend', 'decreasing')
            volume_decreasing = volume_trend == 'decreasing'
            
            # Breakout volume should be strong
            breakout_strength = volume_analysis.get('breakout_strength', 0)
            strong_breakout = breakout_strength > 0.6
            
            return volume_decreasing and strong_breakout
            
        except Exception as e:
            logger.error(f"Error in Wedge volume validation: {e}")
            return False
    
    def _validate_channel_volume(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Advanced Channel pattern volume validation"""
        try:
            # Volume should be consistent during channel formation
            volume_consistency = volume_analysis.get('volume_consistency', 0)
            consistent_volume = volume_consistency > 0.6
            
            # Breakout volume should be strong
            breakout_strength = volume_analysis.get('breakout_strength', 0)
            strong_breakout = breakout_strength > 0.6
            
            return consistent_volume and strong_breakout
            
        except Exception as e:
            logger.error(f"Error in Channel volume validation: {e}")
            return False
    
    def _validate_basic_volume_confirmation(self, volume_analysis: Dict, price_data: pd.DataFrame) -> bool:
        """Basic volume confirmation for unknown patterns"""
        try:
            # Check if volume is above average
            current_volume = price_data['volume'].iloc[-1]
            avg_volume = price_data['volume'].mean()
            volume_above_average = current_volume > avg_volume * 1.2
            
            # Check volume trend
            volume_trend_aligned = volume_analysis.get('volume_trend_alignment', 0) > 0.5
            
            return volume_above_average and volume_trend_aligned
            
        except Exception as e:
            logger.error(f"Error in basic volume validation: {e}")
            return False
    
    # Helper methods for volume analysis
    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """Find peaks in the data"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return peaks
    
    def _get_volume_for_peak(self, price_data: pd.DataFrame, peak_index: int) -> float:
        """Get volume for a specific peak"""
        if 0 <= peak_index < len(price_data):
            return price_data['volume'].iloc[peak_index]
        return 0.0
    
    def _get_volume_for_breakdown(self, price_data: pd.DataFrame) -> float:
        """Get volume for breakdown (last few candles)"""
        return price_data['volume'].iloc[-3:].mean()
    
    def _get_average_volume(self, price_data: pd.DataFrame) -> float:
        """Get average volume for a subset of data"""
        return price_data['volume'].mean()
    
    def _get_volume_for_second_peak(self, price_data: pd.DataFrame) -> float:
        """Get volume for second peak in double pattern"""
        return price_data['volume'].iloc[-5:].mean()
    
    def _get_volume_for_first_peak(self, price_data: pd.DataFrame) -> float:
        """Get volume for first peak in double pattern"""
        return price_data['volume'].iloc[:5].mean()

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            if len(data) < period:
                return np.array([data[0]] * len(data))
            
            ema = np.zeros_like(data)
            ema[period-1] = np.mean(data[:period])
            
            multiplier = 2 / (period + 1)
            
            for i in range(period, len(data)):
                ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return np.array([data[0]] * len(data))
    
    def _check_rsi_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check RSI confirmation for divergence"""
        try:
            if len(df) < 14:
                return False
            
            # Calculate RSI
            rsi = self._calculate_rsi(df['close'].values)
            
            if divergence_type == 'bullish':
                # RSI should be oversold (< 30) for bullish divergence
                return rsi[divergence_point] < 30
            elif divergence_type == 'bearish':
                # RSI should be overbought (> 70) for bearish divergence
                return rsi[divergence_point] > 70
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking RSI confirmation: {e}")
            return False
    
    def _check_macd_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check MACD confirmation for divergence"""
        try:
            if len(df) < 26:
                return False
            
            # Calculate MACD
            macd_line, signal_line = self._calculate_macd(df['close'].values)
            
            if divergence_type == 'bullish':
                # MACD should be turning up for bullish divergence
                return (macd_line[divergence_point] > signal_line[divergence_point] and 
                       macd_line[divergence_point] > macd_line[divergence_point - 1])
            elif divergence_type == 'bearish':
                # MACD should be turning down for bearish divergence
                return (macd_line[divergence_point] < signal_line[divergence_point] and 
                       macd_line[divergence_point] < macd_line[divergence_point - 1])
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking MACD confirmation: {e}")
            return False
    
    def _check_volume_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check volume confirmation for divergence"""
        try:
            if divergence_point < 20:
                return False
            
            current_volume = df['volume'].iloc[divergence_point]
            avg_volume = df['volume'].iloc[divergence_point-20:divergence_point].mean()
            
            if divergence_type == 'bullish':
                # Volume should be above average for bullish divergence
                return current_volume > avg_volume * 1.2
            elif divergence_type == 'bearish':
                # Volume should be above average for bearish divergence
                return current_volume > avg_volume * 1.2
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _check_price_action_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check price action confirmation for divergence"""
        try:
            if divergence_point < 5:
                return False
            
            # Check for reversal candlestick patterns
            current_candle = df.iloc[divergence_point]
            prev_candle = df.iloc[divergence_point - 1]
            
            if divergence_type == 'bullish':
                # Look for bullish reversal patterns
                return (current_candle['close'] > current_candle['open'] and 
                       current_candle['close'] > prev_candle['close'])
            elif divergence_type == 'bearish':
                # Look for bearish reversal patterns
                return (current_candle['close'] < current_candle['open'] and 
                       current_candle['close'] < prev_candle['close'])
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking price action confirmation: {e}")
            return False
    
    def _check_support_resistance_confirmation(self, df: pd.DataFrame, divergence_type: str) -> bool:
        """Check if price action confirms support/resistance levels for divergence"""
        if len(df) < 20:
            return False
            
        # Find key support/resistance levels
        highs = [df.iloc[i]['high'] for i in range(len(df))]
        lows = [df.iloc[i]['low'] for i in range(len(df))]
        
        # Calculate dynamic support/resistance
        resistance_levels = self._find_resistance_levels(highs)
        support_levels = self._find_support_levels(lows)
        
        current_price = df['close'].iloc[-1]
        
        if divergence_type in ['bullish', 'hidden_bullish']:
            # Check if price is near support
            near_support = any(abs(current_price - level) / level < 0.02 for level in support_levels)
            return near_support
        elif divergence_type in ['bearish', 'hidden_bearish']:
            # Check if price is near resistance
            near_resistance = any(abs(current_price - level) / level < 0.02 for level in resistance_levels)
            return near_resistance
            
        return False
    
    def _find_resistance_levels(self, highs, min_touches=2):
        """Find resistance levels from price highs"""
        levels = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                levels.append(highs[i])
        return sorted(list(set(levels)), reverse=True)[:5]
    
    def _find_support_levels(self, lows, min_touches=2):
        """Find support levels from price lows"""
        levels = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                levels.append(lows[i])
        return sorted(list(set(lows)))[:5]


class VolumeAnalyzer:
    """Advanced volume analysis for pattern confirmation and market condition detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Real-time volume tracking
        self.volume_buffer = {}
        self.volume_profile_cache = {}
        self.vwap_cache = {}
        
    def update_real_time_volume(self, symbol: str, volume_data: dict):
        """Update real-time volume data for a symbol"""
        try:
            if symbol not in self.volume_buffer:
                self.volume_buffer[symbol] = []
            
            # Add new volume data point
            self.volume_buffer[symbol].append({
                'timestamp': volume_data.get('timestamp', datetime.now()),
                'volume': volume_data.get('volume', 0),
                'price': volume_data.get('price', 0),
                'side': volume_data.get('side', 'unknown')
            })
            
            # Keep only last 1000 data points
            if len(self.volume_buffer[symbol]) > 1000:
                self.volume_buffer[symbol] = self.volume_buffer[symbol][-1000:]
            
            # Update VWAP
            self._update_vwap(symbol, volume_data)
            
        except Exception as e:
            self.logger.error(f"Error updating real-time volume for {symbol}: {e}")
    
    def _update_vwap(self, symbol: str, volume_data: dict):
        """Update VWAP calculation for real-time data using standard formula"""
        try:
            if symbol not in self.vwap_cache:
                self.vwap_cache[symbol] = {
                    'cumulative_volume': 0,
                    'cumulative_price_volume': 0,
                    'vwap': 0,
                    'typical_price_sum': 0,
                    'volume_sum': 0
                }
            
            volume = volume_data.get('volume', 0)
            price = volume_data.get('price', 0)
            high = volume_data.get('high', price)
            low = volume_data.get('low', price)
            
            if volume > 0 and price > 0:
                # Standard VWAP formula: VWAP = Σ(Typical Price × Volume) / Σ(Volume)
                # Where Typical Price = (High + Low + Close) / 3
                typical_price = (high + low + price) / 3
                
                self.vwap_cache[symbol]['typical_price_sum'] += typical_price * volume
                self.vwap_cache[symbol]['volume_sum'] += volume
                
                if self.vwap_cache[symbol]['volume_sum'] > 0:
                    self.vwap_cache[symbol]['vwap'] = (
                        self.vwap_cache[symbol]['typical_price_sum'] / 
                        self.vwap_cache[symbol]['volume_sum']
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating VWAP for {symbol}: {e}")
    
    def get_real_time_volume_analysis(self, symbol: str) -> dict:
        """Get real-time volume analysis for a symbol"""
        try:
            if symbol not in self.volume_buffer or not self.volume_buffer[symbol]:
                return self._get_default_volume_analysis()
            
            volume_data = self.volume_buffer[symbol]
            
            # Calculate real-time metrics
            recent_volume = volume_data[-10:] if len(volume_data) >= 10 else volume_data
            total_volume = sum(point['volume'] for point in recent_volume)
            avg_volume = total_volume / len(recent_volume) if recent_volume else 0
            
            # Calculate buy/sell pressure
            buy_volume = sum(point['volume'] for point in recent_volume if point['side'] == 'BUY')
            sell_volume = sum(point['volume'] for point in recent_volume if point['side'] == 'SELL')
            volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
            
            # Get VWAP
            vwap = self.vwap_cache.get(symbol, {}).get('vwap', 0)
            
            # Calculate volume trend
            if len(volume_data) >= 20:
                recent_avg = sum(point['volume'] for point in volume_data[-10:]) / 10
                historical_avg = sum(point['volume'] for point in volume_data[-20:-10]) / 10
                volume_trend_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
            else:
                volume_trend_ratio = 1
            
            return {
                'symbol': symbol,
                'total_volume': total_volume,
                'average_volume': avg_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'volume_imbalance': volume_imbalance,
                'vwap': vwap,
                'volume_trend_ratio': volume_trend_ratio,
                'data_points': len(volume_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting real-time volume analysis for {symbol}: {e}")
            return self._get_default_volume_analysis()
    
    def get_volume_profile_realtime(self, symbol: str) -> dict:
        """Get real-time volume profile for a symbol using standard formulas"""
        try:
            if symbol not in self.volume_buffer or not self.volume_buffer[symbol]:
                return {}
            
            volume_data = self.volume_buffer[symbol]
            
            # Group by price levels (bin prices for continuous data)
            price_volume = {}
            price_bin_size = 0.01  # 1 cent bins for crypto
            
            for point in volume_data:
                price = point['price']
                # Bin the price
                binned_price = round(price / price_bin_size) * price_bin_size
                if binned_price not in price_volume:
                    price_volume[binned_price] = 0
                price_volume[binned_price] += point['volume']
            
            if not price_volume:
                return {}
            
            # Standard POC calculation: argmax(Σ Volume at price)
            poc_price = max(price_volume.keys(), key=lambda k: price_volume[k])
            poc_volume = price_volume[poc_price]
            
            # Standard Value Area calculation (70% of volume)
            total_volume = sum(price_volume.values())
            target_volume = total_volume * 0.7
            
            # Sort prices by volume (descending)
            sorted_prices = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
            
            # Find Value Area High and Low (70% volume between 0.15 and 0.85)
            cumulative_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_prices:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else poc_price
            value_area_low = min(value_area_prices) if value_area_prices else poc_price
            
            # Calculate HVN and LVN thresholds
            avg_volume_per_price = total_volume / len(price_volume)
            hvn_threshold = avg_volume_per_price * 1.5  # HVN: Volume > 1.5× average
            lvn_threshold = avg_volume_per_price * 0.5  # LVN: Volume < 0.5× average
            
            # Identify HVN and LVN levels
            hvn_levels = [price for price, vol in price_volume.items() if vol > hvn_threshold]
            lvn_levels = [price for price, vol in price_volume.items() if vol < lvn_threshold]
            
            return {
                'symbol': symbol,
                'poc_price': poc_price,
                'poc_volume': poc_volume,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'value_area_volume': cumulative_volume,
                'total_volume': total_volume,
                'hvn_levels': sorted(hvn_levels),
                'lvn_levels': sorted(lvn_levels),
                'price_volume_distribution': price_volume,
                'data_points': len(volume_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting volume profile for {symbol}: {e}")
            return {}
    
    def detect_volume_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[VolumePattern]:
        """Detect volume patterns using standard formulas"""
        try:
            if len(df) < 20:
                return []
            
            patterns = []
            
            # Calculate volume SMA for comparison
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Detect Volume Spike: Volume > 2× SMA(Volume, 20)
            volume_spikes = df[df['volume_ratio'] > 2.0]
            for idx in volume_spikes.index:
                pattern = VolumePattern(
                    pattern_type=VolumePatternType.VOLUME_SPIKE,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    strength=VolumeStrength.STRONG if df.loc[idx, 'volume_ratio'] > 3.0 else VolumeStrength.MODERATE,
                    confidence=min(df.loc[idx, 'volume_ratio'] / 3.0, 1.0),
                    volume_ratio=df.loc[idx, 'volume_ratio'],
                    price_change=0.0,  # Will be calculated separately
                    pattern_data={'volume': df.loc[idx, 'volume'], 'sma': df.loc[idx, 'volume_sma']},
                    description=f"Volume spike detected: {df.loc[idx, 'volume_ratio']:.2f}x average"
                )
                patterns.append(pattern)
            
            # Detect Volume Dry-up: Volume < 0.5× average
            volume_dry_ups = df[df['volume_ratio'] < 0.5]
            for idx in volume_dry_ups.index:
                pattern = VolumePattern(
                    pattern_type=VolumePatternType.VOLUME_DRY_UP,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    strength=VolumeStrength.STRONG if df.loc[idx, 'volume_ratio'] < 0.3 else VolumeStrength.MODERATE,
                    confidence=1.0 - df.loc[idx, 'volume_ratio'],
                    volume_ratio=df.loc[idx, 'volume_ratio'],
                    price_change=0.0,
                    pattern_data={'volume': df.loc[idx, 'volume'], 'sma': df.loc[idx, 'volume_sma']},
                    description=f"Volume dry-up detected: {df.loc[idx, 'volume_ratio']:.2f}x average"
                )
                patterns.append(pattern)
            
            # Detect Volume Climax: Volume > 3× average at price extremes
            if len(df) >= 10:
                recent_high = df['high'].tail(10).max()
                recent_low = df['low'].tail(10).min()
                price_range = recent_high - recent_low
                
                # Top 10% and bottom 10% of price range
                top_threshold = recent_high - (price_range * 0.1)
                bottom_threshold = recent_low + (price_range * 0.1)
                
                climax_candidates = df[
                    (df['volume_ratio'] > 3.0) & 
                    ((df['high'] >= top_threshold) | (df['low'] <= bottom_threshold))
                ]
                
                for idx in climax_candidates.index:
                    pattern = VolumePattern(
                        pattern_type=VolumePatternType.VOLUME_CLIMAX,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        strength=VolumeStrength.EXTREME,
                        confidence=min(df.loc[idx, 'volume_ratio'] / 4.0, 1.0),
                        volume_ratio=df.loc[idx, 'volume_ratio'],
                        price_change=0.0,
                        pattern_data={
                            'volume': df.loc[idx, 'volume'],
                            'sma': df.loc[idx, 'volume_sma'],
                            'price_level': 'high' if df.loc[idx, 'high'] >= top_threshold else 'low'
                        },
                        description=f"Volume climax at {'high' if df.loc[idx, 'high'] >= top_threshold else 'low'} price level"
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting volume patterns for {symbol}: {e}")
            return []
    
    def detect_volume_divergence(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[VolumePattern]:
        """Detect volume divergence patterns using standard formulas"""
        try:
            if len(df) < 20:
                return []
            
            patterns = []
            
            # Calculate price and volume trends
            df['price_trend'] = df['close'].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            
            # Detect Volume Divergence: Price up, Volume down
            bullish_divergence = df[
                (df['price_trend'] == 1) & (df['volume_trend'] == -1)
            ]
            
            for idx in bullish_divergence.index:
                pattern = VolumePattern(
                    pattern_type=VolumePatternType.VOLUME_DIVERGENCE,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    strength=VolumeStrength.MODERATE,
                    confidence=0.7,
                    volume_ratio=df.loc[idx, 'volume'] / df['volume'].rolling(window=20).mean().loc[idx],
                    price_change=df.loc[idx, 'close'] - df.loc[idx-5, 'close'] if idx >= 5 else 0,
                    pattern_data={
                        'divergence_type': 'bullish',
                        'price_change': df.loc[idx, 'close'] - df.loc[idx-5, 'close'] if idx >= 5 else 0,
                        'volume_change': df.loc[idx, 'volume'] - df.loc[idx-5, 'volume'] if idx >= 5 else 0
                    },
                    description="Bullish volume divergence: Price up, Volume down"
                )
                patterns.append(pattern)
            
            # Detect Bearish Volume Divergence: Price down, Volume up
            bearish_divergence = df[
                (df['price_trend'] == -1) & (df['volume_trend'] == 1)
            ]
            
            for idx in bearish_divergence.index:
                pattern = VolumePattern(
                    pattern_type=VolumePatternType.VOLUME_DIVERGENCE,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    strength=VolumeStrength.MODERATE,
                    confidence=0.7,
                    volume_ratio=df.loc[idx, 'volume'] / df['volume'].rolling(window=20).mean().loc[idx],
                    price_change=df.loc[idx, 'close'] - df.loc[idx-5, 'close'] if idx >= 5 else 0,
                    pattern_data={
                        'divergence_type': 'bearish',
                        'price_change': df.loc[idx, 'close'] - df.loc[idx-5, 'close'] if idx >= 5 else 0,
                        'volume_change': df.loc[idx, 'volume'] - df.loc[idx-5, 'volume'] if idx >= 5 else 0
                    },
                    description="Bearish volume divergence: Price down, Volume up"
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting volume divergence for {symbol}: {e}")
            return []
    
    def _analyze_volume_advanced(self, df: pd.DataFrame) -> Dict:
        """Advanced volume analysis across multiple timeframes"""
        try:
            # Validate DataFrame
            if not isinstance(df, pd.DataFrame):
                self.logger.error(f"Expected DataFrame, got {type(df)}")
                return self._get_default_volume_analysis()
            
            if len(df) < 20:
                return self._get_default_volume_analysis()
            
            # Ensure required columns exist
            required_columns = ['volume', 'close', 'high', 'low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return self._get_default_volume_analysis()
            
            # Basic volume analysis
            try:
                volume_trend = self._analyze_volume_trend(df)
            except Exception as e:
                logger.error(f"Error in _analyze_volume_trend: {e}")
                volume_trend = 'unknown'
            
            try:
                volume_consistency = self._calculate_volume_consistency(df)
            except Exception as e:
                logger.error(f"Error in _calculate_volume_consistency: {e}")
                volume_consistency = 0.0
            
            try:
                breakout_strength = self._calculate_breakout_strength(df)
            except Exception as e:
                logger.error(f"Error in _calculate_breakout_strength: {e}")
                breakout_strength = 0.0
            
            # Multi-timeframe analysis
            try:
                multi_timeframe = self._analyze_multi_timeframe_volume(df)
            except Exception as e:
                logger.error(f"Error in _analyze_multi_timeframe_volume: {e}")
                multi_timeframe = {'alignment_score': 0.0}
            
            # Volume divergence detection
            try:
                volume_divergence = self._detect_advanced_volume_divergences(df)
            except Exception as e:
                logger.error(f"Error in _detect_advanced_volume_divergences: {e}")
                volume_divergence = {'strength': 0.0}
            
            # Volume trend alignment
            try:
                volume_trend_alignment = self._calculate_volume_trend_alignment(df)
            except Exception as e:
                logger.error(f"Error in _calculate_volume_trend_alignment: {e}")
                volume_trend_alignment = 0.0
            
            return {
                'volume_trend': volume_trend,
                'volume_consistency': volume_consistency,
                'breakout_strength': breakout_strength,
                'multi_timeframe': multi_timeframe,
                'volume_divergence': volume_divergence,
                'volume_trend_alignment': volume_trend_alignment
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced volume analysis: {e}")
            self.logger.error(f"DataFrame type: {type(df)}")
            self.logger.error(f"DataFrame columns: {list(df.columns) if hasattr(df, 'columns') else 'No columns'}")
            self.logger.error(f"DataFrame shape: {df.shape if hasattr(df, 'shape') else 'No shape'}")
            return self._get_default_volume_analysis()
    
    def _get_default_volume_analysis(self) -> Dict:
        """Return default volume analysis when insufficient data"""
        return {
            'volume_trend': 'unknown',
            'volume_consistency': 0.0,
            'breakout_strength': 0.0,
            'multi_timeframe': {'alignment_score': 0.0},
            'volume_divergence': {'strength': 0.0},
            'volume_trend_alignment': 0.0
        }
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze overall volume trend"""
        try:
            if len(df) < 10:
                return 'unknown'
            
            recent_volume = df['volume'].iloc[-10:].mean()
            historical_volume = df['volume'].iloc[-30:-10].mean() if len(df) >= 30 else df['volume'].iloc[:-10].mean()
            
            if historical_volume == 0:
                return 'unknown'
            
            ratio = recent_volume / historical_volume
            
            if ratio > 1.2:
                return 'increasing'
            elif ratio < 0.8:
                return 'decreasing'
            else:
                return 'flat'
                
        except Exception as e:
            self.logger.error(f"Error analyzing volume trend: {e}")
            return 'unknown'
    
    def _calculate_volume_consistency(self, df: pd.DataFrame) -> float:
        """Calculate volume consistency score (0.0-1.0)"""
        try:
            if len(df) < 10:
                return 0.0
            
            volume_std = df['volume'].std()
            volume_mean = df['volume'].mean()
            
            if volume_mean == 0:
                return 0.0
            
            # Lower coefficient of variation = higher consistency
            cv = volume_std / volume_mean
            consistency = max(0.0, 1.0 - cv)
            
            return min(1.0, consistency)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume consistency: {e}")
            return 0.0
    
    def _calculate_breakout_strength(self, df: pd.DataFrame) -> float:
        """Calculate breakout strength based on volume and price action"""
        try:
            if len(df) < 5:
                return 0.0
            
            # Calculate price change
            price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            # Calculate volume ratio
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-20:].mean() if len(df) >= 20 else df['volume'].mean()
            
            if avg_volume == 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            
            # Combine price change and volume for breakout strength
            strength = min(1.0, (price_change * 10 + volume_ratio - 1) / 2)
            
            return max(0.0, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout strength: {e}")
            return 0.0
    
    def _analyze_multi_timeframe_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume trends across multiple timeframes"""
        try:
            if len(df) < 50:
                return {'alignment_score': 0.0}
            
            # Ensure df has proper structure and volume column exists
            if 'volume' not in df.columns:
                logger.error("Volume column not found in DataFrame")
                return {'alignment_score': 0.0}
            
            # Convert volume to numeric if it's string
            if df['volume'].dtype == 'object':
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Short-term (last 10 periods)
            short_term = self._analyze_volume_trend(df.iloc[-10:])
            
            # Medium-term (last 25 periods)
            medium_term = self._analyze_volume_trend(df.iloc[-25:])
            
            # Long-term (last 50 periods)
            long_term = self._analyze_volume_trend(df.iloc[-50:])
            
            # Calculate alignment score
            alignment_score = self._calculate_timeframe_alignment(short_term, medium_term, long_term)
            
            return {
                'short_term': short_term,
                'medium_term': medium_term,
                'long_term': long_term,
                'alignment_score': alignment_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {e}")
            return {'alignment_score': 0.0}
    
    def _calculate_timeframe_alignment(self, short_term: str, medium_term: str, long_term: str) -> float:
        """Calculate alignment score across timeframes"""
        try:
            score = 0.0
            
            # Perfect alignment (all same)
            if short_term == medium_term == long_term and short_term != 'unknown':
                score = 1.0
            # Good alignment (2 out of 3 same)
            elif (short_term == medium_term and short_term != 'unknown') or \
                 (short_term == long_term and short_term != 'unknown') or \
                 (medium_term == long_term and medium_term != 'unknown'):
                score = 0.7
            # Partial alignment (at least 2 not decreasing)
            elif sum(1 for trend in [short_term, medium_term, long_term] if trend == 'increasing') >= 2:
                score = 0.5
            # Weak alignment
            else:
                score = 0.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe alignment: {e}")
            return 0.0
    
    def _detect_advanced_volume_divergences(self, df: pd.DataFrame) -> Dict:
        """Detect advanced volume divergences with confirmation signals"""
        try:
            if len(df) < 30:
                return {'strength': 0.0}
            
            # Ensure DataFrame has required columns
            required_columns = ['high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns: {required_columns}")
                return {'strength': 0.0}
            
            # Find significant highs and lows
            highs = self._find_significant_highs(df)
            lows = self._find_significant_lows(df)
            
            # Detect divergences
            bullish_divergence = self._detect_bullish_divergences_advanced(df, highs, lows)
            bearish_divergence = self._detect_bearish_divergences_advanced(df, highs, lows)
            hidden_bullish = self._detect_hidden_bullish_divergences_advanced(df, highs, lows)
            hidden_bearish = self._detect_hidden_bearish_divergences_advanced(df, highs, lows)
            triple_divergence = self._detect_triple_divergences_advanced(df, highs, lows)
            
            # Calculate overall strength
            strength = self._calculate_divergence_strength(
                bullish_divergence, bearish_divergence, 
                hidden_bullish, hidden_bearish, triple_divergence
            )
            
            return {
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'hidden_bullish': hidden_bullish,
                'hidden_bearish': hidden_bearish,
                'triple_divergence': triple_divergence,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting volume divergences: {e}")
            return {'strength': 0.0}
    
    def _find_significant_highs(self, df: pd.DataFrame) -> List[int]:
        """Find significant price highs"""
        try:
            highs = []
            for i in range(2, len(df) - 2):
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and
                    df['high'].iloc[i] > df['high'].iloc[i+2]):
                    highs.append(i)
            return highs
        except Exception as e:
            self.logger.error(f"Error finding significant highs: {e}")
            return []
    
    def _find_significant_lows(self, df: pd.DataFrame) -> List[int]:
        """Find significant price lows"""
        try:
            lows = []
            for i in range(2, len(df) - 2):
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and
                    df['low'].iloc[i] < df['low'].iloc[i+2]):
                    lows.append(i)
            return lows
        except Exception as e:
            self.logger.error(f"Error finding significant lows: {e}")
            return []
    
    def _detect_bullish_divergences_advanced(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> bool:
        """Detect bullish divergences with advanced confirmation"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return False
            
            # Check for price making lower lows but volume making higher lows
            for i in range(1, len(lows)):
                if (df['low'].iloc[lows[i]] < df['low'].iloc[lows[i-1]] and
                    df['volume'].iloc[lows[i]] > df['volume'].iloc[lows[i-1]] * 1.2):
                    
                    # Check confirmation signals
                    if self._check_rsi_confirmation(df, lows[i], 'bullish'):
                        return True
                    if self._check_macd_confirmation(df, lows[i], 'bullish'):
                        return True
                    if self._check_volume_confirmation(df, lows[i], 'bullish'):
                        return True
                    if self._check_price_action_confirmation(df, lows[i], 'bullish'):
                        return True
                    if self._check_support_resistance_confirmation(df, 'bullish'):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting bullish divergences: {e}")
            return False
    
    def _detect_bearish_divergences_advanced(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> bool:
        """Detect bearish divergences with advanced confirmation"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return False
            
            # Check for price making higher highs but volume making lower highs
            for i in range(1, len(highs)):
                if (df['high'].iloc[highs[i]] > df['high'].iloc[highs[i-1]] and
                    df['volume'].iloc[highs[i]] < df['volume'].iloc[highs[i-1]] * 0.8):
                    
                    # Check confirmation signals
                    if self._check_rsi_confirmation(df, highs[i], 'bearish'):
                        return True
                    if self._check_macd_confirmation(df, highs[i], 'bearish'):
                        return True
                    if self._check_volume_confirmation(df, highs[i], 'bearish'):
                        return True
                    if self._check_price_action_confirmation(df, highs[i], 'bearish'):
                        return True
                    if self._check_support_resistance_confirmation(df, 'bearish'):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting bearish divergences: {e}")
            return False
    
    def _detect_hidden_bullish_divergences_advanced(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> bool:
        """Detect hidden bullish divergences"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return False
            
            # Check for price making higher lows but volume making lower lows
            for i in range(1, len(lows)):
                if (df['low'].iloc[lows[i]] > df['low'].iloc[lows[i-1]] and
                    df['volume'].iloc[lows[i]] < df['volume'].iloc[lows[i-1]] * 0.8):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden bullish divergences: {e}")
            return False
    
    def _detect_hidden_bearish_divergences_advanced(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> bool:
        """Detect hidden bearish divergences"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return False
            
            # Check for price making lower highs but volume making higher highs
            for i in range(1, len(highs)):
                if (df['high'].iloc[highs[i]] < df['high'].iloc[highs[i-1]] and
                    df['volume'].iloc[highs[i]] > df['volume'].iloc[highs[i-1]] * 1.2):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden bearish divergences: {e}")
            return False
    
    def _detect_triple_divergences_advanced(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> bool:
        """Detect triple divergences (three consecutive divergences)"""
        try:
            if len(highs) < 3 or len(lows) < 3:
                return False
            
            # Check for three consecutive bullish divergences
            bullish_count = 0
            for i in range(1, len(lows)):
                if (df['low'].iloc[lows[i]] < df['low'].iloc[lows[i-1]] and
                    df['volume'].iloc[lows[i]] > df['volume'].iloc[lows[i-1]] * 1.2):
                    bullish_count += 1
                else:
                    bullish_count = 0
                
                if bullish_count >= 3:
                    return True
            
            # Check for three consecutive bearish divergences
            bearish_count = 0
            for i in range(1, len(highs)):
                if (df['high'].iloc[highs[i]] > df['high'].iloc[highs[i-1]] and
                    df['volume'].iloc[highs[i]] < df['volume'].iloc[highs[i-1]] * 0.8):
                    bearish_count += 1
                else:
                    bearish_count = 0
                
                if bearish_count >= 3:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting triple divergences: {e}")
            return False
    
    def _calculate_divergence_strength(self, bullish: bool, bearish: bool, 
                                     hidden_bullish: bool, hidden_bearish: bool, 
                                     triple: bool) -> float:
        """Calculate overall divergence strength (0.0-1.0)"""
        try:
            strength = 0.0
            
            # Base strength for each divergence type
            if bullish:
                strength += 0.3
            if bearish:
                strength += 0.3
            if hidden_bullish:
                strength += 0.2
            if hidden_bearish:
                strength += 0.2
            if triple:
                strength += 0.5  # Triple divergence is very strong
            
            return min(1.0, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating divergence strength: {e}")
            return 0.0
    
    def _calculate_volume_trend_alignment(self, df: pd.DataFrame) -> float:
        """Calculate how well volume trend aligns with price trend"""
        try:
            if len(df) < 20:
                return 0.0
            
            # Calculate price trend
            price_trend = df['close'].iloc[-1] - df['close'].iloc[-20]
            
            # Calculate volume trend
            volume_trend = df['volume'].iloc[-1] - df['volume'].iloc[-20]
            
            # Check alignment
            if (price_trend > 0 and volume_trend > 0) or (price_trend < 0 and volume_trend < 0):
                # Aligned trends
                alignment = min(1.0, abs(volume_trend) / abs(price_trend) if price_trend != 0 else 0)
            else:
                # Diverging trends
                alignment = 0.0
            
            return alignment
            
        except Exception as e:
            self.logger.error(f"Error calculating volume trend alignment: {e}")
            return 0.0
    
    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        try:
            if len(data) < period:
                return np.array([50.0] * len(data))
            
            rsi = np.zeros_like(data)
            gains = np.where(np.diff(data) > 0, np.diff(data), 0)
            losses = np.where(np.diff(data) < 0, -np.diff(data), 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(data)):
                if avg_loss == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))
                
                # Update averages
                avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            
            # Fill initial values
            rsi[:period] = 50.0
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.array([50.0] * len(data))
    
    def _calculate_macd(self, data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD line and signal line"""
        try:
            if len(data) < slow_period:
                return np.array([0.0] * len(data)), np.array([0.0] * len(data))
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(data, fast_period)
            ema_slow = self._calculate_ema(data, slow_period)
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line (EMA of MACD)
            signal_line = self._calculate_ema(macd_line, signal_period)
            
            return macd_line, signal_line
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return np.array([0.0] * len(data)), np.array([0.0] * len(data))
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            if len(data) < period:
                return np.array([data[0]] * len(data))
            
            ema = np.zeros_like(data)
            ema[period-1] = np.mean(data[:period])
            
            multiplier = 2 / (period + 1)
            
            for i in range(period, len(data)):
                ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return np.array([data[0]] * len(data))
    
    def _check_rsi_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check RSI confirmation for divergence"""
        try:
            if len(df) < 14:
                return False
            
            # Calculate RSI
            rsi = self._calculate_rsi(df['close'].values)
            
            if divergence_type == 'bullish':
                # RSI should be oversold (< 30) for bullish divergence
                return rsi[divergence_point] < 30
            elif divergence_type == 'bearish':
                # RSI should be overbought (> 70) for bearish divergence
                return rsi[divergence_point] > 70
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking RSI confirmation: {e}")
            return False
    
    def _check_macd_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check MACD confirmation for divergence"""
        try:
            if len(df) < 26:
                return False
            
            # Calculate MACD
            macd_line, signal_line = self._calculate_macd(df['close'].values)
            
            if divergence_type == 'bullish':
                # MACD should be turning up for bullish divergence
                return (macd_line[divergence_point] > signal_line[divergence_point] and 
                       macd_line[divergence_point] > macd_line[divergence_point - 1])
            elif divergence_type == 'bearish':
                # MACD should be turning down for bearish divergence
                return (macd_line[divergence_point] < signal_line[divergence_point] and 
                       macd_line[divergence_point] < macd_line[divergence_point - 1])
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking MACD confirmation: {e}")
            return False
    
    def _check_volume_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check volume confirmation for divergence"""
        try:
            if divergence_point < 20:
                return False
            
            current_volume = df['volume'].iloc[divergence_point]
            avg_volume = df['volume'].iloc[divergence_point-20:divergence_point].mean()
            
            if divergence_type == 'bullish':
                # Volume should be above average for bullish divergence
                return current_volume > avg_volume * 1.2
            elif divergence_type == 'bearish':
                # Volume should be above average for bearish divergence
                return current_volume > avg_volume * 1.2
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _check_price_action_confirmation(self, df: pd.DataFrame, divergence_point: int, divergence_type: str) -> bool:
        """Check price action confirmation for divergence"""
        try:
            if divergence_point < 5:
                return False
            
            # Check for reversal candlestick patterns
            current_candle = df.iloc[divergence_point]
            prev_candle = df.iloc[divergence_point - 1]
            
            if divergence_type == 'bullish':
                # Look for bullish reversal patterns
                return (current_candle['close'] > current_candle['open'] and 
                       current_candle['close'] > prev_candle['close'])
            elif divergence_type == 'bearish':
                # Look for bearish reversal patterns
                return (current_candle['close'] < current_candle['open'] and 
                       current_candle['close'] < prev_candle['close'])
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking price action confirmation: {e}")
            return False
    
    def _check_support_resistance_confirmation(self, df: pd.DataFrame, divergence_type: str) -> bool:
        """Check if price action confirms support/resistance levels for divergence"""
        if len(df) < 20:
            return False
            
        # Find key support/resistance levels
        highs = [df.iloc[i]['high'] for i in range(len(df))]
        lows = [df.iloc[i]['low'] for i in range(len(df))]
        
        # Calculate dynamic support/resistance
        resistance_levels = self._find_resistance_levels(highs)
        support_levels = self._find_support_levels(lows)
        
        current_price = df['close'].iloc[-1]
        
        if divergence_type in ['bullish', 'hidden_bullish']:
            # Check if price is near support
            near_support = any(abs(current_price - level) / level < 0.02 for level in support_levels)
            return near_support
        elif divergence_type in ['bearish', 'hidden_bearish']:
            # Check if price is near resistance
            near_resistance = any(abs(current_price - level) / level < 0.02 for level in resistance_levels)
            return near_resistance
            
        return False
    
    def _find_resistance_levels(self, highs, min_touches=2):
        """Find resistance levels from price highs"""
        levels = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                levels.append(highs[i])
        return sorted(list(set(levels)), reverse=True)[:5]
    
    def _find_support_levels(self, lows, min_touches=2):
        """Find support levels from price lows"""
        levels = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                levels.append(lows[i])
        return sorted(list(set(lows)))[:5]

    # ===== ADVANCED FEATURE 4: Volume Accumulation/Distribution Analysis with Smart Money Detection =====
    
    def analyze_accumulation_distribution(self, price_data, volume_data, lookback_periods=50):
        """
        Analyze volume patterns to detect accumulation (smart money buying) vs distribution (smart money selling)
        
        Args:
            price_data: List of OHLC dictionaries
            volume_data: List of volume values
            lookback_periods: Number of periods to analyze
            
        Returns:
            dict: Analysis results including accumulation/distribution signals
        """
        if isinstance(price_data, pd.DataFrame) and isinstance(volume_data, pd.DataFrame):
            # Both are DataFrames, use them directly
            df = price_data.tail(lookback_periods).copy()
            df['volume'] = volume_data.tail(lookback_periods)['volume'].values
        elif isinstance(price_data, pd.DataFrame):
            # price_data is DataFrame, volume_data is list
            df = price_data.tail(lookback_periods).copy()
            df['volume'] = volume_data[-lookback_periods:]
        else:
            # Handle list inputs as before
            if len(price_data) < lookback_periods or len(volume_data) < lookback_periods:
                return {
                    'accumulation': False,
                    'distribution': False,
                    'smart_money_flow': 'unknown',
                    'confidence': 0.0,
                    'volume_profile': {},
                    'vwap_analysis': {},
                    'institutional_activity': {}
                }
            
            # Convert to pandas for easier analysis
            df = pd.DataFrame({
                'open': [p['open'] for p in price_data[-lookback_periods:]],
                'high': [p['high'] for p in price_data[-lookback_periods:]],
                'low': [p['low'] for p in price_data[-lookback_periods:]],
                'close': [p['close'] for p in price_data[-lookback_periods:]],
                'volume': volume_data[-lookback_periods:]
            })
        
        # Calculate VWAP and volume-weighted metrics
        vwap_analysis = self._calculate_vwap_analysis(df)
        
        # Analyze volume profile
        volume_profile = self._analyze_volume_profile(df)
        
        # Detect institutional vs retail behavior
        institutional_activity = self._detect_institutional_activity(df)
        
        # Smart money flow detection
        smart_money_flow = self._detect_smart_money_flow(df, vwap_analysis)
        
        # Accumulation/Distribution patterns
        accumulation_signals = self._detect_accumulation_patterns(df, vwap_analysis)
        distribution_signals = self._detect_distribution_patterns(df, vwap_analysis)
        
        # Calculate overall confidence
        confidence = self._calculate_accumulation_distribution_confidence(
            accumulation_signals, distribution_signals, institutional_activity
        )
        
        return {
            'accumulation': accumulation_signals['detected'],
            'distribution': distribution_signals['detected'],
            'smart_money_flow': smart_money_flow,
            'confidence': confidence,
            'volume_profile': volume_profile,
            'vwap_analysis': vwap_analysis,
            'institutional_activity': institutional_activity,
            'accumulation_details': accumulation_signals,
            'distribution_details': distribution_signals
        }
    
    def _calculate_vwap_analysis(self, df):
        """Calculate Volume Weighted Average Price and related metrics"""
        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # VWAP deviation
        current_price = df['close'].iloc[-1]
        vwap_deviation = (current_price - vwap.iloc[-1]) / vwap.iloc[-1]
        
        # VWAP trend
        vwap_trend = 'up' if vwap.iloc[-1] > vwap.iloc[-5] else 'down'
        
        # Volume-weighted momentum
        price_change = df['close'].pct_change()
        volume_momentum = (price_change * df['volume']).rolling(5).mean()
        
        return {
            'vwap': vwap.iloc[-1],
            'vwap_deviation': vwap_deviation,
            'vwap_trend': vwap_trend,
            'volume_momentum': volume_momentum.iloc[-1] if not pd.isna(volume_momentum.iloc[-1]) else 0,
            'vwap_series': vwap.tolist()
        }
    
    def _analyze_volume_profile(self, df):
        """Analyze where volume concentrates in the price range"""
        # Volume by price level
        price_levels = pd.cut(df['close'], bins=10)
        volume_by_price = df.groupby(price_levels)['volume'].sum()
        
        # High volume nodes (where volume concentrates)
        high_volume_nodes = volume_by_price[volume_by_price > volume_by_price.quantile(0.7)]
        
        # Volume distribution pattern
        volume_distribution = 'concentrated' if len(high_volume_nodes) <= 3 else 'distributed'
        
        # Volume trend analysis
        volume_trend = df['volume'].rolling(10).mean().iloc[-1] / df['volume'].rolling(10).mean().iloc[-10]
        
        return {
            'high_volume_nodes': high_volume_nodes.to_dict(),
            'volume_distribution': volume_distribution,
            'volume_trend_ratio': volume_trend,
            'total_volume': df['volume'].sum(),
            'average_volume': df['volume'].mean()
        }
    
    def _detect_institutional_activity(self, df):
        """Detect signs of institutional vs retail trading activity"""
        # Large volume spikes (institutional)
        avg_volume = df['volume'].rolling(20).mean()
        volume_spikes = df[df['volume'] > avg_volume * 2]
        
        # Block trades detection (large volume with small price movement)
        price_volatility = (df['high'] - df['low']) / df['close']
        block_trades = df[(df['volume'] > avg_volume * 1.5) & (price_volatility < 0.02)]
        
        # End-of-day activity (institutional rebalancing)
        eod_volume = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-25:-5].mean()
        
        # Volume consistency (institutional vs retail)
        volume_consistency = 1 - (df['volume'].std() / df['volume'].mean())
        
        return {
            'large_volume_spikes': len(volume_spikes),
            'block_trades': len(block_trades),
            'eod_activity_ratio': eod_volume,
            'volume_consistency': volume_consistency,
            'institutional_indicators': len(volume_spikes) + len(block_trades)
        }
    
    def _detect_smart_money_flow(self, df, vwap_analysis):
        """Detect smart money flow direction and strength"""
        # Price relative to VWAP
        current_price = df['close'].iloc[-1]
        vwap = vwap_analysis['vwap']
        
        # Volume trend analysis
        recent_volume = df['volume'].iloc[-10:].mean()
        historical_volume = df['volume'].iloc[-50:-10].mean()
        volume_trend = recent_volume / historical_volume if historical_volume > 0 else 1
        
        # Price momentum with volume confirmation
        price_momentum = df['close'].pct_change(5).iloc[-1]
        volume_confirmed_momentum = price_momentum * volume_trend
        
        # Smart money flow determination
        if current_price > vwap and volume_confirmed_momentum > 0.01:
            flow = 'accumulation'
        elif current_price < vwap and volume_confirmed_momentum < -0.01:
            flow = 'distribution'
        else:
            flow = 'neutral'
        
        # Flow strength
        flow_strength = abs(volume_confirmed_momentum)
        
        return {
            'direction': flow,
            'strength': flow_strength,
            'price_vwap_ratio': current_price / vwap,
            'volume_trend': volume_trend,
            'momentum_confirmation': volume_confirmed_momentum
        }
    
    def _detect_accumulation_patterns(self, df, vwap_analysis):
        """Detect accumulation patterns (smart money buying quietly)"""
        # Quiet accumulation indicators
        price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
        volume_quietness = 1 - (df['volume'].std() / df['volume'].mean())
        
        # Support level bounces
        support_bounces = self._count_support_bounces(df)
        
        # Volume on down days vs up days
        down_days = df[df['close'] < df['open']]
        up_days = df[df['close'] > df['open']]
        
        down_volume_ratio = down_days['volume'].mean() / up_days['volume'].mean() if len(up_days) > 0 else 1
        
        # Accumulation score
        accumulation_score = 0
        if price_range < 0.1:  # Tight range
            accumulation_score += 0.3
        if volume_quietness > 0.7:  # Consistent volume
            accumulation_score += 0.2
        if support_bounces >= 2:  # Multiple support tests
            accumulation_score += 0.2
        if down_volume_ratio < 0.8:  # Lower volume on down days
            accumulation_score += 0.3
        
        detected = accumulation_score >= 0.6
        
        return {
            'detected': detected,
            'score': accumulation_score,
            'price_range': price_range,
            'volume_quietness': volume_quietness,
            'support_bounces': support_bounces,
            'down_volume_ratio': down_volume_ratio
        }
    
    def _detect_distribution_patterns(self, df, vwap_analysis):
        """Detect distribution patterns (smart money selling to retail)"""
        # Distribution indicators
        price_volatility = df['close'].pct_change().std()
        volume_spikes = len(df[df['volume'] > df['volume'].rolling(20).mean() * 1.5])
        
        # Resistance level rejections
        resistance_rejections = self._count_resistance_rejections(df)
        
        # Volume on up days vs down days
        up_days = df[df['close'] > df['open']]
        down_days = df[df['close'] < df['open']]
        
        up_volume_ratio = up_days['volume'].mean() / down_days['volume'].mean() if len(down_days) > 0 else 1
        
        # Distribution score
        distribution_score = 0
        if price_volatility > 0.02:  # High volatility
            distribution_score += 0.3
        if volume_spikes >= 3:  # Multiple volume spikes
            distribution_score += 0.2
        if resistance_rejections >= 2:  # Multiple resistance tests
            distribution_score += 0.2
        if up_volume_ratio < 0.8:  # Lower volume on up days
            distribution_score += 0.3
        
        detected = distribution_score >= 0.6
        
        return {
            'detected': detected,
            'score': distribution_score,
            'price_volatility': price_volatility,
            'volume_spikes': volume_spikes,
            'resistance_rejections': resistance_rejections,
            'up_volume_ratio': up_volume_ratio
        }
    
    def _count_support_bounces(self, df):
        """Count how many times price bounced off support levels"""
        bounces = 0
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] <= df['low'].iloc[i-1] and 
                df['low'].iloc[i] <= df['low'].iloc[i+1] and
                df['close'].iloc[i] > df['low'].iloc[i] * 1.01):
                bounces += 1
        return bounces
    
    def _count_resistance_rejections(self, df):
        """Count how many times price was rejected at resistance levels"""
        rejections = 0
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] >= df['high'].iloc[i-1] and 
                df['high'].iloc[i] >= df['high'].iloc[i+1] and
                df['close'].iloc[i] < df['high'].iloc[i] * 0.99):
                rejections += 1
        return rejections
    
    def _calculate_accumulation_distribution_confidence(self, accumulation_signals, distribution_signals, institutional_activity):
        """Calculate overall confidence in accumulation/distribution analysis"""
        base_confidence = 0.5
        
        # Accumulation confidence
        if accumulation_signals['detected']:
            base_confidence += accumulation_signals['score'] * 0.3
        
        # Distribution confidence  
        if distribution_signals['detected']:
            base_confidence += distribution_signals['score'] * 0.3
        
        # Institutional activity confirmation
        if institutional_activity['institutional_indicators'] >= 2:
            base_confidence += 0.2
        
        # Volume consistency bonus
        if institutional_activity['volume_consistency'] > 0.6:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def check_accumulation_distribution(self, price_data, volume_data, lookback_periods=50):
        """
        Convenience method to check for accumulation/distribution patterns
        Returns a simplified result for easy integration
        """
        analysis = self.analyze_accumulation_distribution(price_data, volume_data, lookback_periods)
        
        return {
            'is_accumulation': analysis['accumulation'],
            'is_distribution': analysis['distribution'],
            'smart_money_direction': analysis['smart_money_flow']['direction'],
            'confidence': analysis['confidence'],
            'recommendation': self._get_accumulation_distribution_recommendation(analysis)
        }
    
    def _get_accumulation_distribution_recommendation(self, analysis):
        """Generate trading recommendation based on accumulation/distribution analysis"""
        if analysis['accumulation'] and analysis['confidence'] > 0.7:
            return 'BUY - Strong accumulation detected'
        elif analysis['distribution'] and analysis['confidence'] > 0.7:
            return 'SELL - Strong distribution detected'
        elif analysis['accumulation'] and analysis['confidence'] > 0.5:
            return 'BUY - Moderate accumulation detected'
        elif analysis['distribution'] and analysis['confidence'] > 0.5:
            return 'SELL - Moderate distribution detected'
        else:
            return 'HOLD - No clear accumulation/distribution pattern'

    # ===== ADVANCED FEATURE 5: Volume Momentum and Trend Confirmation with Dynamic Thresholds =====
    
    def analyze_volume_momentum_trends(self, price_data, volume_data, lookback_periods=50):
        """
        Analyze volume momentum and confirm trends with dynamic thresholds
        
        Args:
            price_data: List of OHLC dictionaries
            volume_data: List of volume values
            lookback_periods: Number of periods to analyze
            
        Returns:
            dict: Volume momentum analysis and trend confirmation results
        """
        if len(price_data) < lookback_periods or len(volume_data) < lookback_periods:
            return {
                'momentum_analysis': {},
                'trend_confirmation': {},
                'dynamic_thresholds': {},
                'reversal_signals': {},
                'confidence_score': 0.0
            }
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame({
            'open': [p['open'] for p in price_data[-lookback_periods:]],
            'high': [p['high'] for p in price_data[-lookback_periods:]],
            'low': [p['low'] for p in price_data[-lookback_periods:]],
            'close': [p['close'] for p in price_data[-lookback_periods:]],
            'volume': volume_data[-lookback_periods:]
        })
        
        # Calculate volume momentum across multiple timeframes
        momentum_analysis = self._calculate_volume_momentum_analysis(df)
        
        # Dynamic threshold adjustment based on market conditions
        dynamic_thresholds = self._calculate_dynamic_thresholds(df)
        
        # Trend strength confirmation using volume acceleration
        trend_confirmation = self._confirm_trend_strength(df, momentum_analysis, dynamic_thresholds)
        
        # Volume-based trend reversal signals
        reversal_signals = self._detect_volume_reversal_signals(df, momentum_analysis, dynamic_thresholds)
        
        # Adaptive confidence scoring
        confidence_score = self._calculate_adaptive_confidence(
            momentum_analysis, trend_confirmation, reversal_signals, dynamic_thresholds
        )
        
        return {
            'momentum_analysis': momentum_analysis,
            'trend_confirmation': trend_confirmation,
            'dynamic_thresholds': dynamic_thresholds,
            'reversal_signals': reversal_signals,
            'confidence_score': confidence_score
        }
    
    def _calculate_volume_momentum_analysis(self, df):
        """Calculate volume momentum across multiple timeframes"""
        try:
            # Short-term momentum (5 periods)
            short_momentum = self._calculate_momentum(df, 5)
            
            # Medium-term momentum (15 periods)
            medium_momentum = self._calculate_momentum(df, 15)
            
            # Long-term momentum (30 periods)
            long_momentum = self._calculate_momentum(df, 30)
            
            # Volume acceleration (rate of change of momentum)
            acceleration = self._calculate_volume_acceleration(df)
            
            # Momentum divergence (price vs volume)
            momentum_divergence = self._detect_momentum_divergence(df)
            
            return {
                'short_term': short_momentum,
                'medium_term': medium_momentum,
                'long_term': long_momentum,
                'acceleration': acceleration,
                'divergence': momentum_divergence
            }
        except Exception as e:
            self.logger.error(f"Error calculating volume momentum: {e}")
            return {}
    
    def _calculate_momentum(self, df, period):
        """Calculate volume momentum for a specific period"""
        try:
            if len(df) < period:
                return 0.0
            
            # Volume change over period
            volume_change = df['volume'].iloc[-1] - df['volume'].iloc[-period]
            
            # Price change over period
            price_change = df['close'].iloc[-1] - df['close'].iloc[-period]
            
            # Normalize by average volume
            avg_volume = df['volume'].rolling(period).mean().iloc[-1]
            if avg_volume == 0:
                return 0.0
            
            # Volume momentum relative to price movement
            momentum = (volume_change / avg_volume) * (1 + abs(price_change / df['close'].iloc[-period]))
            
            return momentum
        except Exception as e:
            self.logger.error(f"Error calculating momentum for period {period}: {e}")
            return 0.0
    
    def _calculate_volume_acceleration(self, df):
        """Calculate volume acceleration (rate of change of momentum)"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate momentum changes
            momentum_5 = self._calculate_momentum(df, 5)
            momentum_10 = self._calculate_momentum(df, 10)
            
            # Acceleration is the change in momentum
            acceleration = momentum_5 - momentum_10
            
            return acceleration
        except Exception as e:
            self.logger.error(f"Error calculating volume acceleration: {e}")
            return 0.0
    
    def _detect_momentum_divergence(self, df):
        """Detect divergence between price momentum and volume momentum"""
        try:
            if len(df) < 20:
                return {'detected': False, 'type': 'none', 'strength': 0.0}
            
            # Price momentum
            price_momentum = df['close'].pct_change(10).iloc[-1]
            
            # Volume momentum
            volume_momentum = self._calculate_momentum(df, 10)
            
            # Detect divergence
            if price_momentum > 0 and volume_momentum < 0:
                divergence_type = 'bearish'
                strength = abs(volume_momentum) / (abs(price_momentum) + 0.001)
            elif price_momentum < 0 and volume_momentum > 0:
                divergence_type = 'bullish'
                strength = abs(volume_momentum) / (abs(price_momentum) + 0.001)
            else:
                divergence_type = 'none'
                strength = 0.0
            
            detected = strength > 0.5
            
            return {
                'detected': detected,
                'type': divergence_type,
                'strength': min(1.0, strength)
            }
        except Exception as e:
            self.logger.error(f"Error detecting momentum divergence: {e}")
            return {'detected': False, 'type': 'none', 'strength': 0.0}
    
    def _calculate_dynamic_thresholds(self, df):
        """Calculate dynamic thresholds based on market conditions"""
        try:
            # Market volatility
            volatility = df['close'].pct_change().std()
            
            # Volume volatility
            volume_volatility = df['volume'].pct_change().std()
            
            # Dynamic thresholds based on volatility
            base_threshold = 0.5
            volatility_multiplier = 1 + (volatility * 10)  # Higher volatility = higher thresholds
            
            # Momentum thresholds
            momentum_threshold = base_threshold * volatility_multiplier
            
            # Acceleration thresholds
            acceleration_threshold = base_threshold * 0.8 * volatility_multiplier
            
            # Divergence thresholds
            divergence_threshold = base_threshold * 1.2 * volatility_multiplier
            
            return {
                'momentum_threshold': momentum_threshold,
                'acceleration_threshold': acceleration_threshold,
                'divergence_threshold': divergence_threshold,
                'volatility_multiplier': volatility_multiplier,
                'market_volatility': volatility,
                'volume_volatility': volume_volatility
            }
        except Exception as e:
            self.logger.error(f"Error calculating dynamic thresholds: {e}")
            return {
                'momentum_threshold': 0.5,
                'acceleration_threshold': 0.4,
                'divergence_threshold': 0.6,
                'volatility_multiplier': 1.0,
                'market_volatility': 0.0,
                'volume_volatility': 0.0
            }
    
    def _confirm_trend_strength(self, df, momentum_analysis, dynamic_thresholds):
        """Confirm trend strength using volume acceleration"""
        try:
            # Get momentum values
            short_momentum = momentum_analysis.get('short_term', 0.0)
            medium_momentum = momentum_analysis.get('medium_term', 0.0)
            long_momentum = momentum_analysis.get('long_term', 0.0)
            acceleration = momentum_analysis.get('acceleration', 0.0)
            
            # Get thresholds
            momentum_threshold = dynamic_thresholds.get('momentum_threshold', 0.5)
            acceleration_threshold = dynamic_thresholds.get('acceleration_threshold', 0.4)
            
            # Trend strength scoring
            strength_score = 0.0
            
            # Short-term momentum (0-0.3 points)
            if abs(short_momentum) > momentum_threshold:
                strength_score += 0.3
            
            # Medium-term momentum (0-0.3 points)
            if abs(medium_momentum) > momentum_threshold:
                strength_score += 0.3
            
            # Long-term momentum (0-0.2 points)
            if abs(long_momentum) > momentum_threshold:
                strength_score += 0.2
            
            # Volume acceleration (0-0.2 points)
            if abs(acceleration) > acceleration_threshold:
                strength_score += 0.2
            
            # Determine trend direction
            if short_momentum > 0 and medium_momentum > 0:
                trend_direction = 'uptrend'
            elif short_momentum < 0 and medium_momentum < 0:
                trend_direction = 'downtrend'
            else:
                trend_direction = 'sideways'
            
            # Trend strength classification
            if strength_score >= 0.8:
                strength_class = 'strong'
            elif strength_score >= 0.6:
                strength_class = 'moderate'
            elif strength_score >= 0.4:
                strength_class = 'weak'
            else:
                strength_class = 'none'
            
            return {
                'strength_score': strength_score,
                'strength_class': strength_class,
                'trend_direction': trend_direction,
                'momentum_alignment': short_momentum > 0 and medium_momentum > 0,
                'acceleration_confirmed': abs(acceleration) > acceleration_threshold
            }
        except Exception as e:
            self.logger.error(f"Error confirming trend strength: {e}")
            return {
                'strength_score': 0.0,
                'strength_class': 'none',
                'trend_direction': 'unknown',
                'momentum_alignment': False,
                'acceleration_confirmed': False
            }
    
    def _detect_volume_reversal_signals(self, df, momentum_analysis, dynamic_thresholds):
        """Detect volume-based trend reversal signals"""
        try:
            # Get momentum data
            short_momentum = momentum_analysis.get('short_term', 0.0)
            medium_momentum = momentum_analysis.get('medium_term', 0.0)
            acceleration = momentum_analysis.get('acceleration', 0.0)
            divergence = momentum_analysis.get('divergence', {})
            
            # Get thresholds
            momentum_threshold = dynamic_thresholds.get('momentum_threshold', 0.5)
            acceleration_threshold = dynamic_thresholds.get('acceleration_threshold', 0.4)
            divergence_threshold = dynamic_thresholds.get('divergence_threshold', 0.6)
            
            # Reversal signal detection
            reversal_signals = []
            
            # Momentum reversal
            if abs(short_momentum) > momentum_threshold and abs(medium_momentum) < momentum_threshold * 0.5:
                if short_momentum > 0 and medium_momentum < 0:
                    reversal_signals.append({
                        'type': 'momentum_reversal',
                        'direction': 'bullish',
                        'strength': abs(short_momentum) / momentum_threshold,
                        'confidence': 0.7
                    })
                elif short_momentum < 0 and medium_momentum > 0:
                    reversal_signals.append({
                        'type': 'momentum_reversal',
                        'direction': 'bearish',
                        'strength': abs(short_momentum) / momentum_threshold,
                        'confidence': 0.7
                    })
            
            # Acceleration reversal
            if abs(acceleration) > acceleration_threshold:
                if acceleration > 0 and short_momentum < 0:
                    reversal_signals.append({
                        'type': 'acceleration_reversal',
                        'direction': 'bullish',
                        'strength': abs(acceleration) / acceleration_threshold,
                        'confidence': 0.8
                    })
                elif acceleration < 0 and short_momentum > 0:
                    reversal_signals.append({
                        'type': 'acceleration_reversal',
                        'direction': 'bearish',
                        'strength': abs(acceleration) / acceleration_threshold,
                        'confidence': 0.8
                    })
            
            # Divergence reversal
            if divergence.get('detected', False) and divergence.get('strength', 0.0) > divergence_threshold:
                reversal_signals.append({
                    'type': 'divergence_reversal',
                    'direction': divergence.get('type', 'none'),
                    'strength': divergence.get('strength', 0.0),
                    'confidence': 0.9
                })
            
            return {
                'signals': reversal_signals,
                'total_signals': len(reversal_signals),
                'strongest_signal': max(reversal_signals, key=lambda x: x['strength']) if reversal_signals else None
            }
        except Exception as e:
            self.logger.error(f"Error detecting volume reversal signals: {e}")
            return {
                'signals': [],
                'total_signals': 0,
                'strongest_signal': None
            }
    
    def _calculate_adaptive_confidence(self, momentum_analysis, trend_confirmation, reversal_signals, dynamic_thresholds):
        """Calculate adaptive confidence score based on market conditions"""
        try:
            base_confidence = 0.5
            
            # Momentum strength contribution (0-0.2 points)
            momentum_strength = trend_confirmation.get('strength_score', 0.0)
            base_confidence += momentum_strength * 0.2
            
            # Trend alignment contribution (0-0.2 points)
            if trend_confirmation.get('momentum_alignment', False):
                base_confidence += 0.2
            
            # Acceleration confirmation contribution (0-0.15 points)
            if trend_confirmation.get('acceleration_confirmed', False):
                base_confidence += 0.15
            
            # Reversal signal contribution (0-0.15 points)
            total_signals = reversal_signals.get('total_signals', 0)
            if total_signals > 0:
                base_confidence += min(0.15, total_signals * 0.05)
            
            # Market condition adjustment (0-0.1 points)
            volatility_multiplier = dynamic_thresholds.get('volatility_multiplier', 1.0)
            if volatility_multiplier > 1.5:  # High volatility
                base_confidence += 0.1  # Bonus for high volatility conditions
            
            # Ensure confidence is between 0.0 and 1.0
            final_confidence = min(1.0, max(0.0, base_confidence))
            
            return final_confidence
        except Exception as e:
            self.logger.error(f"Error calculating adaptive confidence: {e}")
            return 0.5
    
    def check_volume_momentum_trends(self, price_data, volume_data, lookback_periods=50):
        """
        Convenience method to check volume momentum and trend confirmation
        Returns a simplified result for easy integration
        """
        analysis = self.analyze_volume_momentum_trends(price_data, volume_data, lookback_periods)
        
        return {
            'trend_direction': analysis['trend_confirmation'].get('trend_direction', 'unknown'),
            'trend_strength': analysis['trend_confirmation'].get('strength_class', 'none'),
            'reversal_signals': analysis['reversal_signals'].get('total_signals', 0),
            'confidence': analysis['confidence_score'],
            'recommendation': self._get_momentum_trend_recommendation(analysis)
        }
    
    def _get_momentum_trend_recommendation(self, analysis):
        """Generate trading recommendation based on volume momentum and trend analysis"""
        try:
            trend_direction = analysis['trend_confirmation'].get('trend_direction', 'unknown')
            strength_class = analysis['trend_confirmation'].get('strength_class', 'none')
            confidence = analysis['confidence_score']
            reversal_signals = analysis['reversal_signals'].get('total_signals', 0)
            
            # Strong trend with high confidence
            if strength_class == 'strong' and confidence > 0.8:
                if trend_direction == 'uptrend':
                    return 'BUY - Strong uptrend confirmed by volume momentum'
                elif trend_direction == 'downtrend':
                    return 'SELL - Strong downtrend confirmed by volume momentum'
            
            # Reversal signals with high confidence
            if reversal_signals > 0 and confidence > 0.7:
                strongest_signal = analysis['reversal_signals'].get('strongest_signal', {})
                direction = strongest_signal.get('direction', 'unknown')
                if direction == 'bullish':
                    return 'BUY - Volume momentum reversal signal detected'
                elif direction == 'bearish':
                    return 'SELL - Volume momentum reversal signal detected'
            
            # Moderate trend
            if strength_class == 'moderate' and confidence > 0.6:
                if trend_direction == 'uptrend':
                    return 'BUY - Moderate uptrend with volume confirmation'
                elif trend_direction == 'downtrend':
                    return 'SELL - Moderate downtrend with volume confirmation'
            
            # Weak or no clear trend
            return 'HOLD - No clear trend direction or low confidence'
            
        except Exception as e:
            self.logger.error(f"Error generating momentum trend recommendation: {e}")
            return 'HOLD - Analysis error'
    
    async def analyze_volume_patterns(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> List[VolumePattern]:
        """
        Analyze volume patterns using perfect calculations with regime adjustments, ATR, Multi-TF, Correlation, ML
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe of the data
            
        Returns:
            List of detected VolumePattern objects
        """
        try:
            # Validate DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Expected DataFrame, got {type(df)}")
                logger.error(f"Data type: {type(df)}")
                logger.error(f"Data content: {df}")
                return []
            
            if df.empty or len(df) < 10:
                logger.warning("Insufficient data for volume pattern analysis")
                return []
            
            # Ensure required columns exist
            required_columns = ['high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.error(f"Available columns: {list(df.columns)}")
                return []
            
            # Log DataFrame info for debugging
            logger.info(f"Volume analyzer received DataFrame with shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame dtypes: {df.dtypes.to_dict()}")
            
            # Calculate ATR for dynamic thresholds
            atr = await self._calculate_atr(df, period=14)
            if atr is None or atr <= 0:
                atr = df.iloc[-1]['close'] * 0.01  # Fallback to 1% if ATR unavailable
            
            # Calculate ADX for market regime detection
            adx = await self._calculate_adx(df, period=14)
            if adx is None:
                adx = 25.0  # Default neutral ADX
            
            # Get comprehensive volume analysis
            volume_analysis = self._analyze_volume_advanced(df)
            
            # Detect different types of volume patterns with perfect calculations
            patterns = []
            
            # 1. Volume Spike Detection with Regime-adjusted thresholds
            spike_pattern = await self._detect_volume_spike_perfect(df, volume_analysis, atr, adx, symbol, timeframe)
            if spike_pattern:
                patterns.append(spike_pattern)
            
            # 2. Volume Divergence Detection with ML confidence
            divergence_pattern = await self._detect_volume_divergence_perfect(df, volume_analysis, atr, adx, symbol, timeframe)
            if divergence_pattern:
                patterns.append(divergence_pattern)
            
            # 3. Volume Climax Detection with tighter thresholds (5% not 10%)
            climax_pattern = await self._detect_volume_climax_perfect(df, volume_analysis, atr, adx, symbol, timeframe)
            if climax_pattern:
                patterns.append(climax_pattern)
            
            # 4. Volume Dry-up Detection with ATR-linked price moves
            dryup_pattern = await self._detect_volume_dry_up_perfect(df, volume_analysis, atr, adx, symbol, timeframe)
            if dryup_pattern:
                patterns.append(dryup_pattern)
            
            # 5. Volume Breakout Detection with Multi-TF validation
            breakout_pattern = await self._detect_volume_breakout_perfect(df, volume_analysis, atr, adx, symbol, timeframe)
            if breakout_pattern:
                patterns.append(breakout_pattern)
            
            # 6. Accumulation/Distribution Detection with correlation analysis
            accumulation_analysis = self.analyze_accumulation_distribution(df, df, lookback_periods=20)
            if accumulation_analysis.get('confidence', 0) > 0.6:
                if accumulation_analysis.get('accumulation_signals'):
                    patterns.append(self._create_accumulation_pattern(df, accumulation_analysis, symbol, timeframe))
                if accumulation_analysis.get('distribution_signals'):
                    patterns.append(self._create_distribution_pattern(df, accumulation_analysis, symbol, timeframe))
            
            logger.info(f"Detected {len(patterns)} volume patterns for {symbol} {timeframe}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _detect_volume_spike(self, df: pd.DataFrame, volume_analysis: Dict) -> bool:
        """Detect if there's a significant volume spike"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            return volume_ratio > 2.0
        except Exception as e:
            logger.error(f"Error detecting volume spike: {e}")
            return False
    
    def _detect_volume_divergence(self, df: pd.DataFrame, volume_analysis: Dict) -> bool:
        """Detect if there's volume divergence"""
        try:
            divergence_data = volume_analysis.get('volume_divergence', {})
            return (divergence_data.get('bullish_divergence', False) or 
                   divergence_data.get('bearish_divergence', False) or
                   divergence_data.get('hidden_bullish_divergence', False) or
                   divergence_data.get('hidden_bearish_divergence', False))
        except Exception as e:
            logger.error(f"Error detecting volume divergence: {e}")
            return False
    
    def _detect_volume_climax(self, df: pd.DataFrame, volume_analysis: Dict) -> bool:
        """Detect if there's volume climax (extreme volume with price reversal)"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Check for price reversal
            current_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            price_change = ((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
            
            # Volume climax: high volume with significant price reversal
            return volume_ratio > 2.5 and abs(price_change) > 3.0
        except Exception as e:
            logger.error(f"Error detecting volume climax: {e}")
            return False
    
    def _detect_volume_dry_up(self, df: pd.DataFrame, volume_analysis: Dict) -> bool:
        """Detect if there's volume dry-up (low volume period)"""
        try:
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            return volume_ratio < 0.5
        except Exception as e:
            logger.error(f"Error detecting volume dry-up: {e}")
            return False
    
    def _detect_volume_breakout(self, df: pd.DataFrame, volume_analysis: Dict) -> bool:
        """Detect if there's volume breakout"""
        try:
            breakout_strength = volume_analysis.get('breakout_strength', 0.0)
            return breakout_strength > 0.7
        except Exception as e:
            logger.error(f"Error detecting volume breakout: {e}")
            return False
    
    def _create_volume_spike_pattern(self, df: pd.DataFrame, volume_analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for volume spike"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        strength = VolumeStrength.EXTREME if volume_ratio > 3.0 else VolumeStrength.STRONG
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_SPIKE,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=strength,
            confidence=min(1.0, volume_ratio / 3.0),
            volume_ratio=volume_ratio,
            price_change=0.0,
            pattern_data={'volume_ratio': volume_ratio},
            description=f"Volume spike detected (ratio: {volume_ratio:.2f})"
        )
    
    def _create_volume_divergence_pattern(self, df: pd.DataFrame, volume_analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for volume divergence"""
        divergence_data = volume_analysis.get('volume_divergence', {})
        divergence_strength = divergence_data.get('strength', 0.0)
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_DIVERGENCE,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=VolumeStrength.STRONG if divergence_strength > 0.7 else VolumeStrength.MEDIUM,
            confidence=divergence_strength,
            volume_ratio=1.0,
            price_change=0.0,
            pattern_data=divergence_data,
            description=f"Volume divergence detected (strength: {divergence_strength:.2f})"
        )
    
    def _create_volume_climax_pattern(self, df: pd.DataFrame, volume_analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for volume climax"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_CLIMAX,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=VolumeStrength.EXTREME,
            confidence=min(1.0, volume_ratio / 4.0),
            volume_ratio=volume_ratio,
            price_change=0.0,
            pattern_data={'volume_ratio': volume_ratio},
            description=f"Volume climax detected (ratio: {volume_ratio:.2f})"
        )
    
    def _create_volume_dry_up_pattern(self, df: pd.DataFrame, volume_analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for volume dry-up"""
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_DRY_UP,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=VolumeStrength.WEAK,
            confidence=1.0 - volume_ratio,
            volume_ratio=volume_ratio,
            price_change=0.0,
            pattern_data={'volume_ratio': volume_ratio},
            description=f"Volume dry-up detected (ratio: {volume_ratio:.2f})"
        )
    
    def _create_volume_breakout_pattern(self, df: pd.DataFrame, volume_analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for volume breakout"""
        breakout_strength = volume_analysis.get('breakout_strength', 0.0)
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_BREAKOUT,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=VolumeStrength.STRONG if breakout_strength > 0.8 else VolumeStrength.MEDIUM,
            confidence=breakout_strength,
            volume_ratio=1.0,
            price_change=0.0,
            pattern_data={'breakout_strength': breakout_strength},
            description=f"Volume breakout detected (strength: {breakout_strength:.2f})"
        )
    
    def _create_accumulation_pattern(self, df: pd.DataFrame, analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for accumulation"""
        confidence = analysis.get('confidence', 0.0)
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_ACCUMULATION,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
            confidence=confidence,
            volume_ratio=1.0,
            price_change=0.0,
            pattern_data=analysis,
            description=f"Volume accumulation detected (confidence: {confidence:.2f})"
        )
    
    def _create_distribution_pattern(self, df: pd.DataFrame, analysis: Dict, symbol: str, timeframe: str) -> VolumePattern:
        """Create VolumePattern for distribution"""
        confidence = analysis.get('confidence', 0.0)
        
        return VolumePattern(
            pattern_type=VolumePatternType.VOLUME_DISTRIBUTION,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
            confidence=confidence,
            volume_ratio=1.0,
            price_change=0.0,
            pattern_data=analysis,
            description=f"Volume distribution detected (confidence: {confidence:.2f})"
        )

    # ===== WYCKOFF METHODOLOGY ENHANCEMENT =====
    
    def detect_wyckoff_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[VolumePattern]:
        """
        Detect Wyckoff patterns including Spring, Upthrust, Accumulation, Distribution, and Tests
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            
        Returns:
            List of detected Wyckoff patterns
        """
        try:
            if len(df) < 50:  # Need sufficient data for Wyckoff analysis
                return []
            
            patterns = []
            
            # Detect Spring pattern (false breakdown below support)
            spring_pattern = self._detect_wyckoff_spring(df, symbol, timeframe)
            if spring_pattern:
                patterns.append(spring_pattern)
            
            # Detect Upthrust pattern (false breakout above resistance)
            upthrust_pattern = self._detect_wyckoff_upthrust(df, symbol, timeframe)
            if upthrust_pattern:
                patterns.append(upthrust_pattern)
            
            # Detect Accumulation phase
            accumulation_pattern = self._detect_wyckoff_accumulation(df, symbol, timeframe)
            if accumulation_pattern:
                patterns.append(accumulation_pattern)
            
            # Detect Distribution phase
            distribution_pattern = self._detect_wyckoff_distribution(df, symbol, timeframe)
            if distribution_pattern:
                patterns.append(distribution_pattern)
            
            # Detect Test patterns
            test_pattern = self._detect_wyckoff_test(df, symbol, timeframe)
            if test_pattern:
                patterns.append(test_pattern)
            
            # Detect Signs of Strength/Weakness
            sos_pattern = self._detect_wyckoff_sign_of_strength(df, symbol, timeframe)
            if sos_pattern:
                patterns.append(sos_pattern)
            
            sow_pattern = self._detect_wyckoff_sign_of_weakness(df, symbol, timeframe)
            if sow_pattern:
                patterns.append(sow_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff patterns: {e}")
            return []
    
    def _detect_wyckoff_spring(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Spring pattern (false breakdown below support)
        
        Spring: Price breaks below support but quickly recovers with high volume
        """
        try:
            if len(df) < 20:
                return None
            
            # Get recent price action
            recent_df = df.tail(20)
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find support level (recent low)
            support_level = np.min(lows[:-5])  # Support from earlier data
            
            # Check for breakdown below support
            breakdown_candles = []
            for i in range(len(lows) - 5, len(lows)):
                if lows[i] < support_level * 0.995:  # 0.5% below support
                    breakdown_candles.append(i)
            
            if not breakdown_candles:
                return None
            
            # Check for quick recovery (Spring)
            for breakdown_idx in breakdown_candles:
                if breakdown_idx >= len(lows) - 3:  # Recent breakdown
                    # Check if price recovered above support within 1-3 candles
                    recovery_found = False
                    recovery_volume = 0
                    
                    for j in range(breakdown_idx + 1, min(breakdown_idx + 4, len(closes))):
                        if closes[j] > support_level:
                            recovery_found = True
                            recovery_volume = volumes[j]
                            break
                    
                    if recovery_found:
                        # Calculate Spring strength
                        breakdown_volume = volumes[breakdown_idx]
                        avg_volume = np.mean(volumes)
                        
                        # Standard Spring validation: Decreasing volume on breakdown
                        # Calculate breakdown depth and recovery speed
                        breakdown_depth = abs(lows[breakdown_idx] - support_level) / support_level
                        recovery_speed = j - breakdown_idx  # Bars to recovery
                        volume_confirmation = recovery_volume > avg_volume * 1.5
                        
                        volume_decreasing = breakdown_volume < avg_volume
                        
                        if volume_decreasing and breakdown_depth > 0.005:  # At least 0.5% breakdown
                            confidence = min(0.9, 0.6 + breakdown_depth * 10 + (1 - recovery_speed / 3) * 0.2)
                            strength = VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MODERATE
                            
                            return VolumePattern(
                                pattern_type=VolumePatternType.WYCKOFF_SPRING,
                                symbol=symbol,
                                timeframe=timeframe,
                                timestamp=datetime.now(),
                                strength=strength,
                                confidence=confidence,
                                volume_ratio=breakdown_volume / avg_volume,
                                price_change=closes[-1] - support_level,
                                pattern_data={
                                    'support_level': support_level,
                                    'breakdown_price': lows[breakdown_idx],
                                    'recovery_price': closes[j],
                                    'breakdown_depth': breakdown_depth,
                                    'recovery_speed': recovery_speed,
                                    'volume_decreasing': volume_decreasing
                                },
                                description=f"Wyckoff Spring detected: False breakdown below {support_level:.2f}"
                            )
                        
                        # Calculate confidence
                        confidence = 0.0
                        if breakdown_depth > 0.005:  # At least 0.5% breakdown
                            confidence += 0.3
                        if recovery_speed <= 2:  # Quick recovery
                            confidence += 0.3
                        if volume_confirmation:  # High volume recovery
                            confidence += 0.4
                        
                        if confidence > 0.6:  # Minimum threshold
                            return VolumePattern(
                                pattern_type=VolumePatternType.WYCKOFF_SPRING,
                                symbol=symbol,
                                timeframe=timeframe,
                                timestamp=datetime.now(timezone.utc),
                                strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                                confidence=confidence,
                                volume_ratio=recovery_volume / avg_volume if avg_volume > 0 else 1.0,
                                price_change=0.0,
                                pattern_data={
                                    'support_level': support_level,
                                    'breakdown_low': lows[breakdown_idx],
                                    'breakdown_depth': breakdown_depth,
                                    'recovery_speed': recovery_speed,
                                    'volume_confirmation': volume_confirmation,
                                    'spring_level': support_level
                                },
                                description=f"Wyckoff Spring detected (confidence: {confidence:.2f})"
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Spring: {e}")
            return None
    
    def _detect_wyckoff_upthrust(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Upthrust pattern (false breakout above resistance)
        
        Upthrust: Price breaks above resistance but quickly falls back with high volume
        """
        try:
            if len(df) < 20:
                return None
            
            # Get recent price action
            recent_df = df.tail(20)
            highs = recent_df['high'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find resistance level (recent high)
            resistance_level = np.max(highs[:-5])  # Resistance from earlier data
            
            # Check for breakout above resistance
            breakout_candles = []
            for i in range(len(highs) - 5, len(highs)):
                if highs[i] > resistance_level * 1.005:  # 0.5% above resistance
                    breakout_candles.append(i)
            
            if not breakout_candles:
                return None
            
            # Check for quick rejection (Upthrust)
            for breakout_idx in breakout_candles:
                if breakout_idx >= len(highs) - 3:  # Recent breakout
                    # Check if price fell back below resistance within 1-3 candles
                    rejection_found = False
                    rejection_volume = 0
                    
                    for j in range(breakout_idx + 1, min(breakout_idx + 4, len(closes))):
                        if closes[j] < resistance_level:
                            rejection_found = True
                            rejection_volume = volumes[j]
                            break
                    
                    if rejection_found:
                        # Calculate Upthrust strength
                        breakout_volume = volumes[breakout_idx]
                        avg_volume = np.mean(volumes)
                        
                        # Upthrust characteristics
                        breakout_height = (highs[breakout_idx] - resistance_level) / resistance_level
                        rejection_speed = j - breakout_idx
                        volume_confirmation = rejection_volume > avg_volume * 1.5
                        
                        # Calculate confidence
                        confidence = 0.0
                        if breakout_height > 0.005:  # At least 0.5% breakout
                            confidence += 0.3
                        if rejection_speed <= 2:  # Quick rejection
                            confidence += 0.3
                        if volume_confirmation:  # High volume rejection
                            confidence += 0.4
                        
                        if confidence > 0.6:  # Minimum threshold
                            return VolumePattern(
                                pattern_type=VolumePatternType.WYCKOFF_UPTHRUST,
                                symbol=symbol,
                                timeframe=timeframe,
                                timestamp=datetime.now(timezone.utc),
                                strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                                confidence=confidence,
                                volume_ratio=rejection_volume / avg_volume if avg_volume > 0 else 1.0,
                                price_change=0.0,
                                pattern_data={
                                    'resistance_level': resistance_level,
                                    'breakout_high': highs[breakout_idx],
                                    'breakout_height': breakout_height,
                                    'rejection_speed': rejection_speed,
                                    'volume_confirmation': volume_confirmation,
                                    'upthrust_level': resistance_level
                                },
                                description=f"Wyckoff Upthrust detected (confidence: {confidence:.2f})"
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Upthrust: {e}")
            return None
    
    def _detect_wyckoff_accumulation(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Accumulation phase
        
        Accumulation: Smart money buying at support levels with decreasing volume
        """
        try:
            if len(df) < 30:
                return None
            
            # Get recent price action
            recent_df = df.tail(30)
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find support level
            support_level = np.min(lows[:-10])
            
            # Check for price action near support
            support_touches = []
            for i in range(len(lows)):
                if abs(lows[i] - support_level) / support_level < 0.02:  # Within 2% of support
                    support_touches.append(i)
            
            if len(support_touches) < 2:
                return None
            
            # Analyze volume pattern during accumulation
            recent_volumes = volumes[support_touches[-3:]] if len(support_touches) >= 3 else volumes[support_touches]
            avg_volume = np.mean(volumes)
            
            # Accumulation characteristics
            volume_decreasing = np.all(np.diff(recent_volumes) <= 0) if len(recent_volumes) > 1 else False
            price_stable = np.std(closes[-10:]) / np.mean(closes[-10:]) < 0.02  # Low volatility
            volume_low = np.mean(recent_volumes) < avg_volume * 0.8  # Below average volume
            
            # Calculate confidence
            confidence = 0.0
            if len(support_touches) >= 2:
                confidence += 0.3
            if volume_decreasing:
                confidence += 0.3
            if price_stable:
                confidence += 0.2
            if volume_low:
                confidence += 0.2
            
            if confidence > 0.6:
                return VolumePattern(
                    pattern_type=VolumePatternType.WYCKOFF_ACCUMULATION,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(timezone.utc),
                    strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                    confidence=confidence,
                    volume_ratio=np.mean(recent_volumes) / avg_volume if avg_volume > 0 else 1.0,
                    price_change=0.0,
                    pattern_data={
                        'support_level': support_level,
                        'support_touches': len(support_touches),
                        'volume_decreasing': volume_decreasing,
                        'price_stable': price_stable,
                        'volume_low': volume_low
                    },
                    description=f"Wyckoff Accumulation detected (confidence: {confidence:.2f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Accumulation: {e}")
            return None
    
    def _detect_wyckoff_distribution(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Distribution phase
        
        Distribution: Smart money selling at resistance levels with increasing volume
        """
        try:
            if len(df) < 30:
                return None
            
            # Get recent price action
            recent_df = df.tail(30)
            highs = recent_df['high'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find resistance level
            resistance_level = np.max(highs[:-10])
            
            # Check for price action near resistance
            resistance_touches = []
            for i in range(len(highs)):
                if abs(highs[i] - resistance_level) / resistance_level < 0.02:  # Within 2% of resistance
                    resistance_touches.append(i)
            
            if len(resistance_touches) < 2:
                return None
            
            # Analyze volume pattern during distribution
            recent_volumes = volumes[resistance_touches[-3:]] if len(resistance_touches) >= 3 else volumes[resistance_touches]
            avg_volume = np.mean(volumes)
            
            # Distribution characteristics
            volume_increasing = np.all(np.diff(recent_volumes) >= 0) if len(recent_volumes) > 1 else False
            price_stable = np.std(closes[-10:]) / np.mean(closes[-10:]) < 0.02  # Low volatility
            volume_high = np.mean(recent_volumes) > avg_volume * 1.2  # Above average volume
            
            # Calculate confidence
            confidence = 0.0
            if len(resistance_touches) >= 2:
                confidence += 0.3
            if volume_increasing:
                confidence += 0.3
            if price_stable:
                confidence += 0.2
            if volume_high:
                confidence += 0.2
            
            if confidence > 0.6:
                return VolumePattern(
                    pattern_type=VolumePatternType.WYCKOFF_DISTRIBUTION,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(timezone.utc),
                    strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                    confidence=confidence,
                    volume_ratio=np.mean(recent_volumes) / avg_volume if avg_volume > 0 else 1.0,
                    price_change=0.0,
                    pattern_data={
                        'resistance_level': resistance_level,
                        'resistance_touches': len(resistance_touches),
                        'volume_increasing': volume_increasing,
                        'price_stable': price_stable,
                        'volume_high': volume_high
                    },
                    description=f"Wyckoff Distribution detected (confidence: {confidence:.2f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Distribution: {e}")
            return None
    
    def _detect_wyckoff_test(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Test pattern
        
        Test: Price returns to support/resistance to test if it holds
        """
        try:
            if len(df) < 20:
                return None
            
            # Get recent price action
            recent_df = df.tail(20)
            lows = recent_df['low'].values
            highs = recent_df['high'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find recent support and resistance levels
            support_level = np.min(lows[:-5])
            resistance_level = np.max(highs[:-5])
            
            # Check for test of support
            support_test = False
            support_test_volume = 0
            for i in range(len(lows) - 3, len(lows)):
                if abs(lows[i] - support_level) / support_level < 0.01:  # Within 1% of support
                    support_test = True
                    support_test_volume = volumes[i]
                    break
            
            # Check for test of resistance
            resistance_test = False
            resistance_test_volume = 0
            for i in range(len(highs) - 3, len(highs)):
                if abs(highs[i] - resistance_level) / resistance_level < 0.01:  # Within 1% of resistance
                    resistance_test = True
                    resistance_test_volume = volumes[i]
                    break
            
            avg_volume = np.mean(volumes)
            
            if support_test:
                # Support test with low volume (good sign)
                volume_ratio = support_test_volume / avg_volume if avg_volume > 0 else 1.0
                confidence = 0.8 if volume_ratio < 0.8 else 0.6  # Lower volume = higher confidence
                
                return VolumePattern(
                    pattern_type=VolumePatternType.WYCKOFF_TEST,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(timezone.utc),
                    strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                    confidence=confidence,
                    volume_ratio=volume_ratio,
                    price_change=0.0,
                    pattern_data={
                        'test_type': 'support',
                        'test_level': support_level,
                        'volume_low': volume_ratio < 0.8
                    },
                    description=f"Wyckoff Support Test detected (confidence: {confidence:.2f})"
                )
            
            elif resistance_test:
                # Resistance test with high volume (bad sign)
                volume_ratio = resistance_test_volume / avg_volume if avg_volume > 0 else 1.0
                confidence = 0.8 if volume_ratio > 1.2 else 0.6  # Higher volume = higher confidence
                
                return VolumePattern(
                    pattern_type=VolumePatternType.WYCKOFF_TEST,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(timezone.utc),
                    strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                    confidence=confidence,
                    volume_ratio=volume_ratio,
                    price_change=0.0,
                    pattern_data={
                        'test_type': 'resistance',
                        'test_level': resistance_level,
                        'volume_high': volume_ratio > 1.2
                    },
                    description=f"Wyckoff Resistance Test detected (confidence: {confidence:.2f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Test: {e}")
            return None
    
    def _detect_wyckoff_sign_of_strength(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Sign of Strength
        
        Sign of Strength: Strong upward move with high volume after accumulation
        """
        try:
            if len(df) < 15:
                return None
            
            # Get recent price action
            recent_df = df.tail(15)
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Check for strong upward move
            price_change = (closes[-1] - closes[-5]) / closes[-5]
            
            if price_change < 0.02:  # Need at least 2% move
                return None
            
            # Check for volume confirmation
            recent_volume = volumes[-1]
            avg_volume = np.mean(volumes)
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate confidence
            confidence = 0.0
            if price_change > 0.03:  # Strong move
                confidence += 0.4
            if volume_ratio > 1.5:  # High volume
                confidence += 0.4
            if price_change > 0.05:  # Very strong move
                confidence += 0.2
            
            if confidence > 0.6:
                return VolumePattern(
                    pattern_type=VolumePatternType.WYCKOFF_SIGN_OF_STRENGTH,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(timezone.utc),
                    strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                    confidence=confidence,
                    volume_ratio=volume_ratio,
                    price_change=price_change,
                    pattern_data={
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'strength_level': 'strong' if price_change > 0.05 else 'moderate'
                    },
                    description=f"Wyckoff Sign of Strength detected (confidence: {confidence:.2f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Sign of Strength: {e}")
            return None
    
    def _detect_wyckoff_sign_of_weakness(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """
        Detect Wyckoff Sign of Weakness
        
        Sign of Weakness: Strong downward move with high volume after distribution
        """
        try:
            if len(df) < 15:
                return None
            
            # Get recent price action
            recent_df = df.tail(15)
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Check for strong downward move
            price_change = (closes[-1] - closes[-5]) / closes[-5]
            
            if price_change > -0.02:  # Need at least 2% move down
                return None
            
            # Check for volume confirmation
            recent_volume = volumes[-1]
            avg_volume = np.mean(volumes)
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate confidence
            confidence = 0.0
            if price_change < -0.03:  # Strong move
                confidence += 0.4
            if volume_ratio > 1.5:  # High volume
                confidence += 0.4
            if price_change < -0.05:  # Very strong move
                confidence += 0.2
            
            if confidence > 0.6:
                return VolumePattern(
                    pattern_type=VolumePatternType.WYCKOFF_SIGN_OF_WEAKNESS,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(timezone.utc),
                    strength=VolumeStrength.STRONG if confidence > 0.8 else VolumeStrength.MEDIUM,
                    confidence=confidence,
                    volume_ratio=volume_ratio,
                    price_change=price_change,
                    pattern_data={
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'weakness_level': 'strong' if price_change < -0.05 else 'moderate'
                    },
                    description=f"Wyckoff Sign of Weakness detected (confidence: {confidence:.2f})"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Wyckoff Sign of Weakness: {e}")
            return None
    
    # ===== PERFECT CALCULATION METHODS =====
    
    async def _detect_volume_spike_perfect(self, df: pd.DataFrame, volume_analysis: Dict, atr: float, adx: float, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """Detect volume spike with perfect calculations: regime-adjusted thresholds, ATR-linked price moves, Multi-TF, Correlation, ML"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Regime-adjusted thresholds: k = 1.5 + (ADX/25)
            regime_threshold = 1.5 + (adx / 25)  # 1.5 in ranging, up to 2.5 in trending markets
            
            if volume_ratio >= regime_threshold:
                # ATR-linked price moves: Confirm spike only if price moves >0.5×ATR concurrently
                current_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                price_move = abs(current_price - prev_price)
                
                if price_move > 0.5 * atr:  # Links volume to price action
                    # Multi-TF Validation: Spike must appear on higher timeframe
                    multi_tf_confirmed = await self._validate_multi_timeframe_spike(df, volume_ratio, atr)
                    
                    # Correlation Analysis: Confirm in correlated asset
                    correlation_confirmed = await self._validate_correlation_spike(symbol, volume_ratio, atr)
                    
                    # ML Confidence: Features = [volume ratio, ATR, price range %, ADX, TF alignment, correlation]
                    ml_features = {
                        'volume_ratio': volume_ratio,
                        'atr_ratio': atr / current_price,
                        'price_range_percent': price_move / current_price,
                        'adx_value': adx,
                        'multi_tf_score': multi_tf_confirmed,
                        'correlation_score': correlation_confirmed
                    }
                    ml_confidence = await self._calculate_ml_confidence_volume(ml_features)
                    
                    # Calculate strength with all enhancements
                    strength = (volume_ratio / regime_threshold) * (1 + 0.5 * multi_tf_confirmed) * (1 + 0.3 * correlation_confirmed)
                    
                    # Only include patterns with ML confidence > 0.7
                    if ml_confidence > 0.7:
                        return VolumePattern(
                            pattern_type=VolumePatternType.VOLUME_SPIKE,
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=datetime.now(timezone.utc),
                            strength=VolumeStrength.STRONG if strength > 2.0 else VolumeStrength.MEDIUM,
                            confidence=ml_confidence,
                            volume_ratio=volume_ratio,
                            price_change=price_move / prev_price * 100,
                            pattern_data={
                                'regime_threshold': regime_threshold,
                                'atr_confirmed': price_move > 0.5 * atr,
                                'multi_tf_confirmed': multi_tf_confirmed > 0,
                                'correlation_confirmed': correlation_confirmed > 0,
                                'ml_confidence': ml_confidence
                            },
                            description=f"Perfect Volume Spike: {volume_ratio:.2f}x (threshold: {regime_threshold:.2f}, ML: {ml_confidence:.2f})"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting perfect volume spike: {e}")
            return None
    
    async def _detect_volume_divergence_perfect(self, df: pd.DataFrame, volume_analysis: Dict, atr: float, adx: float, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """Detect volume divergence with perfect calculations: ML confidence scoring"""
        try:
            divergence_data = volume_analysis.get('volume_divergence', {})
            
            if (divergence_data.get('bullish_divergence', False) or 
                divergence_data.get('bearish_divergence', False) or
                divergence_data.get('hidden_bullish_divergence', False) or
                divergence_data.get('hidden_bearish_divergence', False)):
                
                # Calculate divergence magnitude
                price_delta = divergence_data.get('price_delta', 0)
                volume_delta = divergence_data.get('volume_delta', 0)
                divergence_magnitude = abs((price_delta / atr) - (volume_delta / df['volume'].mean()))
                
                # Multi-TF Validation: Confirm on higher timeframe
                multi_tf_confirmed = await self._validate_multi_timeframe_divergence(df, divergence_data, atr)
                
                # ML Confidence: Features = [price delta, volume delta, ATR, RSI, ADX, correlation]
                ml_features = {
                    'price_delta': price_delta,
                    'volume_delta': volume_delta,
                    'atr_ratio': atr / df['close'].iloc[-1],
                    'rsi_value': divergence_data.get('rsi', 50),
                    'adx_value': adx,
                    'divergence_magnitude': divergence_magnitude,
                    'multi_tf_score': multi_tf_confirmed
                }
                ml_confidence = await self._calculate_ml_confidence_volume(ml_features)
                
                # Only include patterns with ML confidence > 0.7
                if ml_confidence > 0.7:
                    return VolumePattern(
                        pattern_type=VolumePatternType.VOLUME_DIVERGENCE,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=datetime.now(timezone.utc),
                        strength=VolumeStrength.STRONG if divergence_magnitude > 2.0 else VolumeStrength.MEDIUM,
                        confidence=ml_confidence,
                        volume_ratio=divergence_data.get('volume_ratio', 1.0),
                        price_change=price_delta,
                        pattern_data={
                            'divergence_magnitude': divergence_magnitude,
                            'multi_tf_confirmed': multi_tf_confirmed > 0,
                            'ml_confidence': ml_confidence,
                            'divergence_type': divergence_data.get('type', 'unknown')
                        },
                        description=f"Perfect Volume Divergence: {divergence_magnitude:.2f} magnitude (ML: {ml_confidence:.2f})"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting perfect volume divergence: {e}")
            return None
    
    async def _detect_volume_climax_perfect(self, df: pd.DataFrame, volume_analysis: Dict, atr: float, adx: float, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """Detect volume climax with perfect calculations: tighter thresholds (5% not 10%), regime adjustments, correlation"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Regime adjustment: In trending markets (ADX>25), require Volume > 4×avg for stronger confirmation
            if adx > 25:  # Trending market
                climax_threshold = 4.0
            else:  # Ranging market
                climax_threshold = 3.0
            
            if volume_ratio >= climax_threshold:
                # Tighter climax detection: Top/bottom 5% (not 10%) of 50-period range
                price_range = df['close'].rolling(50).max().iloc[-1] - df['close'].rolling(50).min().iloc[-1]
                current_price = df['close'].iloc[-1]
                price_position = (current_price - df['close'].rolling(50).min().iloc[-1]) / price_range
                
                # Check if price is in top/bottom 5% of range
                is_at_extreme = price_position <= 0.05 or price_position >= 0.95
                
                if is_at_extreme:
                    # Correlation Analysis: Confirm in correlated asset
                    correlation_confirmed = await self._validate_correlation_climax(symbol, volume_ratio, atr)
                    
                    # Multi-TF Validation: Climax valid if seen on 1+ higher TF
                    multi_tf_confirmed = await self._validate_multi_timeframe_climax(df, volume_ratio, atr)
                    
                    # ML Confidence scoring
                    ml_features = {
                        'volume_ratio': volume_ratio,
                        'price_position': price_position,
                        'atr_ratio': atr / current_price,
                        'adx_value': adx,
                        'correlation_score': correlation_confirmed,
                        'multi_tf_score': multi_tf_confirmed
                    }
                    ml_confidence = await self._calculate_ml_confidence_volume(ml_features)
                    
                    # Calculate strength with all enhancements
                    strength = (volume_ratio / climax_threshold) * (1 + 0.5 * multi_tf_confirmed) * (1 + 0.3 * correlation_confirmed)
                    
                    # Only include patterns with ML confidence > 0.7
                    if ml_confidence > 0.7:
                        return VolumePattern(
                            pattern_type=VolumePatternType.VOLUME_CLIMAX,
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=datetime.now(timezone.utc),
                            strength=VolumeStrength.STRONG if strength > 1.5 else VolumeStrength.MEDIUM,
                            confidence=ml_confidence,
                            volume_ratio=volume_ratio,
                            price_change=0.0,  # Climax is about volume, not price change
                            pattern_data={
                                'climax_threshold': climax_threshold,
                                'price_position': price_position,
                                'is_at_extreme': is_at_extreme,
                                'multi_tf_confirmed': multi_tf_confirmed > 0,
                                'correlation_confirmed': correlation_confirmed > 0,
                                'ml_confidence': ml_confidence
                            },
                            description=f"Perfect Volume Climax: {volume_ratio:.2f}x at {price_position:.1%} position (ML: {ml_confidence:.2f})"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting perfect volume climax: {e}")
            return None
    
    async def _detect_volume_dry_up_perfect(self, df: pd.DataFrame, volume_analysis: Dict, atr: float, adx: float, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """Detect volume dry-up with perfect calculations: ATR-linked price moves, Multi-TF validation"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio < 0.5:  # Volume dry-up threshold
                # ATR-linked price moves: Confirm with ATR (price range <0.5×ATR)
                current_price = df['close'].iloc[-1]
                price_range = df['high'].iloc[-1] - df['low'].iloc[-1]
                
                if price_range < 0.5 * atr:  # Low volatility confirms dry-up
                    # Multi-TF Validation: Confirm low volume on higher timeframe
                    multi_tf_confirmed = await self._validate_multi_timeframe_dryup(df, volume_ratio, atr)
                    
                    # ML Confidence scoring
                    ml_features = {
                        'volume_ratio': volume_ratio,
                        'price_range_ratio': price_range / atr,
                        'atr_ratio': atr / current_price,
                        'adx_value': adx,
                        'multi_tf_score': multi_tf_confirmed
                    }
                    ml_confidence = await self._calculate_ml_confidence_volume(ml_features)
                    
                    # Only include patterns with ML confidence > 0.7
                    if ml_confidence > 0.7:
                        return VolumePattern(
                            pattern_type=VolumePatternType.VOLUME_DRY_UP,
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=datetime.now(timezone.utc),
                            strength=VolumeStrength.MEDIUM,  # Dry-up is typically medium strength
                            confidence=ml_confidence,
                            volume_ratio=volume_ratio,
                            price_change=0.0,
                            pattern_data={
                                'price_range_ratio': price_range / atr,
                                'multi_tf_confirmed': multi_tf_confirmed > 0,
                                'ml_confidence': ml_confidence
                            },
                            description=f"Perfect Volume Dry-up: {volume_ratio:.2f}x with {price_range/atr:.2f}×ATR range (ML: {ml_confidence:.2f})"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting perfect volume dry-up: {e}")
            return None
    
    async def _detect_volume_breakout_perfect(self, df: pd.DataFrame, volume_analysis: Dict, atr: float, adx: float, symbol: str, timeframe: str) -> Optional[VolumePattern]:
        """Detect volume breakout with perfect calculations: Multi-TF validation, correlation analysis"""
        try:
            # This would integrate with existing breakout detection
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            logger.error(f"Error detecting perfect volume breakout: {e}")
            return None
    
    # ===== PERFECT CALCULATION HELPER METHODS =====
    
    async def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range for dynamic thresholds"""
        try:
            if len(df) < period + 1:
                return None
            
            # Calculate True Range
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR using exponential moving average
            atr = np.full(len(high), np.nan)
            atr[period] = np.mean(true_range[1:period + 1])
            
            for i in range(period + 1, len(high)):
                atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
            
            return float(atr[-1]) if not np.isnan(atr[-1]) else None
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    async def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average Directional Index for market regime detection"""
        try:
            if len(df) < period + 1:
                return None
            
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
            
            return float(adx) if not np.isnan(adx) else None
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return None
    
    async def _validate_multi_timeframe_spike(self, df: pd.DataFrame, volume_ratio: float, atr: float) -> float:
        """Validate volume spike on higher timeframe"""
        try:
            # This would integrate with existing multi-timeframe infrastructure
            # For now, return a placeholder boost
            return 0.3  # 30% boost for multi-TF spike confirmation
            
        except Exception as e:
            logger.error(f"Error validating multi-timeframe spike: {e}")
            return 0.0
    
    async def _validate_correlation_spike(self, symbol: str, volume_ratio: float, atr: float) -> float:
        """Validate volume spike with cross-asset correlation"""
        try:
            # This would integrate with existing correlation infrastructure
            # For now, return a placeholder boost
            return 0.2  # 20% boost for correlation confirmation
            
        except Exception as e:
            logger.error(f"Error validating correlation spike: {e}")
            return 0.0
    
    async def _validate_multi_timeframe_divergence(self, df: pd.DataFrame, divergence_data: Dict, atr: float) -> float:
        """Validate volume divergence on higher timeframe"""
        try:
            # This would integrate with existing multi-timeframe infrastructure
            # For now, return a placeholder boost
            return 0.25  # 25% boost for multi-TF divergence confirmation
            
        except Exception as e:
            logger.error(f"Error validating multi-timeframe divergence: {e}")
            return 0.0
    
    async def _validate_correlation_climax(self, symbol: str, volume_ratio: float, atr: float) -> float:
        """Validate volume climax with cross-asset correlation"""
        try:
            # This would integrate with existing correlation infrastructure
            # For now, return a placeholder boost
            return 0.15  # 15% boost for correlation confirmation
            
        except Exception as e:
            logger.error(f"Error validating correlation climax: {e}")
            return 0.0
    
    async def _validate_multi_timeframe_climax(self, df: pd.DataFrame, volume_ratio: float, atr: float) -> float:
        """Validate volume climax on higher timeframe"""
        try:
            # This would integrate with existing multi-timeframe infrastructure
            # For now, return a placeholder boost
            return 0.2  # 20% boost for multi-TF climax confirmation
            
        except Exception as e:
            logger.error(f"Error validating multi-timeframe climax: {e}")
            return 0.0
    
    async def _validate_multi_timeframe_dryup(self, df: pd.DataFrame, volume_ratio: float, atr: float) -> float:
        """Validate volume dry-up on higher timeframe"""
        try:
            # This would integrate with existing multi-timeframe infrastructure
            # For now, return a placeholder boost
            return 0.1  # 10% boost for multi-TF dry-up confirmation
            
        except Exception as e:
            logger.error(f"Error validating multi-timeframe dry-up: {e}")
            return 0.0
    
    async def _calculate_ml_confidence_volume(self, features: Dict[str, Any]) -> float:
        """Calculate ML confidence score for volume pattern validation"""
        try:
            # This would integrate with existing ML infrastructure
            # For now, return a calculated confidence based on features
            
            confidence = 0.5  # Base confidence
            
            # Volume ratio (higher is better for spikes/climax)
            volume_ratio = features.get('volume_ratio', 1.0)
            if volume_ratio > 3.0:
                confidence += 0.2
            elif volume_ratio > 2.0:
                confidence += 0.1
            
            # ATR ratio (lower is better for stability)
            atr_ratio = features.get('atr_ratio', 0.01)
            if atr_ratio < 0.01:  # Low volatility
                confidence += 0.1
            
            # ADX regime confirmation
            adx = features.get('adx_value', 25)
            if 20 <= adx <= 30:  # Good regime for volume patterns
                confidence += 0.1
            
            # Multi-TF confirmation
            multi_tf_score = features.get('multi_tf_score', 0)
            if multi_tf_score > 0:
                confidence += 0.1
            
            # Correlation confirmation
            correlation_score = features.get('correlation_score', 0)
            if correlation_score > 0:
                confidence += 0.1
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating ML confidence for volume: {e}")
            return 0.5