#!/usr/bin/env python3
"""
Volume Divergence Detector for AlphaPulse
Advanced volume-price divergence detection for pattern confirmation
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

from data.volume_analyzer import VolumeAnalyzer, VolumePattern, VolumePatternType, VolumeStrength

logger = logging.getLogger(__name__)

class DivergenceType(Enum):
    """Types of volume-price divergence"""
    POSITIVE_DIVERGENCE = "positive_divergence"  # Price down, volume up (bullish)
    NEGATIVE_DIVERGENCE = "negative_divergence"  # Price up, volume down (bearish)
    HIDDEN_POSITIVE = "hidden_positive"  # Price up, volume down (bullish continuation)
    HIDDEN_NEGATIVE = "hidden_negative"  # Price down, volume up (bearish continuation)
    NO_DIVERGENCE = "no_divergence"

class DivergenceStrength(Enum):
    """Strength levels of divergence"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"

@dataclass
class VolumeDivergenceSignal:
    """Volume divergence detection signal"""
    divergence_type: DivergenceType
    strength: DivergenceStrength
    confidence: float  # 0.0-1.0
    price_change: float
    volume_change: float
    divergence_score: float  # -1.0 to 1.0 (negative = bearish, positive = bullish)
    timestamp: datetime
    pattern_alignment: str  # "bullish", "bearish", "neutral"
    description: str
    metadata: Dict[str, Any]

class VolumeDivergenceDetector:
    """
    Advanced volume divergence detector for pattern confirmation
    """
    
    def __init__(self):
        self.volume_analyzer = VolumeAnalyzer()
        
        # Divergence detection parameters
        self.divergence_thresholds = {
            'weak': 0.1,      # 10% change
            'moderate': 0.2,  # 20% change
            'strong': 0.3,    # 30% change
            'extreme': 0.5    # 50% change
        }
        
        # Time windows for divergence analysis
        self.analysis_windows = {
            'short_term': 5,   # 5 periods
            'medium_term': 10, # 10 periods
            'long_term': 20    # 20 periods
        }
        
        # Confidence multipliers for different divergence types
        self.divergence_multipliers = {
            DivergenceType.POSITIVE_DIVERGENCE: 1.2,    # Boost for bullish divergence
            DivergenceType.NEGATIVE_DIVERGENCE: 0.8,    # Penalty for bearish divergence
            DivergenceType.HIDDEN_POSITIVE: 1.1,        # Small boost for hidden bullish
            DivergenceType.HIDDEN_NEGATIVE: 0.9,        # Small penalty for hidden bearish
            DivergenceType.NO_DIVERGENCE: 1.0           # No change
        }
        
        logger.info("🚀 Volume Divergence Detector initialized")
    
    def detect_volume_divergence(
        self, 
        df: pd.DataFrame, 
        pattern_type: str = "unknown",
        pattern_direction: str = "neutral"
    ) -> Optional[VolumeDivergenceSignal]:
        """
        Detect volume-price divergence for pattern confirmation
        
        Args:
            df: DataFrame with OHLCV data
            pattern_type: Type of pattern being analyzed
            pattern_direction: Direction of pattern ("bullish", "bearish", "neutral")
            
        Returns:
            VolumeDivergenceSignal if divergence detected, None otherwise
        """
        try:
            if len(df) < 20:
                logger.warning("Insufficient data for divergence detection (minimum 20 candles required)")
                return None
            
            # Analyze price and volume trends
            price_analysis = self._analyze_price_trends(df)
            volume_analysis = self._analyze_volume_trends(df)
            
            # Detect divergence
            divergence_info = self._detect_divergence_pattern(price_analysis, volume_analysis)
            
            if divergence_info['divergence_type'] == DivergenceType.NO_DIVERGENCE:
                return None
            
            # Calculate divergence score and confidence
            divergence_score = self._calculate_divergence_score(divergence_info, pattern_direction)
            confidence = self._calculate_divergence_confidence(divergence_info)
            
            # Determine strength
            strength = self._determine_divergence_strength(divergence_info)
            
            # Create divergence signal
            divergence_signal = VolumeDivergenceSignal(
                divergence_type=divergence_info['divergence_type'],
                strength=strength,
                confidence=confidence,
                price_change=divergence_info['price_change'],
                volume_change=divergence_info['volume_change'],
                divergence_score=divergence_score,
                timestamp=datetime.now(),
                pattern_alignment=self._determine_pattern_alignment(divergence_info, pattern_direction),
                description=self._generate_divergence_description(divergence_info),
                metadata=divergence_info
            )
            
            logger.info(f"🔍 Detected {divergence_signal.divergence_type.value} divergence with {divergence_signal.strength.value} strength")
            
            return divergence_signal
            
        except Exception as e:
            logger.error(f"Error detecting volume divergence: {e}")
            return None
    
    def _analyze_price_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price trends across different timeframes
        """
        try:
            analysis = {}
            
            for window_name, window_size in self.analysis_windows.items():
                if len(df) < window_size:
                    continue
                
                window_data = df.tail(window_size)
                
                # Calculate price change
                price_change = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
                
                # Calculate price momentum
                price_momentum = self._calculate_momentum(window_data['close'])
                
                # Calculate price volatility
                price_volatility = window_data['close'].std() / window_data['close'].mean()
                
                # Determine trend direction
                if price_change > 0.02:  # 2% increase
                    trend_direction = "up"
                elif price_change < -0.02:  # 2% decrease
                    trend_direction = "down"
                else:
                    trend_direction = "sideways"
                
                analysis[window_name] = {
                    'price_change': price_change,
                    'price_momentum': price_momentum,
                    'price_volatility': price_volatility,
                    'trend_direction': trend_direction,
                    'highs': window_data['high'].max(),
                    'lows': window_data['low'].min()
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing price trends: {e}")
            return {}
    
    def _analyze_volume_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume trends across different timeframes
        """
        try:
            analysis = {}
            
            for window_name, window_size in self.analysis_windows.items():
                if len(df) < window_size:
                    continue
                
                window_data = df.tail(window_size)
                
                # Calculate volume change
                volume_change = (window_data['volume'].iloc[-1] - window_data['volume'].iloc[0]) / window_data['volume'].iloc[0]
                
                # Calculate volume momentum
                volume_momentum = self._calculate_momentum(window_data['volume'])
                
                # Calculate volume consistency
                volume_consistency = 1.0 - (window_data['volume'].std() / window_data['volume'].mean())
                
                # Determine volume trend
                if volume_change > 0.1:  # 10% increase
                    volume_trend = "increasing"
                elif volume_change < -0.1:  # 10% decrease
                    volume_trend = "decreasing"
                else:
                    volume_trend = "stable"
                
                analysis[window_name] = {
                    'volume_change': volume_change,
                    'volume_momentum': volume_momentum,
                    'volume_consistency': volume_consistency,
                    'volume_trend': volume_trend,
                    'avg_volume': window_data['volume'].mean(),
                    'volume_peaks': self._find_volume_peaks(window_data)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volume trends: {e}")
            return {}
    
    def _detect_divergence_pattern(
        self, 
        price_analysis: Dict[str, Any], 
        volume_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect divergence patterns between price and volume
        """
        try:
            # Use medium-term analysis for primary divergence detection
            if 'medium_term' not in price_analysis or 'medium_term' not in volume_analysis:
                return self._get_no_divergence_result()
            
            price_info = price_analysis['medium_term']
            volume_info = volume_analysis['medium_term']
            
            price_change = price_info['price_change']
            volume_change = volume_info['volume_change']
            price_trend = price_info['trend_direction']
            volume_trend = volume_info['volume_trend']
            
            # Detect divergence patterns
            divergence_type = DivergenceType.NO_DIVERGENCE
            
            # Positive divergence: Price down, volume up (bullish signal)
            if price_change < -0.05 and volume_change > 0.1:
                divergence_type = DivergenceType.POSITIVE_DIVERGENCE
            
            # Negative divergence: Price up, volume down (bearish signal)
            elif price_change > 0.05 and volume_change < -0.1:
                divergence_type = DivergenceType.NEGATIVE_DIVERGENCE
            
            # Hidden positive: Price up, volume down (bullish continuation)
            elif price_change > 0.05 and volume_change < -0.05:
                divergence_type = DivergenceType.HIDDEN_POSITIVE
            
            # Hidden negative: Price down, volume up (bearish continuation)
            elif price_change < -0.05 and volume_change > 0.05:
                divergence_type = DivergenceType.HIDDEN_NEGATIVE
            
            # Check for confirmation from other timeframes
            confirmation_score = self._check_divergence_confirmation(
                price_analysis, volume_analysis, divergence_type
            )
            
            return {
                'divergence_type': divergence_type,
                'price_change': price_change,
                'volume_change': volume_change,
                'price_trend': price_trend,
                'volume_trend': volume_trend,
                'confirmation_score': confirmation_score,
                'price_analysis': price_analysis,
                'volume_analysis': volume_analysis
            }
            
        except Exception as e:
            logger.error(f"Error detecting divergence pattern: {e}")
            return self._get_no_divergence_result()
    
    def _check_divergence_confirmation(
        self, 
        price_analysis: Dict[str, Any], 
        volume_analysis: Dict[str, Any], 
        divergence_type: DivergenceType
    ) -> float:
        """
        Check if divergence is confirmed across multiple timeframes
        """
        try:
            confirmation_score = 0.0
            total_checks = 0
            
            # Check short-term confirmation
            if 'short_term' in price_analysis and 'short_term' in volume_analysis:
                short_confirmation = self._check_timeframe_confirmation(
                    price_analysis['short_term'], 
                    volume_analysis['short_term'], 
                    divergence_type
                )
                confirmation_score += short_confirmation
                total_checks += 1
            
            # Check long-term confirmation
            if 'long_term' in price_analysis and 'long_term' in volume_analysis:
                long_confirmation = self._check_timeframe_confirmation(
                    price_analysis['long_term'], 
                    volume_analysis['long_term'], 
                    divergence_type
                )
                confirmation_score += long_confirmation
                total_checks += 1
            
            return confirmation_score / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error checking divergence confirmation: {e}")
            return 0.0
    
    def _check_timeframe_confirmation(
        self, 
        price_info: Dict[str, Any], 
        volume_info: Dict[str, Any], 
        divergence_type: DivergenceType
    ) -> float:
        """
        Check if a specific timeframe confirms the divergence
        """
        try:
            price_change = price_info['price_change']
            volume_change = volume_info['volume_change']
            
            if divergence_type == DivergenceType.POSITIVE_DIVERGENCE:
                # Confirm positive divergence: price down, volume up
                if price_change < 0 and volume_change > 0:
                    return 1.0
                elif price_change < 0 and volume_change > -0.05:
                    return 0.5
                    
            elif divergence_type == DivergenceType.NEGATIVE_DIVERGENCE:
                # Confirm negative divergence: price up, volume down
                if price_change > 0 and volume_change < 0:
                    return 1.0
                elif price_change > 0 and volume_change < 0.05:
                    return 0.5
                    
            elif divergence_type == DivergenceType.HIDDEN_POSITIVE:
                # Confirm hidden positive: price up, volume down
                if price_change > 0 and volume_change < 0:
                    return 1.0
                elif price_change > 0 and volume_change < 0.05:
                    return 0.5
                    
            elif divergence_type == DivergenceType.HIDDEN_NEGATIVE:
                # Confirm hidden negative: price down, volume up
                if price_change < 0 and volume_change > 0:
                    return 1.0
                elif price_change < 0 and volume_change > -0.05:
                    return 0.5
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking timeframe confirmation: {e}")
            return 0.0
    
    def _calculate_divergence_score(
        self, 
        divergence_info: Dict[str, Any], 
        pattern_direction: str
    ) -> float:
        """
        Calculate divergence score (-1.0 to 1.0)
        """
        try:
            divergence_type = divergence_info['divergence_type']
            confirmation_score = divergence_info.get('confirmation_score', 0.0)
            
            # Base score based on divergence type
            base_score = 0.0
            
            if divergence_type == DivergenceType.POSITIVE_DIVERGENCE:
                base_score = 0.8  # Bullish
            elif divergence_type == DivergenceType.NEGATIVE_DIVERGENCE:
                base_score = -0.8  # Bearish
            elif divergence_type == DivergenceType.HIDDEN_POSITIVE:
                base_score = 0.4  # Mildly bullish
            elif divergence_type == DivergenceType.HIDDEN_NEGATIVE:
                base_score = -0.4  # Mildly bearish
            
            # Adjust score based on confirmation
            adjusted_score = base_score * confirmation_score
            
            # Check pattern alignment
            if pattern_direction == "bullish" and base_score > 0:
                adjusted_score *= 1.2  # Boost for bullish alignment
            elif pattern_direction == "bearish" and base_score < 0:
                adjusted_score *= 1.2  # Boost for bearish alignment
            elif pattern_direction != "neutral":
                adjusted_score *= 0.8  # Penalty for misalignment
            
            return max(-1.0, min(1.0, adjusted_score))
            
        except Exception as e:
            logger.error(f"Error calculating divergence score: {e}")
            return 0.0
    
    def _calculate_divergence_confidence(self, divergence_info: Dict[str, Any]) -> float:
        """
        Calculate confidence in the divergence detection
        """
        try:
            confirmation_score = divergence_info.get('confirmation_score', 0.0)
            price_change = abs(divergence_info.get('price_change', 0.0))
            volume_change = abs(divergence_info.get('volume_change', 0.0))
            
            # Base confidence from confirmation
            confidence = confirmation_score * 0.6
            
            # Add confidence based on magnitude of changes
            if price_change > 0.1:  # 10% price change
                confidence += 0.2
            elif price_change > 0.05:  # 5% price change
                confidence += 0.1
            
            if volume_change > 0.2:  # 20% volume change
                confidence += 0.2
            elif volume_change > 0.1:  # 10% volume change
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating divergence confidence: {e}")
            return 0.0
    
    def _determine_divergence_strength(self, divergence_info: Dict[str, Any]) -> DivergenceStrength:
        """
        Determine the strength of the divergence
        """
        try:
            price_change = abs(divergence_info.get('price_change', 0.0))
            volume_change = abs(divergence_info.get('volume_change', 0.0))
            confirmation_score = divergence_info.get('confirmation_score', 0.0)
            
            # Calculate overall strength
            strength_score = (price_change + volume_change + confirmation_score) / 3
            
            if strength_score >= 0.4:
                return DivergenceStrength.EXTREME
            elif strength_score >= 0.3:
                return DivergenceStrength.STRONG
            elif strength_score >= 0.2:
                return DivergenceStrength.MODERATE
            else:
                return DivergenceStrength.WEAK
                
        except Exception as e:
            logger.error(f"Error determining divergence strength: {e}")
            return DivergenceStrength.WEAK
    
    def _determine_pattern_alignment(
        self, 
        divergence_info: Dict[str, Any], 
        pattern_direction: str
    ) -> str:
        """
        Determine if divergence aligns with pattern direction
        """
        try:
            divergence_type = divergence_info['divergence_type']
            
            if divergence_type in [DivergenceType.POSITIVE_DIVERGENCE, DivergenceType.HIDDEN_POSITIVE]:
                divergence_direction = "bullish"
            elif divergence_type in [DivergenceType.NEGATIVE_DIVERGENCE, DivergenceType.HIDDEN_NEGATIVE]:
                divergence_direction = "bearish"
            else:
                divergence_direction = "neutral"
            
            if pattern_direction == divergence_direction:
                return "aligned"
            elif pattern_direction == "neutral":
                return "neutral"
            else:
                return "misaligned"
                
        except Exception as e:
            logger.error(f"Error determining pattern alignment: {e}")
            return "neutral"
    
    def _generate_divergence_description(self, divergence_info: Dict[str, Any]) -> str:
        """
        Generate a description of the detected divergence
        """
        try:
            divergence_type = divergence_info['divergence_type']
            strength = self._determine_divergence_strength(divergence_info)
            price_change = divergence_info.get('price_change', 0.0)
            volume_change = divergence_info.get('volume_change', 0.0)
            
            descriptions = {
                DivergenceType.POSITIVE_DIVERGENCE: f"Positive divergence detected: Price down {abs(price_change)*100:.1f}%, Volume up {volume_change*100:.1f}%",
                DivergenceType.NEGATIVE_DIVERGENCE: f"Negative divergence detected: Price up {price_change*100:.1f}%, Volume down {abs(volume_change)*100:.1f}%",
                DivergenceType.HIDDEN_POSITIVE: f"Hidden positive divergence: Price up {price_change*100:.1f}%, Volume down {abs(volume_change)*100:.1f}%",
                DivergenceType.HIDDEN_NEGATIVE: f"Hidden negative divergence: Price down {abs(price_change)*100:.1f}%, Volume up {volume_change*100:.1f}%"
            }
            
            base_description = descriptions.get(divergence_type, "No divergence detected")
            return f"{base_description} ({strength.value} strength)"
            
        except Exception as e:
            logger.error(f"Error generating divergence description: {e}")
            return "Divergence analysis error"
    
    def _calculate_momentum(self, series: pd.Series) -> float:
        """
        Calculate momentum of a series
        """
        try:
            if len(series) < 2:
                return 0.0
            
            # Simple momentum: rate of change
            return (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _find_volume_peaks(self, df: pd.DataFrame) -> List[int]:
        """
        Find volume peaks in the data
        """
        try:
            volume_series = df['volume']
            peaks = []
            
            for i in range(1, len(volume_series) - 1):
                if volume_series.iloc[i] > volume_series.iloc[i-1] and volume_series.iloc[i] > volume_series.iloc[i+1]:
                    peaks.append(i)
            
            return peaks
            
        except Exception as e:
            logger.error(f"Error finding volume peaks: {e}")
            return []
    
    def _get_no_divergence_result(self) -> Dict[str, Any]:
        """
        Return result for no divergence detected
        """
        return {
            'divergence_type': DivergenceType.NO_DIVERGENCE,
            'price_change': 0.0,
            'volume_change': 0.0,
            'price_trend': 'unknown',
            'volume_trend': 'unknown',
            'confirmation_score': 0.0,
            'price_analysis': {},
            'volume_analysis': {}
        }
    
    def get_divergence_multiplier(self, divergence_type: DivergenceType) -> float:
        """
        Get confidence multiplier for divergence type
        """
        return self.divergence_multipliers.get(divergence_type, 1.0)
    
    def analyze_multiple_patterns(
        self, 
        df: pd.DataFrame, 
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, VolumeDivergenceSignal]:
        """
        Analyze divergence for multiple patterns
        """
        try:
            results = {}
            
            for pattern in patterns:
                pattern_type = pattern.get('type', 'unknown')
                pattern_direction = pattern.get('direction', 'neutral')
                
                divergence_signal = self.detect_volume_divergence(
                    df, pattern_type, pattern_direction
                )
                
                if divergence_signal:
                    results[pattern_type] = divergence_signal
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing multiple patterns: {e}")
            return {}
