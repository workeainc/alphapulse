"""
Elliott Wave Analyzer for AlphaPlus
Implements institutional-grade Elliott Wave analysis and pattern recognition
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WaveType(Enum):
    """Types of Elliott Waves"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    EXTENSION = "extension"
    DIAGONAL = "diagonal"
    ZIGZAG = "zigzag"
    FLAT = "flat"
    TRIANGLE = "triangle"
    COMBINATION = "combination"

class WavePosition(Enum):
    """Wave positions in Elliott Wave structure"""
    WAVE_1 = "wave_1"
    WAVE_2 = "wave_2"
    WAVE_3 = "wave_3"
    WAVE_4 = "wave_4"
    WAVE_5 = "wave_5"
    WAVE_A = "wave_a"
    WAVE_B = "wave_b"
    WAVE_C = "wave_c"

@dataclass
class ElliottWave:
    """Elliott Wave Structure"""
    wave_type: WaveType
    position: WavePosition
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    high: float
    low: float
    length: float
    retracement: float  # Retracement percentage of previous wave
    fibonacci_ratios: Dict[str, float]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, any]

@dataclass
class ElliottWaveAnalysis:
    """Complete Elliott Wave Analysis Result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_wave: WavePosition
    wave_count: int
    waves: List[ElliottWave]
    pattern_type: WaveType
    trend_direction: str  # 'bullish' or 'bearish'
    next_target: float
    support_levels: List[float]
    resistance_levels: List[float]
    fibonacci_levels: Dict[str, float]
    confidence_score: float
    processing_time_ms: float
    metadata: Dict[str, any]

class ElliottWaveAnalyzer:
    """
    Advanced Elliott Wave Analyzer for institutional-grade analysis
    Implements wave counting, pattern recognition, and Fibonacci analysis
    """
    
    def __init__(self, 
                 min_wave_length: int = 5,
                 max_wave_length: int = 50,
                 min_retracement: float = 0.236,
                 max_retracement: float = 0.786,
                 fibonacci_levels: List[float] = None):
        """
        Initialize Elliott Wave Analyzer
        
        Args:
            min_wave_length: Minimum number of candles for a wave
            max_wave_length: Maximum number of candles for a wave
            min_retracement: Minimum retracement percentage
            max_retracement: Maximum retracement percentage
            fibonacci_levels: List of Fibonacci ratios to analyze
        """
        self.min_wave_length = min_wave_length
        self.max_wave_length = max_wave_length
        self.min_retracement = min_retracement
        self.max_retracement = max_retracement
        
        if fibonacci_levels is None:
            self.fibonacci_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.618, 2.618]
        else:
            self.fibonacci_levels = fibonacci_levels
            
        self.logger = logging.getLogger(__name__)
        
    def analyze_elliott_waves(self, 
                            df: pd.DataFrame, 
                            symbol: str, 
                            timeframe: str,
                            lookback_periods: int = 200) -> ElliottWaveAnalysis:
        """
        Perform comprehensive Elliott Wave analysis
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe being analyzed
            lookback_periods: Number of periods to analyze
            
        Returns:
            ElliottWaveAnalysis: Complete analysis results
        """
        start_time = datetime.now()
        
        try:
            # Validate input data
            if len(df) < lookback_periods:
                lookback_periods = len(df)
            
            # Use recent data
            recent_df = df.tail(lookback_periods).copy()
            
            # Find swing highs and lows
            swing_points = self._find_swing_points(recent_df)
            
            # Identify Elliott Wave structure
            waves = self._identify_waves(swing_points, recent_df)
            
            # Determine current wave position
            current_wave = self._determine_current_wave(waves)
            
            # Analyze pattern type
            pattern_type = self._analyze_pattern_type(waves)
            
            # Calculate Fibonacci levels
            fibonacci_levels = self._calculate_fibonacci_levels(waves, recent_df)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(waves)
            
            # Calculate next target
            next_target = self._calculate_next_target(waves, fibonacci_levels)
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(waves, fibonacci_levels)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(waves, pattern_type)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ElliottWaveAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                current_wave=current_wave,
                wave_count=len(waves),
                waves=waves,
                pattern_type=pattern_type,
                trend_direction=trend_direction,
                next_target=next_target,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                fibonacci_levels=fibonacci_levels,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                metadata={
                    'lookback_periods': lookback_periods,
                    'swing_points_count': len(swing_points),
                    'min_wave_length': self.min_wave_length,
                    'max_wave_length': self.max_wave_length
                }
            )
            
        except Exception as e:
            self.logger.error(f"Elliott Wave analysis error: {e}")
            raise
    
    def _find_swing_points(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """Find swing highs and lows in the data"""
        try:
            swing_points = []
            
            for i in range(2, len(df) - 2):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                
                # Check for swing high
                if (high > df['high'].iloc[i-1] and high > df['high'].iloc[i-2] and
                    high > df['high'].iloc[i+1] and high > df['high'].iloc[i+2]):
                    swing_points.append({
                        'index': i,
                        'price': high,
                        'type': 'high',
                        'timestamp': df.index[i]
                    })
                
                # Check for swing low
                elif (low < df['low'].iloc[i-1] and low < df['low'].iloc[i-2] and
                      low < df['low'].iloc[i+1] and low < df['low'].iloc[i+2]):
                    swing_points.append({
                        'index': i,
                        'price': low,
                        'type': 'low',
                        'timestamp': df.index[i]
                    })
            
            return swing_points
            
        except Exception as e:
            self.logger.error(f"Swing points detection error: {e}")
            return []
    
    def _identify_waves(self, swing_points: List[Dict[str, any]], df: pd.DataFrame) -> List[ElliottWave]:
        """Identify Elliott Wave structure from swing points"""
        try:
            waves = []
            
            if len(swing_points) < 3:
                return waves
            
            # Start with first swing point
            current_direction = swing_points[0]['type']
            wave_start = swing_points[0]
            wave_count = 0
            
            for i in range(1, len(swing_points)):
                point = swing_points[i]
                
                # Check if direction changed (potential wave end)
                if point['type'] != current_direction:
                    # Create wave
                    if wave_count < 5:  # Only first 5 waves
                        wave_position = self._get_wave_position(wave_count, current_direction)
                        
                        # Calculate wave properties
                        wave_length = point['index'] - wave_start['index']
                        price_change = point['price'] - wave_start['price']
                        
                        # Calculate retracement if not first wave
                        retracement = 0.0
                        if wave_count > 0 and len(waves) > 0:
                            prev_wave = waves[-1]
                            prev_length = abs(prev_wave.end_price - prev_wave.start_price)
                            if prev_length > 0:
                                retracement = abs(price_change) / prev_length
                        
                        # Calculate Fibonacci ratios
                        fibonacci_ratios = self._calculate_wave_fibonacci_ratios(waves, price_change)
                        
                        # Determine wave type
                        wave_type = self._determine_wave_type(wave_count, price_change, retracement)
                        
                        # Calculate confidence
                        confidence = self._calculate_wave_confidence(wave_count, retracement, fibonacci_ratios)
                        
                        wave = ElliottWave(
                            wave_type=wave_type,
                            position=wave_position,
                            start_index=wave_start['index'],
                            end_index=point['index'],
                            start_price=wave_start['price'],
                            end_price=point['price'],
                            high=max(wave_start['price'], point['price']),
                            low=min(wave_start['price'], point['price']),
                            length=wave_length,
                            retracement=retracement,
                            fibonacci_ratios=fibonacci_ratios,
                            confidence=confidence,
                            timestamp=point['timestamp'],
                            metadata={
                                'direction': current_direction,
                                'price_change': price_change,
                                'wave_number': wave_count + 1
                            }
                        )
                        
                        waves.append(wave)
                        wave_count += 1
                    
                    # Reset for next wave
                    wave_start = point
                    current_direction = point['type']
            
            return waves
            
        except Exception as e:
            self.logger.error(f"Wave identification error: {e}")
            return []
    
    def _get_wave_position(self, wave_count: int, direction: str) -> WavePosition:
        """Get wave position based on count and direction"""
        if direction == 'high':  # Impulse waves
            positions = [WavePosition.WAVE_1, WavePosition.WAVE_2, WavePosition.WAVE_3, 
                        WavePosition.WAVE_4, WavePosition.WAVE_5]
        else:  # Corrective waves
            positions = [WavePosition.WAVE_A, WavePosition.WAVE_B, WavePosition.WAVE_C]
        
        if wave_count < len(positions):
            return positions[wave_count]
        else:
            return WavePosition.WAVE_5  # Default
    
    def _calculate_wave_fibonacci_ratios(self, waves: List[ElliottWave], price_change: float) -> Dict[str, float]:
        """Calculate Fibonacci ratios for the wave"""
        try:
            ratios = {}
            
            if len(waves) == 0:
                return ratios
            
            # Calculate ratios relative to previous waves
            for i, wave in enumerate(waves[-2:], 1):  # Last 2 waves
                prev_length = abs(wave.end_price - wave.start_price)
                if prev_length > 0:
                    ratio = abs(price_change) / prev_length
                    ratios[f'fib_{i}'] = ratio
            
            # Calculate common Fibonacci ratios
            for fib_level in self.fibonacci_levels:
                ratios[f'fib_{fib_level}'] = fib_level
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Fibonacci calculation error: {e}")
            return {}
    
    def _determine_wave_type(self, wave_count: int, price_change: float, retracement: float) -> WaveType:
        """Determine the type of Elliott Wave"""
        try:
            if wave_count < 5:
                # Impulse waves (1, 3, 5)
                if wave_count in [0, 2, 4]:  # Waves 1, 3, 5
                    if abs(price_change) > 0:  # Non-zero price change
                        return WaveType.IMPULSE
                    else:
                        return WaveType.CORRECTIVE
                else:  # Waves 2, 4
                    return WaveType.CORRECTIVE
            else:
                # Corrective waves (A, B, C)
                if retracement > 0.618:  # Deep retracement
                    return WaveType.ZIGZAG
                elif retracement < 0.382:  # Shallow retracement
                    return WaveType.FLAT
                else:
                    return WaveType.CORRECTIVE
            
        except Exception as e:
            self.logger.error(f"Wave type determination error: {e}")
            return WaveType.CORRECTIVE
    
    def _calculate_wave_confidence(self, wave_count: int, retracement: float, fibonacci_ratios: Dict[str, float]) -> float:
        """Calculate confidence score for the wave"""
        try:
            confidence_factors = []
            
            # Factor 1: Retracement validity
            if self.min_retracement <= retracement <= self.max_retracement:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
            
            # Factor 2: Fibonacci ratio alignment
            fib_alignment = 0.0
            for ratio in fibonacci_ratios.values():
                if isinstance(ratio, float):
                    for fib_level in self.fibonacci_levels:
                        if abs(ratio - fib_level) < 0.1:  # Within 10% of Fibonacci level
                            fib_alignment = max(fib_alignment, 0.9)
                            break
            
            confidence_factors.append(fib_alignment)
            
            # Factor 3: Wave count validity
            if wave_count < 5:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
            
            # Calculate weighted average
            if confidence_factors:
                weights = [0.4, 0.4, 0.2]
                weighted_sum = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
                return min(weighted_sum, 1.0)
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _determine_current_wave(self, waves: List[ElliottWave]) -> WavePosition:
        """Determine current wave position"""
        try:
            if not waves:
                return WavePosition.WAVE_1
            
            # Return the last wave position
            return waves[-1].position
            
        except Exception as e:
            self.logger.error(f"Current wave determination error: {e}")
            return WavePosition.WAVE_1
    
    def _analyze_pattern_type(self, waves: List[ElliottWave]) -> WaveType:
        """Analyze the overall pattern type"""
        try:
            if not waves:
                return WaveType.CORRECTIVE
            
            # Count wave types
            wave_types = [wave.wave_type for wave in waves]
            
            # Determine pattern based on wave sequence
            if len(wave_types) >= 5:
                if WaveType.IMPULSE in wave_types[:5]:
                    return WaveType.IMPULSE
                elif WaveType.ZIGZAG in wave_types:
                    return WaveType.ZIGZAG
                elif WaveType.FLAT in wave_types:
                    return WaveType.FLAT
                else:
                    return WaveType.CORRECTIVE
            else:
                return WaveType.CORRECTIVE
            
        except Exception as e:
            self.logger.error(f"Pattern type analysis error: {e}")
            return WaveType.CORRECTIVE
    
    def _calculate_fibonacci_levels(self, waves: List[ElliottWave], df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement and extension levels"""
        try:
            fibonacci_levels = {}
            
            if not waves:
                return fibonacci_levels
            
            # Get the overall trend
            first_wave = waves[0]
            last_wave = waves[-1]
            
            trend_start = first_wave.start_price
            trend_end = last_wave.end_price
            trend_range = abs(trend_end - trend_start)
            
            if trend_range == 0:
                return fibonacci_levels
            
            # Calculate retracement levels
            for fib_level in self.fibonacci_levels:
                if trend_end > trend_start:  # Uptrend
                    retracement_level = trend_end - (trend_range * fib_level)
                    fibonacci_levels[f'retracement_{fib_level}'] = retracement_level
                else:  # Downtrend
                    retracement_level = trend_end + (trend_range * fib_level)
                    fibonacci_levels[f'retracement_{fib_level}'] = retracement_level
            
            # Calculate extension levels
            for fib_level in [1.618, 2.618, 3.618]:
                if trend_end > trend_start:  # Uptrend
                    extension_level = trend_end + (trend_range * fib_level)
                    fibonacci_levels[f'extension_{fib_level}'] = extension_level
                else:  # Downtrend
                    extension_level = trend_end - (trend_range * fib_level)
                    fibonacci_levels[f'extension_{fib_level}'] = extension_level
            
            return fibonacci_levels
            
        except Exception as e:
            self.logger.error(f"Fibonacci levels calculation error: {e}")
            return {}
    
    def _determine_trend_direction(self, waves: List[ElliottWave]) -> str:
        """Determine overall trend direction"""
        try:
            if not waves:
                return 'neutral'
            
            # Compare first and last wave
            first_wave = waves[0]
            last_wave = waves[-1]
            
            if last_wave.end_price > first_wave.start_price:
                return 'bullish'
            else:
                return 'bearish'
            
        except Exception as e:
            self.logger.error(f"Trend direction determination error: {e}")
            return 'neutral'
    
    def _calculate_next_target(self, waves: List[ElliottWave], fibonacci_levels: Dict[str, float]) -> float:
        """Calculate next price target based on Elliott Wave analysis"""
        try:
            if not waves:
                return 0.0
            
            current_price = waves[-1].end_price
            
            # Use Fibonacci extension levels
            if fibonacci_levels:
                # Find the next extension level
                extension_levels = [level for key, level in fibonacci_levels.items() if 'extension' in key]
                if extension_levels:
                    # Find the closest extension level above current price (for bullish) or below (for bearish)
                    trend_direction = self._determine_trend_direction(waves)
                    
                    if trend_direction == 'bullish':
                        valid_levels = [level for level in extension_levels if level > current_price]
                        if valid_levels:
                            return min(valid_levels)
                    else:
                        valid_levels = [level for level in extension_levels if level < current_price]
                        if valid_levels:
                            return max(valid_levels)
            
            # Fallback: use wave projection
            if len(waves) >= 2:
                last_wave = waves[-1]
                prev_wave = waves[-2]
                wave_length = abs(last_wave.end_price - last_wave.start_price)
                
                if self._determine_trend_direction(waves) == 'bullish':
                    return current_price + wave_length
                else:
                    return current_price - wave_length
            
            return current_price
            
        except Exception as e:
            self.logger.error(f"Next target calculation error: {e}")
            return 0.0
    
    def _find_support_resistance(self, waves: List[ElliottWave], fibonacci_levels: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        try:
            support_levels = []
            resistance_levels = []
            
            # Add Fibonacci retracement levels
            for key, level in fibonacci_levels.items():
                if 'retracement' in key:
                    if level > 0:
                        if self._determine_trend_direction(waves) == 'bullish':
                            support_levels.append(level)
                        else:
                            resistance_levels.append(level)
            
            # Add wave highs and lows
            for wave in waves:
                if wave.high > 0:
                    resistance_levels.append(wave.high)
                if wave.low > 0:
                    support_levels.append(wave.low)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            
            return support_levels, resistance_levels
            
        except Exception as e:
            self.logger.error(f"Support resistance calculation error: {e}")
            return [], []
    
    def _calculate_confidence_score(self, waves: List[ElliottWave], pattern_type: WaveType) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            if not waves:
                return 0.0
            
            confidence_factors = []
            
            # Factor 1: Number of waves (more waves = higher confidence)
            wave_count_confidence = min(len(waves) / 5.0, 1.0)
            confidence_factors.append(wave_count_confidence)
            
            # Factor 2: Average wave confidence
            avg_wave_confidence = sum(wave.confidence for wave in waves) / len(waves)
            confidence_factors.append(avg_wave_confidence)
            
            # Factor 3: Pattern type confidence
            pattern_confidence = 0.8 if pattern_type == WaveType.IMPULSE else 0.6
            confidence_factors.append(pattern_confidence)
            
            # Factor 4: Wave consistency
            wave_lengths = [wave.length for wave in waves]
            if len(wave_lengths) > 1:
                length_variance = np.var(wave_lengths) / np.mean(wave_lengths) if np.mean(wave_lengths) > 0 else 1.0
                consistency = max(0, 1.0 - length_variance)
                confidence_factors.append(consistency)
            else:
                confidence_factors.append(0.5)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]
            weighted_sum = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
            return min(weighted_sum, 1.0)
            
        except Exception as e:
            self.logger.error(f"Overall confidence calculation error: {e}")
            return 0.5
    
    def get_trading_signals(self, analysis: ElliottWaveAnalysis, current_price: float) -> Dict[str, any]:
        """
        Generate trading signals based on Elliott Wave analysis
        
        Args:
            analysis: Elliott Wave analysis results
            current_price: Current market price
            
        Returns:
            Dict containing trading signals and recommendations
        """
        try:
            signals = {
                'wave_signal': self._analyze_wave_signal(analysis, current_price),
                'pattern_signal': self._analyze_pattern_signal(analysis, current_price),
                'fibonacci_signal': self._analyze_fibonacci_signal(analysis, current_price),
                'trend_signal': self._analyze_trend_signal(analysis, current_price),
                'support_resistance_signals': self._analyze_support_resistance_signals(analysis, current_price),
                'overall_signal': 'neutral',
                'confidence': analysis.confidence_score,
                'recommendations': []
            }
            
            # Determine overall signal
            signal_scores = []
            
            if signals['wave_signal']['strength'] > 0.7:
                signal_scores.append(signals['wave_signal']['score'])
            
            if signals['pattern_signal']['strength'] > 0.7:
                signal_scores.append(signals['pattern_signal']['score'])
            
            if signals['fibonacci_signal']['strength'] > 0.7:
                signal_scores.append(signals['fibonacci_signal']['score'])
            
            if signal_scores:
                avg_score = sum(signal_scores) / len(signal_scores)
                if avg_score > 0.6:
                    signals['overall_signal'] = 'bullish'
                elif avg_score < -0.6:
                    signals['overall_signal'] = 'bearish'
                else:
                    signals['overall_signal'] = 'neutral'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Trading signals generation error: {e}")
            return {'overall_signal': 'neutral', 'confidence': 0.0, 'error': str(e)}
    
    def _analyze_wave_signal(self, analysis: ElliottWaveAnalysis, current_price: float) -> Dict[str, any]:
        """Analyze wave-based trading signal"""
        try:
            if not analysis.waves:
                return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': 'No waves detected'}
            
            current_wave = analysis.waves[-1]
            
            # Analyze based on wave position
            if current_wave.position in [WavePosition.WAVE_1, WavePosition.WAVE_3, WavePosition.WAVE_5]:
                return {
                    'signal': 'bullish' if analysis.trend_direction == 'bullish' else 'bearish',
                    'strength': current_wave.confidence,
                    'score': 0.7 if analysis.trend_direction == 'bullish' else -0.7,
                    'message': f'In impulse wave {current_wave.position.value}'
                }
            elif current_wave.position in [WavePosition.WAVE_2, WavePosition.WAVE_4]:
                return {
                    'signal': 'corrective',
                    'strength': current_wave.confidence,
                    'score': 0.0,
                    'message': f'In corrective wave {current_wave.position.value}'
                }
            else:
                return {
                    'signal': 'neutral',
                    'strength': current_wave.confidence,
                    'score': 0.0,
                    'message': f'In wave {current_wave.position.value}'
                }
                
        except Exception as e:
            return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}
    
    def _analyze_pattern_signal(self, analysis: ElliottWaveAnalysis, current_price: float) -> Dict[str, any]:
        """Analyze pattern-based trading signal"""
        try:
            if analysis.pattern_type == WaveType.IMPULSE:
                return {
                    'signal': 'bullish' if analysis.trend_direction == 'bullish' else 'bearish',
                    'strength': 0.8,
                    'score': 0.8 if analysis.trend_direction == 'bullish' else -0.8,
                    'message': 'Impulse pattern detected - strong trend continuation'
                }
            elif analysis.pattern_type == WaveType.CORRECTIVE:
                return {
                    'signal': 'corrective',
                    'strength': 0.6,
                    'score': 0.0,
                    'message': 'Corrective pattern - consolidation expected'
                }
            else:
                return {
                    'signal': 'neutral',
                    'strength': 0.5,
                    'score': 0.0,
                    'message': f'{analysis.pattern_type.value} pattern detected'
                }
                
        except Exception as e:
            return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}
    
    def _analyze_fibonacci_signal(self, analysis: ElliottWaveAnalysis, current_price: float) -> Dict[str, any]:
        """Analyze Fibonacci-based trading signal"""
        try:
            if not analysis.fibonacci_levels:
                return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': 'No Fibonacci levels'}
            
            # Find nearest Fibonacci level
            nearest_level = None
            min_distance = float('inf')
            
            for key, level in analysis.fibonacci_levels.items():
                distance = abs(current_price - level)
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = (key, level)
            
            if nearest_level and min_distance / current_price < 0.01:  # Within 1%
                level_name, level_value = nearest_level
                return {
                    'signal': 'support_resistance',
                    'strength': 0.7,
                    'score': 0.0,
                    'message': f'Price at Fibonacci level {level_name}: {level_value:.2f}'
                }
            else:
                return {
                    'signal': 'neutral',
                    'strength': 0.3,
                    'score': 0.0,
                    'message': 'Price not at key Fibonacci levels'
                }
                
        except Exception as e:
            return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}
    
    def _analyze_trend_signal(self, analysis: ElliottWaveAnalysis, current_price: float) -> Dict[str, any]:
        """Analyze trend-based trading signal"""
        try:
            if analysis.trend_direction == 'bullish':
                return {
                    'signal': 'bullish',
                    'strength': 0.8,
                    'score': 0.6,
                    'message': 'Bullish trend detected - buy on dips'
                }
            elif analysis.trend_direction == 'bearish':
                return {
                    'signal': 'bearish',
                    'strength': 0.8,
                    'score': -0.6,
                    'message': 'Bearish trend detected - sell on rallies'
                }
            else:
                return {
                    'signal': 'neutral',
                    'strength': 0.5,
                    'score': 0.0,
                    'message': 'Neutral trend - wait for direction'
                }
                
        except Exception as e:
            return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}
    
    def _analyze_support_resistance_signals(self, analysis: ElliottWaveAnalysis, current_price: float) -> List[Dict[str, any]]:
        """Analyze support and resistance signals"""
        try:
            signals = []
            
            # Check support levels
            for support in analysis.support_levels:
                distance = abs(current_price - support) / current_price
                if distance < 0.02:  # Within 2% of support
                    signals.append({
                        'signal': 'support',
                        'strength': 0.7,
                        'score': 0.5,
                        'message': f'Price near support at {support:.2f}',
                        'level_type': 'support'
                    })
            
            # Check resistance levels
            for resistance in analysis.resistance_levels:
                distance = abs(current_price - resistance) / current_price
                if distance < 0.02:  # Within 2% of resistance
                    signals.append({
                        'signal': 'resistance',
                        'strength': 0.7,
                        'score': -0.5,
                        'message': f'Price near resistance at {resistance:.2f}',
                        'level_type': 'resistance'
                    })
            
            return signals
            
        except Exception as e:
            return [{'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}]
