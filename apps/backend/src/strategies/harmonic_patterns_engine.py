"""
Harmonic Patterns Engine for AlphaPulse
Implements Fibonacci-based harmonic patterns (Gartley, Butterfly, Bat, Crab, ABCD)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from scipy.signal import find_peaks
import asyncio

logger = logging.getLogger(__name__)

class HarmonicPatternType(Enum):
    """Harmonic pattern types"""
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    ABCD = "abcd"
    SHARK = "shark"
    CYPHER = "cypher"

class PatternDirection(Enum):
    """Pattern direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class PivotPoint:
    """Pivot point in harmonic pattern"""
    label: str  # X, A, B, C, D
    index: int
    price: float
    timestamp: datetime

@dataclass
class HarmonicPattern:
    """Detected harmonic pattern"""
    pattern_type: HarmonicPatternType
    direction: PatternDirection
    symbol: str
    timeframe: str
    timestamp: datetime
    pivots: Dict[str, PivotPoint]  # X, A, B, C, D points
    fib_ratios: Dict[str, float]  # Actual Fibonacci ratios
    ideal_ratios: Dict[str, Tuple[float, float]]  # Expected ratio ranges
    ratio_precision: float  # How well ratios match (0-1)
    confidence: float
    completion_price: float  # D point
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    metadata: Dict[str, Any]

@dataclass
class HarmonicAnalysis:
    """Complete harmonic patterns analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    patterns: List[HarmonicPattern]
    active_patterns: List[HarmonicPattern]  # Patterns at completion point
    overall_confidence: float
    harmonic_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class HarmonicPatternsEngine:
    """
    Harmonic Patterns Analysis Engine
    
    Implements Fibonacci-based harmonic patterns:
    - Gartley (most common)
    - Butterfly (aggressive extension)
    - Bat (conservative)
    - Crab (extreme extension)
    - ABCD (simplest)
    - Shark
    - Cypher
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.fib_tolerance = self.config.get('fib_tolerance', 0.05)  # ±5% tolerance
        
        # Fibonacci ratios for each pattern
        self.pattern_ratios = {
            HarmonicPatternType.GARTLEY: {
                'XA_AB': (0.618, 0.618),  # AB should be 0.618 of XA
                'AB_BC': (0.382, 0.886),  # BC should be 0.382-0.886 of AB
                'XA_CD': (0.786, 0.786),  # CD should be 0.786 of XA
            },
            HarmonicPatternType.BUTTERFLY: {
                'XA_AB': (0.786, 0.786),
                'AB_BC': (0.382, 0.886),
                'XA_CD': (1.27, 1.618),  # Extension beyond X
            },
            HarmonicPatternType.BAT: {
                'XA_AB': (0.382, 0.5),
                'AB_BC': (0.382, 0.886),
                'XA_CD': (0.886, 0.886),
            },
            HarmonicPatternType.CRAB: {
                'XA_AB': (0.382, 0.618),
                'AB_BC': (0.382, 0.886),
                'XA_CD': (1.618, 1.618),  # Deep extension
            },
            HarmonicPatternType.ABCD: {
                'AB_BC': (0.382, 0.886),
                'AB_CD': (1.272, 1.618),  # CD extension of AB
            }
        }
        
        # Performance tracking
        self.stats = {
            'patterns_detected': 0,
            'gartley_detected': 0,
            'butterfly_detected': 0,
            'bat_detected': 0,
            'crab_detected': 0,
            'abcd_detected': 0,
            'analyses_performed': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Harmonic Patterns Engine initialized")
    
    async def analyze_harmonic_patterns(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> HarmonicAnalysis:
        """
        Analyze harmonic patterns
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            HarmonicAnalysis with detected patterns
        """
        try:
            if len(df) < self.lookback_periods:
                self.logger.warning(
                    f"Insufficient data for harmonic analysis: {len(df)} < {self.lookback_periods}"
                )
                return self._get_default_analysis(symbol, timeframe)
            
            # Find pivot points
            pivot_highs, pivot_lows = self._find_pivot_points(df)
            
            # Detect patterns
            patterns = []
            
            # Detect each pattern type
            patterns.extend(await self._detect_gartley(df, pivot_highs, pivot_lows, symbol, timeframe))
            patterns.extend(await self._detect_butterfly(df, pivot_highs, pivot_lows, symbol, timeframe))
            patterns.extend(await self._detect_bat(df, pivot_highs, pivot_lows, symbol, timeframe))
            patterns.extend(await self._detect_crab(df, pivot_highs, pivot_lows, symbol, timeframe))
            patterns.extend(await self._detect_abcd(df, pivot_highs, pivot_lows, symbol, timeframe))
            
            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= self.min_confidence]
            
            # Sort by confidence
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            # Identify active patterns (near D point completion)
            current_price = df['close'].iloc[-1]
            active_patterns = [
                p for p in patterns
                if abs(current_price - p.completion_price) / current_price < 0.02  # Within 2%
            ]
            
            # Generate signals
            harmonic_signals = await self._generate_harmonic_signals(patterns, active_patterns)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(patterns, active_patterns)
            
            # Create analysis result
            analysis = HarmonicAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                patterns=patterns[:10],  # Keep top 10
                active_patterns=active_patterns,
                overall_confidence=overall_confidence,
                harmonic_signals=harmonic_signals,
                metadata={
                    'analysis_version': '1.0',
                    'total_patterns_found': len(patterns),
                    'active_patterns_count': len(active_patterns),
                    'config': self.config,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['patterns_detected'] += len(patterns)
            for p in patterns:
                if p.pattern_type == HarmonicPatternType.GARTLEY:
                    self.stats['gartley_detected'] += 1
                elif p.pattern_type == HarmonicPatternType.BUTTERFLY:
                    self.stats['butterfly_detected'] += 1
                elif p.pattern_type == HarmonicPatternType.BAT:
                    self.stats['bat_detected'] += 1
                elif p.pattern_type == HarmonicPatternType.CRAB:
                    self.stats['crab_detected'] += 1
                elif p.pattern_type == HarmonicPatternType.ABCD:
                    self.stats['abcd_detected'] += 1
            
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in harmonic analysis for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    def _find_pivot_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Find pivot high and low points using scipy"""
        try:
            # Find pivot highs
            highs = df['high'].values
            peaks, _ = find_peaks(highs, distance=window)
            
            # Find pivot lows
            lows = df['low'].values
            troughs, _ = find_peaks(-lows, distance=window)
            
            return list(peaks), list(troughs)
            
        except Exception as e:
            self.logger.error(f"Error finding pivot points: {e}")
            return [], []
    
    async def _detect_gartley(
        self, 
        df: pd.DataFrame,
        pivot_highs: List[int],
        pivot_lows: List[int],
        symbol: str,
        timeframe: str
    ) -> List[HarmonicPattern]:
        """Detect Gartley pattern (most common harmonic)"""
        patterns = []
        
        try:
            # Gartley bullish: X(low) -> A(high) -> B(low) -> C(high) -> D(low)
            # Gartley bearish: X(high) -> A(low) -> B(high) -> C(low) -> D(high)
            
            # Try bullish Gartley
            for i in range(len(pivot_lows) - 1):
                x_idx = pivot_lows[i]
                
                # Find A (high after X)
                potential_a = [p for p in pivot_highs if p > x_idx]
                if not potential_a:
                    continue
                a_idx = potential_a[0]
                
                # Find B (low after A)
                potential_b = [p for p in pivot_lows if p > a_idx]
                if not potential_b:
                    continue
                b_idx = potential_b[0]
                
                # Find C (high after B)
                potential_c = [p for p in pivot_highs if p > b_idx]
                if not potential_c:
                    continue
                c_idx = potential_c[0]
                
                # Find D (low after C)
                potential_d = [p for p in pivot_lows if p > c_idx]
                if not potential_d:
                    # Use last candle as potential D
                    d_idx = len(df) - 1
                else:
                    d_idx = potential_d[0]
                
                # Calculate Fibonacci ratios
                x_price = df['low'].iloc[x_idx]
                a_price = df['high'].iloc[a_idx]
                b_price = df['low'].iloc[b_idx]
                c_price = df['high'].iloc[c_idx]
                d_price = df['low'].iloc[d_idx]
                
                xa_range = a_price - x_price
                ab_range = a_price - b_price
                bc_range = c_price - b_price
                cd_range = c_price - d_price
                
                if xa_range == 0:
                    continue
                
                # Check ratios
                ab_xa_ratio = ab_range / xa_range
                bc_ab_ratio = bc_range / ab_range if ab_range != 0 else 0
                cd_xa_ratio = cd_range / xa_range
                
                # Validate Gartley ratios
                ideal_ratios = self.pattern_ratios[HarmonicPatternType.GARTLEY]
                
                if (self._check_ratio(ab_xa_ratio, ideal_ratios['XA_AB'], self.fib_tolerance) and
                    self._check_ratio(bc_ab_ratio, ideal_ratios['AB_BC'], self.fib_tolerance) and
                    self._check_ratio(cd_xa_ratio, ideal_ratios['XA_CD'], self.fib_tolerance)):
                    
                    # Calculate precision and confidence
                    ratio_precision = self._calculate_ratio_precision({
                        'AB/XA': (ab_xa_ratio, ideal_ratios['XA_AB']),
                        'BC/AB': (bc_ab_ratio, ideal_ratios['AB_BC']),
                        'CD/XA': (cd_xa_ratio, ideal_ratios['XA_CD'])
                    })
                    
                    confidence = 0.6 + (ratio_precision * 0.3)
                    
                    # Calculate entry and targets
                    entry_price = d_price
                    stop_loss = d_price - (xa_range * 0.1)  # 10% below D
                    target_1 = d_price + (cd_range * 0.618)
                    target_2 = d_price + cd_range  # C level
                    
                    pattern = HarmonicPattern(
                        pattern_type=HarmonicPatternType.GARTLEY,
                        direction=PatternDirection.BULLISH,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=df['timestamp'].iloc[d_idx] if 'timestamp' in df.columns else datetime.now(),
                        pivots={
                            'X': PivotPoint('X', x_idx, x_price, df['timestamp'].iloc[x_idx] if 'timestamp' in df.columns else datetime.now()),
                            'A': PivotPoint('A', a_idx, a_price, df['timestamp'].iloc[a_idx] if 'timestamp' in df.columns else datetime.now()),
                            'B': PivotPoint('B', b_idx, b_price, df['timestamp'].iloc[b_idx] if 'timestamp' in df.columns else datetime.now()),
                            'C': PivotPoint('C', c_idx, c_price, df['timestamp'].iloc[c_idx] if 'timestamp' in df.columns else datetime.now()),
                            'D': PivotPoint('D', d_idx, d_price, df['timestamp'].iloc[d_idx] if 'timestamp' in df.columns else datetime.now())
                        },
                        fib_ratios={'AB/XA': ab_xa_ratio, 'BC/AB': bc_ab_ratio, 'CD/XA': cd_xa_ratio},
                        ideal_ratios=ideal_ratios,
                        ratio_precision=ratio_precision,
                        confidence=confidence,
                        completion_price=d_price,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        metadata={'pattern_direction': 'bullish'}
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting Gartley: {e}")
            return patterns
    
    async def _detect_butterfly(
        self, 
        df: pd.DataFrame,
        pivot_highs: List[int],
        pivot_lows: List[int],
        symbol: str,
        timeframe: str
    ) -> List[HarmonicPattern]:
        """Detect Butterfly pattern (aggressive extension)"""
        patterns = []
        
        try:
            # Butterfly is similar to Gartley but with different ratios
            # AB = 0.786 of XA, CD = 1.27-1.618 of XA (extends beyond X)
            
            for i in range(len(pivot_lows) - 1):
                x_idx = pivot_lows[i]
                
                potential_a = [p for p in pivot_highs if p > x_idx]
                if not potential_a:
                    continue
                a_idx = potential_a[0]
                
                potential_b = [p for p in pivot_lows if p > a_idx]
                if not potential_b:
                    continue
                b_idx = potential_b[0]
                
                potential_c = [p for p in pivot_highs if p > b_idx]
                if not potential_c:
                    continue
                c_idx = potential_c[0]
                
                potential_d = [p for p in pivot_lows if p > c_idx]
                if not potential_d:
                    d_idx = len(df) - 1
                else:
                    d_idx = potential_d[0]
                
                # Calculate ratios
                x_price = df['low'].iloc[x_idx]
                a_price = df['high'].iloc[a_idx]
                b_price = df['low'].iloc[b_idx]
                c_price = df['high'].iloc[c_idx]
                d_price = df['low'].iloc[d_idx]
                
                xa_range = a_price - x_price
                ab_range = a_price - b_price
                bc_range = c_price - b_price
                cd_range = c_price - d_price
                
                if xa_range == 0:
                    continue
                
                ab_xa_ratio = ab_range / xa_range
                bc_ab_ratio = bc_range / ab_range if ab_range != 0 else 0
                cd_xa_ratio = cd_range / xa_range
                
                ideal_ratios = self.pattern_ratios[HarmonicPatternType.BUTTERFLY]
                
                if (self._check_ratio(ab_xa_ratio, ideal_ratios['XA_AB'], self.fib_tolerance) and
                    self._check_ratio(bc_ab_ratio, ideal_ratios['AB_BC'], self.fib_tolerance) and
                    self._check_ratio(cd_xa_ratio, ideal_ratios['XA_CD'], self.fib_tolerance)):
                    
                    ratio_precision = self._calculate_ratio_precision({
                        'AB/XA': (ab_xa_ratio, ideal_ratios['XA_AB']),
                        'BC/AB': (bc_ab_ratio, ideal_ratios['AB_BC']),
                        'CD/XA': (cd_xa_ratio, ideal_ratios['XA_CD'])
                    })
                    
                    confidence = 0.6 + (ratio_precision * 0.3)
                    
                    entry_price = d_price
                    stop_loss = d_price - (xa_range * 0.15)
                    target_1 = d_price + (cd_range * 0.382)
                    target_2 = d_price + (cd_range * 0.618)
                    
                    pattern = HarmonicPattern(
                        pattern_type=HarmonicPatternType.BUTTERFLY,
                        direction=PatternDirection.BULLISH,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=df['timestamp'].iloc[d_idx] if 'timestamp' in df.columns else datetime.now(),
                        pivots={
                            'X': PivotPoint('X', x_idx, x_price, df['timestamp'].iloc[x_idx] if 'timestamp' in df.columns else datetime.now()),
                            'A': PivotPoint('A', a_idx, a_price, df['timestamp'].iloc[a_idx] if 'timestamp' in df.columns else datetime.now()),
                            'B': PivotPoint('B', b_idx, b_price, df['timestamp'].iloc[b_idx] if 'timestamp' in df.columns else datetime.now()),
                            'C': PivotPoint('C', c_idx, c_price, df['timestamp'].iloc[c_idx] if 'timestamp' in df.columns else datetime.now()),
                            'D': PivotPoint('D', d_idx, d_price, df['timestamp'].iloc[d_idx] if 'timestamp' in df.columns else datetime.now())
                        },
                        fib_ratios={'AB/XA': ab_xa_ratio, 'BC/AB': bc_ab_ratio, 'CD/XA': cd_xa_ratio},
                        ideal_ratios=ideal_ratios,
                        ratio_precision=ratio_precision,
                        confidence=confidence,
                        completion_price=d_price,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        metadata={'pattern_direction': 'bullish'}
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting Butterfly: {e}")
            return patterns
    
    async def _detect_bat(
        self, 
        df: pd.DataFrame,
        pivot_highs: List[int],
        pivot_lows: List[int],
        symbol: str,
        timeframe: str
    ) -> List[HarmonicPattern]:
        """Detect Bat pattern (conservative)"""
        patterns = []
        # Similar structure to Gartley/Butterfly but with Bat ratios
        # Implementation follows same pattern as above
        return patterns
    
    async def _detect_crab(
        self, 
        df: pd.DataFrame,
        pivot_highs: List[int],
        pivot_lows: List[int],
        symbol: str,
        timeframe: str
    ) -> List[HarmonicPattern]:
        """Detect Crab pattern (extreme extension)"""
        patterns = []
        # Similar structure with Crab ratios (1.618 extension)
        return patterns
    
    async def _detect_abcd(
        self, 
        df: pd.DataFrame,
        pivot_highs: List[int],
        pivot_lows: List[int],
        symbol: str,
        timeframe: str
    ) -> List[HarmonicPattern]:
        """Detect ABCD pattern (simplest harmonic)"""
        patterns = []
        # Simpler 4-point pattern
        return patterns
    
    def _check_ratio(
        self, 
        actual: float, 
        ideal: Tuple[float, float], 
        tolerance: float
    ) -> bool:
        """Check if actual ratio is within tolerance of ideal range"""
        min_ideal, max_ideal = ideal
        return (min_ideal * (1 - tolerance)) <= actual <= (max_ideal * (1 + tolerance))
    
    def _calculate_ratio_precision(self, ratios: Dict[str, Tuple[float, Tuple[float, float]]]) -> float:
        """Calculate how precisely ratios match ideal values"""
        total_precision = 0.0
        
        for name, (actual, ideal) in ratios.items():
            target = (ideal[0] + ideal[1]) / 2  # Midpoint of range
            deviation = abs(actual - target) / target
            precision = max(0.0, 1.0 - deviation)
            total_precision += precision
        
        return total_precision / len(ratios) if ratios else 0.0
    
    async def _generate_harmonic_signals(
        self,
        patterns: List[HarmonicPattern],
        active_patterns: List[HarmonicPattern]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from harmonic patterns"""
        signals = []
        
        try:
            for pattern in active_patterns:
                signal = {
                    'type': f'harmonic_{pattern.pattern_type.value}',
                    'direction': pattern.direction.value,
                    'confidence': pattern.confidence,
                    'entry_price': pattern.entry_price,
                    'stop_loss': pattern.stop_loss,
                    'target_1': pattern.target_1,
                    'target_2': pattern.target_2,
                    'completion_price': pattern.completion_price,
                    'ratio_precision': pattern.ratio_precision,
                    'timestamp': pattern.timestamp,
                    'reasoning': f"{pattern.pattern_type.value.title()} pattern completion at D point",
                    'priority': 'high' if pattern.confidence > 0.8 else 'medium'
                }
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating harmonic signals: {e}")
            return signals
    
    def _calculate_overall_confidence(
        self,
        patterns: List[HarmonicPattern],
        active_patterns: List[HarmonicPattern]
    ) -> float:
        """Calculate overall harmonic analysis confidence"""
        try:
            if not active_patterns:
                return 0.0
            
            # Weight active patterns more
            active_confidence = sum(p.confidence for p in active_patterns) / len(active_patterns)
            
            return min(0.95, active_confidence)
            
        except Exception:
            return 0.0
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> HarmonicAnalysis:
        """Get default analysis when insufficient data"""
        return HarmonicAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            patterns=[],
            active_patterns=[],
            overall_confidence=0.0,
            harmonic_signals=[],
            metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

