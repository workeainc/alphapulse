"""
Volume Profile Analyzer for AlphaPlus
Implements institutional-grade volume profile analysis including POC and Value Areas
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VolumeProfileType(Enum):
    """Types of volume profile patterns"""
    POC = "poc"  # Point of Control
    VALUE_AREA_HIGH = "value_area_high"
    VALUE_AREA_LOW = "value_area_low"
    VOLUME_NODE = "volume_node"
    VOLUME_GAP = "volume_gap"
    SINGLE_PRINT = "single_print"
    VOLUME_CLIMAX = "volume_climax"
    VOLUME_EXHAUSTION = "volume_exhaustion"

@dataclass
class VolumeProfileLevel:
    """Volume Profile Level Data"""
    price_level: float
    volume: float
    volume_percentage: float
    poc_score: float  # 0.0 to 1.0
    value_area_score: float  # 0.0 to 1.0
    timestamp: datetime
    level_type: VolumeProfileType
    metadata: Dict[str, any]

@dataclass
class VolumeProfileAnalysis:
    """Complete Volume Profile Analysis Result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    poc_level: float
    value_area_high: float
    value_area_low: float
    value_area_volume_percentage: float
    volume_nodes: List[VolumeProfileLevel]
    volume_gaps: List[VolumeProfileLevel]
    single_prints: List[VolumeProfileLevel]
    volume_climax_levels: List[VolumeProfileLevel]
    volume_exhaustion_levels: List[VolumeProfileLevel]
    total_volume: float
    price_range: Tuple[float, float]
    confidence_score: float
    processing_time_ms: float
    metadata: Dict[str, any]

class VolumeProfileAnalyzer:
    """
    Advanced Volume Profile Analyzer for institutional-grade analysis
    Implements POC (Point of Control) and Value Areas analysis
    """
    
    def __init__(self, 
                 value_area_percentage: float = 0.70,
                 volume_threshold: float = 0.05,
                 price_levels: int = 100,
                 min_volume_for_poc: float = 0.02):
        """
        Initialize Volume Profile Analyzer
        
        Args:
            value_area_percentage: Percentage of volume to include in value area (default 70%)
            volume_threshold: Minimum volume percentage for significant levels
            price_levels: Number of price levels to analyze
            min_volume_for_poc: Minimum volume percentage for POC consideration
        """
        self.value_area_percentage = value_area_percentage
        self.volume_threshold = volume_threshold
        self.price_levels = price_levels
        self.min_volume_for_poc = min_volume_for_poc
        self.logger = logging.getLogger(__name__)
        
    def analyze_volume_profile(self, 
                             df: pd.DataFrame, 
                             symbol: str, 
                             timeframe: str,
                             lookback_periods: int = 100) -> VolumeProfileAnalysis:
        """
        Perform comprehensive volume profile analysis
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe being analyzed
            lookback_periods: Number of periods to analyze
            
        Returns:
            VolumeProfileAnalysis: Complete analysis results
        """
        start_time = datetime.now()
        
        try:
            # Validate input data
            if len(df) < lookback_periods:
                lookback_periods = len(df)
            
            # Use recent data
            recent_df = df.tail(lookback_periods).copy()
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(recent_df)
            
            # Find POC (Point of Control)
            poc_level = self._find_poc(volume_profile)
            
            # Calculate Value Areas
            value_area_high, value_area_low = self._calculate_value_areas(volume_profile)
            
            # Detect volume nodes
            volume_nodes = self._detect_volume_nodes(volume_profile, recent_df)
            
            # Detect volume gaps
            volume_gaps = self._detect_volume_gaps(volume_profile, recent_df)
            
            # Detect single prints
            single_prints = self._detect_single_prints(volume_profile, recent_df)
            
            # Detect volume climax levels
            volume_climax = self._detect_volume_climax(volume_profile, recent_df)
            
            # Detect volume exhaustion levels
            volume_exhaustion = self._detect_volume_exhaustion(volume_profile, recent_df)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                volume_profile, poc_level, value_area_high, value_area_low
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return VolumeProfileAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                poc_level=poc_level,
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                value_area_volume_percentage=self.value_area_percentage,
                volume_nodes=volume_nodes,
                volume_gaps=volume_gaps,
                single_prints=single_prints,
                volume_climax_levels=volume_climax,
                volume_exhaustion_levels=volume_exhaustion,
                total_volume=recent_df['volume'].sum(),
                price_range=(recent_df['low'].min(), recent_df['high'].max()),
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                metadata={
                    'lookback_periods': lookback_periods,
                    'price_levels_analyzed': len(volume_profile),
                    'volume_threshold': self.volume_threshold,
                    'value_area_percentage': self.value_area_percentage
                }
            )
            
        except Exception as e:
            self.logger.error(f"Volume profile analysis error: {e}")
            raise
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume profile for given price levels"""
        try:
            # Create price levels
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_range = price_max - price_min
            
            if price_range == 0:
                # Handle flat price scenario
                price_levels = [price_min]
            else:
                price_levels = np.linspace(price_min, price_max, self.price_levels)
            
            volume_profile = []
            
            for i, price_level in enumerate(price_levels):
                if i == len(price_levels) - 1:
                    # Last level
                    volume_at_level = df[
                        (df['low'] <= price_level) & 
                        (df['high'] >= price_level)
                    ]['volume'].sum()
                else:
                    # Regular levels
                    next_level = price_levels[i + 1]
                    volume_at_level = df[
                        (df['low'] <= price_level) & 
                        (df['high'] >= price_level) &
                        (df['high'] < next_level)
                    ]['volume'].sum()
                
                volume_profile.append({
                    'price_level': price_level,
                    'volume': volume_at_level,
                    'volume_percentage': volume_at_level / df['volume'].sum() if df['volume'].sum() > 0 else 0
                })
            
            return pd.DataFrame(volume_profile)
            
        except Exception as e:
            self.logger.error(f"Volume profile calculation error: {e}")
            raise
    
    def _find_poc(self, volume_profile: pd.DataFrame) -> float:
        """Find Point of Control (price level with highest volume)"""
        try:
            if volume_profile.empty:
                return 0.0
            
            # Find level with maximum volume
            max_volume_idx = volume_profile['volume'].idxmax()
            poc_level = volume_profile.loc[max_volume_idx, 'price_level']
            
            # Validate POC meets minimum volume threshold
            max_volume_pct = volume_profile.loc[max_volume_idx, 'volume_percentage']
            if max_volume_pct < self.min_volume_for_poc:
                self.logger.warning(f"POC volume percentage {max_volume_pct:.3f} below threshold {self.min_volume_for_poc}")
            
            return poc_level
            
        except Exception as e:
            self.logger.error(f"POC calculation error: {e}")
            return 0.0
    
    def _calculate_value_areas(self, volume_profile: pd.DataFrame) -> Tuple[float, float]:
        """Calculate Value Area High and Low levels"""
        try:
            if volume_profile.empty:
                return 0.0, 0.0
            
            # Sort by volume (descending)
            sorted_profile = volume_profile.sort_values('volume', ascending=False)
            
            # Calculate cumulative volume percentage
            sorted_profile['cumulative_volume'] = sorted_profile['volume_percentage'].cumsum()
            
            # Find levels within value area percentage
            value_area_levels = sorted_profile[
                sorted_profile['cumulative_volume'] <= self.value_area_percentage
            ]
            
            if value_area_levels.empty:
                # Fallback to top volume level
                value_area_levels = sorted_profile.head(1)
            
            # Get high and low of value area
            value_area_high = value_area_levels['price_level'].max()
            value_area_low = value_area_levels['price_level'].min()
            
            return value_area_high, value_area_low
            
        except Exception as e:
            self.logger.error(f"Value area calculation error: {e}")
            return 0.0, 0.0
    
    def _detect_volume_nodes(self, volume_profile: pd.DataFrame, df: pd.DataFrame) -> List[VolumeProfileLevel]:
        """Detect significant volume nodes (high volume levels)"""
        try:
            nodes = []
            
            # Find levels with volume above threshold
            significant_levels = volume_profile[
                volume_profile['volume_percentage'] > self.volume_threshold
            ]
            
            for _, level in significant_levels.iterrows():
                # Calculate POC score (how close to being POC)
                max_volume = volume_profile['volume'].max()
                poc_score = level['volume'] / max_volume if max_volume > 0 else 0
                
                # Calculate value area score
                value_area_score = min(level['volume_percentage'] / self.value_area_percentage, 1.0)
                
                nodes.append(VolumeProfileLevel(
                    price_level=level['price_level'],
                    volume=level['volume'],
                    volume_percentage=level['volume_percentage'],
                    poc_score=poc_score,
                    value_area_score=value_area_score,
                    timestamp=df.index[-1] if not df.empty else datetime.now(),
                    level_type=VolumeProfileType.VOLUME_NODE,
                    metadata={
                        'threshold_exceeded': level['volume_percentage'] / self.volume_threshold,
                        'is_poc_candidate': poc_score > 0.8
                    }
                ))
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Volume nodes detection error: {e}")
            return []
    
    def _detect_volume_gaps(self, volume_profile: pd.DataFrame, df: pd.DataFrame) -> List[VolumeProfileLevel]:
        """Detect volume gaps (low volume areas)"""
        try:
            gaps = []
            
            # Find levels with very low volume
            gap_threshold = self.volume_threshold * 0.1  # 10% of normal threshold
            gap_levels = volume_profile[
                volume_profile['volume_percentage'] < gap_threshold
            ]
            
            for _, level in gap_levels.iterrows():
                gaps.append(VolumeProfileLevel(
                    price_level=level['price_level'],
                    volume=level['volume'],
                    volume_percentage=level['volume_percentage'],
                    poc_score=0.0,  # Gaps can't be POC
                    value_area_score=0.0,  # Gaps are outside value area
                    timestamp=df.index[-1] if not df.empty else datetime.now(),
                    level_type=VolumeProfileType.VOLUME_GAP,
                    metadata={
                        'gap_strength': gap_threshold / level['volume_percentage'] if level['volume_percentage'] > 0 else 999,
                        'is_significant_gap': level['volume_percentage'] < gap_threshold * 0.5
                    }
                ))
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Volume gaps detection error: {e}")
            return []
    
    def _detect_single_prints(self, volume_profile: pd.DataFrame, df: pd.DataFrame) -> List[VolumeProfileLevel]:
        """Detect single prints (isolated high volume levels)"""
        try:
            single_prints = []
            
            # Find levels with high volume but isolated
            high_volume_threshold = self.volume_threshold * 2  # Double normal threshold
            high_volume_levels = volume_profile[
                volume_profile['volume_percentage'] > high_volume_threshold
            ]
            
            for _, level in high_volume_levels.iterrows():
                # Check if this level is isolated (surrounding levels have low volume)
                price_level = level['price_level']
                surrounding_levels = volume_profile[
                    (volume_profile['price_level'] >= price_level * 0.995) &
                    (volume_profile['price_level'] <= price_level * 1.005) &
                    (volume_profile['price_level'] != price_level)
                ]
                
                avg_surrounding_volume = surrounding_levels['volume_percentage'].mean()
                isolation_ratio = level['volume_percentage'] / avg_surrounding_volume if avg_surrounding_volume > 0 else 999
                
                if isolation_ratio > 3.0:  # 3x higher than surrounding levels
                    single_prints.append(VolumeProfileLevel(
                        price_level=level['price_level'],
                        volume=level['volume'],
                        volume_percentage=level['volume_percentage'],
                        poc_score=level['volume'] / volume_profile['volume'].max() if volume_profile['volume'].max() > 0 else 0,
                        value_area_score=min(level['volume_percentage'] / self.value_area_percentage, 1.0),
                        timestamp=df.index[-1] if not df.empty else datetime.now(),
                        level_type=VolumeProfileType.SINGLE_PRINT,
                        metadata={
                            'isolation_ratio': isolation_ratio,
                            'surrounding_avg_volume': avg_surrounding_volume,
                            'is_extreme_single_print': isolation_ratio > 5.0
                        }
                    ))
            
            return single_prints
            
        except Exception as e:
            self.logger.error(f"Single prints detection error: {e}")
            return []
    
    def _detect_volume_climax(self, volume_profile: pd.DataFrame, df: pd.DataFrame) -> List[VolumeProfileLevel]:
        """Detect volume climax levels (exhaustion signals)"""
        try:
            climax_levels = []
            
            # Find levels with extremely high volume
            climax_threshold = self.volume_threshold * 5  # 5x normal threshold
            extreme_volume_levels = volume_profile[
                volume_profile['volume_percentage'] > climax_threshold
            ]
            
            for _, level in extreme_volume_levels.iterrows():
                # Check if this represents climax (high volume with price rejection)
                price_level = level['price_level']
                
                # Find candles at this level
                level_candles = df[
                    (df['low'] <= price_level) & 
                    (df['high'] >= price_level)
                ]
                
                if not level_candles.empty:
                    # Calculate price rejection (doji-like patterns)
                    avg_body_size = abs(level_candles['close'] - level_candles['open']).mean()
                    avg_wick_size = (level_candles['high'] - level_candles['low']).mean()
                    rejection_ratio = avg_wick_size / avg_body_size if avg_body_size > 0 else 999
                    
                    if rejection_ratio > 2.0:  # High wick to body ratio indicates rejection
                        climax_levels.append(VolumeProfileLevel(
                            price_level=level['price_level'],
                            volume=level['volume'],
                            volume_percentage=level['volume_percentage'],
                            poc_score=level['volume'] / volume_profile['volume'].max() if volume_profile['volume'].max() > 0 else 0,
                            value_area_score=min(level['volume_percentage'] / self.value_area_percentage, 1.0),
                            timestamp=df.index[-1] if not df.empty else datetime.now(),
                            level_type=VolumeProfileType.VOLUME_CLIMAX,
                            metadata={
                                'rejection_ratio': rejection_ratio,
                                'climax_strength': level['volume_percentage'] / climax_threshold,
                                'is_extreme_climax': level['volume_percentage'] > climax_threshold * 2
                            }
                        ))
            
            return climax_levels
            
        except Exception as e:
            self.logger.error(f"Volume climax detection error: {e}")
            return []
    
    def _detect_volume_exhaustion(self, volume_profile: pd.DataFrame, df: pd.DataFrame) -> List[VolumeProfileLevel]:
        """Detect volume exhaustion levels (low volume after high volume)"""
        try:
            exhaustion_levels = []
            
            # Find levels with very low volume after high volume periods
            exhaustion_threshold = self.volume_threshold * 0.05  # 5% of normal threshold
            low_volume_levels = volume_profile[
                volume_profile['volume_percentage'] < exhaustion_threshold
            ]
            
            for _, level in low_volume_levels.iterrows():
                # Check if this follows a high volume period
                price_level = level['price_level']
                
                # Look for recent high volume at nearby levels
                nearby_high_volume = volume_profile[
                    (volume_profile['price_level'] >= price_level * 0.95) &
                    (volume_profile['price_level'] <= price_level * 1.05) &
                    (volume_profile['volume_percentage'] > self.volume_threshold)
                ]
                
                if not nearby_high_volume.empty:
                    # Calculate exhaustion score
                    max_nearby_volume = nearby_high_volume['volume_percentage'].max()
                    exhaustion_ratio = max_nearby_volume / level['volume_percentage'] if level['volume_percentage'] > 0 else 999
                    
                    if exhaustion_ratio > 10.0:  # 10x volume drop indicates exhaustion
                        exhaustion_levels.append(VolumeProfileLevel(
                            price_level=level['price_level'],
                            volume=level['volume'],
                            volume_percentage=level['volume_percentage'],
                            poc_score=0.0,  # Exhaustion levels can't be POC
                            value_area_score=0.0,  # Exhaustion levels are outside value area
                            timestamp=df.index[-1] if not df.empty else datetime.now(),
                            level_type=VolumeProfileType.VOLUME_EXHAUSTION,
                            metadata={
                                'exhaustion_ratio': exhaustion_ratio,
                                'max_nearby_volume': max_nearby_volume,
                                'is_extreme_exhaustion': exhaustion_ratio > 20.0
                            }
                        ))
            
            return exhaustion_levels
            
        except Exception as e:
            self.logger.error(f"Volume exhaustion detection error: {e}")
            return []
    
    def _calculate_confidence_score(self, 
                                  volume_profile: pd.DataFrame, 
                                  poc_level: float, 
                                  value_area_high: float, 
                                  value_area_low: float) -> float:
        """Calculate confidence score for the analysis"""
        try:
            confidence_factors = []
            
            # Factor 1: POC strength
            if poc_level > 0:
                poc_volume = volume_profile[
                    volume_profile['price_level'] == poc_level
                ]['volume_percentage'].iloc[0] if not volume_profile[
                    volume_profile['price_level'] == poc_level
                ].empty else 0
                poc_strength = min(poc_volume / self.min_volume_for_poc, 1.0)
                confidence_factors.append(poc_strength)
            
            # Factor 2: Value area quality
            if value_area_high > value_area_low:
                value_area_range = (value_area_high - value_area_low) / poc_level if poc_level > 0 else 0
                value_area_quality = max(0, 1.0 - value_area_range)  # Smaller range = higher quality
                confidence_factors.append(value_area_quality)
            
            # Factor 3: Volume distribution
            total_volume = volume_profile['volume'].sum()
            if total_volume > 0:
                volume_concentration = volume_profile['volume_percentage'].max()
                volume_distribution = min(volume_concentration * 2, 1.0)  # Higher concentration = better
                confidence_factors.append(volume_distribution)
            
            # Factor 4: Data quality
            data_quality = min(len(volume_profile) / self.price_levels, 1.0)
            confidence_factors.append(data_quality)
            
            # Calculate weighted average
            if confidence_factors:
                weights = [0.3, 0.3, 0.2, 0.2]  # POC, Value Area, Volume, Data Quality
                weighted_sum = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
                return min(weighted_sum, 1.0)
            
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def get_trading_signals(self, analysis: VolumeProfileAnalysis, current_price: float) -> Dict[str, any]:
        """
        Generate trading signals based on volume profile analysis
        
        Args:
            analysis: Volume profile analysis results
            current_price: Current market price
            
        Returns:
            Dict containing trading signals and recommendations
        """
        try:
            signals = {
                'poc_signal': self._analyze_poc_signal(analysis, current_price),
                'value_area_signal': self._analyze_value_area_signal(analysis, current_price),
                'volume_node_signals': self._analyze_volume_node_signals(analysis, current_price),
                'gap_signals': self._analyze_gap_signals(analysis, current_price),
                'single_print_signals': self._analyze_single_print_signals(analysis, current_price),
                'climax_signals': self._analyze_climax_signals(analysis, current_price),
                'exhaustion_signals': self._analyze_exhaustion_signals(analysis, current_price),
                'overall_signal': 'neutral',
                'confidence': analysis.confidence_score,
                'recommendations': []
            }
            
            # Determine overall signal
            signal_scores = []
            
            if signals['poc_signal']['strength'] > 0.7:
                signal_scores.append(signals['poc_signal']['score'])
            
            if signals['value_area_signal']['strength'] > 0.7:
                signal_scores.append(signals['value_area_signal']['score'])
            
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
    
    def _analyze_poc_signal(self, analysis: VolumeProfileAnalysis, current_price: float) -> Dict[str, any]:
        """Analyze POC-based trading signal"""
        try:
            poc_distance = abs(current_price - analysis.poc_level) / analysis.poc_level if analysis.poc_level > 0 else 999
            
            if poc_distance < 0.01:  # Within 1% of POC
                return {
                    'signal': 'support_resistance',
                    'strength': 0.9,
                    'score': 0.0,  # Neutral at POC
                    'message': f'Price at POC level {analysis.poc_level:.2f}'
                }
            elif current_price < analysis.poc_level:
                return {
                    'signal': 'bullish',
                    'strength': max(0.1, 1.0 - poc_distance),
                    'score': 0.5,
                    'message': f'Price below POC, potential support at {analysis.poc_level:.2f}'
                }
            else:
                return {
                    'signal': 'bearish',
                    'strength': max(0.1, 1.0 - poc_distance),
                    'score': -0.5,
                    'message': f'Price above POC, potential resistance at {analysis.poc_level:.2f}'
                }
        except Exception as e:
            return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}
    
    def _analyze_value_area_signal(self, analysis: VolumeProfileAnalysis, current_price: float) -> Dict[str, any]:
        """Analyze Value Area-based trading signal"""
        try:
            if current_price >= analysis.value_area_low and current_price <= analysis.value_area_high:
                return {
                    'signal': 'neutral',
                    'strength': 0.8,
                    'score': 0.0,
                    'message': f'Price within value area ({analysis.value_area_low:.2f} - {analysis.value_area_high:.2f})'
                }
            elif current_price < analysis.value_area_low:
                return {
                    'signal': 'bullish',
                    'strength': 0.7,
                    'score': 0.6,
                    'message': f'Price below value area, potential bounce to {analysis.value_area_low:.2f}'
                }
            else:
                return {
                    'signal': 'bearish',
                    'strength': 0.7,
                    'score': -0.6,
                    'message': f'Price above value area, potential drop to {analysis.value_area_high:.2f}'
                }
        except Exception as e:
            return {'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}
    
    def _analyze_volume_node_signals(self, analysis: VolumeProfileAnalysis, current_price: float) -> List[Dict[str, any]]:
        """Analyze volume node signals"""
        try:
            signals = []
            for node in analysis.volume_nodes:
                distance = abs(current_price - node.price_level) / node.price_level
                if distance < 0.02:  # Within 2% of volume node
                    signals.append({
                        'signal': 'support_resistance',
                        'strength': node.poc_score,
                        'score': 0.0,
                        'message': f'Price near volume node at {node.price_level:.2f}',
                        'node_type': 'volume_node'
                    })
            return signals
        except Exception as e:
            return [{'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}]
    
    def _analyze_gap_signals(self, analysis: VolumeProfileAnalysis, current_price: float) -> List[Dict[str, any]]:
        """Analyze volume gap signals"""
        try:
            signals = []
            for gap in analysis.volume_gaps:
                distance = abs(current_price - gap.price_level) / gap.price_level
                if distance < 0.01:  # Within 1% of gap
                    signals.append({
                        'signal': 'breakout',
                        'strength': 0.6,
                        'score': 0.0,
                        'message': f'Price at volume gap, potential breakout from {gap.price_level:.2f}',
                        'gap_type': 'volume_gap'
                    })
            return signals
        except Exception as e:
            return [{'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}]
    
    def _analyze_single_print_signals(self, analysis: VolumeProfileAnalysis, current_price: float) -> List[Dict[str, any]]:
        """Analyze single print signals"""
        try:
            signals = []
            for single_print in analysis.single_prints:
                distance = abs(current_price - single_print.price_level) / single_print.price_level
                if distance < 0.01:  # Within 1% of single print
                    signals.append({
                        'signal': 'reversal',
                        'strength': single_print.poc_score,
                        'score': 0.0,
                        'message': f'Price at single print level {single_print.price_level:.2f}',
                        'single_print_type': 'reversal_candidate'
                    })
            return signals
        except Exception as e:
            return [{'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}]
    
    def _analyze_climax_signals(self, analysis: VolumeProfileAnalysis, current_price: float) -> List[Dict[str, any]]:
        """Analyze volume climax signals"""
        try:
            signals = []
            for climax in analysis.volume_climax_levels:
                distance = abs(current_price - climax.price_level) / climax.price_level
                if distance < 0.01:  # Within 1% of climax level
                    signals.append({
                        'signal': 'reversal',
                        'strength': climax.poc_score,
                        'score': 0.0,
                        'message': f'Price at volume climax level {climax.price_level:.2f}',
                        'climax_type': 'exhaustion_reversal'
                    })
            return signals
        except Exception as e:
            return [{'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}]
    
    def _analyze_exhaustion_signals(self, analysis: VolumeProfileAnalysis, current_price: float) -> List[Dict[str, any]]:
        """Analyze volume exhaustion signals"""
        try:
            signals = []
            for exhaustion in analysis.volume_exhaustion_levels:
                distance = abs(current_price - exhaustion.price_level) / exhaustion.price_level
                if distance < 0.01:  # Within 1% of exhaustion level
                    signals.append({
                        'signal': 'continuation',
                        'strength': 0.5,
                        'score': 0.0,
                        'message': f'Price at volume exhaustion level {exhaustion.price_level:.2f}',
                        'exhaustion_type': 'low_volume_continuation'
                    })
            return signals
        except Exception as e:
            return [{'signal': 'neutral', 'strength': 0.0, 'score': 0.0, 'message': f'Error: {e}'}]
