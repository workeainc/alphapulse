"""
Dynamic Support/Resistance Analyzer for AlphaPulse
Advanced support/resistance analysis with volume weighting, psychological levels, and dynamic validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class LevelType(Enum):
    """Support/Resistance level types"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    DYNAMIC = "dynamic"
    VWAP_SUPPORT = "vwap_support"
    VWAP_RESISTANCE = "vwap_resistance"
    VOLUME_NODE = "volume_node"

class PsychologicalLevelType(Enum):
    """Psychological level types"""
    ROUND_NUMBER = "round_number"
    FIBONACCI = "fibonacci"
    PREVIOUS_HIGH_LOW = "previous_high_low"

class InteractionType(Enum):
    """Level interaction types"""
    TOUCH = "touch"
    BOUNCE = "bounce"
    PENETRATION = "penetration"
    BREAK = "break"

@dataclass
class TouchPoint:
    """Individual touch point data"""
    timestamp: datetime
    price: float
    volume: float
    distance_to_level: float
    reaction_strength: float
    
@dataclass
class SupportResistanceLevel:
    """Dynamic support/resistance level"""
    level_type: LevelType
    price_level: float
    strength: float
    confidence: float
    touch_count: int = 0
    first_touch_time: Optional[datetime] = None
    last_touch_time: Optional[datetime] = None
    touch_points: List[TouchPoint] = field(default_factory=list)
    volume_confirmation: bool = False
    avg_volume_at_level: float = 0.0
    volume_spike_ratio: float = 1.0
    institutional_activity: bool = False
    level_age_bars: int = 0
    level_range: float = 0.0
    penetration_count: int = 0
    rejection_count: int = 0
    is_active: bool = True
    is_broken: bool = False
    break_time: Optional[datetime] = None
    break_volume: Optional[float] = None
    market_structure_context: Optional[str] = None
    trend_alignment: Optional[str] = None
    psychological_level: bool = False

@dataclass
class VolumeWeightedLevel:
    """Volume-weighted support/resistance level"""
    level_type: LevelType
    price_level: float
    volume_weight: float
    volume_percentage: float
    total_volume_at_level: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    volume_imbalance: float = 0.0
    time_spent_at_level: int = 0
    price_acceptance: bool = False
    level_efficiency: float = 0.0
    validation_score: float = 0.0
    statistical_significance: float = 0.0
    level_reliability: str = "medium"

@dataclass
class PsychologicalLevel:
    """Psychological support/resistance level"""
    level_type: PsychologicalLevelType
    price_level: float
    psychological_strength: float
    round_number_type: Optional[str] = None
    fibonacci_ratio: Optional[float] = None
    historical_significance: float = 0.0
    reaction_count: int = 0
    penetration_difficulty: float = 0.0
    average_reaction_strength: float = 0.0
    back_test_success_rate: float = 0.0
    forward_test_accuracy: float = 0.0
    reliability_score: float = 0.0

@dataclass
class LevelInteraction:
    """Level interaction analysis"""
    level_id: int
    level_type: LevelType
    interaction_type: InteractionType
    approach_price: float
    interaction_price: float
    reaction_price: Optional[float] = None
    price_distance: float = 0.0
    interaction_volume: float = 0.0
    volume_ratio: float = 1.0
    momentum_strength: float = 0.0
    momentum_direction: Optional[str] = None
    reaction_strength: float = 0.0
    reaction_duration: int = 0
    success_probability: float = 0.0
    actual_outcome: Optional[str] = None

@dataclass
class SupportResistanceAnalysis:
    """Complete support/resistance analysis result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    support_levels: List[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: List[SupportResistanceLevel] = field(default_factory=list)
    volume_weighted_levels: List[VolumeWeightedLevel] = field(default_factory=list)
    psychological_levels: List[PsychologicalLevel] = field(default_factory=list)
    recent_interactions: List[LevelInteraction] = field(default_factory=list)
    overall_strength: float = 0.0
    analysis_confidence: float = 0.0
    market_context: Dict[str, Any] = field(default_factory=dict)

class DynamicSupportResistanceAnalyzer:
    """Advanced dynamic support/resistance analyzer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Analysis settings
        self.min_level_strength = self.config.get('min_level_strength', 0.3)
        self.min_touch_count = self.config.get('min_touch_count', 2)
        self.level_tolerance = self.config.get('level_tolerance', 0.002)  # 0.2%
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.lookback_periods = self.config.get('lookback_periods', 100)
        
        # Volume analysis settings
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', 2.0)
        self.institutional_volume_threshold = self.config.get('institutional_volume_threshold', 5.0)
        
        # Psychological level settings
        self.round_number_levels = [10, 50, 100, 500, 1000, 5000, 10000]
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'levels_detected': 0,
            'volume_weighted_levels': 0,
            'psychological_levels': 0,
            'level_interactions': 0,
            'successful_predictions': 0,
            'last_update': None
        }
        
    async def analyze_support_resistance(self, symbol: str, timeframe: str, 
                                       candlestick_data: List[Dict],
                                       volume_data: Optional[List[Dict]] = None) -> SupportResistanceAnalysis:
        """Comprehensive support/resistance analysis"""
        try:
            if len(candlestick_data) < self.lookback_periods:
                self.logger.warning(f"Insufficient data for {symbol}: {len(candlestick_data)} < {self.lookback_periods}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(candlestick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Detect basic support/resistance levels
            support_levels, resistance_levels = await self._detect_basic_levels(df)
            
            # Enhance with volume analysis
            volume_weighted_levels = await self._analyze_volume_weighted_levels(df)
            
            # Detect psychological levels
            psychological_levels = await self._detect_psychological_levels(df, symbol)
            
            # Validate and score all levels
            support_levels = await self._validate_and_score_levels(df, support_levels)
            resistance_levels = await self._validate_and_score_levels(df, resistance_levels)
            
            # Analyze recent level interactions
            recent_interactions = await self._analyze_level_interactions(
                df, support_levels + resistance_levels
            )
            
            # Calculate overall analysis metrics
            overall_strength = await self._calculate_overall_strength(
                support_levels, resistance_levels, volume_weighted_levels
            )
            
            analysis_confidence = await self._calculate_analysis_confidence(
                support_levels, resistance_levels, volume_weighted_levels, psychological_levels
            )
            
            # Get market context
            market_context = await self._get_market_context(df, support_levels, resistance_levels)
            
            # Create analysis result
            analysis = SupportResistanceAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1],
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volume_weighted_levels=volume_weighted_levels,
                psychological_levels=psychological_levels,
                recent_interactions=recent_interactions,
                overall_strength=overall_strength,
                analysis_confidence=analysis_confidence,
                market_context=market_context
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            self.stats['levels_detected'] += len(support_levels) + len(resistance_levels)
            self.stats['volume_weighted_levels'] += len(volume_weighted_levels)
            self.stats['psychological_levels'] += len(psychological_levels)
            self.stats['level_interactions'] += len(recent_interactions)
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing support/resistance for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    async def _detect_basic_levels(self, df: pd.DataFrame) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Detect basic support and resistance levels"""
        support_levels = []
        resistance_levels = []
        
        try:
            # Find local highs and lows
            for i in range(2, len(df) - 2):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check for resistance (local high)
                if (current_high > df.iloc[i-1]['high'] and current_high > df.iloc[i-2]['high'] and
                    current_high > df.iloc[i+1]['high'] and current_high > df.iloc[i+2]['high']):
                    
                    resistance_levels.append(SupportResistanceLevel(
                        level_type=LevelType.RESISTANCE,
                        price_level=current_high,
                        strength=0.5,  # Initial strength
                        confidence=0.5,  # Initial confidence
                        first_touch_time=df.iloc[i]['timestamp'],
                        last_touch_time=df.iloc[i]['timestamp'],
                        level_age_bars=len(df) - i
                    ))
                
                # Check for support (local low)
                if (current_low < df.iloc[i-1]['low'] and current_low < df.iloc[i-2]['low'] and
                    current_low < df.iloc[i+1]['low'] and current_low < df.iloc[i+2]['low']):
                    
                    support_levels.append(SupportResistanceLevel(
                        level_type=LevelType.SUPPORT,
                        price_level=current_low,
                        strength=0.5,  # Initial strength
                        confidence=0.5,  # Initial confidence
                        first_touch_time=df.iloc[i]['timestamp'],
                        last_touch_time=df.iloc[i]['timestamp'],
                        level_age_bars=len(df) - i
                    ))
            
            # Consolidate nearby levels
            support_levels = await self._consolidate_levels(support_levels)
            resistance_levels = await self._consolidate_levels(resistance_levels)
            
            return support_levels, resistance_levels
            
        except Exception as e:
            self.logger.error(f"Error detecting basic levels: {e}")
            return [], []
    
    async def _consolidate_levels(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Consolidate nearby levels into stronger levels"""
        if not levels:
            return levels
        
        try:
            # Sort levels by price
            levels.sort(key=lambda x: x.price_level)
            
            consolidated = []
            current_group = [levels[0]]
            
            for i in range(1, len(levels)):
                # Check if this level is close to the current group
                price_diff = abs(levels[i].price_level - current_group[-1].price_level) / current_group[-1].price_level
                
                if price_diff <= self.level_tolerance:
                    current_group.append(levels[i])
                else:
                    # Consolidate current group
                    if len(current_group) > 1:
                        consolidated_level = await self._merge_levels(current_group)
                        consolidated.append(consolidated_level)
                    else:
                        consolidated.append(current_group[0])
                    
                    current_group = [levels[i]]
            
            # Handle last group
            if len(current_group) > 1:
                consolidated_level = await self._merge_levels(current_group)
                consolidated.append(consolidated_level)
            else:
                consolidated.append(current_group[0])
            
            return consolidated
            
        except Exception as e:
            self.logger.error(f"Error consolidating levels: {e}")
            return levels
    
    async def _merge_levels(self, levels: List[SupportResistanceLevel]) -> SupportResistanceLevel:
        """Merge multiple nearby levels into one stronger level"""
        try:
            # Calculate weighted average price based on strength
            total_weight = sum(level.strength for level in levels)
            if total_weight == 0:
                total_weight = len(levels)
            
            weighted_price = sum(level.price_level * level.strength for level in levels) / total_weight
            
            # Combine strengths
            combined_strength = min(1.0, sum(level.strength for level in levels) / len(levels) * 1.5)
            
            # Take earliest first touch and latest last touch
            first_touch = min(level.first_touch_time for level in levels if level.first_touch_time)
            last_touch = max(level.last_touch_time for level in levels if level.last_touch_time)
            
            # Combine other properties
            total_touches = sum(level.touch_count for level in levels)
            total_penetrations = sum(level.penetration_count for level in levels)
            total_rejections = sum(level.rejection_count for level in levels)
            
            return SupportResistanceLevel(
                level_type=levels[0].level_type,
                price_level=weighted_price,
                strength=combined_strength,
                confidence=combined_strength * 0.9,  # Slightly lower confidence
                touch_count=total_touches,
                first_touch_time=first_touch,
                last_touch_time=last_touch,
                penetration_count=total_penetrations,
                rejection_count=total_rejections,
                level_age_bars=max(level.level_age_bars for level in levels)
            )
            
        except Exception as e:
            self.logger.error(f"Error merging levels: {e}")
            return levels[0]  # Return first level as fallback
    
    async def _analyze_volume_weighted_levels(self, df: pd.DataFrame) -> List[VolumeWeightedLevel]:
        """Analyze volume-weighted support/resistance levels"""
        volume_levels = []
        
        try:
            if 'volume' not in df.columns:
                return volume_levels
            
            # Create price-volume distribution
            price_volume_map = defaultdict(float)
            
            for _, row in df.iterrows():
                # Distribute volume across the price range of the bar
                price_range = row['high'] - row['low']
                if price_range > 0:
                    # Simple distribution - can be enhanced with more sophisticated methods
                    price_levels = np.linspace(row['low'], row['high'], 10)
                    volume_per_level = row['volume'] / 10
                    
                    for price in price_levels:
                        price_volume_map[round(price, 2)] += volume_per_level
            
            # Find significant volume nodes
            total_volume = sum(price_volume_map.values())
            volume_threshold = total_volume * 0.01  # 1% of total volume (more sensitive)
            
            for price, volume in price_volume_map.items():
                if volume >= volume_threshold:
                    volume_percentage = (volume / total_volume) * 100
                    
                    # Determine level type based on recent price action
                    current_price = df['close'].iloc[-1]
                    if price < current_price:
                        level_type = LevelType.VWAP_SUPPORT
                    else:
                        level_type = LevelType.VWAP_RESISTANCE
                    
                    # Calculate validation score
                    validation_score = min(1.0, volume_percentage / 5.0)  # Normalize to 0-1
                    
                    volume_levels.append(VolumeWeightedLevel(
                        level_type=level_type,
                        price_level=price,
                        volume_weight=volume,
                        volume_percentage=volume_percentage,
                        total_volume_at_level=volume,
                        validation_score=validation_score,
                        level_reliability="high" if validation_score > 0.7 else "medium" if validation_score > 0.4 else "low"
                    ))
            
            # Sort by volume weight and take top levels
            volume_levels.sort(key=lambda x: x.volume_weight, reverse=True)
            return volume_levels[:10]  # Top 10 volume-weighted levels
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume-weighted levels: {e}")
            return []
    
    async def _detect_psychological_levels(self, df: pd.DataFrame, symbol: str) -> List[PsychologicalLevel]:
        """Detect psychological support/resistance levels"""
        psychological_levels = []
        
        try:
            current_price = df['close'].iloc[-1]
            price_range = df['high'].max() - df['low'].min()
            
            # Round number levels
            for base in self.round_number_levels:
                # Find round numbers near current price
                lower_bound = current_price - price_range * 0.2
                upper_bound = current_price + price_range * 0.2
                
                # Generate round number levels
                level = base
                while level < upper_bound:
                    if level > lower_bound:
                        # Determine round number type
                        if level >= 1000:
                            round_type = "major"
                        elif level >= 100:
                            round_type = "minor"
                        else:
                            round_type = "micro"
                        
                        # Calculate psychological strength based on round number significance
                        if round_type == "major":
                            strength = 0.9
                        elif round_type == "minor":
                            strength = 0.7
                        else:
                            strength = 0.5
                        
                        psychological_levels.append(PsychologicalLevel(
                            level_type=PsychologicalLevelType.ROUND_NUMBER,
                            price_level=level,
                            psychological_strength=strength,
                            round_number_type=round_type,
                            reliability_score=strength * 0.8
                        ))
                    
                    level += base
                
                # Lower levels
                level = base
                while level > lower_bound:
                    if level < upper_bound:
                        # Same logic for lower levels
                        if level >= 1000:
                            round_type = "major"
                        elif level >= 100:
                            round_type = "minor"
                        else:
                            round_type = "micro"
                        
                        if round_type == "major":
                            strength = 0.9
                        elif round_type == "minor":
                            strength = 0.7
                        else:
                            strength = 0.5
                        
                        psychological_levels.append(PsychologicalLevel(
                            level_type=PsychologicalLevelType.ROUND_NUMBER,
                            price_level=level,
                            psychological_strength=strength,
                            round_number_type=round_type,
                            reliability_score=strength * 0.8
                        ))
                    
                    level -= base
            
            # Remove duplicates and sort
            seen_prices = set()
            unique_levels = []
            for level in psychological_levels:
                if level.price_level not in seen_prices:
                    seen_prices.add(level.price_level)
                    unique_levels.append(level)
            
            return unique_levels[:20]  # Top 20 psychological levels
            
        except Exception as e:
            self.logger.error(f"Error detecting psychological levels: {e}")
            return []
    
    async def _validate_and_score_levels(self, df: pd.DataFrame, 
                                       levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Validate and score support/resistance levels"""
        try:
            for level in levels:
                # Count touches and calculate statistics
                touches = await self._count_level_touches(df, level)
                level.touch_count = len(touches)
                level.touch_points = touches
                
                if touches:
                    level.last_touch_time = max(touch.timestamp for touch in touches)
                    level.avg_volume_at_level = np.mean([touch.volume for touch in touches])
                
                # Calculate level strength based on multiple factors
                level.strength = await self._calculate_level_strength(df, level)
                
                # Calculate confidence
                level.confidence = await self._calculate_level_confidence(df, level)
                
                # Check volume confirmation
                level.volume_confirmation = await self._check_volume_confirmation(df, level)
                
                # Check for institutional activity
                level.institutional_activity = await self._check_institutional_activity(df, level)
                
                # Update level status
                level.is_broken = await self._check_if_level_broken(df, level)
                if level.is_broken:
                    level.is_active = False
                    level.break_time = df['timestamp'].iloc[-1]
                
            # Filter levels by minimum strength
            validated_levels = [level for level in levels if level.strength >= self.min_level_strength]
            
            # Sort by strength
            validated_levels.sort(key=lambda x: x.strength, reverse=True)
            
            return validated_levels
            
        except Exception as e:
            self.logger.error(f"Error validating levels: {e}")
            return levels
    
    async def _count_level_touches(self, df: pd.DataFrame, level: SupportResistanceLevel) -> List[TouchPoint]:
        """Count and analyze touches to a support/resistance level"""
        touches = []
        
        try:
            tolerance = level.price_level * self.level_tolerance
            
            for i, row in df.iterrows():
                # Check if price touched the level
                if level.level_type == LevelType.SUPPORT:
                    if row['low'] <= level.price_level + tolerance and row['low'] >= level.price_level - tolerance:
                        # Calculate reaction strength
                        if i < len(df) - 1:
                            reaction_strength = (df.iloc[i+1]['close'] - row['low']) / row['low']
                        else:
                            reaction_strength = 0.0
                        
                        touches.append(TouchPoint(
                            timestamp=row['timestamp'],
                            price=row['low'],
                            volume=row['volume'] if 'volume' in row else 0.0,
                            distance_to_level=abs(row['low'] - level.price_level),
                            reaction_strength=reaction_strength
                        ))
                
                elif level.level_type == LevelType.RESISTANCE:
                    if row['high'] >= level.price_level - tolerance and row['high'] <= level.price_level + tolerance:
                        # Calculate reaction strength
                        if i < len(df) - 1:
                            reaction_strength = (row['high'] - df.iloc[i+1]['close']) / row['high']
                        else:
                            reaction_strength = 0.0
                        
                        touches.append(TouchPoint(
                            timestamp=row['timestamp'],
                            price=row['high'],
                            volume=row['volume'] if 'volume' in row else 0.0,
                            distance_to_level=abs(row['high'] - level.price_level),
                            reaction_strength=reaction_strength
                        ))
            
            return touches
            
        except Exception as e:
            self.logger.error(f"Error counting level touches: {e}")
            return []
    
    async def _calculate_level_strength(self, df: pd.DataFrame, level: SupportResistanceLevel) -> float:
        """Calculate the strength of a support/resistance level"""
        try:
            strength = 0.0
            
            # Touch count factor (more touches = stronger)
            touch_factor = min(1.0, level.touch_count / 5.0)
            strength += touch_factor * 0.3
            
            # Age factor (older levels that still hold = stronger)
            age_factor = min(1.0, level.level_age_bars / 50.0)
            strength += age_factor * 0.2
            
            # Reaction strength factor
            if level.touch_points:
                avg_reaction = np.mean([abs(touch.reaction_strength) for touch in level.touch_points])
                reaction_factor = min(1.0, avg_reaction * 20)  # Scale reaction strength
                strength += reaction_factor * 0.3
            
            # Volume confirmation factor
            if level.volume_confirmation:
                strength += 0.1
            
            # Institutional activity factor
            if level.institutional_activity:
                strength += 0.1
            
            return min(1.0, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating level strength: {e}")
            return 0.5
    
    async def _calculate_level_confidence(self, df: pd.DataFrame, level: SupportResistanceLevel) -> float:
        """Calculate confidence in the support/resistance level"""
        try:
            confidence = level.strength * 0.8  # Base confidence from strength
            
            # Statistical significance
            if level.touch_count >= 3:
                confidence += 0.1
            
            # Recent activity
            if level.last_touch_time and level.last_touch_time >= df['timestamp'].iloc[-10]:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating level confidence: {e}")
            return 0.5
    
    async def _check_volume_confirmation(self, df: pd.DataFrame, level: SupportResistanceLevel) -> bool:
        """Check if volume confirms the support/resistance level"""
        try:
            if not level.touch_points or 'volume' not in df.columns:
                return False
            
            # Calculate average volume
            avg_volume = df['volume'].mean()
            
            # Check if volume at touches is above average
            touch_volumes = [touch.volume for touch in level.touch_points]
            avg_touch_volume = np.mean(touch_volumes)
            
            return bool(avg_touch_volume > avg_volume * self.volume_threshold)
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    async def _check_institutional_activity(self, df: pd.DataFrame, level: SupportResistanceLevel) -> bool:
        """Check for institutional activity at the level"""
        try:
            if not level.touch_points or 'volume' not in df.columns:
                return False
            
            # Look for volume spikes at level touches
            avg_volume = df['volume'].mean()
            
            for touch in level.touch_points:
                if touch.volume > avg_volume * self.institutional_volume_threshold:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking institutional activity: {e}")
            return False
    
    async def _check_if_level_broken(self, df: pd.DataFrame, level: SupportResistanceLevel) -> bool:
        """Check if a support/resistance level has been broken"""
        try:
            recent_prices = df.tail(5)  # Check last 5 bars
            tolerance = level.price_level * self.level_tolerance
            
            if level.level_type == LevelType.SUPPORT:
                # Support is broken if price closes significantly below
                for _, row in recent_prices.iterrows():
                    if row['close'] < level.price_level - tolerance:
                        return True
            
            elif level.level_type == LevelType.RESISTANCE:
                # Resistance is broken if price closes significantly above
                for _, row in recent_prices.iterrows():
                    if row['close'] > level.price_level + tolerance:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if level broken: {e}")
            return False
    
    async def _analyze_level_interactions(self, df: pd.DataFrame, 
                                        levels: List[SupportResistanceLevel]) -> List[LevelInteraction]:
        """Analyze recent interactions with support/resistance levels"""
        interactions = []
        
        try:
            # Analyze last 20 bars for interactions
            recent_data = df.tail(20)
            
            for level in levels:
                for i, row in recent_data.iterrows():
                    # Check if price interacted with level
                    tolerance = level.price_level * self.level_tolerance
                    
                    if level.level_type == LevelType.SUPPORT:
                        if row['low'] <= level.price_level + tolerance:
                            interaction_type = await self._determine_interaction_type(
                                df, i, level, row['low']
                            )
                            
                            interactions.append(LevelInteraction(
                                level_id=id(level),  # Simple ID for now
                                level_type=level.level_type,
                                interaction_type=interaction_type,
                                approach_price=row['open'],
                                interaction_price=row['low'],
                                reaction_price=row['close'],
                                price_distance=abs(row['low'] - level.price_level),
                                interaction_volume=row['volume'] if 'volume' in row else 0.0
                            ))
                    
                    elif level.level_type == LevelType.RESISTANCE:
                        if row['high'] >= level.price_level - tolerance:
                            interaction_type = await self._determine_interaction_type(
                                df, i, level, row['high']
                            )
                            
                            interactions.append(LevelInteraction(
                                level_id=id(level),  # Simple ID for now
                                level_type=level.level_type,
                                interaction_type=interaction_type,
                                approach_price=row['open'],
                                interaction_price=row['high'],
                                reaction_price=row['close'],
                                price_distance=abs(row['high'] - level.price_level),
                                interaction_volume=row['volume'] if 'volume' in row else 0.0
                            ))
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error analyzing level interactions: {e}")
            return []
    
    async def _determine_interaction_type(self, df: pd.DataFrame, bar_index: int, 
                                        level: SupportResistanceLevel, touch_price: float) -> InteractionType:
        """Determine the type of interaction with a level"""
        try:
            tolerance = level.price_level * self.level_tolerance
            
            # Check if it was a penetration or break
            if level.level_type == LevelType.SUPPORT:
                if touch_price < level.price_level - tolerance:
                    return InteractionType.BREAK
                else:
                    return InteractionType.BOUNCE
            
            elif level.level_type == LevelType.RESISTANCE:
                if touch_price > level.price_level + tolerance:
                    return InteractionType.BREAK
                else:
                    return InteractionType.BOUNCE
            
            return InteractionType.TOUCH
            
        except Exception as e:
            self.logger.error(f"Error determining interaction type: {e}")
            return InteractionType.TOUCH
    
    async def _calculate_overall_strength(self, support_levels: List[SupportResistanceLevel],
                                        resistance_levels: List[SupportResistanceLevel],
                                        volume_levels: List[VolumeWeightedLevel]) -> float:
        """Calculate overall support/resistance strength"""
        try:
            if not support_levels and not resistance_levels:
                return 0.0
            
            # Average strength of all levels
            all_strengths = []
            all_strengths.extend([level.strength for level in support_levels])
            all_strengths.extend([level.strength for level in resistance_levels])
            
            if all_strengths:
                base_strength = np.mean(all_strengths)
            else:
                base_strength = 0.0
            
            # Bonus for volume-weighted levels
            if volume_levels:
                volume_bonus = min(0.2, len(volume_levels) * 0.02)
                base_strength += volume_bonus
            
            return min(1.0, base_strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall strength: {e}")
            return 0.5
    
    async def _calculate_analysis_confidence(self, support_levels: List[SupportResistanceLevel],
                                           resistance_levels: List[SupportResistanceLevel],
                                           volume_levels: List[VolumeWeightedLevel],
                                           psychological_levels: List[PsychologicalLevel]) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Level confidence
            all_confidences = []
            all_confidences.extend([level.confidence for level in support_levels])
            all_confidences.extend([level.confidence for level in resistance_levels])
            
            if all_confidences:
                avg_confidence = np.mean(all_confidences)
                confidence += avg_confidence * 0.3
            
            # Volume confirmation bonus
            volume_confirmed = sum(1 for level in support_levels + resistance_levels if level.volume_confirmation)
            total_levels = len(support_levels) + len(resistance_levels)
            if total_levels > 0:
                volume_confirmation_ratio = volume_confirmed / total_levels
                confidence += volume_confirmation_ratio * 0.1
            
            # Volume-weighted levels bonus
            if volume_levels:
                confidence += min(0.1, len(volume_levels) * 0.01)
            
            # Psychological levels bonus
            if psychological_levels:
                confidence += min(0.1, len(psychological_levels) * 0.005)
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    async def _get_market_context(self, df: pd.DataFrame,
                                support_levels: List[SupportResistanceLevel],
                                resistance_levels: List[SupportResistanceLevel]) -> Dict[str, Any]:
        """Get market context for the analysis"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            nearest_support = None
            nearest_resistance = None
            
            for level in support_levels:
                if level.price_level < current_price:
                    if not nearest_support or level.price_level > nearest_support.price_level:
                        nearest_support = level
            
            for level in resistance_levels:
                if level.price_level > current_price:
                    if not nearest_resistance or level.price_level < nearest_resistance.price_level:
                        nearest_resistance = level
            
            # Calculate distances
            support_distance = ((current_price - nearest_support.price_level) / current_price * 100) if nearest_support else None
            resistance_distance = ((nearest_resistance.price_level - current_price) / current_price * 100) if nearest_resistance else None
            
            return {
                'current_price': current_price,
                'nearest_support': nearest_support.price_level if nearest_support else None,
                'nearest_resistance': nearest_resistance.price_level if nearest_resistance else None,
                'support_distance_pct': support_distance,
                'resistance_distance_pct': resistance_distance,
                'total_support_levels': len(support_levels),
                'total_resistance_levels': len(resistance_levels),
                'strong_support_levels': len([l for l in support_levels if l.strength > 0.7]),
                'strong_resistance_levels': len([l for l in resistance_levels if l.strength > 0.7])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {e}")
            return {}
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> SupportResistanceAnalysis:
        """Get default analysis when insufficient data"""
        return SupportResistanceAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            overall_strength=0.0,
            analysis_confidence=0.0
        )

# Global instance
dynamic_support_resistance_analyzer = DynamicSupportResistanceAnalyzer()
