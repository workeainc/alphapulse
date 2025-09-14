"""
Demand and Supply Zone Analyzer for AlphaPlus
Comprehensive demand and supply zone analysis including zone detection, volume profile analysis, and breakout detection
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ZoneType(Enum):
    """Zone types"""
    DEMAND = 'demand'
    SUPPLY = 'supply'

class ZoneStrength(Enum):
    """Zone strength levels"""
    WEAK = 'weak'
    MODERATE = 'moderate'
    STRONG = 'strong'
    VERY_STRONG = 'very_strong'

class VolumeNodeType(Enum):
    """Volume node types"""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class BreakoutType(Enum):
    """Breakout types"""
    DEMAND_BREAKOUT = 'demand_breakout'
    SUPPLY_BREAKOUT = 'supply_breakout'
    DEMAND_BREAKDOWN = 'demand_breakdown'
    SUPPLY_BREAKDOWN = 'supply_breakdown'

class InteractionType(Enum):
    """Zone interaction types"""
    TOUCH = 'touch'
    BOUNCE = 'bounce'
    PENETRATION = 'penetration'
    REJECTION = 'rejection'

@dataclass
class DemandSupplyZone:
    """Demand or supply zone data structure"""
    timestamp: datetime
    symbol: str
    timeframe: str
    zone_type: ZoneType
    zone_start_price: float
    zone_end_price: float
    zone_volume: float
    zone_strength: float  # 0 to 1
    zone_confidence: float  # 0 to 1
    zone_touches: int
    zone_duration_hours: Optional[int]
    zone_breakout_direction: Optional[str]  # 'up', 'down', 'none'
    zone_breakout_strength: Optional[float]  # 0 to 1
    zone_volume_profile: Dict[str, Any]
    zone_order_flow: Dict[str, Any]
    zone_metadata: Dict[str, Any]

@dataclass
class VolumeProfileNode:
    """Volume profile node data structure"""
    timestamp: datetime
    symbol: str
    timeframe: str
    price_level: float
    volume_at_level: float
    volume_percentage: float  # Percentage of total volume
    volume_node_type: VolumeNodeType
    volume_concentration: float  # 0 to 1
    price_efficiency: Optional[float]  # How efficient price movement is at this level
    volume_trend: Optional[str]  # 'increasing', 'decreasing', 'stable'
    volume_metadata: Dict[str, Any]

@dataclass
class ZoneBreakout:
    """Zone breakout data structure"""
    timestamp: datetime
    symbol: str
    timeframe: str
    zone_id: int
    breakout_type: BreakoutType
    breakout_price: float
    breakout_volume: float
    breakout_strength: float  # 0 to 1
    breakout_confidence: float  # 0 to 1
    breakout_volume_ratio: float  # Volume vs average
    breakout_momentum: Optional[float]  # Price momentum during breakout
    breakout_retest: Optional[bool]  # Whether price retested the zone
    breakout_metadata: Dict[str, Any]

@dataclass
class ZoneInteraction:
    """Zone interaction data structure"""
    timestamp: datetime
    symbol: str
    timeframe: str
    zone_id: int
    interaction_type: InteractionType
    interaction_price: float
    interaction_volume: float
    interaction_strength: float  # 0 to 1
    interaction_confidence: float  # 0 to 1
    interaction_duration_minutes: Optional[int]
    interaction_momentum: Optional[float]  # Price momentum during interaction
    interaction_metadata: Dict[str, Any]

@dataclass
class DemandSupplyAnalysis:
    """Comprehensive demand and supply analysis result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    demand_zones: List[DemandSupplyZone]
    supply_zones: List[DemandSupplyZone]
    volume_profile_nodes: List[VolumeProfileNode]
    zone_breakouts: List[ZoneBreakout]
    zone_interactions: List[ZoneInteraction]
    strongest_demand_zone: Optional[DemandSupplyZone]
    strongest_supply_zone: Optional[DemandSupplyZone]
    volume_profile_summary: Dict[str, Any]
    zone_analysis_summary: Dict[str, Any]
    analysis_confidence: float  # 0 to 1
    market_context: Dict[str, Any]
    analysis_metadata: Dict[str, Any]

class DemandSupplyZoneAnalyzer:
    """
    Advanced Demand and Supply Zone Analyzer
    Provides comprehensive demand and supply zone analysis including zone detection, volume profile analysis, and breakout detection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.min_zone_touches = self.config.get('min_zone_touches', 2)
        self.zone_price_threshold = self.config.get('zone_price_threshold', 0.02)  # 2% price range
        self.volume_threshold = self.config.get('volume_threshold', 0.1)  # 10% of average volume
        self.breakout_threshold = self.config.get('breakout_threshold', 0.03)  # 3% breakout
        self.min_data_points = self.config.get('min_data_points', 100)
        self.volume_profile_bins = self.config.get('volume_profile_bins', 50)
        self.zone_strength_threshold = self.config.get('zone_strength_threshold', 0.6)
        
        # Statistics tracking
        self.stats = {
            'demand_zones_detected': 0,
            'supply_zones_detected': 0,
            'volume_nodes_analyzed': 0,
            'breakouts_detected': 0,
            'interactions_tracked': 0,
            'total_analyses': 0,
            'errors': 0
        }
        
        logger.info("Demand Supply Zone Analyzer initialized")
    
    async def analyze_demand_supply_zones(self, symbol: str, timeframe: str, 
                                        data: pd.DataFrame) -> DemandSupplyAnalysis:
        """
        Perform comprehensive demand and supply zone analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            data: OHLCV data
            
        Returns:
            DemandSupplyAnalysis: Comprehensive demand and supply analysis
        """
        try:
            logger.info(f"🔄 Analyzing demand/supply zones for {symbol} ({timeframe})")
            
            # Validate input data
            if len(data) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} < {self.min_data_points}")
                return await self._create_default_analysis(symbol, timeframe)
            
            # Perform individual analyses
            demand_zones = await self._detect_demand_zones(symbol, timeframe, data)
            supply_zones = await self._detect_supply_zones(symbol, timeframe, data)
            volume_profile_nodes = await self._analyze_volume_profile(symbol, timeframe, data)
            zone_breakouts = await self._detect_zone_breakouts(symbol, timeframe, data, demand_zones + supply_zones)
            zone_interactions = await self._track_zone_interactions(symbol, timeframe, data, demand_zones + supply_zones)
            
            # Find strongest zones
            strongest_demand_zone = max(demand_zones, key=lambda z: z.zone_strength) if demand_zones else None
            strongest_supply_zone = max(supply_zones, key=lambda z: z.zone_strength) if supply_zones else None
            
            # Generate summaries
            volume_profile_summary = await self._generate_volume_profile_summary(volume_profile_nodes)
            zone_analysis_summary = await self._generate_zone_analysis_summary(demand_zones, supply_zones)
            
            # Calculate analysis confidence
            analysis_confidence = await self._calculate_analysis_confidence(
                demand_zones, supply_zones, volume_profile_nodes, zone_breakouts
            )
            
            # Generate market context
            market_context = await self._calculate_market_context(symbol, data)
            
            # Create comprehensive analysis
            analysis = DemandSupplyAnalysis(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                demand_zones=demand_zones,
                supply_zones=supply_zones,
                volume_profile_nodes=volume_profile_nodes,
                zone_breakouts=zone_breakouts,
                zone_interactions=zone_interactions,
                strongest_demand_zone=strongest_demand_zone,
                strongest_supply_zone=strongest_supply_zone,
                volume_profile_summary=volume_profile_summary,
                zone_analysis_summary=zone_analysis_summary,
                analysis_confidence=analysis_confidence,
                market_context=market_context,
                analysis_metadata={
                    'data_points_analyzed': len(data),
                    'analysis_duration_ms': 0,  # TODO: Add timing
                    'config_used': self.config
                }
            )
            
            # Update statistics
            self.stats['total_analyses'] += 1
            self.stats['demand_zones_detected'] += len(demand_zones)
            self.stats['supply_zones_detected'] += len(supply_zones)
            self.stats['volume_nodes_analyzed'] += len(volume_profile_nodes)
            self.stats['breakouts_detected'] += len(zone_breakouts)
            self.stats['interactions_tracked'] += len(zone_interactions)
            
            logger.info(f"✅ Demand/supply zone analysis completed for {symbol}: {len(demand_zones)} demand zones, {len(supply_zones)} supply zones")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing demand/supply zones for {symbol}: {e}")
            self.stats['errors'] += 1
            return await self._create_default_analysis(symbol, timeframe)
    
    async def _detect_demand_zones(self, symbol: str, timeframe: str, 
                                 data: pd.DataFrame) -> List[DemandSupplyZone]:
        """Detect demand zones (support areas)"""
        try:
            demand_zones = []
            
            if len(data) < 20:
                return demand_zones
            
            # Find local minima (potential demand zones)
            window_size = min(10, len(data) // 10)
            local_minima = []
            
            for i in range(window_size, len(data) - window_size):
                current_low = data.iloc[i]['low']
                left_window = data.iloc[i-window_size:i]['low']
                right_window = data.iloc[i+1:i+window_size+1]['low']
                
                # Check if current point is a local minimum
                if (current_low <= left_window.min() and 
                    current_low <= right_window.min()):
                    local_minima.append({
                        'index': i,
                        'price': current_low,
                        'timestamp': data.iloc[i]['timestamp'],
                        'volume': data.iloc[i]['volume']
                    })
            
            # Group nearby minima into zones
            zones = await self._group_price_levels_into_zones(local_minima, 'demand')
            
            # Analyze each zone
            for zone_data in zones:
                zone = await self._analyze_zone(symbol, timeframe, zone_data, ZoneType.DEMAND, data)
                if zone and zone.zone_strength >= self.zone_strength_threshold:
                    demand_zones.append(zone)
            
            return demand_zones
            
        except Exception as e:
            logger.error(f"Error detecting demand zones: {e}")
            return []
    
    async def _detect_supply_zones(self, symbol: str, timeframe: str, 
                                 data: pd.DataFrame) -> List[DemandSupplyZone]:
        """Detect supply zones (resistance areas)"""
        try:
            supply_zones = []
            
            if len(data) < 20:
                return supply_zones
            
            # Find local maxima (potential supply zones)
            window_size = min(10, len(data) // 10)
            local_maxima = []
            
            for i in range(window_size, len(data) - window_size):
                current_high = data.iloc[i]['high']
                left_window = data.iloc[i-window_size:i]['high']
                right_window = data.iloc[i+1:i+window_size+1]['high']
                
                # Check if current point is a local maximum
                if (current_high >= left_window.max() and 
                    current_high >= right_window.max()):
                    local_maxima.append({
                        'index': i,
                        'price': current_high,
                        'timestamp': data.iloc[i]['timestamp'],
                        'volume': data.iloc[i]['volume']
                    })
            
            # Group nearby maxima into zones
            zones = await self._group_price_levels_into_zones(local_maxima, 'supply')
            
            # Analyze each zone
            for zone_data in zones:
                zone = await self._analyze_zone(symbol, timeframe, zone_data, ZoneType.SUPPLY, data)
                if zone and zone.zone_strength >= self.zone_strength_threshold:
                    supply_zones.append(zone)
            
            return supply_zones
            
        except Exception as e:
            logger.error(f"Error detecting supply zones: {e}")
            return []
    
    async def _group_price_levels_into_zones(self, price_levels: List[Dict], zone_type: str) -> List[Dict]:
        """Group nearby price levels into zones"""
        try:
            if not price_levels:
                return []
            
            # Sort by price
            sorted_levels = sorted(price_levels, key=lambda x: x['price'])
            
            zones = []
            current_zone = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                # Check if this level is close to the current zone
                zone_prices = [l['price'] for l in current_zone]
                avg_zone_price = np.mean(zone_prices)
                price_diff = abs(level['price'] - avg_zone_price) / avg_zone_price
                
                if price_diff <= self.zone_price_threshold:
                    # Add to current zone
                    current_zone.append(level)
                else:
                    # Start new zone
                    if len(current_zone) >= self.min_zone_touches:
                        zones.append({
                            'levels': current_zone,
                            'start_price': min(l['price'] for l in current_zone),
                            'end_price': max(l['price'] for l in current_zone),
                            'avg_price': np.mean([l['price'] for l in current_zone]),
                            'total_volume': sum(l['volume'] for l in current_zone),
                            'touches': len(current_zone)
                        })
                    current_zone = [level]
            
            # Add the last zone
            if len(current_zone) >= self.min_zone_touches:
                zones.append({
                    'levels': current_zone,
                    'start_price': min(l['price'] for l in current_zone),
                    'end_price': max(l['price'] for l in current_zone),
                    'avg_price': np.mean([l['price'] for l in current_zone]),
                    'total_volume': sum(l['volume'] for l in current_zone),
                    'touches': len(current_zone)
                })
            
            return zones
            
        except Exception as e:
            logger.error(f"Error grouping price levels into zones: {e}")
            return []
    
    async def _analyze_zone(self, symbol: str, timeframe: str, zone_data: Dict, 
                          zone_type: ZoneType, data: pd.DataFrame) -> Optional[DemandSupplyZone]:
        """Analyze a single zone"""
        try:
            # Calculate zone metrics
            zone_start_price = zone_data['start_price']
            zone_end_price = zone_data['end_price']
            zone_volume = zone_data['total_volume']
            zone_touches = zone_data['touches']
            
            # Calculate zone strength based on touches and volume
            volume_score = min(1.0, zone_volume / (data['volume'].mean() * zone_touches))
            touch_score = min(1.0, zone_touches / 5)  # Normalize to 5 touches
            zone_strength = (volume_score * 0.6 + touch_score * 0.4)
            
            # Calculate zone confidence
            zone_confidence = min(1.0, zone_touches / 3)  # More touches = higher confidence
            
            # Calculate zone duration
            timestamps = [level['timestamp'] for level in zone_data['levels']]
            zone_duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            
            # Analyze volume profile within zone
            zone_volume_profile = await self._analyze_zone_volume_profile(zone_data, data)
            
            # Analyze order flow within zone
            zone_order_flow = await self._analyze_zone_order_flow(zone_data, data)
            
            # Check for breakouts
            breakout_direction, breakout_strength = await self._check_zone_breakout(
                zone_data, zone_type, data
            )
            
            return DemandSupplyZone(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                zone_type=zone_type,
                zone_start_price=zone_start_price,
                zone_end_price=zone_end_price,
                zone_volume=zone_volume,
                zone_strength=zone_strength,
                zone_confidence=zone_confidence,
                zone_touches=zone_touches,
                zone_duration_hours=int(zone_duration_hours) if zone_duration_hours > 0 else None,
                zone_breakout_direction=breakout_direction,
                zone_breakout_strength=breakout_strength,
                zone_volume_profile=zone_volume_profile,
                zone_order_flow=zone_order_flow,
                zone_metadata={
                    'avg_price': zone_data['avg_price'],
                    'price_range': zone_end_price - zone_start_price,
                    'volume_per_touch': zone_volume / zone_touches
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing zone: {e}")
            return None
    
    async def _analyze_zone_volume_profile(self, zone_data: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile within a zone"""
        try:
            zone_prices = [level['price'] for level in zone_data['levels']]
            zone_volumes = [level['volume'] for level in zone_data['levels']]
            
            return {
                'total_volume': sum(zone_volumes),
                'avg_volume_per_touch': np.mean(zone_volumes),
                'volume_distribution': {
                    'min': min(zone_volumes),
                    'max': max(zone_volumes),
                    'std': np.std(zone_volumes)
                },
                'price_volume_correlation': np.corrcoef(zone_prices, zone_volumes)[0, 1] if len(zone_prices) > 1 else 0,
                'volume_concentration': len([v for v in zone_volumes if v > np.mean(zone_volumes)]) / len(zone_volumes)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing zone volume profile: {e}")
            return {}
    
    async def _analyze_zone_order_flow(self, zone_data: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow within a zone"""
        try:
            # This would integrate with order flow analysis in a real implementation
            # For now, return basic metrics
            zone_volumes = [level['volume'] for level in zone_data['levels']]
            
            return {
                'buy_pressure': 0.5,  # Placeholder - would be calculated from order flow
                'sell_pressure': 0.5,  # Placeholder - would be calculated from order flow
                'order_flow_imbalance': 0.0,  # Placeholder
                'volume_intensity': np.mean(zone_volumes) / data['volume'].mean() if data['volume'].mean() > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing zone order flow: {e}")
            return {}
    
    async def _check_zone_breakout(self, zone_data: Dict, zone_type: ZoneType, 
                                 data: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Check if a zone has been broken out of"""
        try:
            zone_start_price = zone_data['start_price']
            zone_end_price = zone_data['end_price']
            
            # Get recent price data
            recent_data = data.tail(20)  # Last 20 periods
            
            if zone_type == ZoneType.DEMAND:
                # Check for breakdown (price below demand zone)
                if recent_data['low'].min() < zone_start_price:
                    breakdown_strength = (zone_start_price - recent_data['low'].min()) / zone_start_price
                    return 'down', min(1.0, breakdown_strength / self.breakout_threshold)
            
            elif zone_type == ZoneType.SUPPLY:
                # Check for breakout (price above supply zone)
                if recent_data['high'].max() > zone_end_price:
                    breakout_strength = (recent_data['high'].max() - zone_end_price) / zone_end_price
                    return 'up', min(1.0, breakout_strength / self.breakout_threshold)
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error checking zone breakout: {e}")
            return None, None
    
    async def _analyze_volume_profile(self, symbol: str, timeframe: str, 
                                    data: pd.DataFrame) -> List[VolumeProfileNode]:
        """Analyze volume profile across price levels"""
        try:
            volume_profile_nodes = []
            
            if len(data) < 20:
                return volume_profile_nodes
            
            # Create price bins for volume profile
            price_range = data['high'].max() - data['low'].min()
            bin_size = price_range / self.volume_profile_bins
            
            if bin_size <= 0:
                return volume_profile_nodes
            
            # Calculate volume at each price level
            for i in range(self.volume_profile_bins):
                price_level = data['low'].min() + (i * bin_size)
                next_price_level = price_level + bin_size
                
                # Find data points within this price range
                mask = ((data['low'] <= next_price_level) & (data['high'] >= price_level))
                level_data = data[mask]
                
                if len(level_data) > 0:
                    volume_at_level = level_data['volume'].sum()
                    total_volume = data['volume'].sum()
                    volume_percentage = volume_at_level / total_volume if total_volume > 0 else 0
                    
                    # Determine node type
                    if volume_percentage > 0.1:  # More than 10% of total volume
                        node_type = VolumeNodeType.HIGH
                    elif volume_percentage > 0.05:  # More than 5% of total volume
                        node_type = VolumeNodeType.MEDIUM
                    else:
                        node_type = VolumeNodeType.LOW
                    
                    # Calculate volume concentration
                    volume_concentration = volume_at_level / len(level_data) if len(level_data) > 0 else 0
                    
                    # Calculate price efficiency (how much price moved at this level)
                    if len(level_data) > 1:
                        price_changes = level_data['close'].pct_change().abs()
                        price_efficiency = price_changes.mean()
                    else:
                        price_efficiency = None
                    
                    # Determine volume trend
                    if len(level_data) > 5:
                        recent_volume = level_data.tail(3)['volume'].mean()
                        earlier_volume = level_data.head(3)['volume'].mean()
                        if recent_volume > earlier_volume * 1.2:
                            volume_trend = 'increasing'
                        elif recent_volume < earlier_volume * 0.8:
                            volume_trend = 'decreasing'
                        else:
                            volume_trend = 'stable'
                    else:
                        volume_trend = None
                    
                    node = VolumeProfileNode(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        timeframe=timeframe,
                        price_level=price_level,
                        volume_at_level=volume_at_level,
                        volume_percentage=volume_percentage,
                        volume_node_type=node_type,
                        volume_concentration=volume_concentration,
                        price_efficiency=price_efficiency,
                        volume_trend=volume_trend,
                        volume_metadata={
                            'data_points': len(level_data),
                            'price_range': next_price_level - price_level
                        }
                    )
                    
                    volume_profile_nodes.append(node)
            
            return volume_profile_nodes
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return []
    
    async def _detect_zone_breakouts(self, symbol: str, timeframe: str, data: pd.DataFrame,
                                   zones: List[DemandSupplyZone]) -> List[ZoneBreakout]:
        """Detect zone breakouts"""
        try:
            breakouts = []
            
            if not zones or len(data) < 10:
                return breakouts
            
            recent_data = data.tail(10)
            
            for zone in zones:
                # Check for breakouts based on zone type
                if zone.zone_type == ZoneType.DEMAND:
                    # Check for breakdown
                    if recent_data['low'].min() < zone.zone_start_price:
                        breakout_price = recent_data['low'].min()
                        breakout_volume = recent_data[recent_data['low'] <= zone.zone_start_price]['volume'].sum()
                        
                        breakout = ZoneBreakout(
                            timestamp=datetime.now(timezone.utc),
                            symbol=symbol,
                            timeframe=timeframe,
                            zone_id=id(zone),  # Use object id as placeholder
                            breakout_type=BreakoutType.DEMAND_BREAKDOWN,
                            breakout_price=breakout_price,
                            breakout_volume=breakout_volume,
                            breakout_strength=zone.zone_breakout_strength or 0.0,
                            breakout_confidence=zone.zone_confidence,
                            breakout_volume_ratio=breakout_volume / data['volume'].mean() if data['volume'].mean() > 0 else 1.0,
                            breakout_momentum=None,  # Would calculate from price movement
                            breakout_retest=None,  # Would track retest behavior
                            breakout_metadata={
                                'zone_strength': zone.zone_strength,
                                'zone_touches': zone.zone_touches
                            }
                        )
                        breakouts.append(breakout)
                
                elif zone.zone_type == ZoneType.SUPPLY:
                    # Check for breakout
                    if recent_data['high'].max() > zone.zone_end_price:
                        breakout_price = recent_data['high'].max()
                        breakout_volume = recent_data[recent_data['high'] >= zone.zone_end_price]['volume'].sum()
                        
                        breakout = ZoneBreakout(
                            timestamp=datetime.now(timezone.utc),
                            symbol=symbol,
                            timeframe=timeframe,
                            zone_id=id(zone),  # Use object id as placeholder
                            breakout_type=BreakoutType.SUPPLY_BREAKOUT,
                            breakout_price=breakout_price,
                            breakout_volume=breakout_volume,
                            breakout_strength=zone.zone_breakout_strength or 0.0,
                            breakout_confidence=zone.zone_confidence,
                            breakout_volume_ratio=breakout_volume / data['volume'].mean() if data['volume'].mean() > 0 else 1.0,
                            breakout_momentum=None,  # Would calculate from price movement
                            breakout_retest=None,  # Would track retest behavior
                            breakout_metadata={
                                'zone_strength': zone.zone_strength,
                                'zone_touches': zone.zone_touches
                            }
                        )
                        breakouts.append(breakout)
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Error detecting zone breakouts: {e}")
            return []
    
    async def _track_zone_interactions(self, symbol: str, timeframe: str, data: pd.DataFrame,
                                     zones: List[DemandSupplyZone]) -> List[ZoneInteraction]:
        """Track zone interactions (touches, bounces, etc.)"""
        try:
            interactions = []
            
            if not zones or len(data) < 10:
                return interactions
            
            recent_data = data.tail(10)
            
            for zone in zones:
                # Check for recent interactions with this zone
                zone_price_range = (zone.zone_start_price, zone.zone_end_price)
                
                for _, row in recent_data.iterrows():
                    # Check if price touched the zone
                    if (zone_price_range[0] <= row['low'] <= zone_price_range[1] or
                        zone_price_range[0] <= row['high'] <= zone_price_range[1]):
                        
                        # Determine interaction type
                        if zone.zone_type == ZoneType.DEMAND:
                            if row['close'] > zone_price_range[1]:  # Bounce
                                interaction_type = InteractionType.BOUNCE
                            else:  # Touch
                                interaction_type = InteractionType.TOUCH
                        else:  # Supply zone
                            if row['close'] < zone_price_range[0]:  # Bounce
                                interaction_type = InteractionType.BOUNCE
                            else:  # Touch
                                interaction_type = InteractionType.TOUCH
                        
                        interaction = ZoneInteraction(
                            timestamp=row['timestamp'],
                            symbol=symbol,
                            timeframe=timeframe,
                            zone_id=id(zone),  # Use object id as placeholder
                            interaction_type=interaction_type,
                            interaction_price=row['close'],
                            interaction_volume=row['volume'],
                            interaction_strength=zone.zone_strength,
                            interaction_confidence=zone.zone_confidence,
                            interaction_duration_minutes=None,  # Would calculate from data
                            interaction_momentum=None,  # Would calculate from price movement
                            interaction_metadata={
                                'zone_strength': zone.zone_strength,
                                'zone_touches': zone.zone_touches
                            }
                        )
                        interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error tracking zone interactions: {e}")
            return []
    
    async def _generate_volume_profile_summary(self, volume_profile_nodes: List[VolumeProfileNode]) -> Dict[str, Any]:
        """Generate volume profile summary"""
        try:
            if not volume_profile_nodes:
                return {'error': 'No volume profile nodes available'}
            
            high_nodes = [node for node in volume_profile_nodes if node.volume_node_type == VolumeNodeType.HIGH]
            medium_nodes = [node for node in volume_profile_nodes if node.volume_node_type == VolumeNodeType.MEDIUM]
            low_nodes = [node for node in volume_profile_nodes if node.volume_node_type == VolumeNodeType.LOW]
            
            return {
                'total_nodes': len(volume_profile_nodes),
                'high_volume_nodes': len(high_nodes),
                'medium_volume_nodes': len(medium_nodes),
                'low_volume_nodes': len(low_nodes),
                'strongest_volume_level': max(volume_profile_nodes, key=lambda x: x.volume_at_level).price_level if volume_profile_nodes else None,
                'volume_distribution': {
                    'high_percentage': sum(node.volume_percentage for node in high_nodes),
                    'medium_percentage': sum(node.volume_percentage for node in medium_nodes),
                    'low_percentage': sum(node.volume_percentage for node in low_nodes)
                },
                'price_range': {
                    'min': min(node.price_level for node in volume_profile_nodes),
                    'max': max(node.price_level for node in volume_profile_nodes)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating volume profile summary: {e}")
            return {'error': str(e)}
    
    async def _generate_zone_analysis_summary(self, demand_zones: List[DemandSupplyZone], 
                                            supply_zones: List[DemandSupplyZone]) -> Dict[str, Any]:
        """Generate zone analysis summary"""
        try:
            return {
                'total_zones': len(demand_zones) + len(supply_zones),
                'demand_zones': {
                    'count': len(demand_zones),
                    'strongest_zone': max(demand_zones, key=lambda z: z.zone_strength).zone_start_price if demand_zones else None,
                    'avg_strength': np.mean([z.zone_strength for z in demand_zones]) if demand_zones else 0.0,
                    'total_touches': sum(z.zone_touches for z in demand_zones)
                },
                'supply_zones': {
                    'count': len(supply_zones),
                    'strongest_zone': max(supply_zones, key=lambda z: z.zone_strength).zone_start_price if supply_zones else None,
                    'avg_strength': np.mean([z.zone_strength for z in supply_zones]) if supply_zones else 0.0,
                    'total_touches': sum(z.zone_touches for z in supply_zones)
                },
                'zone_balance': {
                    'demand_dominance': len(demand_zones) > len(supply_zones),
                    'strength_ratio': (np.mean([z.zone_strength for z in demand_zones]) if demand_zones else 0.0) / 
                                    (np.mean([z.zone_strength for z in supply_zones]) if supply_zones else 1.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating zone analysis summary: {e}")
            return {'error': str(e)}
    
    async def _calculate_analysis_confidence(self, demand_zones: List[DemandSupplyZone],
                                           supply_zones: List[DemandSupplyZone],
                                           volume_profile_nodes: List[VolumeProfileNode],
                                           zone_breakouts: List[ZoneBreakout]) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence_scores = []
            
            # Zone detection confidence
            total_zones = len(demand_zones) + len(supply_zones)
            if total_zones > 0:
                avg_zone_confidence = np.mean([z.zone_confidence for z in demand_zones + supply_zones])
                confidence_scores.append(avg_zone_confidence)
            
            # Volume profile confidence
            if volume_profile_nodes:
                high_volume_nodes = len([n for n in volume_profile_nodes if n.volume_node_type == VolumeNodeType.HIGH])
                volume_confidence = min(1.0, high_volume_nodes / 5)  # More high volume nodes = higher confidence
                confidence_scores.append(volume_confidence)
            
            # Breakout detection confidence
            if zone_breakouts:
                avg_breakout_confidence = np.mean([b.breakout_confidence for b in zone_breakouts])
                confidence_scores.append(avg_breakout_confidence)
            
            return np.mean(confidence_scores) if confidence_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    async def _calculate_market_context(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market context for demand/supply analysis"""
        try:
            return {
                'symbol': symbol,
                'data_points': len(data),
                'time_range': {
                    'start': data['timestamp'].min() if len(data) > 0 else None,
                    'end': data['timestamp'].max() if len(data) > 0 else None
                },
                'price_metrics': {
                    'price_range': float(data['high'].max() - data['low'].min()) if len(data) > 0 else 0,
                    'price_volatility': float(data['close'].std()) if len(data) > 0 else 0,
                    'avg_price': float(data['close'].mean()) if len(data) > 0 else 0
                },
                'volume_metrics': {
                    'total_volume': float(data['volume'].sum()) if len(data) > 0 else 0,
                    'avg_volume': float(data['volume'].mean()) if len(data) > 0 else 0,
                    'volume_volatility': float(data['volume'].std()) if len(data) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating market context: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def _create_default_analysis(self, symbol: str, timeframe: str) -> DemandSupplyAnalysis:
        """Create default analysis when insufficient data"""
        return DemandSupplyAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            demand_zones=[],
            supply_zones=[],
            volume_profile_nodes=[],
            zone_breakouts=[],
            zone_interactions=[],
            strongest_demand_zone=None,
            strongest_supply_zone=None,
            volume_profile_summary={'error': 'Insufficient data'},
            zone_analysis_summary={'error': 'Insufficient data'},
            analysis_confidence=0.0,
            market_context={'symbol': symbol, 'insufficient_data': True},
            analysis_metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def close(self):
        """Close analyzer and cleanup"""
        try:
            logger.info("Demand Supply Zone Analyzer closed")
        except Exception as e:
            logger.error(f"Error closing analyzer: {e}")
