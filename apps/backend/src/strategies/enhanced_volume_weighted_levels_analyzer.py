#!/usr/bin/env python3
"""
Enhanced Volume-Weighted Levels Analyzer for AlphaPlus
Advanced HVN/LVN detection with POC calculation and volume profile analysis
"""

import asyncio
import logging
import asyncpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import math

logger = logging.getLogger(__name__)

class VolumeNodeType(Enum):
    """Volume node types"""
    HIGH_VOLUME_NODE = "high_volume_node"
    LOW_VOLUME_NODE = "low_volume_node"
    POINT_OF_CONTROL = "point_of_control"
    VALUE_AREA_HIGH = "value_area_high"
    VALUE_AREA_LOW = "value_area_low"
    VOLUME_GAP = "volume_gap"
    VOLUME_CLIMAX = "volume_climax"

@dataclass
class VolumeNode:
    """Volume node with detailed analysis"""
    node_type: VolumeNodeType
    price_level: float
    volume_at_level: float
    volume_percentage: float
    node_strength: float
    confidence: float
    timestamp: datetime
    is_active: bool = True
    touch_count: int = 0
    first_touch_time: Optional[datetime] = None
    last_touch_time: Optional[datetime] = None
    volume_trend: str = "stable"  # "increasing", "decreasing", "stable"
    price_efficiency: float = 0.0
    institutional_activity: bool = False
    
    @property
    def price(self) -> float:
        """Alias for price_level for compatibility"""
        return self.price_level
    
    @property
    def volume(self) -> float:
        """Alias for volume_at_level for compatibility"""
        return self.volume_at_level
    
    @property
    def strength(self) -> float:
        """Alias for node_strength for compatibility"""
        return self.node_strength

@dataclass
class VolumeProfileAnalysis:
    """Comprehensive volume profile analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    poc_price: float
    poc_volume: float
    value_area_high: float
    value_area_low: float
    value_area_volume: float
    total_volume: float
    high_volume_nodes: List[VolumeNode]
    low_volume_nodes: List[VolumeNode]
    volume_gaps: List[VolumeNode]
    volume_distribution: Dict[float, float]
    analysis_confidence: float
    algorithm_inputs: Dict[str, Any]

class EnhancedVolumeWeightedLevelsAnalyzer:
    """Enhanced volume-weighted levels analyzer with HVN/LVN detection"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.logger = logger
        self.db_pool = None
        
        # Configuration
        self.config = {
            'lookback_periods': 200,
            'volume_profile_periods': 100,
            'poc_threshold': 0.7,  # 70% of max volume
            'value_area_percentage': 0.68,  # 68% of volume
            'hvn_threshold': 1.5,  # 1.5x average volume
            'lvn_threshold': 0.5,  # 0.5x average volume
            'volume_gap_threshold': 0.1,  # 10% volume drop
            'institutional_threshold': 5.0,  # 5x average volume
            'min_node_strength': 0.3,
            'price_efficiency_threshold': 0.8
        }
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'volume_profiles_created': 0,
            'hvn_detected': 0,
            'lvn_detected': 0,
            'poc_calculated': 0,
            'last_analysis': None
        }
        
        logger.info("🔧 Enhanced Volume-Weighted Levels Analyzer initialized")
    
    async def initialize(self):
        """Initialize database connection pool"""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(self.db_url)
            self.logger.info("✅ Database connection pool created for Volume-Weighted Levels Analyzer")
    
    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            self.logger.info("🔌 Database connection pool closed for Volume-Weighted Levels Analyzer")
    
    async def analyze_volume_weighted_levels(self, symbol: str, timeframe: str) -> VolumeProfileAnalysis:
        """Analyze volume-weighted levels with HVN/LVN detection"""
        try:
            self.logger.info(f"📊 Analyzing volume-weighted levels for {symbol} {timeframe}")
            
            # Get recent OHLCV data
            ohlcv_data = await self._get_recent_ohlcv_data(symbol, timeframe)
            if not ohlcv_data or len(ohlcv_data) < 50:
                self.logger.warning(f"Insufficient data for volume analysis: {symbol}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert decimal columns to float to avoid type conflicts
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Create volume profile
            volume_profile = await self._create_volume_profile(df, symbol, timeframe)
            
            # Detect HVN and LVN
            hvn_nodes = await self._detect_high_volume_nodes(df, volume_profile)
            lvn_nodes = await self._detect_low_volume_nodes(df, volume_profile)
            
            # Detect volume gaps
            volume_gaps = await self._detect_volume_gaps(df, volume_profile)
            
            # Calculate Point of Control (POC)
            poc_node = await self._calculate_point_of_control(df, volume_profile)
            
            # Calculate Value Area
            value_area_nodes = await self._calculate_value_area(df, volume_profile)
            
            # Analyze institutional activity
            institutional_nodes = await self._detect_institutional_activity(df, volume_profile)
            
            # Combine all nodes
            all_nodes = hvn_nodes + lvn_nodes + volume_gaps + [poc_node] + value_area_nodes + institutional_nodes
            
            # Validate and score nodes
            validated_nodes = await self._validate_and_score_nodes(df, all_nodes)
            
            # Calculate analysis confidence
            analysis_confidence = await self._calculate_analysis_confidence(df, validated_nodes)
            
            # Prepare algorithm inputs
            algorithm_inputs = await self._prepare_algorithm_inputs(validated_nodes, volume_profile)
            
            # Create analysis
            analysis = VolumeProfileAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(timezone.utc),
                poc_price=poc_node.price_level,
                poc_volume=poc_node.volume_at_level,
                value_area_high=value_area_nodes[0].price_level if len(value_area_nodes) > 0 else poc_node.price_level,
                value_area_low=value_area_nodes[1].price_level if len(value_area_nodes) > 1 else poc_node.price_level,
                value_area_volume=sum(node.volume_at_level for node in value_area_nodes),
                total_volume=volume_profile['total_volume'],
                high_volume_nodes=[node for node in validated_nodes if node.node_type == VolumeNodeType.HIGH_VOLUME_NODE],
                low_volume_nodes=[node for node in validated_nodes if node.node_type == VolumeNodeType.LOW_VOLUME_NODE],
                volume_gaps=[node for node in validated_nodes if node.node_type == VolumeNodeType.VOLUME_GAP],
                volume_distribution=volume_profile['volume_distribution'],
                analysis_confidence=analysis_confidence,
                algorithm_inputs=algorithm_inputs
            )
            
            # Store analysis in database
            await self._store_volume_profile_analysis(analysis)
            
            # Update statistics
            self.stats['total_analyses'] += 1
            self.stats['successful_analyses'] += 1
            self.stats['volume_profiles_created'] += 1
            self.stats['hvn_detected'] += len(hvn_nodes)
            self.stats['lvn_detected'] += len(lvn_nodes)
            self.stats['poc_calculated'] += 1
            self.stats['last_analysis'] = datetime.now()
            
            self.logger.info(f"✅ Volume-weighted analysis completed for {symbol} {timeframe}")
            self.logger.info(f"📊 Detected {len(hvn_nodes)} HVN, {len(lvn_nodes)} LVN, POC at {poc_node.price_level}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing volume-weighted levels for {symbol} {timeframe}: {e}")
            self.stats['total_analyses'] += 1
            self.stats['failed_analyses'] += 1
            return self._get_default_analysis(symbol, timeframe)
    
    async def _get_recent_ohlcv_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get recent OHLCV data"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY timestamp DESC
                    LIMIT $3
                """
                
                rows = await conn.fetch(query, symbol, timeframe, self.config['lookback_periods'])
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting OHLCV data: {e}")
            return []
    
    async def _create_volume_profile(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Create detailed volume profile from OHLCV data"""
        try:
            # Calculate volume-weighted price levels
            price_levels = []
            volume_at_levels = []
            
            for _, row in df.iterrows():
                # Create price levels within the candle range
                price_range = np.linspace(row['low'], row['high'], 20)  # More granular
                volume_per_level = row['volume'] / 20
                
                for price in price_range:
                    price_levels.append(round(price, 2))
                    volume_at_levels.append(volume_per_level)
            
            # Group by price levels and sum volumes
            volume_distribution = {}
            for price, volume in zip(price_levels, volume_at_levels):
                if price in volume_distribution:
                    volume_distribution[price] += volume
                else:
                    volume_distribution[price] = volume
            
            # Calculate total volume
            total_volume = sum(volume_distribution.values())
            
            # Calculate average volume per level
            avg_volume_per_level = total_volume / len(volume_distribution) if volume_distribution else 0
            
            return {
                'volume_distribution': volume_distribution,
                'total_volume': total_volume,
                'avg_volume_per_level': avg_volume_per_level,
                'price_range': (df['low'].min(), df['high'].max()),
                'volume_range': (df['volume'].min(), df['volume'].max())
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating volume profile: {e}")
            return {
                'volume_distribution': {},
                'total_volume': 0,
                'avg_volume_per_level': 0,
                'price_range': (0, 0),
                'volume_range': (0, 0)
            }
    
    async def _detect_high_volume_nodes(self, df: pd.DataFrame, volume_profile: Dict[str, Any]) -> List[VolumeNode]:
        """Detect High Volume Nodes (HVN)"""
        try:
            hvn_nodes = []
            volume_distribution = volume_profile['volume_distribution']
            avg_volume = volume_profile['avg_volume_per_level']
            hvn_threshold = avg_volume * self.config['hvn_threshold']
            
            for price_level, volume_at_level in volume_distribution.items():
                if volume_at_level >= hvn_threshold:
                    # Calculate node strength
                    node_strength = min(volume_at_level / avg_volume, 3.0) / 3.0
                    
                    # Calculate confidence
                    confidence = await self._calculate_node_confidence(df, price_level, volume_at_level)
                    
                    # Analyze volume trend
                    volume_trend = await self._analyze_volume_trend(df, price_level)
                    
                    # Calculate price efficiency
                    price_efficiency = await self._calculate_price_efficiency(df, price_level)
                    
                    # Check for institutional activity
                    institutional_activity = volume_at_level >= avg_volume * self.config['institutional_threshold']
                    
                    node = VolumeNode(
                        node_type=VolumeNodeType.HIGH_VOLUME_NODE,
                        price_level=price_level,
                        volume_at_level=volume_at_level,
                        volume_percentage=(volume_at_level / volume_profile['total_volume']) * 100,
                        node_strength=node_strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        volume_trend=volume_trend,
                        price_efficiency=price_efficiency,
                        institutional_activity=institutional_activity
                    )
                    hvn_nodes.append(node)
            
            return hvn_nodes
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting HVN: {e}")
            return []
    
    async def _detect_low_volume_nodes(self, df: pd.DataFrame, volume_profile: Dict[str, Any]) -> List[VolumeNode]:
        """Detect Low Volume Nodes (LVN)"""
        try:
            lvn_nodes = []
            volume_distribution = volume_profile['volume_distribution']
            avg_volume = volume_profile['avg_volume_per_level']
            lvn_threshold = avg_volume * self.config['lvn_threshold']
            
            for price_level, volume_at_level in volume_distribution.items():
                if volume_at_level <= lvn_threshold:
                    # Calculate node strength (inverse for LVN)
                    node_strength = max(0, 1 - (volume_at_level / avg_volume))
                    
                    # Calculate confidence
                    confidence = await self._calculate_node_confidence(df, price_level, volume_at_level)
                    
                    # Analyze volume trend
                    volume_trend = await self._analyze_volume_trend(df, price_level)
                    
                    # Calculate price efficiency
                    price_efficiency = await self._calculate_price_efficiency(df, price_level)
                    
                    node = VolumeNode(
                        node_type=VolumeNodeType.LOW_VOLUME_NODE,
                        price_level=price_level,
                        volume_at_level=volume_at_level,
                        volume_percentage=(volume_at_level / volume_profile['total_volume']) * 100,
                        node_strength=node_strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        volume_trend=volume_trend,
                        price_efficiency=price_efficiency,
                        institutional_activity=False
                    )
                    lvn_nodes.append(node)
            
            return lvn_nodes
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting LVN: {e}")
            return []
    
    async def _detect_volume_gaps(self, df: pd.DataFrame, volume_profile: Dict[str, Any]) -> List[VolumeNode]:
        """Detect volume gaps in the profile"""
        try:
            volume_gaps = []
            volume_distribution = volume_profile['volume_distribution']
            sorted_prices = sorted(volume_distribution.keys())
            
            for i in range(len(sorted_prices) - 1):
                current_price = sorted_prices[i]
                next_price = sorted_prices[i + 1]
                current_volume = volume_distribution[current_price]
                next_volume = volume_distribution[next_price]
                
                # Check for significant volume drop
                if current_volume > 0 and next_volume > 0:
                    volume_ratio = next_volume / current_volume
                    
                    if volume_ratio <= self.config['volume_gap_threshold']:
                        gap_price = (current_price + next_price) / 2
                        
                        node = VolumeNode(
                            node_type=VolumeNodeType.VOLUME_GAP,
                            price_level=gap_price,
                            volume_at_level=min(current_volume, next_volume),
                            volume_percentage=(min(current_volume, next_volume) / volume_profile['total_volume']) * 100,
                            node_strength=1 - volume_ratio,
                            confidence=0.7,
                            timestamp=datetime.now(timezone.utc),
                            volume_trend="decreasing",
                            price_efficiency=0.5
                        )
                        volume_gaps.append(node)
            
            return volume_gaps
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting volume gaps: {e}")
            return []
    
    async def _calculate_point_of_control(self, df: pd.DataFrame, volume_profile: Dict[str, Any]) -> VolumeNode:
        """Calculate Point of Control (POC)"""
        try:
            volume_distribution = volume_profile['volume_distribution']
            
            if not volume_distribution:
                return VolumeNode(
                    node_type=VolumeNodeType.POINT_OF_CONTROL,
                    price_level=0.0,
                    volume_at_level=0.0,
                    volume_percentage=0.0,
                    node_strength=0.0,
                    confidence=0.0,
                    timestamp=datetime.now(timezone.utc)
                )
            
            # Find price level with maximum volume
            poc_price = max(volume_distribution.keys(), key=lambda k: volume_distribution[k])
            poc_volume = volume_distribution[poc_price]
            
            # Calculate POC strength
            total_volume = volume_profile['total_volume']
            poc_strength = poc_volume / total_volume
            
            # Calculate confidence
            confidence = await self._calculate_node_confidence(df, poc_price, poc_volume)
            
            return VolumeNode(
                node_type=VolumeNodeType.POINT_OF_CONTROL,
                price_level=poc_price,
                volume_at_level=poc_volume,
                volume_percentage=(poc_volume / total_volume) * 100,
                node_strength=poc_strength,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                volume_trend="stable",
                price_efficiency=1.0,
                institutional_activity=poc_volume >= volume_profile['avg_volume_per_level'] * self.config['institutional_threshold']
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating POC: {e}")
            return VolumeNode(
                node_type=VolumeNodeType.POINT_OF_CONTROL,
                price_level=0.0,
                volume_at_level=0.0,
                volume_percentage=0.0,
                node_strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _calculate_value_area(self, df: pd.DataFrame, volume_profile: Dict[str, Any]) -> List[VolumeNode]:
        """Calculate Value Area High and Low"""
        try:
            volume_distribution = volume_profile['volume_distribution']
            total_volume = volume_profile['total_volume']
            target_volume = total_volume * self.config['value_area_percentage']
            
            # Sort levels by volume (descending)
            sorted_levels = sorted(volume_distribution.items(), key=lambda x: x[1], reverse=True)
            
            # Find levels that make up the value area
            value_area_volume = 0
            value_area_levels = []
            
            for price, volume in sorted_levels:
                if value_area_volume < target_volume:
                    value_area_levels.append((price, volume))
                    value_area_volume += volume
                else:
                    break
            
            if not value_area_levels:
                return []
            
            # Find high and low of value area
            value_area_high = max(level[0] for level in value_area_levels)
            value_area_low = min(level[0] for level in value_area_levels)
            
            nodes = []
            
            # Create Value Area High node
            vah_volume = volume_distribution.get(value_area_high, 0)
            vah_node = VolumeNode(
                node_type=VolumeNodeType.VALUE_AREA_HIGH,
                price_level=value_area_high,
                volume_at_level=vah_volume,
                volume_percentage=(vah_volume / total_volume) * 100,
                node_strength=0.8,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                volume_trend="stable",
                price_efficiency=0.8
            )
            nodes.append(vah_node)
            
            # Create Value Area Low node
            val_volume = volume_distribution.get(value_area_low, 0)
            val_node = VolumeNode(
                node_type=VolumeNodeType.VALUE_AREA_LOW,
                price_level=value_area_low,
                volume_at_level=val_volume,
                volume_percentage=(val_volume / total_volume) * 100,
                node_strength=0.8,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                volume_trend="stable",
                price_efficiency=0.8
            )
            nodes.append(val_node)
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating value area: {e}")
            return []
    
    async def _detect_institutional_activity(self, df: pd.DataFrame, volume_profile: Dict[str, Any]) -> List[VolumeNode]:
        """Detect institutional activity nodes"""
        try:
            institutional_nodes = []
            volume_distribution = volume_profile['volume_distribution']
            avg_volume = volume_profile['avg_volume_per_level']
            institutional_threshold = avg_volume * self.config['institutional_threshold']
            
            for price_level, volume_at_level in volume_distribution.items():
                if volume_at_level >= institutional_threshold:
                    # Calculate institutional strength
                    institutional_strength = min(volume_at_level / institutional_threshold, 2.0) / 2.0
                    
                    # Calculate confidence
                    confidence = await self._calculate_node_confidence(df, price_level, volume_at_level)
                    
                    node = VolumeNode(
                        node_type=VolumeNodeType.HIGH_VOLUME_NODE,  # Use HVN type for institutional
                        price_level=price_level,
                        volume_at_level=volume_at_level,
                        volume_percentage=(volume_at_level / volume_profile['total_volume']) * 100,
                        node_strength=institutional_strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        volume_trend="increasing",
                        price_efficiency=0.9,
                        institutional_activity=True
                    )
                    institutional_nodes.append(node)
            
            return institutional_nodes
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting institutional activity: {e}")
            return []
    
    async def _calculate_node_confidence(self, df: pd.DataFrame, price_level: float, volume_at_level: float) -> float:
        """Calculate confidence in a volume node"""
        try:
            # Base confidence from volume
            avg_volume = df['volume'].mean()
            volume_confidence = min(volume_at_level / avg_volume, 2.0) / 2.0
            
            # Data quality factor
            data_quality = min(len(df) / self.config['lookback_periods'], 1.0)
            
            # Price level consistency
            tolerance = price_level * 0.001  # 0.1% tolerance
            touches = 0
            
            for _, row in df.iterrows():
                if (abs(row['high'] - price_level) <= tolerance or 
                    abs(row['low'] - price_level) <= tolerance):
                    touches += 1
            
            consistency = min(touches / 10.0, 1.0)
            
            # Combine factors
            total_confidence = (volume_confidence * 0.4 + data_quality * 0.3 + consistency * 0.3)
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating node confidence: {e}")
            return 0.0
    
    async def _analyze_volume_trend(self, df: pd.DataFrame, price_level: float) -> str:
        """Analyze volume trend at a price level"""
        try:
            tolerance = price_level * 0.001
            volumes_at_level = []
            
            for _, row in df.iterrows():
                if (abs(row['high'] - price_level) <= tolerance or 
                    abs(row['low'] - price_level) <= tolerance):
                    volumes_at_level.append(row['volume'])
            
            if len(volumes_at_level) < 3:
                return "stable"
            
            # Calculate trend
            recent_volumes = volumes_at_level[-3:]
            if len(recent_volumes) >= 2:
                if recent_volumes[-1] > recent_volumes[-2]:
                    return "increasing"
                elif recent_volumes[-1] < recent_volumes[-2]:
                    return "decreasing"
            
            return "stable"
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing volume trend: {e}")
            return "stable"
    
    async def _calculate_price_efficiency(self, df: pd.DataFrame, price_level: float) -> float:
        """Calculate price efficiency at a level"""
        try:
            tolerance = price_level * 0.001
            price_movements = []
            
            for _, row in df.iterrows():
                if (abs(row['high'] - price_level) <= tolerance or 
                    abs(row['low'] - price_level) <= tolerance):
                    # Calculate price movement efficiency
                    candle_range = row['high'] - row['low']
                    if candle_range > 0:
                        efficiency = abs(row['close'] - row['open']) / candle_range
                        price_movements.append(efficiency)
            
            if not price_movements:
                return 0.0
            
            return np.mean(price_movements)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating price efficiency: {e}")
            return 0.0
    
    async def _validate_and_score_nodes(self, df: pd.DataFrame, nodes: List[VolumeNode]) -> List[VolumeNode]:
        """Validate and score volume nodes"""
        try:
            validated_nodes = []
            
            for node in nodes:
                # Filter by minimum strength
                if node.node_strength >= self.config['min_node_strength']:
                    # Update touch count
                    node.touch_count = await self._count_node_touches(df, node.price_level)
                    
                    # Update timestamps
                    node.first_touch_time = await self._get_first_touch_time(df, node.price_level)
                    node.last_touch_time = await self._get_last_touch_time(df, node.price_level)
                    
                    validated_nodes.append(node)
            
            # Sort by strength and confidence
            validated_nodes.sort(key=lambda x: (x.node_strength, x.confidence), reverse=True)
            
            return validated_nodes
            
        except Exception as e:
            self.logger.error(f"❌ Error validating nodes: {e}")
            return []
    
    async def _count_node_touches(self, df: pd.DataFrame, price_level: float) -> int:
        """Count touches at a price level"""
        try:
            tolerance = price_level * 0.001
            touch_count = 0
            
            for _, row in df.iterrows():
                if (abs(row['high'] - price_level) <= tolerance or 
                    abs(row['low'] - price_level) <= tolerance):
                    touch_count += 1
            
            return touch_count
            
        except Exception as e:
            self.logger.error(f"❌ Error counting node touches: {e}")
            return 0
    
    async def _get_first_touch_time(self, df: pd.DataFrame, price_level: float) -> Optional[datetime]:
        """Get first touch time for a price level"""
        try:
            tolerance = price_level * 0.001
            
            for _, row in df.iterrows():
                if (abs(row['high'] - price_level) <= tolerance or 
                    abs(row['low'] - price_level) <= tolerance):
                    return row['timestamp']
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting first touch time: {e}")
            return None
    
    async def _get_last_touch_time(self, df: pd.DataFrame, price_level: float) -> Optional[datetime]:
        """Get last touch time for a price level"""
        try:
            tolerance = price_level * 0.001
            last_touch = None
            
            for _, row in df.iterrows():
                if (abs(row['high'] - price_level) <= tolerance or 
                    abs(row['low'] - price_level) <= tolerance):
                    last_touch = row['timestamp']
            
            return last_touch
            
        except Exception as e:
            self.logger.error(f"❌ Error getting last touch time: {e}")
            return None
    
    async def _calculate_analysis_confidence(self, df: pd.DataFrame, nodes: List[VolumeNode]) -> float:
        """Calculate overall analysis confidence"""
        try:
            if not nodes:
                return 0.0
            
            # Average confidence of all nodes
            avg_node_confidence = np.mean([node.confidence for node in nodes])
            
            # Data quality factor
            data_quality = min(len(df) / self.config['lookback_periods'], 1.0)
            
            # Node diversity factor
            node_types = set(node.node_type for node in nodes)
            diversity_factor = len(node_types) / 7.0  # 7 different node types
            
            # Combine factors
            total_confidence = (avg_node_confidence * 0.5 + data_quality * 0.3 + diversity_factor * 0.2)
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating analysis confidence: {e}")
            return 0.0
    
    async def _prepare_algorithm_inputs(self, nodes: List[VolumeNode], volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for other algorithms"""
        try:
            # Group nodes by type
            node_groups = {}
            for node in nodes:
                node_type = node.node_type.value
                if node_type not in node_groups:
                    node_groups[node_type] = []
                
                node_groups[node_type].append({
                    'price': node.price_level,
                    'volume': node.volume_at_level,
                    'strength': node.node_strength,
                    'confidence': node.confidence,
                    'volume_percentage': node.volume_percentage,
                    'institutional_activity': node.institutional_activity,
                    'price_efficiency': node.price_efficiency
                })
            
            # Find POC
            poc_node = next((node for node in nodes if node.node_type == VolumeNodeType.POINT_OF_CONTROL), None)
            
            # Find Value Area
            vah_nodes = [node for node in nodes if node.node_type == VolumeNodeType.VALUE_AREA_HIGH]
            val_nodes = [node for node in nodes if node.node_type == VolumeNodeType.VALUE_AREA_LOW]
            
            return {
                'volume_profile': {
                    'poc_price': poc_node.price_level if poc_node else 0.0,
                    'poc_volume': poc_node.volume_at_level if poc_node else 0.0,
                    'value_area_high': vah_nodes[0].price_level if vah_nodes else 0.0,
                    'value_area_low': val_nodes[0].price_level if val_nodes else 0.0,
                    'total_volume': volume_profile['total_volume']
                },
                'volume_nodes': node_groups,
                'high_volume_nodes': node_groups.get('high_volume_node', []),
                'low_volume_nodes': node_groups.get('low_volume_node', []),
                'volume_gaps': node_groups.get('volume_gap', []),
                'institutional_nodes': [node for node in nodes if node.institutional_activity],
                'total_nodes': len(nodes),
                'active_nodes': len([node for node in nodes if node.is_active])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing algorithm inputs: {e}")
            return {}
    
    async def _store_volume_profile_analysis(self, analysis: VolumeProfileAnalysis):
        """Store volume profile analysis in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store main analysis
                query = """
                    INSERT INTO volume_profile_analysis (
                        symbol, timeframe, timestamp, poc_price, poc_volume,
                        value_area_high, value_area_low, value_area_volume,
                        total_volume, analysis_confidence, algorithm_inputs
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                        poc_price = EXCLUDED.poc_price,
                        poc_volume = EXCLUDED.poc_volume,
                        value_area_high = EXCLUDED.value_area_high,
                        value_area_low = EXCLUDED.value_area_low,
                        value_area_volume = EXCLUDED.value_area_volume,
                        total_volume = EXCLUDED.total_volume,
                        analysis_confidence = EXCLUDED.analysis_confidence,
                        algorithm_inputs = EXCLUDED.algorithm_inputs
                """
                
                # Convert VolumeNode objects to serializable format
                serializable_inputs = {
                    'hvn_count': len(analysis.high_volume_nodes),
                    'lvn_count': len(analysis.low_volume_nodes),
                    'volume_gaps_count': len(analysis.volume_gaps),
                    'poc_price': analysis.poc_price,
                    'value_area_high': analysis.value_area_high,
                    'value_area_low': analysis.value_area_low,
                    'analysis_confidence': analysis.analysis_confidence
                }
                
                await conn.execute(
                    query,
                    analysis.symbol,
                    analysis.timeframe,
                    analysis.timestamp,
                    analysis.poc_price,
                    analysis.poc_volume,
                    analysis.value_area_high,
                    analysis.value_area_low,
                    analysis.value_area_volume,
                    analysis.total_volume,
                    analysis.analysis_confidence,
                    json.dumps(serializable_inputs)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error storing volume profile analysis: {e}")
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> VolumeProfileAnalysis:
        """Get default analysis when data is insufficient"""
        return VolumeProfileAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            poc_price=0.0,
            poc_volume=0.0,
            value_area_high=0.0,
            value_area_low=0.0,
            value_area_volume=0.0,
            total_volume=0.0,
            high_volume_nodes=[],
            low_volume_nodes=[],
            volume_gaps=[],
            volume_distribution={},
            analysis_confidence=0.0,
            algorithm_inputs={}
        )

# Example usage
async def main():
    """Test the enhanced volume-weighted levels analyzer"""
    analyzer = EnhancedVolumeWeightedLevelsAnalyzer()
    
    try:
        await analyzer.initialize()
        
        # Analyze volume-weighted levels
        analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
        
        print(f"\n📊 Volume-Weighted Analysis for {analysis.symbol}:")
        print(f"  POC Price: {analysis.poc_price}")
        print(f"  POC Volume: {analysis.poc_volume}")
        print(f"  Value Area: {analysis.value_area_low} - {analysis.value_area_high}")
        print(f"  Total Volume: {analysis.total_volume}")
        print(f"  Analysis Confidence: {analysis.analysis_confidence:.3f}")
        print(f"  HVN Count: {len(analysis.high_volume_nodes)}")
        print(f"  LVN Count: {len(analysis.low_volume_nodes)}")
        print(f"  Volume Gaps: {len(analysis.volume_gaps)}")
        
        print(f"\n🔍 High Volume Nodes:")
        for hvn in analysis.high_volume_nodes[:5]:  # Show top 5
            print(f"  Price: {hvn.price_level}, Volume: {hvn.volume_at_level:.2f}, Strength: {hvn.node_strength:.3f}")
        
        print(f"\n🔍 Low Volume Nodes:")
        for lvn in analysis.low_volume_nodes[:5]:  # Show top 5
            print(f"  Price: {lvn.price_level}, Volume: {lvn.volume_at_level:.2f}, Strength: {lvn.node_strength:.3f}")
        
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
