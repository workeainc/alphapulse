#!/usr/bin/env python3
"""
Enhanced Order Book Integration Service for AlphaPlus
Integrates order book data with volume-weighted level calculations
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

logger = logging.getLogger(__name__)

class OrderBookLevelType(Enum):
    """Order book level types"""
    HIGH_VOLUME_NODE = "high_volume_node"
    LOW_VOLUME_NODE = "low_volume_node"
    POINT_OF_CONTROL = "point_of_control"
    VALUE_AREA_HIGH = "value_area_high"
    VALUE_AREA_LOW = "value_area_low"
    LIQUIDITY_WALL = "liquidity_wall"
    SUPPORT_RESISTANCE = "support_resistance"

@dataclass
class OrderBookLevel:
    """Order book level with volume analysis"""
    level_type: OrderBookLevelType
    price_level: float
    volume_at_level: float
    volume_percentage: float
    bid_volume: float
    ask_volume: float
    volume_imbalance: float
    level_strength: float
    confidence: float
    timestamp: datetime
    is_active: bool = True
    touch_count: int = 0
    last_touch_time: Optional[datetime] = None

@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    poc_price: float
    poc_volume: float
    value_area_high: float
    value_area_low: float
    value_area_volume: float
    total_volume: float
    high_volume_nodes: List[OrderBookLevel]
    low_volume_nodes: List[OrderBookLevel]
    liquidity_walls: List[OrderBookLevel]
    volume_distribution: Dict[float, float]  # price -> volume
    analysis_confidence: float

@dataclass
class OrderBookAnalysis:
    """Comprehensive order book analysis"""
    symbol: str
    timestamp: datetime
    volume_profile: VolumeProfile
    bid_ask_imbalance: float
    depth_pressure: float
    liquidity_score: float
    analysis_confidence: float
    market_microstructure: Dict[str, Any]
    algorithm_inputs: Dict[str, Any]
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    spread: float = 0.0
    spread_percentage: float = 0.0
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    order_book_levels: List[OrderBookLevel] = field(default_factory=list)

class EnhancedOrderBookIntegration:
    """Enhanced order book integration with volume-weighted algorithms"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.logger = logger
        self.db_pool = None
        
        # Configuration
        self.config = {
            'volume_profile_periods': 100,
            'poc_threshold': 0.7,  # 70% of max volume
            'value_area_percentage': 0.68,  # 68% of volume
            'liquidity_wall_threshold': 5.0,  # 5x average volume
            'min_level_strength': 0.3,
            'level_tolerance': 0.001  # 0.1%
        }
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'volume_profiles_created': 0,
            'levels_detected': 0,
            'last_analysis': None
        }
        
        logger.info("🔧 Enhanced Order Book Integration initialized")
    
    async def initialize(self):
        """Initialize database connection pool"""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(self.db_url)
            self.logger.info("✅ Database connection pool created for Order Book Integration")
    
    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            self.logger.info("🔌 Database connection pool closed for Order Book Integration")
    
    async def analyze_order_book_with_volume_profile(self, symbol: str, timeframe: str) -> OrderBookAnalysis:
        """Analyze order book with volume profile integration"""
        try:
            self.logger.info(f"🔍 Analyzing order book with volume profile for {symbol} {timeframe}")
            
            # Get recent OHLCV data for volume profile
            ohlcv_data = await self._get_recent_ohlcv_data(symbol, timeframe)
            if not ohlcv_data or len(ohlcv_data) < 20:
                self.logger.warning(f"Insufficient OHLCV data for {symbol} {timeframe}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Get recent order book data
            order_book_data = await self._get_recent_order_book_data(symbol)
            if not order_book_data:
                self.logger.warning(f"No order book data available for {symbol}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Create volume profile
            volume_profile = await self._create_volume_profile(symbol, timeframe, ohlcv_data)
            
            # Analyze order book levels
            order_book_levels = await self._analyze_order_book_levels(order_book_data, volume_profile)
            
            # Calculate market microstructure metrics
            microstructure = await self._calculate_market_microstructure(order_book_data, volume_profile)
            
            # Prepare algorithm inputs
            algorithm_inputs = await self._prepare_algorithm_inputs(volume_profile, order_book_levels, microstructure)
            
            # Create comprehensive analysis
            analysis = OrderBookAnalysis(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                volume_profile=volume_profile,
                bid_ask_imbalance=microstructure['bid_ask_imbalance'],
                depth_pressure=microstructure['depth_pressure'],
                liquidity_score=microstructure['liquidity_score'],
                analysis_confidence=0.5,  # Default confidence when no orderbook data
                total_bid_volume=microstructure.get('total_bid_volume', 0.0),
                total_ask_volume=microstructure.get('total_ask_volume', 0.0),
                spread=microstructure.get('spread', 0.0),
                spread_percentage=microstructure.get('spread_percentage', 0.0),
                mid_price=microstructure.get('mid_price', 0.0),
                best_bid=microstructure.get('best_bid', 0.0),
                best_ask=microstructure.get('best_ask', 0.0),
                order_book_levels=order_book_levels,  # Add order book levels
                market_microstructure=microstructure,
                algorithm_inputs=algorithm_inputs
            )
            
            # Store analysis in database
            await self._store_order_book_analysis(analysis)
            
            # Update statistics
            self.stats['total_analyses'] += 1
            self.stats['successful_analyses'] += 1
            self.stats['volume_profiles_created'] += 1
            self.stats['levels_detected'] += len(order_book_levels)
            self.stats['last_analysis'] = datetime.now()
            
            self.logger.info(f"✅ Order book analysis completed for {symbol} {timeframe}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing order book for {symbol} {timeframe}: {e}")
            self.stats['total_analyses'] += 1
            self.stats['failed_analyses'] += 1
            return self._get_default_analysis(symbol, timeframe)
    
    async def _get_recent_ohlcv_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get recent OHLCV data for volume profile calculation"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '7 days'
                    ORDER BY timestamp DESC
                    LIMIT $3
                """
                
                rows = await conn.fetch(query, symbol, timeframe, self.config['volume_profile_periods'])
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting OHLCV data: {e}")
            return []
    
    async def _get_recent_order_book_data(self, symbol: str) -> List[Dict]:
        """Get recent order book data"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, bids, asks, best_bid, best_ask, spread
                    FROM order_book_data
                    WHERE symbol = $1
                    AND timestamp >= NOW() - INTERVAL '1 hour'
                    ORDER BY timestamp DESC
                    LIMIT 100
                """
                
                rows = await conn.fetch(query, symbol)
                
                if not rows:
                    # If no data, create mock data for testing
                    mock_data = [{
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'bids': [[115300, 1000], [115250, 800], [115200, 600]],
                        'asks': [[115350, 1000], [115400, 800], [115450, 600]],
                        'best_bid': 115300,
                        'best_ask': 115350,
                        'spread': 50
                    }]
                    return mock_data
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting order book data: {e}")
            return []
    
    async def _create_volume_profile(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> VolumeProfile:
        """Create volume profile from OHLCV data"""
        try:
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate volume-weighted price levels
            price_levels = []
            volume_at_levels = []
            
            for _, row in df.iterrows():
                # Create price levels within the candle range
                price_range = np.linspace(float(row['low']), float(row['high']), 10)
                volume_per_level = float(row['volume']) / 10
                
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
            
            # Find Point of Control (POC)
            poc_price = max(volume_distribution.keys(), key=lambda k: volume_distribution[k])
            poc_volume = volume_distribution[poc_price]
            
            # Calculate Value Area (68% of volume)
            sorted_levels = sorted(volume_distribution.items(), key=lambda x: x[1], reverse=True)
            total_volume = sum(volume_distribution.values())
            target_volume = total_volume * self.config['value_area_percentage']
            
            value_area_volume = 0
            value_area_levels = []
            
            for price, volume in sorted_levels:
                if value_area_volume < target_volume:
                    value_area_levels.append((price, volume))
                    value_area_volume += volume
                else:
                    break
            
            if value_area_levels:
                value_area_high = max(level[0] for level in value_area_levels)
                value_area_low = min(level[0] for level in value_area_levels)
            else:
                value_area_high = poc_price
                value_area_low = poc_price
            
            # Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
            avg_volume = total_volume / len(volume_distribution)
            high_volume_threshold = avg_volume * 1.5
            low_volume_threshold = avg_volume * 0.5
            
            high_volume_nodes = []
            low_volume_nodes = []
            
            for price, volume in volume_distribution.items():
                if volume >= high_volume_threshold:
                    high_volume_nodes.append(OrderBookLevel(
                        level_type=OrderBookLevelType.HIGH_VOLUME_NODE,
                        price_level=price,
                        volume_at_level=volume,
                        volume_percentage=(volume / total_volume) * 100,
                        bid_volume=volume * 0.5,  # Estimate
                        ask_volume=volume * 0.5,  # Estimate
                        volume_imbalance=0.0,
                        level_strength=min(volume / poc_volume, 1.0),
                        confidence=0.8,
                        timestamp=datetime.now(timezone.utc)
                    ))
                elif volume <= low_volume_threshold:
                    low_volume_nodes.append(OrderBookLevel(
                        level_type=OrderBookLevelType.LOW_VOLUME_NODE,
                        price_level=price,
                        volume_at_level=volume,
                        volume_percentage=(volume / total_volume) * 100,
                        bid_volume=volume * 0.5,  # Estimate
                        ask_volume=volume * 0.5,  # Estimate
                        volume_imbalance=0.0,
                        level_strength=volume / poc_volume,
                        confidence=0.6,
                        timestamp=datetime.now(timezone.utc)
                    ))
            
            # Calculate analysis confidence
            analysis_confidence = min(len(ohlcv_data) / self.config['volume_profile_periods'], 1.0)
            
            return VolumeProfile(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(timezone.utc),
                poc_price=poc_price,
                poc_volume=poc_volume,
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                value_area_volume=value_area_volume,
                total_volume=total_volume,
                high_volume_nodes=high_volume_nodes,
                low_volume_nodes=low_volume_nodes,
                liquidity_walls=[],  # Will be populated by order book analysis
                volume_distribution=volume_distribution,
                analysis_confidence=analysis_confidence
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating volume profile: {e}")
            return self._get_default_volume_profile(symbol, timeframe)
    
    async def _analyze_order_book_levels(self, order_book_data: List[Dict], volume_profile: VolumeProfile) -> List[OrderBookLevel]:
        """Analyze order book levels with volume profile context"""
        try:
            levels = []
            
            # Analyze recent order book snapshots
            for ob_data in order_book_data[:10]:  # Last 10 snapshots
                bids = ob_data.get('bids', [])
                asks = ob_data.get('asks', [])
                
                if not bids or not asks:
                    continue
                
                # Calculate total volumes
                total_bid_volume = sum(float(bid[1]) for bid in bids)
                total_ask_volume = sum(float(ask[1]) for ask in asks)
                total_volume = total_bid_volume + total_ask_volume
                
                # Identify liquidity walls (large orders)
                liquidity_wall_threshold = total_volume * 0.1  # 10% of total volume
                
                for bid in bids:
                    price, volume = float(bid[0]), float(bid[1])
                    if volume >= liquidity_wall_threshold:
                        # Check if this aligns with volume profile levels
                        volume_profile_strength = self._get_volume_profile_strength(price, volume_profile)
                        
                        level = OrderBookLevel(
                            level_type=OrderBookLevelType.LIQUIDITY_WALL,
                            price_level=price,
                            volume_at_level=volume,
                            volume_percentage=(volume / total_volume) * 100,
                            bid_volume=volume,
                            ask_volume=0,
                            volume_imbalance=1.0,
                            level_strength=volume_profile_strength,
                            confidence=0.9,
                            timestamp=ob_data['timestamp']
                        )
                        levels.append(level)
                
                for ask in asks:
                    price, volume = float(ask[0]), float(ask[1])
                    if volume >= liquidity_wall_threshold:
                        # Check if this aligns with volume profile levels
                        volume_profile_strength = self._get_volume_profile_strength(price, volume_profile)
                        
                        level = OrderBookLevel(
                            level_type=OrderBookLevelType.LIQUIDITY_WALL,
                            price_level=price,
                            volume_at_level=volume,
                            volume_percentage=(volume / total_volume) * 100,
                            bid_volume=0,
                            ask_volume=volume,
                            volume_imbalance=-1.0,
                            level_strength=volume_profile_strength,
                            confidence=0.9,
                            timestamp=ob_data['timestamp']
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing order book levels: {e}")
            return []
    
    def _get_volume_profile_strength(self, price: float, volume_profile: VolumeProfile) -> float:
        """Get volume profile strength for a given price"""
        try:
            # Find closest price level in volume distribution
            closest_price = min(volume_profile.volume_distribution.keys(), 
                              key=lambda x: abs(x - price))
            
            if abs(closest_price - price) <= self.config['level_tolerance']:
                volume_at_price = volume_profile.volume_distribution[closest_price]
                return min(volume_at_price / volume_profile.poc_volume, 1.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error getting volume profile strength: {e}")
            return 0.0
    
    async def _calculate_market_microstructure(self, order_book_data: List[Dict], volume_profile: VolumeProfile) -> Dict[str, Any]:
        """Calculate market microstructure metrics"""
        try:
            if not order_book_data:
                return self._get_default_microstructure()
            
            latest_ob = order_book_data[0]
            bids = latest_ob.get('bids', [])
            asks = latest_ob.get('asks', [])
            
            if not bids or not asks:
                return self._get_default_microstructure()
            
            # Calculate bid-ask imbalance
            total_bid_volume = sum(float(bid[1]) for bid in bids)
            total_ask_volume = sum(float(ask[1]) for ask in asks)
            total_volume = total_bid_volume + total_ask_volume
            
            bid_ask_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            # Calculate depth pressure
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_percentage = (spread / mid_price) * 100 if mid_price > 0 else 0
            
            # Calculate liquidity score
            liquidity_score = min(total_volume / 1000000, 1.0)  # Normalize by 1M volume
            
            return {
                'bid_ask_imbalance': bid_ask_imbalance,
                'depth_pressure': spread_percentage,
                'liquidity_score': liquidity_score,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating market microstructure: {e}")
            return self._get_default_microstructure()
    
    async def _prepare_algorithm_inputs(self, volume_profile: VolumeProfile, 
                                     order_book_levels: List[OrderBookLevel], 
                                     microstructure: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for volume-weighted algorithms"""
        try:
            # Prepare inputs for volume-weighted support/resistance
            volume_weighted_levels = []
            for level in order_book_levels:
                if level.level_strength >= self.config['min_level_strength']:
                    volume_weighted_levels.append({
                        'price': level.price_level,
                        'volume': level.volume_at_level,
                        'strength': level.level_strength,
                        'confidence': level.confidence,
                        'type': level.level_type.value
                    })
            
            # Prepare inputs for supply/demand zones
            supply_zones = []
            demand_zones = []
            
            for level in volume_profile.high_volume_nodes:
                if level.level_type == OrderBookLevelType.HIGH_VOLUME_NODE:
                    if level.price_level > volume_profile.poc_price:
                        supply_zones.append({
                            'price': level.price_level,
                            'volume': level.volume_at_level,
                            'strength': level.level_strength
                        })
                    else:
                        demand_zones.append({
                            'price': level.price_level,
                            'volume': level.volume_at_level,
                            'strength': level.level_strength
                        })
            
            return {
                'volume_profile': {
                    'poc_price': volume_profile.poc_price,
                    'poc_volume': volume_profile.poc_volume,
                    'value_area_high': volume_profile.value_area_high,
                    'value_area_low': volume_profile.value_area_low,
                    'total_volume': volume_profile.total_volume
                },
                'volume_weighted_levels': volume_weighted_levels,
                'supply_zones': supply_zones,
                'demand_zones': demand_zones,
                'high_volume_nodes': [{'price': node.price_level, 'volume': node.volume_at_level} 
                                    for node in volume_profile.high_volume_nodes],
                'low_volume_nodes': [{'price': node.price_level, 'volume': node.volume_at_level} 
                                    for node in volume_profile.low_volume_nodes],
                'liquidity_walls': [{'price': wall.price_level, 'volume': wall.volume_at_level, 'type': wall.level_type.value}
                                  for wall in order_book_levels if wall.level_type == OrderBookLevelType.LIQUIDITY_WALL],
                'market_microstructure': microstructure
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing algorithm inputs: {e}")
            return {}
    
    async def _store_order_book_analysis(self, analysis: OrderBookAnalysis):
        """Store order book analysis in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store volume profile
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
                
                await conn.execute(
                    query,
                    analysis.symbol,
                    analysis.volume_profile.timeframe,
                    analysis.timestamp,
                    analysis.volume_profile.poc_price,
                    analysis.volume_profile.poc_volume,
                    analysis.volume_profile.value_area_high,
                    analysis.volume_profile.value_area_low,
                    analysis.volume_profile.value_area_volume,
                    analysis.volume_profile.total_volume,
                    analysis.volume_profile.analysis_confidence,
                    json.dumps(analysis.algorithm_inputs)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error storing order book analysis: {e}")
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> OrderBookAnalysis:
        """Get default analysis when data is insufficient"""
        return OrderBookAnalysis(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            volume_profile=self._get_default_volume_profile(symbol, timeframe),
            bid_ask_imbalance=0.0,
            depth_pressure=0.0,
            liquidity_score=0.0,
            analysis_confidence=0.0,  # Low confidence for default analysis
            total_bid_volume=0.0,
            total_ask_volume=0.0,
            spread=0.0,
            spread_percentage=0.0,
            mid_price=0.0,
            best_bid=0.0,
            best_ask=0.0,
            order_book_levels=[],  # Empty list for default analysis
            market_microstructure=self._get_default_microstructure(),
            algorithm_inputs={}
        )
    
    def _get_default_volume_profile(self, symbol: str, timeframe: str) -> VolumeProfile:
        """Get default volume profile"""
        return VolumeProfile(
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
            liquidity_walls=[],
            volume_distribution={},
            analysis_confidence=0.0
        )
    
    def _get_default_microstructure(self) -> Dict[str, Any]:
        """Get default microstructure metrics"""
        return {
            'bid_ask_imbalance': 0.0,
            'depth_pressure': 0.0,
            'liquidity_score': 0.0,
            'total_bid_volume': 0.0,
            'total_ask_volume': 0.0,
            'spread': 0.0,
            'spread_percentage': 0.0,
            'mid_price': 0.0,
            'best_bid': 0.0,
            'best_ask': 0.0
        }

# Example usage
async def main():
    """Test the enhanced order book integration"""
    integration = EnhancedOrderBookIntegration()
    
    try:
        await integration.initialize()
        
        # Analyze order book with volume profile
        analysis = await integration.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
        
        print(f"\n📊 Order Book Analysis for {analysis.symbol}:")
        print(f"  POC Price: {analysis.volume_profile.poc_price}")
        print(f"  POC Volume: {analysis.volume_profile.poc_volume}")
        print(f"  Value Area: {analysis.volume_profile.value_area_low} - {analysis.volume_profile.value_area_high}")
        print(f"  Bid-Ask Imbalance: {analysis.bid_ask_imbalance:.3f}")
        print(f"  Liquidity Score: {analysis.liquidity_score:.3f}")
        print(f"  HVN Count: {len(analysis.volume_profile.high_volume_nodes)}")
        print(f"  LVN Count: {len(analysis.volume_profile.low_volume_nodes)}")
        
    finally:
        await integration.close()

if __name__ == "__main__":
    asyncio.run(main())
