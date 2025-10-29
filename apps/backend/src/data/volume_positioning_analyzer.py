"""
Volume Positioning Analyzer for AlphaPulse
Analyzes volume patterns, order book imbalances, and volume positioning for intelligent trading
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncpg
from dataclasses import dataclass
import ccxt

logger = logging.getLogger(__name__)

@dataclass
class VolumeAnalysis:
    """Volume analysis data structure"""
    timestamp: datetime
    symbol: str
    volume_ratio: float  # Current volume vs average
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    order_book_imbalance: float  # -1 to 1 (negative = bearish, positive = bullish)
    volume_positioning_score: float  # 0 to 1
    buy_volume_ratio: float  # Ratio of buy volume to total volume
    sell_volume_ratio: float  # Ratio of sell volume to total volume
    volume_breakout: bool  # True if volume is significantly above average
    volume_analysis: str  # Detailed analysis text

@dataclass
class LiquidityAnalysis:
    """Liquidity analysis data structure"""
    timestamp: datetime
    symbol: str
    liquidity_score: float  # 0 to 1 (overall liquidity)
    bid_liquidity: float  # Bid side liquidity
    ask_liquidity: float  # Ask side liquidity
    liquidity_walls: List[Dict[str, Any]]  # Large liquidity walls
    order_clusters: List[Dict[str, Any]]  # Order clusters
    depth_pressure: float  # -1 to 1 (pressure on price)
    spread_analysis: Dict[str, float]  # Spread metrics
    liquidity_analysis: str  # Detailed analysis text

@dataclass
class OrderBookAnalysis:
    """Order book analysis data structure"""
    timestamp: datetime
    symbol: str
    bid_ask_imbalance: float  # -1 to 1
    order_flow_toxicity: float  # -1 to 1 (negative = aggressive selling)
    depth_pressure: float  # -1 to 1 (pressure on price direction)
    liquidity_walls: List[Dict[str, Any]]  # Large liquidity walls
    order_clusters: List[Dict[str, Any]]  # Order clusters
    spread_analysis: Dict[str, float]  # Spread metrics
    order_book_analysis: str  # Detailed analysis text

@dataclass
class MarketDepthAnalysis:
    """Market depth analysis data structure"""
    timestamp: datetime
    symbol: str
    analysis_type: str  # 'liquidity_walls', 'order_clusters', 'imbalance'
    price_level: float
    volume_at_level: float
    side: str  # 'bid' or 'ask'
    confidence: float  # 0 to 1
    strength_score: float  # 0 to 1
    distance_from_mid: float  # Percentage from mid price
    wall_thickness: int  # Number of levels
    metadata: Dict[str, Any]

class VolumePositioningAnalyzer:
    """
    Advanced volume positioning analyzer
    Analyzes volume patterns, order book imbalances, and volume positioning
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        self.volume_history = {}  # Cache for volume history
        self.order_book_cache = {}  # Cache for order book data
        
        logger.info("Volume Positioning Analyzer initialized")
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, List]:
        """Get order book data from exchange"""
        try:
            cache_key = f"order_book_{symbol}"
            current_time = datetime.now()
            
            # Check cache (5 second cache)
            if cache_key in self.order_book_cache:
                cached_data = self.order_book_cache[cache_key]
                if (current_time - cached_data['timestamp']).seconds < 5:
                    return cached_data['data']
            
            # Fetch fresh order book data
            if hasattr(self.exchange, 'fetch_order_book'):
                order_book = await self.exchange.fetch_order_book(symbol, limit)
            else:
                # Fallback for non-async exchanges
                order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # Cache the result
            self.order_book_cache[cache_key] = {
                'data': order_book,
                'timestamp': current_time
            }
            
            return order_book
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    async def calculate_order_book_imbalance(self, symbol: str) -> float:
        """Calculate order book imbalance (-1 to 1)"""
        try:
            order_book = await self.get_order_book(symbol)
            
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate total bid and ask volume
            bid_volume = sum(bid[1] for bid in order_book['bids'])
            ask_volume = sum(ask[1] for ask in order_book['asks'])
            
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Calculate imbalance (-1 to 1)
            imbalance = (bid_volume - ask_volume) / total_volume
            
            logger.info(f"Order book imbalance for {symbol}: {imbalance:.3f}")
            return imbalance
            
        except Exception as e:
            logger.error(f"Error calculating order book imbalance: {e}")
            return 0.0
    
    async def get_volume_history(self, symbol: str, timeframe: str = '1h', limit: int = 24) -> List[float]:
        """Get volume history for analysis"""
        try:
            cache_key = f"volume_history_{symbol}_{timeframe}"
            current_time = datetime.now()
            
            # Check cache (1 minute cache)
            if cache_key in self.volume_history:
                cached_data = self.volume_history[cache_key]
                if (current_time - cached_data['timestamp']).seconds < 60:
                    return cached_data['data']
            
            # Fetch OHLCV data
            if hasattr(self.exchange, 'fetch_ohlcv'):
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                # Fallback for non-async exchanges
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Extract volumes
            volumes = [candle[5] for candle in ohlcv]
            
            # Cache the result
            self.volume_history[cache_key] = {
                'data': volumes,
                'timestamp': current_time
            }
            
            return volumes
            
        except Exception as e:
            logger.error(f"Error getting volume history for {symbol}: {e}")
            return []
    
    async def calculate_volume_ratio(self, symbol: str, timeframe: str = '1h') -> float:
        """Calculate current volume ratio compared to average"""
        try:
            volumes = await self.get_volume_history(symbol, timeframe)
            
            if len(volumes) < 2:
                return 1.0
            
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])  # Average excluding current
            
            if avg_volume == 0:
                return 1.0
            
            volume_ratio = current_volume / avg_volume
            
            logger.info(f"Volume ratio for {symbol}: {volume_ratio:.3f}")
            return volume_ratio
            
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    async def determine_volume_trend(self, symbol: str, timeframe: str = '1h') -> str:
        """Determine volume trend direction"""
        try:
            volumes = await self.get_volume_history(symbol, timeframe)
            
            if len(volumes) < 3:
                return "stable"
            
            # Calculate trend using linear regression
            x = np.arange(len(volumes))
            y = np.array(volumes)
            
            # Simple trend calculation
            recent_avg = np.mean(volumes[-3:])
            older_avg = np.mean(volumes[-6:-3]) if len(volumes) >= 6 else volumes[0]
            
            if recent_avg > older_avg * 1.1:
                return "increasing"
            elif recent_avg < older_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error determining volume trend: {e}")
            return "stable"
    
    async def calculate_buy_sell_volume_ratio(self, symbol: str) -> Tuple[float, float]:
        """Calculate buy vs sell volume ratio"""
        try:
            # Get recent trades to analyze buy/sell volume
            if hasattr(self.exchange, 'fetch_trades'):
                trades = await self.exchange.fetch_trades(symbol, limit=100)
            else:
                # Fallback for non-async exchanges
                trades = self.exchange.fetch_trades(symbol, limit=100)
            
            if not trades:
                return 0.5, 0.5
            
            buy_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'buy')
            sell_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'sell')
            
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.5, 0.5
            
            buy_ratio = buy_volume / total_volume
            sell_ratio = sell_volume / total_volume
            
            logger.info(f"Buy/Sell ratio for {symbol}: {buy_ratio:.3f}/{sell_ratio:.3f}")
            return buy_ratio, sell_ratio
            
        except Exception as e:
            logger.error(f"Error calculating buy/sell volume ratio: {e}")
            return 0.5, 0.5
    
    async def detect_volume_breakout(self, symbol: str, threshold: float = 2.0) -> bool:
        """Detect if current volume is significantly above average"""
        try:
            volume_ratio = await self.calculate_volume_ratio(symbol)
            is_breakout = volume_ratio > threshold
            
            logger.info(f"Volume breakout for {symbol}: {is_breakout} (ratio: {volume_ratio:.3f})")
            return is_breakout
            
        except Exception as e:
            logger.error(f"Error detecting volume breakout: {e}")
            return False
    
    async def calculate_volume_positioning_score(self, symbol: str) -> float:
        """Calculate comprehensive volume positioning score (0-1)"""
        try:
            # Get various volume metrics
            volume_ratio = await self.calculate_volume_ratio(symbol)
            order_book_imbalance = await self.calculate_order_book_imbalance(symbol)
            buy_ratio, sell_ratio = await self.calculate_buy_sell_volume_ratio(symbol)
            volume_breakout = await self.detect_volume_breakout(symbol)
            
            # Calculate score components
            volume_score = min(volume_ratio / 3.0, 1.0)  # Normalize volume ratio
            imbalance_score = (order_book_imbalance + 1) / 2  # Convert -1,1 to 0,1
            buy_pressure_score = buy_ratio
            
            # Combine scores with weights
            final_score = (
                volume_score * 0.3 +
                imbalance_score * 0.4 +
                buy_pressure_score * 0.3
            )
            
            # Boost score if volume breakout detected
            if volume_breakout:
                final_score = min(final_score * 1.2, 1.0)
            
            logger.info(f"Volume positioning score for {symbol}: {final_score:.3f}")
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating volume positioning score: {e}")
            return 0.5
    
    async def generate_volume_analysis(self, symbol: str) -> str:
        """Generate detailed volume analysis text"""
        try:
            volume_ratio = await self.calculate_volume_ratio(symbol)
            order_book_imbalance = await self.calculate_order_book_imbalance(symbol)
            buy_ratio, sell_ratio = await self.calculate_buy_sell_volume_ratio(symbol)
            volume_trend = await self.determine_volume_trend(symbol)
            volume_breakout = await self.detect_volume_breakout(symbol)
            
            analysis_parts = []
            
            # Volume level analysis
            if volume_ratio > 2.0:
                analysis_parts.append("Volume is significantly above average")
            elif volume_ratio > 1.5:
                analysis_parts.append("Volume is moderately above average")
            elif volume_ratio < 0.5:
                analysis_parts.append("Volume is significantly below average")
            elif volume_ratio < 0.8:
                analysis_parts.append("Volume is below average")
            else:
                analysis_parts.append("Volume is at normal levels")
            
            # Order book analysis
            if order_book_imbalance > 0.3:
                analysis_parts.append("Strong buying pressure in order book")
            elif order_book_imbalance < -0.3:
                analysis_parts.append("Strong selling pressure in order book")
            else:
                analysis_parts.append("Balanced order book")
            
            # Buy/sell ratio analysis
            if buy_ratio > 0.6:
                analysis_parts.append("Dominant buying activity")
            elif sell_ratio > 0.6:
                analysis_parts.append("Dominant selling activity")
            else:
                analysis_parts.append("Mixed buying and selling activity")
            
            # Volume trend analysis
            analysis_parts.append(f"Volume trend is {volume_trend}")
            
            # Breakout analysis
            if volume_breakout:
                analysis_parts.append("Volume breakout detected")
            
            analysis_text = ". ".join(analysis_parts) + "."
            
            logger.info(f"Volume analysis for {symbol}: {analysis_text}")
            return analysis_text
            
        except Exception as e:
            logger.error(f"Error generating volume analysis: {e}")
            return "Volume analysis unavailable"
    
    async def analyze_volume_positioning(self, symbol: str, timeframe: str = '1h') -> VolumeAnalysis:
        """Perform comprehensive volume positioning analysis"""
        try:
            logger.info(f"🔄 Analyzing volume positioning for {symbol}")
            
            # Calculate all volume metrics
            volume_ratio = await self.calculate_volume_ratio(symbol)
            volume_trend = await self.determine_volume_trend(symbol)
            order_book_imbalance = await self.calculate_order_book_imbalance(symbol)
            volume_positioning_score = await self.calculate_volume_positioning_score(symbol)
            buy_ratio, sell_ratio = await self.calculate_buy_sell_volume_ratio(symbol)
            volume_breakout = await self.detect_volume_breakout(symbol)
            volume_analysis = await self.generate_volume_analysis(symbol)
            
            # Create volume analysis object
            analysis = VolumeAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                order_book_imbalance=order_book_imbalance,
                volume_positioning_score=volume_positioning_score,
                buy_volume_ratio=buy_ratio,
                sell_volume_ratio=sell_ratio,
                volume_breakout=volume_breakout,
                volume_analysis=volume_analysis
            )
            
            logger.info(f"✅ Volume positioning analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing volume positioning for {symbol}: {e}")
            # Return default analysis
            return VolumeAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                volume_ratio=1.0,
                volume_trend="stable",
                order_book_imbalance=0.0,
                volume_positioning_score=0.5,
                buy_volume_ratio=0.5,
                sell_volume_ratio=0.5,
                volume_breakout=False,
                volume_analysis="Volume analysis unavailable"
            )
    
    async def store_volume_analysis(self, analysis: VolumeAnalysis, timeframe: str = '1h') -> bool:
        """Store volume analysis in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO volume_analysis (
                        timestamp, symbol, timeframe, volume_ratio, volume_trend, order_book_imbalance,
                        volume_positioning_score, buy_volume_ratio, sell_volume_ratio,
                        volume_breakout, volume_analysis
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                analysis.timestamp, analysis.symbol, timeframe, analysis.volume_ratio, analysis.volume_trend,
                analysis.order_book_imbalance, analysis.volume_positioning_score,
                analysis.buy_volume_ratio, analysis.sell_volume_ratio,
                analysis.volume_breakout, analysis.volume_analysis
                )
            
            logger.info(f"✅ Volume analysis stored for {analysis.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing volume analysis: {e}")
            return False

    # ==================== ENHANCED LIQUIDITY ANALYSIS METHODS ====================
    
    async def analyze_liquidity(self, symbol: str) -> LiquidityAnalysis:
        """Analyze liquidity patterns and liquidity walls"""
        try:
            logger.info(f"🔍 Analyzing liquidity for {symbol}")
            
            # Get order book data
            order_book = await self.get_order_book(symbol, limit=100)
            
            # Calculate liquidity metrics
            liquidity_score = await self.calculate_liquidity_score(order_book)
            bid_liquidity, ask_liquidity = await self.calculate_bid_ask_liquidity(order_book)
            liquidity_walls = await self.detect_liquidity_walls(order_book)
            order_clusters = await self.detect_order_clusters(order_book)
            depth_pressure = await self.calculate_depth_pressure(order_book)
            spread_analysis = await self.analyze_spread(order_book)
            liquidity_analysis = await self.generate_liquidity_analysis(symbol, liquidity_walls, order_clusters)
            
            # Create liquidity analysis object
            analysis = LiquidityAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                liquidity_score=liquidity_score,
                bid_liquidity=bid_liquidity,
                ask_liquidity=ask_liquidity,
                liquidity_walls=liquidity_walls,
                order_clusters=order_clusters,
                depth_pressure=depth_pressure,
                spread_analysis=spread_analysis,
                liquidity_analysis=liquidity_analysis
            )
            
            logger.info(f"✅ Liquidity analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing liquidity for {symbol}: {e}")
            return self._get_default_liquidity_analysis(symbol)
    
    async def analyze_order_book(self, symbol: str) -> OrderBookAnalysis:
        """Analyze order book patterns and imbalances"""
        try:
            logger.info(f"🔍 Analyzing order book for {symbol}")
            
            # Get order book data
            order_book = await self.get_order_book(symbol, limit=100)
            
            # Calculate order book metrics
            bid_ask_imbalance = await self.calculate_weighted_imbalance(order_book)
            order_flow_toxicity = await self.calculate_order_flow_toxicity(order_book)
            depth_pressure = await self.calculate_depth_pressure(order_book)
            liquidity_walls = await self.detect_liquidity_walls(order_book)
            order_clusters = await self.detect_order_clusters(order_book)
            spread_analysis = await self.analyze_spread(order_book)
            order_book_analysis = await self.generate_order_book_analysis(symbol, bid_ask_imbalance, order_flow_toxicity)
            
            # Create order book analysis object
            analysis = OrderBookAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                bid_ask_imbalance=bid_ask_imbalance,
                order_flow_toxicity=order_flow_toxicity,
                depth_pressure=depth_pressure,
                liquidity_walls=liquidity_walls,
                order_clusters=order_clusters,
                spread_analysis=spread_analysis,
                order_book_analysis=order_book_analysis
            )
            
            logger.info(f"✅ Order book analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing order book for {symbol}: {e}")
            return self._get_default_order_book_analysis(symbol)
    
    async def analyze_market_depth(self, symbol: str) -> List[MarketDepthAnalysis]:
        """Analyze market depth for liquidity walls and order clusters"""
        try:
            logger.info(f"🔍 Analyzing market depth for {symbol}")
            
            # Get order book data
            order_book = await self.get_order_book(symbol, limit=100)
            
            analyses = []
            
            # Analyze liquidity walls
            liquidity_walls = await self.detect_liquidity_walls(order_book)
            for wall in liquidity_walls:
                analysis = MarketDepthAnalysis(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    analysis_type='liquidity_walls',
                    price_level=wall['price_level'],
                    volume_at_level=wall['volume'],
                    side=wall['side'],
                    confidence=wall['confidence'],
                    strength_score=wall['strength_score'],
                    distance_from_mid=wall['distance_from_mid'],
                    wall_thickness=wall['wall_thickness'],
                    metadata=wall.get('metadata', {})
                )
                analyses.append(analysis)
            
            # Analyze order clusters
            order_clusters = await self.detect_order_clusters(order_book)
            for cluster in order_clusters:
                analysis = MarketDepthAnalysis(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    analysis_type='order_clusters',
                    price_level=cluster['price_level'],
                    volume_at_level=cluster['volume'],
                    side=cluster['side'],
                    confidence=cluster['confidence'],
                    strength_score=cluster['strength_score'],
                    distance_from_mid=cluster['distance_from_mid'],
                    wall_thickness=cluster['wall_thickness'],
                    metadata=cluster.get('metadata', {})
                )
                analyses.append(analysis)
            
            logger.info(f"✅ Market depth analysis completed for {symbol}: {len(analyses)} analyses")
            return analyses
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market depth for {symbol}: {e}")
            return []

    async def calculate_liquidity_score(self, order_book: Dict[str, List]) -> float:
        """Calculate overall liquidity score (0 to 1)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate total volume
            bid_volume = sum(bid[1] for bid in order_book['bids'])
            ask_volume = sum(ask[1] for ask in order_book['asks'])
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Calculate spread
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread = (best_ask - best_bid) / best_bid
            
            # Calculate depth (number of levels with significant volume)
            depth_threshold = total_volume * 0.01  # 1% of total volume
            bid_depth = sum(1 for bid in order_book['bids'] if bid[1] > depth_threshold)
            ask_depth = sum(1 for ask in order_book['asks'] if ask[1] > depth_threshold)
            
            # Calculate liquidity score
            volume_score = min(total_volume / 1000, 1.0)  # Normalize to 1000 volume units
            spread_score = max(0, 1 - spread * 100)  # Lower spread = higher score
            depth_score = min((bid_depth + ask_depth) / 20, 1.0)  # Normalize to 20 levels
            
            liquidity_score = (volume_score * 0.4 + spread_score * 0.4 + depth_score * 0.2)
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.0

    async def calculate_bid_ask_liquidity(self, order_book: Dict[str, List]) -> Tuple[float, float]:
        """Calculate bid and ask liquidity separately"""
        try:
            bid_volume = sum(bid[1] for bid in order_book['bids']) if order_book['bids'] else 0
            ask_volume = sum(ask[1] for ask in order_book['asks']) if order_book['asks'] else 0
            
            return bid_volume, ask_volume
            
        except Exception as e:
            logger.error(f"Error calculating bid/ask liquidity: {e}")
            return 0.0, 0.0

    async def detect_liquidity_walls(self, order_book: Dict[str, List]) -> List[Dict[str, Any]]:
        """Detect large liquidity walls at specific price levels"""
        try:
            walls = []
            
            if not order_book['bids'] or not order_book['asks']:
                return walls
            
            # Calculate average volume for threshold
            all_volumes = [bid[1] for bid in order_book['bids']] + [ask[1] for ask in order_book['asks']]
            avg_volume = sum(all_volumes) / len(all_volumes) if all_volumes else 0
            wall_threshold = avg_volume * 3  # 3x average volume
            
            # Detect bid walls
            for i, bid in enumerate(order_book['bids']):
                if bid[1] > wall_threshold:
                    wall = {
                        'price_level': bid[0],
                        'volume': bid[1],
                        'side': 'bid',
                        'confidence': min(bid[1] / wall_threshold, 1.0),
                        'strength_score': min(bid[1] / avg_volume, 5.0),
                        'distance_from_mid': abs(bid[0] - (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2),
                        'wall_thickness': 1,
                        'metadata': {'level': i, 'threshold': wall_threshold}
                    }
                    walls.append(wall)
            
            # Detect ask walls
            for i, ask in enumerate(order_book['asks']):
                if ask[1] > wall_threshold:
                    wall = {
                        'price_level': ask[0],
                        'volume': ask[1],
                        'side': 'ask',
                        'confidence': min(ask[1] / wall_threshold, 1.0),
                        'strength_score': min(ask[1] / avg_volume, 5.0),
                        'distance_from_mid': abs(ask[0] - (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2),
                        'wall_thickness': 1,
                        'metadata': {'level': i, 'threshold': wall_threshold}
                    }
                    walls.append(wall)
            
            return walls
            
        except Exception as e:
            logger.error(f"Error detecting liquidity walls: {e}")
            return []

    async def detect_order_clusters(self, order_book: Dict[str, List]) -> List[Dict[str, Any]]:
        """Detect order clusters at similar price levels"""
        try:
            clusters = []
            
            if not order_book['bids'] or not order_book['asks']:
                return clusters
            
            # Group orders by price proximity
            price_tolerance = 0.001  # 0.1% price tolerance
            
            # Cluster bid orders
            bid_clusters = self._find_price_clusters(order_book['bids'], price_tolerance)
            for cluster in bid_clusters:
                cluster_analysis = self._analyze_cluster(cluster, 'bid', order_book)
                if cluster_analysis:
                    clusters.append(cluster_analysis)
            
            # Cluster ask orders
            ask_clusters = self._find_price_clusters(order_book['asks'], price_tolerance)
            for cluster in ask_clusters:
                cluster_analysis = self._analyze_cluster(cluster, 'ask', order_book)
                if cluster_analysis:
                    clusters.append(cluster_analysis)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting order clusters: {e}")
            return []

    def _find_price_clusters(self, orders: List[List], tolerance: float) -> List[List[List]]:
        """Find clusters of orders at similar price levels"""
        if not orders:
            return []
        
        clusters = []
        current_cluster = [orders[0]]
        
        for i in range(1, len(orders)):
            current_price = orders[i][0]
            prev_price = orders[i-1][0]
            
            if abs(current_price - prev_price) / prev_price <= tolerance:
                current_cluster.append(orders[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [orders[i]]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return clusters

    def _analyze_cluster(self, cluster: List[List], side: str, order_book: Dict[str, List]) -> Dict[str, Any]:
        """Analyze a cluster of orders"""
        try:
            total_volume = sum(order[1] for order in cluster)
            avg_price = sum(order[0] * order[1] for order in cluster) / total_volume
            
            # Calculate cluster metrics
            volume_concentration = total_volume / sum(order[1] for order in (order_book['bids'] if side == 'bid' else order_book['asks']))
            price_range = max(order[0] for order in cluster) - min(order[0] for order in cluster)
            
            return {
                'price_level': avg_price,
                'volume': total_volume,
                'side': side,
                'confidence': min(volume_concentration * 2, 1.0),
                'strength_score': min(volume_concentration * 5, 5.0),
                'distance_from_mid': abs(avg_price - (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2),
                'wall_thickness': len(cluster),
                'metadata': {
                    'volume_concentration': volume_concentration,
                    'price_range': price_range,
                    'order_count': len(cluster)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cluster: {e}")
            return None

    async def calculate_depth_pressure(self, order_book: Dict[str, List]) -> float:
        """Calculate depth pressure (-1 to 1) indicating price direction pressure"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate weighted average prices
            bid_volume = sum(bid[1] for bid in order_book['bids'])
            ask_volume = sum(ask[1] for ask in order_book['asks'])
            
            if bid_volume == 0 or ask_volume == 0:
                return 0.0
            
            weighted_bid_price = sum(bid[0] * bid[1] for bid in order_book['bids']) / bid_volume
            weighted_ask_price = sum(ask[0] * ask[1] for ask in order_book['asks']) / ask_volume
            
            # Calculate pressure based on volume imbalance
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Calculate pressure based on price proximity to mid
            mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            bid_pressure = (mid_price - weighted_bid_price) / mid_price
            ask_pressure = (weighted_ask_price - mid_price) / mid_price
            
            # Combine pressure indicators
            pressure = volume_imbalance * 0.6 + (bid_pressure - ask_pressure) * 0.4
            
            return max(-1.0, min(1.0, pressure))
            
        except Exception as e:
            logger.error(f"Error calculating depth pressure: {e}")
            return 0.0

    async def calculate_weighted_imbalance(self, order_book: Dict[str, List]) -> float:
        """Calculate weighted order book imbalance (-1 to 1)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate weighted volumes (closer to mid = higher weight)
            mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            
            weighted_bid_volume = 0
            for i, bid in enumerate(order_book['bids']):
                weight = 1.0 / (i + 1)  # Decreasing weight with distance from mid
                weighted_bid_volume += bid[1] * weight
            
            weighted_ask_volume = 0
            for i, ask in enumerate(order_book['asks']):
                weight = 1.0 / (i + 1)  # Decreasing weight with distance from mid
                weighted_ask_volume += ask[1] * weight
            
            total_weighted_volume = weighted_bid_volume + weighted_ask_volume
            
            if total_weighted_volume == 0:
                return 0.0
            
            imbalance = (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume
            
            return max(-1.0, min(1.0, imbalance))
            
        except Exception as e:
            logger.error(f"Error calculating weighted imbalance: {e}")
            return 0.0

    async def calculate_order_flow_toxicity(self, order_book: Dict[str, List]) -> float:
        """Calculate order flow toxicity (-1 to 1)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Analyze order size distribution
            bid_sizes = [bid[1] for bid in order_book['bids']]
            ask_sizes = [ask[1] for ask in order_book['asks']]
            
            # Calculate size skewness (large orders = toxic)
            avg_bid_size = sum(bid_sizes) / len(bid_sizes) if bid_sizes else 0
            avg_ask_size = sum(ask_sizes) / len(ask_sizes) if ask_sizes else 0
            
            bid_toxicity = sum(1 for size in bid_sizes if size > avg_bid_size * 2) / len(bid_sizes) if bid_sizes else 0
            ask_toxicity = sum(1 for size in ask_sizes if size > avg_ask_size * 2) / len(ask_sizes) if ask_sizes else 0
            
            # Calculate overall toxicity
            toxicity = (bid_toxicity + ask_toxicity) / 2
            
            # Convert to -1 to 1 scale (negative = aggressive selling)
            return (toxicity - 0.5) * 2
            
        except Exception as e:
            logger.error(f"Error calculating order flow toxicity: {e}")
            return 0.0

    async def analyze_spread(self, order_book: Dict[str, List]) -> Dict[str, float]:
        """Analyze bid-ask spread metrics"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return {'spread': 0.0, 'spread_percentage': 0.0, 'mid_price': 0.0}
            
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
            
            spread = best_ask - best_bid
            spread_percentage = (spread / mid_price) * 100
            
            return {
                'spread': spread,
                'spread_percentage': spread_percentage,
                'mid_price': mid_price
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spread: {e}")
            return {'spread': 0.0, 'spread_percentage': 0.0, 'mid_price': 0.0}

    async def generate_liquidity_analysis(self, symbol: str, liquidity_walls: List[Dict], order_clusters: List[Dict]) -> str:
        """Generate human-readable liquidity analysis"""
        try:
            analysis = f"Liquidity analysis for {symbol}: "
            
            if liquidity_walls:
                wall_count = len(liquidity_walls)
                strongest_wall = max(liquidity_walls, key=lambda x: x['strength_score'])
                analysis += f"Detected {wall_count} liquidity walls. "
                analysis += f"Strongest wall at {strongest_wall['price_level']:.2f} ({strongest_wall['side']}) with strength {strongest_wall['strength_score']:.2f}. "
            else:
                analysis += "No significant liquidity walls detected. "
            
            if order_clusters:
                cluster_count = len(order_clusters)
                analysis += f"Found {cluster_count} order clusters. "
            else:
                analysis += "No significant order clusters detected. "
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating liquidity analysis: {e}")
            return f"Liquidity analysis for {symbol}: Analysis unavailable"

    async def generate_order_book_analysis(self, symbol: str, imbalance: float, toxicity: float) -> str:
        """Generate human-readable order book analysis"""
        try:
            analysis = f"Order book analysis for {symbol}: "
            
            if imbalance > 0.1:
                analysis += f"Bullish imbalance ({imbalance:.3f}). "
            elif imbalance < -0.1:
                analysis += f"Bearish imbalance ({imbalance:.3f}). "
            else:
                analysis += f"Balanced order book ({imbalance:.3f}). "
            
            if toxicity > 0.1:
                analysis += f"High order flow toxicity ({toxicity:.3f}). "
            elif toxicity < -0.1:
                analysis += f"Low order flow toxicity ({toxicity:.3f}). "
            else:
                analysis += f"Normal order flow toxicity ({toxicity:.3f}). "
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating order book analysis: {e}")
            return f"Order book analysis for {symbol}: Analysis unavailable"

    def _get_default_liquidity_analysis(self, symbol: str) -> LiquidityAnalysis:
        """Get default liquidity analysis when analysis fails"""
        return LiquidityAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            liquidity_score=0.0,
            bid_liquidity=0.0,
            ask_liquidity=0.0,
            liquidity_walls=[],
            order_clusters=[],
            depth_pressure=0.0,
            spread_analysis={'spread': 0.0, 'spread_percentage': 0.0, 'mid_price': 0.0},
            liquidity_analysis=f"Liquidity analysis for {symbol}: Analysis unavailable"
        )

    def _get_default_order_book_analysis(self, symbol: str) -> OrderBookAnalysis:
        """Get default order book analysis when analysis fails"""
        return OrderBookAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            bid_ask_imbalance=0.0,
            order_flow_toxicity=0.0,
            depth_pressure=0.0,
            liquidity_walls=[],
            order_clusters=[],
            spread_analysis={'spread': 0.0, 'spread_percentage': 0.0, 'mid_price': 0.0},
            order_book_analysis=f"Order book analysis for {symbol}: Analysis unavailable"
        )

# Example usage
async def main():
    """Example usage of Volume Positioning Analyzer"""
    # Initialize database pool
    db_pool = await asyncpg.create_pool(
        host='postgres',
        port=5432,
        database='alphapulse',
        user='alpha_emon',
        password='Emon_@17711',
        min_size=5,
        max_size=20
    )
    
    # Initialize exchange - Use SPOT API, not futures
    from safe_exchange_config import create_safe_binance_exchange
    exchange = create_safe_binance_exchange()
    
    # Create analyzer
    analyzer = VolumePositioningAnalyzer(db_pool, exchange)
    
    # Analyze volume positioning for BTC/USDT
    analysis = await analyzer.analyze_volume_positioning('BTC/USDT')
    
    print(f"Volume Analysis for {analysis.symbol}:")
    print(f"Volume Ratio: {analysis.volume_ratio:.3f}")
    print(f"Order Book Imbalance: {analysis.order_book_imbalance:.3f}")
    print(f"Volume Positioning Score: {analysis.volume_positioning_score:.3f}")
    print(f"Analysis: {analysis.volume_analysis}")

if __name__ == "__main__":
    asyncio.run(main())

    # ==================== ENHANCED LIQUIDITY ANALYSIS METHODS ====================
    
    async def analyze_liquidity(self, symbol: str) -> LiquidityAnalysis:
        """Analyze liquidity patterns and liquidity walls"""
        try:
            logger.info(f"🔍 Analyzing liquidity for {symbol}")
            
            # Get order book data
            order_book = await self.get_order_book(symbol, limit=100)
            
            # Calculate liquidity metrics
            liquidity_score = await self.calculate_liquidity_score(order_book)
            bid_liquidity, ask_liquidity = await self.calculate_bid_ask_liquidity(order_book)
            liquidity_walls = await self.detect_liquidity_walls(order_book)
            order_clusters = await self.detect_order_clusters(order_book)
            depth_pressure = await self.calculate_depth_pressure(order_book)
            spread_analysis = await self.analyze_spread(order_book)
            liquidity_analysis = await self.generate_liquidity_analysis(symbol, liquidity_walls, order_clusters)
            
            # Create liquidity analysis object
            analysis = LiquidityAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                liquidity_score=liquidity_score,
                bid_liquidity=bid_liquidity,
                ask_liquidity=ask_liquidity,
                liquidity_walls=liquidity_walls,
                order_clusters=order_clusters,
                depth_pressure=depth_pressure,
                spread_analysis=spread_analysis,
                liquidity_analysis=liquidity_analysis
            )
            
            logger.info(f"✅ Liquidity analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing liquidity for {symbol}: {e}")
            return self._get_default_liquidity_analysis(symbol)
    
    async def analyze_order_book(self, symbol: str) -> OrderBookAnalysis:
        """Analyze order book patterns and imbalances"""
        try:
            logger.info(f"🔍 Analyzing order book for {symbol}")
            
            # Get order book data
            order_book = await self.get_order_book(symbol, limit=100)
            
            # Calculate order book metrics
            bid_ask_imbalance = await self.calculate_weighted_imbalance(order_book)
            order_flow_toxicity = await self.calculate_order_flow_toxicity(order_book)
            depth_pressure = await self.calculate_depth_pressure(order_book)
            liquidity_walls = await self.detect_liquidity_walls(order_book)
            order_clusters = await self.detect_order_clusters(order_book)
            spread_analysis = await self.analyze_spread(order_book)
            order_book_analysis = await self.generate_order_book_analysis(symbol, bid_ask_imbalance, order_flow_toxicity)
            
            # Create order book analysis object
            analysis = OrderBookAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                bid_ask_imbalance=bid_ask_imbalance,
                order_flow_toxicity=order_flow_toxicity,
                depth_pressure=depth_pressure,
                liquidity_walls=liquidity_walls,
                order_clusters=order_clusters,
                spread_analysis=spread_analysis,
                order_book_analysis=order_book_analysis
            )
            
            logger.info(f"✅ Order book analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing order book for {symbol}: {e}")
            return self._get_default_order_book_analysis(symbol)
    
    async def analyze_market_depth(self, symbol: str) -> List[MarketDepthAnalysis]:
        """Analyze market depth for liquidity walls and order clusters"""
        try:
            logger.info(f"🔍 Analyzing market depth for {symbol}")
            
            # Get order book data
            order_book = await self.get_order_book(symbol, limit=100)
            
            analyses = []
            
            # Analyze liquidity walls
            liquidity_walls = await self.detect_liquidity_walls(order_book)
            for wall in liquidity_walls:
                analysis = MarketDepthAnalysis(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    analysis_type='liquidity_walls',
                    price_level=wall['price_level'],
                    volume_at_level=wall['volume'],
                    side=wall['side'],
                    confidence=wall['confidence'],
                    strength_score=wall['strength'],
                    distance_from_mid=wall['distance_from_mid'],
                    wall_thickness=wall['thickness'],
                    metadata=wall
                )
                analyses.append(analysis)
            
            # Analyze order clusters
            order_clusters = await self.detect_order_clusters(order_book)
            for cluster in order_clusters:
                analysis = MarketDepthAnalysis(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    analysis_type='order_clusters',
                    price_level=cluster['price_level'],
                    volume_at_level=cluster['volume'],
                    side=cluster['side'],
                    confidence=cluster['confidence'],
                    strength_score=cluster['strength'],
                    distance_from_mid=cluster['distance_from_mid'],
                    wall_thickness=cluster['thickness'],
                    metadata=cluster
                )
                analyses.append(analysis)
            
            logger.info(f"✅ Market depth analysis completed for {symbol}: {len(analyses)} analyses")
            return analyses
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market depth for {symbol}: {e}")
            return []
    
    # ==================== LIQUIDITY ANALYSIS HELPER METHODS ====================
    
    async def calculate_liquidity_score(self, order_book: Dict[str, List]) -> float:
        """Calculate overall liquidity score (0 to 1)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate total volume
            total_bid_volume = sum(bid[1] for bid in order_book['bids'])
            total_ask_volume = sum(ask[1] for ask in order_book['asks'])
            total_volume = total_bid_volume + total_ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Calculate spread
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread = (best_ask - best_bid) / best_bid
            
            # Calculate depth (volume within 1% of mid price)
            mid_price = (best_bid + best_ask) / 2
            depth_threshold = mid_price * 0.01
            
            bid_depth = sum(bid[1] for bid in order_book['bids'] if bid[0] >= mid_price - depth_threshold)
            ask_depth = sum(ask[1] for ask in order_book['asks'] if ask[0] <= mid_price + depth_threshold)
            
            # Calculate liquidity score based on volume, spread, and depth
            volume_score = min(total_volume / 1000, 1.0)  # Normalize to 0-1
            spread_score = max(0, 1 - spread * 100)  # Lower spread = higher score
            depth_score = min((bid_depth + ask_depth) / 100, 1.0)  # Normalize to 0-1
            
            liquidity_score = (volume_score * 0.4 + spread_score * 0.3 + depth_score * 0.3)
            
            return min(max(liquidity_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    async def calculate_bid_ask_liquidity(self, order_book: Dict[str, List]) -> Tuple[float, float]:
        """Calculate bid and ask side liquidity separately"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0, 0.0
            
            # Calculate bid liquidity
            bid_volume = sum(bid[1] for bid in order_book['bids'])
            bid_liquidity = min(bid_volume / 500, 1.0)  # Normalize to 0-1
            
            # Calculate ask liquidity
            ask_volume = sum(ask[1] for ask in order_book['asks'])
            ask_liquidity = min(ask_volume / 500, 1.0)  # Normalize to 0-1
            
            return bid_liquidity, ask_liquidity
            
        except Exception as e:
            logger.error(f"Error calculating bid/ask liquidity: {e}")
            return 0.5, 0.5
    
    async def detect_liquidity_walls(self, order_book: Dict[str, List]) -> List[Dict[str, Any]]:
        """Detect large liquidity walls in the order book"""
        try:
            walls = []
            
            if not order_book['bids'] or not order_book['asks']:
                return walls
            
            # Calculate average volume for comparison
            all_volumes = [bid[1] for bid in order_book['bids']] + [ask[1] for ask in order_book['asks']]
            avg_volume = np.mean(all_volumes) if all_volumes else 0
            std_volume = np.std(all_volumes) if len(all_volumes) > 1 else 0
            
            # Detect bid walls (large buy orders)
            for i, bid in enumerate(order_book['bids']):
                price, volume = bid
                
                # Check if volume is significantly above average (2+ standard deviations)
                if volume > avg_volume + 2 * std_volume:
                    # Check if adjacent levels also have high volume (wall thickness)
                    thickness = 1
                    for j in range(i + 1, min(i + 5, len(order_book['bids']))):
                        if order_book['bids'][j][1] > avg_volume + std_volume:
                            thickness += 1
                        else:
                            break
                    
                    wall = {
                        'price_level': price,
                        'volume': volume,
                        'side': 'bid',
                        'confidence': min(volume / (avg_volume + 2 * std_volume), 1.0),
                        'strength': min(volume / avg_volume, 3.0) / 3.0,
                        'distance_from_mid': 0.0,  # Will be calculated later
                        'thickness': thickness,
                        'type': 'liquidity_wall'
                    }
                    walls.append(wall)
            
            # Detect ask walls (large sell orders)
            for i, ask in enumerate(order_book['asks']):
                price, volume = ask
                
                # Check if volume is significantly above average
                if volume > avg_volume + 2 * std_volume:
                    # Check wall thickness
                    thickness = 1
                    for j in range(i + 1, min(i + 5, len(order_book['asks']))):
                        if order_book['asks'][j][1] > avg_volume + std_volume:
                            thickness += 1
                        else:
                            break
                    
                    wall = {
                        'price_level': price,
                        'volume': volume,
                        'side': 'ask',
                        'confidence': min(volume / (avg_volume + 2 * std_volume), 1.0),
                        'strength': min(volume / avg_volume, 3.0) / 3.0,
                        'distance_from_mid': 0.0,  # Will be calculated later
                        'thickness': thickness,
                        'type': 'liquidity_wall'
                    }
                    walls.append(wall)
            
            # Calculate distance from mid price
            if order_book['bids'] and order_book['asks']:
                mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
                for wall in walls:
                    wall['distance_from_mid'] = abs(wall['price_level'] - mid_price) / mid_price
            
            return walls
            
        except Exception as e:
            logger.error(f"Error detecting liquidity walls: {e}")
            return []
    
    async def detect_order_clusters(self, order_book: Dict[str, List]) -> List[Dict[str, Any]]:
        """Detect order clusters (groups of orders at similar price levels)"""
        try:
            clusters = []
            
            if not order_book['bids'] or not order_book['asks']:
                return clusters
            
            # Calculate average volume and price spacing
            all_volumes = [bid[1] for bid in order_book['bids']] + [ask[1] for ask in order_book['asks']]
            avg_volume = np.mean(all_volumes) if all_volumes else 0
            
            # Detect bid clusters
            bid_clusters = self._find_price_clusters(order_book['bids'], avg_volume)
            for cluster in bid_clusters:
                cluster['side'] = 'bid'
                clusters.append(cluster)
            
            # Detect ask clusters
            ask_clusters = self._find_price_clusters(order_book['asks'], avg_volume)
            for cluster in ask_clusters:
                cluster['side'] = 'ask'
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting order clusters: {e}")
            return []
    
    def _find_price_clusters(self, orders: List[List[float]], avg_volume: float) -> List[Dict[str, Any]]:
        """Find clusters of orders at similar price levels"""
        clusters = []
        
        if len(orders) < 2:
            return clusters
        
        # Group orders by price proximity (within 0.1% of each other)
        current_cluster = []
        cluster_start_idx = 0
        
        for i, (price, volume) in enumerate(orders):
            if not current_cluster:
                current_cluster = [(price, volume)]
                cluster_start_idx = i
            else:
                # Check if price is close to cluster
                cluster_avg_price = np.mean([p for p, _ in current_cluster])
                price_diff = abs(price - cluster_avg_price) / cluster_avg_price
                
                if price_diff <= 0.001:  # Within 0.1%
                    current_cluster.append((price, volume))
                else:
                    # End current cluster and start new one
                    if len(current_cluster) >= 2:  # Only keep clusters with 2+ orders
                        cluster_data = self._analyze_cluster(current_cluster, cluster_start_idx, avg_volume)
                        clusters.append(cluster_data)
                    
                    current_cluster = [(price, volume)]
                    cluster_start_idx = i
        
        # Handle last cluster
        if len(current_cluster) >= 2:
            cluster_data = self._analyze_cluster(current_cluster, cluster_start_idx, avg_volume)
            clusters.append(cluster_data)
        
        return clusters
    
    def _analyze_cluster(self, cluster: List[Tuple[float, float]], start_idx: int, avg_volume: float) -> Dict[str, Any]:
        """Analyze a cluster of orders"""
        prices = [price for price, _ in cluster]
        volumes = [volume for _, volume in cluster]
        
        avg_price = np.mean(prices)
        total_volume = sum(volumes)
        thickness = len(cluster)
        
        return {
            'price_level': avg_price,
            'volume': total_volume,
            'confidence': min(total_volume / (avg_volume * thickness), 1.0),
            'strength': min(total_volume / avg_volume, 2.0) / 2.0,
            'distance_from_mid': 0.0,  # Will be calculated later
            'thickness': thickness,
            'type': 'order_cluster',
            'start_index': start_idx,
            'price_range': (min(prices), max(prices))
        }
    
    async def calculate_depth_pressure(self, order_book: Dict[str, List]) -> float:
        """Calculate depth pressure (-1 to 1, negative = downward pressure)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate weighted volume within 1% of mid price
            mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            depth_threshold = mid_price * 0.01
            
            bid_pressure = 0
            ask_pressure = 0
            
            # Calculate bid pressure (weighted by distance from mid)
            for price, volume in order_book['bids']:
                if price >= mid_price - depth_threshold:
                    distance = (mid_price - price) / mid_price
                    weight = 1 - distance  # Closer to mid = higher weight
                    bid_pressure += volume * weight
            
            # Calculate ask pressure (weighted by distance from mid)
            for price, volume in order_book['asks']:
                if price <= mid_price + depth_threshold:
                    distance = (price - mid_price) / mid_price
                    weight = 1 - distance  # Closer to mid = higher weight
                    ask_pressure += volume * weight
            
            # Calculate pressure ratio
            total_pressure = bid_pressure + ask_pressure
            if total_pressure == 0:
                return 0.0
            
            pressure_ratio = (bid_pressure - ask_pressure) / total_pressure
            
            return max(min(pressure_ratio, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating depth pressure: {e}")
            return 0.0
    
    async def calculate_weighted_imbalance(self, order_book: Dict[str, List]) -> float:
        """Calculate weighted bid/ask imbalance (-1 to 1)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # Calculate weighted volumes (closer to mid price = higher weight)
            mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
            
            weighted_bid_volume = 0
            weighted_ask_volume = 0
            
            # Weight bid volumes
            for price, volume in order_book['bids']:
                distance = (mid_price - price) / mid_price
                weight = 1 - distance  # Closer to mid = higher weight
                weighted_bid_volume += volume * weight
            
            # Weight ask volumes
            for price, volume in order_book['asks']:
                distance = (price - mid_price) / mid_price
                weight = 1 - distance  # Closer to mid = higher weight
                weighted_ask_volume += volume * weight
            
            # Calculate weighted imbalance
            total_weighted_volume = weighted_bid_volume + weighted_ask_volume
            if total_weighted_volume == 0:
                return 0.0
            
            imbalance = (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume
            
            return max(min(imbalance, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating weighted imbalance: {e}")
            return 0.0
    
    async def calculate_order_flow_toxicity(self, order_book: Dict[str, List]) -> float:
        """Calculate order flow toxicity (-1 to 1, negative = aggressive selling)"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return 0.0
            
            # This is a simplified version - in practice, you'd need trade data
            # For now, we'll use order book imbalance as a proxy
            imbalance = await self.calculate_weighted_imbalance(order_book)
            
            # Convert imbalance to toxicity (negative imbalance = aggressive selling)
            toxicity = -imbalance
            
            return max(min(toxicity, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating order flow toxicity: {e}")
            return 0.0
    
    async def analyze_spread(self, order_book: Dict[str, List]) -> Dict[str, float]:
        """Analyze bid/ask spread metrics"""
        try:
            if not order_book['bids'] or not order_book['asks']:
                return {'spread': 0.0, 'spread_percentage': 0.0, 'spread_quality': 0.0}
            
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100
            
            # Calculate spread quality (lower spread = higher quality)
            spread_quality = max(0, 1 - spread_percentage * 10)
            
            return {
                'spread': spread,
                'spread_percentage': spread_percentage,
                'spread_quality': spread_quality
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spread: {e}")
            return {'spread': 0.0, 'spread_percentage': 0.0, 'spread_quality': 0.0}
    
    # ==================== ANALYSIS GENERATION METHODS ====================
    
    async def generate_liquidity_analysis(self, symbol: str, liquidity_walls: List[Dict], order_clusters: List[Dict]) -> str:
        """Generate detailed liquidity analysis text"""
        try:
            analysis_parts = []
            
            # Analyze liquidity walls
            if liquidity_walls:
                bid_walls = [w for w in liquidity_walls if w['side'] == 'bid']
                ask_walls = [w for w in liquidity_walls if w['side'] == 'ask']
                
                if bid_walls:
                    strongest_bid_wall = max(bid_walls, key=lambda x: x['strength'])
                    analysis_parts.append(f"Strong bid wall at {strongest_bid_wall['price_level']:.2f} with {strongest_bid_wall['volume']:.2f} volume")
                
                if ask_walls:
                    strongest_ask_wall = max(ask_walls, key=lambda x: x['strength'])
                    analysis_parts.append(f"Strong ask wall at {strongest_ask_wall['price_level']:.2f} with {strongest_ask_wall['volume']:.2f} volume")
            else:
                analysis_parts.append("No significant liquidity walls detected")
            
            # Analyze order clusters
            if order_clusters:
                analysis_parts.append(f"Detected {len(order_clusters)} order clusters")
            
            return ". ".join(analysis_parts) if analysis_parts else "Liquidity analysis unavailable"
            
        except Exception as e:
            logger.error(f"Error generating liquidity analysis: {e}")
            return "Liquidity analysis unavailable"
    
    async def generate_order_book_analysis(self, symbol: str, imbalance: float, toxicity: float) -> str:
        """Generate detailed order book analysis text"""
        try:
            analysis_parts = []
            
            # Analyze imbalance
            if imbalance > 0.1:
                analysis_parts.append(f"Bullish order book imbalance ({imbalance:.3f})")
            elif imbalance < -0.1:
                analysis_parts.append(f"Bearish order book imbalance ({imbalance:.3f})")
            else:
                analysis_parts.append("Balanced order book")
            
            # Analyze toxicity
            if toxicity > 0.1:
                analysis_parts.append("Aggressive buying pressure")
            elif toxicity < -0.1:
                analysis_parts.append("Aggressive selling pressure")
            else:
                analysis_parts.append("Neutral order flow")
            
            return ". ".join(analysis_parts) if analysis_parts else "Order book analysis unavailable"
            
        except Exception as e:
            logger.error(f"Error generating order book analysis: {e}")
            return "Order book analysis unavailable"
    
    # ==================== DEFAULT ANALYSIS METHODS ====================
    
    def _get_default_liquidity_analysis(self, symbol: str) -> LiquidityAnalysis:
        """Get default liquidity analysis when analysis fails"""
        return LiquidityAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            liquidity_score=0.5,
            bid_liquidity=0.5,
            ask_liquidity=0.5,
            liquidity_walls=[],
            order_clusters=[],
            depth_pressure=0.0,
            spread_analysis={'spread': 0.0, 'spread_percentage': 0.0, 'spread_quality': 0.0},
            liquidity_analysis="Liquidity analysis unavailable"
        )
    
    def _get_default_order_book_analysis(self, symbol: str) -> OrderBookAnalysis:
        """Get default order book analysis when analysis fails"""
        return OrderBookAnalysis(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            bid_ask_imbalance=0.0,
            order_flow_toxicity=0.0,
            depth_pressure=0.0,
            liquidity_walls=[],
            order_clusters=[],
            spread_analysis={'spread': 0.0, 'spread_percentage': 0.0, 'spread_quality': 0.0},
            order_book_analysis="Order book analysis unavailable"
        )
    
    # ML Feature Engineering Methods
    
    async def engineer_liquidation_features(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Engineer features specifically for liquidation prediction"""
        try:
            features = []
            
            # Get current analysis
            volume_analysis = await self.analyze_volume_positioning(symbol)
            liquidity_analysis = await self.analyze_liquidity(symbol)
            order_book_analysis = await self.analyze_order_book(symbol)
            
            # Core liquidation features
            features.extend([
                order_book_analysis.order_flow_toxicity,  # Feature 0: Order flow toxicity
                order_book_analysis.bid_ask_imbalance,    # Feature 1: Bid-ask imbalance
                volume_analysis.volume_ratio,             # Feature 2: Volume ratio
                liquidity_analysis.depth_pressure,       # Feature 3: Depth pressure
                order_book_analysis.depth_pressure,      # Feature 4: Order book depth pressure
            ])
            
            # Liquidity features
            features.extend([
                liquidity_analysis.liquidity_score,      # Feature 5: Overall liquidity score
                liquidity_analysis.bid_liquidity,        # Feature 6: Bid liquidity
                liquidity_analysis.ask_liquidity,        # Feature 7: Ask liquidity
                len(liquidity_analysis.liquidity_walls) / 10.0,  # Feature 8: Liquidity walls count (normalized)
                len(liquidity_analysis.order_clusters) / 10.0,   # Feature 9: Order clusters count (normalized)
            ])
            
            # Volume-based features
            features.extend([
                volume_analysis.buy_volume_ratio,        # Feature 10: Buy volume ratio
                volume_analysis.sell_volume_ratio,       # Feature 11: Sell volume ratio
                float(volume_analysis.volume_breakout),  # Feature 12: Volume breakout indicator
                volume_analysis.volume_positioning_score, # Feature 13: Volume positioning score
            ])
            
            # Spread and market microstructure features
            spread_metrics = liquidity_analysis.spread_analysis
            features.extend([
                spread_metrics.get('spread', 0.0),           # Feature 14: Absolute spread
                spread_metrics.get('spread_percentage', 0.0), # Feature 15: Spread percentage
                spread_metrics.get('spread_quality', 0.0),    # Feature 16: Spread quality
            ])
            
            # Market data features (if available)
            if market_data:
                features.extend([
                    market_data.get('historical_volatility', 0.0),     # Feature 17: Historical volatility
                    market_data.get('price_change_24h', 0.0) / 100.0,  # Feature 18: 24h price change (normalized)
                    market_data.get('volume_24h', 0.0) / 1000000.0,    # Feature 19: 24h volume (normalized)
                    market_data.get('recent_liquidation_volume', 0.0) / 1000.0,  # Feature 20: Recent liquidation volume
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # Default values
            
            # Advanced derived features
            features.extend([
                # Feature 21: Imbalance-volatility interaction
                abs(order_book_analysis.bid_ask_imbalance) * features[17] if len(features) > 17 else 0.0,
                # Feature 22: Volume-toxicity interaction
                volume_analysis.volume_ratio * order_book_analysis.order_flow_toxicity,
                # Feature 23: Liquidity-depth pressure interaction
                liquidity_analysis.liquidity_score * liquidity_analysis.depth_pressure,
                # Feature 24: Spread-volume interaction
                spread_metrics.get('spread_percentage', 0.0) * volume_analysis.volume_ratio,
            ])
            
            # Ensure all features are finite and within reasonable bounds
            features = [max(-10.0, min(10.0, f)) if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering liquidation features for {symbol}: {e}")
            # Return default feature vector
            return [0.0] * 25
    
    async def engineer_order_book_features(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Engineer features for order book forecasting"""
        try:
            features = []
            
            # Get current analysis
            order_book_analysis = await self.analyze_order_book(symbol)
            liquidity_analysis = await self.analyze_liquidity(symbol)
            volume_analysis = await self.analyze_volume_positioning(symbol)
            
            # Order book structure features
            features.extend([
                order_book_analysis.bid_ask_imbalance,    # Feature 0: Current imbalance
                order_book_analysis.order_flow_toxicity,  # Feature 1: Order flow toxicity
                order_book_analysis.depth_pressure,      # Feature 2: Depth pressure
            ])
            
            # Liquidity features
            features.extend([
                liquidity_analysis.bid_liquidity,        # Feature 3: Bid side liquidity
                liquidity_analysis.ask_liquidity,        # Feature 4: Ask side liquidity
                liquidity_analysis.liquidity_score,      # Feature 5: Overall liquidity
                liquidity_analysis.depth_pressure,       # Feature 6: Liquidity depth pressure
            ])
            
            # Spread dynamics
            spread_metrics = liquidity_analysis.spread_analysis
            features.extend([
                spread_metrics.get('spread', 0.0),           # Feature 7: Current spread
                spread_metrics.get('spread_percentage', 0.0), # Feature 8: Spread percentage
                spread_metrics.get('spread_quality', 0.0),    # Feature 9: Spread quality
            ])
            
            # Volume characteristics
            features.extend([
                volume_analysis.volume_ratio,             # Feature 10: Volume ratio
                volume_analysis.buy_volume_ratio,         # Feature 11: Buy volume ratio
                volume_analysis.sell_volume_ratio,        # Feature 12: Sell volume ratio
                volume_analysis.volume_positioning_score, # Feature 13: Volume positioning
            ])
            
            # Market microstructure features
            liquidity_walls_strength = sum(wall.get('strength', 0.0) for wall in liquidity_analysis.liquidity_walls[:5])
            order_clusters_volume = sum(cluster.get('total_volume', 0.0) for cluster in liquidity_analysis.order_clusters[:5])
            
            features.extend([
                len(liquidity_analysis.liquidity_walls) / 10.0,  # Feature 14: Liquidity walls count
                liquidity_walls_strength / 10.0,                 # Feature 15: Liquidity walls strength
                len(liquidity_analysis.order_clusters) / 10.0,   # Feature 16: Order clusters count
                order_clusters_volume / 1000.0,                  # Feature 17: Order clusters volume
            ])
            
            # Derived interaction features
            features.extend([
                # Feature 18: Imbalance-liquidity interaction
                order_book_analysis.bid_ask_imbalance * liquidity_analysis.liquidity_score,
                # Feature 19: Toxicity-volume interaction
                order_book_analysis.order_flow_toxicity * volume_analysis.volume_ratio,
                # Feature 20: Spread-depth interaction
                spread_metrics.get('spread_percentage', 0.0) * order_book_analysis.depth_pressure,
            ])
            
            # Ensure all features are finite and bounded
            features = [max(-5.0, min(5.0, f)) if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering order book features for {symbol}: {e}")
            # Return default feature vector
            return [0.0] * 21
    
    async def engineer_market_microstructure_features(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Engineer features for market microstructure analysis"""
        try:
            features = []
            
            # Get comprehensive analysis
            volume_analysis = await self.analyze_volume_positioning(symbol)
            liquidity_analysis = await self.analyze_liquidity(symbol)
            order_book_analysis = await self.analyze_order_book(symbol)
            
            # Core microstructure features
            features.extend([
                order_book_analysis.order_flow_toxicity,  # Feature 0: Order flow toxicity
                order_book_analysis.bid_ask_imbalance,    # Feature 1: Price discovery efficiency
                volume_analysis.volume_positioning_score, # Feature 2: Volume positioning quality
                liquidity_analysis.liquidity_score,      # Feature 3: Market liquidity
            ])
            
            # Information flow features
            features.extend([
                order_book_analysis.depth_pressure,      # Feature 4: Information pressure
                volume_analysis.volume_ratio,            # Feature 5: Volume information content
                liquidity_analysis.depth_pressure,       # Feature 6: Liquidity information
            ])
            
            # Market efficiency indicators
            spread_metrics = liquidity_analysis.spread_analysis
            features.extend([
                spread_metrics.get('spread_percentage', 0.0),     # Feature 7: Efficiency via spread
                volume_analysis.buy_volume_ratio - 0.5,          # Feature 8: Flow imbalance
                abs(order_book_analysis.bid_ask_imbalance),      # Feature 9: Price discovery quality
            ])
            
            # Market resilience features
            liquidity_walls_count = len(liquidity_analysis.liquidity_walls)
            order_clusters_count = len(liquidity_analysis.order_clusters)
            
            features.extend([
                liquidity_walls_count / 10.0,            # Feature 10: Structural resilience
                order_clusters_count / 10.0,             # Feature 11: Order flow resilience
                liquidity_analysis.bid_liquidity + liquidity_analysis.ask_liquidity,  # Feature 12: Total resilience
            ])
            
            # Advanced microstructure indicators
            features.extend([
                # Feature 13: Information asymmetry proxy
                order_book_analysis.order_flow_toxicity * abs(order_book_analysis.bid_ask_imbalance),
                # Feature 14: Market impact efficiency
                volume_analysis.volume_ratio * (1.0 - spread_metrics.get('spread_percentage', 0.0)),
                # Feature 15: Liquidity efficiency
                liquidity_analysis.liquidity_score * (1.0 / (1.0 + spread_metrics.get('spread_percentage', 0.01))),
            ])
            
            # Ensure all features are finite and bounded
            features = [max(-3.0, min(3.0, f)) if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering microstructure features for {symbol}: {e}")
            # Return default feature vector
            return [0.0] * 16
    
    def calculate_feature_importance_scores(self, features: List[float], analysis_type: str = 'liquidation') -> Dict[str, float]:
        """Calculate feature importance scores for interpretability"""
        try:
            if analysis_type == 'liquidation':
                feature_names = [
                    'order_flow_toxicity', 'bid_ask_imbalance', 'volume_ratio', 'depth_pressure', 'ob_depth_pressure',
                    'liquidity_score', 'bid_liquidity', 'ask_liquidity', 'liquidity_walls_count', 'order_clusters_count',
                    'buy_volume_ratio', 'sell_volume_ratio', 'volume_breakout', 'volume_positioning_score',
                    'spread', 'spread_percentage', 'spread_quality', 'volatility', 'price_change_24h', 'volume_24h',
                    'recent_liquidation_volume', 'imbalance_volatility', 'volume_toxicity', 'liquidity_depth', 'spread_volume'
                ]
            elif analysis_type == 'order_book':
                feature_names = [
                    'bid_ask_imbalance', 'order_flow_toxicity', 'depth_pressure', 'bid_liquidity', 'ask_liquidity',
                    'liquidity_score', 'liq_depth_pressure', 'spread', 'spread_percentage', 'spread_quality',
                    'volume_ratio', 'buy_volume_ratio', 'sell_volume_ratio', 'volume_positioning',
                    'liquidity_walls_count', 'liquidity_walls_strength', 'order_clusters_count', 'order_clusters_volume',
                    'imbalance_liquidity', 'toxicity_volume', 'spread_depth'
                ]
            else:  # microstructure
                feature_names = [
                    'order_flow_toxicity', 'bid_ask_imbalance', 'volume_positioning_score', 'liquidity_score',
                    'depth_pressure', 'volume_ratio', 'liq_depth_pressure', 'spread_percentage', 'flow_imbalance',
                    'price_discovery_quality', 'structural_resilience', 'order_flow_resilience', 'total_resilience',
                    'information_asymmetry', 'market_impact_efficiency', 'liquidity_efficiency'
                ]
            
            # Calculate relative importance based on feature magnitudes and distributions
            feature_scores = {}
            total_magnitude = sum(abs(f) for f in features)
            
            for i, name in enumerate(feature_names[:len(features)]):
                if total_magnitude > 0:
                    importance = abs(features[i]) / total_magnitude
                else:
                    importance = 1.0 / len(features)
                feature_scores[name] = importance
            
            return feature_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
