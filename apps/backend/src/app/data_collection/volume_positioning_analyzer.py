"""
Volume Positioning Analyzer for AlphaPulse
Analyzes volume patterns, order book imbalances, and volume positioning
"""

import asyncio
import logging
import asyncpg
import ccxt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VolumeAnalysis:
    """Volume analysis data structure"""
    timestamp: datetime
    symbol: str
    volume_ratio: float
    volume_trend: str
    order_book_imbalance: float
    volume_positioning_score: float
    buy_volume_ratio: float
    sell_volume_ratio: float
    volume_breakout: bool
    volume_analysis: str

class VolumePositioningAnalyzer:
    """
    Analyzes volume patterns and positioning for trading signals
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        self.cache = {}
        self.cache_duration = 60  # 1 minute
        
        logger.info("Volume Positioning Analyzer initialized")
    
    async def analyze_volume_positioning(self, symbol: str, timeframe: str = '1h') -> VolumeAnalysis:
        """Analyze volume positioning for a symbol"""
        try:
            logger.info(f"🔄 Analyzing volume positioning for {symbol}")
            
            # Get recent OHLCV data
            ohlcv_data = self._get_ohlcv_data(symbol)
            
            # Calculate volume metrics
            volume_ratio = await self._calculate_volume_ratio(ohlcv_data)
            volume_trend = await self._determine_volume_trend(ohlcv_data)
            
            # Get order book data
            order_book = self._get_order_book(symbol)
            order_book_imbalance = await self._calculate_order_book_imbalance(order_book)
            
            # Calculate volume positioning score
            volume_positioning_score = await self._calculate_volume_positioning_score(
                volume_ratio, volume_trend, order_book_imbalance
            )
            
            # Calculate buy/sell volume ratios
            buy_volume_ratio, sell_volume_ratio = await self._calculate_volume_ratios(ohlcv_data)
            
            # Detect volume breakouts
            volume_breakout = await self._detect_volume_breakout(ohlcv_data)
            
            # Generate analysis text
            volume_analysis = await self._generate_volume_analysis(
                volume_ratio, volume_trend, order_book_imbalance, 
                volume_positioning_score, buy_volume_ratio, sell_volume_ratio, volume_breakout
            )
            
            # Create volume analysis data
            analysis_data = VolumeAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                order_book_imbalance=order_book_imbalance,
                volume_positioning_score=volume_positioning_score,
                buy_volume_ratio=buy_volume_ratio,
                sell_volume_ratio=sell_volume_ratio,
                volume_breakout=volume_breakout,
                volume_analysis=volume_analysis
            )
            
            logger.info(f"✅ Volume analysis completed for {symbol}")
            return analysis_data
            
        except Exception as e:
            logger.error(f"❌ Error analyzing volume positioning for {symbol}: {e}")
            # Return default analysis on error
            return VolumeAnalysis(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                volume_ratio=1.0,
                volume_trend='neutral',
                order_book_imbalance=0.0,
                volume_positioning_score=0.5,
                buy_volume_ratio=0.5,
                sell_volume_ratio=0.5,
                volume_breakout=False,
                volume_analysis="Volume analysis unavailable due to error"
            )
    
    def _get_ohlcv_data(self, symbol: str, limit: int = 100) -> List[List]:
        """Get OHLCV data for volume analysis"""
        try:
            cache_key = f'ohlcv_{symbol}'
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_duration:
                    return cached_data['data']
            
            # Fetch OHLCV data from exchange (synchronous)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=limit)
            
            # Cache the result
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': ohlcv
            }
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {e}")
            return []
    
    def _get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book data for imbalance analysis"""
        try:
            cache_key = f'orderbook_{symbol}'
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_duration:
                    return cached_data['data']
            
            # Fetch order book from exchange (synchronous)
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # Cache the result
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': order_book
            }
            
            return order_book
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    async def _calculate_volume_ratio(self, ohlcv_data: List[List]) -> float:
        """Calculate current volume ratio compared to average"""
        try:
            if len(ohlcv_data) < 20:
                return 1.0
            
            # Get recent volumes (last 5 periods)
            recent_volumes = [candle[5] for candle in ohlcv_data[-5:]]
            current_volume = recent_volumes[-1]
            
            # Calculate average volume (last 20 periods)
            avg_volume = sum([candle[5] for candle in ohlcv_data[-20:]]) / 20
            
            if avg_volume == 0:
                return 1.0
            
            volume_ratio = current_volume / avg_volume
            return max(0.1, min(10.0, volume_ratio))  # Clamp between 0.1 and 10
            
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    async def _determine_volume_trend(self, ohlcv_data: List[List]) -> str:
        """Determine volume trend direction"""
        try:
            if len(ohlcv_data) < 10:
                return 'neutral'
            
            # Get recent volumes
            recent_volumes = [candle[5] for candle in ohlcv_data[-10:]]
            
            # Calculate volume trend using simple moving average
            first_half_avg = sum(recent_volumes[:5]) / 5
            second_half_avg = sum(recent_volumes[5:]) / 5
            
            if second_half_avg > first_half_avg * 1.1:
                return 'increasing'
            elif second_half_avg < first_half_avg * 0.9:
                return 'decreasing'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining volume trend: {e}")
            return 'neutral'
    
    async def _calculate_order_book_imbalance(self, order_book: Dict[str, Any]) -> float:
        """Calculate order book imbalance (-1 to 1)"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            # Calculate total bid and ask volume
            bid_volume = sum(bid[1] for bid in bids)
            ask_volume = sum(ask[1] for ask in asks)
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            
            # Calculate imbalance (-1 to 1)
            imbalance = (bid_volume - ask_volume) / total_volume
            return max(-1.0, min(1.0, imbalance))
            
        except Exception as e:
            logger.error(f"Error calculating order book imbalance: {e}")
            return 0.0
    
    async def _calculate_volume_positioning_score(self, volume_ratio: float, volume_trend: str, order_book_imbalance: float) -> float:
        """Calculate overall volume positioning score (0-1)"""
        try:
            # Volume ratio component (0-1)
            volume_ratio_score = min(1.0, volume_ratio / 3.0)  # Normalize to 0-1
            
            # Volume trend component (0-1)
            trend_score = 0.5  # neutral
            if volume_trend == 'increasing':
                trend_score = 0.8
            elif volume_trend == 'decreasing':
                trend_score = 0.2
            
            # Order book imbalance component (0-1)
            imbalance_score = (order_book_imbalance + 1) / 2  # Convert -1,1 to 0,1
            
            # Weighted average
            positioning_score = (
                volume_ratio_score * 0.4 +
                trend_score * 0.3 +
                imbalance_score * 0.3
            )
            
            return max(0.0, min(1.0, positioning_score))
            
        except Exception as e:
            logger.error(f"Error calculating volume positioning score: {e}")
            return 0.5
    
    async def _calculate_volume_ratios(self, ohlcv_data: List[List]) -> tuple[float, float]:
        """Calculate buy vs sell volume ratios"""
        try:
            if len(ohlcv_data) < 10:
                return 0.5, 0.5
            
            # Simple heuristic: if close > open, consider it buy volume
            buy_volume = 0
            sell_volume = 0
            
            for candle in ohlcv_data[-10:]:
                open_price = candle[1]
                close_price = candle[4]
                volume = candle[5]
                
                if close_price > open_price:
                    buy_volume += volume
                else:
                    sell_volume += volume
            
            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return 0.5, 0.5
            
            buy_ratio = buy_volume / total_volume
            sell_ratio = sell_volume / total_volume
            
            return buy_ratio, sell_ratio
            
        except Exception as e:
            logger.error(f"Error calculating volume ratios: {e}")
            return 0.5, 0.5
    
    async def _detect_volume_breakout(self, ohlcv_data: List[List]) -> bool:
        """Detect volume breakout patterns"""
        try:
            if len(ohlcv_data) < 20:
                return False
            
            # Get recent volumes
            recent_volumes = [candle[5] for candle in ohlcv_data[-5:]]
            historical_volumes = [candle[5] for candle in ohlcv_data[-20:-5]]
            
            if not historical_volumes:
                return False
            
            # Calculate average historical volume
            avg_historical = sum(historical_volumes) / len(historical_volumes)
            
            # Check if recent volumes are significantly higher
            recent_avg = sum(recent_volumes) / len(recent_volumes)
            
            # Volume breakout if recent average is 2x historical average
            return recent_avg > avg_historical * 2
            
        except Exception as e:
            logger.error(f"Error detecting volume breakout: {e}")
            return False
    
    async def _generate_volume_analysis(self, volume_ratio: float, volume_trend: str, 
                                      order_book_imbalance: float, volume_positioning_score: float,
                                      buy_volume_ratio: float, sell_volume_ratio: float, 
                                      volume_breakout: bool) -> str:
        """Generate human-readable volume analysis"""
        try:
            analysis_parts = []
            
            # Volume ratio analysis
            if volume_ratio > 2.0:
                analysis_parts.append("High volume activity")
            elif volume_ratio > 1.5:
                analysis_parts.append("Above average volume")
            elif volume_ratio < 0.5:
                analysis_parts.append("Low volume activity")
            
            # Volume trend analysis
            if volume_trend == 'increasing':
                analysis_parts.append("Volume trending up")
            elif volume_trend == 'decreasing':
                analysis_parts.append("Volume trending down")
            
            # Order book analysis
            if order_book_imbalance > 0.3:
                analysis_parts.append("Strong buying pressure")
            elif order_book_imbalance < -0.3:
                analysis_parts.append("Strong selling pressure")
            
            # Volume positioning analysis
            if volume_positioning_score > 0.7:
                analysis_parts.append("Bullish volume positioning")
            elif volume_positioning_score < 0.3:
                analysis_parts.append("Bearish volume positioning")
            
            # Buy/sell ratio analysis
            if buy_volume_ratio > 0.6:
                analysis_parts.append("Dominant buying volume")
            elif sell_volume_ratio > 0.6:
                analysis_parts.append("Dominant selling volume")
            
            # Volume breakout analysis
            if volume_breakout:
                analysis_parts.append("Volume breakout detected")
            
            if not analysis_parts:
                analysis_parts.append("Neutral volume conditions")
            
            return "; ".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating volume analysis: {e}")
            return "Volume analysis unavailable"
    
    async def store_volume_analysis(self, data: VolumeAnalysis, timeframe: str = '1h') -> bool:
        """Store volume analysis data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO volume_analysis (
                        timestamp, symbol, volume_ratio, volume_trend, order_book_imbalance,
                        volume_positioning_score, buy_volume_ratio, sell_volume_ratio,
                        volume_breakout, volume_analysis
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 
                data.timestamp, data.symbol, data.volume_ratio, data.volume_trend,
                data.order_book_imbalance, data.volume_positioning_score,
                data.buy_volume_ratio, data.sell_volume_ratio,
                data.volume_breakout, data.volume_analysis
                )
            
            logger.info(f"✅ Volume analysis stored for {data.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing volume analysis: {e}")
            return False
    
    async def get_latest_volume_analysis(self, symbol: str) -> Optional[VolumeAnalysis]:
        """Get latest volume analysis for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM volume_analysis 
                    WHERE symbol = $1
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, symbol)
                
                if row:
                    return VolumeAnalysis(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        volume_ratio=float(row['volume_ratio']),
                        volume_trend=row['volume_trend'],
                        order_book_imbalance=float(row['order_book_imbalance']),
                        volume_positioning_score=float(row['volume_positioning_score']),
                        buy_volume_ratio=float(row['buy_volume_ratio']),
                        sell_volume_ratio=float(row['sell_volume_ratio']),
                        volume_breakout=row['volume_breakout'],
                        volume_analysis=row['volume_analysis']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting latest volume analysis: {e}")
            return None
