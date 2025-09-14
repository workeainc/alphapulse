"""
Free API Database Integration Service
Handles storing and retrieving free API data for signal generation
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FreeAPIMarketData:
    """Market data from free APIs"""
    symbol: str
    timestamp: datetime
    source: str
    price: float
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_change_24h: Optional[float] = None
    market_cap_change_24h: Optional[float] = None
    fear_greed_index: Optional[int] = None
    liquidation_events: Optional[Dict] = None
    raw_data: Optional[Dict] = None
    data_quality_score: float = 1.0

@dataclass
class FreeAPISentimentData:
    """Sentiment data from free APIs"""
    symbol: str
    timestamp: datetime
    source: str
    sentiment_type: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    volume: Optional[int] = None
    keywords: Optional[Dict] = None
    raw_data: Optional[Dict] = None
    data_quality_score: float = 1.0

@dataclass
class FreeAPINewsData:
    """News data from free APIs"""
    symbol: str
    timestamp: datetime
    source: str
    title: str
    content: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    relevance_score: Optional[float] = None
    keywords: Optional[Dict] = None
    raw_data: Optional[Dict] = None

@dataclass
class FreeAPISocialData:
    """Social media data from free APIs"""
    symbol: str
    timestamp: datetime
    platform: str
    content: str
    post_id: Optional[str] = None
    author: Optional[str] = None
    engagement_metrics: Optional[Dict] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    influence_score: Optional[float] = None
    keywords: Optional[Dict] = None
    raw_data: Optional[Dict] = None

@dataclass
class FreeAPILiquidationEvent:
    """Liquidation event data from free APIs"""
    symbol: str
    timestamp: datetime
    source: str
    liquidation_type: str
    price: float
    quantity: float
    value_usd: float
    side: str
    raw_data: Optional[Dict] = None

class FreeAPIDatabaseService:
    """Service for managing free API data in the database"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
    
    async def store_market_data(self, market_data: FreeAPIMarketData) -> bool:
        """Store market data from free APIs"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO free_api_market_data (
                        symbol, timestamp, source, price, volume_24h, market_cap,
                        price_change_24h, volume_change_24h, market_cap_change_24h,
                        fear_greed_index, liquidation_events, raw_data, data_quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT DO NOTHING
                """
                
                await conn.execute(
                    query,
                    market_data.symbol,
                    market_data.timestamp,
                    market_data.source,
                    market_data.price,
                    market_data.volume_24h,
                    market_data.market_cap,
                    market_data.price_change_24h,
                    market_data.volume_change_24h,
                    market_data.market_cap_change_24h,
                    market_data.fear_greed_index,
                    json.dumps(market_data.liquidation_events) if market_data.liquidation_events else None,
                    json.dumps(market_data.raw_data) if market_data.raw_data else None,
                    market_data.data_quality_score
                )
                
                self.logger.debug(f"✅ Stored market data: {market_data.symbol} from {market_data.source}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing market data: {e}")
            return False
    
    async def store_sentiment_data(self, sentiment_data: FreeAPISentimentData) -> bool:
        """Store sentiment data from free APIs"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO free_api_sentiment_data (
                        symbol, timestamp, source, sentiment_type, sentiment_score,
                        sentiment_label, confidence, volume, keywords, raw_data, data_quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT DO NOTHING
                """
                
                await conn.execute(
                    query,
                    sentiment_data.symbol,
                    sentiment_data.timestamp,
                    sentiment_data.source,
                    sentiment_data.sentiment_type,
                    sentiment_data.sentiment_score,
                    sentiment_data.sentiment_label,
                    sentiment_data.confidence,
                    sentiment_data.volume,
                    json.dumps(sentiment_data.keywords) if sentiment_data.keywords else None,
                    json.dumps(sentiment_data.raw_data) if sentiment_data.raw_data else None,
                    sentiment_data.data_quality_score
                )
                
                self.logger.debug(f"✅ Stored sentiment data: {sentiment_data.symbol} from {sentiment_data.source}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing sentiment data: {e}")
            return False
    
    async def store_news_data(self, news_data: FreeAPINewsData) -> bool:
        """Store news data from free APIs"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO free_api_news_data (
                        symbol, timestamp, source, title, content, url, published_at,
                        sentiment_score, sentiment_label, relevance_score, keywords, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT DO NOTHING
                """
                
                await conn.execute(
                    query,
                    news_data.symbol,
                    news_data.timestamp,
                    news_data.source,
                    news_data.title,
                    news_data.content,
                    news_data.url,
                    news_data.published_at,
                    news_data.sentiment_score,
                    news_data.sentiment_label,
                    news_data.relevance_score,
                    json.dumps(news_data.keywords) if news_data.keywords else None,
                    json.dumps(news_data.raw_data) if news_data.raw_data else None
                )
                
                self.logger.debug(f"✅ Stored news data: {news_data.symbol} from {news_data.source}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing news data: {e}")
            return False
    
    async def store_social_data(self, social_data: FreeAPISocialData) -> bool:
        """Store social media data from free APIs"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO free_api_social_data (
                        symbol, timestamp, platform, post_id, content, author,
                        engagement_metrics, sentiment_score, sentiment_label,
                        influence_score, keywords, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT DO NOTHING
                """
                
                await conn.execute(
                    query,
                    social_data.symbol,
                    social_data.timestamp,
                    social_data.platform,
                    social_data.post_id,
                    social_data.content,
                    social_data.author,
                    json.dumps(social_data.engagement_metrics) if social_data.engagement_metrics else None,
                    social_data.sentiment_score,
                    social_data.sentiment_label,
                    social_data.influence_score,
                    json.dumps(social_data.keywords) if social_data.keywords else None,
                    json.dumps(social_data.raw_data) if social_data.raw_data else None
                )
                
                self.logger.debug(f"✅ Stored social data: {social_data.symbol} from {social_data.platform}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing social data: {e}")
            return False
    
    async def store_liquidation_events(self, liquidation_events: List[FreeAPILiquidationEvent]) -> bool:
        """Store liquidation events from free APIs"""
        try:
            if not liquidation_events:
                return True
                
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO free_api_liquidation_events (
                        symbol, timestamp, source, liquidation_type, price,
                        quantity, value_usd, side, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT DO NOTHING
                """
                
                for event in liquidation_events:
                    await conn.execute(
                        query,
                        event.symbol,
                        event.timestamp,
                        event.source,
                        event.liquidation_type,
                        event.price,
                        event.quantity,
                        event.value_usd,
                        event.side,
                        json.dumps(event.raw_data) if event.raw_data else None
                    )
                
                self.logger.debug(f"✅ Stored {len(liquidation_events)} liquidation events")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing liquidation events: {e}")
            return False
    
    async def get_latest_market_data(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get latest market data for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT symbol, timestamp, source, price, volume_24h, market_cap,
                           price_change_24h, volume_change_24h, market_cap_change_24h,
                           fear_greed_index, liquidation_events, raw_data, data_quality_score
                    FROM free_api_market_data
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting market data: {e}")
            return []
    
    async def get_latest_sentiment_data(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get latest sentiment data for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT symbol, timestamp, source, sentiment_type, sentiment_score,
                           sentiment_label, confidence, volume, keywords, raw_data, data_quality_score
                    FROM free_api_sentiment_data
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting sentiment data: {e}")
            return []
    
    async def get_aggregated_sentiment(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get aggregated sentiment data for signal generation"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT 
                        sentiment_type,
                        AVG(sentiment_score) as avg_sentiment_score,
                        COUNT(*) as sentiment_count,
                        AVG(confidence) as avg_confidence,
                        MAX(timestamp) as last_updated
                    FROM free_api_sentiment_data
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY sentiment_type
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                
                aggregated = {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'sentiment_by_type': {},
                    'overall_sentiment': 0.0,
                    'overall_confidence': 0.0,
                    'total_sentiment_count': 0,
                    'last_updated': None
                }
                
                total_weighted_sentiment = 0.0
                total_weight = 0.0
                
                for row in rows:
                    sentiment_type = row['sentiment_type']
                    avg_sentiment = float(row['avg_sentiment_score'])
                    count = int(row['sentiment_count'])
                    avg_confidence = float(row['avg_confidence'])
                    
                    # Weight by confidence and count
                    weight = avg_confidence * count
                    total_weighted_sentiment += avg_sentiment * weight
                    total_weight += weight
                    
                    aggregated['sentiment_by_type'][sentiment_type] = {
                        'sentiment_score': avg_sentiment,
                        'count': count,
                        'confidence': avg_confidence,
                        'last_updated': row['last_updated']
                    }
                    
                    aggregated['total_sentiment_count'] += count
                    if not aggregated['last_updated'] or row['last_updated'] > aggregated['last_updated']:
                        aggregated['last_updated'] = row['last_updated']
                
                if total_weight > 0:
                    aggregated['overall_sentiment'] = total_weighted_sentiment / total_weight
                    aggregated['overall_confidence'] = total_weight / aggregated['total_sentiment_count']
                
                return aggregated
                
        except Exception as e:
            self.logger.error(f"❌ Error getting aggregated sentiment: {e}")
            return {
                'symbol': symbol,
                'timeframe_hours': hours,
                'sentiment_by_type': {},
                'overall_sentiment': 0.0,
                'overall_confidence': 0.0,
                'total_sentiment_count': 0,
                'last_updated': None
            }
    
    async def get_aggregated_market_data(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get aggregated market data for signal generation"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT 
                        source,
                        AVG(price) as avg_price,
                        AVG(volume_24h) as avg_volume_24h,
                        AVG(market_cap) as avg_market_cap,
                        AVG(price_change_24h) as avg_price_change_24h,
                        AVG(fear_greed_index) as avg_fear_greed_index,
                        COUNT(*) as data_points,
                        MAX(timestamp) as last_updated
                    FROM free_api_market_data
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY source
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                
                aggregated = {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'market_data_by_source': {},
                    'consensus_price': 0.0,
                    'consensus_volume': 0.0,
                    'consensus_market_cap': 0.0,
                    'consensus_price_change': 0.0,
                    'consensus_fear_greed': 0.0,
                    'total_data_points': 0,
                    'last_updated': None
                }
                
                prices = []
                volumes = []
                market_caps = []
                price_changes = []
                fear_greed_indices = []
                
                for row in rows:
                    source = row['source']
                    avg_price = float(row['avg_price'])
                    avg_volume = float(row['avg_volume_24h']) if row['avg_volume_24h'] else None
                    avg_market_cap = float(row['avg_market_cap']) if row['avg_market_cap'] else None
                    avg_price_change = float(row['avg_price_change_24h']) if row['avg_price_change_24h'] else None
                    avg_fear_greed = float(row['avg_fear_greed_index']) if row['avg_fear_greed_index'] else None
                    data_points = int(row['data_points'])
                    
                    aggregated['market_data_by_source'][source] = {
                        'price': avg_price,
                        'volume_24h': avg_volume,
                        'market_cap': avg_market_cap,
                        'price_change_24h': avg_price_change,
                        'fear_greed_index': avg_fear_greed,
                        'data_points': data_points,
                        'last_updated': row['last_updated']
                    }
                    
                    prices.append(avg_price)
                    if avg_volume:
                        volumes.append(avg_volume)
                    if avg_market_cap:
                        market_caps.append(avg_market_cap)
                    if avg_price_change:
                        price_changes.append(avg_price_change)
                    if avg_fear_greed:
                        fear_greed_indices.append(avg_fear_greed)
                    
                    aggregated['total_data_points'] += data_points
                    if not aggregated['last_updated'] or row['last_updated'] > aggregated['last_updated']:
                        aggregated['last_updated'] = row['last_updated']
                
                # Calculate consensus values
                if prices:
                    aggregated['consensus_price'] = np.mean(prices)
                if volumes:
                    aggregated['consensus_volume'] = np.mean(volumes)
                if market_caps:
                    aggregated['consensus_market_cap'] = np.mean(market_caps)
                if price_changes:
                    aggregated['consensus_price_change'] = np.mean(price_changes)
                if fear_greed_indices:
                    aggregated['consensus_fear_greed'] = np.mean(fear_greed_indices)
                
                return aggregated
                
        except Exception as e:
            self.logger.error(f"❌ Error getting aggregated market data: {e}")
            return {
                'symbol': symbol,
                'timeframe_hours': hours,
                'market_data_by_source': {},
                'consensus_price': 0.0,
                'consensus_volume': 0.0,
                'consensus_market_cap': 0.0,
                'consensus_price_change': 0.0,
                'consensus_fear_greed': 0.0,
                'total_data_points': 0,
                'last_updated': None
            }
    
    async def get_recent_liquidation_events(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get recent liquidation events for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT symbol, timestamp, source, liquidation_type, price,
                           quantity, value_usd, side, raw_data
                    FROM free_api_liquidation_events
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting liquidation events: {e}")
            return []
    
    async def update_data_quality_metrics(self, source: str, data_type: str, 
                                         availability_score: float, accuracy_score: float,
                                         completeness_score: float, timeliness_score: float,
                                         error_count: int = 0, success_count: int = 0,
                                         rate_limit_status: str = "ok",
                                         last_error_message: str = None) -> bool:
        """Update data quality metrics for monitoring"""
        try:
            async with self.db_pool.acquire() as conn:
                overall_score = (availability_score + accuracy_score + completeness_score + timeliness_score) / 4.0
                
                query = """
                    INSERT INTO free_api_data_quality (
                        source, data_type, timestamp, availability_score, accuracy_score,
                        completeness_score, timeliness_score, overall_score, error_count,
                        success_count, rate_limit_status, last_successful_fetch, last_error_message
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT DO NOTHING
                """
                
                await conn.execute(
                    query,
                    source,
                    data_type,
                    datetime.now(),
                    availability_score,
                    accuracy_score,
                    completeness_score,
                    timeliness_score,
                    overall_score,
                    error_count,
                    success_count,
                    rate_limit_status,
                    datetime.now() if success_count > 0 else None,
                    last_error_message
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error updating data quality metrics: {e}")
            return False
    
    async def cleanup_old_data(self) -> bool:
        """Clean up old data to maintain performance"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT cleanup_old_free_api_data()")
                self.logger.info("✅ Cleaned up old free API data")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old data: {e}")
            return False
