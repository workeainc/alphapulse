#!/usr/bin/env python3
"""
Auto-Labeling Service for Self-Training ML System
Implements distant supervision to automatically generate training labels
"""

import asyncio
import logging
import asyncpg
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LabelingConfig:
    """Configuration for auto-labeling"""
    # Return thresholds for binary classification
    threshold_30m: float = 0.01  # 1.0%
    threshold_2h: float = 0.015  # 1.5%
    threshold_24h: float = 0.04  # 4.0%
    
    # Time windows for return calculation
    window_30m: int = 30  # minutes
    window_2h: int = 120  # minutes
    window_24h: int = 1440  # minutes
    
    # Minimum time after publish to start calculating returns
    min_delay_minutes: int = 1
    
    # Confidence thresholds
    min_confidence: float = 0.5
    max_confidence: float = 0.95
    
    # Batch processing
    batch_size: int = 100
    max_processing_time: int = 3600  # seconds

@dataclass
class LabeledDataPoint:
    """A labeled data point for training"""
    news_id: int
    symbol: str
    publish_time: datetime
    y_30m: bool
    y_2h: bool
    y_24h: bool
    ret_30m: float
    ret_2h: float
    ret_24h: float
    price_at_publish: float
    volume_at_publish: float
    confidence_score: float
    labeling_metadata: Dict[str, Any]

class AutoLabelingService:
    """Service for automatically labeling news articles with market impact"""
    
    def __init__(self, db_pool: asyncpg.Pool, config: LabelingConfig = None):
        self.db_pool = db_pool
        self.config = config or LabelingConfig()
        
    async def generate_labels_for_news_articles(self, 
                                              start_time: datetime = None,
                                              end_time: datetime = None,
                                              symbols: List[str] = None) -> List[LabeledDataPoint]:
        """
        Generate labels for news articles using distant supervision
        
        Args:
            start_time: Start time for processing (default: 24 hours ago)
            end_time: End time for processing (default: now)
            symbols: List of symbols to process (default: all)
            
        Returns:
            List of labeled data points
        """
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.utcnow()
            
        logger.info(f"[LABELING] Starting auto-labeling for period {start_time} to {end_time}")
        
        try:
            # Get news articles that need labeling
            news_articles = await self._get_news_articles_for_labeling(start_time, end_time, symbols)
            logger.info(f"[NEWS] Found {len(news_articles)} news articles to label")
            
            labeled_data = []
            
            for article in news_articles:
                try:
                    # Extract symbols mentioned in the article
                    article_symbols = await self._extract_symbols_from_article(article)
                    
                    if not article_symbols:
                        continue
                    
                    # Generate labels for each symbol
                    for symbol in article_symbols:
                        labeled_point = await self._generate_label_for_article_symbol(article, symbol)
                        if labeled_point:
                            labeled_data.append(labeled_point)
                            
                except Exception as e:
                    logger.error(f"[ERROR] Error labeling article {article['id']}: {e}")
                    continue
            
            logger.info(f"[SUCCESS] Generated {len(labeled_data)} labeled data points")
            return labeled_data
            
        except Exception as e:
            logger.error(f"[ERROR] Error in generate_labels_for_news_articles: {e}")
            return []
    
    async def _get_news_articles_for_labeling(self, 
                                            start_time: datetime,
                                            end_time: datetime,
                                            symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Get news articles that need labeling"""
        
        query = """
            SELECT
                id, title, description, content, source, published_at,
                sentiment_score, breaking_news, verified_source,
                entities, metadata, keywords
            FROM raw_news_content
            WHERE published_at BETWEEN $1 AND $2
            AND id NOT IN (
                SELECT DISTINCT news_id 
                FROM labels_news_market 
                WHERE publish_time BETWEEN $1 AND $2
            )
            ORDER BY published_at DESC
        """
        
        params = [start_time, end_time]
        
        if symbols:
            # Filter by symbols mentioned in keywords field
            symbol_conditions = []
            for symbol in symbols:
                symbol_conditions.append(f"keywords @> '[\"{symbol}\"]'")
            if symbol_conditions:
                query += f" AND ({' OR '.join(symbol_conditions)})"
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def _extract_symbols_from_article(self, article: Dict[str, Any]) -> List[str]:
        """Extract cryptocurrency symbols mentioned in the article"""
        
        # Try to get symbols from keywords field first
        keywords = article.get('keywords', [])
        if keywords and isinstance(keywords, list):
            # Filter for crypto symbols
            crypto_symbols = [kw for kw in keywords if len(kw) <= 5 and kw.isupper()]
            if crypto_symbols:
                return crypto_symbols
        
        # Fallback: extract from entities or content
        symbols = set()
        
        # Common crypto symbols to look for
        crypto_symbols = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX', 'MATIC',
            'LINK', 'UNI', 'LTC', 'BCH', 'XLM', 'ATOM', 'ETC', 'FIL', 'VET', 'ICP'
        ]
        
        # Search in title and content
        text_to_search = f"{article.get('title', '')} {article.get('content', '')}"
        text_to_search = text_to_search.upper()
        
        for symbol in crypto_symbols:
            if symbol in text_to_search:
                symbols.add(symbol)
        
        return list(symbols)
    
    async def _generate_label_for_article_symbol(self, 
                                               article: Dict[str, Any],
                                               symbol: str) -> Optional[LabeledDataPoint]:
        """Generate labels for a specific article-symbol pair"""
        
        try:
            publish_time = article['published_at']
            
            # Get price data around publish time
            price_data = await self._get_price_data_around_publish_time(symbol, publish_time)
            
            if not price_data:
                return None
            
            # Calculate returns for different time windows
            returns = await self._calculate_returns(price_data, publish_time)
            
            if not returns:
                return None
            
            # Generate binary labels based on thresholds
            y_30m = abs(returns['ret_30m']) >= self.config.threshold_30m
            y_2h = abs(returns['ret_2h']) >= self.config.threshold_2h
            y_24h = abs(returns['ret_24h']) >= self.config.threshold_24h
            
            # Calculate confidence score
            confidence_score = self._calculate_labeling_confidence(article, returns)
            
            # Create labeling metadata
            labeling_metadata = {
                'source': article['source'],
                'sentiment_score': article.get('sentiment_score'),
                'breaking_news': article.get('breaking_news', False),
                'verified_source': article.get('verified_source', False),
                'price_data_points': len(price_data),
                'return_volatility': np.std([returns['ret_30m'], returns['ret_2h'], returns['ret_24h']])
            }
            
            return LabeledDataPoint(
                news_id=article['id'],
                symbol=symbol,
                publish_time=publish_time,
                y_30m=y_30m,
                y_2h=y_2h,
                y_24h=y_24h,
                ret_30m=returns['ret_30m'],
                ret_2h=returns['ret_2h'],
                ret_24h=returns['ret_24h'],
                price_at_publish=returns['price_at_publish'],
                volume_at_publish=returns['volume_at_publish'],
                confidence_score=confidence_score,
                labeling_metadata=labeling_metadata
            )
            
        except Exception as e:
            logger.error(f"[ERROR] Error generating label for article {article['id']} symbol {symbol}: {e}")
            return None
    
    async def _get_price_data_around_publish_time(self, 
                                                 symbol: str,
                                                 publish_time: datetime) -> List[Dict[str, Any]]:
        """Get price data around the publish time"""
        
        # Calculate time range for price data
        start_time = publish_time - timedelta(minutes=30)  # 30 minutes before
        end_time = publish_time + timedelta(hours=25)  # 25 hours after (for 24h return)
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = $1 
            AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp ASC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_time, end_time)
            return [dict(row) for row in rows]
    
    async def _calculate_returns(self, 
                               price_data: List[Dict[str, Any]],
                               publish_time: datetime) -> Optional[Dict[str, float]]:
        """Calculate returns for different time windows"""
        
        if len(price_data) < 10:  # Need minimum data points
            return None
        
        try:
            # Find the first price bar after publish time
            publish_timestamp = publish_time.replace(second=0, microsecond=0)  # Round to minute
            
            # Find the first bar strictly after publish time
            post_publish_data = [p for p in price_data if p['timestamp'] > publish_timestamp]
            
            if not post_publish_data:
                return None
            
            # Get price at publish time (use the last bar before publish)
            pre_publish_data = [p for p in price_data if p['timestamp'] <= publish_timestamp]
            if not pre_publish_data:
                return None
            
            price_at_publish = pre_publish_data[-1]['close']
            volume_at_publish = pre_publish_data[-1]['volume']
            
            # Calculate returns for different windows
            returns = {}
            
            # 30-minute return
            ret_30m = self._calculate_window_return(post_publish_data, self.config.window_30m)
            returns['ret_30m'] = ret_30m if ret_30m is not None else 0.0
            
            # 2-hour return
            ret_2h = self._calculate_window_return(post_publish_data, self.config.window_2h)
            returns['ret_2h'] = ret_2h if ret_2h is not None else 0.0
            
            # 24-hour return
            ret_24h = self._calculate_window_return(post_publish_data, self.config.window_24h)
            returns['ret_24h'] = ret_24h if ret_24h is not None else 0.0
            
            returns['price_at_publish'] = price_at_publish
            returns['volume_at_publish'] = volume_at_publish
            
            return returns
            
        except Exception as e:
            logger.error(f"[ERROR] Error calculating returns: {e}")
            return None
    
    def _calculate_window_return(self, 
                               price_data: List[Dict[str, Any]],
                               window_minutes: int) -> Optional[float]:
        """Calculate return for a specific time window"""
        
        if not price_data:
            return None
        
        # Find the start and end of the window
        start_time = price_data[0]['timestamp']
        end_time = start_time + timedelta(minutes=window_minutes)
        
        # Get price data within the window
        window_data = [p for p in price_data if p['timestamp'] <= end_time]
        
        if len(window_data) < 2:
            return None
        
        # Calculate return: (end_price - start_price) / start_price
        start_price = window_data[0]['close']
        end_price = window_data[-1]['close']
        
        return (end_price - start_price) / start_price
    
    def _calculate_labeling_confidence(self, 
                                     article: Dict[str, Any],
                                     returns: Dict[str, float]) -> float:
        """Calculate confidence score for the labeling"""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence for verified sources
        if article.get('verified_source', False):
            confidence += 0.2
        
        # Boost confidence for breaking news
        if article.get('breaking_news', False):
            confidence += 0.1
        
        # Boost confidence for high sentiment scores
        sentiment_score = abs(article.get('sentiment_score', 0))
        if sentiment_score > 0.7:
            confidence += 0.1
        
        # Boost confidence for high return volatility
        return_volatility = abs(returns['ret_30m']) + abs(returns['ret_2h']) + abs(returns['ret_24h'])
        if return_volatility > 0.05:  # 5% total volatility
            confidence += 0.1
        
        # Cap confidence at maximum
        return min(confidence, self.config.max_confidence)
    
    async def store_labeled_data(self, labeled_data: List[LabeledDataPoint]) -> int:
        """Store labeled data in the database"""
        
        if not labeled_data:
            return 0
        
        query = """
            INSERT INTO labels_news_market (
                timestamp, news_id, symbol, publish_time,
                y_30m, y_2h, y_24h, ret_30m, ret_2h, ret_24h,
                price_at_publish, volume_at_publish,
                labeling_method, confidence_score, labeling_metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (news_id, symbol, publish_time) DO UPDATE SET
                y_30m = EXCLUDED.y_30m,
                y_2h = EXCLUDED.y_2h,
                y_24h = EXCLUDED.y_24h,
                ret_30m = EXCLUDED.ret_30m,
                ret_2h = EXCLUDED.ret_2h,
                ret_24h = EXCLUDED.ret_24h,
                confidence_score = EXCLUDED.confidence_score,
                labeling_metadata = EXCLUDED.labeling_metadata
        """
        
        async with self.db_pool.acquire() as conn:
            stored_count = 0
            for data_point in labeled_data:
                try:
                    await conn.execute(
                        query,
                        datetime.utcnow(),
                        data_point.news_id,
                        data_point.symbol,
                        data_point.publish_time,
                        data_point.y_30m,
                        data_point.y_2h,
                        data_point.y_24h,
                        data_point.ret_30m,
                        data_point.ret_2h,
                        data_point.ret_24h,
                        data_point.price_at_publish,
                        data_point.volume_at_publish,
                        'auto',
                        data_point.confidence_score,
                        json.dumps(data_point.labeling_metadata)
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"[ERROR] Error storing labeled data point: {e}")
                    continue
        
        logger.info(f"[STORED] Stored {stored_count} labeled data points")
        return stored_count
    
    async def get_labeling_statistics(self, 
                                    start_time: datetime = None,
                                    end_time: datetime = None) -> Dict[str, Any]:
        """Get statistics about the labeling process"""
        
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.utcnow()
        
        query = """
            SELECT 
                COUNT(*) as total_labels,
                COUNT(*) FILTER (WHERE y_30m = true) as positive_30m,
                COUNT(*) FILTER (WHERE y_2h = true) as positive_2h,
                COUNT(*) FILTER (WHERE y_24h = true) as positive_24h,
                AVG(confidence_score) as avg_confidence,
                COUNT(DISTINCT news_id) as unique_articles,
                COUNT(DISTINCT symbol) as unique_symbols
            FROM labels_news_market
            WHERE timestamp BETWEEN $1 AND $2
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, start_time, end_time)
            
            if row:
                return {
                    'total_labels': row['total_labels'],
                    'positive_30m': row['positive_30m'],
                    'positive_2h': row['positive_2h'],
                    'positive_24h': row['positive_24h'],
                    'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0.0,
                    'unique_articles': row['unique_articles'],
                    'unique_symbols': row['unique_symbols'],
                    'positive_rate_30m': row['positive_30m'] / row['total_labels'] if row['total_labels'] > 0 else 0.0,
                    'positive_rate_2h': row['positive_2h'] / row['total_labels'] if row['total_labels'] > 0 else 0.0,
                    'positive_rate_24h': row['positive_24h'] / row['total_labels'] if row['total_labels'] > 0 else 0.0
                }
            
            return {}
