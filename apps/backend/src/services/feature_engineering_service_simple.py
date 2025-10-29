#!/usr/bin/env python3
"""
Simplified Feature Engineering Service for Self-Training ML System
Implements basic feature extraction without sentence-transformers dependency
"""

import asyncio
import logging
import asyncpg
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Text features
    tfidf_max_features: int = 1000  # Reduced for simplicity
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    
    # Event tags
    event_tags: List[str] = None
    
    # Social metrics
    social_window_minutes: int = 30
    dev_activity_window_days: int = 7
    whale_tx_window_hours: int = 24
    
    # Market regime
    market_regime_window_days: int = 10
    volatility_window_days: int = 14
    
    def __post_init__(self):
        if self.event_tags is None:
            self.event_tags = [
                'hack', 'listing', 'etf', 'airdrop', 'lawsuit', 'upgrade', 
                'partnership', 'regulation', 'adoption', 'institutional',
                'deflation', 'inflation', 'halving', 'fork', 'merge'
            ]

@dataclass
class FeatureSet:
    """Complete feature set for a news article"""
    news_id: int
    symbol: str
    
    # Text features
    title_tfidf_ngrams: Dict[str, float]
    embedding_384d: List[float]  # Placeholder for now
    entities: Dict[str, Any]
    event_tags: List[str]
    
    # Meta features
    source_trust: float
    is_breaking: bool
    is_important: bool
    is_hot: bool
    publish_hour: int
    day_of_week: int
    dedup_cluster_size: int
    
    # Social & on-chain context
    social_volume_zscore_30m: float
    social_volume_zscore_neg30m: float
    dev_activity_7d_change: float
    whale_tx_usd_1m_plus_24h_change: float
    
    # Market regime controls
    btc_dominance: float
    total_mc_zscore: float
    asset_vol_10d: float
    atr_14: float
    funding_rate: float
    
    # Metadata
    feature_version: str
    feature_metadata: Dict[str, Any]

class SimpleFeatureEngineeringService:
    """Simplified service for extracting features from news articles"""
    
    def __init__(self, db_pool: asyncpg.Pool, config: FeatureConfig = None):
        self.db_pool = db_pool
        self.config = config or FeatureConfig()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            stop_words='english'
        )
        
        logger.info("[INIT] Simple feature engineering service initialized")
    
    async def extract_features_for_news_article(self, 
                                             news_id: int,
                                             symbol: str,
                                             article_data: Dict[str, Any]) -> Optional[FeatureSet]:
        """Extract comprehensive features for a news article"""
        
        try:
            # Extract text features
            text_features = await self._extract_text_features(article_data)
            
            # Extract meta features
            meta_features = await self._extract_meta_features(article_data)
            
            # Extract social and on-chain context
            social_features = await self._extract_social_features(symbol, article_data['published_at'])
            
            # Extract market regime controls
            market_features = await self._extract_market_features(symbol, article_data['published_at'])
            
            # Combine all features
            feature_set = FeatureSet(
                news_id=news_id,
                symbol=symbol,
                **text_features,
                **meta_features,
                **social_features,
                **market_features,
                feature_version='v1.0-simple',
                feature_metadata={
                    'extraction_timestamp': datetime.utcnow().isoformat(),
                    'article_source': article_data.get('source'),
                    'article_published_at': article_data['published_at'].isoformat()
                }
            )
            
            return feature_set
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting features for news {news_id}: {e}")
            return None
    
    async def _extract_text_features(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text-based features"""
        
        # Combine title and content for processing
        title = article_data.get('title', '')
        content = article_data.get('content', '')
        description = article_data.get('description', '')
        
        combined_text = f"{title} {description} {content}".strip()
        
        # TF-IDF features
        title_tfidf_ngrams = await self._extract_tfidf_features(title)
        
        # Simple embedding features (placeholder)
        embedding_384d = [0.0] * 384  # Placeholder for now
        
        # Entity recognition (simplified)
        entities = await self._extract_entities_simple(combined_text)
        
        # Event tag detection
        event_tags = await self._extract_event_tags(combined_text)
        
        return {
            'title_tfidf_ngrams': title_tfidf_ngrams,
            'embedding_384d': embedding_384d,
            'entities': entities,
            'event_tags': event_tags
        }
    
    async def _extract_tfidf_features(self, text: str) -> Dict[str, float]:
        """Extract TF-IDF features from text"""
        try:
            # Fit and transform the text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Convert to dictionary
            tfidf_dict = {}
            for i, feature_name in enumerate(feature_names):
                tfidf_dict[feature_name] = float(tfidf_matrix[0, i])
            
            return tfidf_dict
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting TF-IDF features: {e}")
            return {}
    
    async def _extract_entities_simple(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text (simplified version)"""
        try:
            entities = {
                'organizations': [],
                'projects': [],
                'exchanges': [],
                'chains': [],
                'people': [],
                'locations': []
            }
            
            # Common crypto entities
            crypto_entities = {
                'organizations': ['Binance', 'Coinbase', 'Kraken', 'FTX', 'MicroStrategy', 'Tesla'],
                'projects': ['Bitcoin', 'Ethereum', 'Cardano', 'Solana', 'Polkadot', 'Chainlink'],
                'exchanges': ['Binance', 'Coinbase', 'Kraken', 'Kucoin', 'Bybit', 'OKX'],
                'chains': ['Ethereum', 'Bitcoin', 'Solana', 'Cardano', 'Polkadot', 'Polygon']
            }
            
            # Simple text matching
            text_upper = text.upper()
            for category, entity_list in crypto_entities.items():
                for entity in entity_list:
                    if entity.upper() in text_upper:
                        entities[category].append(entity)
            
            # Remove duplicates
            for category in entities:
                entities[category] = list(set(entities[category]))
            
            return entities
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting entities: {e}")
            return {'organizations': [], 'projects': [], 'exchanges': [], 'chains': [], 'people': [], 'locations': []}
    
    async def _extract_event_tags(self, text: str) -> List[str]:
        """Extract event tags from text"""
        try:
            text_lower = text.lower()
            detected_tags = []
            
            for tag in self.config.event_tags:
                if tag.lower() in text_lower:
                    detected_tags.append(tag)
            
            return detected_tags
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting event tags: {e}")
            return []
    
    async def _extract_meta_features(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meta features from article data"""
        
        publish_time = article_data['published_at']
        
        # Source trust score
        source_trust = await self._calculate_source_trust(article_data.get('source', ''))
        
        # Breaking news indicators
        is_breaking = article_data.get('breaking_news', False)
        is_important = article_data.get('important', False)
        is_hot = article_data.get('hot', False)
        
        # Temporal features
        publish_hour = publish_time.hour
        day_of_week = publish_time.weekday()
        
        # Deduplication cluster size
        dedup_cluster_size = await self._calculate_dedup_cluster_size(article_data)
        
        return {
            'source_trust': source_trust,
            'is_breaking': is_breaking,
            'is_important': is_important,
            'is_hot': is_hot,
            'publish_hour': publish_hour,
            'day_of_week': day_of_week,
            'dedup_cluster_size': dedup_cluster_size
        }
    
    async def _calculate_source_trust(self, source: str) -> float:
        """Calculate trust score for a news source"""
        
        # Define trust scores for known sources
        trusted_sources = {
            'reuters': 0.9,
            'bloomberg': 0.9,
            'cnbc': 0.8,
            'coindesk': 0.8,
            'cointelegraph': 0.7,
            'decrypt': 0.7,
            'theblock': 0.8,
            'cryptopanic': 0.6
        }
        
        source_lower = source.lower()
        
        # Check for exact matches
        for trusted_source, score in trusted_sources.items():
            if trusted_source in source_lower:
                return score
        
        # Default trust score for unknown sources
        return 0.5
    
    async def _calculate_dedup_cluster_size(self, article_data: Dict[str, Any]) -> int:
        """Calculate deduplication cluster size"""
        
        try:
            title = article_data.get('title', '')
            publish_time = article_data['published_at']
            
            # Look for similar articles within a time window
            start_time = publish_time - timedelta(hours=1)
            end_time = publish_time + timedelta(hours=1)
            
            query = """
                SELECT title, published_at
                FROM raw_news_content
                WHERE published_at BETWEEN $1 AND $2
                AND id != $3
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, start_time, end_time, article_data['id'])
                
                cluster_size = 1  # Start with current article
                
                for row in rows:
                    similarity = fuzz.ratio(title.lower(), row['title'].lower())
                    if similarity > 80:  # 80% similarity threshold
                        cluster_size += 1
                
                return cluster_size
                
        except Exception as e:
            logger.error(f"[ERROR] Error calculating dedup cluster size: {e}")
            return 1
    
    async def _extract_social_features(self, symbol: str, publish_time: datetime) -> Dict[str, float]:
        """Extract social and on-chain context features"""
        
        try:
            # Social volume z-scores
            social_volume_zscore_30m = await self._get_social_volume_zscore(symbol, publish_time, 30)
            social_volume_zscore_neg30m = await self._get_social_volume_zscore(symbol, publish_time, -30)
            
            # Developer activity change
            dev_activity_7d_change = await self._get_dev_activity_change(symbol, publish_time)
            
            # Whale transaction change
            whale_tx_change = await self._get_whale_tx_change(symbol, publish_time)
            
            return {
                'social_volume_zscore_30m': social_volume_zscore_30m,
                'social_volume_zscore_neg30m': social_volume_zscore_neg30m,
                'dev_activity_7d_change': dev_activity_7d_change,
                'whale_tx_usd_1m_plus_24h_change': whale_tx_change
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting social features: {e}")
            return {
                'social_volume_zscore_30m': 0.0,
                'social_volume_zscore_neg30m': 0.0,
                'dev_activity_7d_change': 0.0,
                'whale_tx_usd_1m_plus_24h_change': 0.0
            }
    
    async def _get_social_volume_zscore(self, symbol: str, publish_time: datetime, offset_minutes: int) -> float:
        """Get social volume z-score for a time window"""
        
        try:
            # Calculate time window
            if offset_minutes > 0:
                start_time = publish_time
                end_time = publish_time + timedelta(minutes=offset_minutes)
            else:
                start_time = publish_time + timedelta(minutes=offset_minutes)
                end_time = publish_time
            
            # Query social sentiment data
            query = """
                SELECT AVG(sentiment_score) as avg_sentiment, COUNT(*) as volume
                FROM social_sentiment
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, symbol, start_time, end_time)
                
                if row and row['volume'] > 0:
                    # Calculate z-score (simplified)
                    return float(row['avg_sentiment']) if row['avg_sentiment'] else 0.0
                
                return 0.0
                
        except Exception as e:
            logger.error(f"[ERROR] Error getting social volume z-score: {e}")
            return 0.0
    
    async def _get_dev_activity_change(self, symbol: str, publish_time: datetime) -> float:
        """Get developer activity change over 7 days"""
        
        try:
            # This would typically come from GitHub API or similar
            # For now, return a placeholder value
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting dev activity change: {e}")
            return 0.0
    
    async def _get_whale_tx_change(self, symbol: str, publish_time: datetime) -> float:
        """Get whale transaction change over 24 hours"""
        
        try:
            # This would typically come from blockchain data
            # For now, return a placeholder value
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting whale tx change: {e}")
            return 0.0
    
    async def _extract_market_features(self, symbol: str, publish_time: datetime) -> Dict[str, float]:
        """Extract market regime control features"""
        
        try:
            # BTC dominance
            btc_dominance = await self._get_btc_dominance(publish_time)
            
            # Total market cap z-score
            total_mc_zscore = await self._get_total_mc_zscore(publish_time)
            
            # Asset volatility
            asset_vol_10d = await self._get_asset_volatility(symbol, publish_time, 10)
            
            # ATR (Average True Range)
            atr_14 = await self._get_atr_14(symbol, publish_time)
            
            # Funding rate (for futures)
            funding_rate = await self._get_funding_rate(symbol, publish_time)
            
            return {
                'btc_dominance': btc_dominance,
                'total_mc_zscore': total_mc_zscore,
                'asset_vol_10d': asset_vol_10d,
                'atr_14': atr_14,
                'funding_rate': funding_rate
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting market features: {e}")
            return {
                'btc_dominance': 0.0,
                'total_mc_zscore': 0.0,
                'asset_vol_10d': 0.0,
                'atr_14': 0.0,
                'funding_rate': 0.0
            }
    
    async def _get_btc_dominance(self, timestamp: datetime) -> float:
        """Get BTC dominance at timestamp"""
        
        try:
            query = """
                SELECT btc_dominance
                FROM market_intelligence
                WHERE timestamp <= $1
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, timestamp)
                return float(row['btc_dominance']) if row and row['btc_dominance'] else 0.0
                
        except Exception as e:
            logger.error(f"[ERROR] Error getting BTC dominance: {e}")
            return 0.0
    
    async def _get_total_mc_zscore(self, timestamp: datetime) -> float:
        """Get total market cap z-score at timestamp"""
        
        try:
            # This would calculate z-score based on historical market cap data
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting total MC z-score: {e}")
            return 0.0
    
    async def _get_asset_volatility(self, symbol: str, timestamp: datetime, days: int) -> float:
        """Get asset volatility over specified days"""
        
        try:
            start_time = timestamp - timedelta(days=days)
            
            query = """
                SELECT STDDEV(close) as volatility
                FROM market_data
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, symbol, start_time, timestamp)
                return float(row['volatility']) if row and row['volatility'] else 0.0
                
        except Exception as e:
            logger.error(f"[ERROR] Error getting asset volatility: {e}")
            return 0.0
    
    async def _get_atr_14(self, symbol: str, timestamp: datetime) -> float:
        """Get 14-period ATR for asset"""
        
        try:
            # This would calculate ATR from OHLC data
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting ATR: {e}")
            return 0.0
    
    async def _get_funding_rate(self, symbol: str, timestamp: datetime) -> float:
        """Get funding rate for asset (futures only)"""
        
        try:
            # This would come from exchange API
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting funding rate: {e}")
            return 0.0
    
    async def store_feature_set(self, feature_set: FeatureSet) -> bool:
        """Store feature set in database"""
        
        try:
            query = """
                INSERT INTO feature_engineering_pipeline (
                    timestamp, news_id, symbol,
                    title_tfidf_ngrams, embedding_384d, entities, event_tags,
                    source_trust, is_breaking, is_important, is_hot,
                    publish_hour, day_of_week, dedup_cluster_size,
                    social_volume_zscore_30m, social_volume_zscore_neg30m,
                    dev_activity_7d_change, whale_tx_usd_1m_plus_24h_change,
                    btc_dominance, total_mc_zscore, asset_vol_10d, atr_14, funding_rate,
                    feature_version, feature_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
                ON CONFLICT (news_id, symbol) DO UPDATE SET
                    title_tfidf_ngrams = EXCLUDED.title_tfidf_ngrams,
                    embedding_384d = EXCLUDED.embedding_384d,
                    entities = EXCLUDED.entities,
                    event_tags = EXCLUDED.event_tags,
                    source_trust = EXCLUDED.source_trust,
                    is_breaking = EXCLUDED.is_breaking,
                    is_important = EXCLUDED.is_important,
                    is_hot = EXCLUDED.is_hot,
                    publish_hour = EXCLUDED.publish_hour,
                    day_of_week = EXCLUDED.day_of_week,
                    dedup_cluster_size = EXCLUDED.dedup_cluster_size,
                    social_volume_zscore_30m = EXCLUDED.social_volume_zscore_30m,
                    social_volume_zscore_neg30m = EXCLUDED.social_volume_zscore_neg30m,
                    dev_activity_7d_change = EXCLUDED.dev_activity_7d_change,
                    whale_tx_usd_1m_plus_24h_change = EXCLUDED.whale_tx_usd_1m_plus_24h_change,
                    btc_dominance = EXCLUDED.btc_dominance,
                    total_mc_zscore = EXCLUDED.total_mc_zscore,
                    asset_vol_10d = EXCLUDED.asset_vol_10d,
                    atr_14 = EXCLUDED.atr_14,
                    funding_rate = EXCLUDED.funding_rate,
                    feature_version = EXCLUDED.feature_version,
                    feature_metadata = EXCLUDED.feature_metadata
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.utcnow(),
                    feature_set.news_id,
                    feature_set.symbol,
                    json.dumps(feature_set.title_tfidf_ngrams),
                    feature_set.embedding_384d,
                    json.dumps(feature_set.entities),
                    feature_set.event_tags,
                    feature_set.source_trust,
                    feature_set.is_breaking,
                    feature_set.is_important,
                    feature_set.is_hot,
                    feature_set.publish_hour,
                    feature_set.day_of_week,
                    feature_set.dedup_cluster_size,
                    feature_set.social_volume_zscore_30m,
                    feature_set.social_volume_zscore_neg30m,
                    feature_set.dev_activity_7d_change,
                    feature_set.whale_tx_usd_1m_plus_24h_change,
                    feature_set.btc_dominance,
                    feature_set.total_mc_zscore,
                    feature_set.asset_vol_10d,
                    feature_set.atr_14,
                    feature_set.funding_rate,
                    feature_set.feature_version,
                    json.dumps(feature_set.feature_metadata)
                )
            
            logger.info(f"[STORED] Stored features for news {feature_set.news_id} symbol {feature_set.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error storing feature set: {e}")
            return False
