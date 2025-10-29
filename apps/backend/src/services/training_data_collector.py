"""
Training Data Collector for ML Models
Collects historical news data with market impact labels for model training
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class TrainingDataPoint:
    """Data class for training data points"""
    article_id: str
    title: str
    content: str
    published_at: datetime
    source: str
    sentiment_score: float
    entities: List[str]
    market_regime: str
    btc_dominance: float
    market_volatility: float
    social_volume: float
    cross_source_validation: float
    feed_credibility: float
    price_impact_24h: float
    enhanced_sentiment: float
    optimal_timing_score: float
    features: Dict[str, float]
    labels: Dict[str, float]

class TrainingDataCollector:
    """Collects and prepares training data for ML models"""
    
    def __init__(self, db_pool, config: Dict[str, Any]):
        self.db_pool = db_pool
        self.config = config
        self.ml_config = config.get('machine_learning', {})
        
        # Training data storage
        self.training_data = []
        self.validation_data = []
        
        # Data collection settings
        self.lookback_days = 30
        self.min_samples_per_model = 1000
        self.validation_split = 0.2
        
        logger.info("📊 TrainingDataCollector initialized successfully")
    
    async def collect_training_data(self) -> Dict[str, List[TrainingDataPoint]]:
        """Collect comprehensive training data for all ML models"""
        try:
            logger.info("🔄 Starting training data collection...")
            
            # Collect historical news data
            news_data = await self._collect_historical_news()
            
            # Collect market data
            market_data = await self._collect_market_data()
            
            # Collect price impact data
            price_impact_data = await self._collect_price_impact_data()
            
            # Merge and prepare training data
            training_data = await self._prepare_training_dataset(
                news_data, market_data, price_impact_data
            )
            
            # Split into training and validation sets
            train_data, val_data = self._split_training_data(training_data)
            
            self.training_data = train_data
            self.validation_data = val_data
            
            logger.info(f"✅ Collected {len(train_data)} training samples and {len(val_data)} validation samples")
            
            return {
                'training': train_data,
                'validation': val_data,
                'total_samples': len(training_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Error collecting training data: {e}")
            return {'training': [], 'validation': [], 'total_samples': 0}
    
    async def _collect_historical_news(self) -> List[Dict[str, Any]]:
        """Collect historical news articles from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get news articles from the last N days
                lookback_date = datetime.utcnow() - timedelta(days=self.lookback_days)
                
                query = """
                    SELECT
                        id, title, content, url, source, published_at,
                        sentiment_score, entities, language, category,
                        feed_credibility, cross_source_validation,
                        social_volume_current as social_volume, regime_aware_score as market_regime_score,
                        btc_dominance, market_volatility,
                        correlation_30m, correlation_2h, correlation_24h,
                        normalized_sentiment, sentiment_confidence,
                        EXTRACT(hour FROM published_at) as hour_of_day,
                        EXTRACT(dow FROM published_at) as day_of_week,
                        CASE WHEN EXTRACT(hour FROM published_at) BETWEEN 9 AND 16 THEN 1 ELSE 0 END as is_market_hours
                    FROM raw_news_content
                    WHERE published_at >= $1
                    AND sentiment_score IS NOT NULL
                    ORDER BY published_at DESC
                    LIMIT 10000;
                """
                
                rows = await conn.fetch(query, lookback_date)
                
                news_data = []
                for row in rows:
                    news_data.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'url': row['url'],
                        'source': row['source'],
                        'published_at': row['published_at'],
                        'sentiment_score': float(row['sentiment_score'] or 0.0),
                        'entities': row['entities'] or [],
                        'language': row['language'],
                        'category': row['category'],
                        'feed_credibility': float(row['feed_credibility'] or 0.5),
                        'cross_source_validation': float(row['cross_source_validation'] or 0.0),
                        'social_volume': float(row['social_volume'] or 0.0),
                        'market_regime_score': float(row['market_regime_score'] or 0.0),
                        'btc_dominance': float(row['btc_dominance'] or 50.0),
                        'market_volatility': float(row['market_volatility'] or 0.02),
                        'correlation_30m': float(row['correlation_30m'] or 0.0),
                        'correlation_2h': float(row['correlation_2h'] or 0.0),
                        'correlation_24h': float(row['correlation_24h'] or 0.0),
                        'normalized_sentiment': float(row['normalized_sentiment'] or 0.0),
                        'sentiment_confidence': float(row['sentiment_confidence'] or 0.5),
                        'hour_of_day': int(row['hour_of_day'] or 12),
                        'day_of_week': int(row['day_of_week'] or 3),
                        'is_market_hours': int(row['is_market_hours'] or 1)
                    })
                
                logger.info(f"📰 Collected {len(news_data)} historical news articles")
                return news_data
                
        except Exception as e:
            logger.error(f"❌ Error collecting historical news: {e}")
            return []
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect market data for training"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get market regime data
                regime_query = """
                    SELECT timestamp, market_regime as regime_type, 0.8 as confidence_score
                    FROM market_regime_data 
                    WHERE timestamp >= $1
                    ORDER BY timestamp DESC
                    LIMIT 1000;
                """
                
                lookback_date = datetime.utcnow() - timedelta(days=self.lookback_days)
                regime_rows = await conn.fetch(regime_query, lookback_date)
                
                market_data = {
                    'regime_data': [
                        {
                            'timestamp': row['timestamp'],
                            'regime_type': row['regime_type'],
                            'confidence': float(row['confidence_score'] or 0.5)
                        }
                        for row in regime_rows
                    ]
                }
                
                logger.info(f"📈 Collected {len(market_data['regime_data'])} market regime data points")
                return market_data
                
        except Exception as e:
            logger.error(f"❌ Error collecting market data: {e}")
            return {'regime_data': []}
    
    async def _collect_price_impact_data(self) -> Dict[str, float]:
        """Collect price impact data for training labels"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get price impact analysis
                impact_query = """
                    SELECT
                        news_id, symbol, impact_30m as impact_score_30m, impact_2h as impact_score_2h, impact_24h as impact_score_24h,
                        impact_30m as price_change_30m, impact_2h as price_change_2h, impact_24h as price_change_24h,
                        0.0 as volume_change_30m, 0.0 as volume_change_2h, 0.0 as volume_change_24h
                    FROM news_market_impact
                    WHERE created_at >= $1
                    ORDER BY created_at DESC
                    LIMIT 5000;
                """
                
                lookback_date = datetime.utcnow() - timedelta(days=self.lookback_days)
                impact_rows = await conn.fetch(impact_query, lookback_date)
                
                price_impact_data = {}
                for row in impact_rows:
                    news_id = row['news_id']
                    if news_id not in price_impact_data:
                        price_impact_data[news_id] = {}
                    
                    symbol = row['symbol']
                    price_impact_data[news_id][symbol] = {
                        'impact_score_30m': float(row['impact_score_30m'] or 0.0),
                        'impact_score_2h': float(row['impact_score_2h'] or 0.0),
                        'impact_score_24h': float(row['impact_score_24h'] or 0.0),
                        'price_change_30m': float(row['price_change_30m'] or 0.0),
                        'price_change_2h': float(row['price_change_2h'] or 0.0),
                        'price_change_24h': float(row['price_change_24h'] or 0.0),
                        'volume_change_30m': float(row['volume_change_30m'] or 0.0),
                        'volume_change_2h': float(row['volume_change_2h'] or 0.0),
                        'volume_change_24h': float(row['volume_change_24h'] or 0.0)
                    }
                
                logger.info(f"💰 Collected price impact data for {len(price_impact_data)} news articles")
                return price_impact_data
                
        except Exception as e:
            logger.error(f"❌ Error collecting price impact data: {e}")
            return {}
    
    async def _prepare_training_dataset(
        self, 
        news_data: List[Dict[str, Any]], 
        market_data: Dict[str, Any], 
        price_impact_data: Dict[str, float]
    ) -> List[TrainingDataPoint]:
        """Prepare comprehensive training dataset"""
        try:
            training_points = []
            
            for news in news_data:
                # Extract features
                features = self._extract_features(news, market_data)
                
                # Get price impact labels
                price_impact = price_impact_data.get(news['id'], {})
                btc_impact = price_impact.get('BTCUSDT', {})
                
                # Calculate labels
                labels = self._calculate_labels(news, btc_impact)
                
                # Create training data point
                training_point = TrainingDataPoint(
                    article_id=news['id'],
                    title=news['title'],
                    content=news['content'],
                    published_at=news['published_at'],
                    source=news['source'],
                    sentiment_score=news['sentiment_score'],
                    entities=news['entities'],
                    market_regime=self._get_market_regime(news['published_at'], market_data),
                    btc_dominance=news['btc_dominance'],
                    market_volatility=news['market_volatility'],
                    social_volume=news['social_volume'],
                    cross_source_validation=news['cross_source_validation'],
                    feed_credibility=news['feed_credibility'],
                    price_impact_24h=labels['price_impact_24h'],
                    enhanced_sentiment=labels['enhanced_sentiment'],
                    optimal_timing_score=labels['optimal_timing_score'],
                    features=features,
                    labels=labels
                )
                
                training_points.append(training_point)
            
            logger.info(f"✅ Prepared {len(training_points)} training data points")
            return training_points
            
        except Exception as e:
            logger.error(f"❌ Error preparing training dataset: {e}")
            return []
    
    def _extract_features(self, news: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML models"""
        features = {}
        
        # Text features
        features['title_length'] = len(news['title'])
        features['content_length'] = len(news['content']) if news['content'] else 0
        features['entity_count'] = len(news['entities'])
        
        # Sentiment features
        features['sentiment_score'] = news['sentiment_score']
        features['normalized_sentiment'] = news['normalized_sentiment']
        features['sentiment_confidence'] = news['sentiment_confidence']
        
        # Market features
        features['market_regime_score'] = news['market_regime_score']
        features['btc_dominance'] = news['btc_dominance']
        features['market_volatility'] = news['market_volatility']
        features['correlation_30m'] = news['correlation_30m']
        features['correlation_2h'] = news['correlation_2h']
        features['correlation_24h'] = news['correlation_24h']
        
        # Temporal features
        features['hour_of_day'] = news['hour_of_day']
        features['day_of_week'] = news['day_of_week']
        features['is_market_hours'] = news['is_market_hours']
        
        # Social features
        features['social_volume'] = news['social_volume']
        features['cross_source_validation'] = news['cross_source_validation']
        features['feed_credibility'] = news['feed_credibility']
        
        return features
    
    def _calculate_labels(self, news: Dict[str, Any], btc_impact: Dict[str, float]) -> Dict[str, float]:
        """Calculate labels for ML models"""
        labels = {}
        
        # Price impact label (24h)
        price_impact_24h = abs(btc_impact.get('price_change_24h', 0.0))
        labels['price_impact_24h'] = min(1.0, price_impact_24h / 0.1)  # Normalize to 0-1
        
        # Enhanced sentiment label
        base_sentiment = news['normalized_sentiment']
        market_regime_boost = news['market_regime_score'] * 0.1
        enhanced_sentiment = base_sentiment + market_regime_boost
        labels['enhanced_sentiment'] = max(-1.0, min(1.0, enhanced_sentiment))
        
        # Optimal timing label
        hour_score = 1.0 if 9 <= news['hour_of_day'] <= 16 else 0.5
        day_score = 1.0 if news['day_of_week'] < 5 else 0.3
        market_hours_score = news['is_market_hours']
        timing_score = (hour_score + day_score + market_hours_score) / 3
        labels['optimal_timing_score'] = timing_score
        
        return labels
    
    def _get_market_regime(self, timestamp: datetime, market_data: Dict[str, Any]) -> str:
        """Get market regime for a specific timestamp"""
        try:
            # Find the closest market regime data point
            regime_data = market_data.get('regime_data', [])
            if not regime_data:
                return 'neutral'
            
            # Find the regime data point closest to the timestamp
            closest_regime = 'neutral'
            min_diff = float('inf')
            
            for regime in regime_data:
                diff = abs((regime['timestamp'] - timestamp).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_regime = regime['regime_type']
            
            return closest_regime
            
        except Exception as e:
            logger.error(f"❌ Error getting market regime: {e}")
            return 'neutral'
    
    def _split_training_data(self, data: List[TrainingDataPoint]) -> Tuple[List[TrainingDataPoint], List[TrainingDataPoint]]:
        """Split data into training and validation sets"""
        try:
            # Shuffle data
            np.random.shuffle(data)
            
            # Calculate split index
            split_idx = int(len(data) * (1 - self.validation_split))
            
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            logger.info(f"📊 Split data: {len(train_data)} training, {len(val_data)} validation")
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"❌ Error splitting training data: {e}")
            return [], []
    
    async def get_training_data_summary(self) -> Dict[str, Any]:
        """Get summary of training data"""
        try:
            if not self.training_data:
                return {'error': 'No training data available'}
            
            # Calculate statistics
            total_samples = len(self.training_data) + len(self.validation_data)
            
            # Feature statistics
            feature_names = list(self.training_data[0].features.keys())
            feature_stats = {}
            
            for feature in feature_names:
                values = [point.features[feature] for point in self.training_data]
                feature_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            # Label statistics
            label_names = list(self.training_data[0].labels.keys())
            label_stats = {}
            
            for label in label_names:
                values = [point.labels[label] for point in self.training_data]
                label_stats[label] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            return {
                'total_samples': total_samples,
                'training_samples': len(self.training_data),
                'validation_samples': len(self.validation_data),
                'feature_count': len(feature_names),
                'label_count': len(label_names),
                'feature_names': feature_names,
                'label_names': label_names,
                'feature_statistics': feature_stats,
                'label_statistics': label_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting training data summary: {e}")
            return {'error': str(e)}
    
    async def save_training_data(self, filepath: str):
        """Save training data to file"""
        try:
            # Convert to serializable format
            data = {
                'training': [
                    {
                        'article_id': point.article_id,
                        'title': point.title,
                        'content': point.content,
                        'published_at': point.published_at.isoformat(),
                        'source': point.source,
                        'features': point.features,
                        'labels': point.labels
                    }
                    for point in self.training_data
                ],
                'validation': [
                    {
                        'article_id': point.article_id,
                        'title': point.title,
                        'content': point.content,
                        'published_at': point.published_at.isoformat(),
                        'source': point.source,
                        'features': point.features,
                        'labels': point.labels
                    }
                    for point in self.validation_data
                ]
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Training data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving training data: {e}")
    
    async def load_training_data(self, filepath: str):
        """Load training data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert back to TrainingDataPoint objects
            self.training_data = [
                TrainingDataPoint(
                    article_id=item['article_id'],
                    title=item['title'],
                    content=item['content'],
                    published_at=datetime.fromisoformat(item['published_at']),
                    source=item['source'],
                    sentiment_score=0.0,  # Will be filled from features
                    entities=[],  # Will be filled from features
                    market_regime='neutral',
                    btc_dominance=50.0,
                    market_volatility=0.02,
                    social_volume=0.0,
                    cross_source_validation=0.0,
                    feed_credibility=0.5,
                    price_impact_24h=item['labels']['price_impact_24h'],
                    enhanced_sentiment=item['labels']['enhanced_sentiment'],
                    optimal_timing_score=item['labels']['optimal_timing_score'],
                    features=item['features'],
                    labels=item['labels']
                )
                for item in data['training']
            ]
            
            self.validation_data = [
                TrainingDataPoint(
                    article_id=item['article_id'],
                    title=item['title'],
                    content=item['content'],
                    published_at=datetime.fromisoformat(item['published_at']),
                    source=item['source'],
                    sentiment_score=0.0,
                    entities=[],
                    market_regime='neutral',
                    btc_dominance=50.0,
                    market_volatility=0.02,
                    social_volume=0.0,
                    cross_source_validation=0.0,
                    feed_credibility=0.5,
                    price_impact_24h=item['labels']['price_impact_24h'],
                    enhanced_sentiment=item['labels']['enhanced_sentiment'],
                    optimal_timing_score=item['labels']['optimal_timing_score'],
                    features=item['features'],
                    labels=item['labels']
                )
                for item in data['validation']
            ]
            
            logger.info(f"✅ Training data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error loading training data: {e}")
