"""
Enhanced Sentiment Service for AlphaPlus
Integrates advanced sentiment analysis with real-time processing and multi-source data collection
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
import redis
from dataclasses import dataclass
import json
import os

# Import enhanced sentiment analyzer
from src.ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer, SentimentData, SentimentAggregation

logger = logging.getLogger(__name__)

@dataclass
class SentimentSummary:
    """Sentiment summary for API responses"""
    symbol: str
    timestamp: datetime
    overall_sentiment: float
    sentiment_label: str
    confidence: float
    source_breakdown: Dict[str, float]
    volume_metrics: Dict[str, int]
    trend: str
    trend_strength: float
    fear_greed_index: Optional[int] = None
    market_mood: str = 'neutral'

class EnhancedSentimentService:
    """Enhanced sentiment service with real-time processing and multi-source integration"""
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        
        # Initialize enhanced sentiment analyzer
        self.sentiment_analyzer = EnhancedSentimentAnalyzer(db_pool, redis_client)
        
        # Supported symbols
        self.supported_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        
        # Cache settings
        self.cache_timeout = 300  # 5 minutes
        self.sentiment_cache = {}
        
        # Background task settings
        self.collection_interval = 60  # 1 minute
        self.aggregation_interval = 300  # 5 minutes
        self.is_running = False
        
        logger.info("✅ Enhanced Sentiment Service initialized")
    
    async def start_service(self):
        """Start the sentiment service"""
        if self.is_running:
            logger.warning("Sentiment service is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Enhanced Sentiment Service...")
        
        # Start background tasks
        asyncio.create_task(self._background_sentiment_collection())
        asyncio.create_task(self._background_sentiment_aggregation())
        
        logger.info("✅ Enhanced Sentiment Service started successfully")
    
    async def stop_service(self):
        """Stop the sentiment service"""
        if not self.is_running:
            logger.warning("Sentiment service is not running")
            return
        
        self.is_running = False
        logger.info("🛑 Stopping Enhanced Sentiment Service...")
        
        # Close sentiment analyzer
        await self.sentiment_analyzer.close()
        
        logger.info("✅ Enhanced Sentiment Service stopped")
    
    async def _background_sentiment_collection(self):
        """Background task for collecting sentiment data"""
        while self.is_running:
            try:
                logger.debug("🔄 Collecting sentiment data...")
                
                for symbol in self.supported_symbols:
                    # Collect sentiment from all sources
                    sentiment_data = await self.sentiment_analyzer.collect_all_sentiment(symbol)
                    
                    if sentiment_data:
                        logger.debug(f"📊 Collected {len(sentiment_data)} sentiment records for {symbol}")
                    
                    # Small delay between symbols to avoid rate limiting
                    await asyncio.sleep(1)
                
                # Wait for next collection cycle
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in background sentiment collection: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retrying
    
    async def _background_sentiment_aggregation(self):
        """Background task for aggregating sentiment data"""
        while self.is_running:
            try:
                logger.debug("🔄 Aggregating sentiment data...")
                
                for symbol in self.supported_symbols:
                    # Aggregate sentiment for different time windows
                    for window_size in ['1min', '5min', '15min', '1hour']:
                        aggregation = await self.sentiment_analyzer.aggregate_sentiment(symbol, window_size)
                        
                        if aggregation:
                            logger.debug(f"📈 Aggregated {window_size} sentiment for {symbol}")
                    
                    # Small delay between symbols
                    await asyncio.sleep(0.5)
                
                # Wait for next aggregation cycle
                await asyncio.sleep(self.aggregation_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in background sentiment aggregation: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retrying
    
    async def get_sentiment_summary(self, symbol: str) -> Optional[SentimentSummary]:
        """Get sentiment summary for a symbol"""
        try:
            # Check cache first
            cache_key = f"sentiment_summary_{symbol}"
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data:
                return SentimentSummary(**cached_data)
            
            # Get latest sentiment aggregation
            aggregation = await self.sentiment_analyzer.aggregate_sentiment(symbol, '5min')
            
            if not aggregation:
                return None
            
            # Get fear & greed index
            fear_greed_index = await self._get_fear_greed_index()
            
            # Determine market mood
            market_mood = self._determine_market_mood(aggregation.overall_sentiment_score)
            
            # Create sentiment summary
            summary = SentimentSummary(
                symbol=aggregation.symbol,
                timestamp=aggregation.timestamp,
                overall_sentiment=aggregation.overall_sentiment_score,
                sentiment_label=self._get_sentiment_label(aggregation.overall_sentiment_score),
                confidence=aggregation.confidence_weighted_score,
                source_breakdown=aggregation.source_breakdown,
                volume_metrics=aggregation.volume_metrics,
                trend=aggregation.sentiment_trend,
                trend_strength=aggregation.trend_strength,
                fear_greed_index=fear_greed_index,
                market_mood=market_mood
            )
            
            # Cache the result
            await self._set_cache(cache_key, summary.__dict__, self.cache_timeout)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {e}")
            return None
    
    async def get_multi_symbol_sentiment(self, symbols: List[str]) -> Dict[str, SentimentSummary]:
        """Get sentiment summary for multiple symbols"""
        try:
            results = {}
            
            for symbol in symbols:
                summary = await self.get_sentiment_summary(symbol)
                if summary:
                    results[symbol] = summary
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting multi-symbol sentiment: {e}")
            return {}
    
    async def collect_all_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect sentiment data from all sources for a symbol"""
        try:
            # Use the sentiment analyzer's collection method
            sentiment_data = await self.sentiment_analyzer.collect_all_sentiment(symbol)
            return sentiment_data or []
        except Exception as e:
            logger.error(f"Error collecting sentiment for {symbol}: {e}")
            return []
    
    async def aggregate_sentiment(self, symbol: str, window_size: str) -> Optional[Dict[str, Any]]:
        """Aggregate sentiment data for a symbol and time window"""
        try:
            # Use the sentiment analyzer's aggregation method
            aggregation = await self.sentiment_analyzer.aggregate_sentiment(symbol, window_size)
            
            if aggregation:
                return {
                    'symbol': aggregation.symbol,
                    'timestamp': aggregation.timestamp,
                    'window_size': aggregation.window_size,
                    'overall_sentiment_score': aggregation.overall_sentiment_score,
                    'positive_sentiment_score': aggregation.positive_sentiment_score,
                    'negative_sentiment_score': aggregation.negative_sentiment_score,
                    'neutral_sentiment_score': aggregation.neutral_sentiment_score,
                    'total_volume': aggregation.total_volume,
                    'confidence_weighted_score': aggregation.confidence_weighted_score,
                    'source_diversity_score': aggregation.source_diversity_score,
                    'sentiment_trend': aggregation.sentiment_trend,
                    'trend_strength': aggregation.trend_strength,
                    'source_breakdown': aggregation.source_breakdown,
                    'volume_metrics': aggregation.volume_metrics
                }
            return None
        except Exception as e:
            logger.error(f"Error aggregating sentiment for {symbol}: {e}")
            return None

    async def get_sentiment_trends(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment trends over time"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Query sentiment aggregation data
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, overall_sentiment_score, confidence_weighted_score,
                           sentiment_trend, trend_strength
                    FROM real_time_sentiment_aggregation
                    WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                    AND window_size = '5min'
                    ORDER BY timestamp ASC
                """
                rows = await conn.fetch(query, symbol, start_time, end_time)
            
            if not rows:
                return {}
            
            # Process trend data
            timestamps = [row['timestamp'] for row in rows]
            sentiment_scores = [row['overall_sentiment_score'] for row in rows]
            confidence_scores = [row['confidence_weighted_score'] for row in rows]
            trends = [row['sentiment_trend'] for row in rows]
            trend_strengths = [row['trend_strength'] for row in rows]
            
            # Calculate trend statistics
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            trend_changes = sum(1 for i in range(1, len(trends)) if trends[i] != trends[i-1])
            
            return {
                'symbol': symbol,
                'time_range': f'{hours}h',
                'data_points': len(rows),
                'timestamps': [ts.isoformat() for ts in timestamps],
                'sentiment_scores': sentiment_scores,
                'confidence_scores': confidence_scores,
                'trends': trends,
                'trend_strengths': trend_strengths,
                'statistics': {
                    'average_sentiment': avg_sentiment,
                    'sentiment_volatility': sentiment_volatility,
                    'trend_changes': trend_changes,
                    'current_trend': trends[-1] if trends else 'stable',
                    'current_strength': trend_strengths[-1] if trend_strengths else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment trends for {symbol}: {e}")
            return {}
    
    async def get_sentiment_alerts(self, symbol: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get sentiment alerts for significant changes"""
        try:
            # Get recent sentiment data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, overall_sentiment_score, confidence_weighted_score,
                           sentiment_trend, trend_strength
                    FROM real_time_sentiment_aggregation
                    WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                    AND window_size = '5min'
                    ORDER BY timestamp DESC
                """
                rows = await conn.fetch(query, symbol, start_time, end_time)
            
            if len(rows) < 2:
                return []
            
            alerts = []
            
            # Check for significant sentiment changes
            for i in range(1, len(rows)):
                current = rows[i-1]
                previous = rows[i]
                
                sentiment_change = abs(current['overall_sentiment_score'] - previous['overall_sentiment_score'])
                
                if sentiment_change >= threshold:
                    alerts.append({
                        'timestamp': current['timestamp'].isoformat(),
                        'alert_type': 'sentiment_spike',
                        'sentiment_change': sentiment_change,
                        'current_sentiment': current['overall_sentiment_score'],
                        'previous_sentiment': previous['overall_sentiment_score'],
                        'confidence': current['confidence_weighted_score'],
                        'trend': current['sentiment_trend'],
                        'severity': 'high' if sentiment_change >= 0.5 else 'medium' if sentiment_change >= 0.3 else 'low'
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting sentiment alerts for {symbol}: {e}")
            return []
    
    async def _get_fear_greed_index(self) -> Optional[int]:
        """Get current fear & greed index"""
        try:
            # Check cache first
            cache_key = 'fear_greed_index'
            cached_data = await self._get_from_cache(cache_key)
            
            if cached_data:
                return cached_data.get('value')
            
            # Simulate fear & greed index (in production, fetch from API)
            fear_greed_value = 65  # Simulated value
            
            # Cache the result
            await self._set_cache(cache_key, {'value': fear_greed_value}, 300)
            
            return fear_greed_value
            
        except Exception as e:
            logger.error(f"Error getting fear & greed index: {e}")
            return None
    
    def _determine_market_mood(self, sentiment_score: float) -> str:
        """Determine market mood based on sentiment score"""
        if sentiment_score >= 0.3:
            return 'bullish'
        elif sentiment_score <= -0.3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Get sentiment label based on score"""
        if sentiment_score >= 0.1:
            return 'positive'
        elif sentiment_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    async def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get data from Redis cache"""
        try:
            if self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
        return None
    
    async def _set_cache(self, key: str, data: Dict, timeout: int):
        """Set data in Redis cache"""
        try:
            if self.redis_client:
                await self.redis_client.setex(key, timeout, json.dumps(data))
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status and health metrics"""
        try:
            # Get basic service status
            status = {
                'service': 'enhanced_sentiment_service',
                'status': 'running' if self.is_running else 'stopped',
                'timestamp': datetime.utcnow().isoformat(),
                'supported_symbols': self.supported_symbols,
                'collection_interval': self.collection_interval,
                'aggregation_interval': self.aggregation_interval
            }
            
            # Get database connection status
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                status['database_status'] = 'connected'
            except Exception as e:
                status['database_status'] = f'error: {str(e)}'
            
            # Get Redis connection status
            try:
                if self.redis_client:
                    await self.redis_client.ping()
                    status['redis_status'] = 'connected'
                else:
                    status['redis_status'] = 'not_configured'
            except Exception as e:
                status['redis_status'] = f'error: {str(e)}'
            
            # Get recent sentiment statistics
            try:
                async with self.db_pool.acquire() as conn:
                    query = """
                        SELECT COUNT(*) as total_records,
                               COUNT(DISTINCT symbol) as symbols_covered,
                               MAX(timestamp) as latest_update
                        FROM enhanced_sentiment_data
                        WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    """
                    row = await conn.fetchrow(query)
                    
                    if row:
                        status['recent_statistics'] = {
                            'total_records_last_hour': row['total_records'],
                            'symbols_covered': row['symbols_covered'],
                            'latest_update': row['latest_update'].isoformat() if row['latest_update'] else None
                        }
            except Exception as e:
                status['recent_statistics'] = f'error: {str(e)}'
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                'service': 'enhanced_sentiment_service',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    # ===== PHASE 4A: PREDICTIVE ANALYTICS METHODS =====
    
    async def get_price_prediction(self, symbol: str, time_horizon: str = '4h') -> Dict[str, Any]:
        """Get price movement prediction for a symbol"""
        try:
            # Check cache first
            cache_key = f"price_prediction:{symbol}:{time_horizon}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # Get prediction from analyzer
            prediction = await self.sentiment_analyzer.predict_price_movement(symbol, time_horizon)
            
            # Cache the result (shorter cache for predictions)
            await self.redis_client.setex(
                cache_key, 
                60,  # 1 minute cache for predictions
                json.dumps(prediction, default=str)
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting price prediction for {symbol}: {e}")
            return {}
    
    async def get_multi_horizon_predictions(self, symbol: str) -> Dict[str, Any]:
        """Get predictions for multiple time horizons"""
        try:
            horizons = ['1h', '4h', '1d', '1w']
            predictions = {}
            
            for horizon in horizons:
                prediction = await self.get_price_prediction(symbol, horizon)
                if prediction:
                    predictions[horizon] = prediction
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error getting multi-horizon predictions for {symbol}: {e}")
            return {}
    
    async def get_prediction_confidence_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed confidence analysis for predictions"""
        try:
            # Get recent predictions
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT prediction_probability, confidence, direction, strength, timestamp
                    FROM sentiment_predictions 
                    WHERE symbol = $1 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """
                rows = await conn.fetch(query, symbol)
            
            if not rows:
                return {'symbol': symbol, 'confidence_analysis': 'No recent predictions'}
            
            # Analyze confidence patterns
            confidences = [row['confidence'] for row in rows]
            probabilities = [row['prediction_probability'] for row in rows]
            directions = [row['direction'] for row in rows]
            
            analysis = {
                'symbol': symbol,
                'average_confidence': np.mean(confidences),
                'confidence_volatility': np.std(confidences),
                'average_probability': np.mean(probabilities),
                'prediction_consistency': len(set(directions)) / len(directions),
                'recent_predictions': len(rows),
                'confidence_trend': 'increasing' if confidences[0] > confidences[-1] else 'decreasing'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting prediction confidence analysis for {symbol}: {e}")
            return {}

    # ===== PHASE 4B: CROSS-ASSET CORRELATION METHODS =====
    
    async def get_cross_asset_analysis(self, primary_symbol: str, secondary_symbols: List[str] = None) -> Dict[str, Any]:
        """Get cross-asset sentiment correlation analysis"""
        try:
            # Check cache first
            cache_key = f"cross_asset:{primary_symbol}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # Get cross-asset analysis from analyzer
            analysis = await self.sentiment_analyzer.analyze_cross_asset_sentiment(primary_symbol, secondary_symbols)
            
            # Cache the result
            await self.redis_client.setex(
                cache_key, 
                300,  # 5 minutes cache
                json.dumps(analysis, default=str)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting cross-asset analysis for {primary_symbol}: {e}")
            return {}
    
    async def get_market_sentiment_overview(self) -> Dict[str, Any]:
        """Get overall market sentiment overview"""
        try:
            # Check cache first
            cache_key = "market_sentiment_overview"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # Get sentiment for all supported symbols
            all_sentiments = {}
            for symbol in self.supported_symbols:
                sentiment = await self.get_sentiment_summary(symbol)
                if sentiment:
                    all_sentiments[symbol] = {
                        'overall_sentiment': sentiment.overall_sentiment,
                        'sentiment_label': sentiment.sentiment_label,
                        'confidence': sentiment.confidence,
                        'trend': sentiment.trend
                    }
            
            # Calculate market-wide metrics
            sentiment_scores = [data['overall_sentiment'] for data in all_sentiments.values()]
            confidence_scores = [data['confidence'] for data in all_sentiments.values()]
            
            market_overview = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_assets': len(all_sentiments),
                'average_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0.0,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'sentiment_volatility': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0,
                'bullish_assets': len([s for s in sentiment_scores if s > 0.1]),
                'bearish_assets': len([s for s in sentiment_scores if s < -0.1]),
                'neutral_assets': len([s for s in sentiment_scores if -0.1 <= s <= 0.1]),
                'individual_sentiments': all_sentiments
            }
            
            # Cache the result
            await self.redis_client.setex(
                cache_key, 
                300,  # 5 minutes cache
                json.dumps(market_overview, default=str)
            )
            
            return market_overview
            
        except Exception as e:
            logger.error(f"Error getting market sentiment overview: {e}")
            return {}

    # ===== PHASE 4C: MODEL PERFORMANCE METHODS =====
    
    async def get_model_performance_summary(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get model performance summary"""
        try:
            # Check cache first
            cache_key = f"model_performance:{symbol or 'all'}:{days}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # Get performance from analyzer
            performance = await self.sentiment_analyzer.get_model_performance_summary(symbol, days)
            
            # Cache the result
            await self.redis_client.setex(
                cache_key, 
                600,  # 10 minutes cache
                json.dumps(performance, default=str)
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {}
    
    async def update_model_performance(self, actual_outcomes: List[Dict]):
        """Update model performance with actual outcomes"""
        try:
            await self.sentiment_analyzer.update_model_performance(actual_outcomes)
            logger.info(f"Updated model performance with {len(actual_outcomes)} outcomes")
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    async def get_prediction_alerts(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get prediction alerts for significant sentiment changes"""
        try:
            alerts = []
            
            # Get recent predictions with high confidence
            async with self.db_pool.acquire() as conn:
                if symbol:
                    query = """
                        SELECT symbol, prediction_probability, direction, strength, confidence, timestamp
                        FROM sentiment_predictions 
                        WHERE symbol = $1 AND confidence > 0.8 AND timestamp >= NOW() - INTERVAL '1 hour'
                        ORDER BY timestamp DESC
                    """
                    rows = await conn.fetch(query, symbol)
                else:
                    query = """
                        SELECT symbol, prediction_probability, direction, strength, confidence, timestamp
                        FROM sentiment_predictions 
                        WHERE confidence > 0.8 AND timestamp >= NOW() - INTERVAL '1 hour'
                        ORDER BY timestamp DESC
                    """
                    rows = await conn.fetch(query)
            
            for row in rows:
                # Create alert for high-confidence predictions
                alert = {
                    'symbol': row['symbol'],
                    'type': 'high_confidence_prediction',
                    'message': f"High confidence {row['direction']} prediction for {row['symbol']}",
                    'prediction_probability': row['prediction_probability'],
                    'direction': row['direction'],
                    'strength': row['strength'],
                    'confidence': row['confidence'],
                    'timestamp': row['timestamp'].isoformat(),
                    'severity': 'high' if row['confidence'] > 0.9 else 'medium'
                }
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting prediction alerts: {e}")
            return []
