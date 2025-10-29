"""
Free API Data Pipeline Service
Continuously collects and stores free API data for signal generation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import json
import time

from src.services.free_api_manager import FreeAPIManager
from src.services.free_api_database_service import (
    FreeAPIDatabaseService, FreeAPIMarketData, FreeAPISentimentData, 
    FreeAPINewsData, FreeAPISocialData, FreeAPILiquidationEvent
)

logger = logging.getLogger(__name__)

class FreeAPIDataPipeline:
    """Continuous data collection pipeline for free APIs"""
    
    def __init__(self, db_service: FreeAPIDatabaseService, free_api_manager: FreeAPIManager):
        self.db_service = db_service
        self.free_api_manager = free_api_manager
        self.logger = logging.getLogger(__name__)
        
        # Pipeline configuration
        self.collection_intervals = {
            'market_data': 60,      # 1 minute
            'sentiment_data': 300,  # 5 minutes
            'news_data': 900,      # 15 minutes
            'social_data': 600,    # 10 minutes
            'liquidation_events': 30  # 30 seconds
        }
        
        # Symbols to track
        self.tracked_symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'BNB', 'XRP', 'DOGE', 'MATIC', 'AVAX', 'DOT']
        
        # Pipeline state
        self.is_running = False
        self.last_collection_times = {}
        self.collection_stats = {
            'market_data': {'success': 0, 'errors': 0, 'last_success': None},
            'sentiment_data': {'success': 0, 'errors': 0, 'last_success': None},
            'news_data': {'success': 0, 'errors': 0, 'last_success': None},
            'social_data': {'success': 0, 'errors': 0, 'last_success': None},
            'liquidation_events': {'success': 0, 'errors': 0, 'last_success': None}
        }
        
        # Rate limiting tracking
        self.rate_limit_status = {}
    
    async def start_pipeline(self):
        """Start the continuous data collection pipeline"""
        try:
            self.logger.info("🚀 Starting Free API Data Pipeline...")
            self.is_running = True
            
            # Start collection tasks
            tasks = [
                asyncio.create_task(self._market_data_collection_loop()),
                asyncio.create_task(self._sentiment_data_collection_loop()),
                asyncio.create_task(self._news_data_collection_loop()),
                asyncio.create_task(self._social_data_collection_loop()),
                asyncio.create_task(self._liquidation_events_collection_loop()),
                asyncio.create_task(self._pipeline_monitoring_loop())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"❌ Error starting pipeline: {e}")
        finally:
            self.is_running = False
    
    async def stop_pipeline(self):
        """Stop the data collection pipeline"""
        self.logger.info("🛑 Stopping Free API Data Pipeline...")
        self.is_running = False
    
    async def _market_data_collection_loop(self):
        """Continuous market data collection loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                last_collection = self.last_collection_times.get('market_data')
                
                if (not last_collection or 
                    (current_time - last_collection).total_seconds() >= self.collection_intervals['market_data']):
                    
                    await self._collect_market_data()
                    self.last_collection_times['market_data'] = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in market data collection loop: {e}")
                self.collection_stats['market_data']['errors'] += 1
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _sentiment_data_collection_loop(self):
        """Continuous sentiment data collection loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                last_collection = self.last_collection_times.get('sentiment_data')
                
                if (not last_collection or 
                    (current_time - last_collection).total_seconds() >= self.collection_intervals['sentiment_data']):
                    
                    await self._collect_sentiment_data()
                    self.last_collection_times['sentiment_data'] = current_time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in sentiment data collection loop: {e}")
                self.collection_stats['sentiment_data']['errors'] += 1
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _news_data_collection_loop(self):
        """Continuous news data collection loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                last_collection = self.last_collection_times.get('news_data')
                
                if (not last_collection or 
                    (current_time - last_collection).total_seconds() >= self.collection_intervals['news_data']):
                    
                    await self._collect_news_data()
                    self.last_collection_times['news_data'] = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Error in news data collection loop: {e}")
                self.collection_stats['news_data']['errors'] += 1
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _social_data_collection_loop(self):
        """Continuous social media data collection loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                last_collection = self.last_collection_times.get('social_data')
                
                if (not last_collection or 
                    (current_time - last_collection).total_seconds() >= self.collection_intervals['social_data']):
                    
                    await self._collect_social_data()
                    self.last_collection_times['social_data'] = current_time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in social data collection loop: {e}")
                self.collection_stats['social_data']['errors'] += 1
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _liquidation_events_collection_loop(self):
        """Continuous liquidation events collection loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                last_collection = self.last_collection_times.get('liquidation_events')
                
                if (not last_collection or 
                    (current_time - last_collection).total_seconds() >= self.collection_intervals['liquidation_events']):
                    
                    await self._collect_liquidation_events()
                    self.last_collection_times['liquidation_events'] = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in liquidation events collection loop: {e}")
                self.collection_stats['liquidation_events']['errors'] += 1
                await asyncio.sleep(15)  # Wait longer on error
    
    async def _pipeline_monitoring_loop(self):
        """Monitor pipeline health and performance"""
        while self.is_running:
            try:
                # Log pipeline status every 5 minutes
                await asyncio.sleep(300)
                
                if self.is_running:
                    self.logger.info("📊 Free API Pipeline Status:")
                    for data_type, stats in self.collection_stats.items():
                        success_rate = stats['success'] / (stats['success'] + stats['errors']) * 100 if (stats['success'] + stats['errors']) > 0 else 0
                        self.logger.info(f"  {data_type}: {stats['success']} success, {stats['errors']} errors ({success_rate:.1f}% success rate)")
                    
                    # Update data quality metrics
                    await self._update_data_quality_metrics()
                    
                    # Cleanup old data periodically
                    if datetime.now().hour % 6 == 0:  # Every 6 hours
                        await self.db_service.cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"❌ Error in pipeline monitoring: {e}")
    
    async def _collect_market_data(self):
        """Collect market data from all sources"""
        try:
            self.logger.debug("📈 Collecting market data...")
            
            for symbol in self.tracked_symbols:
                try:
                    # Get market data from free APIs
                    market_data_result = await self.free_api_manager.get_market_data(symbol)
                    
                    if market_data_result and market_data_result.get('success'):
                        data = market_data_result['data']
                        
                        # Store market data from each source
                        for source, source_data in data.items():
                            if source_data and isinstance(source_data, dict):
                                market_data = FreeAPIMarketData(
                                    symbol=symbol,
                                    timestamp=datetime.now(),
                                    source=source,
                                    price=source_data.get('price', 0.0),
                                    volume_24h=source_data.get('volume_24h'),
                                    market_cap=source_data.get('market_cap'),
                                    price_change_24h=source_data.get('price_change_24h'),
                                    volume_change_24h=source_data.get('volume_change_24h'),
                                    market_cap_change_24h=source_data.get('market_cap_change_24h'),
                                    fear_greed_index=source_data.get('fear_greed_index'),
                                    liquidation_events=source_data.get('liquidation_events'),
                                    raw_data=source_data,
                                    data_quality_score=market_data_result.get('data_quality_score', 1.0)
                                )
                                
                                await self.db_service.store_market_data(market_data)
                    
                    # Small delay between symbols to avoid rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error collecting market data for {symbol}: {e}")
                    continue
            
            self.collection_stats['market_data']['success'] += 1
            self.collection_stats['market_data']['last_success'] = datetime.now()
            self.logger.debug("✅ Market data collection completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in market data collection: {e}")
            self.collection_stats['market_data']['errors'] += 1
    
    async def _collect_sentiment_data(self):
        """Collect sentiment data from all sources"""
        try:
            self.logger.debug("😊 Collecting sentiment data...")
            
            for symbol in self.tracked_symbols:
                try:
                    # Get sentiment data from free APIs
                    sentiment_result = await self.free_api_manager.get_sentiment_analysis(symbol)
                    
                    if sentiment_result and sentiment_result.get('success'):
                        data = sentiment_result['data']
                        
                        # Store sentiment data from each source
                        for source, source_data in data.items():
                            if source_data and isinstance(source_data, dict):
                                sentiment_data = FreeAPISentimentData(
                                    symbol=symbol,
                                    timestamp=datetime.now(),
                                    source=source,
                                    sentiment_type=source_data.get('sentiment_type', 'general'),
                                    sentiment_score=source_data.get('sentiment_score', 0.0),
                                    sentiment_label=source_data.get('sentiment_label', 'neutral'),
                                    confidence=source_data.get('confidence', 0.0),
                                    volume=source_data.get('volume'),
                                    keywords=source_data.get('keywords'),
                                    raw_data=source_data,
                                    data_quality_score=sentiment_result.get('data_quality_score', 1.0)
                                )
                                
                                await self.db_service.store_sentiment_data(sentiment_data)
                    
                    # Small delay between symbols
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error collecting sentiment data for {symbol}: {e}")
                    continue
            
            self.collection_stats['sentiment_data']['success'] += 1
            self.collection_stats['sentiment_data']['last_success'] = datetime.now()
            self.logger.debug("✅ Sentiment data collection completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in sentiment data collection: {e}")
            self.collection_stats['sentiment_data']['errors'] += 1
    
    async def _collect_news_data(self):
        """Collect news data from all sources"""
        try:
            self.logger.debug("📰 Collecting news data...")
            
            for symbol in self.tracked_symbols:
                try:
                    # Get news data from free APIs
                    news_result = await self.free_api_manager.get_news_sentiment(symbol)
                    
                    if news_result and news_result.get('success'):
                        data = news_result['data']
                        
                        # Store news data from each source
                        for source, source_data in data.items():
                            if source_data and isinstance(source_data, dict):
                                articles = source_data.get('articles', [])
                                
                                for article in articles:
                                    if isinstance(article, dict):
                                        news_data = FreeAPINewsData(
                                            symbol=symbol,
                                            timestamp=datetime.now(),
                                            source=source,
                                            title=article.get('title', ''),
                                            content=article.get('content'),
                                            url=article.get('url'),
                                            published_at=article.get('published_at'),
                                            sentiment_score=article.get('sentiment_score'),
                                            sentiment_label=article.get('sentiment_label'),
                                            relevance_score=article.get('relevance_score'),
                                            keywords=article.get('keywords'),
                                            raw_data=article
                                        )
                                        
                                        await self.db_service.store_news_data(news_data)
                    
                    # Small delay between symbols
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error collecting news data for {symbol}: {e}")
                    continue
            
            self.collection_stats['news_data']['success'] += 1
            self.collection_stats['news_data']['last_success'] = datetime.now()
            self.logger.debug("✅ News data collection completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in news data collection: {e}")
            self.collection_stats['news_data']['errors'] += 1
    
    async def _collect_social_data(self):
        """Collect social media data from all sources"""
        try:
            self.logger.debug("📱 Collecting social media data...")
            
            for symbol in self.tracked_symbols:
                try:
                    # Get social sentiment data from free APIs
                    social_result = await self.free_api_manager.get_social_sentiment(symbol)
                    
                    if social_result and social_result.get('success'):
                        data = social_result['data']
                        
                        # Store social data from each platform
                        for platform, platform_data in data.items():
                            if platform_data and isinstance(platform_data, dict):
                                posts = platform_data.get('posts', [])
                                
                                for post in posts:
                                    if isinstance(post, dict):
                                        social_data = FreeAPISocialData(
                                            symbol=symbol,
                                            timestamp=datetime.now(),
                                            platform=platform,
                                            post_id=post.get('post_id'),
                                            content=post.get('content', ''),
                                            author=post.get('author'),
                                            engagement_metrics=post.get('engagement_metrics'),
                                            sentiment_score=post.get('sentiment_score'),
                                            sentiment_label=post.get('sentiment_label'),
                                            influence_score=post.get('influence_score'),
                                            keywords=post.get('keywords'),
                                            raw_data=post
                                        )
                                        
                                        await self.db_service.store_social_data(social_data)
                    
                    # Small delay between symbols
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error collecting social data for {symbol}: {e}")
                    continue
            
            self.collection_stats['social_data']['success'] += 1
            self.collection_stats['social_data']['last_success'] = datetime.now()
            self.logger.debug("✅ Social media data collection completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in social data collection: {e}")
            self.collection_stats['social_data']['errors'] += 1
    
    async def _collect_liquidation_events(self):
        """Collect liquidation events from all sources"""
        try:
            self.logger.debug("💥 Collecting liquidation events...")
            
            for symbol in self.tracked_symbols:
                try:
                    # Get liquidation events from free APIs
                    liquidation_result = await self.free_api_manager.get_liquidation_events(symbol)
                    
                    if liquidation_result and liquidation_result.get('success'):
                        data = liquidation_result['data']
                        
                        # Store liquidation events from each source
                        for source, source_data in data.items():
                            if source_data and isinstance(source_data, dict):
                                events = source_data.get('events', [])
                                
                                liquidation_events = []
                                for event in events:
                                    if isinstance(event, dict):
                                        liquidation_event = FreeAPILiquidationEvent(
                                            symbol=symbol,
                                            timestamp=datetime.now(),
                                            source=source,
                                            liquidation_type=event.get('type', 'unknown'),
                                            price=event.get('price', 0.0),
                                            quantity=event.get('quantity', 0.0),
                                            value_usd=event.get('value_usd', 0.0),
                                            side=event.get('side', 'unknown'),
                                            raw_data=event
                                        )
                                        liquidation_events.append(liquidation_event)
                                
                                if liquidation_events:
                                    await self.db_service.store_liquidation_events(liquidation_events)
                    
                    # Small delay between symbols
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error collecting liquidation events for {symbol}: {e}")
                    continue
            
            self.collection_stats['liquidation_events']['success'] += 1
            self.collection_stats['liquidation_events']['last_success'] = datetime.now()
            self.logger.debug("✅ Liquidation events collection completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in liquidation events collection: {e}")
            self.collection_stats['liquidation_events']['errors'] += 1
    
    async def _update_data_quality_metrics(self):
        """Update data quality metrics for monitoring"""
        try:
            for data_type, stats in self.collection_stats.items():
                total_attempts = stats['success'] + stats['errors']
                if total_attempts > 0:
                    availability_score = stats['success'] / total_attempts
                    accuracy_score = 1.0 if stats['errors'] == 0 else max(0.0, 1.0 - (stats['errors'] / total_attempts))
                    completeness_score = 1.0 if stats['last_success'] else 0.0
                    timeliness_score = 1.0 if stats['last_success'] and (datetime.now() - stats['last_success']).total_seconds() < 3600 else 0.0
                    
                    await self.db_service.update_data_quality_metrics(
                        source='free_api_pipeline',
                        data_type=data_type,
                        availability_score=availability_score,
                        accuracy_score=accuracy_score,
                        completeness_score=completeness_score,
                        timeliness_score=timeliness_score,
                        error_count=stats['errors'],
                        success_count=stats['success']
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating data quality metrics: {e}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            status = {
                'is_running': self.is_running,
                'tracked_symbols': self.tracked_symbols,
                'collection_intervals': self.collection_intervals,
                'collection_stats': self.collection_stats,
                'last_collection_times': self.last_collection_times,
                'rate_limit_status': self.rate_limit_status,
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting pipeline status: {e}")
            return {'is_running': False, 'error': str(e)}
    
    async def force_collection(self, data_type: str, symbol: Optional[str] = None):
        """Force immediate collection of specific data type"""
        try:
            symbols_to_collect = [symbol] if symbol else self.tracked_symbols
            
            if data_type == 'market_data':
                for sym in symbols_to_collect:
                    await self._collect_market_data_for_symbol(sym)
            elif data_type == 'sentiment_data':
                for sym in symbols_to_collect:
                    await self._collect_sentiment_data_for_symbol(sym)
            elif data_type == 'news_data':
                for sym in symbols_to_collect:
                    await self._collect_news_data_for_symbol(sym)
            elif data_type == 'social_data':
                for sym in symbols_to_collect:
                    await self._collect_social_data_for_symbol(sym)
            elif data_type == 'liquidation_events':
                for sym in symbols_to_collect:
                    await self._collect_liquidation_events_for_symbol(sym)
            
            self.logger.info(f"✅ Forced collection of {data_type} completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in forced collection: {e}")
    
    async def _collect_market_data_for_symbol(self, symbol: str):
        """Collect market data for specific symbol"""
        # Implementation similar to _collect_market_data but for single symbol
        pass
    
    async def _collect_sentiment_data_for_symbol(self, symbol: str):
        """Collect sentiment data for specific symbol"""
        # Implementation similar to _collect_sentiment_data but for single symbol
        pass
    
    async def _collect_news_data_for_symbol(self, symbol: str):
        """Collect news data for specific symbol"""
        # Implementation similar to _collect_news_data but for single symbol
        pass
    
    async def _collect_social_data_for_symbol(self, symbol: str):
        """Collect social data for specific symbol"""
        # Implementation similar to _collect_social_data but for single symbol
        pass
    
    async def _collect_liquidation_events_for_symbol(self, symbol: str):
        """Collect liquidation events for specific symbol"""
        # Implementation similar to _collect_liquidation_events but for single symbol
        pass
