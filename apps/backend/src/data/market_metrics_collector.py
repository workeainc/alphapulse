#!/usr/bin/env python3
"""
Market Metrics Collector for AlphaPulse
Collects BTC dominance and total crypto market cap for correlation analysis
"""

import asyncio
import logging
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

from .storage import DataStorage

logger = logging.getLogger(__name__)

@dataclass
class MarketMetrics:
    """Market metrics data"""
    timestamp: datetime
    total_market_cap: float
    btc_dominance: float
    btc_price: float
    btc_market_cap: float
    total_volume_24h: float
    btc_volume_24h: float
    market_cap_change_24h: float
    btc_dominance_change_24h: float
    source: str
    confidence: float

class MarketMetricsCollector:
    """
    Collects market metrics from multiple sources for correlation analysis
    with price action
    """
    
    def __init__(self, config: Dict = None):
        """Initialize market metrics collector"""
        self.config = config or {}
        
        # Initialize components
        self.storage = DataStorage(self.config.get('storage_path', 'data'))
        
        # API configuration
        self.coingecko_api_url = "https://api.coingecko.com/api/v3"
        self.coinmarketcap_api_url = "https://pro-api.coinmarketcap.com/v1"
        self.coinmarketcap_api_key = self.config.get('coinmarketcap_api_key')
        
        # Collection settings
        self.collection_interval = self.config.get('collection_interval', 60)  # seconds
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        
        # Data sources priority
        self.data_sources = [
            'coingecko',  # Primary source (free, reliable)
            'coinmarketcap'  # Fallback (requires API key)
        ]
        
        # Performance tracking
        self.stats = {
            'metrics_collected': 0,
            'api_calls': 0,
            'errors': 0,
            'last_update': None,
            'collection_start': None
        }
        
        # Callbacks
        self.metrics_callbacks = []
        
        # Session management
        self.session = None
        
        logger.info("📊 Market Metrics Collector initialized")
    
    async def start_collection(self):
        """Start collecting market metrics"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'AlphaPulse/1.0 (Market Metrics Collector)'
                }
            )
            
            # Start collection loop
            self.stats['collection_start'] = datetime.now()
            asyncio.create_task(self._collection_loop())
            
            logger.info("✅ Market metrics collection started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start market metrics collection: {e}")
            raise
    
    async def stop_collection(self):
        """Stop collecting market metrics"""
        try:
            if self.session:
                await self.session.close()
            
            logger.info("🛑 Market metrics collection stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping collection: {e}")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while True:
            try:
                # Collect metrics from primary source
                metrics = await self._collect_metrics()
                
                if metrics:
                    # Store metrics
                    await self._store_metrics(metrics)
                    
                    # Notify callbacks
                    await self._notify_metrics_callbacks(metrics)
                    
                    # Update stats
                    self._update_stats()
                    
                    logger.debug(f"📊 Collected market metrics: {metrics.total_market_cap:,.0f} MC, {metrics.btc_dominance:.2f}% BTC")
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in collection loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(self.retry_delay)
    
    async def _collect_metrics(self) -> Optional[MarketMetrics]:
        """Collect market metrics from available sources"""
        for source in self.data_sources:
            try:
                if source == 'coingecko':
                    metrics = await self._collect_from_coingecko()
                elif source == 'coinmarketcap':
                    metrics = await self._collect_from_coinmarketcap()
                else:
                    continue
                
                if metrics:
                    return metrics
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to collect from {source}: {e}")
                continue
        
        logger.error("❌ Failed to collect metrics from all sources")
        return None
    
    async def _collect_from_coingecko(self) -> Optional[MarketMetrics]:
        """Collect metrics from CoinGecko API"""
        try:
            # Get global market data
            global_url = f"{self.coingecko_api_url}/global"
            
            async with self.session.get(global_url) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ CoinGecko API returned status {response.status}")
                    return None
                
                data = await response.json()
                self.stats['api_calls'] += 1
                
                if 'data' not in data:
                    logger.warning("⚠️ CoinGecko API response missing 'data' field")
                    return None
                
                global_data = data['data']
                
                # Get BTC specific data
                btc_url = f"{self.coingecko_api_url}/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
                
                async with self.session.get(btc_url) as btc_response:
                    if btc_response.status == 200:
                        btc_data = await btc_response.json()
                        self.stats['api_calls'] += 1
                    else:
                        btc_data = {}
                
                # Extract metrics
                total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
                total_volume = global_data.get('total_volume', {}).get('usd', 0)
                btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
                
                btc_info = btc_data.get('bitcoin', {})
                btc_price = btc_info.get('usd', 0)
                btc_market_cap = btc_info.get('usd_market_cap', 0)
                btc_volume = btc_info.get('usd_24h_vol', 0)
                btc_change = btc_info.get('usd_24h_change', 0)
                
                # Calculate changes (would need historical data for accurate calculation)
                market_cap_change = 0.0  # Placeholder
                btc_dominance_change = 0.0  # Placeholder
                
                # Calculate confidence based on data completeness
                confidence = self._calculate_data_confidence([
                    total_market_cap, total_volume, btc_dominance, 
                    btc_price, btc_market_cap, btc_volume
                ])
                
                return MarketMetrics(
                    timestamp=datetime.now(),
                    total_market_cap=total_market_cap,
                    btc_dominance=btc_dominance,
                    btc_price=btc_price,
                    btc_market_cap=btc_market_cap,
                    total_volume_24h=total_volume,
                    btc_volume_24h=btc_volume,
                    market_cap_change_24h=market_cap_change,
                    btc_dominance_change_24h=btc_dominance_change,
                    source='coingecko',
                    confidence=confidence
                )
                
        except Exception as e:
            logger.error(f"❌ Error collecting from CoinGecko: {e}")
            return None
    
    async def _collect_from_coinmarketcap(self) -> Optional[MarketMetrics]:
        """Collect metrics from CoinMarketCap API"""
        try:
            if not self.coinmarketcap_api_key:
                logger.warning("⚠️ CoinMarketCap API key not configured")
                return None
            
            # Get global market data
            global_url = f"{self.coinmarketcap_api_url}/global-metrics/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key
            }
            
            async with self.session.get(global_url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ CoinMarketCap API returned status {response.status}")
                    return None
                
                data = await response.json()
                self.stats['api_calls'] += 1
                
                if 'data' not in data:
                    logger.warning("⚠️ CoinMarketCap API response missing 'data' field")
                    return None
                
                global_data = data['data']
                quote = global_data.get('quote', {}).get('USD', {})
                
                # Get BTC specific data
                btc_url = f"{self.coinmarketcap_api_url}/cryptocurrency/quotes/latest?symbol=BTC"
                
                async with self.session.get(btc_url, headers=headers) as btc_response:
                    if btc_response.status == 200:
                        btc_data = await btc_response.json()
                        self.stats['api_calls'] += 1
                    else:
                        btc_data = {}
                
                # Extract metrics
                total_market_cap = quote.get('total_market_cap', 0)
                total_volume = quote.get('total_volume_24h', 0)
                btc_dominance = global_data.get('btc_dominance', 0)
                
                btc_info = btc_data.get('data', {}).get('BTC', {})
                btc_quote = btc_info.get('quote', {}).get('USD', {})
                
                btc_price = btc_quote.get('price', 0)
                btc_market_cap = btc_quote.get('market_cap', 0)
                btc_volume = btc_quote.get('volume_24h', 0)
                btc_change = btc_quote.get('percent_change_24h', 0)
                
                # Calculate changes
                market_cap_change = quote.get('total_market_cap_yesterday_percentage_change', 0)
                btc_dominance_change = 0.0  # Would need historical data
                
                # Calculate confidence
                confidence = self._calculate_data_confidence([
                    total_market_cap, total_volume, btc_dominance,
                    btc_price, btc_market_cap, btc_volume
                ])
                
                return MarketMetrics(
                    timestamp=datetime.now(),
                    total_market_cap=total_market_cap,
                    btc_dominance=btc_dominance,
                    btc_price=btc_price,
                    btc_market_cap=btc_market_cap,
                    total_volume_24h=total_volume,
                    btc_volume_24h=btc_volume,
                    market_cap_change_24h=market_cap_change,
                    btc_dominance_change_24h=btc_dominance_change,
                    source='coinmarketcap',
                    confidence=confidence
                )
                
        except Exception as e:
            logger.error(f"❌ Error collecting from CoinMarketCap: {e}")
            return None
    
    def _calculate_data_confidence(self, values: List[float]) -> float:
        """Calculate confidence level based on data completeness and validity"""
        try:
            if not values:
                return 0.0
            
            # Check for valid (non-zero, non-negative) values
            valid_count = sum(1 for v in values if v and v > 0)
            completeness = valid_count / len(values)
            
            # Check for reasonable ranges
            reasonable_count = 0
            for value in values:
                if value:
                    if 0 < value < 1e15:  # Reasonable range for market data
                        reasonable_count += 1
            
            validity = reasonable_count / len(values) if values else 0
            
            # Overall confidence
            confidence = (completeness + validity) / 2
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    async def _store_metrics(self, metrics: MarketMetrics):
        """Store market metrics in database"""
        try:
            await self.storage.store_market_metrics(
                timestamp=metrics.timestamp,
                total_market_cap=metrics.total_market_cap,
                btc_dominance=metrics.btc_dominance,
                btc_price=metrics.btc_price,
                btc_market_cap=metrics.btc_market_cap,
                total_volume_24h=metrics.total_volume_24h,
                btc_volume_24h=metrics.btc_volume_24h,
                market_cap_change_24h=metrics.market_cap_change_24h,
                btc_dominance_change_24h=metrics.btc_dominance_change_24h,
                source=metrics.source,
                confidence=metrics.confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Error storing market metrics: {e}")
    
    async def _notify_metrics_callbacks(self, metrics: MarketMetrics):
        """Notify metrics callbacks"""
        for callback in self.metrics_callbacks:
            try:
                await callback(metrics)
            except Exception as e:
                logger.error(f"❌ Error in metrics callback: {e}")
    
    def _update_stats(self):
        """Update collection statistics"""
        self.stats['metrics_collected'] += 1
        self.stats['last_update'] = datetime.now()
    
    def add_metrics_callback(self, callback):
        """Add metrics callback"""
        self.metrics_callbacks.append(callback)
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return self.stats.copy()
    
    async def get_latest_metrics(self) -> Optional[MarketMetrics]:
        """Get the latest market metrics"""
        try:
            # This would typically fetch from storage
            # For now, collect fresh data
            return await self._collect_metrics()
        except Exception as e:
            logger.error(f"❌ Error getting latest metrics: {e}")
            return None
    
    async def get_metrics_history(self, hours: int = 24) -> List[MarketMetrics]:
        """Get market metrics history"""
        try:
            # This would typically fetch from storage
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"❌ Error getting metrics history: {e}")
            return []
    
    def get_market_correlation_data(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Get market correlation data for analysis"""
        try:
            # This would fetch market metrics and price data for correlation analysis
            return {
                'market_cap_correlation': 0.0,
                'btc_dominance_correlation': 0.0,
                'volume_correlation': 0.0,
                'correlation_period': '24h'
            }
        except Exception as e:
            logger.error(f"❌ Error getting correlation data: {e}")
            return {}

async def test_market_metrics_collector():
    """Test the market metrics collector"""
    config = {
        'collection_interval': 30,  # 30 seconds for testing
        'storage_path': 'test_data'
    }
    
    collector = MarketMetricsCollector(config)
    
    try:
        await collector.start_collection()
        
        # Run for 2 minutes to test collection
        await asyncio.sleep(120)
        
        # Get stats
        stats = collector.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Get latest metrics
        latest = await collector.get_latest_metrics()
        if latest:
            print(f"Latest metrics: {latest.total_market_cap:,.0f} MC, {latest.btc_dominance:.2f}% BTC")
        
    finally:
        await collector.stop_collection()

if __name__ == "__main__":
    asyncio.run(test_market_metrics_collector())
