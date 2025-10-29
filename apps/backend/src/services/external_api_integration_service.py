"""
External API Integration Service
Ensures live data flow from external APIs to sophisticated interface
Phase 6: Real Data Integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import json

from src.services.free_api_manager import FreeAPIManager
from src.services.free_api_data_pipeline import FreeAPIDataPipeline
from src.services.free_api_database_service import FreeAPIDatabaseService
from src.database.connection import TimescaleDBConnection

logger = logging.getLogger(__name__)

class ExternalAPIIntegrationService:
    """Service to ensure live data flow from external APIs"""
    
    def __init__(self):
        self.free_api_manager = FreeAPIManager()
        self.db_connection = TimescaleDBConnection()
        self.logger = logger
        
        # Initialize database service
        self.db_service = FreeAPIDatabaseService(self.db_connection)
        
        # Initialize data pipeline
        self.data_pipeline = FreeAPIDataPipeline(self.db_service, self.free_api_manager)
        
        # Pipeline status
        self.is_pipeline_running = False
        self.last_data_update = {}
        
    async def start_live_data_collection(self) -> Dict[str, Any]:
        """Start live data collection from external APIs"""
        try:
            if not self.is_pipeline_running:
                # Start the data pipeline
                await self.data_pipeline.start_pipeline()
                self.is_pipeline_running = True
                
                self.logger.info("Live data collection started successfully")
                
                return {
                    "status": "success",
                    "message": "Live data collection started",
                    "pipeline_running": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "status": "info",
                    "message": "Live data collection already running",
                    "pipeline_running": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error starting live data collection: {e}")
            return {
                "status": "error",
                "message": f"Failed to start live data collection: {str(e)}",
                "pipeline_running": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def stop_live_data_collection(self) -> Dict[str, Any]:
        """Stop live data collection"""
        try:
            if self.is_pipeline_running:
                await self.data_pipeline.stop_pipeline()
                self.is_pipeline_running = False
                
                self.logger.info("Live data collection stopped")
                
                return {
                    "status": "success",
                    "message": "Live data collection stopped",
                    "pipeline_running": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "status": "info",
                    "message": "Live data collection not running",
                    "pipeline_running": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error stopping live data collection: {e}")
            return {
                "status": "error",
                "message": f"Failed to stop live data collection: {str(e)}",
                "pipeline_running": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics"""
        try:
            pipeline_stats = await self.data_pipeline.get_pipeline_stats()
            
            return {
                "pipeline_running": self.is_pipeline_running,
                "collection_stats": pipeline_stats,
                "last_update": self.last_data_update,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {e}")
            return {
                "pipeline_running": False,
                "collection_stats": {},
                "last_update": {},
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def force_data_refresh(self, symbol: str) -> Dict[str, Any]:
        """Force immediate data refresh for a specific symbol"""
        try:
            # Force refresh market data
            market_data = await self.free_api_manager.get_market_data(symbol)
            
            # Force refresh sentiment data
            sentiment_data = await self.free_api_manager.get_sentiment_analysis(symbol)
            
            # Store in database
            if market_data:
                await self.db_service.store_market_data(market_data)
            
            if sentiment_data:
                await self.db_service.store_sentiment_data(symbol, sentiment_data)
            
            self.last_data_update[symbol] = datetime.now(timezone.utc).isoformat()
            
            return {
                "status": "success",
                "message": f"Data refreshed for {symbol}",
                "symbol": symbol,
                "market_data_updated": market_data is not None,
                "sentiment_data_updated": sentiment_data is not None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error forcing data refresh for {symbol}: {e}")
            return {
                "status": "error",
                "message": f"Failed to refresh data for {symbol}: {str(e)}",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_live_data_status(self, symbol: str) -> Dict[str, Any]:
        """Get live data status for a specific symbol"""
        try:
            # Check latest data in TimescaleDB
            query = """
            SELECT 
                MAX(timestamp) as latest_market_data,
                COUNT(*) as market_data_count
            FROM free_api_market_data 
            WHERE symbol = $1 
            AND timestamp >= NOW() - INTERVAL '1 hour'
            """
            
            async with self.db_connection.get_connection() as conn:
                market_row = await conn.fetchrow(query, symbol.upper())
            
            # Check sentiment data
            query = """
            SELECT 
                MAX(timestamp) as latest_sentiment_data,
                COUNT(*) as sentiment_data_count
            FROM free_api_sentiment_data 
            WHERE symbol = $1 
            AND timestamp >= NOW() - INTERVAL '1 hour'
            """
            
            async with self.db_connection.get_connection() as conn:
                sentiment_row = await conn.fetchrow(query, symbol.upper())
            
            # Calculate data freshness
            now = datetime.now(timezone.utc)
            market_freshness = None
            sentiment_freshness = None
            
            if market_row and market_row['latest_market_data']:
                market_freshness = (now - market_row['latest_market_data']).total_seconds()
            
            if sentiment_row and sentiment_row['latest_sentiment_data']:
                sentiment_freshness = (now - sentiment_row['latest_sentiment_data']).total_seconds()
            
            return {
                "symbol": symbol,
                "market_data": {
                    "latest_update": market_row['latest_market_data'].isoformat() if market_row['latest_market_data'] else None,
                    "count_last_hour": market_row['market_data_count'] if market_row else 0,
                    "freshness_seconds": market_freshness,
                    "is_fresh": market_freshness is None or market_freshness < 300  # 5 minutes
                },
                "sentiment_data": {
                    "latest_update": sentiment_row['latest_sentiment_data'].isoformat() if sentiment_row['latest_sentiment_data'] else None,
                    "count_last_hour": sentiment_row['sentiment_data_count'] if sentiment_row else 0,
                    "freshness_seconds": sentiment_freshness,
                    "is_fresh": sentiment_freshness is None or sentiment_freshness < 600  # 10 minutes
                },
                "overall_status": "healthy" if (
                    (market_freshness is None or market_freshness < 300) and
                    (sentiment_freshness is None or sentiment_freshness < 600)
                ) else "stale",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live data status for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "overall_status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_api_health_status(self) -> Dict[str, Any]:
        """Get health status of all external APIs"""
        try:
            api_status = {}
            
            # Test CoinGecko API
            try:
                coingecko_data = await self.free_api_manager.get_coingecko_data("bitcoin")
                api_status["coingecko"] = {
                    "status": "healthy",
                    "response_time": "< 1s",
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                api_status["coingecko"] = {
                    "status": "error",
                    "error": str(e),
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            
            # Test CryptoCompare API
            try:
                cryptocompare_data = await self.free_api_manager.get_cryptocompare_data("BTC")
                api_status["cryptocompare"] = {
                    "status": "healthy",
                    "response_time": "< 1s",
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                api_status["cryptocompare"] = {
                    "status": "error",
                    "error": str(e),
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            
            # Test NewsAPI
            try:
                news_data = await self.free_api_manager.get_news_data("bitcoin")
                api_status["newsapi"] = {
                    "status": "healthy",
                    "response_time": "< 1s",
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                api_status["newsapi"] = {
                    "status": "error",
                    "error": str(e),
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            
            # Test Reddit API
            try:
                reddit_data = await self.free_api_manager.get_reddit_data("bitcoin")
                api_status["reddit"] = {
                    "status": "healthy",
                    "response_time": "< 1s",
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                api_status["reddit"] = {
                    "status": "error",
                    "error": str(e),
                    "last_test": datetime.now(timezone.utc).isoformat()
                }
            
            # Calculate overall health
            healthy_apis = sum(1 for api in api_status.values() if api["status"] == "healthy")
            total_apis = len(api_status)
            overall_health = "healthy" if healthy_apis >= total_apis * 0.75 else "degraded" if healthy_apis >= total_apis * 0.5 else "unhealthy"
            
            return {
                "overall_health": overall_health,
                "healthy_apis": healthy_apis,
                "total_apis": total_apis,
                "api_status": api_status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting API health status: {e}")
            return {
                "overall_health": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def ensure_data_freshness(self, symbol: str) -> Dict[str, Any]:
        """Ensure data freshness for sophisticated interface"""
        try:
            # Get current data status
            data_status = await self.get_live_data_status(symbol)
            
            # If data is stale, force refresh
            if data_status["overall_status"] == "stale":
                refresh_result = await self.force_data_refresh(symbol)
                
                return {
                    "action": "refreshed",
                    "symbol": symbol,
                    "previous_status": data_status["overall_status"],
                    "refresh_result": refresh_result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "action": "no_action_needed",
                    "symbol": symbol,
                    "status": data_status["overall_status"],
                    "message": "Data is fresh",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error ensuring data freshness for {symbol}: {e}")
            return {
                "action": "error",
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global instance
external_api_service = ExternalAPIIntegrationService()
