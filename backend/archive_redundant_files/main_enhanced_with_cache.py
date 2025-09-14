#!/usr/bin/env python3
"""
Enhanced AlphaPlus Main Application with Cache Integration
Ultra-low latency data processing with Redis cache and TimescaleDB
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import enhanced components
from services.enhanced_cache_manager import EnhancedCacheManager
from services.enhanced_data_pipeline import EnhancedDataPipeline
from services.enhanced_websocket_service import EnhancedWebSocketService

# Import existing components
from core.websocket_binance import BinanceWebSocketClient
from database.connection import TimescaleDBConnection

# Import Phase 4 sentiment components
from ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from app.services.enhanced_sentiment_service import EnhancedSentimentService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlphaPlus Enhanced with Cache",
    description="Ultra-low latency trading system with Redis cache integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
cache_manager = None
data_pipeline = None
websocket_service = None
db_connection = None
binance_websocket = None
sentiment_analyzer = None
sentiment_service = None

# Configuration
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']
REDIS_URL = "redis://localhost:6379"

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced AlphaPlus system with cache integration"""
    global cache_manager, data_pipeline, websocket_service, db_connection, binance_websocket
    
    try:
        logger.info("🚀 Starting AlphaPlus Enhanced with Cache Integration...")
        
        # Initialize cache manager
        cache_manager = EnhancedCacheManager(
            redis_url=REDIS_URL,
            enable_redis=True,
            max_memory_cache_size=10000
        )
        logger.info("✅ Cache manager initialized")
        
        # Initialize database connection
        db_connection = TimescaleDBConnection({
            'host': 'postgres',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        await db_connection.initialize()
        logger.info("✅ Database connection established")
        
        # Initialize data pipeline
        data_pipeline = EnhancedDataPipeline(
            redis_url=REDIS_URL,
            database_config={
                'host': 'postgres',
                'port': 5432,
                'database': 'alphapulse',
                'user': 'alpha_emon',
                'password': 'Emon_@17711'
            },
            symbols=SYMBOLS,
            timeframes=TIMEFRAMES,
            enable_cache=True
        )
        await data_pipeline.initialize()
        logger.info("✅ Data pipeline initialized")
        
        # Initialize WebSocket service
        
        # Initialize Phase 4 sentiment services
        import redis.asyncio as redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Initialize sentiment analyzer
        sentiment_analyzer = EnhancedSentimentAnalyzer(db_connection.pool, redis_client)
        logger.info("✅ Enhanced sentiment analyzer initialized")
        
        # Initialize sentiment service
        sentiment_service = EnhancedSentimentService(db_connection.pool, redis_client)
        logger.info("✅ Enhanced sentiment service initialized")
        websocket_service = EnhancedWebSocketService(
            cache_manager=cache_manager,
            data_pipeline=data_pipeline,
            symbols=SYMBOLS,
            timeframes=TIMEFRAMES
        )
        await websocket_service.start()
        logger.info("✅ WebSocket service started")
        
        # Initialize Binance WebSocket client
        binance_websocket = BinanceWebSocketClient(
            symbols=[s.replace('/', '') for s in SYMBOLS],
            timeframes=TIMEFRAMES
        )
        await binance_websocket.connect()
        logger.info("✅ Binance WebSocket client initialized")
        
        # Start data collection
        asyncio.create_task(start_enhanced_data_collection())
        logger.info("✅ Enhanced data collection started")
        
        logger.info("🎉 AlphaPlus Enhanced with Cache Integration fully activated!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start enhanced system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the enhanced AlphaPlus system"""
    global cache_manager, data_pipeline, websocket_service, db_connection, binance_websocket
    
    try:
        logger.info("🛑 Shutting down AlphaPlus Enhanced system...")
        
        # Stop WebSocket service
        if websocket_service:
            await websocket_service.stop()
        
        # Close data pipeline
        if data_pipeline:
            await data_pipeline.close()
        
        # Close cache manager
        if cache_manager:
            await cache_manager.close()
        
        # Close database connection
        if db_connection:
            await db_connection.close()
        
        # Close Binance WebSocket
        if binance_websocket:
            await binance_websocket.disconnect()
        
        logger.info("✅ AlphaPlus Enhanced system shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

async def start_enhanced_data_collection():
    """Start enhanced real-time data collection with cache integration"""
    global binance_websocket, data_pipeline, websocket_service
    
    try:
        logger.info("🔄 Starting enhanced data collection with cache integration...")
        
        async for data in binance_websocket.listen():
            try:
                # Process incoming WebSocket data
                if 'e' in data and data['e'] == 'kline':
                    processed_data = await process_binance_data(data)
                    
                    if processed_data:
                        # Process through enhanced pipeline
                        success = await data_pipeline.process_market_data(processed_data)
                        
                        if success:
                            # Queue for WebSocket broadcast
                            await websocket_service.queue_market_data(processed_data)
                            
                            # Store real-time data in cache
                            real_time_data = {
                                'symbol': processed_data['symbol'],
                                'timeframe': processed_data['timeframe'],
                                'timestamp': processed_data['timestamp'].isoformat(),
                                'price': processed_data['close'],
                                'volume': processed_data['volume'],
                                'indicators': processed_data.get('indicators', {}),
                                'source': 'binance_websocket'
                            }
                            await data_pipeline.store_real_time_data(
                                processed_data['symbol'], 
                                processed_data['timeframe'], 
                                real_time_data
                            )
                
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket data: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error in enhanced data collection: {e}")

async def process_binance_data(data: Dict) -> Optional[Dict]:
    """Process Binance WebSocket data"""
    try:
        kline = data['k']
        symbol = data['s']
        timeframe = convert_timeframe(kline['i'])
        
        processed_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_complete': kline['x'],
            'source': 'binance_websocket'
        }
        
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Error processing Binance data: {e}")
        return None

def convert_timeframe(interval: str) -> str:
    """Convert Binance interval to standard timeframe"""
    timeframe_map = {
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h',
        '12h': '12h', '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
    }
    return timeframe_map.get(interval, interval)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AlphaPlus Enhanced with Cache Integration",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "cache_manager": cache_manager is not None,
            "data_pipeline": data_pipeline is not None,
            "websocket_service": websocket_service is not None,
            "database": db_connection is not None,
            "binance_websocket": binance_websocket is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache performance statistics"""
    try:
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        stats = await cache_manager.get_cache_stats()
        return {
            "cache_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@app.get("/api/pipeline/stats")
async def get_pipeline_stats():
    """Get data pipeline performance statistics"""
    try:
        if not data_pipeline:
            raise HTTPException(status_code=503, detail="Data pipeline not available")
        
        stats = await data_pipeline.get_pipeline_stats()
        return {
            "pipeline_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline stats: {str(e)}")

@app.get("/api/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket service statistics"""
    try:
        if not websocket_service:
            raise HTTPException(status_code=503, detail="WebSocket service not available")
        
        stats = await websocket_service.get_service_stats()
        return {
            "websocket_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket stats: {str(e)}")

@app.get("/api/market/data")
async def get_market_data(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    limit: int = Query(100, description="Number of data points")
):
    """Get market data from cache"""
    try:
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        data = await cache_manager.get_market_data(symbol, timeframe, limit)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data or [],
            "count": len(data) if data else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market data: {str(e)}")

@app.get("/api/signals")
async def get_signals(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    limit: int = Query(50, description="Number of signals")
):
    """Get trading signals from cache"""
    try:
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        signals = await cache_manager.get_signals(symbol, timeframe, limit)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signals": signals or [],
            "count": len(signals) if signals else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {str(e)}")

@app.get("/api/real-time/{symbol}/{timeframe}")
async def get_real_time_data(symbol: str, timeframe: str):
    """Get real-time data from cache"""
    try:
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        data = await cache_manager.get_real_time_data(symbol, timeframe)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time data: {str(e)}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with cache integration"""
    try:
        if not websocket_service:
            await websocket.close(code=1000, reason="WebSocket service not available")
            return
        
        await websocket_service.handle_connection(websocket)
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except Exception:
            pass

# Phase 4 Sentiment Analysis Endpoints
@app.get("/api/sentiment/predictions/{symbol}")
async def get_price_prediction(
    symbol: str,
    time_horizon: str = Query('4h', description="Prediction time horizon: 1h, 4h, 1d, 1w")
):
    """Get price movement prediction for a symbol"""
    try:
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        prediction = await sentiment_service.get_price_prediction(symbol, time_horizon)
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction available for {symbol}"
            )
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price prediction for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving price prediction for {symbol}"
        )

@app.get("/api/sentiment/cross-asset/{primary_symbol}")
async def get_cross_asset_analysis(
    primary_symbol: str,
    symbols: str = Query('ETH/USDT,BNB/USDT', description="Comma-separated list of symbols")
):
    """Get cross-asset sentiment analysis"""
    try:
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        symbol_list = [s.strip() for s in symbols.split(',')]
        analysis = await sentiment_service.get_cross_asset_analysis(primary_symbol, symbol_list)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"No cross-asset analysis available for {primary_symbol}"
            )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cross-asset analysis for {primary_symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cross-asset analysis for {primary_symbol}"
        )

@app.get("/api/sentiment/model-performance")
async def get_model_performance():
    """Get model performance summary"""
    try:
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        performance = await sentiment_service.get_model_performance_summary()
        return performance
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model performance"
        )

@app.get("/api/sentiment/summary/{symbol}")
async def get_sentiment_summary(symbol: str):
    """Get sentiment summary for a symbol"""
    try:
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        summary = await sentiment_service.get_latest_sentiment_summary(symbol)
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment summary available for {symbol}"
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment summary for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sentiment summary for {symbol}"
        )

@app.post("/api/cache/clear")
async def clear_cache(pattern: str = None):
    """Clear cache entries"""
    try:
        if not cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        cleared_count = await cache_manager.clear_cache(pattern)
        
        return {
            "message": f"Cleared {cleared_count} cache entries",
            "cleared_count": cleared_count,
            "pattern": pattern,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/api/system/overview")
async def get_system_overview():
    """Get comprehensive system overview"""
    try:
        overview = {
            "system": {
                "name": "AlphaPlus Enhanced with Cache",
                "version": "2.0.0",
                "status": "running",
                "uptime": "N/A"  # Could add uptime tracking
            },
            "components": {
                "cache_manager": cache_manager is not None,
                "data_pipeline": data_pipeline is not None,
                "websocket_service": websocket_service is not None,
                "database": db_connection is not None,
                "binance_websocket": binance_websocket is not None
            },
            "configuration": {
                "symbols": SYMBOLS,
                "timeframes": TIMEFRAMES,
                "redis_url": REDIS_URL
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add component stats if available
        if cache_manager:
            try:
                overview["cache_stats"] = await cache_manager.get_cache_stats()
            except Exception:
                overview["cache_stats"] = "unavailable"
        
        if data_pipeline:
            try:
                overview["pipeline_stats"] = await data_pipeline.get_pipeline_stats()
            except Exception:
                overview["pipeline_stats"] = "unavailable"
        
        if websocket_service:
            try:
                overview["websocket_stats"] = await websocket_service.get_service_stats()
            except Exception:
                overview["websocket_stats"] = "unavailable"
        
        return overview
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system overview: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting AlphaPlus Enhanced with Cache Integration...")
        uvicorn.run(
            "main_enhanced_with_cache:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down AlphaPlus Enhanced system...")
    except Exception as e:
        logger.error(f"❌ Failed to start AlphaPlus Enhanced system: {e}")
        raise
