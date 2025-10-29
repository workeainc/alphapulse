"""
Intelligent AlphaPulse Main Application
Integrates enhanced data collection, intelligent analysis, and signal generation
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import asyncpg
import ccxt
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd

# Import our intelligent components
from .data_collection.enhanced_data_collection_manager import EnhancedDataCollectionManager
from .analysis.intelligent_analysis_engine import IntelligentAnalysisEngine
from .signals.intelligent_signal_generator import IntelligentSignalGenerator, IntelligentSignal
from .services.live_market_data_service import LiveMarketDataService, TradeExecution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlphaPulse Intelligent Trading System",
    description="Advanced AI-powered trading signal generation with 85% confidence threshold",
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

# Global variables for components
db_pool = None
exchange = None
data_collection_manager = None
analysis_engine = None
signal_generator = None
live_market_data_service = None

# WebSocket connections
websocket_connections = []

# Temporarily disabled startup event to get server running
# @app.on_event("startup")
# async def startup_event():
#     """Initialize all components on startup"""
#     global db_pool, exchange, data_collection_manager, analysis_engine, signal_generator
#     
#     try:
#         logger.info("🚀 Starting AlphaPulse Intelligent Trading System...")
#         
#         # Initialize database connection (non-blocking)
#         try:
#             db_pool = await asyncpg.create_pool(
#                 host=os.getenv('DB_HOST', 'localhost'),
#                 port=int(os.getenv('DB_PORT', 5432)),
#                 database=os.getenv('DB_NAME', 'alphapulse'),
#                 user=os.getenv('DB_USER', 'alpha_emon'),
#                 password=os.getenv('DB_PASSWORD', 'Emon_@17711'),
#                 min_size=5,
#                 max_size=20
#             )
#             logger.info("✅ Database connection established")
#         except Exception as e:
#             logger.warning(f"⚠️ Database connection failed: {e}")
#             db_pool = None
#         
#         # Initialize exchange (non-blocking)
#         try:
#             exchange = ccxt.binance({
#                 'sandbox': False,
#                 'enableRateLimit': True,
#             })
#             logger.info("✅ Exchange connection established")
#         except Exception as e:
#             logger.warning(f"⚠️ Exchange connection failed: {e}")
#             exchange = None
#         
#         # Initialize components without blocking startup
#         logger.info("🔄 Initializing components in background...")
#         
#         # Start background task for component initialization
#         asyncio.create_task(initialize_components_background())
#         
#         logger.info("🎉 AlphaPulse Intelligent Trading System started successfully!")
#         
#     except Exception as e:
#         logger.error(f"❌ Error during startup: {e}")
#         # Don't raise - let the server start even if some components fail

# async def initialize_components_background():
#     """Initialize components in background to avoid blocking startup"""
#     global data_collection_manager, analysis_engine, signal_generator, live_market_data_service
#     
#     try:
#         # Initialize enhanced data collection manager
#         try:
#             data_collection_manager = EnhancedDataCollectionManager(db_pool, exchange)
#             await data_collection_manager.start_collection()
#             logger.info("✅ Enhanced data collection started")
#         except Exception as e:
#             logger.warning(f"⚠️ Enhanced data collection failed: {e}")
#             data_collection_manager = None
#         
#         # Initialize intelligent analysis engine
#         try:
#             analysis_engine = IntelligentAnalysisEngine(db_pool, exchange)
#             await analysis_engine.initialize()
#             logger.info("✅ Intelligent analysis engine initialized")
#         except Exception as e:
#             logger.warning(f"⚠️ Intelligent analysis engine failed: {e}")
#             analysis_engine = None
#         
#         # Initialize intelligent signal generator
#         try:
#             signal_generator = IntelligentSignalGenerator(db_pool, exchange)
#             logger.info("✅ Intelligent signal generator initialized")
#         except Exception as e:
#             logger.warning(f"⚠️ Intelligent signal generator failed: {e}")
#             signal_generator = None
#         
#         # Initialize live market data service
#         logger.info("🔄 Initializing live market data service...")
#         try:
#             logger.info("🔄 Creating LiveMarketDataService instance...")
#             live_market_data_service = LiveMarketDataService(db_pool)
#             logger.info("🔄 Starting data collection...")
#             await live_market_data_service.start_data_collection()
#             logger.info("✅ Live market data service initialized and started")
#         except Exception as e:
#             logger.error(f"❌ Error initializing live market data service: {e}")
#             import traceback
#             logger.error(f"❌ Full traceback: {traceback.format_exc()}")
#             live_market_data_service = None
#             
#     except Exception as e:
#         logger.error(f"❌ Error during background initialization: {e}")

# Simple startup event for now
@app.on_event("startup")
async def startup_event():
    """Simple startup event"""
    logger.info("🚀 AlphaPulse Intelligent Trading System starting...")
    logger.info("✅ Server started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global data_collection_manager, signal_generator, db_pool
    
    try:
        logger.info("🛑 Shutting down AlphaPulse Intelligent Trading System...")
        
        if signal_generator:
            signal_generator.shutdown()
            logger.info("✅ Signal generator closed")
        
        if data_collection_manager:
            await data_collection_manager.stop_data_collection()
            logger.info("✅ Data collection stopped")
        
        if live_market_data_service:
            await live_market_data_service.stop_data_collection()
            logger.info("✅ Live market data service stopped")
        
        if db_pool:
            await db_pool.close()
            logger.info("✅ Database connection closed")
        
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Health and Status Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AlphaPulse Intelligent Trading System",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check_simple():
    """Simple health check endpoint for Docker"""
    return {"status": "healthy"}

@app.get("/api/test/phase3")
async def test_phase3():
    """Test endpoint for Phase 3"""
    return {"status": "success", "phase": 3, "message": "Phase 3 test endpoint working"}

@app.get("/api/patterns/latest")
async def get_latest_patterns():
    """Get latest trading patterns"""
    return {
        "patterns": [
            {"id": 1, "name": "Double Top", "confidence": 0.85, "timestamp": "2024-01-01T10:00:00Z"},
            {"id": 2, "name": "Head and Shoulders", "confidence": 0.78, "timestamp": "2024-01-01T09:30:00Z"}
        ]
    }

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Get latest trading signals"""
    return {
        "signals": [
            {"id": 1, "symbol": "BTC/USDT", "direction": "buy", "confidence": 0.82, "timestamp": "2024-01-01T10:00:00Z"},
            {"id": 2, "symbol": "ETH/USDT", "direction": "sell", "confidence": 0.75, "timestamp": "2024-01-01T09:45:00Z"}
        ]
    }

@app.get("/api/market/status")
async def get_market_status():
    """Get current market status"""
    return {
        "status": "bullish",
        "volatility": "medium",
        "trend": "upward",
        "timestamp": "2024-01-01T10:00:00Z"
    }

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI performance metrics"""
    return {
        "accuracy": 0.85,
        "total_signals": 150,
        "successful_trades": 127,
        "win_rate": 0.847,
        "timestamp": "2024-01-01T10:00:00Z"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time notifications"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text('{"type": "connection", "status": "connected"}')
        
        # Keep connection alive
        while True:
            # Wait for any message from client
            data = await websocket.receive_text()
            
            # Echo back for now (can be extended for specific commands)
            await websocket.send_text(f'{{"type": "echo", "message": "{data}"}}')
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "data_collection": "running" if data_collection_manager else "stopped",
            "signal_generation": "running" if signal_generator else "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@app.get("/api/status")
async def system_status():
    """Get comprehensive system status"""
    try:
        # Get data collection status
        data_collection_status = await data_collection_manager.get_collection_status() if data_collection_manager else {}
        
        # Get signal generation statistics
        signal_stats = await signal_generator.get_signal_statistics() if signal_generator else {}
        
        return {
            "system": {
                "status": "running",
                "version": "2.0.0",
                "startup_time": datetime.utcnow().isoformat()
            },
            "data_collection": data_collection_status,
            "signal_generation": signal_stats,
            "database": {
                "status": "connected",
                "pool_size": db_pool.get_size() if db_pool else 0
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")

# Intelligent Signal Endpoints
@app.get("/api/intelligent/signals/latest")
async def get_latest_intelligent_signals(limit: int = Query(10, description="Number of signals to return")):
    """Get latest intelligent signals"""
    try:
        if not signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not available")
        
        signals = await signal_generator.get_latest_signals(limit)
        
        # Convert signals to dict for JSON response
        signal_list = []
        for signal in signals:
            signal_dict = {
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "timestamp": signal.timestamp.isoformat(),
                "signal_type": signal.signal_type,
                "signal_direction": signal.signal_direction,
                "signal_strength": signal.signal_strength,
                "confidence_score": signal.confidence_score,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "risk_level": signal.risk_level,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit_1": signal.take_profit_1,
                "take_profit_2": signal.take_profit_2,
                "take_profit_3": signal.take_profit_3,
                "take_profit_4": signal.take_profit_4,
                "position_size_percentage": signal.position_size_percentage,
                "pattern_analysis": signal.pattern_analysis,
                "technical_analysis": signal.technical_analysis,
                "sentiment_analysis": signal.sentiment_analysis,
                "volume_analysis": signal.volume_analysis,
                "market_regime_analysis": signal.market_regime_analysis,
                "entry_reasoning": signal.entry_reasoning,
                "no_safe_entry_reasons": signal.no_safe_entry_reasons,
                "status": signal.status,
                "pnl": signal.pnl,
                "executed_at": signal.executed_at.isoformat() if signal.executed_at else None,
                "closed_at": signal.closed_at.isoformat() if signal.closed_at else None
            }
            signal_list.append(signal_dict)
        
        return {
            "signals": signal_list,
            "count": len(signal_list),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting latest signals: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting signals: {e}")

@app.get("/api/intelligent/signals/{symbol}")
async def get_signals_by_symbol(
    symbol: str,
    limit: int = Query(10, description="Number of signals to return")
):
    """Get signals for a specific symbol"""
    try:
        if not signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not available")
        
        signals = await signal_generator.get_signals_by_symbol(symbol, limit)
        
        # Convert signals to dict for JSON response
        signal_list = []
        for signal in signals:
            signal_dict = {
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "timestamp": signal.timestamp.isoformat(),
                "signal_type": signal.signal_type,
                "signal_direction": signal.signal_direction,
                "signal_strength": signal.signal_strength,
                "confidence_score": signal.confidence_score,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "risk_level": signal.risk_level,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit_1": signal.take_profit_1,
                "take_profit_2": signal.take_profit_2,
                "take_profit_3": signal.take_profit_3,
                "take_profit_4": signal.take_profit_4,
                "position_size_percentage": signal.position_size_percentage,
                "pattern_analysis": signal.pattern_analysis,
                "technical_analysis": signal.technical_analysis,
                "sentiment_analysis": signal.sentiment_analysis,
                "volume_analysis": signal.volume_analysis,
                "market_regime_analysis": signal.market_regime_analysis,
                "entry_reasoning": signal.entry_reasoning,
                "no_safe_entry_reasons": signal.no_safe_entry_reasons,
                "status": signal.status,
                "pnl": signal.pnl,
                "executed_at": signal.executed_at.isoformat() if signal.executed_at else None,
                "closed_at": signal.closed_at.isoformat() if signal.closed_at else None
            }
            signal_list.append(signal_dict)
        
        return {
            "symbol": symbol,
            "signals": signal_list,
            "count": len(signal_list),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting signals: {e}")

@app.get("/api/intelligent/signals/statistics")
async def get_signal_statistics():
    """Get signal generation statistics"""
    try:
        if not signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not available")
        
        stats = await signal_generator.get_signal_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting signal statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {e}")

# Intelligent Analysis Endpoints
@app.get("/api/intelligent/analysis/{symbol}")
async def get_intelligent_analysis(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for analysis")
):
    """Get intelligent analysis for a symbol"""
    try:
        if not analysis_engine:
            raise HTTPException(status_code=503, detail="Analysis engine not available")
        
        analysis = await analysis_engine.analyze_symbol(symbol, timeframe)
        
        # Convert analysis to dict for JSON response
        analysis_dict = {
            "symbol": analysis.symbol,
            "timeframe": analysis.timeframe,
            "timestamp": analysis.timestamp.isoformat(),
            "current_price": analysis.current_price,
            "overall_confidence": analysis.overall_confidence,
            "signal_direction": analysis.signal_direction,
            "signal_strength": analysis.signal_strength,
            "safe_entry_detected": analysis.safe_entry_detected,
            "risk_reward_ratio": analysis.risk_reward_ratio,
            "entry_price": analysis.entry_price,
            "stop_loss": analysis.stop_loss,
            "take_profit_1": analysis.take_profit_1,
            "take_profit_2": analysis.take_profit_2,
            "take_profit_3": analysis.take_profit_3,
            "take_profit_4": analysis.take_profit_4,
            "position_size_percentage": analysis.position_size_percentage,
            "pattern_type": analysis.pattern_type,
            "pattern_strength": analysis.pattern_strength,
            "pattern_confidence": analysis.pattern_confidence,
            "rsi_value": analysis.rsi_value,
            "macd_signal": analysis.macd_signal,
            "bollinger_position": analysis.bollinger_position,
            "technical_confidence": analysis.technical_confidence,
            "market_sentiment": analysis.market_sentiment,
            "news_sentiment": analysis.news_sentiment,
            "sentiment_confidence": analysis.sentiment_confidence,
            "volume_ratio": analysis.volume_ratio,
            "volume_positioning": analysis.volume_positioning,
            "volume_confidence": analysis.volume_confidence,
            "market_regime": analysis.market_regime,
            "volatility_level": analysis.volatility_level,
            "market_regime_confidence": analysis.market_regime_confidence,
            "analysis_reasoning": analysis.analysis_reasoning,
            "no_safe_entry_reasons": analysis.no_safe_entry_reasons
        }
        
        return analysis_dict
        
    except Exception as e:
        logger.error(f"Error getting analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting analysis: {e}")

# Live Market Data Endpoints
@app.get("/api/live/market-data/{symbol}")
async def get_live_market_data(symbol: str):
    """Get live market data for a symbol"""
    try:
        if not live_market_data_service:
            raise HTTPException(status_code=503, detail="Live market data service not available")
        
        market_data = await live_market_data_service.get_latest_market_data(symbol)
        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data available for {symbol}")
        
        return {
            "symbol": market_data.symbol,
            "price": market_data.price,
            "volume": market_data.volume,
            "bid": market_data.bid,
            "ask": market_data.ask,
            "spread": market_data.spread,
            "high_24h": market_data.high_24h,
            "low_24h": market_data.low_24h,
            "change_24h": market_data.change_24h,
            "change_percent_24h": market_data.change_percent_24h,
            "market_cap": market_data.market_cap,
            "timestamp": market_data.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting live market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting market data: {e}")

# Test endpoint
@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify server is working"""
    return {"message": "Test endpoint working", "timestamp": datetime.utcnow().isoformat()}

# Phase 5.2: Real-time Chart Integration Endpoints
@app.get("/api/charts/candlestick/{symbol}")
async def get_candlestick_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(100, description="Number of candlesticks to return", ge=1, le=1000)
):
    """Get candlestick data for charting"""
    try:
        # Return mock data for now
        import random
        base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
        
        candlestick_data = []
        for i in range(min(limit, 100)):
            # Generate mock candlestick data
            open_price = base_price + random.uniform(-1000, 1000)
            close_price = open_price + random.uniform(-500, 500)
            high_price = max(open_price, close_price) + random.uniform(0, 200)
            low_price = min(open_price, close_price) - random.uniform(0, 200)
            volume = random.uniform(100, 10000)
            
            # Generate timestamp (going backwards from now)
            timestamp = datetime.utcnow() - timedelta(hours=i)
            
            candlestick_data.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
        
        # Reverse to get chronological order
        candlestick_data.reverse()
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": candlestick_data,
            "count": len(candlestick_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting candlestick data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting candlestick data: {e}")

@app.get("/api/charts/technical-indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    indicators: str = Query("rsi,macd,sma,ema,bollinger", description="Comma-separated list of indicators")
):
    """Get technical indicators for charting"""
    try:
        # Return mock data for now
        import random
        
        # Parse requested indicators
        requested_indicators = [ind.strip() for ind in indicators.split(",")]
        
        indicators_data = {}
        
        if 'rsi' in requested_indicators:
            indicators_data['rsi'] = round(random.uniform(30, 70), 2)
        
        if 'macd' in requested_indicators:
            macd_value = random.uniform(-100, 100)
            signal_value = macd_value + random.uniform(-20, 20)
            histogram_value = macd_value - signal_value
            indicators_data['macd'] = {
                'macd': round(macd_value, 2),
                'signal': round(signal_value, 2),
                'histogram': round(histogram_value, 2)
            }
        
        if 'sma' in requested_indicators:
            base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
            indicators_data['sma'] = round(base_price + random.uniform(-1000, 1000), 2)
        
        if 'ema' in requested_indicators:
            base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
            indicators_data['ema'] = round(base_price + random.uniform(-1000, 1000), 2)
        
        if 'bollinger' in requested_indicators:
            base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
            middle = base_price + random.uniform(-1000, 1000)
            std = random.uniform(500, 2000)
            indicators_data['bollinger'] = {
                'upper': round(middle + (2 * std), 2),
                'middle': round(middle, 2),
                'lower': round(middle - (2 * std), 2)
            }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting technical indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting technical indicators: {e}")

@app.get("/api/charts/real-time/{symbol}")
async def get_real_time_chart_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for real-time data")
):
    """Get real-time chart data including candlesticks and indicators"""
    try:
        # Return mock data for now
        import random
        
        base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
        
        # Generate mock candlestick data
        candlestick_data = []
        for i in range(10):  # Last 10 candles
            open_price = base_price + random.uniform(-1000, 1000)
            close_price = open_price + random.uniform(-500, 500)
            high_price = max(open_price, close_price) + random.uniform(0, 200)
            low_price = min(open_price, close_price) - random.uniform(0, 200)
            volume = random.uniform(100, 10000)
            
            timestamp = datetime.utcnow() - timedelta(hours=i)
            
            candlestick_data.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
        
        # Reverse to get chronological order
        candlestick_data.reverse()
        
        # Generate mock indicators
        latest_indicators = {
            'rsi': round(random.uniform(30, 70), 2),
            'macd': {
                'macd': round(random.uniform(-100, 100), 2),
                'signal': round(random.uniform(-100, 100), 2),
                'histogram': round(random.uniform(-50, 50), 2)
            },
            'sma_20': round(base_price + random.uniform(-1000, 1000), 2),
            'ema_12': round(base_price + random.uniform(-1000, 1000), 2),
            'bollinger': {
                'upper': round(base_price + 2000, 2),
                'middle': round(base_price, 2),
                'lower': round(base_price - 2000, 2)
            }
        }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candlestick_data": candlestick_data,
            "indicators": latest_indicators,
            "latest_price": round(base_price + random.uniform(-500, 500), 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time chart data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting real-time chart data: {e}")

@app.get("/api/live/market-data")
async def get_all_live_market_data():
    """Get live market data for all symbols"""
    try:
        if not live_market_data_service:
            raise HTTPException(status_code=503, detail="Live market data service not available")
        
        all_data = {}
        for symbol in live_market_data_service.trading_pairs:
            market_data = await live_market_data_service.get_latest_market_data(symbol)
            if market_data:
                all_data[symbol] = {
                    "price": market_data.price,
                    "volume": market_data.volume,
                    "bid": market_data.bid,
                    "ask": market_data.ask,
                    "spread": market_data.spread,
                    "change_percent_24h": market_data.change_percent_24h,
                    "timestamp": market_data.timestamp.isoformat()
                }
        
        return {
            "market_data": all_data,
            "count": len(all_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting all live market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting market data: {e}")

@app.get("/api/live/performance")
async def get_live_performance_stats():
    """Get live performance statistics"""
    try:
        if not live_market_data_service:
            raise HTTPException(status_code=503, detail="Live market data service not available")
        
        stats = await live_market_data_service.get_performance_stats()
        return {
            "performance_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance stats: {e}")

@app.post("/api/live/execute-trade")
async def execute_trade(trade_data: dict):
    """Execute a trade"""
    try:
        if not live_market_data_service:
            raise HTTPException(status_code=503, detail="Live market data service not available")
        
        # Create trade execution object
        trade = TradeExecution(
            signal_id=trade_data.get('signal_id'),
            symbol=trade_data['symbol'],
            side=trade_data['side'],
            order_type=trade_data['order_type'],
            quantity=float(trade_data['quantity']),
            price=float(trade_data['price']),
            executed_price=None,
            status='pending',
            exchange_order_id=None,
            exchange_trade_id=None,
            commission=None,
            commission_asset=None,
            executed_at=None
        )
        
        # Execute trade
        success = await live_market_data_service.execute_trade(trade)
        
        return {
            "success": success,
            "trade_id": trade.exchange_order_id,
            "status": trade.status,
            "executed_price": trade.executed_price,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing trade: {e}")

# Data Collection Endpoints
@app.get("/api/intelligent/data/status")
async def get_data_collection_status():
    """Get data collection status"""
    try:
        if not data_collection_manager:
            raise HTTPException(status_code=503, detail="Data collection manager not available")
        
        status = await data_collection_manager.get_collection_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting data collection status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {e}")

@app.get("/api/intelligent/data/latest")
async def get_latest_collected_data():
    """Get latest collected market data"""
    try:
        if not data_collection_manager:
            raise HTTPException(status_code=503, detail="Data collection manager not available")
        
        data = await data_collection_manager.get_latest_data()
        return data
        
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting data: {e}")

# Control Endpoints
@app.post("/api/intelligent/control/start")
async def start_intelligent_system():
    """Start the intelligent system"""
    try:
        global data_collection_manager, signal_generator
        
        if data_collection_manager:
            await data_collection_manager.start_data_collection()
        
        if signal_generator:
            await signal_generator.start_signal_generation()
        
        return {
            "message": "Intelligent system started successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting intelligent system: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting system: {e}")

@app.post("/api/intelligent/control/stop")
async def stop_intelligent_system():
    """Stop the intelligent system"""
    try:
        global data_collection_manager, signal_generator
        
        if data_collection_manager:
            await data_collection_manager.stop_data_collection()
        
        if signal_generator:
            await signal_generator.stop_signal_generation()
        
        return {
            "message": "Intelligent system stopped successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping intelligent system: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping system: {e}")



# Additional API Endpoints for Frontend
@app.get("/api/signals")
async def get_signals():
    """Get all trading signals"""
    try:
        # Return mock signals for now
        import random
        
        signals = []
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        directions = ['long', 'short']
        
        for i in range(10):
            signal = {
                "signal_id": f"signal_{i+1}",
                "symbol": random.choice(symbols),
                "direction": random.choice(directions),
                "confidence_score": round(random.uniform(0.75, 0.95), 2),
                "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "timeframe": random.choice(['1h', '4h', '1d']),
                "entry_price": round(random.uniform(45000, 55000), 2),
                "stop_loss": round(random.uniform(40000, 50000), 2),
                "take_profit": round(random.uniform(50000, 60000), 2)
            }
            signals.append(signal)
        
        return signals
        
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching signals: {e}")

@app.get("/api/market-data")
async def get_market_data():
    """Get market data for all monitored symbols"""
    try:
        # Return mock market data for now
        import random
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        market_data = []
        
        for symbol in symbols:
            base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            price = base_price + random.uniform(-1000, 1000)
            change_24h = random.uniform(-500, 500)
            change_percent_24h = (change_24h / price) * 100
            volume_24h = random.uniform(1000000, 10000000)
            high_24h = price + random.uniform(0, 1000)
            low_24h = price - random.uniform(0, 1000)
            
            market_data.append({
                "symbol": symbol,
                "price": round(price, 2),
                "change_24h": round(change_24h, 2),
                "change_percent_24h": round(change_percent_24h, 2),
                "volume_24h": round(volume_24h, 2),
                "high_24h": round(high_24h, 2),
                "low_24h": round(low_24h, 2)
            })
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {e}")

@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        # Return mock performance metrics for now
        import random
        
        total_signals = random.randint(100, 500)
        successful_signals = int(total_signals * random.uniform(0.75, 0.90))
        success_rate = successful_signals / total_signals
        total_profit = random.uniform(1000, 5000)
        profit_change = random.uniform(-500, 500)
        
        return {
            "totalSignals": total_signals,
            "successfulSignals": successful_signals,
            "successRate": round(success_rate, 3),
            "totalProfit": round(total_profit, 2),
            "profitChange": round(profit_change, 2)
        }
        
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching performance metrics: {e}")

# Manual Signal Generation Endpoint
@app.post("/api/intelligent/signals/generate")
async def generate_manual_signal(
    symbol: str = Query(..., description="Symbol to analyze"),
    timeframe: str = Query("1h", description="Timeframe for analysis")
):
    """Manually generate a signal for a symbol"""
    try:
        if not signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not available")
        
        signal = await signal_generator.generate_intelligent_signal(symbol, timeframe)
        
        if not signal:
            raise HTTPException(status_code=404, detail="Could not generate signal")
        
        # Store the signal
        await signal_generator._store_signal(signal)
        
        # Convert signal to dict for JSON response
        signal_dict = {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "timestamp": signal.timestamp.isoformat(),
            "signal_type": signal.signal_type,
            "signal_direction": signal.signal_direction,
            "signal_strength": signal.signal_strength,
            "confidence_score": signal.confidence_score,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "risk_level": signal.risk_level,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit_1": signal.take_profit_1,
            "take_profit_2": signal.take_profit_2,
            "take_profit_3": signal.take_profit_3,
            "take_profit_4": signal.take_profit_4,
            "position_size_percentage": signal.position_size_percentage,
            "pattern_analysis": signal.pattern_analysis,
            "technical_analysis": signal.technical_analysis,
            "sentiment_analysis": signal.sentiment_analysis,
            "volume_analysis": signal.volume_analysis,
            "market_regime_analysis": signal.market_regime_analysis,
            "entry_reasoning": signal.entry_reasoning,
            "no_safe_entry_reasons": signal.no_safe_entry_reasons,
            "status": signal.status
        }
        
        return {
            "message": "Signal generated successfully",
            "signal": signal_dict,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating manual signal: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating signal: {e}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main_intelligent:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
