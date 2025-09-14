"""
Unified Main Application for AlphaPlus - Fixed Integration
FastAPI application with proper service management and error handling
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime, timedelta
import random

# Import core components
from app.core.config import settings, get_settings
from app.core.database_manager import db_manager, initialize_database, close_database
from app.core.service_manager import service_manager, initialize_all_services, shutdown_all_services, register_service

# Import services
from app.services.market_data_service import MarketDataService
from app.services.sentiment_service import SentimentService
from app.services.risk_manager import RiskManager
from app.services.live_market_data_service import LiveMarketDataService

# Import strategies
from app.strategies.strategy_manager import StrategyManager
from app.strategies.real_time_signal_generator import RealTimeSignalGenerator

# Import data components
from app.data.real_time_processor import RealTimeCandlestickProcessor
from app.core.unified_websocket_client import UnifiedWebSocketClient, UnifiedWebSocketManager

# Import AI components
try:
    from ai.real_time_pipeline import RealTimePipeline
    from ai.signal_generator import SignalGenerator
    from strategies.advanced_pattern_detector import AdvancedPatternDetector
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI components not available - AI features disabled")

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Advanced algorithmic trading system with AI/ML capabilities",
    version=settings.app_version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
websocket_connections = []

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    try:
        logger.info("🚀 Starting AlphaPlus Trading System...")
        
        # Validate configuration
        if not settings.validate_configuration():
            logger.error("❌ Configuration validation failed")
            raise RuntimeError("Invalid configuration")
        
        # Initialize database
        logger.info("📊 Initializing database...")
        db_config = {
            'host': settings.database.host,
            'port': settings.database.port,
            'database': settings.database.database,
            'username': settings.database.username,
            'password': settings.database.password,
            'min_size': settings.database.min_size,
            'max_size': settings.database.max_size,
            'command_timeout': settings.database.command_timeout
        }
        
        if not await initialize_database(db_config):
            logger.error("❌ Database initialization failed")
            raise RuntimeError("Database initialization failed")
        
        # Register services with proper dependencies
        logger.info("🔧 Registering services...")
        
        # Core services (no dependencies)
        register_service("database", db_manager)
        
        # Data services (depend on database)
        market_data_service = MarketDataService(db_manager)
        register_service("market_data", market_data_service, dependencies=["database"])
        
        live_market_data_service = LiveMarketDataService(db_manager)
        register_service("live_market_data", live_market_data_service, dependencies=["database"])
        
        # Strategy services (depend on market data)
        strategy_manager = StrategyManager(db_manager)
        register_service("strategy_manager", strategy_manager, dependencies=["market_data"])
        
        real_time_signal_generator = RealTimeSignalGenerator(db_manager)
        register_service("signal_generator", real_time_signal_generator, dependencies=["market_data"])
        
        # AI services (if available)
        if AI_AVAILABLE and settings.enable_ai:
            try:
                ai_pipeline = RealTimePipeline(
                    max_queue_size=10000,
                    num_workers=4,
                    enable_parallel_processing=True,
                    enable_caching=True
                )
                register_service("ai_pipeline", ai_pipeline, dependencies=["market_data"])
                
                pattern_detector = AdvancedPatternDetector({
                    'min_confidence': 0.7,
                    'volume_threshold': 1.5,
                    'enable_ml_enhancement': True
                })
                register_service("pattern_detector", pattern_detector, dependencies=["ai_pipeline"])
                
                signal_generator = SignalGenerator()
                register_service("ai_signal_generator", signal_generator, dependencies=["pattern_detector"])
                
                logger.info("✅ AI services registered")
            except Exception as e:
                logger.warning(f"Failed to register AI services: {e}")
        
        # Initialize all services
        logger.info("🚀 Initializing all services...")
        if not await initialize_all_services():
            logger.error("❌ Service initialization failed")
            raise RuntimeError("Service initialization failed")
        
        logger.info("🎉 AlphaPlus Trading System started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    try:
        logger.info("🛑 Shutting down AlphaPlus Trading System...")
        
        # Close WebSocket connections
        for websocket in websocket_connections:
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        # Shutdown services
        await shutdown_all_services()
        
        # Close database
        await close_database()
        
        logger.info("✅ AlphaPlus Trading System shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_healthy = db_manager.is_initialized
        
        # Check services
        services_status = service_manager.get_all_services_status()
        all_services_healthy = service_manager.is_all_services_healthy()
        
        return {
            "status": "healthy" if db_healthy and all_services_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "status": "healthy" if db_healthy else "unhealthy",
                "initialized": db_healthy
            },
            "services": services_status,
            "ai_available": AI_AVAILABLE,
            "websocket_connections": len(websocket_connections)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive data)"""
    config = settings.to_dict()
    # Remove sensitive information
    config['database']['password'] = '***'
    config['exchange']['binance_secret_key'] = '***'
    return config

# Market data endpoints
@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Get market data for a symbol"""
    try:
        market_data_service = service_manager.get_service("market_data")
        if not market_data_service:
            raise HTTPException(status_code=503, detail="Market data service not available")
        
        data = await market_data_service.get_market_data(symbol, timeframe, limit)
        return {"symbol": symbol, "timeframe": timeframe, "data": data}
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Signal endpoints
@app.get("/signals")
async def get_signals(symbol: Optional[str] = None, limit: int = 50):
    """Get trading signals"""
    try:
        signal_generator = service_manager.get_service("signal_generator")
        if not signal_generator:
            raise HTTPException(status_code=503, detail="Signal generator not available")
        
        signals = await signal_generator.get_signals(symbol, limit)
        return {"signals": signals}
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            data = {
                "timestamp": datetime.now().isoformat(),
                "type": "heartbeat",
                "connections": len(websocket_connections)
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# Service status endpoint
@app.get("/services/status")
async def get_services_status():
    """Get status of all services"""
    return service_manager.get_all_services_status()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AlphaPlus Trading System",
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "config": "/config",
            "market_data": "/market-data/{symbol}",
            "signals": "/signals",
            "websocket": "/ws",
            "services": "/services/status"
        }
    }

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting AlphaPlus Trading System...")
        uvicorn.run(
            "main_unified_fixed:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)
