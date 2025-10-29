"""
Main AI Trading System for AlphaPlus
Activates the real AI/ML pattern recognition and signal generation system
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import asyncio
import asyncpg
import pandas as pd

# Import your AI/ML components
from src.ai.real_time_pipeline import RealTimePipeline, DataPoint, ProcessingResult
from src.ai.signal_generator import SignalGenerator
from src.strategies.advanced_pattern_detector import AdvancedPatternDetector, PatternResult
from src.data.enhanced_real_time_pipeline import EnhancedRealTimePipeline
from src.data.ccxt_integration_service import CCXTIntegrationService
from src.data.social_sentiment_service import SocialSentimentService
from src.services.mtf_orchestrator import MTFOrchestrator
from src.database.connection import TimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AlphaPlus AI Trading System",
    description="Real-time AI/ML pattern recognition and signal generation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI system components
ai_pipeline = None
pattern_detector = None
signal_generator = None
data_pipeline = None
ccxt_service = None
sentiment_service = None
mtf_orchestrator = None
db_connection = None

# Data buffers for real-time processing
market_data_buffer = {}
signal_buffer = []
pattern_buffer = []

@app.on_event("startup")
async def startup_event():
    """Initialize the AI trading system"""
    global ai_pipeline, pattern_detector, signal_generator, data_pipeline
    global ccxt_service, sentiment_service, mtf_orchestrator, db_connection
    
    try:
        logger.info("🚀 Starting AlphaPlus AI Trading System...")
        
        # Initialize database connection
        db_connection = TimescaleDBConnection({
            'host': 'postgres',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        await db_connection.initialize()
        logger.info("✅ Database connection established")
        
        # Initialize AI components
        ai_pipeline = RealTimePipeline(
            max_queue_size=10000,
            num_workers=4,
            enable_parallel_processing=True,
            enable_caching=True
        )
        logger.info("✅ AI Pipeline initialized")
        
        # Initialize pattern detector
        pattern_detector = AdvancedPatternDetector({
            'min_confidence': 0.7,
            'volume_threshold': 1.5,
            'enable_ml_enhancement': True
        })
        await pattern_detector.initialize()
        logger.info("✅ Pattern Detector initialized")
        
        # Initialize signal generator
        signal_generator = SignalGenerator()
        logger.info("✅ Signal Generator initialized")
        
        # Initialize data pipeline
        data_pipeline = EnhancedRealTimePipeline({
            'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT'],
            'exchanges': ['binance', 'okx', 'bybit'],
            'update_frequency': 1.0,
            'analysis_enabled': True,
            'signal_generation_enabled': True
        })
        logger.info("✅ Data Pipeline initialized")
        
        # Initialize CCXT service for market data
        ccxt_service = CCXTIntegrationService({
            'exchanges': ['binance', 'okx', 'bybit'],
            'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT'],
            'update_frequency': 1.0
        })
        logger.info("✅ CCXT Service initialized")
        
        # Initialize sentiment service
        sentiment_service = SocialSentimentService({
            'twitter_enabled': True,
            'news_enabled': True,
            'reddit_enabled': True,
            'update_frequency': 30.0
        })
        logger.info("✅ Sentiment Service initialized")
        
        # Initialize MTF orchestrator
        mtf_orchestrator = MTFOrchestrator({
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'fusion_method': 'weighted_average',
            'confidence_threshold': 0.75
        })
        logger.info("✅ MTF Orchestrator initialized")
        
        # Start AI pipeline
        await ai_pipeline.start()
        logger.info("✅ AI Pipeline started")
        
        # Start data collection
        asyncio.create_task(start_data_collection())
        logger.info("✅ Data collection started")
        
        # Start pattern detection
        asyncio.create_task(start_pattern_detection())
        logger.info("✅ Pattern detection started")
        
        # Start signal generation
        asyncio.create_task(start_signal_generation())
        logger.info("✅ Signal generation started")
        
        logger.info("🎉 AlphaPlus AI Trading System fully activated!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AI system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the AI trading system"""
    global ai_pipeline, data_pipeline
    
    try:
        if ai_pipeline:
            await ai_pipeline.stop()
        if data_pipeline:
            await data_pipeline.stop()
        logger.info("✅ AI Trading System shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

async def start_data_collection():
    """Start real-time data collection"""
    global ccxt_service, sentiment_service
    
    try:
        while True:
            # Collect market data
            market_data = await ccxt_service.get_market_data()
            if market_data:
                for symbol, data in market_data.items():
                    if symbol not in market_data_buffer:
                        market_data_buffer[symbol] = []
                    market_data_buffer[symbol].append(data)
                    
                    # Keep only recent data
                    if len(market_data_buffer[symbol]) > 1000:
                        market_data_buffer[symbol] = market_data_buffer[symbol][-1000:]
            
            # Collect sentiment data
            sentiment_data = await sentiment_service.get_sentiment_data()
            if sentiment_data:
                # Process sentiment data
                pass
            
            await asyncio.sleep(1)  # 1 second update frequency
            
    except Exception as e:
        logger.error(f"❌ Data collection error: {e}")

async def start_pattern_detection():
    """Start real-time pattern detection"""
    global pattern_detector, market_data_buffer, pattern_buffer
    
    try:
        while True:
            for symbol, data_points in market_data_buffer.items():
                if len(data_points) >= 20:  # Need minimum data for pattern detection
                    # Convert to DataFrame for pattern detection
                    df = pd.DataFrame(data_points)
                    
                    # Detect patterns
                    patterns = await pattern_detector.detect_patterns(symbol, df)
                    
                    if patterns:
                        for pattern in patterns:
                            pattern_buffer.append(pattern)
                            logger.info(f"🎯 Pattern detected: {symbol} - {pattern.pattern_type} (confidence: {pattern.confidence})")
            
            # Keep only recent patterns
            if len(pattern_buffer) > 100:
                pattern_buffer = pattern_buffer[-100:]
            
            await asyncio.sleep(5)  # 5 second pattern detection frequency
            
    except Exception as e:
        logger.error(f"❌ Pattern detection error: {e}")

async def start_signal_generation():
    """Start real-time signal generation"""
    global signal_generator, pattern_buffer, signal_buffer, mtf_orchestrator
    
    try:
        while True:
            if pattern_buffer:
                # Generate signals from patterns
                for pattern in pattern_buffer[-10:]:  # Process recent patterns
                    if pattern.confidence >= 0.7:  # High confidence patterns
                        signal = await signal_generator.generate_signal_from_pattern(pattern)
                        if signal:
                            signal_buffer.append(signal)
                            logger.info(f"🚨 Signal generated: {pattern.symbol} - {signal['direction']} (confidence: {signal['confidence']})")
                
                # Multi-timeframe fusion
                if len(signal_buffer) >= 5:
                    fused_signals = await mtf_orchestrator.fuse_signals(signal_buffer[-5:])
                    if fused_signals:
                        logger.info(f"🔗 MTF Fusion: {len(fused_signals)} fused signals")
            
            # Keep only recent signals
            if len(signal_buffer) > 50:
                signal_buffer = signal_buffer[-50:]
            
            await asyncio.sleep(10)  # 10 second signal generation frequency
            
    except Exception as e:
        logger.error(f"❌ Signal generation error: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AlphaPlus AI Trading System",
        "status": "running",
        "version": "1.0.0",
        "components": {
            "ai_pipeline": "active" if ai_pipeline else "inactive",
            "pattern_detector": "active" if pattern_detector else "inactive",
            "signal_generator": "active" if signal_generator else "inactive",
            "data_pipeline": "active" if data_pipeline else "inactive"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AlphaPlus AI Trading System",
        "database": "connected" if db_connection else "disconnected",
        "ai_pipeline": "running" if ai_pipeline and ai_pipeline.running else "stopped",
        "patterns_detected": len(pattern_buffer),
        "signals_generated": len(signal_buffer),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/patterns/latest")
async def get_latest_patterns():
    """Get latest detected patterns"""
    if not pattern_buffer:
        return {"patterns": []}
    
    return {
        "patterns": [
            {
                "symbol": p.symbol,
                "pattern_type": p.pattern_type.value,
                "confidence": p.confidence,
                "strength": p.strength.value,
                "timestamp": p.timestamp.isoformat(),
                "entry_price": p.entry_price,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit
            }
            for p in pattern_buffer[-10:]  # Last 10 patterns
        ]
    }

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Get latest generated signals"""
    if not signal_buffer:
        return {"signals": []}
    
    return {
        "signals": [
            {
                "symbol": s.get('symbol', 'Unknown'),
                "direction": s.get('direction', 'Unknown'),
                "confidence": s.get('confidence', 0.0),
                "pattern_type": s.get('pattern_type', 'Unknown'),
                "timestamp": s.get('timestamp', datetime.utcnow().isoformat()),
                "entry_price": s.get('entry_price', 0.0),
                "stop_loss": s.get('stop_loss', 0.0),
                "take_profit": s.get('take_profit', 0.0)
            }
            for s in signal_buffer[-10:]  # Last 10 signals
        ]
    }

@app.get("/api/market/status")
async def get_market_status():
    """Get current market status"""
    if not market_data_buffer:
        return {"status": "no_data"}
    
    market_status = {}
    for symbol, data_points in market_data_buffer.items():
        if data_points:
            latest = data_points[-1]
            market_status[symbol] = {
                "price": latest.get('close', 0.0),
                "volume": latest.get('volume', 0.0),
                "timestamp": latest.get('timestamp', datetime.utcnow().isoformat()),
                "data_points": len(data_points)
            }
    
    return {"market_status": market_status}

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send real-time updates every 5 seconds
            await asyncio.sleep(5)
            
            # Prepare real-time data
            real_time_data = {
                "type": "real_time_update",
                "timestamp": datetime.utcnow().isoformat(),
                "patterns_count": len(pattern_buffer),
                "signals_count": len(signal_buffer),
                "market_symbols": list(market_data_buffer.keys()),
                "latest_patterns": [
                    {
                        "symbol": p.symbol,
                        "pattern_type": p.pattern_type.value,
                        "confidence": p.confidence
                    }
                    for p in pattern_buffer[-5:]  # Last 5 patterns
                ],
                "latest_signals": [
                    {
                        "symbol": s.get('symbol', 'Unknown'),
                        "direction": s.get('direction', 'Unknown'),
                        "confidence": s.get('confidence', 0.0)
                    }
                    for s in signal_buffer[-5:]  # Last 5 signals
                ]
            }
            
            await websocket.send_text(json.dumps(real_time_data))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting AlphaPlus AI Trading System...")
        uvicorn.run(
            "main_ai_system:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start AI system: {e}")
        sys.exit(1)
