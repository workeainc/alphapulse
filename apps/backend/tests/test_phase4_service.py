#!/usr/bin/env python3
"""
Phase 4 Test Service
Simple FastAPI service to test Phase 4 enhancements
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import asyncpg
import redis.asyncio as redis
import json

# Add current directory to path
sys.path.append('.')

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from src.app.services.enhanced_sentiment_service import EnhancedSentimentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Phase 4 Test Service",
    description="Test service for Phase 4 enhancements",
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

# Global services
sentiment_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services"""
    global sentiment_service
    
    try:
        logger.info("🚀 Starting Phase 4 Test Service...")
        
        # Initialize database connection
        db_pool = await asyncpg.create_pool(
            host='localhost', port=5432, database='alphapulse',
            user='alpha_emon', password='Emon_@17711'
        )
        
        # Initialize Redis connection
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connections
        await redis_client.ping()
        async with db_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        
        # Initialize services
        sentiment_analyzer = EnhancedSentimentAnalyzer(db_pool, redis_client)
        sentiment_service = EnhancedSentimentService(db_pool, redis_client)
        
        logger.info("✅ Phase 4 Test Service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start Phase 4 Test Service: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Phase 4 Test Service is running!", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/sentiment/predictions/{symbol}")
async def get_price_prediction(
    symbol: str,
    time_horizon: str = Query('4h', description="Prediction time horizon: 1h, 4h, 1d, 1w")
):
    """Get price movement prediction for a symbol"""
    try:
        if not sentiment_service:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
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
            raise HTTPException(status_code=500, detail="Service not initialized")
        
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
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        performance = await sentiment_service.get_model_performance_summary()
        return performance
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model performance"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
