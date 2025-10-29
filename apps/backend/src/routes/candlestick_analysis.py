#!/usr/bin/env python3
"""
Candlestick Analysis API Routes for AlphaPulse
Provides endpoints for real-time pattern detection and signal generation
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Path, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import logging
import asyncio
import re

from ..src.database.connection import get_db
from src.data.real_time_processor import RealTimeCandlestickProcessor
from ..src.strategies.ml_pattern_detector import MLPatternDetector
from ..src.strategies.real_time_signal_generator import RealTimeSignalGenerator
from src.app.core.unified_config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/candlestick", tags=["candlestick-analysis"])

# Validation constants
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]+$')
TIMEFRAME_PATTERN = re.compile(r'^(1m|5m|15m|30m|1h|4h|1d|1w)$')
MAX_LIMIT = 1000
MIN_LIMIT = 1

def get_processor() -> RealTimeCandlestickProcessor:
    """Get or create candlestick processor instance using dependency injection"""
    try:
        processor = RealTimeCandlestickProcessor({
            'min_confidence': 0.7,
            'min_strength': 0.6,
            'confirmation_required': True,
            'volume_confirmation': True,
            'trend_confirmation': True,
            'min_data_points': 50,
            'max_data_points': 1000,
            'signal_cooldown': 300
        })
        return processor
    except Exception as e:
        logger.error(f"Failed to create processor: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize processor")

def validate_symbol(symbol: str) -> str:
    """Validate trading symbol format"""
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format. Use uppercase alphanumeric characters only.")
    if len(symbol) > 20:
        raise HTTPException(status_code=400, detail="Symbol too long. Maximum 20 characters allowed.")
    return symbol.upper()

def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe format"""
    if not TIMEFRAME_PATTERN.match(timeframe):
        raise HTTPException(status_code=400, detail="Invalid timeframe. Use: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w")
    return timeframe

def validate_limit(limit: int) -> int:
    """Validate limit parameter"""
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise HTTPException(status_code=400, detail=f"Limit must be between {MIN_LIMIT} and {MAX_LIMIT}")
    return limit

@router.get("/status")
async def get_analysis_status():
    """Get current status of candlestick analysis system"""
    try:
        processor = get_processor()
        
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "processing_stats": processor.get_processing_stats(),
            "active_symbols": list(processor.candlestick_data.keys()),
            "total_signals": sum(len(signals) for signals in processor.signal_history.values())
        }
    except ConnectionError as e:
        logger.error(f"Database connection error in analysis status: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")
    except ImportError as e:
        logger.error(f"Module import error in analysis status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="System configuration error")
    except Exception as e:
        logger.error(f"Error getting analysis status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get analysis status")

@router.get("/patterns/{symbol}")
async def get_patterns(
    symbol: str = Path(..., description="Trading symbol (e.g., BTC, ETH)"),
    timeframe: str = Query("15m", description="Timeframe for analysis"),
    limit: int = Query(100, description="Maximum number of patterns to return")
):
    """Get detected patterns for a symbol"""
    try:
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        limit = validate_limit(limit)
        
        processor = get_processor()
        
        # Get symbol data
        symbol_data = processor.get_symbol_data(symbol, timeframe)
        
        if not symbol_data['candlesticks']:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol} {timeframe}")
        
        # Convert to DataFrame for pattern detection
        df = processor._to_dataframe(symbol, timeframe)
        
        # Detect patterns with batch error handling
        try:
            ml_detector = MLPatternDetector()
            patterns = ml_detector.detect_patterns_ml(df)
        except ImportError as e:
            logger.error(f"Pattern detection module not available: {e}", exc_info=True)
            patterns = []
        except Exception as e:
            logger.error(f"Pattern detection failed for {symbol}: {e}", exc_info=True)
            patterns = []
        
        # Format patterns for response with error handling
        formatted_patterns = []
        for pattern in patterns[-limit:]:  # Get latest patterns
            try:
                formatted_patterns.append({
                    "pattern": pattern.pattern,
                    "type": pattern.type,
                    "strength": pattern.strength,
                    "confidence": pattern.ml_confidence,
                    "market_regime": pattern.market_regime,
                    "timestamp": pattern.timestamp,
                    "features": pattern.features
                })
            except AttributeError as e:
                logger.warning(f"Pattern formatting error: {e}, skipping pattern")
                continue
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_patterns": len(patterns),
            "patterns": formatted_patterns,
            "last_update": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Database connection error for patterns {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")
    except ValueError as e:
        logger.error(f"Invalid data format for patterns {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid data format")
    except Exception as e:
        logger.error(f"Error getting patterns for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/signals/{symbol}")
async def get_signals(
    symbol: str = Path(..., description="Trading symbol (e.g., BTC, ETH)"),
    timeframe: str = Query("15m", description="Timeframe for analysis"),
    limit: int = Query(50, description="Maximum number of signals to return")
):
    """Get generated trading signals for a symbol"""
    try:
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        limit = validate_limit(limit)
        
        processor = get_processor()
        
        # Get signals for symbol
        signals = processor.signal_history.get(symbol, [])
        
        if not signals:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_signals": 0,
                "signals": [],
                "message": "No signals generated yet"
            }
        
        # Format signals for response
        formatted_signals = []
        for signal in sorted(signals, key=lambda x: x.timestamp, reverse=True)[:limit]:
            formatted_signals.append({
                "signal_type": signal.signal_type,
                "pattern": signal.pattern,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "price": signal.price,
                "timestamp": signal.timestamp.isoformat(),
                "timeframe": signal.timeframe,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "market_regime": signal.market_regime,
                "volume_confirmation": signal.volume_confirmation,
                "trend_alignment": signal.trend_alignment,
                "support_resistance_levels": signal.support_resistance_levels,
                "additional_indicators": signal.additional_indicators
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_signals": len(signals),
            "signals": formatted_signals,
            "last_update": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/signals/summary")
async def get_signals_summary():
    """Get summary of all generated signals"""
    try:
        processor = get_processor()
        summary = processor.get_signal_summary()
        
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting signals summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analysis/{symbol}")
async def get_comprehensive_analysis(
    symbol: str = Path(..., description="Trading symbol (e.g., BTC, ETH)"),
    timeframe: str = Query("15m", description="Timeframe for analysis")
):
    """Get comprehensive analysis including patterns, signals, and indicators"""
    try:
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        
        processor = get_processor()
        
        # Get symbol data
        symbol_data = processor.get_symbol_data(symbol, timeframe)
        
        if not symbol_data['candlesticks']:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol} {timeframe}")
        
        # Get latest processed data
        processed_data = symbol_data['processed']
        if not processed_data:
            raise HTTPException(status_code=404, detail=f"No processed data found for {symbol} {timeframe}")
        
        latest_data = processed_data[-1]
        
        # Get patterns
        df = processor._to_dataframe(symbol, timeframe)
        ml_detector = MLPatternDetector()
        patterns = ml_detector.detect_patterns_ml(df)
        
        # Get signals
        signals = processor.signal_history.get(symbol, [])
        
        # Format analysis
        analysis = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_price": {
                "open": latest_data.open,
                "high": latest_data.high,
                "low": latest_data.low,
                "close": latest_data.close,
                "volume": latest_data.volume
            },
            "technical_indicators": latest_data.indicators,
            "detected_patterns": [p.pattern for p in patterns[-5:]],  # Last 5 patterns
            "pattern_summary": {
                "total_patterns": len(patterns),
                "bullish_patterns": len([p for p in patterns if p.type == 'bullish']),
                "bearish_patterns": len([p for p in patterns if p.type == 'bearish']),
                "neutral_patterns": len([p for p in patterns if p.type == 'neutral'])
            },
            "signals": {
                "total_signals": len(signals),
                "buy_signals": len([s for s in signals if s.signal_type == 'buy']),
                "sell_signals": len([s for s in signals if s.signal_type == 'sell']),
                "latest_signals": [
                    {
                        "type": s.signal_type,
                        "pattern": s.pattern,
                        "confidence": s.confidence,
                        "timestamp": s.timestamp.isoformat()
                    }
                    for s in sorted(signals, key=lambda x: x.timestamp, reverse=True)[:5]
                ]
            },
            "market_regime": patterns[-1].market_regime if patterns else None,
            "data_quality": {
                "total_candlesticks": len(symbol_data['candlesticks']),
                "data_points_available": len(symbol_data['candlesticks']) >= processor.min_data_points
            }
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comprehensive analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/train-model")
async def train_ml_model(
    symbol: str = Path(..., description="Trading symbol (e.g., BTC, ETH)"),
    timeframe: str = Query("15m", description="Timeframe for training data"),
    training_data_size: int = Query(1000, description="Number of data points for training")
):
    """Train the ML pattern detection model with historical data"""
    try:
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        training_data_size = validate_limit(training_data_size)
        
        processor = get_processor()
        
        # Get historical data
        symbol_data = processor.get_symbol_data(symbol, timeframe)
        
        if len(symbol_data['candlesticks']) < training_data_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data. Need {training_data_size}, have {len(symbol_data['candlesticks'])}"
            )
        
        # Convert to DataFrame
        df = processor._to_dataframe(symbol, timeframe)
        
        # Generate training labels (simplified - in production you'd use actual pattern labels)
        # This is a placeholder - you'd need to implement proper labeling
        pattern_labels = ['no_pattern'] * len(df)
        
        # Train model
        ml_detector = MLPatternDetector()
        success = ml_detector.train_model(df, pattern_labels)
        
        if success:
            return {
                "message": "ML model trained successfully",
                "symbol": symbol,
                "timeframe": timeframe,
                "training_samples": len(df),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to train ML model")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/model-info")
async def get_model_info():
    """Get information about the trained ML model"""
    try:
        ml_detector = MLPatternDetector()
        model_info = ml_detector.get_model_info()
        
        return {
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/start-processing")
async def start_real_time_processing(
    symbols: List[str] = Query(..., description="List of symbols to process (e.g., BTC, ETH)"),
    timeframes: List[str] = Query(default=["15m"], description="List of timeframes to process (e.g., 1m, 5m, 15m)")
):
    """Start real-time candlestick processing for specified symbols"""
    try:
        # Validate symbols and timeframes
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for tf in timeframes:
            if tf not in valid_timeframes:
                raise HTTPException(status_code=400, detail=f"Invalid timeframe: {tf}")
        
        processor = get_processor()
        
        # Start processing (this would integrate with WebSocket client)
        # For now, just update the processor configuration
        processor.config.update({
            'active_symbols': symbols,
            'active_timeframes': timeframes
        })
        
        return {
            "message": "Real-time processing started",
            "symbols": symbols,
            "timeframes": timeframes,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting real-time processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/stop-processing")
async def stop_real_time_processing():
    """Stop real-time candlestick processing"""
    try:
        processor = get_processor()
        await processor.stop_processing()
        
        return {
            "message": "Real-time processing stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping real-time processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/clear-data")
async def clear_analysis_data(
    symbol: Optional[str] = Query(None, description="Symbol to clear data for (e.g., BTC)"),
    timeframe: Optional[str] = Query(None, description="Timeframe to clear data for (e.g., 15m)")
):
    """Clear stored analysis data"""
    try:
        processor = get_processor()
        if symbol and timeframe:
            processor.clear_data(symbol, timeframe)
            message = f"Data cleared for {symbol} {timeframe}"
        else:
            processor.clear_all_data()
            message = "All data cleared"
        
        return {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.websocket("/ws/analysis/{symbol}")
async def websocket_analysis(
    websocket: WebSocket,
    symbol: str = Path(..., description="Trading symbol (e.g., BTC, ETH)"),
    timeframe: str = Query("15m", description="Timeframe for analysis")
):
    """WebSocket endpoint for real-time analysis updates"""
    await websocket.accept()
    
    try:
        # Validate inputs
        symbol = validate_symbol(symbol)
        timeframe = validate_timeframe(timeframe)
        
        processor = get_processor()
        
        # Add data callback to send updates via WebSocket
        async def send_analysis_update(processed_data):
            if processed_data.symbol == symbol and processed_data.timeframe == timeframe:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "analysis_update",
                        "symbol": processed_data.symbol,
                        "timeframe": processed_data.timeframe,
                        "timestamp": processed_data.timestamp.isoformat(),
                        "price": {
                            "open": processed_data.open,
                            "high": processed_data.high,
                            "low": processed_data.low,
                            "close": processed_data.close,
                            "volume": processed_data.volume
                        },
                        "indicators": processed_data.indicators,
                        "patterns": processed_data.patterns,
                        "signals": processed_data.signals
                    }))
                except Exception as e:
                    logger.error(f"Error sending WebSocket update: {e}")
        
        # Add signal callback
        async def send_signal_update(signal):
            if signal.symbol == symbol:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "signal_update",
                        "signal": {
                            "type": signal.signal_type,
                            "pattern": signal.pattern,
                            "strength": signal.strength,
                            "confidence": signal.confidence,
                            "price": signal.price,
                            "timestamp": signal.timestamp.isoformat(),
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "risk_reward_ratio": signal.risk_reward_ratio
                        }
                    }))
                except Exception as e:
                    logger.error(f"Error sending signal update: {e}")
        
        # Register callbacks
        processor.add_data_callback(send_analysis_update)
        processor.add_signal_callback(send_signal_update)
        
        # Send initial data
        symbol_data = processor.get_symbol_data(symbol, timeframe)
        if symbol_data['processed']:
            latest = symbol_data['processed'][-1]
            await websocket.send_text(json.dumps({
                "type": "initial_data",
                "symbol": symbol,
                "timeframe": timeframe,
                "data": {
                    "timestamp": latest.timestamp.isoformat(),
                    "price": {
                        "open": latest.open,
                        "high": latest.high,
                        "low": latest.low,
                        "close": latest.close,
                        "volume": latest.volume
                    },
                    "indicators": latest.indicators,
                    "patterns": latest.patterns,
                    "signals": latest.signals
                }
            }))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client message (ping/pong for keep-alive)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for {symbol}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WebSocket analysis error: {e}")
        try:
            await websocket.close()
        except:
            pass

@router.get("/health")
async def health_check():
    """Health check for candlestick analysis system"""
    try:
        processor = get_processor()
        stats = processor.get_processing_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "processing_stats": stats,
            "active_connections": len(processor.signal_callbacks) + len(processor.data_callbacks)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

