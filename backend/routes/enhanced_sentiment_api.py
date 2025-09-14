"""
Enhanced Sentiment API Routes for AlphaPlus
Provides comprehensive sentiment analysis endpoints with real-time data
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio

# Import enhanced sentiment service
from app.services.enhanced_sentiment_service import EnhancedSentimentService, SentimentSummary

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sentiment", tags=["Enhanced Sentiment Analysis"])

# Global sentiment service instance (will be initialized in main app)
sentiment_service: Optional[EnhancedSentimentService] = None

def get_sentiment_service() -> EnhancedSentimentService:
    """Dependency to get sentiment service instance"""
    if sentiment_service is None:
        raise HTTPException(status_code=503, detail="Sentiment service not initialized")
    return sentiment_service

@router.get("/health")
async def sentiment_health_check():
    """Health check for sentiment service"""
    try:
        if sentiment_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "service": "enhanced_sentiment_service",
                    "status": "not_initialized",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        status = await sentiment_service.get_service_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error in sentiment health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "service": "enhanced_sentiment_service",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/summary/{symbol}")
async def get_sentiment_summary(
    symbol: str,
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get sentiment summary for a specific symbol"""
    try:
        summary = await service.get_sentiment_summary(symbol)
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data available for {symbol}"
            )
        
        return JSONResponse(content={
            "symbol": summary.symbol,
            "timestamp": summary.timestamp.isoformat(),
            "overall_sentiment": summary.overall_sentiment,
            "sentiment_label": summary.sentiment_label,
            "confidence": summary.confidence,
            "source_breakdown": summary.source_breakdown,
            "volume_metrics": summary.volume_metrics,
            "trend": summary.trend,
            "trend_strength": summary.trend_strength,
            "fear_greed_index": summary.fear_greed_index,
            "market_mood": summary.market_mood
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment summary for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sentiment data for {symbol}"
        )

@router.get("/multi-symbol")
async def get_multi_symbol_sentiment(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get sentiment summary for multiple symbols"""
    try:
        # Parse symbols from query parameter
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if not symbol_list:
            raise HTTPException(
                status_code=400,
                detail="No symbols provided"
            )
        
        # Limit to reasonable number of symbols
        if len(symbol_list) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed per request"
            )
        
        results = await service.get_multi_symbol_sentiment(symbol_list)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No sentiment data available for the requested symbols"
            )
        
        # Convert to response format
        response_data = {}
        for symbol, summary in results.items():
            response_data[symbol] = {
                "timestamp": summary.timestamp.isoformat(),
                "overall_sentiment": summary.overall_sentiment,
                "sentiment_label": summary.sentiment_label,
                "confidence": summary.confidence,
                "source_breakdown": summary.source_breakdown,
                "volume_metrics": summary.volume_metrics,
                "trend": summary.trend,
                "trend_strength": summary.trend_strength,
                "fear_greed_index": summary.fear_greed_index,
                "market_mood": summary.market_mood
            }
        
        return JSONResponse(content={
            "symbols": response_data,
            "total_symbols": len(response_data),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting multi-symbol sentiment: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving multi-symbol sentiment data"
        )

@router.get("/trends/{symbol}")
async def get_sentiment_trends(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get sentiment trends over time for a symbol"""
    try:
        trends = await service.get_sentiment_trends(symbol, hours)
        
        if not trends:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment trend data available for {symbol}"
            )
        
        return JSONResponse(content=trends)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment trends for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sentiment trends for {symbol}"
        )

@router.get("/alerts/{symbol}")
async def get_sentiment_alerts(
    symbol: str,
    threshold: float = Query(0.3, ge=0.1, le=1.0, description="Alert threshold for sentiment changes"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get sentiment alerts for significant changes"""
    try:
        alerts = await service.get_sentiment_alerts(symbol, threshold)
        
        return JSONResponse(content={
            "symbol": symbol,
            "threshold": threshold,
            "alerts": alerts,
            "total_alerts": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment alerts for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sentiment alerts for {symbol}"
        )

@router.get("/market-overview")
async def get_market_sentiment_overview(
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get market-wide sentiment overview"""
    try:
        # Get sentiment for all supported symbols
        all_symbols = service.supported_symbols
        results = await service.get_multi_symbol_sentiment(all_symbols)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No market sentiment data available"
            )
        
        # Calculate market-wide metrics
        total_sentiment = 0.0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_confidence = 0.0
        
        symbol_data = {}
        
        for symbol, summary in results.items():
            total_sentiment += summary.overall_sentiment
            total_confidence += summary.confidence
            
            if summary.market_mood == 'bullish':
                bullish_count += 1
            elif summary.market_mood == 'bearish':
                bearish_count += 1
            else:
                neutral_count += 1
            
            symbol_data[symbol] = {
                "sentiment": summary.overall_sentiment,
                "label": summary.sentiment_label,
                "confidence": summary.confidence,
                "trend": summary.trend,
                "market_mood": summary.market_mood
            }
        
        # Calculate averages
        avg_sentiment = total_sentiment / len(results) if results else 0.0
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        # Determine overall market mood
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_mood = "bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_mood = "bearish"
        else:
            overall_mood = "neutral"
        
        return JSONResponse(content={
            "market_overview": {
                "overall_sentiment": avg_sentiment,
                "overall_confidence": avg_confidence,
                "overall_mood": overall_mood,
                "bullish_symbols": bullish_count,
                "bearish_symbols": bearish_count,
                "neutral_symbols": neutral_count,
                "total_symbols": len(results)
            },
            "symbols": symbol_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market sentiment overview: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving market sentiment overview"
        )

@router.post("/collect/{symbol}")
async def trigger_sentiment_collection(
    symbol: str,
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Trigger immediate sentiment collection for a symbol"""
    try:
        # Collect sentiment data
        sentiment_data = await service.sentiment_analyzer.collect_all_sentiment(symbol)
        
        return JSONResponse(content={
            "symbol": symbol,
            "collected_records": len(sentiment_data),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error triggering sentiment collection for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error collecting sentiment data for {symbol}"
        )

@router.get("/sources/{symbol}")
async def get_sentiment_sources(
    symbol: str,
    hours: int = Query(1, ge=1, le=24, description="Number of hours to analyze"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get sentiment breakdown by source for a symbol"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Query sentiment data by source
        async with service.db_pool.acquire() as conn:
            query = """
                SELECT source, 
                       COUNT(*) as record_count,
                       AVG(sentiment_score) as avg_sentiment,
                       AVG(confidence) as avg_confidence,
                       SUM(volume) as total_volume
                FROM enhanced_sentiment_data
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                GROUP BY source
                ORDER BY record_count DESC
            """
            rows = await conn.fetch(query, symbol, start_time, end_time)
        
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment source data available for {symbol}"
            )
        
        sources_data = {}
        total_records = 0
        
        for row in rows:
            source = row['source']
            sources_data[source] = {
                "record_count": row['record_count'],
                "avg_sentiment": float(row['avg_sentiment']) if row['avg_sentiment'] else 0.0,
                "avg_confidence": float(row['avg_confidence']) if row['avg_confidence'] else 0.0,
                "total_volume": row['total_volume'] or 0
            }
            total_records += row['record_count']
        
        return JSONResponse(content={
            "symbol": symbol,
            "time_range": f"{hours}h",
            "total_records": total_records,
            "sources": sources_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment sources for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sentiment source data for {symbol}"
        )

@router.get("/quality/{symbol}")
async def get_sentiment_quality_metrics(
    symbol: str,
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get sentiment quality metrics for a symbol"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Query sentiment quality metrics
        async with service.db_pool.acquire() as conn:
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    AVG(confidence) as avg_confidence,
                    STDDEV(confidence) as confidence_stddev,
                    AVG(context_score) as avg_context_score,
                    COUNT(CASE WHEN sarcasm_detected THEN 1 END) as sarcasm_count,
                    COUNT(CASE WHEN topic_classification = 'price_moving' THEN 1 END) as price_moving_count,
                    COUNT(CASE WHEN topic_classification = 'noise' THEN 1 END) as noise_count,
                    COUNT(DISTINCT source) as source_diversity
                FROM enhanced_sentiment_data
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            row = await conn.fetchrow(query, symbol, start_time, end_time)
        
        if not row or row['total_records'] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment quality data available for {symbol}"
            )
        
        # Calculate quality metrics
        total_records = row['total_records']
        avg_confidence = float(row['avg_confidence']) if row['avg_confidence'] else 0.0
        confidence_stddev = float(row['confidence_stddev']) if row['confidence_stddev'] else 0.0
        avg_context_score = float(row['avg_context_score']) if row['avg_context_score'] else 0.0
        
        # Calculate percentages
        sarcasm_percentage = (row['sarcasm_count'] / total_records) * 100 if total_records > 0 else 0
        price_moving_percentage = (row['price_moving_count'] / total_records) * 100 if total_records > 0 else 0
        noise_percentage = (row['noise_count'] / total_records) * 100 if total_records > 0 else 0
        
        # Quality score calculation
        quality_score = (
            avg_confidence * 0.4 +
            avg_context_score * 0.3 +
            (1 - noise_percentage / 100) * 0.2 +
            (row['source_diversity'] / 6) * 0.1  # Normalize to max 6 sources
        )
        
        return JSONResponse(content={
            "symbol": symbol,
            "time_range": f"{hours}h",
            "quality_metrics": {
                "total_records": total_records,
                "avg_confidence": avg_confidence,
                "confidence_stddev": confidence_stddev,
                "avg_context_score": avg_context_score,
                "sarcasm_percentage": sarcasm_percentage,
                "price_moving_percentage": price_moving_percentage,
                "noise_percentage": noise_percentage,
                "source_diversity": row['source_diversity'],
                "overall_quality_score": quality_score
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment quality metrics for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sentiment quality metrics for {symbol}"
        )

@router.get("/supported-symbols")
async def get_supported_symbols(
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get list of supported symbols for sentiment analysis"""
    try:
        return JSONResponse(content={
            "supported_symbols": service.supported_symbols,
            "total_symbols": len(service.supported_symbols),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting supported symbols: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving supported symbols"
        )

# WebSocket endpoint for real-time sentiment updates
@router.websocket("/ws/{symbol}")
async def sentiment_websocket(
    websocket,
    symbol: str,
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """WebSocket endpoint for real-time sentiment updates"""
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection established for {symbol}")
        
        while True:
            try:
                # Get latest sentiment data
                summary = await service.get_sentiment_summary(symbol)
                
                if summary:
                    # Send sentiment data
                    await websocket.send_json({
                        "symbol": summary.symbol,
                        "timestamp": summary.timestamp.isoformat(),
                        "overall_sentiment": summary.overall_sentiment,
                        "sentiment_label": summary.sentiment_label,
                        "confidence": summary.confidence,
                        "trend": summary.trend,
                        "market_mood": summary.market_mood
                    })
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in sentiment WebSocket for {symbol}: {e}")
                await websocket.send_json({
                    "error": f"Error retrieving sentiment data: {str(e)}"
                })
                break
                
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")
    finally:
        logger.info(f"WebSocket connection closed for {symbol}")

# ===== PHASE 4A: PREDICTIVE ANALYTICS ENDPOINTS =====

@router.get("/predictions/{symbol}")
async def get_price_prediction(
    symbol: str,
    time_horizon: str = Query('4h', description="Prediction time horizon: 1h, 4h, 1d, 1w"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get price movement prediction for a symbol"""
    try:
        prediction = await service.get_price_prediction(symbol, time_horizon)
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction available for {symbol}"
            )
        
        return JSONResponse(content=prediction)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price prediction for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving price prediction for {symbol}"
        )

@router.get("/predictions/{symbol}/multi-horizon")
async def get_multi_horizon_predictions(
    symbol: str,
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get predictions for multiple time horizons"""
    try:
        predictions = await service.get_multi_horizon_predictions(symbol)
        
        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available for {symbol}"
            )
        
        return JSONResponse(content=predictions)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting multi-horizon predictions for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving multi-horizon predictions for {symbol}"
        )

@router.get("/predictions/{symbol}/confidence-analysis")
async def get_prediction_confidence_analysis(
    symbol: str,
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get detailed confidence analysis for predictions"""
    try:
        analysis = await service.get_prediction_confidence_analysis(symbol)
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        logger.error(f"Error getting prediction confidence analysis for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving prediction confidence analysis for {symbol}"
        )

# ===== PHASE 4B: CROSS-ASSET CORRELATION ENDPOINTS =====

@router.get("/cross-asset/{primary_symbol}")
async def get_cross_asset_analysis(
    primary_symbol: str,
    secondary_symbols: str = Query(None, description="Comma-separated list of secondary symbols"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get cross-asset sentiment correlation analysis"""
    try:
        # Parse secondary symbols
        symbols_list = None
        if secondary_symbols:
            symbols_list = [s.strip() for s in secondary_symbols.split(',')]
        
        analysis = await service.get_cross_asset_analysis(primary_symbol, symbols_list)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"No cross-asset analysis available for {primary_symbol}"
            )
        
        return JSONResponse(content=analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cross-asset analysis for {primary_symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving cross-asset analysis for {primary_symbol}"
        )

@router.get("/market-overview")
async def get_market_sentiment_overview(
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get overall market sentiment overview"""
    try:
        overview = await service.get_market_sentiment_overview()
        
        if not overview:
            raise HTTPException(
                status_code=404,
                detail="No market sentiment overview available"
            )
        
        return JSONResponse(content=overview)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market sentiment overview: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving market sentiment overview"
        )

# ===== PHASE 4C: MODEL PERFORMANCE ENDPOINTS =====

@router.get("/model-performance")
async def get_model_performance_summary(
    symbol: str = Query(None, description="Optional symbol filter"),
    days: int = Query(30, description="Number of days to analyze"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get model performance summary"""
    try:
        performance = await service.get_model_performance_summary(symbol, days)
        
        return JSONResponse(content=performance)
        
    except Exception as e:
        logger.error(f"Error getting model performance summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model performance summary"
        )

@router.post("/model-performance/update")
async def update_model_performance(
    actual_outcomes: List[Dict[str, Any]],
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Update model performance with actual outcomes"""
    try:
        await service.update_model_performance(actual_outcomes)
        
        return JSONResponse(content={
            "message": f"Model performance updated with {len(actual_outcomes)} outcomes",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating model performance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error updating model performance"
        )

# ===== PHASE 4D: PREDICTION ALERTS ENDPOINTS =====

@router.get("/alerts/predictions")
async def get_prediction_alerts(
    symbol: str = Query(None, description="Optional symbol filter"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get prediction alerts for significant sentiment changes"""
    try:
        alerts = await service.get_prediction_alerts(symbol)
        
        return JSONResponse(content={
            "alerts": alerts,
            "total_alerts": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving prediction alerts"
        )

# ===== PHASE 4E: ENHANCED WEBSOCKET FOR PREDICTIONS =====

@router.websocket("/ws/predictions/{symbol}")
async def prediction_websocket(
    websocket,
    symbol: str,
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """WebSocket endpoint for real-time prediction updates"""
    try:
        await websocket.accept()
        logger.info(f"Prediction WebSocket connection established for {symbol}")
        
        while True:
            try:
                # Get latest prediction data
                prediction = await service.get_price_prediction(symbol, '4h')
                
                if prediction:
                    # Send prediction data
                    await websocket.send_json({
                        "symbol": prediction.get('symbol'),
                        "timestamp": prediction.get('timestamp'),
                        "prediction_probability": prediction.get('prediction_probability'),
                        "direction": prediction.get('direction'),
                        "strength": prediction.get('strength'),
                        "confidence": prediction.get('confidence'),
                        "sentiment_score": prediction.get('sentiment_score'),
                        "technical_indicators": prediction.get('technical_indicators'),
                        "factors": prediction.get('factors')
                    })
                
                # Wait before next update
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in prediction WebSocket for {symbol}: {e}")
                await websocket.send_json({
                    "error": f"Error retrieving prediction data: {str(e)}"
                })
                break
                
    except Exception as e:
        logger.error(f"Prediction WebSocket error for {symbol}: {e}")
    finally:
        logger.info(f"Prediction WebSocket connection closed for {symbol}")

# ===== PHASE 4F: BULK OPERATIONS ENDPOINTS =====

@router.get("/bulk/predictions")
async def get_bulk_predictions(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    time_horizon: str = Query('4h', description="Prediction time horizon"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get predictions for multiple symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        predictions = {}
        
        for symbol in symbol_list:
            prediction = await service.get_price_prediction(symbol, time_horizon)
            if prediction:
                predictions[symbol] = prediction
        
        return JSONResponse(content={
            "predictions": predictions,
            "total_symbols": len(symbol_list),
            "successful_predictions": len(predictions),
            "time_horizon": time_horizon,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting bulk predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving bulk predictions"
        )

@router.get("/bulk/cross-asset")
async def get_bulk_cross_asset_analysis(
    primary_symbols: str = Query(..., description="Comma-separated list of primary symbols"),
    service: EnhancedSentimentService = Depends(get_sentiment_service)
):
    """Get cross-asset analysis for multiple primary symbols"""
    try:
        symbol_list = [s.strip() for s in primary_symbols.split(',')]
        analyses = {}
        
        for symbol in symbol_list:
            analysis = await service.get_cross_asset_analysis(symbol)
            if analysis:
                analyses[symbol] = analysis
        
        return JSONResponse(content={
            "cross_asset_analyses": analyses,
            "total_symbols": len(symbol_list),
            "successful_analyses": len(analyses),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting bulk cross-asset analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving bulk cross-asset analysis"
        )
