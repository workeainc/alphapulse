"""
Single-Pair Trading API Endpoints
Dedicated endpoints for sophisticated single-pair trading interface
Phase 4: Backend Integration
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import json
from dataclasses import asdict

# Import with error handling to prevent startup failures
try:
    from ..signals.intelligent_signal_generator import IntelligentSignalGenerator, IntelligentSignal
except ImportError:
    IntelligentSignalGenerator = None
    IntelligentSignal = None

try:
    from ..analysis.intelligent_analysis_engine import IntelligentAnalysisEngine
except ImportError:
    IntelligentAnalysisEngine = None

try:
    from src.database.connection import TimescaleDBConnection
except ImportError:
    TimescaleDBConnection = None

try:
    from src.services.free_api_manager import FreeAPIManager
except ImportError:
    FreeAPIManager = None

try:
    from src.services.news_sentiment_service import NewsSentimentService
except ImportError:
    NewsSentimentService = None

try:
    from src.services.real_data_integration_service import real_data_service
except ImportError:
    real_data_service = None

try:
    from src.services.external_api_integration_service import external_api_service
except ImportError:
    external_api_service = None

# Create router
router = APIRouter(prefix="/api/single-pair", tags=["single-pair"])

# Global instances
signal_generator = IntelligentSignalGenerator() if IntelligentSignalGenerator else None
analysis_engine = IntelligentAnalysisEngine() if IntelligentAnalysisEngine else None
db_connection = TimescaleDBConnection() if TimescaleDBConnection else None
free_api_manager = FreeAPIManager() if FreeAPIManager else None
news_service = NewsSentimentService() if NewsSentimentService else None

# WebSocket connection manager
class SinglePairConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.pair_subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, pair: str = "BTCUSDT"):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if pair not in self.pair_subscriptions:
            self.pair_subscriptions[pair] = []
        self.pair_subscriptions[pair].append(websocket)
        
        logging.info(f"WebSocket connected for pair: {pair}")

    def disconnect(self, websocket: WebSocket, pair: str = "BTCUSDT"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if pair in self.pair_subscriptions and websocket in self.pair_subscriptions[pair]:
            self.pair_subscriptions[pair].remove(websocket)
        
        logging.info(f"WebSocket disconnected for pair: {pair}")

    async def send_to_pair(self, pair: str, message: dict):
        if pair in self.pair_subscriptions:
            for connection in self.pair_subscriptions[pair]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    # Remove broken connections
                    self.pair_subscriptions[pair].remove(connection)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

# Global connection manager
connection_manager = SinglePairConnectionManager()

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify router is working"""
    return JSONResponse(content={
        "status": "success",
        "message": "Single-pair router is working",
        "timestamp": datetime.utcnow().isoformat()
    })

@router.get("/status")
async def get_system_status():
    """Get single-pair system status"""
    try:
        status = {
            "system_status": "running",
            "active_connections": len(connection_manager.active_connections),
            "subscribed_pairs": list(connection_manager.pair_subscriptions.keys()),
            "components": {
                "signal_generator": "available" if signal_generator else "unavailable",
                "analysis_engine": "available" if analysis_engine else "unavailable",
                "db_connection": "available" if db_connection else "unavailable",
                "free_api_manager": "available" if free_api_manager else "unavailable",
                "news_service": "available" if news_service else "unavailable",
                "real_data_service": "available" if real_data_service else "unavailable",
                "external_api_service": "available" if external_api_service else "unavailable"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        return JSONResponse(content=status)
    except Exception as e:
        logging.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/analysis/{pair}")
async def get_single_pair_analysis(pair: str, timeframe: str = "1h"):
    """Get comprehensive analysis for a single pair using real data"""
    try:
        # Validate pair format
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        # Check if real_data_service is available
        if not real_data_service:
            return JSONResponse(content={
                "error": "Real data service not available",
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Get real analysis data
        analysis_data = await real_data_service.get_real_analysis_data(pair, timeframe)
        
        return JSONResponse(content=analysis_data)
        
    except Exception as e:
        logging.error(f"Error getting analysis for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis for {pair}")

@router.get("/confidence/{pair}")
async def get_confidence_building(pair: str, timeframe: str = "1h"):
    """Get real-time confidence building for a single pair"""
    try:
        # Validate pair format
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        # Check if signal_generator is available
        if not signal_generator:
            return JSONResponse(content={
                "error": "Signal generator not available",
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Get current confidence
        confidence_data = await signal_generator.get_confidence_building(
            symbol=pair,
            timeframe=timeframe
        )
        
        return JSONResponse(content={
            "pair": pair,
            "timeframe": timeframe,
            "confidence": confidence_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error getting confidence for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get confidence for {pair}")

@router.get("/signal/{pair}")
async def get_single_pair_signal(pair: str, timeframe: str = "1h"):
    """Get the current signal for a single pair"""
    try:
        # Validate pair format
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        # Check if signal_generator is available
        if not signal_generator:
            return JSONResponse(content={
                "error": "Signal generator not available",
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Generate signal
        signal = await signal_generator.generate_single_pair_signal(
            symbol=pair,
            timeframe=timeframe
        )
        
        if signal:
            return JSONResponse(content={
                "pair": pair,
                "timeframe": timeframe,
                "signal": asdict(signal) if hasattr(signal, '__dict__') else signal,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return JSONResponse(content={
                "pair": pair,
                "timeframe": timeframe,
                "signal": None,
                "message": "No signal available",
                "timestamp": datetime.utcnow().isoformat()
            })
        
    except Exception as e:
        logging.error(f"Error getting signal for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get signal for {pair}")

@router.post("/signal/{pair}/execute")
async def execute_single_pair_signal(
    pair: str,
    execution_data: Dict[str, Any]
):
    """Execute a signal for a single pair"""
    try:
        # Validate pair format
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        # Execute the signal
        execution_result = await signal_generator.execute_single_pair_signal(
            symbol=pair,
            execution_data=execution_data
        )
        
        return JSONResponse(content={
            "pair": pair,
            "execution": execution_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error executing signal for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute signal for {pair}")

@router.get("/pairs")
async def get_available_pairs():
    """Get list of available trading pairs"""
    try:
        pairs = await free_api_manager.get_available_pairs()
        return JSONResponse(content={
            "pairs": pairs,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logging.error(f"Error getting available pairs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available pairs")

@router.get("/timeframes")
async def get_available_timeframes():
    """Get list of available timeframes"""
    timeframes = ["15m", "1h", "4h", "1d", "1w"]
    return JSONResponse(content={
        "timeframes": timeframes,
        "timestamp": datetime.utcnow().isoformat()
    })

@router.websocket("/ws/{pair}")
async def websocket_endpoint(websocket: WebSocket, pair: str):
    """WebSocket endpoint for real-time single-pair data"""
    await connection_manager.connect(websocket, pair)
    
    try:
        while True:
            # Send real-time updates every 5 seconds
            await asyncio.sleep(5)
            
            try:
                # Get real data from services instead of calling endpoints
                analysis_data = await real_data_service.get_real_analysis_data(pair, "1h")
                confidence_data = await signal_generator.get_confidence_building(pair, "1h")
                signal_data = await signal_generator.generate_single_pair_signal(pair, "1h")
                
                # Send real data update
                update_data = {
                    "type": "real_time_update",
                    "pair": pair,
                    "analysis": analysis_data,
                    "confidence": confidence_data,
                    "signal": signal_data.__dict__ if signal_data else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await connection_manager.send_to_pair(pair, update_data)
                
            except Exception as e:
                logging.error(f"Error getting real data for {pair}: {e}")
                # Send error update
                error_data = {
                    "type": "error",
                    "pair": pair,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await connection_manager.send_to_pair(pair, error_data)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, pair)
    except Exception as e:
        logging.error(f"WebSocket error for {pair}: {e}")
        connection_manager.disconnect(websocket, pair)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "service": "single-pair-api",
        "timestamp": datetime.utcnow().isoformat()
    })

# External API Management Endpoints
@router.post("/data/start-collection")
async def start_live_data_collection():
    """Start live data collection from external APIs"""
    try:
        result = await external_api_service.start_live_data_collection()
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start data collection: {str(e)}")

@router.post("/data/stop-collection")
async def stop_live_data_collection():
    """Stop live data collection"""
    try:
        result = await external_api_service.stop_live_data_collection()
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error stopping data collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop data collection: {str(e)}")

@router.get("/data/status")
async def get_data_collection_status():
    """Get data collection status and statistics"""
    try:
        result = await external_api_service.get_pipeline_status()
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error getting data collection status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data collection status: {str(e)}")

@router.post("/data/refresh/{pair}")
async def force_data_refresh(pair: str):
    """Force immediate data refresh for a specific pair"""
    try:
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        result = await external_api_service.force_data_refresh(pair)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error forcing data refresh for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh data for {pair}: {str(e)}")

@router.get("/data/freshness/{pair}")
async def get_data_freshness(pair: str):
    """Get data freshness status for a specific pair"""
    try:
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        result = await external_api_service.get_live_data_status(pair)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error getting data freshness for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data freshness for {pair}: {str(e)}")

@router.get("/apis/health")
async def get_api_health_status():
    """Get health status of all external APIs"""
    try:
        result = await external_api_service.get_api_health_status()
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error getting API health status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get API health status: {str(e)}")

@router.post("/data/ensure-freshness/{pair}")
async def ensure_data_freshness(pair: str):
    """Ensure data freshness for sophisticated interface"""
    try:
        if not pair.endswith("USDT"):
            pair = f"{pair}USDT"
        
        result = await external_api_service.ensure_data_freshness(pair)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error ensuring data freshness for {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ensure data freshness for {pair}: {str(e)}")
