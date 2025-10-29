"""
AlphaPulse Main Entry Point - 100 Symbol Scaled System
Uses full orchestration layer for managing 50 futures + 50 spot symbols
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from src.services.startup_orchestrator import startup_orchestrator
from src.services.orchestration_monitor import OrchestrationMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alphapulse_scaled.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global instances
monitor: OrchestrationMonitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global monitor
    
    logger.info("=" * 80)
    logger.info("🚀 ALPHAPULSE - 100 SYMBOL SCALED SYSTEM STARTING")
    logger.info("=" * 80)
    
    # Get database URL from environment (Docker port 55433)
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse'
    )
    
    # Start orchestration
    try:
        success = await startup_orchestrator.startup(database_url)
        
        if not success:
            logger.error("❌ Startup failed - check logs above")
            raise RuntimeError("Startup orchestrator failed to initialize")
        
        # Start monitoring
        monitor = OrchestrationMonitor(startup_orchestrator, update_interval_seconds=30)
        await monitor.start()
        
        logger.info("✅ System fully operational - ready to generate signals!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Fatal error during startup: {e}")
        raise
    
    # Shutdown
    logger.info("🛑 Initiating system shutdown...")
    
    if monitor:
        await monitor.stop()
    
    await startup_orchestrator.shutdown()
    
    logger.info("✅ Shutdown complete")

# Create FastAPI app with orchestration
app = FastAPI(
    title="AlphaPulse - Scaled Trading System",
    description="100 Symbol Trading Signal Generation with 9-Head AI Consensus",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        if startup_orchestrator.is_initialized:
            status = await startup_orchestrator.get_system_status()
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "system": status
            }
        else:
            return {
                "status": "initializing",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Metrics endpoint
@app.get("/api/v1/metrics")
async def get_metrics():
    """Get current system metrics"""
    try:
        if monitor:
            metrics = await monitor.get_current_metrics()
            if metrics:
                return {
                    "success": True,
                    "metrics": metrics.__dict__,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return {
            "success": False,
            "error": "Metrics not available",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance report endpoint
@app.get("/api/v1/performance")
async def get_performance_report():
    """Get comprehensive performance report"""
    try:
        if monitor:
            report = await monitor.get_performance_report()
            return {
                "success": True,
                "report": report,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise HTTPException(status_code=503, detail="Monitor not initialized")
        
    except Exception as e:
        logger.error(f"❌ Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Symbols endpoint
@app.get("/api/v1/symbols")
async def get_active_symbols():
    """Get list of actively tracked symbols"""
    try:
        if startup_orchestrator.symbol_manager:
            futures = await startup_orchestrator.symbol_manager.get_futures_symbols()
            spot = await startup_orchestrator.symbol_manager.get_spot_symbols()
            
            return {
                "success": True,
                "symbols": {
                    "futures": futures,
                    "spot": spot,
                    "total": len(futures) + len(spot)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise HTTPException(status_code=503, detail="Symbol manager not initialized")
        
    except Exception as e:
        logger.error(f"❌ Error getting symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Signal generation endpoint for specific symbol
@app.get("/api/v1/signals/{symbol}")
async def get_signal_for_symbol(symbol: str, timeframe: str = "1h"):
    """Generate signal for specific symbol"""
    try:
        if startup_orchestrator.ai_service:
            signal = await startup_orchestrator.ai_service.generate_ai_signal(symbol, timeframe)
            
            if signal:
                return {
                    "success": True,
                    "signal": {
                        "symbol": signal.symbol,
                        "direction": signal.signal_direction,
                        "confidence": signal.confidence_score,
                        "probability": signal.probability,
                        "consensus_achieved": signal.consensus_achieved,
                        "agreeing_heads": signal.agreeing_heads,
                        "reasoning": signal.model_reasoning
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": True,
                    "signal": None,
                    "message": "No consensus achieved - no trade signal",
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        raise HTTPException(status_code=503, detail="AI service not initialized")
        
    except Exception as e:
        logger.error(f"❌ Error generating signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket status endpoint
@app.get("/api/v1/websocket/status")
async def get_websocket_status():
    """Get WebSocket connection status"""
    try:
        if startup_orchestrator.websocket_orchestrator:
            health = await startup_orchestrator.websocket_orchestrator.get_health_status()
            stats = startup_orchestrator.websocket_orchestrator.get_stats()
            
            return {
                "success": True,
                "health": health,
                "stats": stats.__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise HTTPException(status_code=503, detail="WebSocket orchestrator not initialized")
        
    except Exception as e:
        logger.error(f"❌ Error getting WebSocket status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scheduler status endpoint
@app.get("/api/v1/scheduler/status")
async def get_scheduler_status():
    """Get signal generation scheduler status"""
    try:
        if startup_orchestrator.signal_scheduler:
            stats = startup_orchestrator.signal_scheduler.get_stats()
            
            return {
                "success": True,
                "scheduler": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise HTTPException(status_code=503, detail="Signal scheduler not initialized")
        
    except Exception as e:
        logger.error(f"❌ Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "name": "AlphaPulse Scaled Trading System",
        "version": "2.0.0",
        "description": "100 Symbol Trading Signal Generation with 9-Head AI Consensus",
        "status": "operational" if startup_orchestrator.is_initialized else "initializing",
        "endpoints": {
            "health": "/health",
            "metrics": "/api/v1/metrics",
            "symbols": "/api/v1/symbols",
            "signals": "/api/v1/signals/{symbol}",
            "websocket_status": "/api/v1/websocket/status",
            "scheduler_status": "/api/v1/scheduler/status",
            "performance": "/api/v1/performance"
        }
    }

if __name__ == "__main__":
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main_scaled:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )

