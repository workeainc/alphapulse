"""
Simplified Main Application for AlphaPlus
FastAPI application with direct database connection
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
from typing import Dict, Any, List
import json
from datetime import datetime
import asyncpg
import asyncio
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AlphaPlus Trading System",
    description="Advanced algorithmic trading system with AI/ML capabilities",
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

# Database connection pool
db_pool = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    global db_pool
    try:
        # Create connection pool
        db_pool = await asyncpg.create_pool(
            host='postgres',
            port=5432,
            database='alphapulse',
            user='alpha_emon',
            password='Emon_@17711',
            min_size=5,
            max_size=20
        )
        logger.info("Database connection pool initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue without database for now

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connection on shutdown"""
    global db_pool
    if db_pool:
        await db_pool.close()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AlphaPlus Trading System API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global db_pool
    db_status = "connected" if db_pool else "disconnected"
    
    return {
        "status": "healthy", 
        "service": "AlphaPlus",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "components": ["trading", "ai", "monitoring"],
        "database": "connected" if db_pool else "disconnected"
    }

# Portfolio endpoints
@app.get("/api/portfolio/overview")
async def get_portfolio_overview():
    """Get portfolio overview data from database"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Get total balance from trades
            balance_query = """
                SELECT 
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 AND status = 'closed' THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 AND status = 'closed' THEN 1 END) as losing_trades
                FROM trades
            """
            balance_result = await conn.fetchrow(balance_query)
            
            # Get open positions
            open_query = """
                SELECT 
                    COUNT(*) as open_count,
                    COALESCE(SUM(pnl), 0) as unrealized_pnl
                FROM trades 
                WHERE status = 'open'
            """
            open_result = await conn.fetchrow(open_query)
            
            # Calculate portfolio metrics
            total_balance = 100000.0  # Starting balance
            total_pnl = float(balance_result['total_pnl'] or 0)
            available_balance = total_balance + total_pnl
            open_positions = int(open_result['open_count'] or 0)
            unrealized_pnl = float(open_result['unrealized_pnl'] or 0)
            
            return {
                "total_balance": total_balance,
                "available_balance": available_balance,
                "total_pnl": total_pnl,
                "total_pnl_percentage": (total_pnl / total_balance * 100) if total_balance > 0 else 0,
                "daily_pnl": 0.0,  # TODO: Calculate daily P&L
                "daily_pnl_percentage": 0.0,
                "open_positions": open_positions,
                "unrealized_pnl": unrealized_pnl,
                "consecutive_losses": 0,
                "daily_loss_limit": 5000.0
            }
                
    except Exception as e:
        logger.error(f"Error getting portfolio overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio data")

@app.get("/api/portfolio/risk")
async def get_portfolio_risk():
    """Get portfolio risk metrics from database"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Get latest risk metrics
            risk_query = """
                SELECT var_95, max_drawdown, sharpe_ratio, sortino_ratio
                FROM performance_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """
            risk_result = await conn.fetchrow(risk_query)
            
            if risk_result:
                return {
                    "var_95": float(risk_result['var_95'] or 2.5),
                    "max_drawdown": float(risk_result['max_drawdown'] or 8.2),
                    "sharpe_ratio": float(risk_result['sharpe_ratio'] or 1.85),
                    "sortino_ratio": float(risk_result['sortino_ratio'] or 2.1),
                    "current_risk": "medium",
                    "daily_loss_limit": 5000.0,
                    "position_size_limit": 10000.0
                }
            else:
                # Return default risk metrics if no data exists
                return {
                    "var_95": 2.5,
                    "max_drawdown": 8.2,
                    "sharpe_ratio": 1.85,
                    "sortino_ratio": 2.1,
                    "current_risk": "medium",
                    "daily_loss_limit": 5000.0,
                    "position_size_limit": 10000.0
                }
                    
    except Exception as e:
        logger.error(f"Error getting portfolio risk: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve risk data")

@app.get("/api/trading/status")
async def get_trading_status():
    """Get current trading status from database"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Get trading statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 AND status = 'closed' THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 AND status = 'closed' THEN 1 END) as losing_trades
                FROM trades
            """
            stats_result = await conn.fetchrow(stats_query)
            
            return {
                "is_running": False,  # TODO: Implement trading bot status
                "last_start": None,   # TODO: Implement trading bot tracking
                "total_trades": int(stats_result['total_trades'] or 0),
                "winning_trades": int(stats_result['winning_trades'] or 0),
                "losing_trades": int(stats_result['losing_trades'] or 0),
                "current_strategy": "mean_reversion"  # TODO: Implement strategy tracking
            }
                
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trading status")

@app.get("/api/trading/history")
async def get_trading_history():
    """Get trading history from database"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Get recent trades
            trades_query = """
                SELECT id, symbol, side, quantity, entry_price, entry_time, pnl
                FROM trades
                ORDER BY entry_time DESC
                LIMIT 10
            """
            trades_result = await conn.fetch(trades_query)
            
            trade_list = []
            for trade in trades_result:
                trade_list.append({
                    "id": str(trade['id']),
                    "symbol": trade['symbol'],
                    "side": trade['side'],
                    "amount": float(trade['quantity']),
                    "price": float(trade['entry_price']),
                    "timestamp": trade['entry_time'].isoformat(),
                    "pnl": float(trade['pnl']) if trade['pnl'] else 0
                })
            
            return {"trades": trade_list}
                
    except Exception as e:
        logger.error(f"Error getting trading history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trading history")

@app.get("/api/strategies/performance")
async def get_strategy_performance():
    """Get strategy performance metrics from database"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Get performance by strategy
            strategy_query = """
                SELECT 
                    strategy_name,
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 AND status = 'closed' THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 AND status = 'closed' THEN 1 END) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                GROUP BY strategy_name
            """
            strategy_result = await conn.fetch(strategy_query)
            
            strategies = {}
            for row in strategy_result:
                strategy_name = row['strategy_name']
                total_trades = int(row['total_trades'] or 0)
                winning_trades = int(row['winning_trades'] or 0)
                
                strategies[strategy_name] = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": int(row['losing_trades'] or 0),
                    "total_pnl": float(row['total_pnl'] or 0),
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    "avg_win": float(row['avg_pnl'] or 0),
                    "avg_loss": 0  # TODO: Calculate average loss separately
                }
            
            # If no strategies found, return default data
            if not strategies:
                strategies = {
                    "mean_reversion": {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_pnl": 0,
                        "win_rate": 0,
                        "avg_win": 0,
                        "avg_loss": 0
                    }
                }
            
            return {"strategies": strategies}
                
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve strategy performance")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send periodic portfolio updates
            await asyncio.sleep(5)
            
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        # Get real-time portfolio data
                        portfolio_query = """
                            SELECT 
                                COALESCE(SUM(pnl), 0) as total_pnl,
                                COUNT(*) as total_trades,
                                COUNT(CASE WHEN status = 'open' THEN 1 END) as open_positions
                            FROM trades
                        """
                        portfolio_result = await conn.fetchrow(portfolio_query)
                        
                        # Get latest market data (simulated for now)
                        real_time_data = {
                            "type": "portfolio_update",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "total_balance": 100000.0,
                                "available_balance": 100000.0 + float(portfolio_result['total_pnl'] or 0),
                                "total_pnl": float(portfolio_result['total_pnl'] or 0),
                                "open_positions": int(portfolio_result['open_positions'] or 0),
                                "total_trades": int(portfolio_result['total_trades'] or 0)
                            },
                            "market_data": {
                                "BTC/USDT": {
                                    "price": 45000 + random.uniform(-500, 500),
                                    "volume": random.uniform(1000, 5000),
                                    "change_24h": random.uniform(-5, 5)
                                },
                                "ETH/USDT": {
                                    "price": 2800 + random.uniform(-100, 100),
                                    "volume": random.uniform(500, 2000),
                                    "change_24h": random.uniform(-3, 3)
                                }
                            }
                        }
                        
                        await websocket.send_text(json.dumps(real_time_data))
                        
                except Exception as e:
                    logger.error(f"Error getting real-time data: {e}")
                    # Send fallback data
                    fallback_data = {
                        "type": "portfolio_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "total_balance": 100000.0,
                            "available_balance": 100000.0,
                            "total_pnl": 0.0,
                            "open_positions": 0,
                            "total_trades": 0
                        }
                    }
                    await websocket.send_text(json.dumps(fallback_data))
            else:
                # Send fallback data if database not available
                fallback_data = {
                    "type": "portfolio_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "total_balance": 100000.0,
                        "available_balance": 100000.0,
                        "total_pnl": 0.0,
                        "open_positions": 0,
                        "total_trades": 0
                    }
                }
                await websocket.send_text(json.dumps(fallback_data))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    try:
        logger.info("Starting AlphaPlus Trading System...")
        uvicorn.run(
            "main_simple:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
