"""
Performance Monitoring API Endpoints
Provides access to system performance metrics and monitoring data
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncpg

from ..monitoring.performance_monitor import PerformanceMonitor
from ..database.connection import get_db_pool

router = APIRouter(prefix="/performance", tags=["Performance Monitoring"])

# Global performance monitor instance
performance_monitor: Optional[PerformanceMonitor] = None

async def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        db_pool = await get_db_pool()
        performance_monitor = PerformanceMonitor(db_pool)
    return performance_monitor

@router.get("/summary")
async def get_performance_summary(
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
) -> Dict[str, Any]:
    """Get current performance summary"""
    try:
        summary = monitor.get_performance_summary()
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance summary: {str(e)}")

@router.get("/alerts")
async def get_performance_alerts(
    limit: int = 10,
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
) -> Dict[str, Any]:
    """Get recent performance alerts"""
    try:
        alerts = monitor.get_recent_alerts(limit)
        return {
            "status": "success",
            "data": alerts,
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@router.get("/metrics/system")
async def get_system_metrics(
    hours: int = 24,
    db_pool: asyncpg.Pool = Depends(get_db_pool)
) -> Dict[str, Any]:
    """Get system metrics for the specified time period"""
    try:
        async with db_pool.acquire() as conn:
            # Get system metrics for the last N hours
            query = """
                SELECT timestamp, cpu_percent, memory_percent, memory_used_gb,
                       disk_usage_percent, network_io_mb, active_connections,
                       database_connections, redis_connections
                FROM system_metrics
                WHERE timestamp >= $1
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            rows = await conn.fetch(query, cutoff_time)
            
            metrics = []
            for row in rows:
                metrics.append({
                    "timestamp": row['timestamp'].isoformat(),
                    "cpu_percent": float(row['cpu_percent']) if row['cpu_percent'] else 0.0,
                    "memory_percent": float(row['memory_percent']) if row['memory_percent'] else 0.0,
                    "memory_used_gb": float(row['memory_used_gb']) if row['memory_used_gb'] else 0.0,
                    "disk_usage_percent": float(row['disk_usage_percent']) if row['disk_usage_percent'] else 0.0,
                    "network_io_mb": float(row['network_io_mb']) if row['network_io_mb'] else 0.0,
                    "active_connections": row['active_connections'] or 0,
                    "database_connections": row['database_connections'] or 0,
                    "redis_connections": row['redis_connections'] or 0
                })
            
            return {
                "status": "success",
                "data": metrics,
                "count": len(metrics),
                "period_hours": hours,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")

@router.get("/metrics/trading")
async def get_trading_metrics(
    hours: int = 24,
    db_pool: asyncpg.Pool = Depends(get_db_pool)
) -> Dict[str, Any]:
    """Get trading metrics for the specified time period"""
    try:
        async with db_pool.acquire() as conn:
            # Get trading metrics for the last N hours
            query = """
                SELECT timestamp, symbols_processed, patterns_detected, signals_generated,
                       avg_processing_time_ms, data_collection_latency_ms, websocket_latency_ms,
                       database_query_time_ms, cache_hit_rate, error_rate
                FROM trading_metrics
                WHERE timestamp >= $1
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            rows = await conn.fetch(query, cutoff_time)
            
            metrics = []
            for row in rows:
                metrics.append({
                    "timestamp": row['timestamp'].isoformat(),
                    "symbols_processed": row['symbols_processed'] or 0,
                    "patterns_detected": row['patterns_detected'] or 0,
                    "signals_generated": row['signals_generated'] or 0,
                    "avg_processing_time_ms": float(row['avg_processing_time_ms']) if row['avg_processing_time_ms'] else 0.0,
                    "data_collection_latency_ms": float(row['data_collection_latency_ms']) if row['data_collection_latency_ms'] else 0.0,
                    "websocket_latency_ms": float(row['websocket_latency_ms']) if row['websocket_latency_ms'] else 0.0,
                    "database_query_time_ms": float(row['database_query_time_ms']) if row['database_query_time_ms'] else 0.0,
                    "cache_hit_rate": float(row['cache_hit_rate']) if row['cache_hit_rate'] else 0.0,
                    "error_rate": float(row['error_rate']) if row['error_rate'] else 0.0
                })
            
            return {
                "status": "success",
                "data": metrics,
                "count": len(metrics),
                "period_hours": hours,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trading metrics: {str(e)}")

@router.post("/monitoring/start")
async def start_performance_monitoring(
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
) -> Dict[str, Any]:
    """Start performance monitoring"""
    try:
        await monitor.start_monitoring()
        return {
            "status": "success",
            "message": "Performance monitoring started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")

@router.post("/monitoring/stop")
async def stop_performance_monitoring(
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
) -> Dict[str, Any]:
    """Stop performance monitoring"""
    try:
        await monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "Performance monitoring stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")

@router.get("/health")
async def get_performance_health(
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
) -> Dict[str, Any]:
    """Get performance health status"""
    try:
        summary = monitor.get_performance_summary()
        
        # Determine overall health
        health_status = "healthy"
        issues = []
        
        if summary:
            system = summary.get('system', {})
            trading = summary.get('trading', {})
            
            # Check CPU
            cpu_percent = system.get('cpu_percent', 0)
            if cpu_percent > 90:
                health_status = "critical"
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                health_status = "warning"
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            # Check memory
            memory_percent = system.get('memory_percent', 0)
            if memory_percent > 95:
                health_status = "critical"
                issues.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > 80:
                health_status = "warning"
                issues.append(f"Memory usage high: {memory_percent:.1f}%")
            
            # Check latency
            avg_latency = trading.get('avg_processing_time_ms', 0)
            if avg_latency > 500:
                health_status = "critical"
                issues.append(f"Processing latency critical: {avg_latency:.1f}ms")
            elif avg_latency > 100:
                health_status = "warning"
                issues.append(f"Processing latency high: {avg_latency:.1f}ms")
            
            # Check error rate
            error_rate = trading.get('error_rate', 0)
            if error_rate > 0.10:
                health_status = "critical"
                issues.append(f"Error rate critical: {error_rate:.2%}")
            elif error_rate > 0.05:
                health_status = "warning"
                issues.append(f"Error rate high: {error_rate:.2%}")
        
        return {
            "status": "success",
            "health": {
                "overall_status": health_status,
                "issues": issues,
                "monitoring_active": monitor.is_monitoring,
                "alerts_count": len(monitor.alerts)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting health status: {str(e)}")

@router.get("/dashboard")
async def get_performance_dashboard(
    monitor: PerformanceMonitor = Depends(get_performance_monitor),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
) -> Dict[str, Any]:
    """Get comprehensive performance dashboard data"""
    try:
        # Get current summary
        summary = monitor.get_performance_summary()
        
        # Get recent alerts
        alerts = monitor.get_recent_alerts(5)
        
        # Get latest metrics
        async with db_pool.acquire() as conn:
            # Latest system metrics
            latest_system = await conn.fetchrow("""
                SELECT * FROM system_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            # Latest trading metrics
            latest_trading = await conn.fetchrow("""
                SELECT * FROM trading_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            # Get metrics for the last hour
            hour_ago = datetime.now() - timedelta(hours=1)
            
            # System metrics trend
            system_trend = await conn.fetch("""
                SELECT timestamp, cpu_percent, memory_percent
                FROM system_metrics
                WHERE timestamp >= $1
                ORDER BY timestamp ASC
            """, hour_ago)
            
            # Trading metrics trend
            trading_trend = await conn.fetch("""
                SELECT timestamp, patterns_detected, signals_generated, avg_processing_time_ms
                FROM trading_metrics
                WHERE timestamp >= $1
                ORDER BY timestamp ASC
            """, hour_ago)
        
        return {
            "status": "success",
            "dashboard": {
                "current_summary": summary,
                "recent_alerts": alerts,
                "latest_system": dict(latest_system) if latest_system else None,
                "latest_trading": dict(latest_trading) if latest_trading else None,
                "system_trend": [dict(row) for row in system_trend],
                "trading_trend": [dict(row) for row in trading_trend],
                "monitoring_active": monitor.is_monitoring
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard: {str(e)}")
