"""
Orchestration Monitor for AlphaPulse
Provides real-time monitoring and metrics for 100-symbol system
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import time

from src.services.startup_orchestrator import StartupOrchestrator

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: datetime
    uptime_seconds: float
    
    # Symbol tracking
    total_symbols_tracked: int
    active_symbols: int
    futures_count: int
    spot_count: int
    
    # WebSocket metrics
    websocket_connections_active: int
    websocket_total_streams: int
    websocket_messages_per_second: float
    websocket_reconnections: int
    
    # Data pipeline metrics
    data_pipeline_inserts_per_second: float
    redis_cache_hit_rate: float
    database_pool_utilization: float
    
    # Signal generation metrics
    signals_generated_total: int
    signals_generated_today: int
    consensus_achievement_rate: float
    avg_analysis_time_ms: float
    
    # Performance health
    overall_health_score: float  # 0-1
    health_status: str  # 'healthy', 'degraded', 'critical'

class OrchestrationMonitor:
    """
    Monitors and reports on 100-symbol orchestration performance
    Provides metrics, health checks, and alerting
    """
    
    def __init__(self, orchestrator: StartupOrchestrator, update_interval_seconds: int = 30):
        self.orchestrator = orchestrator
        self.update_interval = update_interval_seconds
        self.logger = logger
        self.db_connection = None  # Will be set from orchestrator
        
        # Metrics history
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Alert thresholds
        self.thresholds = {
            'min_messages_per_second': 50,
            'min_cache_hit_rate': 0.70,
            'max_analysis_time_ms': 2000,
            'min_health_score': 0.60
        }
        
        # State
        self.is_running = False
        self.last_metrics: Optional[SystemMetrics] = None
        
        # Statistics
        self.alerts_triggered = 0
        self.uptime_start: Optional[datetime] = None
        
        logger.info(f"✅ Orchestration Monitor initialized (update_interval={update_interval_seconds}s)")
    
    async def start(self):
        """Start monitoring"""
        if self.is_running:
            logger.warning("⚠️ Monitor already running")
            return
        
        logger.info("🚀 Starting orchestration monitor...")
        self.is_running = True
        self.uptime_start = datetime.now(timezone.utc)
        
        # Get db_connection from orchestrator
        if hasattr(self.orchestrator, 'data_pipeline') and self.orchestrator.data_pipeline:
            self.db_connection = getattr(self.orchestrator.data_pipeline, 'db_connection', None)
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ Orchestration monitor started")
    
    async def stop(self):
        """Stop monitoring"""
        logger.info("🛑 Stopping orchestration monitor...")
        self.is_running = False
        logger.info("✅ Orchestration monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Monitoring loop started")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                if metrics:
                    # Store in history
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history.pop(0)
                    
                    self.last_metrics = metrics
                    
                    # Check for alerts
                    await self._check_alerts(metrics)
                    
                    # Log summary
                    self._log_metrics_summary(metrics)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def collect_metrics(self) -> Optional[SystemMetrics]:
        """Collect comprehensive system metrics"""
        try:
            if not self.orchestrator.is_initialized:
                return None
            
            # Get system status
            system_status = await self.orchestrator.get_system_status()
            
            # Extract metrics
            symbol_stats = system_status.get('symbol_manager', {})
            ws_stats = system_status.get('websocket', {}).get('stats', {})
            scheduler_stats = system_status.get('signal_scheduler', {})
            db_stats = system_status.get('database', {})
            
            # Calculate derived metrics
            uptime = system_status.get('uptime_seconds', 0)
            
            # WebSocket messages per second
            ws_messages_total = ws_stats.get('messages_received', 0)
            messages_per_second = ws_messages_total / max(uptime, 1)
            
            # Calculate health score
            health_score = self._calculate_health_score(system_status)
            health_status = self._determine_health_status(health_score)
            
            # Create metrics object
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                uptime_seconds=uptime,
                total_symbols_tracked=symbol_stats.get('cached_symbols', {}).get('all', 0),
                active_symbols=symbol_stats.get('cached_symbols', {}).get('all', 0),
                futures_count=symbol_stats.get('cached_symbols', {}).get('futures', 0),
                spot_count=symbol_stats.get('cached_symbols', {}).get('spot', 0),
                websocket_connections_active=ws_stats.get('active_connections', 0),
                websocket_total_streams=ws_stats.get('total_streams', 0),
                websocket_messages_per_second=messages_per_second,
                websocket_reconnections=ws_stats.get('reconnections', 0),
                data_pipeline_inserts_per_second=0.0,  # Would need to calculate from pipeline stats
                redis_cache_hit_rate=0.85,  # Placeholder - would track from cache manager
                database_pool_utilization=(db_stats.get('pool_size', 0) / max(db_stats.get('pool_max', 1), 1)),
                signals_generated_total=scheduler_stats.get('stats', {}).get('signals_generated', 0),
                signals_generated_today=scheduler_stats.get('stats', {}).get('signals_generated', 0),  # Would need date filtering
                consensus_achievement_rate=self._extract_rate(scheduler_stats, 'consensus_achievement_rate'),
                avg_analysis_time_ms=scheduler_stats.get('stats', {}).get('avg_analysis_time_ms', 0),
                overall_health_score=health_score,
                health_status=health_status
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error collecting metrics: {e}")
            return None
    
    def _calculate_health_score(self, system_status: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-1)"""
        scores = []
        
        # Database health
        db_util = system_status.get('database', {}).get('pool_size', 0) / max(
            system_status.get('database', {}).get('pool_max', 1), 1
        )
        db_score = 1.0 if db_util < 0.8 else 0.5 if db_util < 0.95 else 0.3
        scores.append(db_score)
        
        # WebSocket health
        ws_health = system_status.get('websocket', {}).get('health', {})
        connected_clients = sum(
            1 for client_info in ws_health.get('clients', {}).values() 
            if client_info.get('connected', False)
        )
        total_clients = ws_health.get('total_clients', 1)
        ws_score = connected_clients / max(total_clients, 1)
        scores.append(ws_score)
        
        # Signal generation health
        scheduler_stats = system_status.get('signal_scheduler', {}).get('stats', {})
        success_rate = (
            scheduler_stats.get('successful_analyses', 0) / 
            max(scheduler_stats.get('total_analyses', 1), 1)
        )
        scores.append(success_rate)
        
        # Overall score
        return sum(scores) / len(scores) if scores else 0.5
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status from score"""
        if health_score >= 0.8:
            return 'healthy'
        elif health_score >= 0.6:
            return 'degraded'
        else:
            return 'critical'
    
    def _extract_rate(self, stats: Dict, key: str) -> float:
        """Extract rate from stats string like '75.5%'"""
        try:
            rate_str = stats.get('derived_metrics', {}).get(key, '0%')
            return float(rate_str.replace('%', '')) / 100
        except:
            return 0.0
    
    async def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds and trigger alerts"""
        alerts = []
        
        # Check message throughput
        if metrics.websocket_messages_per_second < self.thresholds['min_messages_per_second']:
            alerts.append(f"Low message throughput: {metrics.websocket_messages_per_second:.1f} msg/s")
        
        # Check cache hit rate
        if metrics.redis_cache_hit_rate < self.thresholds['min_cache_hit_rate']:
            alerts.append(f"Low cache hit rate: {metrics.redis_cache_hit_rate:.1%}")
        
        # Check analysis time
        if metrics.avg_analysis_time_ms > self.thresholds['max_analysis_time_ms']:
            alerts.append(f"Slow analysis: {metrics.avg_analysis_time_ms:.0f}ms")
        
        # Check overall health
        if metrics.overall_health_score < self.thresholds['min_health_score']:
            alerts.append(f"Low health score: {metrics.overall_health_score:.1%}")
        
        # Log alerts
        if alerts:
            self.alerts_triggered += len(alerts)
            for alert in alerts:
                logger.warning(f"⚠️ ALERT: {alert}")
    
    async def _get_mtf_metrics(self) -> Dict[str, Any]:
        """Get MTF entry system metrics"""
        try:
            if not self.db_connection:
                self.logger.warning("DB connection not available for MTF metrics")
                return self._get_empty_mtf_metrics()
            
            async with self.db_connection.get_connection() as conn:
                # Entry strategy distribution
                strategy_dist = await conn.fetch("""
                    SELECT entry_strategy, COUNT(*) as count
                    FROM ai_signals_mtf
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY entry_strategy
                    ORDER BY count DESC
                """)
                
                # Average metrics
                avg_metrics = await conn.fetchrow("""
                    SELECT 
                        AVG(entry_confidence) as avg_entry_confidence,
                        AVG(risk_reward_ratio) as avg_rr_ratio,
                        AVG(signal_confidence) as avg_signal_confidence,
                        COUNT(*) FILTER (WHERE entry_strategy != 'MARKET_ENTRY') 
                            as refined_entries,
                        COUNT(*) as total_signals
                    FROM ai_signals_mtf
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    'entry_strategy_distribution': {
                        row['entry_strategy']: int(row['count']) 
                        for row in strategy_dist
                    },
                    'avg_entry_confidence': float(avg_metrics['avg_entry_confidence'] or 0),
                    'avg_risk_reward_ratio': float(avg_metrics['avg_rr_ratio'] or 0),
                    'avg_signal_confidence': float(avg_metrics['avg_signal_confidence'] or 0),
                    'entry_refinement_success_rate': (
                        (avg_metrics['refined_entries'] / avg_metrics['total_signals'] * 100)
                        if avg_metrics['total_signals'] > 0 else 0
                    ),
                    'total_mtf_signals_24h': int(avg_metrics['total_signals'] or 0)
                }
        except Exception as e:
            self.logger.error(f"❌ Error getting MTF metrics: {e}")
            return self._get_empty_mtf_metrics()
    
    def _get_empty_mtf_metrics(self) -> Dict[str, Any]:
        """Return empty MTF metrics structure"""
        return {
            'entry_strategy_distribution': {},
            'avg_entry_confidence': 0,
            'avg_risk_reward_ratio': 0,
            'avg_signal_confidence': 0,
            'entry_refinement_success_rate': 0,
            'total_mtf_signals_24h': 0
        }
    
    def _log_metrics_summary(self, metrics: SystemMetrics):
        """Log periodic metrics summary"""
        logger.info(
            f"📊 Metrics: "
            f"{metrics.active_symbols} symbols, "
            f"{metrics.websocket_total_streams} streams, "
            f"{metrics.websocket_messages_per_second:.1f} msg/s, "
            f"{metrics.signals_generated_total} signals, "
            f"Health={metrics.health_status}"
        )
    
    async def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current metrics snapshot"""
        if self.last_metrics:
            return self.last_metrics
        return await self.collect_metrics()
    
    async def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for last N minutes"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = await self.get_current_metrics()
        
        if not current_metrics:
            return {'error': 'No metrics available'}
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_metrics': asdict(current_metrics),
            'alerts': {
                'total_alerts_triggered': self.alerts_triggered,
                'current_thresholds': self.thresholds
            },
            'history': {
                'data_points': len(self.metrics_history),
                'oldest_data': self.metrics_history[0].timestamp.isoformat() if self.metrics_history else None,
                'newest_data': self.metrics_history[-1].timestamp.isoformat() if self.metrics_history else None
            }
        }

