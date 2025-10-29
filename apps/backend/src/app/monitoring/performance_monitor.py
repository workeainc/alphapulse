"""
Performance Monitor for AlphaPlus
Comprehensive system monitoring and metrics collection
"""

import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncpg

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    network_io_mb: float
    active_connections: int
    database_connections: int
    redis_connections: int

@dataclass
class TradingMetrics:
    """Trading system performance metrics"""
    timestamp: datetime
    symbols_processed: int
    patterns_detected: int
    signals_generated: int
    avg_processing_time_ms: float
    data_collection_latency_ms: float
    websocket_latency_ms: float
    database_query_time_ms: float
    cache_hit_rate: float
    error_rate: float

@dataclass
class EnhancedAlgorithmMetrics:
    """Enhanced algorithm performance metrics"""
    timestamp: datetime
    psychological_levels_processed: int
    volume_weighted_levels_processed: int
    orderbook_analyses_processed: int
    algorithm_integration_cycles: int
    avg_psychological_analysis_time_ms: float
    avg_volume_analysis_time_ms: float
    avg_orderbook_analysis_time_ms: float
    algorithm_success_rate: float
    enhanced_signals_generated: int
    algorithm_data_points_collected: int

@dataclass
class PerformanceAlert:
    """Performance alert"""
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, Any]

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        
        # Metrics storage
        self.system_metrics = deque(maxlen=1000)
        self.trading_metrics = deque(maxlen=1000)
        self.enhanced_algorithm_metrics = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'latency_warning': 100.0,  # ms
            'latency_critical': 500.0,  # ms
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.10,  # 10%
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Performance counters
        self.counters = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'patterns_detected': 0,
            'signals_generated': 0,
            'data_points_collected': 0,
            # Enhanced algorithm counters
            'psychological_levels_processed': 0,
            'volume_weighted_levels_processed': 0,
            'orderbook_analyses_processed': 0,
            'algorithm_integration_cycles': 0,
            'enhanced_signals_generated': 0,
            'algorithm_data_points_collected': 0,
        }
        
        # Latency tracking
        self.latency_times = deque(maxlen=1000)
        self.psychological_analysis_times = deque(maxlen=1000)
        self.volume_analysis_times = deque(maxlen=1000)
        self.orderbook_analysis_times = deque(maxlen=1000)
        
        logger.info("🚀 Performance Monitor initialized")
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            logger.warning("⚠️ Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("✅ Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect trading metrics
                trading_metrics = await self._collect_trading_metrics()
                self.trading_metrics.append(trading_metrics)
                
                # Collect enhanced algorithm metrics
                enhanced_algorithm_metrics = await self._collect_enhanced_algorithm_metrics()
                self.enhanced_algorithm_metrics.append(enhanced_algorithm_metrics)
                
                # Check for alerts
                await self._check_alerts(system_metrics, trading_metrics, enhanced_algorithm_metrics)
                
                # Store metrics in database
                await self._store_metrics(system_metrics, trading_metrics, enhanced_algorithm_metrics)
                
                # Log performance summary
                await self._log_performance_summary()
                
                # Wait for next collection cycle
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io_mb = (network.bytes_sent + network.bytes_recv) / (1024**2)
            
            # Connection counts (approximate)
            active_connections = len(psutil.net_connections())
            database_connections = await self._get_database_connections()
            redis_connections = await self._get_redis_connections()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                disk_usage_percent=disk_usage_percent,
                network_io_mb=network_io_mb,
                active_connections=active_connections,
                database_connections=database_connections,
                redis_connections=redis_connections
            )
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                disk_usage_percent=0.0,
                network_io_mb=0.0,
                active_connections=0,
                database_connections=0,
                redis_connections=0
            )
    
    async def _collect_trading_metrics(self) -> TradingMetrics:
        """Collect trading system metrics"""
        try:
            # Calculate average processing time
            avg_processing_time = 0.0
            if self.latency_times:
                avg_processing_time = sum(self.latency_times) / len(self.latency_times)
            
            # Calculate error rate
            total_requests = self.counters['total_requests']
            failed_requests = self.counters['failed_requests']
            error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
            
            # Calculate cache hit rate (approximate)
            cache_hit_rate = 0.85  # Placeholder - would need Redis metrics
            
            return TradingMetrics(
                timestamp=datetime.now(),
                symbols_processed=len(['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']),
                patterns_detected=self.counters['patterns_detected'],
                signals_generated=self.counters['signals_generated'],
                avg_processing_time_ms=avg_processing_time,
                data_collection_latency_ms=50.0,  # Placeholder
                websocket_latency_ms=10.0,  # Placeholder
                database_query_time_ms=20.0,  # Placeholder
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"❌ Error collecting trading metrics: {e}")
            return TradingMetrics(
                timestamp=datetime.now(),
                symbols_processed=0,
                patterns_detected=0,
                signals_generated=0,
                avg_processing_time_ms=0.0,
                data_collection_latency_ms=0.0,
                websocket_latency_ms=0.0,
                database_query_time_ms=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0
            )
    
    async def _collect_enhanced_algorithm_metrics(self) -> EnhancedAlgorithmMetrics:
        """Collect enhanced algorithm performance metrics"""
        try:
            # Calculate average analysis times
            avg_psychological_time = 0.0
            if self.psychological_analysis_times:
                avg_psychological_time = sum(self.psychological_analysis_times) / len(self.psychological_analysis_times)
            
            avg_volume_time = 0.0
            if self.volume_analysis_times:
                avg_volume_time = sum(self.volume_analysis_times) / len(self.volume_analysis_times)
            
            avg_orderbook_time = 0.0
            if self.orderbook_analysis_times:
                avg_orderbook_time = sum(self.orderbook_analysis_times) / len(self.orderbook_analysis_times)
            
            # Calculate algorithm success rate
            total_algorithm_requests = (
                self.counters['psychological_levels_processed'] +
                self.counters['volume_weighted_levels_processed'] +
                self.counters['orderbook_analyses_processed']
            )
            successful_algorithm_requests = (
                self.counters['psychological_levels_processed'] +
                self.counters['volume_weighted_levels_processed'] +
                self.counters['orderbook_analyses_processed']
            )  # Simplified - would need actual success tracking
            
            algorithm_success_rate = (
                successful_algorithm_requests / total_algorithm_requests 
                if total_algorithm_requests > 0 else 1.0
            )
            
            return EnhancedAlgorithmMetrics(
                timestamp=datetime.now(),
                psychological_levels_processed=self.counters['psychological_levels_processed'],
                volume_weighted_levels_processed=self.counters['volume_weighted_levels_processed'],
                orderbook_analyses_processed=self.counters['orderbook_analyses_processed'],
                algorithm_integration_cycles=self.counters['algorithm_integration_cycles'],
                avg_psychological_analysis_time_ms=avg_psychological_time,
                avg_volume_analysis_time_ms=avg_volume_time,
                avg_orderbook_analysis_time_ms=avg_orderbook_time,
                algorithm_success_rate=algorithm_success_rate,
                enhanced_signals_generated=self.counters['enhanced_signals_generated'],
                algorithm_data_points_collected=self.counters['algorithm_data_points_collected']
            )
            
        except Exception as e:
            logger.error(f"❌ Error collecting enhanced algorithm metrics: {e}")
            return EnhancedAlgorithmMetrics(
                timestamp=datetime.now(),
                psychological_levels_processed=0,
                volume_weighted_levels_processed=0,
                orderbook_analyses_processed=0,
                algorithm_integration_cycles=0,
                avg_psychological_analysis_time_ms=0.0,
                avg_volume_analysis_time_ms=0.0,
                avg_orderbook_analysis_time_ms=0.0,
                algorithm_success_rate=0.0,
                enhanced_signals_generated=0,
                algorithm_data_points_collected=0
            )
    
    async def _get_database_connections(self) -> int:
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                )
                return result or 0
        except Exception:
            return 0
    
    async def _get_redis_connections(self) -> int:
        """Get active Redis connections"""
        try:
            # This would need Redis client to get actual connection count
            return 5  # Placeholder
        except Exception:
            return 0
    
    async def _check_alerts(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics, enhanced_algorithm_metrics: EnhancedAlgorithmMetrics):
        """Check for performance alerts"""
        alerts = []
        
        # CPU alerts
        if system_metrics.cpu_percent > self.thresholds['cpu_critical']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='CPU_CRITICAL',
                severity='CRITICAL',
                message=f"CPU usage critical: {system_metrics.cpu_percent:.1f}%",
                metrics={'cpu_percent': system_metrics.cpu_percent}
            ))
        elif system_metrics.cpu_percent > self.thresholds['cpu_warning']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='CPU_WARNING',
                severity='WARNING',
                message=f"CPU usage high: {system_metrics.cpu_percent:.1f}%",
                metrics={'cpu_percent': system_metrics.cpu_percent}
            ))
        
        # Memory alerts
        if system_metrics.memory_percent > self.thresholds['memory_critical']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='MEMORY_CRITICAL',
                severity='CRITICAL',
                message=f"Memory usage critical: {system_metrics.memory_percent:.1f}%",
                metrics={'memory_percent': system_metrics.memory_percent}
            ))
        elif system_metrics.memory_percent > self.thresholds['memory_warning']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='MEMORY_WARNING',
                severity='WARNING',
                message=f"Memory usage high: {system_metrics.memory_percent:.1f}%",
                metrics={'memory_percent': system_metrics.memory_percent}
            ))
        
        # Latency alerts
        if trading_metrics.avg_processing_time_ms > self.thresholds['latency_critical']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='LATENCY_CRITICAL',
                severity='CRITICAL',
                message=f"Processing latency critical: {trading_metrics.avg_processing_time_ms:.1f}ms",
                metrics={'avg_processing_time_ms': trading_metrics.avg_processing_time_ms}
            ))
        elif trading_metrics.avg_processing_time_ms > self.thresholds['latency_warning']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='LATENCY_WARNING',
                severity='WARNING',
                message=f"Processing latency high: {trading_metrics.avg_processing_time_ms:.1f}ms",
                metrics={'avg_processing_time_ms': trading_metrics.avg_processing_time_ms}
            ))
        
        # Error rate alerts
        if trading_metrics.error_rate > self.thresholds['error_rate_critical']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='ERROR_RATE_CRITICAL',
                severity='CRITICAL',
                message=f"Error rate critical: {trading_metrics.error_rate:.2%}",
                metrics={'error_rate': trading_metrics.error_rate}
            ))
        elif trading_metrics.error_rate > self.thresholds['error_rate_warning']:
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='ERROR_RATE_WARNING',
                severity='WARNING',
                message=f"Error rate high: {trading_metrics.error_rate:.2%}",
                metrics={'error_rate': trading_metrics.error_rate}
            ))
        
        # Enhanced algorithm alerts
        if enhanced_algorithm_metrics.algorithm_success_rate < 0.8:  # Less than 80% success rate
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='ALGORITHM_SUCCESS_RATE_LOW',
                severity='WARNING',
                message=f"Algorithm success rate low: {enhanced_algorithm_metrics.algorithm_success_rate:.2%}",
                metrics={'algorithm_success_rate': enhanced_algorithm_metrics.algorithm_success_rate}
            ))
        
        if enhanced_algorithm_metrics.avg_psychological_analysis_time_ms > 1000:  # More than 1 second
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='PSYCHOLOGICAL_ANALYSIS_SLOW',
                severity='WARNING',
                message=f"Psychological analysis slow: {enhanced_algorithm_metrics.avg_psychological_analysis_time_ms:.1f}ms",
                metrics={'avg_psychological_analysis_time_ms': enhanced_algorithm_metrics.avg_psychological_analysis_time_ms}
            ))
        
        if enhanced_algorithm_metrics.avg_volume_analysis_time_ms > 1000:  # More than 1 second
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='VOLUME_ANALYSIS_SLOW',
                severity='WARNING',
                message=f"Volume analysis slow: {enhanced_algorithm_metrics.avg_volume_analysis_time_ms:.1f}ms",
                metrics={'avg_volume_analysis_time_ms': enhanced_algorithm_metrics.avg_volume_analysis_time_ms}
            ))
        
        if enhanced_algorithm_metrics.avg_orderbook_analysis_time_ms > 1000:  # More than 1 second
            alerts.append(PerformanceAlert(
                timestamp=datetime.now(),
                alert_type='ORDERBOOK_ANALYSIS_SLOW',
                severity='WARNING',
                message=f"Orderbook analysis slow: {enhanced_algorithm_metrics.avg_orderbook_analysis_time_ms:.1f}ms",
                metrics={'avg_orderbook_analysis_time_ms': enhanced_algorithm_metrics.avg_orderbook_analysis_time_ms}
            ))
        
        # Add alerts to queue
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"🚨 {alert.severity}: {alert.message}")
    
    async def _store_metrics(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics, enhanced_algorithm_metrics: EnhancedAlgorithmMetrics):
        """Store metrics in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store system metrics
                await conn.execute("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, memory_used_gb,
                        disk_usage_percent, network_io_mb, active_connections,
                        database_connections, redis_connections
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, system_metrics.timestamp, system_metrics.cpu_percent,
                     system_metrics.memory_percent, system_metrics.memory_used_gb,
                     system_metrics.disk_usage_percent, system_metrics.network_io_mb,
                     system_metrics.active_connections, system_metrics.database_connections,
                     system_metrics.redis_connections)
                
                # Store trading metrics
                await conn.execute("""
                    INSERT INTO trading_metrics (
                        timestamp, symbols_processed, patterns_detected, signals_generated,
                        avg_processing_time_ms, data_collection_latency_ms, websocket_latency_ms,
                        database_query_time_ms, cache_hit_rate, error_rate
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, trading_metrics.timestamp, trading_metrics.symbols_processed,
                     trading_metrics.patterns_detected, trading_metrics.signals_generated,
                     trading_metrics.avg_processing_time_ms, trading_metrics.data_collection_latency_ms,
                     trading_metrics.websocket_latency_ms, trading_metrics.database_query_time_ms,
                     trading_metrics.cache_hit_rate, trading_metrics.error_rate)
                
                # Store enhanced algorithm metrics
                await conn.execute("""
                    INSERT INTO enhanced_algorithm_metrics (
                        timestamp, psychological_levels_processed, volume_weighted_levels_processed,
                        orderbook_analyses_processed, algorithm_integration_cycles,
                        avg_psychological_analysis_time_ms, avg_volume_analysis_time_ms,
                        avg_orderbook_analysis_time_ms, algorithm_success_rate,
                        enhanced_signals_generated, algorithm_data_points_collected
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, enhanced_algorithm_metrics.timestamp, enhanced_algorithm_metrics.psychological_levels_processed,
                     enhanced_algorithm_metrics.volume_weighted_levels_processed, enhanced_algorithm_metrics.orderbook_analyses_processed,
                     enhanced_algorithm_metrics.algorithm_integration_cycles, enhanced_algorithm_metrics.avg_psychological_analysis_time_ms,
                     enhanced_algorithm_metrics.avg_volume_analysis_time_ms, enhanced_algorithm_metrics.avg_orderbook_analysis_time_ms,
                     enhanced_algorithm_metrics.algorithm_success_rate, enhanced_algorithm_metrics.enhanced_signals_generated,
                     enhanced_algorithm_metrics.algorithm_data_points_collected)
                
        except Exception as e:
            logger.error(f"❌ Error storing metrics: {e}")
    
    async def _log_performance_summary(self):
        """Log performance summary including enhanced algorithms"""
        if not self.system_metrics or not self.trading_metrics or not self.enhanced_algorithm_metrics:
            return
        
        latest_system = self.system_metrics[-1]
        latest_trading = self.trading_metrics[-1]
        latest_enhanced = self.enhanced_algorithm_metrics[-1]
        
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   CPU: {latest_system.cpu_percent:.1f}% | Memory: {latest_system.memory_percent:.1f}%")
        logger.info(f"   Patterns: {latest_trading.patterns_detected} | Signals: {latest_trading.signals_generated}")
        logger.info(f"   Avg Latency: {latest_trading.avg_processing_time_ms:.1f}ms | Error Rate: {latest_trading.error_rate:.2%}")
        logger.info(f"   Enhanced Algorithms:")
        logger.info(f"     Psychological: {latest_enhanced.psychological_levels_processed} | Volume: {latest_enhanced.volume_weighted_levels_processed}")
        logger.info(f"     Orderbook: {latest_enhanced.orderbook_analyses_processed} | Success Rate: {latest_enhanced.algorithm_success_rate:.2%}")
        logger.info(f"     Enhanced Signals: {latest_enhanced.enhanced_signals_generated} | Data Points: {latest_enhanced.algorithm_data_points_collected}")
    
    def record_request(self, success: bool, processing_time_ms: float):
        """Record a request for metrics"""
        self.counters['total_requests'] += 1
        if success:
            self.counters['successful_requests'] += 1
        else:
            self.counters['failed_requests'] += 1
        
        self.latency_times.append(processing_time_ms)
    
    def record_pattern_detected(self):
        """Record pattern detection"""
        self.counters['patterns_detected'] += 1
    
    def record_signal_generated(self):
        """Record signal generation"""
        self.counters['signals_generated'] += 1
    
    def record_psychological_analysis(self, processing_time_ms: float):
        """Record psychological levels analysis"""
        self.counters['psychological_levels_processed'] += 1
        self.psychological_analysis_times.append(processing_time_ms)
    
    def record_volume_analysis(self, processing_time_ms: float):
        """Record volume-weighted levels analysis"""
        self.counters['volume_weighted_levels_processed'] += 1
        self.volume_analysis_times.append(processing_time_ms)
    
    def record_orderbook_analysis(self, processing_time_ms: float):
        """Record orderbook analysis"""
        self.counters['orderbook_analyses_processed'] += 1
        self.orderbook_analysis_times.append(processing_time_ms)
    
    def record_algorithm_integration_cycle(self):
        """Record algorithm integration cycle"""
        self.counters['algorithm_integration_cycles'] += 1
    
    def record_enhanced_signal_generated(self):
        """Record enhanced signal generation"""
        self.counters['enhanced_signals_generated'] += 1
    
    def record_algorithm_data_points(self, count: int):
        """Record algorithm data points collected"""
        self.counters['algorithm_data_points_collected'] += count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary including enhanced algorithms"""
        if not self.system_metrics or not self.trading_metrics or not self.enhanced_algorithm_metrics:
            return {}
        
        latest_system = self.system_metrics[-1]
        latest_trading = self.trading_metrics[-1]
        latest_enhanced = self.enhanced_algorithm_metrics[-1]
        
        return {
            'system': asdict(latest_system),
            'trading': asdict(latest_trading),
            'enhanced_algorithms': asdict(latest_enhanced),
            'counters': self.counters.copy(),
            'alerts_count': len(self.alerts),
            'is_monitoring': self.is_monitoring
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        return [asdict(alert) for alert in list(self.alerts)[-limit:]]
