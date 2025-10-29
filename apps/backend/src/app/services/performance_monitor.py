#!/usr/bin/env python3
"""
Performance Monitoring Service for Ultra-Optimized Pattern Detection
Tracks real-time performance metrics and provides optimization recommendations
"""

import asyncio
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import os
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any] = None

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    timestamp: datetime
    alert_type: str  # 'warning', 'critical', 'info'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    recommendation: str

class PerformanceMonitor:
    """
    Real-time performance monitoring for ultra-optimized pattern detection
    """
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        """Initialize performance monitor"""
        self.alert_thresholds = alert_thresholds or {
            'processing_time_ms': 1000.0,  # 1 second
            'memory_usage_mb': 2048.0,     # 2GB
            'cpu_usage_percent': 80.0,     # 80%
            'cache_hit_rate': 0.3,         # 30%
            'error_rate': 0.05,            # 5%
            'patterns_per_second': 1000.0  # 1000 patterns/sec
        }
        
        # Performance tracking
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_history = deque(maxlen=100)
        self.current_metrics = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # 5 seconds
        
        # Performance counters
        self.total_patterns_detected = 0
        self.total_processing_time = 0.0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("📊 Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("⏹️ Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self._collect_pattern_detection_metrics()
                self._check_alerts()
                self._save_metrics()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric('cpu_usage_percent', cpu_percent, '%')
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self._add_metric('memory_usage_mb', memory_mb, 'MB')
            self._add_metric('memory_usage_percent', memory.percent, '%')
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._add_metric('disk_usage_percent', disk_percent, '%')
            
            # Network I/O
            network = psutil.net_io_counters()
            self._add_metric('network_bytes_sent', network.bytes_sent, 'bytes')
            self._add_metric('network_bytes_recv', network.bytes_recv, 'bytes')
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_pattern_detection_metrics(self):
        """Collect pattern detection specific metrics"""
        try:
            # Calculate derived metrics
            if self.total_patterns_detected > 0:
                avg_processing_time = self.total_processing_time / self.total_patterns_detected
                self._add_metric('avg_processing_time_ms', avg_processing_time, 'ms')
                
                patterns_per_second = self.total_patterns_detected / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0
                self._add_metric('patterns_per_second', patterns_per_second, 'patterns/sec')
            
            # Cache metrics
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                cache_hit_rate = self.cache_hits / total_cache_requests
                self._add_metric('cache_hit_rate', cache_hit_rate, 'ratio')
            
            # Error rate
            total_operations = self.total_patterns_detected + self.total_errors
            if total_operations > 0:
                error_rate = self.total_errors / total_operations
                self._add_metric('error_rate', error_rate, 'ratio')
            
        except Exception as e:
            logger.error(f"Error collecting pattern detection metrics: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str, context: Dict[str, Any] = None):
        """Add a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            context=context or {}
        )
        
        self.metrics_history[name].append(metric)
        self.current_metrics[name] = metric
        
        # Store in database if available
        self._store_metric_in_db(metric)
    
    def _store_metric_in_db(self, metric: PerformanceMetric):
        """Store metric in database for historical analysis"""
        try:
            # This would integrate with your TimescaleDB for time-series storage
            # For now, we'll just log it
            logger.debug(f"Metric: {metric.metric_name} = {metric.value} {metric.unit}")
            
        except Exception as e:
            logger.error(f"Error storing metric in database: {e}")
    
    def _check_alerts(self):
        """Check for performance alerts"""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in self.current_metrics:
                current_value = self.current_metrics[metric_name].value
                
                if current_value > threshold:
                    alert_type = 'critical' if current_value > threshold * 1.5 else 'warning'
                    message = f"{metric_name} exceeded threshold: {current_value:.2f} > {threshold:.2f}"
                    recommendation = self._get_recommendation(metric_name, current_value, threshold)
                    
                    alert = PerformanceAlert(
                        timestamp=datetime.now(),
                        alert_type=alert_type,
                        message=message,
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=threshold,
                        recommendation=recommendation
                    )
                    
                    self.alerts_history.append(alert)
                    logger.warning(f"🚨 Performance Alert: {message}")
    
    def _get_recommendation(self, metric_name: str, current_value: float, threshold: float) -> str:
        """Get optimization recommendation based on metric"""
        recommendations = {
            'processing_time_ms': "Consider increasing max_workers or optimizing pattern detection algorithms",
            'memory_usage_mb': "Consider reducing buffer sizes or implementing memory cleanup",
            'cpu_usage_percent': "Consider distributing load across multiple processes",
            'cache_hit_rate': "Consider adjusting cache TTL or increasing cache size",
            'error_rate': "Investigate pattern detection errors and improve error handling",
            'patterns_per_second': "Consider optimizing vectorized operations or using GPU acceleration"
        }
        
        return recommendations.get(metric_name, "Monitor and investigate performance degradation")
    
    def _save_metrics(self):
        """Save metrics to file for analysis"""
        try:
            metrics_file = f"logs/performance_metrics_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs('logs', exist_ok=True)
            
            # Save current metrics snapshot
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {name: asdict(metric) for name, metric in self.current_metrics.items()},
                'alerts': [asdict(alert) for alert in list(self.alerts_history)[-10:]]  # Last 10 alerts
            }
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def record_pattern_detection(self, patterns_count: int, processing_time_ms: float, 
                               cache_hit: bool = False, error: bool = False):
        """Record pattern detection performance"""
        self.total_patterns_detected += patterns_count
        self.total_processing_time += processing_time_ms
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if error:
            self.total_errors += 1
        
        # Add specific metrics
        self._add_metric('patterns_detected', patterns_count, 'count')
        self._add_metric('processing_time_ms', processing_time_ms, 'ms')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'current_metrics': {name: asdict(metric) for name, metric in self.current_metrics.items()},
            'total_patterns_detected': self.total_patterns_detected,
            'total_processing_time_ms': self.total_processing_time,
            'avg_processing_time_ms': self.total_processing_time / max(self.total_patterns_detected, 1),
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'error_rate': self.total_errors / max(self.total_patterns_detected + self.total_errors, 1),
            'recent_alerts': [asdict(alert) for alert in list(self.alerts_history)[-5:]],
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check cache performance
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        if cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate - consider increasing cache size or TTL")
        
        # Check processing time
        avg_processing_time = self.total_processing_time / max(self.total_patterns_detected, 1)
        if avg_processing_time > 500:
            recommendations.append("High processing time - consider optimizing algorithms or increasing workers")
        
        # Check error rate
        error_rate = self.total_errors / max(self.total_patterns_detected + self.total_errors, 1)
        if error_rate > 0.01:
            recommendations.append("High error rate - investigate and improve error handling")
        
        # Check memory usage
        if 'memory_usage_mb' in self.current_metrics:
            memory_usage = self.current_metrics['memory_usage_mb'].value
            if memory_usage > 1024:  # 1GB
                recommendations.append("High memory usage - consider implementing memory cleanup")
        
        return recommendations
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get historical metrics for analysis"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [metric for metric in self.metrics_history[metric_name] 
                if metric.timestamp > cutoff_time]
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file"""
        if not filename:
            filename = f"performance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'metrics_history': {
                name: [asdict(metric) for metric in metrics]
                for name, metrics in self.metrics_history.items()
            },
            'alerts_history': [asdict(alert) for alert in self.alerts_history]
        }
        
        os.makedirs('exports', exist_ok=True)
        filepath = f"exports/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"📊 Performance metrics exported to {filepath}")
        return filepath

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor
