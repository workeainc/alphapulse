#!/usr/bin/env python3
"""
Real-Time Performance Dashboard for AlphaPulse Trading Bot
Provides live monitoring of pattern storage pipeline performance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
from dataclasses import dataclass, asdict

from src.app.services.pattern_integration_service import PatternIntegrationService

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    value: float
    metric_type: str
    unit: str
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """Performance alert"""
    timestamp: datetime
    severity: str  # "info", "warning", "error", "critical"
    message: str
    metric: str
    current_value: float
    threshold: float
    recommendation: str

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self):
        self.pattern_service = PatternIntegrationService()
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics storage
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {
            "patterns_per_second": [],
            "memory_usage_mb": [],
            "storage_efficiency": [],
            "parallel_efficiency": [],
            "active_workers": [],
            "batch_size": [],
            "storage_time_ms": []
        }
        
        # Alerts storage
        self.alerts: List[Alert] = []
        
        # Dashboard configuration
        self.dashboard_config = {
            "metrics_retention_hours": 24,  # Keep metrics for 24 hours
            "update_interval_seconds": 5,   # Update dashboard every 5 seconds
            "max_metrics_per_type": 1000,   # Maximum metrics to store per type
            "alert_retention_hours": 48,    # Keep alerts for 48 hours
            "enable_real_time_updates": True,
            "enable_performance_alerts": True,
            "enable_optimization_suggestions": True
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "patterns_per_second": {
                "warning": 100,
                "critical": 50
            },
            "memory_usage_mb": {
                "warning": 500,
                "critical": 800
            },
            "storage_efficiency": {
                "warning": 80,
                "critical": 60
            },
            "parallel_efficiency": {
                "warning": 30,
                "critical": 10
            }
        }
        
        # Dashboard state
        self.dashboard_state = {
            "last_update": None,
            "is_monitoring": False,
            "total_updates": 0,
            "last_alert_check": None,
            "performance_score": 100
        }
    
    async def initialize(self):
        """Initialize the performance dashboard"""
        if self._initialized:
            return
        
        try:
            # Initialize pattern service
            await self.pattern_service.initialize()
            
            # Start monitoring if enabled
            if self.dashboard_config["enable_real_time_updates"]:
                asyncio.create_task(self._start_monitoring())
            
            self._initialized = True
            self.logger.info("✅ Performance Dashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Performance Dashboard: {e}")
            raise
    
    async def _start_monitoring(self):
        """Start real-time performance monitoring"""
        self.dashboard_state["is_monitoring"] = True
        self.logger.info("🚀 Starting real-time performance monitoring")
        
        while self.dashboard_state["is_monitoring"]:
            try:
                await self._update_dashboard()
                await asyncio.sleep(self.dashboard_config["update_interval_seconds"])
            except Exception as e:
                self.logger.error(f"❌ Error in performance monitoring: {e}")
                await asyncio.sleep(10)  # Wait 10 seconds before retrying
    
    async def _update_dashboard(self):
        """Update dashboard with latest performance data"""
        try:
            # Get latest performance report
            performance_report = await self.pattern_service.get_production_performance_report()
            
            if "error" in performance_report:
                self.logger.warning(f"⚠️ Could not get performance report: {performance_report['error']}")
                return
            
            # Extract metrics
            await self._extract_and_store_metrics(performance_report)
            
            # Check for alerts
            if self.dashboard_config["enable_performance_alerts"]:
                await self._check_performance_alerts(performance_report)
            
            # Update dashboard state
            self.dashboard_state["last_update"] = datetime.now()
            self.dashboard_state["total_updates"] += 1
            
            # Clean up old metrics
            await self._cleanup_old_metrics()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating dashboard: {e}")
    
    async def _extract_and_store_metrics(self, performance_report: Dict[str, Any]):
        """Extract and store performance metrics from report"""
        try:
            timestamp = datetime.now()
            
            # Extract key metrics
            metrics_to_store = [
                ("patterns_per_second", performance_report.get("performance_metrics", {}).get("current_throughput", 0), "patterns/s"),
                ("memory_usage_mb", performance_report.get("real_time_metrics", {}).get("memory_usage_mb", 0), "MB"),
                ("storage_efficiency", performance_report.get("performance_metrics", {}).get("storage_efficiency_percentage", 0), "%"),
                ("parallel_efficiency", performance_report.get("performance_metrics", {}).get("parallel_efficiency_percentage", 0), "%"),
                ("active_workers", performance_report.get("real_time_metrics", {}).get("active_workers", 0), "workers"),
                ("batch_size", performance_report.get("real_time_metrics", {}).get("current_batch_size", 0), "patterns"),
                ("storage_time_ms", performance_report.get("storage_performance", {}).get("avg_storage_time_ms", 0), "ms")
            ]
            
            for metric_name, value, unit in metrics_to_store:
                metric = PerformanceMetric(
                    timestamp=timestamp,
                    value=value,
                    metric_type=metric_name,
                    unit=unit
                )
                
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].append(metric)
                    
                    # Limit metrics per type
                    if len(self.metrics_history[metric_name]) > self.dashboard_config["max_metrics_per_type"]:
                        self.metrics_history[metric_name] = self.metrics_history[metric_name][-self.dashboard_config["max_metrics_per_type"]:]
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting metrics: {e}")
    
    async def _check_performance_alerts(self, performance_report: Dict[str, Any]):
        """Check for performance alerts based on thresholds"""
        try:
            current_time = datetime.now()
            
            # Check if we should run alert checks
            if (self.dashboard_state["last_alert_check"] and 
                (current_time - self.dashboard_state["last_alert_check"]).seconds < 60):
                return  # Check alerts at most once per minute
            
            self.dashboard_state["last_alert_check"] = current_time
            
            # Get current metrics
            performance_metrics = performance_report.get("performance_metrics", {})
            real_time_metrics = performance_report.get("real_time_metrics", {})
            
            # Check patterns per second
            current_throughput = performance_metrics.get("current_throughput", 0)
            if current_throughput < self.performance_thresholds["patterns_per_second"]["critical"]:
                await self._create_alert("critical", "patterns_per_second", current_throughput, 
                                       self.performance_thresholds["patterns_per_second"]["critical"],
                                       "Critical: Pattern throughput is critically low")
            elif current_throughput < self.performance_thresholds["patterns_per_second"]["warning"]:
                await self._create_alert("warning", "patterns_per_second", current_throughput,
                                       self.performance_thresholds["patterns_per_second"]["warning"],
                                       "Warning: Pattern throughput is below optimal levels")
            
            # Check memory usage
            memory_usage = real_time_metrics.get("memory_usage_mb", 0)
            if memory_usage > self.performance_thresholds["memory_usage_mb"]["critical"]:
                await self._create_alert("critical", "memory_usage_mb", memory_usage,
                                       self.performance_thresholds["memory_usage_mb"]["critical"],
                                       "Critical: Memory usage is critically high")
            elif memory_usage > self.performance_thresholds["memory_usage_mb"]["warning"]:
                await self._create_alert("warning", "memory_usage_mb", memory_usage,
                                       self.performance_thresholds["memory_usage_mb"]["warning"],
                                       "Warning: Memory usage is above optimal levels")
            
            # Check storage efficiency
            storage_efficiency = performance_metrics.get("storage_efficiency_percentage", 100)
            if storage_efficiency < self.performance_thresholds["storage_efficiency"]["critical"]:
                await self._create_alert("critical", "storage_efficiency", storage_efficiency,
                                       self.performance_thresholds["storage_efficiency"]["critical"],
                                       "Critical: Storage efficiency is critically low")
            elif storage_efficiency < self.performance_thresholds["storage_efficiency"]["warning"]:
                await self._create_alert("warning", "storage_efficiency", storage_efficiency,
                                       self.performance_thresholds["storage_efficiency"]["warning"],
                                       "Warning: Storage efficiency is below optimal levels")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking performance alerts: {e}")
    
    async def _create_alert(self, severity: str, metric: str, current_value: float, 
                           threshold: float, message: str):
        """Create a new performance alert"""
        try:
            # Generate recommendation based on alert type
            recommendation = self._generate_alert_recommendation(severity, metric, current_value, threshold)
            
            alert = Alert(
                timestamp=datetime.now(),
                severity=severity,
                message=message,
                metric=metric,
                current_value=current_value,
                threshold=threshold,
                recommendation=recommendation
            )
            
            self.alerts.append(alert)
            
            # Log alert
            log_message = f"🚨 {severity.upper()} ALERT: {message} (Current: {current_value}, Threshold: {threshold})"
            if severity == "critical":
                self.logger.critical(log_message)
            elif severity == "warning":
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            # Clean up old alerts
            await self._cleanup_old_alerts()
            
        except Exception as e:
            self.logger.error(f"❌ Error creating alert: {e}")
    
    def _generate_alert_recommendation(self, severity: str, metric: str, current_value: float, threshold: float) -> str:
        """Generate recommendation based on alert type"""
        recommendations = {
            "patterns_per_second": {
                "low": "Check for bottlenecks in pattern detection pipeline, consider increasing batch sizes or enabling parallel processing",
                "critical": "Immediate action required: Check system resources, database connections, and processing pipeline"
            },
            "memory_usage_mb": {
                "high": "Consider reducing batch sizes, optimizing memory usage, or increasing system memory",
                "critical": "Immediate action required: Reduce batch sizes, check for memory leaks, consider restarting service"
            },
            "storage_efficiency": {
                "low": "Check database performance, optimize batch sizes, consider using COPY operations for large batches",
                "critical": "Immediate action required: Check database health, optimize storage pipeline, verify data integrity"
            }
        }
        
        metric_recs = recommendations.get(metric, {})
        if current_value < threshold:
            return metric_recs.get("low", "Review system configuration and performance")
        else:
            return metric_recs.get("high", "Review system configuration and performance")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.dashboard_config["metrics_retention_hours"])
            
            for metric_type in self.metrics_history:
                self.metrics_history[metric_type] = [
                    metric for metric in self.metrics_history[metric_type]
                    if metric.timestamp > cutoff_time
                ]
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old metrics: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts based on retention policy"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.dashboard_config["alert_retention_hours"])
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old alerts: {e}")
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get current dashboard summary"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Calculate current performance score
            performance_score = self._calculate_performance_score()
            
            # Get latest metrics
            latest_metrics = {}
            for metric_type, metrics in self.metrics_history.items():
                if metrics:
                    latest_metrics[metric_type] = {
                        "current_value": metrics[-1].value,
                        "unit": metrics[-1].unit,
                        "timestamp": metrics[-1].timestamp.isoformat()
                    }
            
            # Get recent alerts
            recent_alerts = [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "message": alert.message,
                    "recommendation": alert.recommendation
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
            
            return {
                "dashboard_status": "active" if self.dashboard_state["is_monitoring"] else "inactive",
                "last_update": self.dashboard_state["last_update"].isoformat() if self.dashboard_state["last_update"] else None,
                "total_updates": self.dashboard_state["total_updates"],
                "performance_score": performance_score,
                "current_metrics": latest_metrics,
                "recent_alerts": recent_alerts,
                "alert_count": len(self.alerts),
                "metrics_count": sum(len(metrics) for metrics in self.metrics_history.values())
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting dashboard summary: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_score(self) -> int:
        """Calculate overall performance score (0-100)"""
        try:
            score = 100
            
            # Deduct points for alerts
            critical_alerts = len([a for a in self.alerts if a.severity == "critical"])
            warning_alerts = len([a for a in self.alerts if a.severity == "warning"])
            
            score -= critical_alerts * 20  # Critical alerts cost 20 points each
            score -= warning_alerts * 5    # Warning alerts cost 5 points each
            
            # Deduct points for low performance metrics
            for metric_type, metrics in self.metrics_history.items():
                if metrics and len(metrics) > 0:
                    latest_value = metrics[-1].value
                    
                    if metric_type == "patterns_per_second" and latest_value < 100:
                        score -= 10
                    elif metric_type == "memory_usage_mb" and latest_value > 500:
                        score -= 15
                    elif metric_type == "storage_efficiency" and latest_value < 80:
                        score -= 10
            
            return max(0, score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance score: {e}")
            return 50  # Return neutral score on error
    
    async def get_metrics_history(self, metric_type: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics for a specific type"""
        try:
            if metric_type not in self.metrics_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = [
                metric for metric in self.metrics_history[metric_type]
                if metric.timestamp > cutoff_time
            ]
            
            return [
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value,
                    "unit": metric.unit
                }
                for metric in metrics
            ]
            
        except Exception as e:
            self.logger.error(f"❌ Error getting metrics history: {e}")
            return []
    
    async def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends and analysis"""
        try:
            trends = {}
            
            for metric_type, metrics in self.metrics_history.items():
                if len(metrics) < 2:
                    continue
                
                # Calculate trend (positive/negative/stable)
                recent_values = [m.value for m in metrics[-10:]]  # Last 10 values
                if len(recent_values) >= 2:
                    first_half = recent_values[:len(recent_values)//2]
                    second_half = recent_values[len(recent_values)//2:]
                    
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    
                    if second_avg > first_avg * 1.1:
                        trend = "improving"
                    elif second_avg < first_avg * 0.9:
                        trend = "declining"
                    else:
                        trend = "stable"
                    
                    trends[metric_type] = {
                        "trend": trend,
                        "change_percentage": ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0,
                        "current_value": recent_values[-1] if recent_values else 0
                    }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"❌ Error getting performance trends: {e}")
            return {}
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.dashboard_state["is_monitoring"] = False
        self.logger.info("🛑 Performance monitoring stopped")
    
    async def close(self):
        """Close the performance dashboard"""
        try:
            await self.stop_monitoring()
            self._initialized = False
            self.logger.info("🔌 Performance Dashboard closed")
        except Exception as e:
            self.logger.error(f"❌ Error closing Performance Dashboard: {e}")

# Global instance for easy access
performance_dashboard = PerformanceDashboard()
