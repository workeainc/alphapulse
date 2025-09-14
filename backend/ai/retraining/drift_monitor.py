#!/usr/bin/env python3
"""
Drift Detection Monitor for AlphaPulse
Phase 5: Consolidated Retraining System

Implements:
1. Unified drift detection across all systems
2. Feature drift detection using PSI
3. Concept drift detection using AUC/F1 and calibration
4. Latency drift detection using p95 thresholds
5. Comprehensive drift analytics and reporting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import time
import numpy as np

# Local imports
from ..feature_drift_detector import FeatureDriftDetector
from ..concept_drift_detector import ConceptDriftDetector
from ..production_monitoring import production_monitoring

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionSummary:
    """Comprehensive summary of all drift detection systems"""
    timestamp: datetime
    feature_drift: Dict[str, Any]
    concept_drift: Dict[str, Any]
    latency_drift: Dict[str, Any]
    overall_status: str  # 'healthy', 'warning', 'critical'
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class DriftAlert:
    """Unified drift alert across all systems"""
    alert_id: str
    timestamp: datetime
    alert_type: str  # 'feature', 'concept', 'latency', 'combined'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    drift_score: float
    source_system: str
    recommendations: List[str]
    metadata: Dict[str, Any]

class DriftDetectionMonitor:
    """
    Unified monitor for all drift detection systems with Phase 4D advanced detection
    Provides comprehensive monitoring, alerting, and status reporting
    """
    
    def __init__(self):
        self.is_running = False
        
        # Initialize all drift detection systems
        self.feature_drift_detector = FeatureDriftDetector()
        self.concept_drift_detector = ConceptDriftDetector()
        
        # Phase 4D: Advanced drift detection systems
        self.adwin_detector = None  # ADWIN for gradual concept drift
        self.page_hinkley_detector = None  # Page-Hinkley for sudden drift
        self.kl_divergence_detector = None  # KL-divergence for distribution shift
        self.calibration_drift_detector = None  # Calibration drift (Brier/ECE)
        
        # Monitoring state
        self.drift_history = []
        self.alert_history = []
        self.last_comprehensive_check = None
        
        # Phase 4D: Enhanced configuration
        self.monitoring_config = {
            'comprehensive_check_interval': 300,  # 5 minutes
            'alert_retention_hours': 168,  # 1 week
            'drift_history_retention_hours': 720,  # 1 month
            'thresholds': {
                'combined_drift_critical': 0.8,  # 80% combined drift score
                'combined_drift_warning': 0.6,   # 60% combined drift score
                'alert_cooldown_minutes': 30,    # Prevent alert spam
                # Phase 4D: Advanced thresholds
                'calibration_ece_threshold': 0.03,  # ECE increase > 0.03 triggers alert
                'rolling_winrate_drop_threshold': 0.10,  # 10% drop vs 30-day baseline
                'population_shift_pvalue_threshold': 0.01,  # p-value < 0.01 for distribution shift
                'adwin_delta_threshold': 0.05,  # ADWIN delta threshold
                'page_hinkley_threshold': 0.1,  # Page-Hinkley threshold
                'kl_divergence_threshold': 0.5,  # KL-divergence threshold
            }
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_checks': 0,
            'drifts_detected': 0,
            'alerts_generated': 0,
            'retrains_triggered': 0,
            'avg_check_duration': 0.0,
            # Phase 4D: Advanced metrics
            'adwin_detections': 0,
            'page_hinkley_detections': 0,
            'kl_divergence_detections': 0,
            'calibration_drift_detections': 0,
            'false_positive_rate': 0.0,
            'detection_latency_ms': 0.0
        }
        
        # Alert cooldown tracking
        self.last_alert_time = {}
        
        # Phase 4D: Advanced detection state
        self.adwin_window = []
        self.page_hinkley_statistic = 0.0
        self.kl_divergence_history = []
        self.calibration_history = []
        self.baseline_distributions = {}
        
        logger.info("🚀 Drift Detection Monitor initialized with Phase 4D advanced detection")
    
    async def start(self):
        """Start the drift detection monitor"""
        if self.is_running:
            logger.warning("Drift Detection Monitor is already running")
            return
        
        try:
            self.is_running = True
            asyncio.create_task(self._run_monitoring_loop())
            logger.info("✅ Drift Detection Monitor started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Drift Detection Monitor: {e}")
            raise
    
    async def stop(self):
        """Stop the drift detection monitor"""
        if not self.is_running:
            logger.warning("Drift Detection Monitor is not running")
            return
        
        try:
            self.is_running = False
            logger.info("✅ Drift Detection Monitor stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping Drift Detection Monitor: {e}")
            raise
    
    async def _run_monitoring_loop(self):
        """Main monitoring loop for comprehensive drift detection"""
        logger.info("🔄 Starting drift monitoring loop...")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Perform comprehensive drift check
                await self._perform_comprehensive_drift_check()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Calculate execution time
                execution_time = time.time() - start_time
                self.performance_metrics['avg_check_duration'] = (
                    (self.performance_metrics['avg_check_duration'] * 
                     self.performance_metrics['total_checks'] + execution_time) /
                    (self.performance_metrics['total_checks'] + 1)
                )
                self.performance_metrics['total_checks'] += 1
                
                # Wait for next check
                await asyncio.sleep(self.monitoring_config['comprehensive_check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in drift monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _perform_comprehensive_drift_check(self):
        """Perform comprehensive drift check across all systems"""
        try:
            # Check feature drift
            feature_status = await self.check_feature_drift()
            
            # Check concept drift
            concept_status = await self.check_concept_drift()
            
            # Check latency drift
            latency_status = await self.check_latency_drift()
            
            # Generate comprehensive summary
            summary = await self._generate_comprehensive_summary(
                feature_status, concept_status, latency_status
            )
            
            # Check combined drift conditions
            await self._check_combined_drift_conditions(summary)
            
            # Log summary
            logger.info(f"🔍 Drift check completed - Overall status: {summary['overall_status']}")
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive drift check: {e}")
    
    async def check_feature_drift(self) -> Dict[str, Any]:
        """Check for feature drift across all monitored features"""
        try:
            # Get drift summary from feature drift detector
            drift_summary = await self.feature_drift_detector.get_drift_summary()
            
            if not drift_summary:
                return {
                    'status': 'healthy',
                    'message': 'No feature drift detected',
                    'drift_score': 0.0,
                    'features_checked': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate overall feature drift status
            total_features = len(drift_summary)
            high_drift_features = 0
            critical_drift_features = 0
            max_drift_score = 0.0
            
            for feature_name, drift_info in drift_summary.items():
                drift_score = drift_info.get('drift_score', 0)
                max_drift_score = max(max_drift_score, drift_score)
                
                if drift_score > 0.6:
                    high_drift_features += 1
                if drift_score > 0.8:
                    critical_drift_features += 1
            
            # Determine overall status
            if critical_drift_features > 0:
                status = 'critical'
                message = f'Critical drift in {critical_drift_features} features'
            elif high_drift_features > 0:
                status = 'warning'
                message = f'High drift in {high_drift_features} features'
            else:
                status = 'healthy'
                message = 'No significant feature drift'
            
            return {
                'status': status,
                'message': message,
                'drift_score': max_drift_score,
                'features_checked': total_features,
                'high_drift_features': high_drift_features,
                'critical_drift_features': critical_drift_features,
                'feature_details': drift_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking feature drift: {e}")
            return {
                'status': 'error',
                'message': f'Error checking feature drift: {str(e)}',
                'drift_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_concept_drift(self) -> Dict[str, Any]:
        """Check for concept drift in model performance"""
        try:
            # Get drift summary from concept drift detector
            drift_summary = await self.concept_drift_detector.get_drift_summary()
            
            if not drift_summary:
                return {
                    'status': 'healthy',
                    'message': 'No concept drift detected',
                    'drift_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Extract key metrics
            auc_f1_drift = drift_summary.get('auc_f1_drift', 0)
            calibration_drift = drift_summary.get('calibration_drift', 0)
            
            # Calculate overall concept drift score
            overall_drift_score = max(auc_f1_drift, calibration_drift)
            
            # Determine status
            if overall_drift_score > 0.25:
                status = 'critical'
                message = 'Critical concept drift detected'
            elif overall_drift_score > 0.15:
                status = 'warning'
                message = 'Warning-level concept drift detected'
            else:
                status = 'healthy'
                message = 'No significant concept drift'
            
            return {
                'status': status,
                'message': message,
                'drift_score': overall_drift_score,
                'auc_f1_drift': auc_f1_drift,
                'calibration_drift': calibration_drift,
                'drift_details': drift_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking concept drift: {e}")
            return {
                'status': 'error',
                'message': f'Error checking concept drift: {str(e)}',
                'drift_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_latency_drift(self) -> Dict[str, Any]:
        """Check for latency drift in inference performance"""
        try:
            # Get latency summary from production monitoring
            latency_summary = production_monitoring.get_latency_summary('inference', hours=1)
            
            if not latency_summary or 'error' in latency_summary:
                return {
                    'status': 'unknown',
                    'message': 'Unable to retrieve latency data',
                    'drift_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Extract p95 latency
            p95_latency = latency_summary.get('statistics', {}).get('p95', 0)
            
            # Check against thresholds
            if p95_latency > 200:  # 200ms threshold
                status = 'critical'
                message = f'Critical latency drift: p95={p95_latency:.2f}ms'
                drift_score = min(1.0, p95_latency / 500)  # Normalize to 0-1
            elif p95_latency > 100:  # 100ms threshold
                status = 'warning'
                message = f'Warning latency drift: p95={p95_latency:.2f}ms'
                drift_score = min(1.0, p95_latency / 200)
            else:
                status = 'healthy'
                message = f'Latency within normal range: p95={p95_latency:.2f}ms'
                drift_score = 0.0
            
            return {
                'status': status,
                'message': message,
                'drift_score': drift_score,
                'p95_latency': p95_latency,
                'latency_details': latency_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking latency drift: {e}")
            return {
                'status': 'error',
                'message': f'Error checking latency drift: {str(e)}',
                'drift_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _generate_comprehensive_summary(self, feature_status: Dict, 
                                           concept_status: Dict, 
                                           latency_status: Dict) -> Dict[str, Any]:
        """Generate comprehensive drift summary"""
        try:
            # Calculate overall status
            overall_status = 'healthy'
            recommendations = []
            
            # Check feature drift
            if feature_status.get('status') == 'critical':
                overall_status = 'critical'
                recommendations.append("Immediate action required - consider emergency retraining")
            elif feature_status.get('status') == 'warning':
                if overall_status == 'healthy':
                    overall_status = 'warning'
                recommendations.append("Monitor closely and prepare for potential retraining")
            
            # Check concept drift
            if concept_status.get('status') == 'critical':
                overall_status = 'critical'
                recommendations.append("Model performance degradation detected - urgent retraining needed")
            elif concept_status.get('status') == 'warning':
                if overall_status == 'healthy':
                    overall_status = 'warning'
                recommendations.append("Performance monitoring required")
            
            # Check latency drift
            if latency_status.get('status') == 'critical':
                overall_status = 'critical'
                recommendations.append("Inference latency critical - model size/complexity review required")
            elif latency_status.get('status') == 'warning':
                if overall_status == 'healthy':
                    overall_status = 'warning'
                recommendations.append("Latency monitoring required")
            
            # Create summary
            summary = DriftDetectionSummary(
                timestamp=datetime.now(),
                feature_drift=feature_status,
                concept_drift=concept_status,
                latency_drift=latency_status,
                overall_status=overall_status,
                recommendations=recommendations,
                metadata={
                    'check_duration': time.time(),
                    'systems_checked': ['feature', 'concept', 'latency']
                }
            )
            
            # Store in history
            self.drift_history.append(summary)
            
            return {
                'timestamp': summary.timestamp.isoformat(),
                'feature_drift': summary.feature_drift,
                'concept_drift': summary.concept_drift,
                'latency_drift': summary.latency_drift,
                'overall_status': summary.overall_status,
                'recommendations': summary.recommendations,
                'metadata': summary.metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive summary: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _check_combined_drift_conditions(self, summary: Dict[str, Any]):
        """Check combined drift conditions and generate alerts"""
        try:
            if summary['overall_status'] == 'critical':
                await self._generate_critical_alert(summary)
            elif summary['overall_status'] == 'warning':
                await self._generate_warning_alert(summary)
                
        except Exception as e:
            logger.error(f"❌ Error checking combined drift conditions: {e}")
    
    async def _generate_critical_alert(self, summary: Dict[str, Any]):
        """Generate critical drift alert"""
        try:
            alert_id = f"critical_{int(time.time())}"
            
            # Check cooldown
            if not self._should_generate_alert('critical', alert_id):
                return
            
            # Create alert
            alert = DriftAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                alert_type='combined',
                severity='critical',
                message=f"CRITICAL: Combined drift detected - {summary['overall_status']}",
                drift_score=1.0,
                source_system='drift_monitor',
                recommendations=summary['recommendations'],
                metadata=summary
            )
            
            # Store alert
            self.alert_history.append(alert)
            self.last_alert_time['critical'] = datetime.now()
            
            # Log critical alert
            logger.critical(f"🚨 CRITICAL ALERT: {alert.message}")
            logger.critical(f"🚨 Recommendations: {alert.recommendations}")
            
            # Update metrics
            self.performance_metrics['alerts_generated'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error generating critical alert: {e}")
    
    async def _generate_warning_alert(self, summary: Dict[str, Any]):
        """Generate warning drift alert"""
        try:
            alert_id = f"warning_{int(time.time())}"
            
            # Check cooldown
            if not self._should_generate_alert('warning', alert_id):
                return
            
            # Create alert
            alert = DriftAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                alert_type='combined',
                severity='warning',
                message=f"WARNING: Drift detected - {summary['overall_status']}",
                drift_score=0.6,
                source_system='drift_monitor',
                recommendations=summary['recommendations'],
                metadata=summary
            )
            
            # Store alert
            self.alert_history.append(alert)
            self.last_alert_time['warning'] = datetime.now()
            
            # Log warning alert
            logger.warning(f"⚠️ WARNING ALERT: {alert.message}")
            logger.warning(f"⚠️ Recommendations: {alert.recommendations}")
            
            # Update metrics
            self.performance_metrics['alerts_generated'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error generating warning alert: {e}")
    
    def _should_generate_alert(self, alert_type: str, alert_id: str) -> bool:
        """Check if alert should be generated (cooldown check)"""
        try:
            cooldown_minutes = self.monitoring_config['thresholds']['alert_cooldown_minutes']
            
            if alert_type not in self.last_alert_time:
                return True
            
            time_since_last = datetime.now() - self.last_alert_time[alert_type]
            return time_since_last.total_seconds() > (cooldown_minutes * 60)
            
        except Exception as e:
            logger.error(f"❌ Error checking alert cooldown: {e}")
            return True
    
    async def _update_performance_metrics(self):
        """Update performance metrics for monitoring"""
        try:
            # Count total drifts detected
            total_drifts = sum([
                1 for summary in self.drift_history[-100:]  # Last 100 checks
                if summary.overall_status in ['warning', 'critical']
            ])
            
            self.performance_metrics['drifts_detected'] = total_drifts
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old drift detection and alert data"""
        try:
            # Clean up old drift history
            cutoff_time = datetime.now() - timedelta(
                hours=self.monitoring_config['drift_history_retention_hours']
            )
            
            # Clean up old alerts
            alert_cutoff = datetime.now() - timedelta(
                hours=self.monitoring_config['alert_retention_hours']
            )
            
            # Remove old drift summaries
            self.drift_history = [
                summary for summary in self.drift_history
                if summary.timestamp >= cutoff_time
            ]
            
            # Remove old alerts
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp >= alert_cutoff
            ]
            
            logger.debug("🧹 Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error in cleanup: {e}")
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get comprehensive orchestration summary"""
        try:
            return {
                'status': 'running' if self.is_running else 'stopped',
                'performance_metrics': self.performance_metrics.copy(),
                'drift_history_count': len(self.drift_history),
                'alert_history_count': len(self.alert_history),
                'last_check': self.last_comprehensive_check.isoformat() if self.last_comprehensive_check else None,
                'monitoring_config': self.monitoring_config.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting orchestration summary: {e}")
            return {'error': str(e)}
    
    def get_current_status(self) -> str:
        """Get current monitor status"""
        if not self.is_running:
            return "stopped"
        
        # Check recent drift history
        if self.drift_history:
            recent_summaries = self.drift_history[-10:]  # Last 10 checks
            critical_count = sum(1 for s in recent_summaries if s.overall_status == 'critical')
            warning_count = sum(1 for s in recent_summaries if s.overall_status == 'warning')
            
            if critical_count > 0:
                return "critical_drift_detected"
            elif warning_count > 0:
                return "warning_drift_detected"
        
        return "healthy"

# Global instance
drift_detection_monitor = DriftDetectionMonitor()

# Phase 4D: Advanced Drift Detection Methods

class ADWINDetector:
    """ADWIN (ADaptive WINdowing) detector for gradual concept drift"""
    
    def __init__(self, delta: float = 0.05, min_window_size: int = 10):
        self.delta = delta
        self.min_window_size = min_window_size
        self.window = []
        self.total_mean = 0.0
        self.total_variance = 0.0
        
    def add_element(self, value: float) -> bool:
        """Add element and check for drift"""
        self.window.append(value)
        
        if len(self.window) < self.min_window_size:
            return False
        
        # Update statistics
        self._update_statistics()
        
        # Check for drift using ADWIN algorithm
        return self._check_drift()
    
    def _update_statistics(self):
        """Update running statistics"""
        n = len(self.window)
        self.total_mean = sum(self.window) / n
        self.total_variance = sum((x - self.total_mean) ** 2 for x in self.window) / (n - 1) if n > 1 else 0
    
    def _check_drift(self) -> bool:
        """Check for drift using ADWIN algorithm"""
        try:
            n = len(self.window)
            
            # Try different window splits
            for i in range(1, n):
                left_window = self.window[:i]
                right_window = self.window[i:]
                
                if len(left_window) < self.min_window_size or len(right_window) < self.min_window_size:
                    continue
                
                left_mean = sum(left_window) / len(left_window)
                right_mean = sum(right_window) / len(right_window)
                
                # Calculate confidence interval
                confidence = self._calculate_confidence(len(left_window), len(right_window))
                
                # Check if means are significantly different
                if abs(left_mean - right_mean) > confidence:
                    # Drift detected - remove old elements
                    self.window = self.window[i:]
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in ADWIN drift check: {e}")
            return False
    
    def _calculate_confidence(self, n1: int, n2: int) -> float:
        """Calculate confidence interval for ADWIN"""
        try:
            # Simplified confidence calculation
            # In practice, you'd use a more sophisticated approach
            return 2 * ((1 / n1 + 1 / n2) ** 0.5) * (self.total_variance ** 0.5) * (np.log(2 / self.delta) ** 0.5)
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.1

class PageHinkleyDetector:
    """Page-Hinkley test for sudden concept drift detection"""
    
    def __init__(self, delta: float = 0.05, alpha: float = 0.005):
        self.delta = delta
        self.alpha = alpha
        self.mean = 0.0
        self.variance = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = float('inf')
        self.drift_detected = False
        
    def add_element(self, value: float) -> bool:
        """Add element and check for drift"""
        try:
            # Update statistics
            self._update_statistics(value)
            
            # Calculate Page-Hinkley statistic
            self.cumulative_sum += (value - self.mean - self.alpha)
            self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
            
            # Check for drift
            threshold = self.delta * np.log(len(self.window)) if hasattr(self, 'window') else self.delta
            
            if self.cumulative_sum - self.min_cumulative_sum > threshold:
                self.drift_detected = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in Page-Hinkley drift check: {e}")
            return False
    
    def _update_statistics(self, value: float):
        """Update running statistics"""
        try:
            if not hasattr(self, 'window'):
                self.window = []
            
            self.window.append(value)
            n = len(self.window)
            
            # Update mean
            self.mean = sum(self.window) / n
            
            # Update variance
            if n > 1:
                self.variance = sum((x - self.mean) ** 2 for x in self.window) / (n - 1)
            else:
                self.variance = 0
                
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

class KLDivergenceDetector:
    """KL-divergence detector for distribution shift detection"""
    
    def __init__(self, num_bins: int = 10, threshold: float = 0.5):
        self.num_bins = num_bins
        self.threshold = threshold
        self.baseline_distribution = None
        self.current_distribution = None
        
    def set_baseline(self, data: List[float]):
        """Set baseline distribution"""
        try:
            self.baseline_distribution = self._calculate_histogram(data)
        except Exception as e:
            logger.error(f"Error setting baseline: {e}")
    
    def check_drift(self, data: List[float]) -> Tuple[bool, float]:
        """Check for distribution drift"""
        try:
            if self.baseline_distribution is None:
                return False, 0.0
            
            self.current_distribution = self._calculate_histogram(data)
            kl_divergence = self._calculate_kl_divergence(
                self.baseline_distribution, 
                self.current_distribution
            )
            
            return kl_divergence > self.threshold, kl_divergence
            
        except Exception as e:
            logger.error(f"Error checking KL-divergence drift: {e}")
            return False, 0.0
    
    def _calculate_histogram(self, data: List[float]) -> List[float]:
        """Calculate histogram of data"""
        try:
            if not data:
                return [0.0] * self.num_bins
            
            min_val, max_val = min(data), max(data)
            bin_width = (max_val - min_val) / self.num_bins if max_val > min_val else 1.0
            
            histogram = [0.0] * self.num_bins
            
            for value in data:
                bin_index = min(int((value - min_val) / bin_width), self.num_bins - 1)
                histogram[bin_index] += 1.0
            
            # Normalize
            total = sum(histogram)
            if total > 0:
                histogram = [h / total for h in histogram]
            
            return histogram
            
        except Exception as e:
            logger.error(f"Error calculating histogram: {e}")
            return [0.0] * self.num_bins
    
    def _calculate_kl_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate KL-divergence between distributions"""
        try:
            kl_div = 0.0
            
            for i in range(len(p)):
                if p[i] > 0 and q[i] > 0:
                    kl_div += p[i] * np.log(p[i] / q[i])
                elif p[i] > 0 and q[i] == 0:
                    kl_div += float('inf')
            
            return kl_div
            
        except Exception as e:
            logger.error(f"Error calculating KL-divergence: {e}")
            return 0.0

class CalibrationDriftDetector:
    """Calibration drift detector using Brier score and ECE"""
    
    def __init__(self, num_bins: int = 10, threshold: float = 0.03):
        self.num_bins = num_bins
        self.threshold = threshold
        self.baseline_ece = None
        self.baseline_brier = None
        
    def set_baseline(self, predictions: List[float], labels: List[int]):
        """Set baseline calibration metrics"""
        try:
            self.baseline_ece = self._calculate_ece(predictions, labels)
            self.baseline_brier = self._calculate_brier_score(predictions, labels)
        except Exception as e:
            logger.error(f"Error setting calibration baseline: {e}")
    
    def check_drift(self, predictions: List[float], labels: List[int]) -> Tuple[bool, Dict[str, float]]:
        """Check for calibration drift"""
        try:
            if self.baseline_ece is None:
                return False, {}
            
            current_ece = self._calculate_ece(predictions, labels)
            current_brier = self._calculate_brier_score(predictions, labels)
            
            ece_increase = current_ece - self.baseline_ece
            brier_change = current_brier - self.baseline_brier
            
            drift_detected = ece_increase > self.threshold
            
            return drift_detected, {
                'ece_increase': ece_increase,
                'brier_change': brier_change,
                'current_ece': current_ece,
                'current_brier': current_brier,
                'baseline_ece': self.baseline_ece,
                'baseline_brier': self.baseline_brier
            }
            
        except Exception as e:
            logger.error(f"Error checking calibration drift: {e}")
            return False, {}
    
    def _calculate_ece(self, predictions: List[float], labels: List[int]) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        try:
            if len(predictions) != len(labels):
                return 0.0
            
            # Create bins
            bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = np.logical_and(predictions > bin_lower, predictions <= bin_upper)
                bin_size = np.sum(in_bin)
                
                if bin_size > 0:
                    bin_accuracy = np.mean(labels[in_bin])
                    bin_confidence = np.mean(predictions[in_bin])
                    ece += bin_size * abs(bin_accuracy - bin_confidence)
            
            return ece / len(predictions)
            
        except Exception as e:
            logger.error(f"Error calculating ECE: {e}")
            return 0.0
    
    def _calculate_brier_score(self, predictions: List[float], labels: List[int]) -> float:
        """Calculate Brier score"""
        try:
            if len(predictions) != len(labels):
                return 0.0
            
            brier_score = np.mean((np.array(predictions) - np.array(labels)) ** 2)
            return brier_score
            
        except Exception as e:
            logger.error(f"Error calculating Brier score: {e}")
            return 0.0
