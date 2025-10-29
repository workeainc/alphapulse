#!/usr/bin/env python3
"""
Concept Drift Detection System
Phase 2: Concept Drift Detection

Implements:
1. Rolling AUC/F1 score drift detection (> 10% over 7 days)
2. Calibration error drift monitoring beyond tolerance
3. Model performance degradation alerts
4. Integration with existing drift detection system
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class ConceptDriftMetrics:
    """Metrics for concept drift detection"""
    timestamp: datetime
    drift_type: str  # 'auc_f1', 'calibration'
    metric_name: str  # 'auc', 'f1', 'calibration_error'
    current_value: float
    reference_value: float
    drift_score: float
    severity: str    # 'low', 'medium', 'high', 'critical'
    confidence: float
    details: Dict[str, Any]

@dataclass
class ConceptDriftAlert:
    """Alert for detected concept drift"""
    alert_id: str
    drift_type: str
    timestamp: datetime
    drift_score: float
    severity: str
    message: str
    recommendations: List[str]
    metadata: Dict[str, Any]

class ConceptDriftDetector:
    """Advanced concept drift detection system for model performance metrics"""
    
    def __init__(self, 
                 rolling_window_days: int = 7,
                 auc_f1_threshold: float = 0.1,  # 10% drop threshold
                 calibration_tolerance: float = 0.05,  # 5% tolerance
                 min_data_points: int = 50):
        self.rolling_window_days = rolling_window_days
        self.auc_f1_threshold = auc_f1_threshold
        self.calibration_tolerance = calibration_tolerance
        self.min_data_points = min_data_points
        
        # Performance history storage
        self.performance_history = {
            'auc': [],
            'f1': [],
            'calibration_error': []
        }
        
        # Drift detection results
        self.drift_history = []
        self.alerts = []
        
        # Reference baseline (established after sufficient data)
        self.reference_baseline = {}
        self.baseline_established = False
        
        logger.info("🚀 Concept Drift Detector initialized")
    
    def add_performance_metrics(self, 
                               timestamp: datetime,
                               auc_score: Optional[float] = None,
                               f1_score: Optional[float] = None,
                               calibration_error: Optional[float] = None,
                               y_true: Optional[np.ndarray] = None,
                               y_pred_proba: Optional[np.ndarray] = None):
        """Add new performance metrics for drift detection"""
        try:
            # Store timestamp
            if 'auc' in self.performance_history and self.performance_history['auc']:
                last_timestamp = self.performance_history['auc'][-1]['timestamp']
                if timestamp <= last_timestamp:
                    logger.warning(f"⚠️ Timestamp {timestamp} is not newer than last recorded {last_timestamp}")
                    return False
            
            # Calculate calibration error if raw predictions provided
            if y_true is not None and y_pred_proba is not None:
                try:
                    calibration_error = self._calculate_calibration_error(y_true, y_pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to calculate calibration error: {e}")
            
            # Store metrics
            if auc_score is not None:
                self.performance_history['auc'].append({
                    'timestamp': timestamp,
                    'value': auc_score
                })
            
            if f1_score is not None:
                self.performance_history['f1'].append({
                    'timestamp': timestamp,
                    'value': f1_score
                })
            
            if calibration_error is not None:
                self.performance_history['calibration_error'].append({
                    'timestamp': timestamp,
                    'value': calibration_error
                })
            
            # Clean old data
            self._cleanup_old_data()
            
            # Establish baseline if we have enough data
            if not self.baseline_established:
                self._establish_baseline()
            
            # Check for drift if baseline is established
            if self.baseline_established:
                self._check_for_drift()
            
            logger.debug(f"✅ Added performance metrics for {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add performance metrics: {e}")
            return False
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate calibration error using Brier score"""
        try:
            # Brier score for binary classification
            if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]  # Take positive class probabilities
            
            # Calculate Brier score (lower is better)
            brier_score = brier_score_loss(y_true, y_pred_proba)
            
            # Convert to calibration error (0 = perfectly calibrated, 1 = worst)
            # Typical good calibration error: < 0.05
            return brier_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate calibration error: {e}")
            return 0.0
    
    def _establish_baseline(self):
        """Establish reference baseline from initial data"""
        try:
            # Check if we have enough data points
            min_points = max(self.min_data_points, self.rolling_window_days)
            
            for metric_name, history in self.performance_history.items():
                if len(history) < min_points:
                    return  # Not enough data yet
            
            # Calculate baseline for each metric
            for metric_name, history in self.performance_history.items():
                if history:
                    # Use first 30% of data for baseline (or first rolling window)
                    baseline_size = min(len(history) // 3, self.rolling_window_days)
                    baseline_data = history[:baseline_size]
                    
                    baseline_value = np.mean([point['value'] for point in baseline_data])
                    baseline_std = np.std([point['value'] for point in baseline_data])
                    
                    self.reference_baseline[metric_name] = {
                        'value': baseline_value,
                        'std': baseline_std,
                        'established_at': datetime.now(),
                        'data_points': len(baseline_data)
                    }
            
            self.baseline_established = True
            logger.info(f"✅ Baseline established with {len(self.reference_baseline)} metrics")
            
        except Exception as e:
            logger.error(f"❌ Failed to establish baseline: {e}")
    
    def _check_for_drift(self):
        """Check for concept drift in all metrics"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.rolling_window_days)
            
            for metric_name, history in self.performance_history.items():
                if metric_name not in self.reference_baseline:
                    continue
                
                # Get recent data within rolling window
                recent_data = [
                    point for point in history
                    if point['timestamp'] >= cutoff_time
                ]
                
                if len(recent_data) < 3:  # Need at least 3 points for trend
                    continue
                
                # Calculate current performance
                current_values = [point['value'] for point in recent_data]
                current_performance = np.mean(current_values)
                
                # Get reference baseline
                reference = self.reference_baseline[metric_name]
                reference_value = reference['value']
                
                # Detect drift based on metric type
                if metric_name in ['auc', 'f1']:
                    drift_metrics = self._detect_auc_f1_drift(
                        metric_name, current_performance, reference_value, recent_data
                    )
                elif metric_name == 'calibration_error':
                    drift_metrics = self._detect_calibration_drift(
                        current_performance, reference_value, recent_data
                    )
                else:
                    continue
                
                if drift_metrics:
                    self.drift_history.append(drift_metrics)
                    
                    # Generate alert if drift is significant
                    if drift_metrics.severity in ['high', 'critical']:
                        self._generate_concept_drift_alert(drift_metrics)
            
        except Exception as e:
            logger.error(f"❌ Error checking for concept drift: {e}")
    
    def _detect_auc_f1_drift(self, metric_name: str, current_value: float, 
                             reference_value: float, recent_data: List[Dict]) -> Optional[ConceptDriftMetrics]:
        """Detect AUC/F1 score drift"""
        try:
            # Calculate percentage change
            if reference_value == 0:
                return None
            
            percentage_change = (current_value - reference_value) / reference_value
            
            # Check if drop exceeds threshold (10% for AUC/F1)
            if abs(percentage_change) < self.auc_f1_threshold:
                return None
            
            # Calculate drift score (0-1, higher = more drift)
            drift_score = min(1.0, abs(percentage_change) / self.auc_f1_threshold)
            
            # Determine severity
            if abs(percentage_change) >= 0.25:  # 25% drop
                severity = 'critical'
            elif abs(percentage_change) >= 0.15:  # 15% drop
                severity = 'high'
            elif abs(percentage_change) >= 0.10:  # 10% drop
                severity = 'medium'
            else:
                severity = 'low'
            
            # Calculate confidence based on data consistency
            recent_values = [point['value'] for point in recent_data]
            confidence = 1.0 - (np.std(recent_values) / (np.mean(recent_values) + 1e-8))
            confidence = max(0.1, min(1.0, confidence))
            
            drift_metrics = ConceptDriftMetrics(
                timestamp=datetime.now(),
                drift_type='auc_f1',
                metric_name=metric_name,
                current_value=current_value,
                reference_value=reference_value,
                drift_score=drift_score,
                severity=severity,
                confidence=confidence,
                details={
                    'percentage_change': percentage_change,
                    'rolling_window_days': self.rolling_window_days,
                    'data_points': len(recent_data),
                    'recent_values': recent_values
                }
            )
            
            logger.info(f"🚨 AUC/F1 drift detected for {metric_name}: {percentage_change:.2%} change, severity: {severity}")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to detect AUC/F1 drift: {e}")
            return None
    
    def _detect_calibration_drift(self, current_value: float, reference_value: float,
                                 recent_data: List[Dict]) -> Optional[ConceptDriftMetrics]:
        """Detect calibration error drift"""
        try:
            # For calibration error, higher values indicate worse calibration
            # Check if calibration error increased beyond tolerance
            error_increase = current_value - reference_value
            
            if error_increase <= self.calibration_tolerance:
                return None
            
            # Calculate drift score
            drift_score = min(1.0, error_increase / (self.calibration_tolerance * 2))
            
            # Determine severity
            if error_increase >= self.calibration_tolerance * 3:  # 3x tolerance
                severity = 'critical'
            elif error_increase >= self.calibration_tolerance * 2:  # 2x tolerance
                severity = 'high'
            elif error_increase >= self.calibration_tolerance:  # 1x tolerance
                severity = 'medium'
            else:
                severity = 'low'
            
            # Calculate confidence
            recent_values = [point['value'] for point in recent_data]
            confidence = 1.0 - (np.std(recent_values) / (np.mean(recent_values) + 1e-8))
            confidence = max(0.1, min(1.0, confidence))
            
            drift_metrics = ConceptDriftMetrics(
                timestamp=datetime.now(),
                drift_type='calibration',
                metric_name='calibration_error',
                current_value=current_value,
                reference_value=reference_value,
                drift_score=drift_score,
                severity=severity,
                confidence=confidence,
                details={
                    'error_increase': error_increase,
                    'tolerance': self.calibration_tolerance,
                    'rolling_window_days': self.rolling_window_days,
                    'data_points': len(recent_data),
                    'recent_values': recent_values
                }
            )
            
            logger.info(f"🚨 Calibration drift detected: {error_increase:.4f} increase, severity: {severity}")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to detect calibration drift: {e}")
            return None
    
    def _generate_concept_drift_alert(self, drift_metrics: ConceptDriftMetrics):
        """Generate alert for concept drift"""
        try:
            alert_id = f"concept_drift_{drift_metrics.drift_type}_{drift_metrics.metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate message
            if drift_metrics.drift_type == 'auc_f1':
                change_pct = drift_metrics.details['percentage_change']
                message = f"Concept drift detected in {drift_metrics.metric_name.upper()}: {change_pct:.2%} change over {self.rolling_window_days} days"
            else:
                error_increase = drift_metrics.details['error_increase']
                message = f"Calibration drift detected: {error_increase:.4f} increase beyond tolerance"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(drift_metrics)
            
            alert = ConceptDriftAlert(
                alert_id=alert_id,
                drift_type=drift_metrics.drift_type,
                timestamp=drift_metrics.timestamp,
                drift_score=drift_metrics.drift_score,
                severity=drift_metrics.severity,
                message=message,
                recommendations=recommendations,
                metadata={
                    'metric_name': drift_metrics.metric_name,
                    'current_value': drift_metrics.current_value,
                    'reference_value': drift_metrics.reference_value,
                    'confidence': drift_metrics.confidence
                }
            )
            
            self.alerts.append(alert)
            logger.warning(f"🚨 Concept drift alert generated: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate concept drift alert: {e}")
    
    def _generate_recommendations(self, drift_metrics: ConceptDriftMetrics) -> List[str]:
        """Generate recommendations based on drift type and severity"""
        recommendations = []
        
        if drift_metrics.drift_type == 'auc_f1':
            recommendations.append("Review feature engineering pipeline")
            recommendations.append("Check for data quality issues")
            recommendations.append("Consider model retraining")
            if drift_metrics.severity in ['high', 'critical']:
                recommendations.append("Trigger urgent model retraining")
        
        elif drift_metrics.drift_type == 'calibration':
            recommendations.append("Review model calibration")
            recommendations.append("Check prediction confidence scores")
            recommendations.append("Consider recalibration or retraining")
            if drift_metrics.severity in ['high', 'critical']:
                recommendations.append("Trigger urgent model retraining")
        
        recommendations.append("Monitor drift metrics over time")
        recommendations.append("Update reference baseline if drift is expected")
        
        return recommendations
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.rolling_window_days * 2)
            
            for metric_name in self.performance_history:
                self.performance_history[metric_name] = [
                    point for point in self.performance_history[metric_name]
                    if point['timestamp'] >= cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
    
    def get_drift_summary(self, start_time: datetime = None, 
                          end_time: datetime = None) -> Dict[str, Any]:
        """Get summary of concept drift detection results"""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.now()
            
            # Filter drift history by time
            filtered_drifts = [
                drift for drift in self.drift_history
                if start_time <= drift.timestamp <= end_time
            ]
            
            if not filtered_drifts:
                return {
                    'total_drifts': 0,
                    'baseline_established': self.baseline_established,
                    'period': {'start': start_time, 'end': end_time}
                }
            
            # Calculate summary statistics
            drift_scores = [drift.drift_score for drift in filtered_drifts]
            severities = [drift.severity for drift in filtered_drifts]
            drift_types = [drift.drift_type for drift in filtered_drifts]
            
            summary = {
                'total_drifts': len(filtered_drifts),
                'baseline_established': self.baseline_established,
                'period': {'start': start_time, 'end': end_time},
                'drift_statistics': {
                    'mean_score': np.mean(drift_scores),
                    'max_score': np.max(drift_scores),
                    'min_score': np.min(drift_scores),
                    'std_score': np.std(drift_scores)
                },
                'severity_distribution': {
                    severity: severities.count(severity) 
                    for severity in set(severities)
                },
                'drift_type_distribution': {
                    drift_type: drift_types.count(drift_type) 
                    for drift_type in set(drift_types)
                },
                'recent_alerts': len([a for a in self.alerts if start_time <= a.timestamp <= end_time])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get drift summary: {e}")
            return {'error': str(e)}
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of concept drift detection"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.rolling_window_days)
            
            # Get recent performance for each metric
            current_status = {}
            for metric_name, history in self.performance_history.items():
                if not history:
                    current_status[metric_name] = {'status': 'no_data', 'value': None}
                    continue
                
                recent_data = [
                    point for point in history
                    if point['timestamp'] >= cutoff_time
                ]
                
                if not recent_data:
                    current_status[metric_name] = {'status': 'no_recent_data', 'value': None}
                    continue
                
                current_value = np.mean([point['value'] for point in recent_data])
                reference_value = self.reference_baseline.get(metric_name, {}).get('value')
                
                if reference_value is None:
                    current_status[metric_name] = {'status': 'no_baseline', 'value': current_value}
                else:
                    # Check if drift detected
                    if metric_name in ['auc', 'f1']:
                        percentage_change = (current_value - reference_value) / reference_value
                        drift_detected = abs(percentage_change) >= self.auc_f1_threshold
                    else:  # calibration_error
                        error_increase = current_value - reference_value
                        drift_detected = error_increase >= self.calibration_tolerance
                    
                    current_status[metric_name] = {
                        'status': 'drift_detected' if drift_detected else 'stable',
                        'value': current_value,
                        'reference_value': reference_value,
                        'drift_detected': drift_detected
                    }
            
            return {
                'baseline_established': self.baseline_established,
                'rolling_window_days': self.rolling_window_days,
                'current_status': current_status,
                'total_drifts': len(self.drift_history),
                'total_alerts': len(self.alerts)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get current status: {e}")
            return {'error': str(e)}

# Convenience functions
def detect_concept_drift(performance_data: List[Dict], 
                        rolling_window_days: int = 7) -> Optional[ConceptDriftMetrics]:
    """Detect concept drift using default settings"""
    detector = ConceptDriftDetector(rolling_window_days=rolling_window_days)
    
    # Add performance data
    for data_point in performance_data:
        detector.add_performance_metrics(
            timestamp=data_point['timestamp'],
            auc_score=data_point.get('auc'),
            f1_score=data_point.get('f1'),
            calibration_error=data_point.get('calibration_error')
        )
    
    # Return latest drift if any
    return detector.drift_history[-1] if detector.drift_history else None

# Global detector instance
concept_drift_detector = ConceptDriftDetector()
