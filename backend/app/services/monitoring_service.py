#!/usr/bin/env python3
"""
Enhanced Monitoring Service for AlphaPulse
Phase 1 - Model Monitoring + Drift Detection Implementation

Exposes Prometheus metrics for trading performance, model drift, latency, and system health.
Enhanced with live-vs-backtest performance tracking and advanced drift detection.
"""

import asyncio
import logging
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import json
import os
from sqlalchemy import create_engine, text

# Prometheus metrics
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, generate_latest, 
        CONTENT_TYPE_LATEST, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

# Database imports
try:
    from ..database.connection_simple import get_async_session
    from ..database.data_versioning_dao import DataVersioningDAO
    # Test database connection
    engine = create_engine("postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    DATABASE_AVAILABLE = True
except Exception as e:
    DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

if DATABASE_AVAILABLE:
    logger.info("Database connection available")
else:
    logger.warning("Database components not available")


@dataclass
class SystemMetrics:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    database_health: float
    active_connections: int


@dataclass
class DriftDetectionResult:
    """Drift detection result"""
    timestamp: datetime
    model_id: str
    drift_type: str  # 'data_drift', 'concept_drift', 'performance_drift'
    severity: float  # 0-100
    threshold: float
    is_drift_detected: bool
    features_affected: List[str]
    details: Dict[str, Any]


@dataclass
class LivePerformanceMetrics:
    """Live performance metrics"""
    timestamp: datetime
    model_id: str
    precision: float
    recall: float
    f1_score: float
    hit_rate: float
    sharpe_ratio: float
    profit_factor: float
    total_signals: int
    win_rate: float
    avg_confidence: float


@dataclass
class BacktestComparison:
    """Live vs backtest comparison"""
    timestamp: datetime
    model_id: str
    live_precision: float
    backtest_precision: float
    precision_delta: float
    live_sharpe: float
    backtest_sharpe: float
    sharpe_delta: float
    performance_degradation: float  # 0-100
    alert_threshold: float
    is_alert_triggered: bool


@dataclass
class MonitoringAlert:
    """Monitoring alert for closed-loop integration"""
    alert_id: str
    model_id: str
    alert_type: str  # 'drift', 'performance', 'risk', 'data_quality'
    severity_level: str  # 'low', 'medium', 'high', 'critical'
    trigger_condition: Dict[str, Any]
    current_value: float
    threshold_value: float
    is_triggered: bool
    triggered_at: Optional[datetime]
    alert_metadata: Dict[str, Any]


@dataclass
class ClosedLoopAction:
    """Closed-loop action configuration"""
    action_id: str
    alert_id: str
    model_id: str
    action_type: str  # 'trigger_retraining', 'deploy_shadow', 'rollback', 'alert'
    action_status: str  # 'pending', 'executing', 'completed', 'failed'
    trigger_source: str
    action_config: Dict[str, Any]
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    action_metadata: Dict[str, Any] = None


class LivePerformanceTracker:
    """Live performance tracking with backtest comparison"""
    
    def __init__(self, alert_threshold: float = 0.1):
        self.alert_threshold = alert_threshold
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.backtest_baselines = {}
        self.alerts = deque(maxlen=100)
        
    async def update_live_performance(self, model_id: str, metrics: LivePerformanceMetrics):
        """Update live performance metrics"""
        self.performance_history[model_id].append(metrics)
        
        # Compare with backtest baseline
        if model_id in self.backtest_baselines:
            comparison = await self._compare_with_backtest(model_id, metrics)
            if comparison.is_alert_triggered:
                self.alerts.append(comparison)
                logger.warning(f"🚨 Performance degradation detected for {model_id}: {comparison.performance_degradation:.2f}%")
    
    async def _compare_with_backtest(self, model_id: str, live_metrics: LivePerformanceMetrics) -> BacktestComparison:
        """Compare live performance with backtest baseline"""
        baseline = self.backtest_baselines.get(model_id, {})
        
        precision_delta = live_metrics.precision - baseline.get('precision', live_metrics.precision)
        sharpe_delta = live_metrics.sharpe_ratio - baseline.get('sharpe_ratio', live_metrics.sharpe_ratio)
        
        # Calculate performance degradation
        precision_degradation = max(0, -precision_delta / baseline.get('precision', 1.0)) * 100
        sharpe_degradation = max(0, -sharpe_delta / baseline.get('sharpe_ratio', 1.0)) * 100
        performance_degradation = (precision_degradation + sharpe_degradation) / 2
        
        is_alert_triggered = performance_degradation > (self.alert_threshold * 100)
        
        return BacktestComparison(
            timestamp=live_metrics.timestamp,
            model_id=model_id,
            live_precision=live_metrics.precision,
            backtest_precision=baseline.get('precision', live_metrics.precision),
            precision_delta=precision_delta,
            live_sharpe=live_metrics.sharpe_ratio,
            backtest_sharpe=baseline.get('sharpe_ratio', live_metrics.sharpe_ratio),
            sharpe_delta=sharpe_delta,
            performance_degradation=performance_degradation,
            alert_threshold=self.alert_threshold * 100,
            is_alert_triggered=is_alert_triggered
        )
    
    def set_backtest_baseline(self, model_id: str, baseline_metrics: Dict[str, float]):
        """Set backtest baseline for comparison"""
        self.backtest_baselines[model_id] = baseline_metrics
        logger.info(f"📊 Set backtest baseline for {model_id}: {baseline_metrics}")
    
    def get_performance_alerts(self) -> List[BacktestComparison]:
        """Get recent performance alerts"""
        return list(self.alerts)


class DriftDetector:
    """Advanced drift detection using multiple methods"""
    
    def __init__(self, psi_threshold: float = 0.1, kl_threshold: float = 0.1):
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.feature_distributions = defaultdict(lambda: deque(maxlen=1000))
        self.drift_history = deque(maxlen=100)
        
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins
            min_val = min(np.min(expected), np.min(actual))
            max_val = max(np.max(expected), np.max(actual))
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate histograms
            expected_hist, _ = np.histogram(expected, bins=bin_edges)
            actual_hist, _ = np.histogram(actual, bins=bin_edges)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            expected_hist = expected_hist.astype(float) + epsilon
            actual_hist = actual_hist.astype(float) + epsilon
            
            # Normalize
            expected_pct = expected_hist / np.sum(expected_hist)
            actual_pct = actual_hist / np.sum(actual_hist)
            
            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            
            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between distributions"""
        try:
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            p = p.astype(float) + epsilon
            q = q.astype(float) + epsilon
            
            # Normalize
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # Calculate KL divergence
            kl_div = np.sum(p * np.log(p / q))
            
            return float(kl_div)
        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def detect_data_drift(self, model_id: str, feature_name: str, new_values: np.ndarray) -> DriftDetectionResult:
        """Detect data drift for a specific feature"""
        if len(self.feature_distributions[f"{model_id}_{feature_name}"]) < 100:
            # Not enough historical data yet
            self.feature_distributions[f"{model_id}_{feature_name}"].extend(new_values)
            return DriftDetectionResult(
                timestamp=datetime.now(),
                model_id=model_id,
                drift_type='data_drift',
                severity=0.0,
                threshold=self.psi_threshold,
                is_drift_detected=False,
                features_affected=[feature_name],
                details={'reason': 'insufficient_historical_data'}
            )
        
        # Get historical distribution
        historical_values = np.array(list(self.feature_distributions[f"{model_id}_{feature_name}"]))
        
        # Calculate PSI
        psi_score = self.calculate_psi(historical_values, new_values)
        
        # Determine severity (0-100)
        severity = min(100.0, (psi_score / self.psi_threshold) * 50)
        
        # Check if drift detected
        is_drift_detected = psi_score > self.psi_threshold
        
        # Update historical distribution
        self.feature_distributions[f"{model_id}_{feature_name}"].extend(new_values)
        
        result = DriftDetectionResult(
            timestamp=datetime.now(),
            model_id=model_id,
            drift_type='data_drift',
            severity=severity,
            threshold=self.psi_threshold,
            is_drift_detected=is_drift_detected,
            features_affected=[feature_name],
            details={
                'psi_score': psi_score,
                'historical_mean': float(np.mean(historical_values)),
                'new_mean': float(np.mean(new_values)),
                'historical_std': float(np.std(historical_values)),
                'new_std': float(np.std(new_values))
            }
        )
        
        self.drift_history.append(result)
        return result
    
    def detect_concept_drift(self, model_id: str, predictions: np.ndarray, actuals: np.ndarray) -> DriftDetectionResult:
        """Detect concept drift based on prediction accuracy"""
        try:
            # Calculate prediction accuracy
            accuracy = np.mean(predictions == actuals)
            
            # Simple concept drift detection based on accuracy drop
            # In practice, you might use more sophisticated methods
            baseline_accuracy = 0.7  # This should come from historical data
            accuracy_drop = baseline_accuracy - accuracy
            
            # Calculate severity
            severity = min(100.0, max(0.0, accuracy_drop * 100))
            
            # Check if drift detected
            is_drift_detected = accuracy_drop > 0.1  # 10% accuracy drop threshold
            
            result = DriftDetectionResult(
                timestamp=datetime.now(),
                model_id=model_id,
                drift_type='concept_drift',
                severity=severity,
                threshold=0.1,
                is_drift_detected=is_drift_detected,
                features_affected=['prediction_accuracy'],
                details={
                    'current_accuracy': accuracy,
                    'baseline_accuracy': baseline_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'predictions_count': len(predictions)
                }
            )
            
            self.drift_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}")
            return DriftDetectionResult(
                timestamp=datetime.now(),
                model_id=model_id,
                drift_type='concept_drift',
                severity=0.0,
                threshold=0.1,
                is_drift_detected=False,
                features_affected=[],
                details={'error': str(e)}
            )
    
    def get_drift_alerts(self) -> List[DriftDetectionResult]:
        """Get recent drift alerts"""
        return [drift for drift in self.drift_history if drift.is_drift_detected]


class ModelInterpretabilityTracker:
    """Track model interpretability and explanations"""
    
    def __init__(self):
        self.attention_weights = defaultdict(lambda: deque(maxlen=100))
        self.feature_importance = defaultdict(lambda: deque(maxlen=100))
        self.explanation_history = deque(maxlen=1000)
        
    def record_lstm_attention(self, model_id: str, attention_weights: np.ndarray, timesteps: List[str]):
        """Record LSTM attention weights"""
        attention_data = {
            'timestamp': datetime.now(),
            'weights': attention_weights.tolist(),
            'timesteps': timesteps,
            'max_attention_step': timesteps[np.argmax(attention_weights)],
            'attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-10)))
        }
        self.attention_weights[model_id].append(attention_data)
        
    def record_transformer_attention(self, model_id: str, attention_weights: np.ndarray, timeframes: List[str]):
        """Record Transformer attention weights"""
        attention_data = {
            'timestamp': datetime.now(),
            'weights': attention_weights.tolist(),
            'timeframes': timeframes,
            'max_attention_timeframe': timeframes[np.argmax(attention_weights)],
            'attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-10)))
        }
        self.attention_weights[model_id].append(attention_data)
        
    def record_feature_importance(self, model_id: str, feature_names: List[str], importance_scores: np.ndarray):
        """Record feature importance scores"""
        importance_data = {
            'timestamp': datetime.now(),
            'features': feature_names,
            'scores': importance_scores.tolist(),
            'top_features': [feature_names[i] for i in np.argsort(importance_scores)[-5:]],  # Top 5 features
            'importance_entropy': float(-np.sum(importance_scores * np.log(importance_scores + 1e-10)))
        }
        self.feature_importance[model_id].append(importance_data)
        
    def record_explanation(self, model_id: str, prediction: float, explanation: Dict[str, Any]):
        """Record model explanation"""
        explanation_data = {
            'timestamp': datetime.now(),
            'model_id': model_id,
            'prediction': prediction,
            'explanation': explanation
        }
        self.explanation_history.append(explanation_data)
        
    def get_recent_attention(self, model_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent attention weights for a model"""
        return list(self.attention_weights[model_id])[-limit:]
        
    def get_recent_feature_importance(self, model_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent feature importance for a model"""
        return list(self.feature_importance[model_id])[-limit:]
        
    def get_recent_explanations(self, model_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent explanations"""
        explanations = list(self.explanation_history)
        if model_id:
            explanations = [exp for exp in explanations if exp['model_id'] == model_id]
        return explanations[-limit:]


class MonitoringService:
    """
    Enhanced Monitoring Service for AlphaPulse with advanced drift detection and performance tracking
    """
    
    def __init__(self, registry=None):
        """Initialize the monitoring service with Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Prometheus client not available")
            return
        
        # Use provided registry or default
        self.registry = registry or REGISTRY
        
        # Trading Performance Metrics
        self.win_rate = Gauge('alphapulse_win_rate', 'Trading win rate', registry=self.registry)
        self.precision = Gauge('alphapulse_precision', 'Trading precision', registry=self.registry)
        self.profit_factor = Gauge('alphapulse_profit_factor', 'Trading profit factor', registry=self.registry)
        self.avg_rr = Gauge('alphapulse_avg_rr', 'Average risk/reward ratio', registry=self.registry)
        
        # Model Performance Metrics
        self.model_accuracy = Gauge('alphapulse_model_accuracy', 'Model accuracy by type', ['model_type'], registry=self.registry)
        self.model_training_failures = Counter('alphapulse_model_training_failures_total', 'Model training failures', registry=self.registry)
        
        # Enhanced Drift Detection Metrics
        self.psi_drift_score = Gauge('alphapulse_psi_drift_score', 'PSI drift score', ['model_id'], registry=self.registry)
        self.kl_drift_score = Gauge('alphapulse_kl_drift_score', 'KL divergence drift score', ['model_id'], registry=self.registry)
        self.concept_drift_score = Gauge('alphapulse_concept_drift_score', 'Concept drift score', ['model_id'], registry=self.registry)
        self.performance_degradation = Gauge('alphapulse_performance_degradation', 'Performance degradation percentage', ['model_id'], registry=self.registry)
        
        # Live Performance Tracking
        self.live_precision = Gauge('alphapulse_live_precision', 'Live precision', ['model_id'], registry=self.registry)
        self.live_recall = Gauge('alphapulse_live_recall', 'Live recall', ['model_id'], registry=self.registry)
        self.live_f1_score = Gauge('alphapulse_live_f1_score', 'Live F1 score', ['model_id'], registry=self.registry)
        self.live_sharpe_ratio = Gauge('alphapulse_live_sharpe_ratio', 'Live Sharpe ratio', ['model_id'], registry=self.registry)
        self.live_hit_rate = Gauge('alphapulse_live_hit_rate', 'Live hit rate', ['model_id'], registry=self.registry)
        
        # Interpretability Metrics
        self.attention_entropy = Gauge('alphapulse_attention_entropy', 'Attention entropy', ['model_id'], registry=self.registry)
        self.feature_importance_entropy = Gauge('alphapulse_feature_importance_entropy', 'Feature importance entropy', ['model_id'], registry=self.registry)
        self.explanation_confidence = Gauge('alphapulse_explanation_confidence', 'Explanation confidence', ['model_id'], registry=self.registry)
        
        # Latency Metrics
        self.inference_duration = Histogram(
            'alphapulse_inference_duration_seconds',
            'Model inference duration',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Signal Metrics
        self.signals_generated = Counter('alphapulse_signals_generated_total', 'Total signals generated', ['symbol', 'model_id'], registry=self.registry)
        self.signals_executed = Counter('alphapulse_signals_executed_total', 'Total signals executed', ['symbol', 'model_id'], registry=self.registry)
        self.signals_rejected = Counter('alphapulse_signals_rejected_total', 'Total signals rejected', ['symbol', 'model_id'], registry=self.registry)
        
        # Active Learning Metrics
        self.active_learning_pending_items = Gauge('alphapulse_active_learning_pending_items', 'Active learning pending items', registry=self.registry)
        self.active_learning_labeled_items = Gauge('alphapulse_active_learning_labeled_items', 'Active learning labeled items', registry=self.registry)
        self.active_learning_processed_items_total = Counter('alphapulse_active_learning_processed_items_total', 'Total active learning processed items', registry=self.registry)
        
        # Portfolio Metrics
        self.portfolio_total_value = Gauge('alphapulse_portfolio_total_value', 'Total portfolio value', registry=self.registry)
        self.portfolio_pnl = Gauge('alphapulse_portfolio_pnl', 'Portfolio P&L', registry=self.registry)
        self.portfolio_drawdown = Gauge('alphapulse_portfolio_drawdown', 'Portfolio drawdown', registry=self.registry)
        
        # System Health Metrics
        self.system_cpu_usage = Gauge('alphapulse_system_cpu_usage', 'System CPU usage percentage', registry=self.registry)
        self.system_memory_usage = Gauge('alphapulse_system_memory_usage', 'System memory usage percentage', registry=self.registry)
        self.system_disk_usage = Gauge('alphapulse_system_disk_usage', 'System disk usage percentage', registry=self.registry)
        self.database_connection_health = Gauge('alphapulse_database_connection_health', 'Database connection health', registry=self.registry)
        
        # Initialize enhanced components
        self.live_performance_tracker = LivePerformanceTracker()
        self.drift_detector = DriftDetector()
        self.interpretability_tracker = ModelInterpretabilityTracker()
        
        # Service state
        self.is_running = False
        self.last_update = None
        
        logger.info("Enhanced monitoring service initialized with drift detection and performance tracking")
    
    async def start(self):
        """Start the monitoring service"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Cannot start monitoring service: Prometheus client not available")
            return
        
        if self.is_running:
            logger.warning("Monitoring service is already running")
            return
        
        logger.info("🚀 Starting Enhanced Monitoring Service...")
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._update_metrics_loop())
        
        logger.info("✅ Enhanced Monitoring Service started successfully")
    
    async def stop(self):
        """Stop the monitoring service"""
        if not self.is_running:
            logger.warning("Monitoring service is not running")
            return
        
        logger.info("🛑 Stopping Enhanced Monitoring Service...")
        self.is_running = False
        logger.info("✅ Enhanced Monitoring Service stopped successfully")
    
    async def _update_metrics_loop(self):
        """Background task to update metrics periodically"""
        while self.is_running:
            try:
                await self._update_all_metrics()
                self.last_update = datetime.now()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"❌ Error updating metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_all_metrics(self):
        """Update all metrics"""
        await asyncio.gather(
            self._update_trading_metrics(),
            self._update_model_metrics(),
            self._update_enhanced_drift_metrics(),
            self._update_live_performance_metrics(),
            self._update_interpretability_metrics(),
            self._update_active_learning_metrics(),
            self._update_system_metrics(),
            self._update_portfolio_metrics(),
            return_exceptions=True
        )
    
    async def _update_trading_metrics(self):
        """Update trading performance metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Calculate trading metrics from signals table
                result = await session.execute("""
                    SELECT 
                        AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                        AVG(CASE WHEN pred = label THEN 1.0 ELSE 0.0 END) as precision,
                        AVG(CASE WHEN realized_rr > 0 THEN realized_rr ELSE 0 END) as profit_factor,
                        AVG(ABS(realized_rr)) as avg_rr
                    FROM signals 
                    WHERE ts >= NOW() - INTERVAL '1 hour'
                """)
                
                row = result.fetchone()
                if row:
                    self.win_rate.set(row.win_rate or 0.0)
                    self.precision.set(row.precision or 0.0)
                    self.profit_factor.set(row.profit_factor or 0.0)
                    self.avg_rr.set(row.avg_rr or 0.0)
                    
        except Exception as e:
            logger.error(f"❌ Error updating trading metrics: {e}")
    
    async def _update_model_metrics(self):
        """Update model performance metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Get model accuracy by type
                result = await session.execute("""
                    SELECT 
                        model_id,
                        AVG(CASE WHEN pred = label THEN 1.0 ELSE 0.0 END) as accuracy
                    FROM signals 
                    WHERE ts >= NOW() - INTERVAL '1 hour'
                    GROUP BY model_id
                """)
                
                # Reset all model accuracy metrics
                for metric in self.model_accuracy._metrics.values():
                    metric._value.set(0.0)
                
                # Set new values
                for row in result.fetchall():
                    model_type = row.model_id.split('_')[0] if '_' in row.model_id else 'unknown'
                    self.model_accuracy.labels(model_type=model_type).set(row.accuracy or 0.0)
                    
        except Exception as e:
            logger.error(f"❌ Error updating model metrics: {e}")
    
    async def _update_enhanced_drift_metrics(self):
        """Update enhanced drift detection metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Get drift metrics from signals
                result = await session.execute("""
                    SELECT 
                        model_id,
                        AVG(ABS(predicted_probability - 0.5)) as psi_drift,
                        AVG(CASE 
                            WHEN predicted_probability > 0.7 AND outcome = 'win' THEN 0.1
                            WHEN predicted_probability < 0.3 AND outcome = 'loss' THEN 0.1
                            ELSE 0.0
                        END) as auc_delta,
                        AVG(CASE 
                            WHEN ABS(predicted_probability - 0.5) > 0.2 THEN 0.2
                            ELSE 0.0
                        END) as concept_drift
                    FROM signals 
                    WHERE ts >= NOW() - INTERVAL '1 hour'
                    GROUP BY model_id
                """)
                
                # Reset metrics
                for metric in self.psi_drift_score._metrics.values():
                    metric._value.set(0.0)
                for metric in self.kl_drift_score._metrics.values():
                    metric._value.set(0.0)
                for metric in self.concept_drift_score._metrics.values():
                    metric._value.set(0.0)
                for metric in self.performance_degradation._metrics.values():
                    metric._value.set(0.0)
                
                # Set new values
                for row in result.fetchall():
                    model_id = row.model_id
                    self.psi_drift_score.labels(model_id=model_id).set(row.psi_drift or 0.0)
                    self.kl_drift_score.labels(model_id=model_id).set(0.0) # KL score not available in this query
                    self.concept_drift_score.labels(model_id=model_id).set(row.concept_drift or 0.0)
                    self.performance_degradation.labels(model_id=model_id).set(0.0) # Performance degradation not available in this query
                    
        except Exception as e:
            logger.error(f"❌ Error updating enhanced drift metrics: {e}")
    
    async def _update_live_performance_metrics(self):
        """Update live performance tracking metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Get live performance metrics from signals
                result = await session.execute("""
                    SELECT 
                        model_id,
                        AVG(CASE WHEN pred = label THEN 1.0 ELSE 0.0 END) as precision,
                        AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                        AVG(CASE WHEN pred = label AND outcome = 'win' THEN 1.0 ELSE 0.0 END) as hit_rate,
                        AVG(CASE WHEN pred = label AND outcome = 'win' THEN realized_rr ELSE 0 END) as profit_factor,
                        AVG(ABS(realized_rr)) as avg_rr,
                        AVG(CASE WHEN pred = label THEN 1.0 ELSE 0.0 END) as recall,
                        AVG(CASE WHEN pred = label AND outcome = 'win' THEN 1.0 ELSE 0.0 END) as f1_score,
                        AVG(CASE WHEN pred = label THEN 1.0 ELSE 0.0 END) as avg_confidence
                    FROM signals 
                    WHERE ts >= NOW() - INTERVAL '1 hour'
                    GROUP BY model_id
                """)
                
                # Reset metrics
                for metric in self.live_precision._metrics.values():
                    metric._value.set(0.0)
                for metric in self.live_recall._metrics.values():
                    metric._value.set(0.0)
                for metric in self.live_f1_score._metrics.values():
                    metric._value.set(0.0)
                for metric in self.live_sharpe_ratio._metrics.values():
                    metric._value.set(0.0)
                for metric in self.live_hit_rate._metrics.values():
                    metric._value.set(0.0)
                
                # Set new values
                for row in result.fetchall():
                    model_id = row.model_id
                    self.live_precision.labels(model_id=model_id).set(row.precision or 0.0)
                    self.live_recall.labels(model_id=model_id).set(row.recall or 0.0)
                    self.live_f1_score.labels(model_id=model_id).set(row.f1_score or 0.0)
                    self.live_sharpe_ratio.labels(model_id=model_id).set(0.0) # Sharpe ratio not available in this query
                    self.live_hit_rate.labels(model_id=model_id).set(row.hit_rate or 0.0)
                    
        except Exception as e:
            logger.error(f"❌ Error updating live performance metrics: {e}")
    
    async def _update_interpretability_metrics(self):
        """Update model interpretability metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Get attention and feature importance from model_interpretability table
                result = await session.execute("""
                    SELECT 
                        model_id,
                        AVG(CASE WHEN attention_type = 'lstm' THEN attention_weights ELSE NULL END) as lstm_attention_weights,
                        AVG(CASE WHEN attention_type = 'transformer' THEN attention_weights ELSE NULL END) as transformer_attention_weights,
                        AVG(CASE WHEN attention_type = 'feature_importance' THEN feature_importance ELSE NULL END) as feature_importance_scores
                    FROM model_interpretability 
                    WHERE ts >= NOW() - INTERVAL '1 hour'
                    GROUP BY model_id
                """)
                
                # Reset metrics
                for metric in self.attention_entropy._metrics.values():
                    metric._value.set(0.0)
                for metric in self.feature_importance_entropy._metrics.values():
                    metric._value.set(0.0)
                for metric in self.explanation_confidence._metrics.values():
                    metric._value.set(0.0)
                
                # Set new values
                for row in result.fetchall():
                    model_id = row.model_id
                    lstm_weights = row.lstm_attention_weights
                    transformer_weights = row.transformer_attention_weights
                    feature_importance_scores = row.feature_importance_scores
                    
                    if lstm_weights is not None:
                        self.attention_entropy.labels(model_id=model_id).set(float(-np.sum(lstm_weights * np.log(lstm_weights + 1e-10))))
                    if transformer_weights is not None:
                        self.attention_entropy.labels(model_id=model_id).set(float(-np.sum(transformer_weights * np.log(transformer_weights + 1e-10))))
                    if feature_importance_scores is not None:
                        self.feature_importance_entropy.labels(model_id=model_id).set(float(-np.sum(feature_importance_scores * np.log(feature_importance_scores + 1e-10))))
                    
        except Exception as e:
            logger.error(f"❌ Error updating interpretability metrics: {e}")
    
    async def _update_active_learning_metrics(self):
        """Update active learning metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Get active learning queue status
                result = await session.execute("""
                    SELECT 
                        status,
                        COUNT(*) as count
                    FROM active_learning_queue
                    GROUP BY status
                """)
                
                # Reset metrics
                self.active_learning_pending_items.set(0)
                self.active_learning_labeled_items.set(0)
                
                # Set new values
                for row in result.fetchall():
                    if row.status == 'pending':
                        self.active_learning_pending_items.set(row.count)
                    elif row.status == 'labeled':
                        self.active_learning_labeled_items.set(row.count)
                    elif row.status == 'processed':
                        pass  # Use total counter instead
                        
        except Exception as e:
            logger.error(f"❌ Error updating active learning metrics: {e}")
    
    async def _update_system_metrics(self):
        """Update system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_disk_usage.set((disk.used / disk.total) * 100)
            
            # Database health (simplified)
            if DATABASE_AVAILABLE:
                try:
                    async with get_async_session() as session:
                        await session.execute("SELECT 1")
                        self.database_connection_health.set(1.0)
                except Exception:
                    self.database_connection_health.set(0.0)
            else:
                self.database_connection_health.set(0.0)
                
        except Exception as e:
            logger.error(f"❌ Error updating system metrics: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            async with get_async_session() as session:
                # Calculate portfolio metrics (simplified)
                result = await session.execute("""
                    SELECT 
                        SUM(realized_rr) as total_pnl,
                        AVG(realized_rr) as avg_pnl
                    FROM signals 
                    WHERE ts >= NOW() - INTERVAL '24 hours'
                """)
                
                row = result.fetchone()
                if row:
                    # Simplified portfolio value (assuming starting with 10000)
                    portfolio_value = 10000 + (row.total_pnl or 0)
                    self.portfolio_total_value.set(portfolio_value)
                    self.portfolio_pnl.set(row.total_pnl or 0.0)
                    
                    # Simplified drawdown calculation
                    if row.total_pnl and row.total_pnl < 0:
                        drawdown = abs(row.total_pnl) / 10000
                        self.portfolio_drawdown.set(-drawdown)
                    else:
                        self.portfolio_drawdown.set(0.0)
                        
        except Exception as e:
            logger.error(f"❌ Error updating portfolio metrics: {e}")
    
    # Enhanced Drift Detection Methods
    async def detect_model_drift(self, model_id: str, feature_data: Dict[str, np.ndarray], 
                                predictions: np.ndarray = None, actuals: np.ndarray = None) -> List[DriftDetectionResult]:
        """Detect drift for a specific model"""
        drift_results = []
        
        # Data drift detection for each feature
        for feature_name, feature_values in feature_data.items():
            drift_result = self.drift_detector.detect_data_drift(model_id, feature_name, feature_values)
            drift_results.append(drift_result)
            
            # Update Prometheus metrics
            if drift_result.is_drift_detected:
                self.psi_drift_score.labels(model_id=model_id).set(drift_result.severity)
                logger.warning(f"🚨 Data drift detected for {model_id} - {feature_name}: {drift_result.severity:.2f}")
        
        # Concept drift detection if predictions and actuals provided
        if predictions is not None and actuals is not None:
            concept_drift_result = self.drift_detector.detect_concept_drift(model_id, predictions, actuals)
            drift_results.append(concept_drift_result)
            
            # Update Prometheus metrics
            if concept_drift_result.is_drift_detected:
                self.concept_drift_score.labels(model_id=model_id).set(concept_drift_result.severity)
                logger.warning(f"🚨 Concept drift detected for {model_id}: {concept_drift_result.severity:.2f}")
        
        return drift_results
    
    async def update_live_performance(self, model_id: str, metrics: LivePerformanceMetrics):
        """Update live performance metrics"""
        await self.live_performance_tracker.update_live_performance(model_id, metrics)
        
        # Update Prometheus metrics
        self.live_precision.labels(model_id=model_id).set(metrics.precision)
        self.live_recall.labels(model_id=model_id).set(metrics.recall)
        self.live_f1_score.labels(model_id=model_id).set(metrics.f1_score)
        self.live_sharpe_ratio.labels(model_id=model_id).set(metrics.sharpe_ratio)
        self.live_hit_rate.labels(model_id=model_id).set(metrics.hit_rate)
        
        logger.info(f"📊 Updated live performance for {model_id}: precision={metrics.precision:.3f}, sharpe={metrics.sharpe_ratio:.3f}")
    
    def set_backtest_baseline(self, model_id: str, baseline_metrics: Dict[str, float]):
        """Set backtest baseline for performance comparison"""
        self.live_performance_tracker.set_backtest_baseline(model_id, baseline_metrics)
    
    # Interpretability Methods
    def record_lstm_attention(self, model_id: str, attention_weights: np.ndarray, timesteps: List[str]):
        """Record LSTM attention weights for interpretability"""
        self.interpretability_tracker.record_lstm_attention(model_id, attention_weights, timesteps)
        
        # Calculate and update attention entropy
        attention_entropy = float(-np.sum(attention_weights * np.log(attention_weights + 1e-10)))
        self.attention_entropy.labels(model_id=model_id).set(attention_entropy)
    
    def record_transformer_attention(self, model_id: str, attention_weights: np.ndarray, timeframes: List[str]):
        """Record Transformer attention weights for interpretability"""
        self.interpretability_tracker.record_transformer_attention(model_id, attention_weights, timeframes)
        
        # Calculate and update attention entropy
        attention_entropy = float(-np.sum(attention_weights * np.log(attention_weights + 1e-10)))
        self.attention_entropy.labels(model_id=model_id).set(attention_entropy)
    
    def record_feature_importance(self, model_id: str, feature_names: List[str], importance_scores: np.ndarray):
        """Record feature importance scores for interpretability"""
        self.interpretability_tracker.record_feature_importance(model_id, feature_names, importance_scores)
        
        # Calculate and update feature importance entropy
        importance_entropy = float(-np.sum(importance_scores * np.log(importance_scores + 1e-10)))
        self.feature_importance_entropy.labels(model_id=model_id).set(importance_entropy)
    
    def record_explanation(self, model_id: str, prediction: float, explanation: Dict[str, Any]):
        """Record model explanation for interpretability"""
        self.interpretability_tracker.record_explanation(model_id, prediction, explanation)
        
        # Calculate explanation confidence (simplified)
        confidence = explanation.get('confidence', 0.5)
        self.explanation_confidence.labels(model_id=model_id).set(confidence)
    
    # Enhanced Monitoring Methods
    def get_drift_alerts(self) -> List[DriftDetectionResult]:
        """Get recent drift alerts"""
        return self.drift_detector.get_drift_alerts()
    
    def get_performance_alerts(self) -> List[BacktestComparison]:
        """Get recent performance alerts"""
        return self.live_performance_tracker.get_performance_alerts()
    
    def get_interpretability_data(self, model_id: str = None) -> Dict[str, Any]:
        """Get interpretability data for models"""
        return {
            'attention_weights': self.interpretability_tracker.get_recent_attention(model_id) if model_id else {},
            'feature_importance': self.interpretability_tracker.get_recent_feature_importance(model_id) if model_id else {},
            'explanations': self.interpretability_tracker.get_recent_explanations(model_id)
        }
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        system_metrics = self.get_system_metrics()
        drift_alerts = self.get_drift_alerts()
        performance_alerts = self.get_performance_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_usage': system_metrics.cpu_usage,
                'memory_usage': system_metrics.memory_usage,
                'disk_usage': system_metrics.disk_usage,
                'database_health': system_metrics.database_health,
                'active_connections': system_metrics.active_connections
            },
            'drift_alerts': {
                'total_alerts': len(drift_alerts),
                'recent_alerts': [alert.__dict__ for alert in drift_alerts[-5:]]  # Last 5 alerts
            },
            'performance_alerts': {
                'total_alerts': len(performance_alerts),
                'recent_alerts': [alert.__dict__ for alert in performance_alerts[-5:]]  # Last 5 alerts
            },
            'service_status': {
                'is_running': self.is_running,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'database_available': DATABASE_AVAILABLE
            }
        }
    
    def record_inference_duration(self, duration_seconds: float):
        """Record model inference duration"""
        if PROMETHEUS_AVAILABLE:
            self.inference_duration.observe(duration_seconds)
    
    def record_signal_generated(self, symbol: str, model_id: str):
        """Record a signal generation"""
        if PROMETHEUS_AVAILABLE:
            self.signals_generated.labels(symbol=symbol, model_id=model_id).inc()
    
    def record_signal_executed(self, symbol: str, model_id: str):
        """Record a signal execution"""
        if PROMETHEUS_AVAILABLE:
            self.signals_executed.labels(symbol=symbol, model_id=model_id).inc()
    
    def record_signal_rejected(self, symbol: str, model_id: str):
        """Record a signal rejection"""
        if PROMETHEUS_AVAILABLE:
            self.signals_rejected.labels(symbol=symbol, model_id=model_id).inc()
    
    def record_model_training_failure(self):
        """Record a model training failure"""
        if PROMETHEUS_AVAILABLE:
            self.model_training_failures.inc()
    
    def record_active_learning_processed(self):
        """Record an active learning item processed"""
        if PROMETHEUS_AVAILABLE:
            self.active_learning_processed_items_total.inc()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics as string"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available"
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Error generating metrics: {e}")
            return f"# Error generating metrics: {e}"
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            # Database health check
            db_health = 0.0
            if DATABASE_AVAILABLE:
                try:
                    # This would need to be async in practice
                    db_health = 1.0
                except Exception:
                    db_health = 0.0
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                database_health=db_health,
                active_connections=0  # Would need to track actual connections
            )
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, 0)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'service_running': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'database_available': DATABASE_AVAILABLE,
            'drift_alerts_count': len(self.get_drift_alerts()),
            'performance_alerts_count': len(self.get_performance_alerts())
        }

    # ==================== CLOSED-LOOP MONITORING METHODS ====================

    async def create_monitoring_alert(self, alert: MonitoringAlert) -> bool:
        """Create a monitoring alert for closed-loop integration"""
        try:
            if not DATABASE_AVAILABLE:
                logger.warning("Database not available for monitoring alert")
                return False
            
            # Use synchronous database connection for now
            engine = create_engine("postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
            with engine.connect() as conn:
                # Insert alert into monitoring_alert_triggers table
                query = text("""
                    INSERT INTO monitoring_alert_triggers (
                        alert_id, model_id, alert_type, severity_level, trigger_condition,
                        current_value, threshold_value, is_triggered, triggered_at, alert_metadata
                    ) VALUES (
                        :alert_id, :model_id, :alert_type, :severity_level, :trigger_condition,
                        :current_value, :threshold_value, :is_triggered, :triggered_at, :alert_metadata
                    )
                """)
                
                conn.execute(query, {
                    'alert_id': alert.alert_id,
                    'model_id': alert.model_id,
                    'alert_type': alert.alert_type,
                    'severity_level': alert.severity_level,
                    'trigger_condition': json.dumps(alert.trigger_condition),
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'is_triggered': alert.is_triggered,
                    'triggered_at': alert.triggered_at,
                    'alert_metadata': json.dumps(alert.alert_metadata or {})
                })
                
                conn.commit()
                
                logger.info(f"Created monitoring alert: {alert.alert_id} for model: {alert.model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating monitoring alert: {e}")
            return False

    async def trigger_closed_loop_action(self, action: ClosedLoopAction) -> bool:
        """Trigger a closed-loop action based on monitoring alert"""
        try:
            if not DATABASE_AVAILABLE:
                logger.warning("Database not available for closed-loop action")
                return False
            
            # Use synchronous database connection for now
            engine = create_engine("postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
            with engine.connect() as conn:
                # Insert action into closed_loop_actions table
                query = text("""
                    INSERT INTO closed_loop_actions (
                        action_id, alert_id, model_id, action_type, action_status,
                        trigger_source, action_config, execution_start, action_metadata
                    ) VALUES (
                        :action_id, :alert_id, :model_id, :action_type, :action_status,
                        :trigger_source, :action_config, :execution_start, :action_metadata
                    )
                """)
                
                conn.execute(query, {
                    'action_id': action.action_id,
                    'alert_id': action.alert_id,
                    'model_id': action.model_id,
                    'action_type': action.action_type,
                    'action_status': action.action_status,
                    'trigger_source': action.trigger_source,
                    'action_config': json.dumps(action.action_config),
                    'execution_start': action.execution_start,
                    'action_metadata': json.dumps(action.action_metadata or {})
                })
                
                conn.commit()
                
                logger.info(f"Triggered closed-loop action: {action.action_id} for model: {action.model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error triggering closed-loop action: {e}")
            return False

    async def check_automated_response_rules(self, model_id: str, alert_type: str, 
                                           current_value: float) -> List[Dict[str, Any]]:
        """Check automated response rules for a given alert"""
        try:
            if not DATABASE_AVAILABLE:
                logger.warning("Database not available for response rules check")
                return []
            
            # Use synchronous database connection for now
            engine = create_engine("postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
            with engine.connect() as conn:
                # Get active rules for the model and alert type
                query = text("""
                    SELECT rule_id, rule_name, rule_type, trigger_conditions, response_actions,
                           priority, cooldown_period_minutes, max_triggers_per_hour
                    FROM automated_response_rules
                    WHERE (model_id = :model_id OR model_id IS NULL)
                    AND rule_type LIKE :rule_type_pattern
                    AND is_active = TRUE
                    ORDER BY priority ASC
                """)
                
                result = conn.execute(query, {
                    'model_id': model_id,
                    'rule_type_pattern': f'%{alert_type}%'
                })
                
                triggered_rules = []
                for row in result:
                    trigger_conditions = json.loads(row.trigger_conditions)
                    response_actions = json.loads(row.response_actions)
                    
                    # Check if rule should be triggered
                    if self._should_trigger_rule(trigger_conditions, current_value, alert_type):
                        triggered_rules.append({
                            'rule_id': row.rule_id,
                            'rule_name': row.rule_name,
                            'rule_type': row.rule_type,
                            'trigger_conditions': trigger_conditions,
                            'response_actions': response_actions,
                            'priority': row.priority,
                            'cooldown_period_minutes': row.cooldown_period_minutes,
                            'max_triggers_per_hour': row.max_triggers_per_hour
                        })
                
                return triggered_rules
                
        except Exception as e:
            logger.error(f"Error checking automated response rules: {e}")
            return []

    def _should_trigger_rule(self, trigger_conditions: Dict[str, Any], 
                           current_value: float, alert_type: str) -> bool:
        """Check if a rule should be triggered based on conditions"""
        try:
            if alert_type == 'drift':
                drift_threshold = trigger_conditions.get('drift_score_threshold', 0.25)
                return current_value >= drift_threshold
                
            elif alert_type == 'performance':
                performance_drop = trigger_conditions.get('performance_drop_threshold', 0.1)
                return current_value >= performance_drop
                
            elif alert_type == 'risk':
                risk_threshold = trigger_conditions.get('risk_score_threshold', 80)
                return current_value >= risk_threshold
                
            elif alert_type == 'data_quality':
                quality_threshold = trigger_conditions.get('quality_threshold', 0.8)
                return current_value <= quality_threshold
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking rule trigger conditions: {e}")
            return False

    async def update_alert_trigger_status(self, alert_id: str, is_triggered: bool, 
                                        retraining_job_id: str = None) -> bool:
        """Update alert trigger status and link to retraining job"""
        try:
            if not DATABASE_AVAILABLE:
                logger.warning("Database not available for alert status update")
                return False
            
            # Use synchronous database connection for now
            engine = create_engine("postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
            with engine.connect() as conn:
                query = text("""
                    UPDATE monitoring_alert_triggers
                    SET is_triggered = :is_triggered,
                        triggered_at = CASE WHEN :is_triggered THEN NOW() ELSE triggered_at END,
                        retraining_job_id = :retraining_job_id,
                        retraining_status = CASE WHEN :retraining_job_id IS NOT NULL THEN 'triggered' ELSE retraining_status END,
                        updated_at = NOW()
                    WHERE alert_id = :alert_id
                """)
                
                conn.execute(query, {
                    'alert_id': alert_id,
                    'is_triggered': is_triggered,
                    'retraining_job_id': retraining_job_id
                })
                
                conn.commit()
                
                logger.info(f"Updated alert trigger status: {alert_id} -> {is_triggered}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating alert trigger status: {e}")
            return False

    async def log_feedback_loop_metrics(self, model_id: str, loop_type: str, 
                                      metrics: Dict[str, Any]) -> bool:
        """Log feedback loop metrics for analysis"""
        try:
            if not DATABASE_AVAILABLE:
                logger.warning("Database not available for feedback loop metrics")
                return False
            
            # Use synchronous database connection for now
            engine = create_engine("postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
            with engine.connect() as conn:
                query = text("""
                    INSERT INTO feedback_loop_metrics (
                        metric_id, model_id, loop_type, trigger_to_action_latency_seconds,
                        action_success_rate, performance_improvement, drift_reduction,
                        false_positive_rate, false_negative_rate, total_triggers,
                        successful_actions, failed_actions, metrics_metadata
                    ) VALUES (
                        :metric_id, :model_id, :loop_type, :latency, :success_rate,
                        :performance_improvement, :drift_reduction, :false_positive_rate,
                        :false_negative_rate, :total_triggers, :successful_actions,
                        :failed_actions, :metrics_metadata
                    )
                """)
                
                metric_id = f"feedback_loop_{model_id}_{loop_type}_{int(time.time())}"
                
                conn.execute(query, {
                    'metric_id': metric_id,
                    'model_id': model_id,
                    'loop_type': loop_type,
                    'latency': metrics.get('trigger_to_action_latency_seconds', 0),
                    'success_rate': metrics.get('action_success_rate', 0),
                    'performance_improvement': metrics.get('performance_improvement', 0),
                    'drift_reduction': metrics.get('drift_reduction', 0),
                    'false_positive_rate': metrics.get('false_positive_rate', 0),
                    'false_negative_rate': metrics.get('false_negative_rate', 0),
                    'total_triggers': metrics.get('total_triggers', 0),
                    'successful_actions': metrics.get('successful_actions', 0),
                    'failed_actions': metrics.get('failed_actions', 0),
                    'metrics_metadata': json.dumps(metrics.get('metadata', {}))
                })
                
                conn.commit()
                
                logger.info(f"Logged feedback loop metrics: {metric_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error logging feedback loop metrics: {e}")
            return False


# FastAPI integration for metrics endpoint
class MetricsEndpoint:
    """FastAPI endpoint for Prometheus metrics"""
    
    def __init__(self, monitoring_service: MonitoringService):
        self.monitoring_service = monitoring_service
    
    async def get_metrics(self):
        """Get Prometheus metrics"""
        from fastapi import Response
        
        metrics = self.monitoring_service.get_metrics()
        return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
    
    async def get_health(self):
        """Get health check endpoint"""
        health_report = self.monitoring_service.get_comprehensive_health_report()
        return health_report
    
    async def get_drift_alerts(self):
        """Get drift alerts endpoint"""
        drift_alerts = self.monitoring_service.get_drift_alerts()
        return {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(drift_alerts),
            'alerts': [alert.__dict__ for alert in drift_alerts[-10:]]  # Last 10 alerts
        }
    
    async def get_performance_alerts(self):
        """Get performance alerts endpoint"""
        performance_alerts = self.monitoring_service.get_performance_alerts()
        return {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(performance_alerts),
            'alerts': [alert.__dict__ for alert in performance_alerts[-10:]]  # Last 10 alerts
        }
    
    async def get_interpretability(self, model_id: str = None):
        """Get interpretability data endpoint"""
        interpretability_data = self.monitoring_service.get_interpretability_data(model_id)
        return {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'data': interpretability_data
        }


# Example usage and integration
async def setup_monitoring_endpoints(app):
    """Setup monitoring endpoints for FastAPI app"""
    try:
        from fastapi import APIRouter
        
        # Initialize monitoring service
        monitoring_service = MonitoringService()
        await monitoring_service.start()
        
        # Create metrics endpoint
        metrics_endpoint = MetricsEndpoint(monitoring_service)
        
        # Create router
        router = APIRouter(prefix="/monitoring", tags=["monitoring"])
        
        # Add endpoints
        router.add_api_route("/metrics", metrics_endpoint.get_metrics, methods=["GET"])
        router.add_api_route("/health", metrics_endpoint.get_health, methods=["GET"])
        router.add_api_route("/drift-alerts", metrics_endpoint.get_drift_alerts, methods=["GET"])
        router.add_api_route("/performance-alerts", metrics_endpoint.get_performance_alerts, methods=["GET"])
        router.add_api_route("/interpretability", metrics_endpoint.get_interpretability, methods=["GET"])
        
        # Include router in app
        app.include_router(router)
        
        # Store service reference for later use
        app.state.monitoring_service = monitoring_service
        
        logger.info("✅ Enhanced monitoring endpoints setup completed")
        
    except Exception as e:
        logger.error(f"❌ Error setting up monitoring endpoints: {e}")


# Context manager for inference timing
class InferenceTimer:
    """Context manager for timing model inference"""
    
    def __init__(self, monitoring_service: MonitoringService):
        self.monitoring_service = monitoring_service
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitoring_service.record_inference_duration(duration)
