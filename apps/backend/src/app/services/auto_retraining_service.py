#!/usr/bin/env python3
"""
Auto-Retraining Service
Handles automatic model retraining, drift detection, and performance monitoring
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

# ML imports for drift detection
try:
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available for drift detection")

logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL = "manual"

class DriftType(Enum):
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    STATISTICAL = "statistical"

@dataclass
class DriftMetrics:
    """Data drift detection metrics"""
    drift_type: DriftType
    feature_name: str
    drift_score: float
    threshold: float
    is_drift_detected: bool
    metadata: Dict[str, Any]

@dataclass
class RetrainingConfig:
    """Configuration for auto-retraining"""
    model_name: str
    symbol: str
    timeframe: str
    retraining_schedule_days: int = 7  # Weekly retraining
    drift_threshold: float = 0.25  # PSI threshold for drift detection
    performance_degradation_threshold: float = 0.1  # 10% performance drop
    min_samples_for_retraining: int = 1000
    max_samples_for_retraining: int = 50000

class AutoRetrainingService:
    """Service for automatic model retraining and drift detection"""
    
    def __init__(self, db_pool, ml_training_service):
        self.db_pool = db_pool
        self.ml_training_service = ml_training_service
        self.logger = logging.getLogger(__name__)
        
        # Retraining configurations
        self.retraining_configs = {}
        
        # Drift detection parameters
        self.drift_detection_params = {
            'psi_threshold': 0.25,
            'kl_divergence_threshold': 0.1,
            'statistical_threshold': 0.05,
            'min_samples': 100,
            'window_size': 1000
        }
        
        self.logger.info("🔄 Auto-Retraining Service initialized")
    
    async def register_model_for_auto_retraining(self, config: RetrainingConfig):
        """Register a model for automatic retraining"""
        try:
            self.retraining_configs[f"{config.model_name}_{config.symbol}_{config.timeframe}"] = config
            self.logger.info(f"✅ Registered {config.model_name} for auto-retraining")
            
            # Store configuration in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_retraining_history (
                        model_name, retraining_trigger, retraining_start, status
                    ) VALUES ($1, $2, $3, $4)
                """, config.model_name, RetrainingTrigger.SCHEDULED.value, 
                     datetime.now(), 'registered')
            
        except Exception as e:
            self.logger.error(f"❌ Error registering model for auto-retraining: {e}")
    
    async def check_retraining_schedule(self):
        """Check if any models need scheduled retraining"""
        try:
            current_time = datetime.now()
            models_to_retrain = []
            
            for config_key, config in self.retraining_configs.items():
                # Check last retraining time
                last_retraining = await self._get_last_retraining_time(config.model_name)
                
                if last_retraining is None or \
                   (current_time - last_retraining).days >= config.retraining_schedule_days:
                    models_to_retrain.append(config)
            
            # Trigger retraining for models that need it
            for config in models_to_retrain:
                await self._trigger_retraining(config, RetrainingTrigger.SCHEDULED)
            
            if models_to_retrain:
                self.logger.info(f"🔄 Triggered scheduled retraining for {len(models_to_retrain)} models")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking retraining schedule: {e}")
    
    async def detect_data_drift(self, model_name: str, symbol: str, timeframe: str) -> List[DriftMetrics]:
        """Detect data drift for a specific model"""
        try:
            if not ML_AVAILABLE:
                self.logger.warning("⚠️ ML libraries not available for drift detection")
                return []
            
            drift_metrics = []
            
            # Get recent data for drift detection
            recent_data = await self._get_recent_ml_data(symbol, timeframe, days=7)
            historical_data = await self._get_historical_ml_data(symbol, timeframe, days=30)
            
            if len(recent_data) < self.drift_detection_params['min_samples'] or \
               len(historical_data) < self.drift_detection_params['min_samples']:
                return []
            
            # Convert to DataFrames
            recent_df = pd.DataFrame(recent_data)
            historical_df = pd.DataFrame(historical_data)
            
            # Detect drift for each feature
            features = ['volume_ratio', 'volume_positioning_score', 'order_book_imbalance']
            
            for feature in features:
                if feature in recent_df.columns and feature in historical_df.columns:
                    # PSI (Population Stability Index)
                    psi_score = self._calculate_psi(recent_df[feature], historical_df[feature])
                    psi_drift = DriftMetrics(
                        drift_type=DriftType.PSI,
                        feature_name=feature,
                        drift_score=psi_score,
                        threshold=self.drift_detection_params['psi_threshold'],
                        is_drift_detected=psi_score > self.drift_detection_params['psi_threshold'],
                        metadata={'psi_score': psi_score}
                    )
                    drift_metrics.append(psi_drift)
                    
                    # Statistical test (KS test)
                    ks_statistic, ks_pvalue = ks_2samp(recent_df[feature], historical_df[feature])
                    statistical_drift = DriftMetrics(
                        drift_type=DriftType.STATISTICAL,
                        feature_name=feature,
                        drift_score=ks_pvalue,
                        threshold=self.drift_detection_params['statistical_threshold'],
                        is_drift_detected=ks_pvalue < self.drift_detection_params['statistical_threshold'],
                        metadata={'ks_statistic': ks_statistic, 'ks_pvalue': ks_pvalue}
                    )
                    drift_metrics.append(statistical_drift)
            
            # Store drift metrics in database
            await self._store_drift_metrics(model_name, symbol, timeframe, drift_metrics)
            
            # Check if significant drift detected
            significant_drift = any(metric.is_drift_detected for metric in drift_metrics)
            if significant_drift:
                config = self.retraining_configs.get(f"{model_name}_{symbol}_{timeframe}")
                if config:
                    await self._trigger_retraining(config, RetrainingTrigger.DRIFT_DETECTED)
                    self.logger.warning(f"🚨 Data drift detected for {model_name}, triggering retraining")
            
            return drift_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting data drift: {e}")
            return []
    
    async def check_performance_degradation(self, model_name: str, symbol: str, timeframe: str) -> bool:
        """Check if model performance has degraded significantly"""
        try:
            # Get recent performance metrics
            recent_performance = await self._get_recent_model_performance(model_name, symbol, timeframe, days=7)
            historical_performance = await self._get_recent_model_performance(model_name, symbol, timeframe, days=30)
            
            if not recent_performance or not historical_performance:
                return False
            
            # Calculate performance change
            recent_auc = np.mean([p['metric_value'] for p in recent_performance if p['metric_name'] == 'auc'])
            historical_auc = np.mean([p['metric_value'] for p in historical_performance if p['metric_name'] == 'auc'])
            
            if historical_auc > 0:
                performance_change = (recent_auc - historical_auc) / historical_auc
                
                if performance_change < -self.drift_detection_params['performance_degradation_threshold']:
                    config = self.retraining_configs.get(f"{model_name}_{symbol}_{timeframe}")
                    if config:
                        await self._trigger_retraining(config, RetrainingTrigger.PERFORMANCE_DEGRADATION)
                        self.logger.warning(f"📉 Performance degradation detected for {model_name}, triggering retraining")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking performance degradation: {e}")
            return False
    
    async def _trigger_retraining(self, config: RetrainingConfig, trigger: RetrainingTrigger):
        """Trigger model retraining"""
        try:
            # Record retraining start
            retraining_id = await self._record_retraining_start(config, trigger)
            
            # Get training data
            training_data = await self._get_training_data(config.symbol, config.timeframe)
            
            if len(training_data) < config.min_samples_for_retraining:
                self.logger.warning(f"⚠️ Insufficient data for retraining {config.model_name}")
                await self._record_retraining_failure(retraining_id, "Insufficient data")
                return
            
            # Limit data size
            if len(training_data) > config.max_samples_for_retraining:
                training_data = training_data[-config.max_samples_for_retraining:]
            
            # Train new model using existing ML training service
            model_config = self._create_model_config(config)
            new_model_version = await self.ml_training_service.train_model(model_config, training_data)
            
            if new_model_version:
                # Activate new model
                await self.ml_training_service.activate_model(config.model_name, new_model_version)
                
                # Record successful retraining
                await self._record_retraining_success(retraining_id, new_model_version, len(training_data))
                
                self.logger.info(f"✅ Successfully retrained {config.model_name} -> {new_model_version}")
            else:
                await self._record_retraining_failure(retraining_id, "Training failed")
                
        except Exception as e:
            self.logger.error(f"❌ Error during retraining: {e}")
            await self._record_retraining_failure(retraining_id, str(e))
    
    def _calculate_psi(self, recent_data: pd.Series, historical_data: pd.Series) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins for both datasets
            bins = np.linspace(min(historical_data.min(), recent_data.min()),
                             max(historical_data.max(), recent_data.max()), 11)
            
            # Calculate histograms
            hist_historical, _ = np.histogram(historical_data, bins=bins)
            hist_recent, _ = np.histogram(recent_data, bins=bins)
            
            # Normalize to probabilities
            hist_historical = hist_historical / hist_historical.sum()
            hist_recent = hist_recent / hist_recent.sum()
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            hist_historical = hist_historical + epsilon
            hist_recent = hist_recent + epsilon
            
            # Calculate PSI
            psi = np.sum((hist_recent - hist_historical) * np.log(hist_recent / hist_historical))
            
            return float(psi)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating PSI: {e}")
            return 0.0
    
    async def _get_recent_ml_data(self, symbol: str, timeframe: str, days: int) -> List[Dict]:
        """Get recent ML dataset for drift detection"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM volume_analysis_ml_dataset 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 day' * $3
                    ORDER BY timestamp DESC
                """, symbol, timeframe, days)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting recent ML data: {e}")
            return []
    
    async def _get_historical_ml_data(self, symbol: str, timeframe: str, days: int) -> List[Dict]:
        """Get historical ML dataset for drift detection"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM volume_analysis_ml_dataset 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 day' * $3
                    AND timestamp < NOW() - INTERVAL '1 day' * 7
                    ORDER BY timestamp DESC
                """, symbol, timeframe, days + 7)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting historical ML data: {e}")
            return []
    
    async def _get_training_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get training data for model retraining"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM volume_analysis_ml_dataset 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY timestamp ASC
                """, symbol, timeframe)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting training data: {e}")
            return []
    
    async def _get_last_retraining_time(self, model_name: str) -> Optional[datetime]:
        """Get the last retraining time for a model"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT retraining_start FROM model_retraining_history 
                    WHERE model_name = $1 AND status = 'completed'
                    ORDER BY retraining_start DESC LIMIT 1
                """, model_name)
                
                return row['retraining_start'] if row else None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting last retraining time: {e}")
            return None
    
    async def _get_recent_model_performance(self, model_name: str, symbol: str, timeframe: str, days: int) -> List[Dict]:
        """Get recent model performance metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM model_performance 
                    WHERE model_name = $1 AND symbol = $2 AND timeframe = $3
                    AND timestamp >= NOW() - INTERVAL '1 day' * $4
                    ORDER BY timestamp DESC
                """, model_name, symbol, timeframe, days)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting model performance: {e}")
            return []
    
    async def _record_retraining_start(self, config: RetrainingConfig, trigger: RetrainingTrigger) -> int:
        """Record the start of a retraining session"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO model_retraining_history (
                        model_name, retraining_trigger, retraining_start, status
                    ) VALUES ($1, $2, $3, $4) RETURNING id
                """, config.model_name, trigger.value, datetime.now(), 'in_progress')
                
                return row['id']
                
        except Exception as e:
            self.logger.error(f"❌ Error recording retraining start: {e}")
            return 0
    
    async def _record_retraining_success(self, retraining_id: int, new_version: str, samples: int):
        """Record successful retraining"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE model_retraining_history 
                    SET retraining_end = $1, new_model_version = $2, 
                        training_samples = $3, status = 'completed'
                    WHERE id = $4
                """, datetime.now(), new_version, samples, retraining_id)
                
        except Exception as e:
            self.logger.error(f"❌ Error recording retraining success: {e}")
    
    async def _record_retraining_failure(self, retraining_id: int, error_message: str):
        """Record failed retraining"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE model_retraining_history 
                    SET retraining_end = $1, status = 'failed',
                        retraining_metadata = $2
                    WHERE id = $3
                """, datetime.now(), json.dumps({'error': error_message}), retraining_id)
                
        except Exception as e:
            self.logger.error(f"❌ Error recording retraining failure: {e}")
    
    async def _store_drift_metrics(self, model_name: str, symbol: str, timeframe: str, drift_metrics: List[DriftMetrics]):
        """Store drift detection metrics in database"""
        try:
            async with self.db_pool.acquire() as conn:
                for metric in drift_metrics:
                    await conn.execute("""
                        INSERT INTO data_drift_metrics (
                            model_name, symbol, timeframe, timestamp, drift_type,
                            feature_name, drift_score, threshold, is_drift_detected, drift_metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, model_name, symbol, timeframe, datetime.now(), metric.drift_type.value,
                         metric.feature_name, metric.drift_score, metric.threshold,
                         metric.is_drift_detected, json.dumps(metric.metadata))
                
        except Exception as e:
            self.logger.error(f"❌ Error storing drift metrics: {e}")
    
    def _create_model_config(self, config: RetrainingConfig):
        """Create model configuration for retraining"""
        from .ml_model_training_service import ModelConfig, ModelType, LabelType
        
        return ModelConfig(
            model_type=ModelType.LIGHTGBM,
            label_type=LabelType.BINARY_BREAKOUT,
            symbol=config.symbol,
            timeframe=config.timeframe,
            features=['volume_ratio', 'volume_positioning_score', 'order_book_imbalance'],
            hyperparameters={
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 127,
                "max_depth": 8,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42
            },
            training_window_days=30,
            validation_window_days=7,
            min_samples=config.min_samples_for_retraining,
            max_samples=config.max_samples_for_retraining
        )
    
    async def run_auto_retraining_cycle(self):
        """Run a complete auto-retraining cycle"""
        try:
            self.logger.info("🔄 Starting auto-retraining cycle")
            
            # 1. Check scheduled retraining
            await self.check_retraining_schedule()
            
            # 2. Check for data drift and performance degradation
            for config_key, config in self.retraining_configs.items():
                # Detect data drift
                drift_metrics = await self.detect_data_drift(config.model_name, config.symbol, config.timeframe)
                
                # Check performance degradation
                await self.check_performance_degradation(config.model_name, config.symbol, config.timeframe)
            
            self.logger.info("✅ Auto-retraining cycle completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in auto-retraining cycle: {e}")
    
    async def get_retraining_status(self, model_name: str = None) -> List[Dict]:
        """Get retraining status for models"""
        try:
            async with self.db_pool.acquire() as conn:
                if model_name:
                    rows = await conn.fetch("""
                        SELECT * FROM model_retraining_history 
                        WHERE model_name = $1 
                        ORDER BY retraining_start DESC LIMIT 10
                    """, model_name)
                else:
                    rows = await conn.fetch("""
                        SELECT * FROM model_retraining_history 
                        ORDER BY retraining_start DESC LIMIT 20
                    """)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting retraining status: {e}")
            return []
    
    async def get_drift_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent drift alerts"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM data_drift_metrics 
                    WHERE is_drift_detected = TRUE 
                    AND timestamp >= NOW() - INTERVAL '1 hour' * $1
                    ORDER BY timestamp DESC
                """, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting drift alerts: {e}")
            return []
