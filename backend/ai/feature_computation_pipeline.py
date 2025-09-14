#!/usr/bin/env python3
"""
Feature Computation Pipeline
Phase 2A: Feature Store Implementation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from dataclasses import dataclass

# Import our feature store
from .feature_store_timescaledb import TimescaleDBFeatureStore, FeatureDefinition, FeatureSet

logger = logging.getLogger(__name__)

@dataclass
class ComputationJob:
    """Feature computation job definition"""
    job_id: str
    feature_names: List[str]
    entity_ids: List[str]
    start_time: datetime
    end_time: datetime
    priority: int = 1
    status: str = "pending"
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    error_message: str = None
    metadata: Dict[str, Any] = None

class FeatureComputationPipeline:
    """Automated feature computation pipeline"""
    
    def __init__(self, feature_store: TimescaleDBFeatureStore, max_workers: int = 4):
        self.feature_store = feature_store
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._active_jobs = {}
        self._job_queue = []
        
        # Initialize default features
        self._initialize_default_features()
        
        logger.info(f"🚀 Feature Computation Pipeline initialized with {max_workers} workers")
    
    def _initialize_default_features(self):
        """Initialize default technical indicator features"""
        try:
            # Technical indicator features
            technical_features = [
                FeatureDefinition(
                    name="rsi_14",
                    description="Relative Strength Index (14-period)",
                    data_type="float",
                    source_table="candles",
                    computation_rule="rsi_14_period",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "momentum", "oscillator"]
                ),
                FeatureDefinition(
                    name="macd",
                    description="MACD Line (12,26,9)",
                    data_type="float",
                    source_table="candles",
                    computation_rule="macd_line",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "trend", "momentum"]
                ),
                FeatureDefinition(
                    name="macd_signal",
                    description="MACD Signal Line",
                    data_type="float",
                    source_table="candles",
                    computation_rule="macd_signal",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "trend", "momentum"]
                ),
                FeatureDefinition(
                    name="ema_20",
                    description="Exponential Moving Average (20-period)",
                    data_type="float",
                    source_table="candles",
                    computation_rule="ema_20_period",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "trend", "moving_average"]
                ),
                FeatureDefinition(
                    name="ema_50",
                    description="Exponential Moving Average (50-period)",
                    data_type="float",
                    source_table="candles",
                    computation_rule="ema_50_period",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "trend", "moving_average"]
                ),
                FeatureDefinition(
                    name="bb_position",
                    description="Bollinger Bands Position (-1 to 1)",
                    data_type="float",
                    source_table="candles",
                    computation_rule="bollinger_bands_position",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "volatility", "mean_reversion"]
                ),
                FeatureDefinition(
                    name="atr",
                    description="Average True Range (14-period)",
                    data_type="float",
                    source_table="candles",
                    computation_rule="atr_14_period",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "volatility"]
                ),
                FeatureDefinition(
                    name="volume_sma_ratio",
                    description="Volume to Simple Moving Average Ratio",
                    data_type="float",
                    source_table="candles",
                    computation_rule="volume_sma_ratio",
                    version="1.0.0",
                    created_at=datetime.now(),
                    tags=["technical", "volume", "momentum"]
                )
            ]
            
            # Register all technical features
            for feature in technical_features:
                self.feature_store.register_feature(feature)
            
            # Create feature sets
            momentum_features = FeatureSet(
                name="momentum_indicators",
                description="Momentum-based technical indicators",
                features=["rsi_14", "macd", "macd_signal"],
                version="1.0.0",
                created_at=datetime.now(),
                metadata={"category": "momentum", "timeframe": "1h"}
            )
            
            trend_features = FeatureSet(
                name="trend_indicators",
                description="Trend-following technical indicators",
                features=["ema_20", "ema_50"],
                version="1.0.0",
                created_at=datetime.now(),
                metadata={"category": "trend", "timeframe": "1h"}
            )
            
            volatility_features = FeatureSet(
                name="volatility_indicators",
                description="Volatility-based technical indicators",
                features=["bb_position", "atr"],
                version="1.0.0",
                created_at=datetime.now(),
                metadata={"category": "volatility", "timeframe": "1h"}
            )
            
            # Register feature sets
            self.feature_store.register_feature_set(momentum_features)
            self.feature_store.register_feature_set(trend_features)
            self.feature_store.register_feature_set(volatility_features)
            
            logger.info("✅ Default technical features initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize default features: {e}")
    
    def submit_computation_job(self, job: ComputationJob) -> str:
        """Submit a feature computation job to the pipeline"""
        try:
            with self._lock:
                job.job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._job_queue)}"
                job.created_at = datetime.now()
                job.status = "pending"
                
                self._job_queue.append(job)
                
                # Sort queue by priority (higher priority first)
                self._job_queue.sort(key=lambda x: x.priority, reverse=True)
                
                logger.info(f"📋 Submitted computation job {job.job_id} for {len(job.feature_names)} features")
                
                # Start processing if not already running
                if not self._active_jobs:
                    self._process_job_queue()
                
                return job.job_id
                
        except Exception as e:
            logger.error(f"❌ Failed to submit computation job: {e}")
            return None
    
    def _process_job_queue(self):
        """Process jobs from the queue"""
        try:
            with self._lock:
                if not self._job_queue:
                    return
                
                # Get next job
                job = self._job_queue.pop(0)
                job.status = "processing"
                job.started_at = datetime.now()
                
                self._active_jobs[job.job_id] = job
                
                # Submit job for processing
                future = self.executor.submit(self._execute_computation_job, job)
                future.add_done_callback(lambda f: self._job_completed(job.job_id, f))
                
                logger.info(f"🚀 Started processing job {job.job_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to process job queue: {e}")
    
    def _execute_computation_job(self, job: ComputationJob) -> bool:
        """Execute a computation job"""
        try:
            logger.info(f"⚙️ Executing job {job.job_id} for {len(job.entity_ids)} entities")
            
            total_features = len(job.feature_names)
            total_entities = len(job.entity_ids)
            completed_features = 0
            
            # Process each entity
            for entity_id in job.entity_ids:
                try:
                    # Compute all features for this entity
                    features = self.feature_store.compute_feature_set(
                        "technical_indicators",  # Use a comprehensive feature set
                        entity_id,
                        job.end_time
                    )
                    
                    completed_features += len(features)
                    
                    # Update progress
                    progress = (completed_features / (total_features * total_entities)) * 100
                    logger.debug(f"📊 Job {job.job_id} progress: {progress:.1f}%")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to compute features for entity {entity_id}: {e}")
                    continue
            
            logger.info(f"✅ Job {job.job_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Job {job.job_id} failed: {e}")
            return False
    
    def _job_completed(self, job_id: str, future):
        """Handle job completion"""
        try:
            with self._lock:
                if job_id in self._active_jobs:
                    job = self._active_jobs[job_id]
                    
                    try:
                        success = future.result()
                        if success:
                            job.status = "completed"
                            job.completed_at = datetime.now()
                            logger.info(f"✅ Job {job_id} completed successfully")
                        else:
                            job.status = "failed"
                            job.error_message = "Job execution failed"
                            logger.error(f"❌ Job {job_id} failed")
                    except Exception as e:
                        job.status = "failed"
                        job.error_message = str(e)
                        logger.error(f"❌ Job {job_id} failed with exception: {e}")
                    
                    # Remove from active jobs
                    del self._active_jobs[job_id]
                    
                    # Process next job in queue
                    if self._job_queue:
                        self._process_job_queue()
                
        except Exception as e:
            logger.error(f"❌ Error handling job completion for {job_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[ComputationJob]:
        """Get the status of a computation job"""
        try:
            with self._lock:
                # Check active jobs
                if job_id in self._active_jobs:
                    return self._active_jobs[job_id]
                
                # Check if job was completed (you might want to persist this)
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get job status for {job_id}: {e}")
            return None
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        try:
            with self._lock:
                return {
                    'active_jobs': len(self._active_jobs),
                    'queued_jobs': len(self._job_queue),
                    'total_workers': self.max_workers,
                    'active_job_ids': list(self._active_jobs.keys()),
                    'queued_job_ids': [job.job_id for job in self._job_queue]
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get pipeline status: {e}")
            return {}
    
    def compute_features_for_entity(self, entity_id: str, timestamp: datetime, 
                                   feature_set_name: str = "technical_indicators") -> Dict[str, float]:
        """Compute features for a specific entity at a specific timestamp"""
        try:
            logger.debug(f"🔍 Computing features for entity {entity_id} at {timestamp}")
            
            # Use the feature store to compute the feature set
            features = self.feature_store.compute_feature_set(
                feature_set_name,
                entity_id,
                timestamp
            )
            
            logger.debug(f"✅ Computed {len(features)} features for {entity_id}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to compute features for entity {entity_id}: {e}")
            return {}
    
    def batch_compute_features(self, entity_ids: List[str], timestamp: datetime,
                              feature_set_name: str = "technical_indicators") -> Dict[str, Dict[str, float]]:
        """Batch compute features for multiple entities"""
        try:
            logger.info(f"📦 Batch computing features for {len(entity_ids)} entities")
            
            results = {}
            
            # Use ThreadPoolExecutor for parallel computation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all computation tasks
                future_to_entity = {
                    executor.submit(
                        self.compute_features_for_entity, 
                        entity_id, 
                        timestamp, 
                        feature_set_name
                    ): entity_id
                    for entity_id in entity_ids
                }
                
                # Collect results
                for future in future_to_entity:
                    entity_id = future_to_entity[future]
                    try:
                        features = future.result(timeout=60)  # 60 second timeout
                        if features:
                            results[entity_id] = features
                    except Exception as e:
                        logger.error(f"❌ Failed to compute features for entity {entity_id}: {e}")
                        results[entity_id] = {}
            
            logger.info(f"✅ Batch computation completed: {len(results)} entities processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch computation failed: {e}")
            return {}
    
    def get_feature_quality_metrics(self, feature_name: str, 
                                   start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get quality metrics for a feature"""
        try:
            # Get feature statistics
            stats = self.feature_store.get_feature_statistics(feature_name, start_time, end_time)
            
            if not stats:
                return {}
            
            # Calculate additional quality metrics
            quality_metrics = {
                'feature_name': feature_name,
                'data_quality': {
                    'completeness': stats.get('count', 0) / max(1, (end_time - start_time).total_seconds() / 3600),  # Assuming hourly data
                    'consistency': 1.0 - (stats.get('stddev', 0) / max(1, abs(stats.get('mean', 1)))),  # Lower stddev = higher consistency
                    'validity': 1.0  # Placeholder - in production you'd check for valid ranges
                },
                'statistical_quality': {
                    'mean': stats.get('mean', 0),
                    'stddev': stats.get('stddev', 0),
                    'min': stats.get('min', 0),
                    'max': stats.get('max', 0),
                    'median': stats.get('median', 0),
                    'iqr': (stats.get('max', 0) - stats.get('min', 0)) / 4  # Approximate IQR
                },
                'period': stats.get('period', {}),
                'overall_score': 0.0
            }
            
            # Calculate overall quality score
            completeness = quality_metrics['data_quality']['completeness']
            consistency = quality_metrics['data_quality']['consistency']
            validity = quality_metrics['data_quality']['validity']
            
            quality_metrics['overall_score'] = (completeness + consistency + validity) / 3
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get quality metrics for {feature_name}: {e}")
            return {}
    
    def cleanup_expired_data(self, retention_days: int = 90):
        """Clean up expired feature data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up expired cache
            self.feature_store.cleanup_expired_cache()
            
            logger.info(f"🧹 Cleaned up data older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired data: {e}")
    
    def shutdown(self):
        """Shutdown the pipeline gracefully"""
        try:
            logger.info("🛑 Shutting down feature computation pipeline...")
            
            # Cancel all active jobs
            with self._lock:
                for job in self._active_jobs.values():
                    job.status = "cancelled"
                    job.error_message = "Pipeline shutdown"
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("✅ Feature computation pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during pipeline shutdown: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
