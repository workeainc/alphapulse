#!/usr/bin/env python3
"""
ML Drift Monitoring System
Detects concept and data drift to trigger automatic retraining
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import existing ML components
from ai.ml_auto_retraining.train_model import MLModelTrainer
from ai.ml_auto_retraining.ml_inference_engine import MLInferenceEngine
from ai.noise_filter_engine import NoiseFilterEngine
from ai.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

@dataclass
class DriftResult:
    """Drift detection result"""
    feature_name: str
    drift_type: str  # 'data_drift', 'concept_drift', 'label_drift'
    drift_score: float
    threshold: float
    is_drift: bool
    p_value: float
    reference_stats: Dict[str, Any]
    current_stats: Dict[str, Any]

@dataclass
class DriftAlert:
    """Drift alert configuration"""
    symbol: str
    regime: str
    model_name: str
    drift_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    features_affected: List[str]
    overall_drift_score: float
    timestamp: datetime
    action_required: str

class DriftMonitor:
    """ML Drift Monitoring System"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.drift_thresholds = {
            'ks_test': 0.15,  # Kolmogorov-Smirnov test threshold
            'psi': 0.25,      # Population Stability Index threshold
            'chi2': 0.05,     # Chi-square test p-value threshold
            'mutual_info': 0.1  # Mutual information threshold
        }
        self.reference_data = {}  # Cache for reference feature distributions
        self.ml_engine = None
        self.trainer = None
        
    async def initialize(self):
        """Initialize drift monitoring components"""
        logger.info("🔧 Initializing Drift Monitor...")
        
        try:
            self.ml_engine = MLInferenceEngine(self.db_config)
            self.trainer = MLModelTrainer(self.db_config)
            await self.trainer.initialize_components()
            
            # Load reference data for all models
            await self._load_reference_data()
            
            logger.info("✅ Drift Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Drift Monitor: {e}")
            raise
    
    async def _load_reference_data(self):
        """Load reference feature distributions for drift detection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get all production models
            cursor.execute("""
                SELECT model_name, regime, symbol, artifact_uri
                FROM ml_models
                WHERE status = 'production'
                ORDER BY created_at DESC
            """)
            
            models = cursor.fetchall()
            
            for model_name, regime, symbol, artifact_uri in models:
                model_key = f"{model_name}_{regime}_{symbol}"
                
                # Load reference data (last 30 days of training data)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                reference_data = await self._get_reference_features(
                    symbol, regime, start_date, end_date
                )
                
                if reference_data is not None and len(reference_data) > 0:
                    self.reference_data[model_key] = reference_data
                    logger.info(f"✅ Loaded reference data for {model_key}: {len(reference_data)} samples")
                
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to load reference data: {e}")
            raise
    
    async def _get_reference_features(self, symbol: str, regime: str, 
                                    start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get reference features for drift detection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Load OHLCV data
            ohlcv_query = """
                SELECT time, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = %s AND time >= %s AND time <= %s
                ORDER BY time ASC
            """
            
            ohlcv_df = pd.read_sql(ohlcv_query, conn, params=(symbol, start_date, end_date))
            
            if len(ohlcv_df) < 100:  # Need sufficient data
                return None
            
            # Create features using existing pipeline
            feature_df = self.ml_engine.create_features_from_ohlcv(ohlcv_df)
            
            # Filter by regime
            regime_classifier = MarketRegimeClassifier(self.db_config)
            await regime_classifier.initialize()
            
            feature_df['regime'] = await regime_classifier.classify_market_regime_batch(feature_df)
            feature_df = feature_df[feature_df['regime'] == regime]
            
            conn.close()
            
            return feature_df
            
        except Exception as e:
            logger.error(f"❌ Failed to get reference features: {e}")
            return None
    
    async def detect_drift(self, symbol: str, regime: str, 
                          model_name: str = 'alphaplus_pattern_classifier') -> List[DriftResult]:
        """Detect drift for a specific model"""
        model_key = f"{model_name}_{regime}_{symbol}"
        
        if model_key not in self.reference_data:
            logger.warning(f"⚠️ No reference data available for {model_key}")
            return []
        
        try:
            # Get current data (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            current_data = await self._get_reference_features(symbol, regime, start_date, end_date)
            
            if current_data is None or len(current_data) < 50:
                logger.warning(f"⚠️ Insufficient current data for drift detection: {model_key}")
                return []
            
            # Detect drift for each feature
            drift_results = []
            reference_data = self.reference_data[model_key]
            
            # Get common features
            common_features = list(set(reference_data.columns) & set(current_data.columns))
            feature_columns = [f for f in common_features if f not in ['time', 'regime']]
            
            for feature in feature_columns:
                try:
                    # Remove NaN values
                    ref_clean = reference_data[feature].dropna()
                    cur_clean = current_data[feature].dropna()
                    
                    if len(ref_clean) < 10 or len(cur_clean) < 10:
                        continue
                    
                    # KS test for data drift
                    ks_stat, ks_pvalue = ks_2samp(ref_clean, cur_clean)
                    
                    # Calculate PSI (Population Stability Index)
                    psi_score = self._calculate_psi(ref_clean, cur_clean)
                    
                    # Determine drift type and score
                    drift_score = max(ks_stat, psi_score)
                    threshold = self.drift_thresholds['ks_test']
                    
                    drift_type = 'data_drift'
                    if ks_pvalue < 0.05 and ks_stat > threshold:
                        drift_type = 'concept_drift'
                    
                    drift_result = DriftResult(
                        feature_name=feature,
                        drift_type=drift_type,
                        drift_score=drift_score,
                        threshold=threshold,
                        is_drift=drift_score > threshold,
                        p_value=ks_pvalue,
                        reference_stats={
                            'mean': float(ref_clean.mean()),
                            'std': float(ref_clean.std()),
                            'count': len(ref_clean)
                        },
                        current_stats={
                            'mean': float(cur_clean.mean()),
                            'std': float(cur_clean.std()),
                            'count': len(cur_clean)
                        }
                    )
                    
                    drift_results.append(drift_result)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to detect drift for feature {feature}: {e}")
                    continue
            
            return drift_results
            
        except Exception as e:
            logger.error(f"❌ Failed to detect drift: {e}")
            return []
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if max_val == min_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to probabilities
            ref_prob = ref_hist / ref_hist.sum()
            cur_prob = cur_hist / cur_hist.sum()
            
            # Calculate PSI
            psi = 0.0
            for i in range(len(ref_prob)):
                if ref_prob[i] > 0 and cur_prob[i] > 0:
                    psi += (cur_prob[i] - ref_prob[i]) * np.log(cur_prob[i] / ref_prob[i])
            
            return float(psi)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate PSI: {e}")
            return 0.0
    
    async def check_drift_alerts(self) -> List[DriftAlert]:
        """Check for drift alerts across all models"""
        alerts = []
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get all production models
            cursor.execute("""
                SELECT model_name, regime, symbol
                FROM ml_models
                WHERE status = 'production'
                ORDER BY created_at DESC
            """)
            
            models = cursor.fetchall()
            conn.close()
            
            for model_name, regime, symbol in models:
                drift_results = await self.detect_drift(symbol, regime, model_name)
                
                if drift_results:
                    # Calculate overall drift score
                    drift_scores = [r.drift_score for r in drift_results if r.is_drift]
                    
                    if drift_scores:
                        overall_score = np.mean(drift_scores)
                        affected_features = [r.feature_name for r in drift_results if r.is_drift]
                        
                        # Determine severity
                        if overall_score > 0.3:
                            severity = 'critical'
                            action = 'immediate_retrain'
                        elif overall_score > 0.2:
                            severity = 'high'
                            action = 'schedule_retrain'
                        elif overall_score > 0.15:
                            severity = 'medium'
                            action = 'monitor_closely'
                        else:
                            severity = 'low'
                            action = 'continue_monitoring'
                        
                        alert = DriftAlert(
                            symbol=symbol,
                            regime=regime,
                            model_name=model_name,
                            drift_type='data_drift',
                            severity=severity,
                            features_affected=affected_features,
                            overall_drift_score=overall_score,
                            timestamp=datetime.now(),
                            action_required=action
                        )
                        
                        alerts.append(alert)
                        
                        # Store alert in database
                        await self._store_drift_alert(alert, drift_results)
                        
                        logger.warning(f"🚨 Drift alert: {symbol} {regime} - {severity} severity")
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to check drift alerts: {e}")
            return []
    
    async def _store_drift_alert(self, alert: DriftAlert, drift_results: List[DriftResult]):
        """Store drift alert in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Store alert
            cursor.execute("""
                INSERT INTO ml_drift_alerts (
                    symbol, regime, model_name, drift_type, severity,
                    features_affected, overall_drift_score, action_required,
                    alert_timestamp, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, NOW()
                )
            """, (
                alert.symbol,
                alert.regime,
                alert.model_name,
                alert.drift_type,
                alert.severity,
                json.dumps(alert.features_affected),
                alert.overall_drift_score,
                alert.action_required,
                alert.timestamp
            ))
            
            # Store detailed drift results
            for result in drift_results:
                cursor.execute("""
                    INSERT INTO ml_drift_details (
                        symbol, regime, model_name, feature_name, drift_type,
                        drift_score, threshold, is_drift, p_value,
                        reference_stats, current_stats, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW()
                    )
                """, (
                    alert.symbol,
                    alert.regime,
                    alert.model_name,
                    result.feature_name,
                    result.drift_type,
                    result.drift_score,
                    result.threshold,
                    result.is_drift,
                    result.p_value,
                    json.dumps(result.reference_stats),
                    json.dumps(result.current_stats)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store drift alert: {e}")
    
    async def trigger_retraining_on_drift(self, alert: DriftAlert):
        """Trigger retraining when drift is detected"""
        try:
            logger.info(f"🔄 Triggering retraining due to drift: {alert.symbol} {alert.regime}")
            
            # Trigger manual retraining
            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)
            
            model_path = await self.trainer.train_model(
                symbol=alert.symbol,
                regime=alert.regime,
                start_date=start_date,
                end_date=end_date,
                model_name=alert.model_name
            )
            
            if model_path:
                logger.info(f"✅ Retraining completed for drift: {alert.symbol} {alert.regime}")
                
                # Update reference data
                await self._update_reference_data(alert.symbol, alert.regime, alert.model_name)
                
            else:
                logger.error(f"❌ Retraining failed for drift: {alert.symbol} {alert.regime}")
                
        except Exception as e:
            logger.error(f"❌ Failed to trigger retraining: {e}")
    
    async def _update_reference_data(self, symbol: str, regime: str, model_name: str):
        """Update reference data after retraining"""
        try:
            model_key = f"{model_name}_{regime}_{symbol}"
            
            # Reload reference data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            reference_data = await self._get_reference_features(symbol, regime, start_date, end_date)
            
            if reference_data is not None and len(reference_data) > 0:
                self.reference_data[model_key] = reference_data
                logger.info(f"✅ Updated reference data for {model_key}")
                
        except Exception as e:
            logger.error(f"❌ Failed to update reference data: {e}")
    
    async def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get drift monitoring summary"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get recent drift alerts
            cursor.execute("""
                SELECT 
                    symbol, regime, severity, overall_drift_score,
                    action_required, alert_timestamp
                FROM ml_drift_alerts
                WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY alert_timestamp DESC
            """, (days,))
            
            alerts = cursor.fetchall()
            
            # Get drift statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
                    COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_alerts,
                    AVG(overall_drift_score) as avg_drift_score
                FROM ml_drift_alerts
                WHERE alert_timestamp >= NOW() - INTERVAL '%s days'
            """, (days,))
            
            stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'period_days': days,
                'total_alerts': stats[0] if stats else 0,
                'critical_alerts': stats[1] if stats else 0,
                'high_alerts': stats[2] if stats else 0,
                'avg_drift_score': float(stats[3]) if stats and stats[3] else 0.0,
                'recent_alerts': [
                    {
                        'symbol': alert[0],
                        'regime': alert[1],
                        'severity': alert[2],
                        'drift_score': float(alert[3]),
                        'action': alert[4],
                        'timestamp': alert[5].isoformat()
                    }
                    for alert in alerts
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get drift summary: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Drift Monitor...")
        
        if self.trainer:
            await self.trainer.cleanup()
        
        logger.info("✅ Drift Monitor cleanup completed")

# CLI interface
async def main():
    """Main CLI function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Drift Monitor')
    parser.add_argument('--action', choices=['check', 'summary', 'retrain'], required=True)
    parser.add_argument('--symbol', help='Symbol for specific drift check')
    parser.add_argument('--regime', help='Regime for specific drift check')
    parser.add_argument('--days', type=int, default=7, help='Days for summary')
    
    args = parser.parse_args()
    
    monitor = DriftMonitor(DB_CONFIG)
    
    try:
        await monitor.initialize()
        
        if args.action == 'check':
            if args.symbol and args.regime:
                # Check specific model
                drift_results = await monitor.detect_drift(args.symbol, args.regime)
                print(json.dumps([{
                    'feature': r.feature_name,
                    'drift_type': r.drift_type,
                    'drift_score': r.drift_score,
                    'is_drift': r.is_drift,
                    'p_value': r.p_value
                } for r in drift_results], indent=2))
            else:
                # Check all models
                alerts = await monitor.check_drift_alerts()
                print(json.dumps([{
                    'symbol': a.symbol,
                    'regime': a.regime,
                    'severity': a.severity,
                    'drift_score': a.overall_drift_score,
                    'action': a.action_required
                } for a in alerts], indent=2))
                
        elif args.action == 'summary':
            summary = await monitor.get_drift_summary(args.days)
            print(json.dumps(summary, indent=2))
            
        elif args.action == 'retrain':
            if not args.symbol or not args.regime:
                logger.error("❌ Symbol and regime required for retraining")
                return
                
            # Create mock alert for retraining
            alert = DriftAlert(
                symbol=args.symbol,
                regime=args.regime,
                model_name='alphaplus_pattern_classifier',
                drift_type='data_drift',
                severity='high',
                features_affected=['volume_ratio', 'rsi_14'],
                overall_drift_score=0.25,
                timestamp=datetime.now(),
                action_required='schedule_retrain'
            )
            
            await monitor.trigger_retraining_on_drift(alert)
            
    except Exception as e:
        logger.error(f"❌ Drift monitor error: {e}")
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
