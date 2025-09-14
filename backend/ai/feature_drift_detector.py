#!/usr/bin/env python3
"""
Feature Drift Detection System
Phase 2C: Enhanced Feature Engineering
Enhanced with standardized interfaces for surgical upgrades
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import asyncio
import asyncpg

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class DriftMetrics:
    """Metrics for feature drift detection"""
    feature_name: str
    timestamp: datetime
    drift_score: float
    drift_type: str  # 'statistical', 'distributional', 'concept'
    severity: str    # 'low', 'medium', 'high', 'critical'
    confidence: float
    details: Dict[str, Any]

@dataclass
class DriftAlert:
    """Alert for detected feature drift"""
    feature_name: str
    drift_type: str
    severity: str
    message: str
    recommendations: List[str]

@dataclass
class StandardizedDriftResult:
    """Standardized drift detection result"""
    psi: float
    ks_p: float
    missing_rate: float
    z_anom: float
    alert: bool
    drift_score: float
    drift_type: str
    severity: str
    confidence: float
    details: Dict[str, Any]

class FeatureDriftDetector:
    """Advanced feature drift detection system with standardized interfaces"""
    
    def __init__(self, reference_window_days: int = 30, 
                 detection_window_days: int = 7,
                 drift_threshold: float = 0.1,
                 db_pool: Optional[asyncpg.Pool] = None):
        self.reference_window_days = reference_window_days
        self.detection_window_days = detection_window_days
        self.drift_threshold = drift_threshold
        self.db_pool = db_pool
        
        # Drift detection methods
        self.detection_methods = {
            'statistical': self._detect_statistical_drift,
            'distributional': self._detect_distributional_drift,
            'concept': self._detect_concept_drift,
            'isolation_forest': self._detect_isolation_forest_drift,
            'pca': self._detect_pca_drift,
            'psi': self._detect_psi_drift
        }
        
        # Reference data storage
        self.reference_data = {}
        self.drift_history = []
        self.alerts = []
        
        # Register interface for standardization
        self._register_interface()
        
        logger.info("🚀 Feature Drift Detector initialized with standardized interfaces")
    
    def _register_interface(self):
        """Register this component's interface for standardization"""
        if self.db_pool:
            try:
                asyncio.create_task(self._register_interface_async())
            except Exception as e:
                logger.warning(f"Could not register interface: {e}")
    
    async def _register_interface_async(self):
        """Register interface asynchronously"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO component_interface_registry 
                    (component_name, interface_type, interface_version, method_name, input_signature, output_signature) 
                    VALUES 
                    ('FeatureDriftDetector', 'drift', '1.0', 'score', 
                     '{"feature_vector": "ndarray", "feature_names": "list"}', 
                     '{"psi": "float", "ks_p": "float", "missing_rate": "float", "z_anom": "float", "alert": "boolean"}')
                    ON CONFLICT (component_name, interface_type, method_name) DO NOTHING;
                """)
        except Exception as e:
            logger.warning(f"Interface registration failed: {e}")
    
    async def score(self, feature_vector: np.ndarray, feature_names: List[str]) -> StandardizedDriftResult:
        """
        Standardized score method for feature drift detection.
        
        Args:
            feature_vector: Feature vector as numpy array
            feature_names: List of feature names
            
        Returns:
            StandardizedDriftResult with drift metrics
        """
        start_time = datetime.now()
        
        try:
            # Convert feature vector to pandas Series for each feature
            feature_data = {}
            for i, feature_name in enumerate(feature_names):
                if i < len(feature_vector):
                    feature_data[feature_name] = pd.Series([feature_vector[i]])
            
            # Calculate drift metrics for each feature
            drift_results = {}
            for feature_name, feature_series in feature_data.items():
                drift_metrics = await self._calculate_feature_drift(feature_name, feature_series)
                drift_results[feature_name] = drift_metrics
            
            # Aggregate results
            psi_scores = [result.psi for result in drift_results.values()]
            ks_p_values = [result.ks_p for result in drift_results.values()]
            missing_rates = [result.missing_rate for result in drift_results.values()]
            z_anom_values = [result.z_anom for result in drift_results.values()]
            
            # Calculate aggregate metrics
            avg_psi = np.mean(psi_scores) if psi_scores else 0.0
            avg_ks_p = np.mean(ks_p_values) if ks_p_values else 1.0
            avg_missing_rate = np.mean(missing_rates) if missing_rates else 0.0
            avg_z_anom = np.mean(z_anom_values) if z_anom_values else 0.0
            
            # Determine overall drift score and alert
            drift_score = max(avg_psi, 1 - avg_ks_p, avg_z_anom)
            alert = drift_score > self.drift_threshold
            
            # Determine severity
            if drift_score > 0.3:
                severity = 'critical'
            elif drift_score > 0.2:
                severity = 'high'
            elif drift_score > 0.1:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Calculate confidence based on data quality
            confidence = max(0.0, 1.0 - avg_missing_rate - (drift_score * 0.5))
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._track_interface_performance('FeatureDriftDetector', 'drift', 'score', processing_time, True)
            
            return StandardizedDriftResult(
                psi=avg_psi,
                ks_p=avg_ks_p,
                missing_rate=avg_missing_rate,
                z_anom=avg_z_anom,
                alert=alert,
                drift_score=drift_score,
                drift_type='feature',
                severity=severity,
                confidence=confidence,
                details={
                    'feature_results': {name: result.__dict__ for name, result in drift_results.items()},
                    'processing_time_ms': processing_time
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._track_interface_performance('FeatureDriftDetector', 'drift', 'score', processing_time, False)
            logger.error(f"❌ Feature drift detection failed: {e}")
            
            # Return default result on error
            return StandardizedDriftResult(
                psi=0.0,
                ks_p=1.0,
                missing_rate=0.0,
                z_anom=0.0,
                alert=False,
                drift_score=0.0,
                drift_type='feature',
                severity='low',
                confidence=0.5,
                details={'error': str(e)}
            )
    
    async def _calculate_feature_drift(self, feature_name: str, current_data: pd.Series) -> StandardizedDriftResult:
        """Calculate drift for a single feature"""
        try:
            # Get reference data
            reference = self.reference_data.get(feature_name)
            if reference is None:
                # No reference data available
                return StandardizedDriftResult(
                    psi=0.0, ks_p=1.0, missing_rate=0.0, z_anom=0.0,
                    alert=False, drift_score=0.0, drift_type='feature',
                    severity='low', confidence=0.5, details={'no_reference': True}
                )
            
            # Calculate PSI
            psi = self._calculate_psi(reference['data'], current_data)
            
            # Calculate KS test
            ks_stat, ks_p = self._calculate_ks_test(reference['data'], current_data)
            
            # Calculate missing rate
            missing_rate = current_data.isnull().sum() / len(current_data)
            
            # Calculate Z-score anomaly
            z_anom = self._calculate_z_anomaly(current_data, reference['mean'], reference['std'])
            
            # Determine drift score
            drift_score = max(psi, 1 - ks_p, abs(z_anom))
            alert = drift_score > self.drift_threshold
            
            # Determine severity
            if drift_score > 0.3:
                severity = 'critical'
            elif drift_score > 0.2:
                severity = 'high'
            elif drift_score > 0.1:
                severity = 'medium'
            else:
                severity = 'low'
            
            return StandardizedDriftResult(
                psi=psi,
                ks_p=ks_p,
                missing_rate=missing_rate,
                z_anom=z_anom,
                alert=alert,
                drift_score=drift_score,
                drift_type='feature',
                severity=severity,
                confidence=1.0 - missing_rate,
                details={
                    'feature_name': feature_name,
                    'ks_statistic': ks_stat,
                    'reference_stats': reference
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating drift for {feature_name}: {e}")
            return StandardizedDriftResult(
                psi=0.0, ks_p=1.0, missing_rate=0.0, z_anom=0.0,
                alert=False, drift_score=0.0, drift_type='feature',
                severity='low', confidence=0.5, details={'error': str(e)}
            )
    
    async def _track_interface_performance(self, component_name: str, interface_type: str, 
                                         method_name: str, execution_time_ms: float, success: bool):
        """Track interface performance for standardization"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO interface_performance_metrics 
                    (component_name, interface_type, method_name, execution_time_ms, success_rate, error_count) 
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, component_name, interface_type, method_name, execution_time_ms, 
                    1.0 if success else 0.0, 0 if success else 1)
        except Exception as e:
            logger.warning(f"Performance tracking failed: {e}")
    
    async def store_interface_result(self, signal_id: str, component_name: str, 
                                   interface_type: str, input_data: Dict, output_data: Dict,
                                   confidence_score: float, processing_time_ms: float):
        """Store standardized interface result"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO standardized_interface_results 
                    (signal_id, component_name, interface_type, input_data, output_data, 
                     confidence_score, processing_time_ms) 
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, signal_id, component_name, interface_type, input_data, output_data,
                     confidence_score, processing_time_ms)
        except Exception as e:
            logger.warning(f"Interface result storage failed: {e}")
    
    def update_reference_data(self, feature_name: str, data: pd.Series, 
                             timestamp: datetime = None) -> bool:
        """Update reference data for a feature"""
        try:
            timestamp = timestamp or datetime.now()
            
            if feature_name not in self.reference_data:
                self.reference_data[feature_name] = []
            
            # Store reference data point
            self.reference_data[feature_name].append({
                'timestamp': timestamp,
                'value': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'percentiles': data.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'data': data
            })
            
            # Keep only recent reference data
            cutoff_time = timestamp - timedelta(days=self.reference_window_days)
            self.reference_data[feature_name] = [
                ref for ref in self.reference_data[feature_name]
                if ref['timestamp'] >= cutoff_time
            ]
            
            logger.debug(f"✅ Updated reference data for {feature_name}: {len(self.reference_data[feature_name])} points")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update reference data for {feature_name}: {e}")
            return False
    
    def detect_drift(self, feature_name: str, current_data: pd.Series, 
                     timestamp: datetime = None) -> Optional[DriftMetrics]:
        """Detect drift for a specific feature"""
        try:
            timestamp = timestamp or datetime.now()
            
            if feature_name not in self.reference_data or not self.reference_data[feature_name]:
                logger.warning(f"⚠️ No reference data available for {feature_name}")
                return None
            
            # Get latest reference data
            reference = self.reference_data[feature_name][-1]
            
            # Run all drift detection methods
            drift_scores = {}
            drift_types = {}
            
            for method_name, method_func in self.detection_methods.items():
                try:
                    score, drift_type = method_func(reference, current_data)
                    drift_scores[method_name] = score
                    drift_types[method_name] = drift_type
                except Exception as e:
                    logger.warning(f"⚠️ Drift detection method {method_name} failed: {e}")
                    continue
            
            if not drift_scores:
                logger.warning(f"⚠️ All drift detection methods failed for {feature_name}")
                return None
            
            # Aggregate drift scores
            overall_score = np.mean(list(drift_scores.values()))
            primary_method = max(drift_scores.keys(), key=lambda k: drift_scores[k])
            primary_type = drift_types[primary_method]
            
            # Determine severity
            severity = self._determine_severity(overall_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(drift_scores)
            
            # Create drift metrics
            drift_metrics = DriftMetrics(
                feature_name=feature_name,
                timestamp=timestamp,
                drift_score=overall_score,
                drift_type=primary_type,
                severity=severity,
                confidence=confidence,
                details={
                    'method_scores': drift_scores,
                    'method_types': drift_types,
                    'primary_method': primary_method,
                    'reference_stats': reference
                }
            )
            
            # Store drift history
            self.drift_history.append(drift_metrics)
            
            # Check if alert should be generated
            if overall_score > self.drift_threshold:
                self._generate_drift_alert(drift_metrics)
            
            logger.info(f"✅ Drift detection completed for {feature_name}: score={overall_score:.3f}, severity={severity}")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to detect drift for {feature_name}: {e}")
            return None
    
    def _detect_statistical_drift(self, reference: Dict[str, Any], 
                                  current_data: pd.Series) -> Tuple[float, str]:
        """Detect statistical drift using basic statistics"""
        try:
            # Calculate current statistics
            current_mean = current_data.mean()
            current_std = current_data.std()
            
            # Calculate drift scores
            mean_drift = abs(current_mean - reference['value']) / (reference['std'] + 1e-8)
            std_drift = abs(current_std - reference['std']) / (reference['std'] + 1e-8)
            
            # Combined drift score
            drift_score = (mean_drift + std_drift) / 2
            
            return drift_score, 'statistical'
            
        except Exception as e:
            logger.error(f"❌ Statistical drift detection failed: {e}")
            return 0.0, 'statistical'
    
    def _detect_distributional_drift(self, reference: Dict[str, Any], 
                                     current_data: pd.Series) -> Tuple[float, str]:
        """Detect distributional drift using KS test"""
        try:
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                reference['data'], current_data
            )
            
            # Convert to drift score (0 = no drift, 1 = complete drift)
            drift_score = 1 - p_value
            
            return drift_score, 'distributional'
            
        except Exception as e:
            logger.error(f"❌ Distributional drift detection failed: {e}")
            return 0.0, 'distributional'
    
    def _detect_concept_drift(self, reference: Dict[str, Any], 
                              current_data: pd.Series) -> Tuple[float, str]:
        """Detect concept drift using advanced metrics"""
        try:
            # Calculate distribution similarity metrics
            reference_percentiles = pd.Series(reference['percentiles'])
            current_percentiles = current_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            
            # Percentile drift
            percentile_drift = np.mean(np.abs(
                current_percentiles - reference_percentiles
            )) / (reference['std'] + 1e-8)
            
            # Shape drift (skewness and kurtosis)
            current_skew = current_data.skew()
            current_kurt = current_data.kurtosis()
            
            skew_drift = abs(current_skew - reference['skewness']) / (abs(reference['skewness']) + 1e-8)
            kurt_drift = abs(current_kurt - reference['kurtosis']) / (abs(reference['kurtosis']) + 1e-8)
            
            # Combined concept drift score
            drift_score = (percentile_drift + skew_drift + kurt_drift) / 3
            
            return drift_score, 'concept'
            
        except Exception as e:
            logger.error(f"❌ Concept drift detection failed: {e}")
            return 0.0, 'concept'
    
    def _detect_isolation_forest_drift(self, reference: Dict[str, Any], 
                                       current_data: pd.Series) -> Tuple[float, str]:
        """Detect drift using Isolation Forest anomaly detection"""
        try:
            # Prepare data for isolation forest
            reference_features = np.column_stack([
                reference['data'].values,
                reference['data'].rolling(window=5).mean().fillna(method='bfill').values,
                reference['data'].rolling(window=5).std().fillna(method='bfill').values
            ])
            
            current_features = np.column_stack([
                current_data.values,
                current_data.rolling(window=5).mean().fillna(method='bfill').values,
                current_data.rolling(window=5).std().fillna(method='bfill').values
            ])
            
            # Train isolation forest on reference data
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(reference_features)
            
            # Predict anomalies in current data
            anomaly_scores = iso_forest.decision_function(current_features)
            
            # Convert to drift score (higher anomaly score = more drift)
            drift_score = 1 - np.mean(anomaly_scores)
            
            return drift_score, 'isolation_forest'
            
        except Exception as e:
            logger.error(f"❌ Isolation Forest drift detection failed: {e}")
            return 0.0, 'isolation_forest'
    
    def _detect_pca_drift(self, reference: Dict[str, Any], 
                           current_data: pd.Series) -> Tuple[float, str]:
        """Detect drift using PCA-based reconstruction error"""
        try:
            # Prepare data for PCA
            reference_features = np.column_stack([
                reference['data'].values,
                reference['data'].rolling(window=5).mean().fillna(method='bfill').values,
                reference['data'].rolling(window=5).std().fillna(method='bfill').values
            ])
            
            current_features = np.column_stack([
                current_data.values,
                current_data.rolling(window=5).mean().fillna(method='bfill').values,
                current_data.rolling(window=5).std().fillna(method='bfill').values
            ])
            
            # Remove NaN values
            reference_features = reference_features[~np.isnan(reference_features).any(axis=1)]
            current_features = current_features[~np.isnan(current_features).any(axis=1)]
            
            if len(reference_features) < 2 or len(current_features) < 2:
                return 0.0, 'pca'
            
            # Fit PCA on reference data
            pca = PCA(n_components=min(2, reference_features.shape[1]))
            pca.fit(reference_features)
            
            # Calculate reconstruction error for current data
            current_reconstructed = pca.inverse_transform(
                pca.transform(current_features)
            )
            
            reconstruction_error = np.mean(
                np.sqrt(np.sum((current_features - current_reconstructed) ** 2, axis=1))
            )
            
            # Normalize error to get drift score
            drift_score = min(1.0, reconstruction_error / (reference['std'] + 1e-8))
            
            return drift_score, 'pca'
            
        except Exception as e:
            logger.error(f"❌ PCA drift detection failed: {e}")
            return 0.0, 'pca'
    
    def _detect_psi_drift(self, reference: Dict[str, Any], 
                           current_data: pd.Series) -> Tuple[float, str]:
        """Detect drift using Population Stability Index (PSI)"""
        try:
            # Create bins for PSI calculation (10 equal-width bins)
            reference_data = reference['data']
            min_val = min(reference_data.min(), current_data.min())
            max_val = max(reference_data.max(), current_data.max())
            
            if max_val == min_val:
                return 0.0, 'psi'
            
            # Create 10 bins
            bins = np.linspace(min_val, max_val, 11)
            
            # Calculate expected (reference) distribution
            expected_counts, _ = np.histogram(reference_data, bins=bins)
            expected_pct = expected_counts / len(reference_data)
            
            # Calculate actual (current) distribution
            actual_counts, _ = np.histogram(current_data, bins=bins)
            actual_pct = actual_counts / len(current_data)
            
            # Handle zero probabilities (add small epsilon)
            epsilon = 1e-10
            expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
            actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
            
            # Calculate PSI: Σ(actual % - expected %) * ln(actual % / expected %)
            psi_score = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            
            # Normalize PSI to 0-1 range (typical PSI > 0.25 indicates significant drift)
            # PSI interpretation: < 0.1 (insignificant), 0.1-0.25 (minor), > 0.25 (significant)
            normalized_psi = min(1.0, psi_score / 0.5)  # Normalize to 0-1 range
            
            return normalized_psi, 'psi'
            
        except Exception as e:
            logger.error(f"❌ PSI drift detection failed: {e}")
            return 0.0, 'psi'
    
    def _determine_severity(self, drift_score: float) -> str:
        """Determine severity level based on drift score"""
        if drift_score < 0.1:
            return 'low'
        elif drift_score < 0.3:
            return 'medium'
        elif drift_score < 0.6:
            return 'high'
        else:
            return 'critical'
    
    def _calculate_confidence(self, method_scores: Dict[str, float]) -> float:
        """Calculate confidence in drift detection based on method agreement"""
        if not method_scores:
            return 0.0
        
        # Higher confidence when methods agree
        scores = list(method_scores.values())
        std_score = np.std(scores)
        mean_score = np.mean(scores)
        
        # Confidence decreases with standard deviation
        confidence = max(0.0, 1.0 - (std_score / (mean_score + 1e-8)))
        
        return confidence
    
    def _generate_drift_alert(self, drift_metrics: DriftMetrics):
        """Generate drift alert"""
        try:
            alert_id = f"drift_{drift_metrics.feature_name}_{drift_metrics.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Generate recommendations based on severity
            recommendations = self._generate_recommendations(drift_metrics)
            
            # Create alert
            alert = DriftAlert(
                feature_name=drift_metrics.feature_name,
                drift_type=drift_metrics.drift_type,
                severity=drift_metrics.severity,
                message=f"Feature drift detected in {drift_metrics.feature_name}: {drift_metrics.drift_type} drift with {drift_metrics.severity} severity",
                recommendations=recommendations,
            )
            
            self.alerts.append(alert)
            logger.warning(f"🚨 DRIFT ALERT: {alert.message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate drift alert: {e}")
    
    def _generate_recommendations(self, drift_metrics: DriftMetrics) -> List[str]:
        """Generate recommendations based on drift severity and type"""
        recommendations = []
        
        if drift_metrics.severity in ['high', 'critical']:
            recommendations.append("Immediate investigation required")
            recommendations.append("Consider retraining models using this feature")
            recommendations.append("Review data pipeline for potential issues")
        
        if drift_metrics.drift_type == 'statistical':
            recommendations.append("Check for data quality issues or preprocessing changes")
            recommendations.append("Verify data source consistency")
        
        elif drift_metrics.drift_type == 'distributional':
            recommendations.append("Review feature engineering pipeline")
            recommendations.append("Check for changes in underlying data distribution")
        
        elif drift_metrics.drift_type == 'concept':
            recommendations.append("Market conditions may have changed")
            recommendations.append("Consider updating feature definitions")
        
        recommendations.append("Monitor drift metrics over time")
        recommendations.append("Update reference data if drift is expected")
        
        return recommendations
    
    def get_drift_summary(self, feature_name: str = None, 
                          start_time: datetime = None,
                          end_time: datetime = None) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.now()
            
            # Filter drift history by time and feature
            filtered_history = [
                drift for drift in self.drift_history
                if start_time <= drift.timestamp <= end_time
                and (feature_name is None or drift.feature_name == feature_name)
            ]
            
            if not filtered_history:
                return {
                    'total_drifts': 0,
                    'features_monitored': list(self.reference_data.keys()),
                    'period': {'start': start_time, 'end': end_time}
                }
            
            # Calculate summary statistics
            drift_scores = [drift.drift_score for drift in filtered_history]
            severities = [drift.severity for drift in filtered_history]
            drift_types = [drift.drift_type for drift in filtered_history]
            
            summary = {
                'total_drifts': len(filtered_history),
                'features_monitored': list(self.reference_data.keys()),
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
    
    def get_feature_health_score(self, feature_name: str) -> float:
        """Get overall health score for a feature (0 = unhealthy, 1 = healthy)"""
        try:
            if feature_name not in self.reference_data:
                return 0.0
            
            # Get recent drift metrics
            recent_drifts = [
                drift for drift in self.drift_history
                if drift.feature_name == feature_name
                and drift.timestamp >= datetime.now() - timedelta(days=7)
            ]
            
            if not recent_drifts:
                return 1.0  # No recent drift detected
            
            # Calculate health score based on drift severity and frequency
            avg_drift_score = np.mean([drift.drift_score for drift in recent_drifts])
            drift_frequency = len(recent_drifts) / 7  # Drifts per day
            
            # Health score decreases with drift score and frequency
            health_score = max(0.0, 1.0 - (avg_drift_score * 0.7 + drift_frequency * 0.3))
            
            return health_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate health score for {feature_name}: {e}")
            return 0.0
    
    def cleanup_old_data(self, max_history_days: int = 90):
        """Clean up old drift history and alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_history_days)
            
            # Clean drift history
            self.drift_history = [
                drift for drift in self.drift_history
                if drift.timestamp >= cutoff_time
            ]
            
            # Clean alerts
            self.alerts = [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]
            
            logger.info(f"🧹 Cleaned up drift data older than {max_history_days} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")

# Convenience functions
def detect_feature_drift(feature_name: str, reference_data: pd.Series, 
                         current_data: pd.Series) -> Optional[DriftMetrics]:
    """Detect drift for a feature using default settings"""
    detector = FeatureDriftDetector()
    detector.update_reference_data(feature_name, reference_data)
    return detector.detect_drift(feature_name, current_data)

def get_feature_health_score(feature_name: str, detector: FeatureDriftDetector) -> float:
    """Get health score for a feature"""
    return detector.get_feature_health_score(feature_name)

def calculate_psi_drift(reference_data: pd.Series, current_data: pd.Series) -> float:
    """Calculate Population Stability Index (PSI) for drift detection"""
    detector = FeatureDriftDetector()
    detector.update_reference_data("temp_feature", reference_data)
    drift_metrics = detector.detect_drift("temp_feature", current_data)
    return drift_metrics.drift_score if drift_metrics else 0.0
