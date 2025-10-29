#!/usr/bin/env python3
"""
Feature Quality Validation System
Phase 2C: Enhanced Feature Engineering
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
from sklearn.ensemble import IsolationForest

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Metrics for feature quality assessment"""
    feature_name: str
    timestamp: datetime
    completeness_score: float
    consistency_score: float
    reliability_score: float
    overall_score: float
    quality_grade: str  # 'A', 'B', 'C', 'D', 'F'
    issues: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class QualityAlert:
    """Alert for feature quality issues"""
    alert_id: str
    feature_name: str
    timestamp: datetime
    issue_type: str
    severity: str
    message: str
    recommendations: List[str]
    metadata: Dict[str, Any]

class FeatureQualityValidator:
    """Advanced feature quality validation system"""
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        
        # Quality assessment methods
        self.assessment_methods = {
            'completeness': self._assess_completeness,
            'consistency': self._assess_consistency,
            'reliability': self._assess_reliability,
            'outlier_detection': self._assess_outliers,
            'distribution_analysis': self._assess_distribution,
            'temporal_stability': self._assess_temporal_stability
        }
        
        # Quality history storage
        self.quality_history = []
        self.alerts = []
        
        logger.info("🚀 Feature Quality Validator initialized")
    
    def validate_feature_quality(self, feature_name: str, data: pd.Series, 
                                metadata: Dict[str, Any] = None,
                                timestamp: datetime = None) -> QualityMetrics:
        """Comprehensive feature quality validation"""
        try:
            timestamp = timestamp or datetime.now()
            metadata = metadata or {}
            
            # Run all quality assessment methods
            assessment_results = {}
            
            for method_name, method_func in self.assessment_methods.items():
                try:
                    score = method_func(data, metadata)
                    assessment_results[method_name] = score
                except Exception as e:
                    logger.warning(f"⚠️ Quality assessment method {method_name} failed: {e}")
                    assessment_results[method_name] = 0.0
            
            # Calculate component scores
            completeness_score = assessment_results.get('completeness', 0.0)
            consistency_score = assessment_results.get('consistency', 0.0)
            reliability_score = assessment_results.get('reliability', 0.0)
            
            # Calculate overall quality score (weighted average)
            overall_score = (
                completeness_score * 0.3 +
                consistency_score * 0.3 +
                reliability_score * 0.4
            )
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(overall_score)
            
            # Identify issues and generate recommendations
            issues = self._identify_quality_issues(assessment_results, data)
            recommendations = self._generate_quality_recommendations(assessment_results, issues)
            
            # Create quality metrics
            quality_metrics = QualityMetrics(
                feature_name=feature_name,
                timestamp=timestamp,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                reliability_score=reliability_score,
                overall_score=overall_score,
                quality_grade=quality_grade,
                issues=issues,
                recommendations=recommendations,
                metadata={
                    'assessment_results': assessment_results,
                    'data_stats': self._calculate_data_statistics(data),
                    'metadata': metadata
                }
            )
            
            # Store quality history
            self.quality_history.append(quality_metrics)
            
            # Check if alert should be generated
            if overall_score < self.quality_threshold:
                self._generate_quality_alert(quality_metrics)
            
            logger.info(f"✅ Quality validation completed for {feature_name}: score={overall_score:.3f}, grade={quality_grade}")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to validate quality for {feature_name}: {e}")
            return None
    
    def _assess_completeness(self, data: pd.Series, metadata: Dict[str, Any]) -> float:
        """Assess data completeness"""
        try:
            if len(data) == 0:
                return 0.0
            
            # Calculate missing data percentage
            missing_count = data.isna().sum()
            total_count = len(data)
            missing_percentage = missing_count / total_count
            
            # Completeness score (1.0 = no missing data, 0.0 = all missing)
            completeness_score = 1.0 - missing_percentage
            
            # Apply penalties for excessive missing data
            if missing_percentage > 0.5:
                completeness_score *= 0.5
            elif missing_percentage > 0.2:
                completeness_score *= 0.8
            
            return completeness_score
            
        except Exception as e:
            logger.error(f"❌ Completeness assessment failed: {e}")
            return 0.0
    
    def _assess_consistency(self, data: pd.Series, metadata: Dict[str, Any]) -> float:
        """Assess data consistency"""
        try:
            if len(data) < 2:
                return 1.0
            
            # Remove NaN values for consistency analysis
            clean_data = data.dropna()
            if len(clean_data) < 2:
                return 0.0
            
            # Check for data type consistency
            type_consistency = 1.0
            if not pd.api.types.is_numeric_dtype(clean_data):
                # For non-numeric data, check if all values are of the same type
                unique_types = clean_data.apply(type).nunique()
                type_consistency = 1.0 / unique_types
            
            # Check for value range consistency
            range_consistency = 1.0
            if pd.api.types.is_numeric_dtype(clean_data):
                # Check if values are within reasonable bounds
                q1, q3 = clean_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                outlier_percentage = len(outliers) / len(clean_data)
                range_consistency = 1.0 - outlier_percentage
            
            # Check for temporal consistency (if timestamp index)
            temporal_consistency = 1.0
            if isinstance(data.index, pd.DatetimeIndex):
                # Check for gaps in time series
                time_diff = data.index.to_series().diff().dropna()
                if len(time_diff) > 0:
                    median_diff = time_diff.median()
                    large_gaps = time_diff[time_diff > median_diff * 3]
                    gap_percentage = len(large_gaps) / len(time_diff)
                    temporal_consistency = 1.0 - gap_percentage
            
            # Combined consistency score
            consistency_score = (type_consistency + range_consistency + temporal_consistency) / 3
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"❌ Consistency assessment failed: {e}")
            return 0.0
    
    def _assess_reliability(self, data: pd.Series, metadata: Dict[str, Any]) -> float:
        """Assess data reliability"""
        try:
            if len(data) < 10:
                return 0.5  # Insufficient data for reliable assessment
            
            clean_data = data.dropna()
            if len(clean_data) < 10:
                return 0.0
            
            # Statistical reliability indicators
            reliability_scores = []
            
            # 1. Coefficient of variation (lower is better for most financial data)
            if clean_data.mean() != 0:
                cv = clean_data.std() / abs(clean_data.mean())
                cv_score = max(0.0, 1.0 - min(cv, 2.0) / 2.0)  # Normalize to 0-1
                reliability_scores.append(cv_score)
            
            # 2. Skewness (closer to 0 is better for normal distributions)
            skewness = clean_data.skew()
            skewness_score = max(0.0, 1.0 - abs(skewness) / 3.0)  # Normalize to 0-1
            reliability_scores.append(skewness_score)
            
            # 3. Kurtosis (closer to 3 is better for normal distributions)
            kurtosis = clean_data.kurtosis()
            kurtosis_score = max(0.0, 1.0 - abs(kurtosis - 3) / 5.0)  # Normalize to 0-1
            reliability_scores.append(kurtosis_score)
            
            # 4. Data stability (variance of rolling statistics)
            if len(clean_data) >= 20:
                rolling_mean = clean_data.rolling(window=5).mean().dropna()
                rolling_std = clean_data.rolling(window=5).std().dropna()
                
                mean_stability = 1.0 - (rolling_mean.std() / clean_data.std())
                std_stability = 1.0 - (rolling_std.std() / clean_data.std())
                
                reliability_scores.extend([max(0.0, mean_stability), max(0.0, std_stability)])
            
            # 5. Outlier detection using Isolation Forest
            try:
                if len(clean_data) >= 20:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    features = clean_data.values.reshape(-1, 1)
                    iso_forest.fit(features)
                    anomaly_scores = iso_forest.decision_function(features)
                    
                    # Lower anomaly scores indicate more reliable data
                    anomaly_score = 1.0 - np.mean(anomaly_scores)
                    reliability_scores.append(max(0.0, anomaly_score))
            except Exception:
                pass  # Skip if Isolation Forest fails
            
            # Calculate average reliability score
            if reliability_scores:
                reliability_score = np.mean(reliability_scores)
            else:
                reliability_score = 0.5
            
            return reliability_score
            
        except Exception as e:
            logger.error(f"❌ Reliability assessment failed: {e}")
            return 0.0
    
    def _assess_outliers(self, data: pd.Series, metadata: Dict[str, Any]) -> float:
        """Assess outlier presence and severity"""
        try:
            if len(data) < 10:
                return 1.0
            
            clean_data = data.dropna()
            if len(clean_data) < 10:
                return 0.0
            
            # Multiple outlier detection methods
            outlier_scores = []
            
            # 1. IQR method
            q1, q3 = clean_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            iqr_score = 1.0 - (len(iqr_outliers) / len(clean_data))
            outlier_scores.append(iqr_score)
            
            # 2. Z-score method
            z_scores = np.abs(stats.zscore(clean_data))
            z_outliers = clean_data[z_scores > 3]
            z_score = 1.0 - (len(z_outliers) / len(clean_data))
            outlier_scores.append(z_score)
            
            # 3. Modified Z-score method (more robust)
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))
            if mad != 0:
                modified_z_scores = 0.6745 * (clean_data - median) / mad
                modified_z_outliers = clean_data[np.abs(modified_z_scores) > 3.5]
                modified_z_score = 1.0 - (len(modified_z_outliers) / len(clean_data))
                outlier_scores.append(modified_z_score)
            
            # Calculate average outlier score
            outlier_score = np.mean(outlier_scores)
            
            return outlier_score
            
        except Exception as e:
            logger.error(f"❌ Outlier assessment failed: {e}")
            return 0.0
    
    def _assess_distribution(self, data: pd.Series, metadata: Dict[str, Any]) -> float:
        """Assess data distribution characteristics"""
        try:
            if len(data) < 20:
                return 0.5
            
            clean_data = data.dropna()
            if len(clean_data) < 20:
                return 0.0
            
            distribution_scores = []
            
            # 1. Normality test (Shapiro-Wilk)
            try:
                if len(clean_data) <= 5000:  # Shapiro-Wilk has size limit
                    shapiro_stat, shapiro_p = stats.shapiro(clean_data)
                    normality_score = shapiro_p  # Higher p-value = more normal
                    distribution_scores.append(normality_score)
            except Exception:
                pass
            
            # 2. Distribution symmetry
            skewness = clean_data.skew()
            symmetry_score = max(0.0, 1.0 - abs(skewness) / 2.0)
            distribution_scores.append(symmetry_score)
            
            # 3. Distribution peakedness
            kurtosis = clean_data.kurtosis()
            peakedness_score = max(0.0, 1.0 - abs(kurtosis - 3) / 4.0)
            distribution_scores.append(peakedness_score)
            
            # 4. Quantile-quantile plot correlation
            try:
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_data)))
                qq_correlation = np.corrcoef(clean_data.sort_values(), theoretical_quantiles)[0, 1]
                qq_score = max(0.0, qq_correlation) if not np.isnan(qq_correlation) else 0.0
                distribution_scores.append(qq_score)
            except Exception:
                pass
            
            # Calculate average distribution score
            if distribution_scores:
                distribution_score = np.mean(distribution_scores)
            else:
                distribution_score = 0.5
            
            return distribution_score
            
        except Exception as e:
            logger.error(f"❌ Distribution assessment failed: {e}")
            return 0.0
    
    def _assess_temporal_stability(self, data: pd.Series, metadata: Dict[str, Any]) -> float:
        """Assess temporal stability of the feature"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 20:
                return 0.5
            
            clean_data = data.dropna()
            if len(clean_data) < 20:
                return 0.0
            
            stability_scores = []
            
            # 1. Rolling mean stability
            rolling_mean = clean_data.rolling(window=5).mean().dropna()
            if len(rolling_mean) > 1:
                mean_stability = 1.0 - (rolling_mean.std() / clean_data.std())
                stability_scores.append(max(0.0, mean_stability))
            
            # 2. Rolling variance stability
            rolling_var = clean_data.rolling(window=5).var().dropna()
            if len(rolling_var) > 1:
                var_stability = 1.0 - (rolling_var.std() / clean_data.var())
                stability_scores.append(max(0.0, var_stability))
            
            # 3. Trend analysis
            if len(clean_data) >= 20:
                x = np.arange(len(clean_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_data.values)
                
                # Lower slope indicates more stability
                trend_stability = max(0.0, 1.0 - abs(slope) / clean_data.std())
                stability_scores.append(trend_stability)
                
                # Higher R-squared indicates more linear trend (more stable)
                r_squared_stability = r_value ** 2
                stability_scores.append(r_squared_stability)
            
            # 4. Seasonal decomposition (if enough data)
            if len(clean_data) >= 50:
                try:
                    # Simple seasonal decomposition
                    period = min(20, len(clean_data) // 4)
                    seasonal_means = []
                    for i in range(period):
                        seasonal_values = clean_data.iloc[i::period]
                        if len(seasonal_values) > 0:
                            seasonal_means.append(seasonal_values.mean())
                    
                    if len(seasonal_means) > 1:
                        seasonal_stability = 1.0 - (np.std(seasonal_means) / clean_data.std())
                        stability_scores.append(max(0.0, seasonal_stability))
                except Exception:
                    pass
            
            # Calculate average stability score
            if stability_scores:
                stability_score = np.mean(stability_scores)
            else:
                stability_score = 0.5
            
            return stability_score
            
        except Exception as e:
            logger.error(f"❌ Temporal stability assessment failed: {e}")
            return 0.0
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """Determine quality grade based on overall score"""
        if overall_score >= 0.9:
            return 'A'
        elif overall_score >= 0.8:
            return 'B'
        elif overall_score >= 0.7:
            return 'C'
        elif overall_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _identify_quality_issues(self, assessment_results: Dict[str, float], 
                                 data: pd.Series) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        # Check completeness
        if assessment_results.get('completeness', 1.0) < 0.8:
            missing_percentage = (1.0 - assessment_results['completeness']) * 100
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        
        # Check consistency
        if assessment_results.get('consistency', 1.0) < 0.7:
            issues.append("Data consistency issues detected")
        
        # Check reliability
        if assessment_results.get('reliability', 1.0) < 0.6:
            issues.append("Low data reliability")
        
        # Check outliers
        if assessment_results.get('outlier_detection', 1.0) < 0.7:
            issues.append("Excessive outliers detected")
        
        # Check distribution
        if assessment_results.get('distribution_analysis', 1.0) < 0.6:
            issues.append("Non-normal distribution characteristics")
        
        # Check temporal stability
        if assessment_results.get('temporal_stability', 1.0) < 0.6:
            issues.append("Temporal instability detected")
        
        # Check data volume
        if len(data) < 100:
            issues.append("Insufficient data volume for reliable analysis")
        
        return issues
    
    def _generate_quality_recommendations(self, assessment_results: Dict[str, float], 
                                         issues: List[str]) -> List[str]:
        """Generate recommendations for improving feature quality"""
        recommendations = []
        
        # Completeness recommendations
        if assessment_results.get('completeness', 1.0) < 0.8:
            recommendations.append("Implement data imputation strategies")
            recommendations.append("Review data collection pipeline for gaps")
        
        # Consistency recommendations
        if assessment_results.get('consistency', 1.0) < 0.7:
            recommendations.append("Standardize data preprocessing steps")
            recommendations.append("Implement data validation rules")
        
        # Reliability recommendations
        if assessment_results.get('reliability', 1.0) < 0.6:
            recommendations.append("Review feature engineering methodology")
            recommendations.append("Consider feature transformation techniques")
        
        # Outlier recommendations
        if assessment_results.get('outlier_detection', 1.0) < 0.7:
            recommendations.append("Implement outlier detection and handling")
            recommendations.append("Review data source for anomalies")
        
        # Distribution recommendations
        if assessment_results.get('distribution_analysis', 1.0) < 0.6:
            recommendations.append("Consider data normalization techniques")
            recommendations.append("Review feature scaling methods")
        
        # Temporal stability recommendations
        if assessment_results.get('temporal_stability', 1.0) < 0.6:
            recommendations.append("Implement adaptive feature updates")
            recommendations.append("Consider time-varying feature engineering")
        
        # General recommendations
        if len(issues) > 3:
            recommendations.append("Comprehensive data quality review required")
            recommendations.append("Consider feature selection alternatives")
        
        recommendations.append("Monitor quality metrics over time")
        recommendations.append("Implement automated quality checks")
        
        return recommendations
    
    def _calculate_data_statistics(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive data statistics"""
        try:
            clean_data = data.dropna()
            
            if len(clean_data) == 0:
                return {
                    'total_count': len(data),
                    'valid_count': 0,
                    'missing_count': len(data),
                    'missing_percentage': 100.0
                }
            
            stats_dict = {
                'total_count': len(data),
                'valid_count': len(clean_data),
                'missing_count': len(data) - len(clean_data),
                'missing_percentage': ((len(data) - len(clean_data)) / len(data)) * 100,
                'min_value': float(clean_data.min()),
                'max_value': float(clean_data.max()),
                'mean_value': float(clean_data.mean()),
                'median_value': float(clean_data.median()),
                'std_value': float(clean_data.std()),
                'skewness': float(clean_data.skew()),
                'kurtosis': float(clean_data.kurtosis())
            }
            
            # Add percentiles
            percentiles = clean_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            for p, value in percentiles.items():
                stats_dict[f'p{int(p*100)}'] = float(value)
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate data statistics: {e}")
            return {'error': str(e)}
    
    def _generate_quality_alert(self, quality_metrics: QualityMetrics):
        """Generate quality alert"""
        try:
            alert_id = f"quality_{quality_metrics.feature_name}_{quality_metrics.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Determine issue type based on lowest scores
            scores = {
                'completeness': quality_metrics.completeness_score,
                'consistency': quality_metrics.consistency_score,
                'reliability': quality_metrics.reliability_score
            }
            
            primary_issue = min(scores.keys(), key=lambda k: scores[k])
            
            # Determine severity
            if quality_metrics.overall_score < 0.5:
                severity = 'critical'
            elif quality_metrics.overall_score < 0.7:
                severity = 'high'
            else:
                severity = 'medium'
            
            # Create alert
            alert = QualityAlert(
                alert_id=alert_id,
                feature_name=quality_metrics.feature_name,
                timestamp=quality_metrics.timestamp,
                issue_type=primary_issue,
                severity=severity,
                message=f"Quality issue detected in {quality_metrics.feature_name}: {primary_issue} score is {scores[primary_issue]:.3f}",
                recommendations=quality_metrics.recommendations[:3],  # Top 3 recommendations
                metadata={
                    'overall_score': quality_metrics.overall_score,
                    'quality_grade': quality_metrics.quality_grade,
                    'all_scores': scores,
                    'issues': quality_metrics.issues
                }
            )
            
            self.alerts.append(alert)
            logger.warning(f"🚨 QUALITY ALERT: {alert.message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate quality alert: {e}")
    
    def get_quality_summary(self, feature_name: str = None, 
                           start_time: datetime = None,
                           end_time: datetime = None) -> Dict[str, Any]:
        """Get summary of quality validation results"""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.now()
            
            # Filter quality history by time and feature
            filtered_history = [
                quality for quality in self.quality_history
                if start_time <= quality.timestamp <= end_time
                and (feature_name is None or quality.feature_name == feature_name)
            ]
            
            if not filtered_history:
                return {
                    'total_validations': 0,
                    'features_monitored': [],
                    'period': {'start': start_time, 'end': end_time}
                }
            
            # Calculate summary statistics
            overall_scores = [quality.overall_score for quality in filtered_history]
            completeness_scores = [quality.completeness_score for quality in filtered_history]
            consistency_scores = [quality.consistency_score for quality in filtered_history]
            reliability_scores = [quality.reliability_score for quality in filtered_history]
            grades = [quality.quality_grade for quality in filtered_history]
            
            summary = {
                'total_validations': len(filtered_history),
                'features_monitored': list(set([q.feature_name for q in filtered_history])),
                'period': {'start': start_time, 'end': end_time},
                'overall_quality': {
                    'mean_score': np.mean(overall_scores),
                    'min_score': np.min(overall_scores),
                    'max_score': np.max(overall_scores),
                    'std_score': np.std(overall_scores)
                },
                'component_scores': {
                    'completeness': {
                        'mean': np.mean(completeness_scores),
                        'std': np.std(completeness_scores)
                    },
                    'consistency': {
                        'mean': np.mean(consistency_scores),
                        'std': np.std(consistency_scores)
                    },
                    'reliability': {
                        'mean': np.mean(reliability_scores),
                        'std': np.std(reliability_scores)
                    }
                },
                'grade_distribution': {
                    grade: grades.count(grade) for grade in set(grades)
                },
                'recent_alerts': len([a for a in self.alerts if start_time <= a.timestamp <= end_time])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get quality summary: {e}")
            return {'error': str(e)}
    
    def get_feature_quality_trend(self, feature_name: str, 
                                  days: int = 30) -> Dict[str, Any]:
        """Get quality trend for a specific feature"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            # Filter quality history for the feature
            feature_history = [
                quality for quality in self.quality_history
                if quality.feature_name == feature_name
                and quality.timestamp >= start_time
            ]
            
            if not feature_history:
                return {
                    'feature_name': feature_name,
                    'trend_data': [],
                    'trend_direction': 'insufficient_data'
                }
            
            # Sort by timestamp
            feature_history.sort(key=lambda x: x.timestamp)
            
            # Extract trend data
            trend_data = [
                {
                    'timestamp': quality.timestamp.isoformat(),
                    'overall_score': quality.overall_score,
                    'completeness_score': quality.completeness_score,
                    'consistency_score': quality.consistency_score,
                    'reliability_score': quality.reliability_score,
                    'quality_grade': quality.quality_grade
                }
                for quality in feature_history
            ]
            
            # Calculate trend direction
            if len(trend_data) >= 2:
                first_score = trend_data[0]['overall_score']
                last_score = trend_data[-1]['overall_score']
                score_change = last_score - first_score
                
                if score_change > 0.05:
                    trend_direction = 'improving'
                elif score_change < -0.05:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'insufficient_data'
            
            return {
                'feature_name': feature_name,
                'trend_data': trend_data,
                'trend_direction': trend_direction,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get quality trend for {feature_name}: {e}")
            return {'error': str(e)}

# Convenience functions
def validate_feature_quality(feature_name: str, data: pd.Series, 
                            metadata: Dict[str, Any] = None) -> Optional[QualityMetrics]:
    """Validate feature quality using default settings"""
    validator = FeatureQualityValidator()
    return validator.validate_feature_quality(feature_name, data, metadata)

def get_feature_quality_trend(feature_name: str, validator: FeatureQualityValidator, 
                             days: int = 30) -> Dict[str, Any]:
    """Get quality trend for a feature"""
    return validator.get_feature_quality_trend(feature_name, days)
