#!/usr/bin/env python3
"""
Timeframe Range Verification Service for AlphaPulse
Phase 5: Timeframe Range Verification
Handles range validation, edge case detection, and streaming mode handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RangeValidationLevel(Enum):
    """Validation levels for timeframe ranges"""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"

class StreamingMode(Enum):
    """Streaming modes for real-time data"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"

@dataclass
class RangeVerificationConfig:
    """Configuration for timeframe range verification"""
    validation_level: RangeValidationLevel = RangeValidationLevel.NORMAL
    streaming_mode: StreamingMode = StreamingMode.BATCH
    min_range_coverage: float = 0.95
    max_gap_threshold: timedelta = timedelta(hours=1)
    allow_future_data: bool = False
    timezone_aware: bool = True
    validate_weekends: bool = True
    validate_holidays: bool = False

@dataclass
class RangeVerificationResult:
    """Result of timeframe range verification"""
    is_valid: bool
    coverage_score: float
    gap_analysis: Dict[str, Any]
    edge_cases: List[Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    verification_report: Dict[str, Any]

class TimeframeRangeVerificationService:
    """Comprehensive timeframe range verification service"""
    
    def __init__(self):
        self.trading_hours = {
            'crypto': {'start': '00:00', 'end': '23:59', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']},
            'forex': {'start': '22:00', 'end': '22:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']},
            'stocks': {'start': '13:30', 'end': '20:00', 'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']}
        }
        
        self.stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'range_issues_detected': 0,
            'total_processing_time_ms': 0.0,
            'avg_verification_time_ms': 0.0
        }
        
        logger.info("🚀 Timeframe Range Verification Service initialized")
    
    def verify_timeframe_ranges(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        config: Optional[RangeVerificationConfig] = None,
        market_type: str = 'crypto'
    ) -> RangeVerificationResult:
        """Verify timeframe ranges for completeness and validity"""
        start_time = datetime.now()
        
        try:
            if config is None:
                config = RangeVerificationConfig()
            
            verification_report = {
                'timeframes_verified': len(data_dict),
                'market_type': market_type,
                'validation_level': config.validation_level.value,
                'streaming_mode': config.streaming_mode.value,
                'timestamp': datetime.now().isoformat()
            }
            
            warnings = []
            errors = []
            edge_cases = []
            recommendations = []
            
            # Validate each timeframe range
            for timeframe, data in data_dict.items():
                if data.empty:
                    warnings.append(f"⚠️ {timeframe}: Empty dataset")
                    continue
                
                timeframe_result = self._verify_single_timeframe_range(
                    data, timeframe, config, market_type
                )
                
                warnings.extend([f"{timeframe}: {w}" for w in timeframe_result['warnings']])
                edge_cases.extend([{**ec, 'timeframe': timeframe} for ec in timeframe_result['edge_cases']])
                recommendations.extend([f"{timeframe}: {r}" for r in timeframe_result['recommendations']])
            
            # Analyze gaps and calculate coverage
            gap_analysis = self._analyze_cross_timeframe_gaps(data_dict, config)
            coverage_score = self._calculate_overall_coverage(data_dict, config)
            
            # Generate recommendations
            recommendations.extend(self._generate_range_recommendations(
                data_dict, gap_analysis, coverage_score, config
            ))
            
            # Determine overall validity
            is_valid = self._determine_overall_validity(
                coverage_score, warnings, errors, config
            )
            
            # Update report and stats
            verification_report.update({
                'coverage_score': coverage_score,
                'gap_analysis': gap_analysis,
                'edge_cases_count': len(edge_cases),
                'warnings_count': len(warnings),
                'recommendations_count': len(recommendations)
            })
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(is_valid, processing_time)
            
            logger.info(f"✅ Timeframe range verification completed: {len(data_dict)} timeframes in {processing_time:.2f}ms")
            
            return RangeVerificationResult(
                is_valid=is_valid,
                coverage_score=coverage_score,
                gap_analysis=gap_analysis,
                edge_cases=edge_cases,
                recommendations=recommendations,
                warnings=warnings,
                errors=errors,
                verification_report=verification_report
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(False, processing_time)
            error_msg = f"Timeframe range verification failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return RangeVerificationResult(
                is_valid=False,
                coverage_score=0.0,
                gap_analysis={},
                edge_cases=[],
                recommendations=[],
                warnings=[],
                errors=[error_msg],
                verification_report={'error': error_msg}
            )
    
    def _verify_single_timeframe_range(
        self, 
        data: pd.DataFrame, 
        timeframe: str, 
        config: RangeVerificationConfig,
        market_type: str
    ) -> Dict[str, Any]:
        """Verify range for a single timeframe"""
        result = {'warnings': [], 'edge_cases': [], 'recommendations': []}
        
        try:
            expected_interval = self._parse_timeframe_interval(timeframe)
            if expected_interval is None:
                result['warnings'].append(f"Unsupported timeframe format: {timeframe}")
                return result
            
            start_time = data.index.min()
            end_time = data.index.max()
            current_time = datetime.now()
            
            # Check for future data
            if not config.allow_future_data and end_time > current_time:
                result['warnings'].append(f"Data extends into future: {end_time}")
                result['edge_cases'].append({
                    'type': 'future_data',
                    'timestamp': end_time,
                    'severity': 'medium'
                })
            
            # Check data continuity
            continuity_issues = self._check_data_continuity(data, expected_interval, config)
            result['warnings'].extend(continuity_issues['warnings'])
            result['edge_cases'].extend(continuity_issues['edge_cases'])
            
            # Check for anomalies
            anomaly_issues = self._detect_range_anomalies(data, timeframe, config)
            result['warnings'].extend(anomaly_issues['warnings'])
            result['edge_cases'].extend(anomaly_issues['edge_cases'])
            
        except Exception as e:
            result['warnings'].append(f"Error during verification: {str(e)}")
        
        return result
    
    def _parse_timeframe_interval(self, timeframe: str) -> Optional[timedelta]:
        """Parse timeframe string to get expected interval"""
        try:
            if timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                return timedelta(minutes=minutes)
            elif timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                return timedelta(hours=hours)
            elif timeframe.endswith('d'):
                days = int(timeframe[:-1])
                return timedelta(days=days)
            elif timeframe.endswith('w'):
                weeks = int(timeframe[:-1])
                return timedelta(weeks=weeks)
            elif timeframe.endswith('M'):
                months = int(timeframe[:-1])
                return timedelta(days=months * 30)
            else:
                return None
        except ValueError:
            return None
    
    def _check_data_continuity(
        self, 
        data: pd.DataFrame, 
        expected_interval: timedelta,
        config: RangeVerificationConfig
    ) -> Dict[str, Any]:
        """Check for gaps and continuity issues in data"""
        result = {'warnings': [], 'edge_cases': [], 'recommendations': []}
        
        try:
            sorted_data = data.sort_index()
            start_time = sorted_data.index[0]
            end_time = sorted_data.index[-1]
            
            expected_timestamps = pd.date_range(
                start=start_time,
                end=end_time,
                freq=expected_interval
            )
            
            actual_timestamps = set(sorted_data.index)
            missing_timestamps = set(expected_timestamps) - actual_timestamps
            
            if missing_timestamps:
                missing_count = len(missing_timestamps)
                missing_percentage = (missing_count / len(expected_timestamps)) * 100
                
                # Always report gaps when they exist
                result['warnings'].append(
                    f"Data gaps detected: {missing_count} missing timestamps ({missing_percentage:.1f}%)"
                )
                
                # Check if gaps are significant
                if missing_percentage > (1 - config.min_range_coverage) * 100:
                    result['warnings'].append(
                        f"Significant data gaps detected: {missing_count} missing timestamps ({missing_percentage:.1f}%)"
                    )
                
                largest_gap = self._find_largest_gap(sorted_data, expected_interval)
                if largest_gap > config.max_gap_threshold:
                    result['edge_cases'].append({
                        'type': 'large_gap',
                        'gap_duration': largest_gap,
                        'severity': 'high' if largest_gap > timedelta(hours=24) else 'medium'
                    })
                    result['warnings'].append(f"Large gap detected: {largest_gap}")
                
                result['recommendations'].append("Consider gap filling or data recovery")
            
        except Exception as e:
            result['warnings'].append(f"Error checking continuity: {str(e)}")
        
        return result
    
    def _find_largest_gap(self, data: pd.DataFrame, expected_interval: timedelta) -> timedelta:
        """Find the largest gap in the data"""
        if len(data) < 2:
            return timedelta(0)
        
        sorted_data = data.sort_index()
        gaps = []
        
        for i in range(len(sorted_data) - 1):
            current_time = sorted_data.index[i]
            next_time = sorted_data.index[i + 1]
            gap = next_time - current_time
            
            if gap > expected_interval:
                gaps.append(gap)
        
        return max(gaps) if gaps else timedelta(0)
    
    def _detect_range_anomalies(
        self, 
        data: pd.DataFrame, 
        timeframe: str,
        config: RangeVerificationConfig
    ) -> Dict[str, Any]:
        """Detect anomalies in timeframe data"""
        result = {'warnings': [], 'edge_cases': []}
        
        try:
            # Check for duplicate timestamps
            duplicate_timestamps = data.index.duplicated()
            if duplicate_timestamps.any():
                duplicate_count = duplicate_timestamps.sum()
                result['warnings'].append(f"Duplicate timestamps detected: {duplicate_count}")
                result['edge_cases'].append({
                    'type': 'duplicate_timestamps',
                    'count': duplicate_count,
                    'severity': 'medium'
                })
            
            # Check for timestamp ordering issues
            if not data.index.is_monotonic_increasing:
                result['warnings'].append("Timestamps are not in chronological order")
                result['edge_cases'].append({
                    'type': 'unordered_timestamps',
                    'severity': 'high'
                })
            
            # Check for extreme price movements
            if 'close' in data.columns:
                price_changes = data['close'].pct_change().abs()
                extreme_moves = price_changes > 0.1
                
                if extreme_moves.any():
                    extreme_count = extreme_moves.sum()
                    result['warnings'].append(f"Extreme price movements detected: {extreme_count} instances")
                    result['edge_cases'].append({
                        'type': 'extreme_price_moves',
                        'count': extreme_count,
                        'severity': 'medium'
                    })
            
        except Exception as e:
            result['warnings'].append(f"Error detecting anomalies: {str(e)}")
        
        return result
    
    def _analyze_cross_timeframe_gaps(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        config: RangeVerificationConfig
    ) -> Dict[str, Any]:
        """Analyze gaps across different timeframes"""
        gap_analysis = {
            'timeframe_coverage': {},
            'cross_timeframe_gaps': [],
            'overall_coverage': 0.0
        }
        
        try:
            for timeframe, data in data_dict.items():
                if data.empty:
                    gap_analysis['timeframe_coverage'][timeframe] = 0.0
                    continue
                
                expected_interval = self._parse_timeframe_interval(timeframe)
                if expected_interval:
                    start_time = data.index.min()
                    end_time = data.index.max()
                    expected_points = int((end_time - start_time) / expected_interval) + 1
                    actual_points = len(data)
                    coverage = actual_points / expected_points if expected_points > 0 else 0.0
                    gap_analysis['timeframe_coverage'][timeframe] = coverage
            
            coverage_values = list(gap_analysis['timeframe_coverage'].values())
            if coverage_values:
                gap_analysis['overall_coverage'] = np.mean(coverage_values)
            
        except Exception as e:
            logger.warning(f"Error analyzing cross-timeframe gaps: {e}")
        
        return gap_analysis
    
    def _calculate_overall_coverage(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        config: RangeVerificationConfig
    ) -> float:
        """Calculate overall coverage score across all timeframes"""
        try:
            if not data_dict:
                return 0.0
            
            coverage_scores = []
            
            for timeframe, data in data_dict.items():
                if data.empty:
                    coverage_scores.append(0.0)
                    continue
                
                expected_interval = self._parse_timeframe_interval(timeframe)
                if expected_interval:
                    start_time = data.index.min()
                    end_time = data.index.max()
                    expected_points = int((end_time - start_time) / expected_interval) + 1
                    actual_points = len(data)
                    coverage = actual_points / expected_points if expected_points > 0 else 0.0
                    coverage_scores.append(coverage)
                else:
                    coverage_scores.append(0.0)
            
            if coverage_scores:
                return np.mean(coverage_scores)
            
        except Exception as e:
            logger.warning(f"Error calculating overall coverage: {e}")
        
        return 0.0
    
    def _generate_range_recommendations(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        gap_analysis: Dict[str, Any],
        coverage_score: float,
        config: RangeVerificationConfig
    ) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        try:
            if coverage_score < config.min_range_coverage:
                recommendations.append(f"Increase data coverage: Current {coverage_score:.1%}, target {config.min_range_coverage:.1%}")
            
            if coverage_score < 0.8:
                recommendations.append("Consider implementing data recovery procedures for missing periods")
            
            if gap_analysis.get('cross_timeframe_gaps'):
                recommendations.append("Address cross-timeframe coverage inconsistencies")
            
            if config.streaming_mode == StreamingMode.REAL_TIME:
                recommendations.append("Real-time mode: Monitor for data gaps and implement alerting")
            
            if coverage_score > 0.95:
                recommendations.append("Excellent data coverage - consider optimization for performance")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _determine_overall_validity(
        self, 
        coverage_score: float,
        warnings: List[str],
        errors: List[str],
        config: RangeVerificationConfig
    ) -> bool:
        """Determine overall validity based on verification results"""
        try:
            if coverage_score < config.min_range_coverage:
                return False
            
            if errors:
                return False
            
            if config.validation_level == RangeValidationLevel.STRICT:
                if len(warnings) > 5:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update performance statistics"""
        self.stats['total_verifications'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        
        if success:
            self.stats['successful_verifications'] += 1
        else:
            self.stats['failed_verifications'] += 1
        
        self.stats['avg_verification_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_verifications']
        )
    
    def get_verification_stats(self) -> Dict:
        """Get timeframe range verification service statistics"""
        return self.stats.copy()

# Example usage and testing
def test_timeframe_range_verification_service():
    """Test the timeframe range verification service"""
    # Create sample data with some gaps
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # 1-minute data with gaps
    dates_1m = pd.date_range(base_time, periods=120, freq='1min')
    # Remove some timestamps to create gaps
    dates_1m = dates_1m.drop(dates_1m[30:35])  # Gap from 30-35 minutes
    dates_1m = dates_1m.drop(dates_1m[80:85])  # Gap from 80-85 minutes
    
    data_1m = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(dates_1m)),
        'high': np.random.uniform(200, 300, len(dates_1m)),
        'low': np.random.uniform(50, 100, len(dates_1m)),
        'close': np.random.uniform(100, 200, len(dates_1m)),
        'volume': np.random.uniform(1000, 10000, len(dates_1m))
    }, index=dates_1m)
    
    # 5-minute data
    dates_5m = pd.date_range(base_time, periods=24, freq='5min')
    data_5m = pd.DataFrame({
        'open': np.random.uniform(100, 200, 24),
        'high': np.random.uniform(200, 300, 24),
        'low': np.random.uniform(50, 100, 24),
        'close': np.random.uniform(100, 200, 24),
        'volume': np.random.uniform(1000, 10000, 24)
    }, index=dates_5m)
    
    # Create data dictionary
    data_dict = {
        '1m': data_1m,
        '5m': data_5m
    }
    
    # Initialize service
    service = TimeframeRangeVerificationService()
    
    # Test range verification
    config = RangeVerificationConfig(
        validation_level=RangeValidationLevel.NORMAL,
        streaming_mode=StreamingMode.BATCH,
        min_range_coverage=0.9,
        max_gap_threshold=timedelta(minutes=10)
    )
    
    result = service.verify_timeframe_ranges(data_dict, config, 'crypto')
    
    # Print results
    print("=== Timeframe Range Verification Test Results ===")
    print(f"Valid: {result.is_valid}")
    print(f"Coverage Score: {result.coverage_score:.2%}")
    print(f"Edge Cases: {len(result.edge_cases)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Recommendations: {len(result.recommendations)}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:5]:  # Show first 5 warnings
            print(f"  - {warning}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations[:5]:  # Show first 5 recommendations
            print(f"  - {rec}")
    
    if result.edge_cases:
        print("\nEdge Cases:")
        for case in result.edge_cases[:3]:  # Show first 3 edge cases
            print(f"  - {case}")
    
    # Print statistics
    stats = service.get_verification_stats()
    print(f"\nService Statistics:")
    print(f"Total Verifications: {stats['total_verifications']}")
    print(f"Success Rate: {stats['successful_verifications']/stats['total_verifications']:.1%}")
    print(f"Avg Processing Time: {stats['avg_verification_time_ms']:.2f}ms")
    
    return result

if __name__ == "__main__":
    # Run test if script is executed directly
    result = test_timeframe_range_verification_service()
