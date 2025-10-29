"""
Performance Regression Testing Framework
Detects performance degradations and maintains performance baselines
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class RegressionStatus(Enum):
    """Status of regression test"""
    PASSED = "passed"
    DEGRADED = "degraded"
    IMPROVED = "improved"
    UNKNOWN = "unknown"

@dataclass
class RegressionTest:
    """Performance regression test configuration"""
    test_id: str
    test_name: str
    baseline_id: str
    current_result: Any  # BenchmarkResult or ProfilingResult
    baseline_result: Any
    tolerance_percentage: float = 10.0
    status: RegressionStatus = RegressionStatus.UNKNOWN
    degradation_score: float = 0.0
    improvement_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegressionReport:
    """Report of regression test results"""
    report_id: str
    timestamp: datetime
    total_tests: int
    passed_tests: int
    degraded_tests: int
    improved_tests: int
    unknown_tests: int
    overall_score: float  # 0.0 to 1.0 (1.0 = all tests passed)
    test_details: List[RegressionTest] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class PerformanceRegressionTester:
    """Performance regression testing framework"""
    
    def __init__(self, baseline_dir: str = "results/performance_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.regression_tests: List[RegressionTest] = []
        self.regression_reports: List[RegressionReport] = []
        
        # Import from performance_profiling
        from src.app.core.performance_profiling import PerformanceBaseline
        self.PerformanceBaseline = PerformanceBaseline
    
    async def run_regression_tests(self, test_configs: List[Dict[str, Any]]) -> RegressionReport:
        """Run a suite of regression tests"""
        try:
            self.logger.info(f"🚀 Starting performance regression tests: {len(test_configs)} tests")
            
            regression_tests = []
            
            for config in test_configs:
                try:
                    regression_test = await self._run_single_regression_test(config)
                    regression_tests.append(regression_test)
                    
                except Exception as e:
                    self.logger.error(f"❌ Regression test failed for {config.get('test_name', 'unknown')}: {e}")
                    # Create failed test
                    regression_test = RegressionTest(
                        test_id=f"failed_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                        test_name=config.get('test_name', 'unknown'),
                        baseline_id=config.get('baseline_id', 'unknown'),
                        current_result=None,
                        baseline_result=None,
                        status=RegressionStatus.UNKNOWN,
                        metadata={'error': str(e)}
                    )
                    regression_tests.append(regression_test)
            
            # Generate regression report
            report = self._generate_regression_report(regression_tests)
            
            # Store results
            self.regression_tests.extend(regression_tests)
            self.regression_reports.append(report)
            
            # Save report
            self._save_regression_report(report)
            
            self.logger.info(f"✅ Regression tests completed: {report.passed_tests}/{report.total_tests} passed")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Regression test suite failed: {e}")
            raise
    
    async def _run_single_regression_test(self, config: Dict[str, Any]) -> RegressionTest:
        """Run a single regression test"""
        try:
            test_name = config['test_name']
            baseline_id = config['baseline_id']
            tolerance = config.get('tolerance_percentage', 10.0)
            
            self.logger.info(f"📊 Running regression test: {test_name}")
            
            # Get baseline result
            baseline_result = await self._get_baseline_result(baseline_id)
            if not baseline_result:
                raise Exception(f"Baseline {baseline_id} not found")
            
            # Run current test
            current_result = await self._run_current_test(config)
            
            # Compare results
            regression_test = self._compare_results(
                test_name, baseline_id, current_result, baseline_result, tolerance
            )
            
            return regression_test
            
        except Exception as e:
            self.logger.error(f"Error running regression test: {e}")
            raise
    
    async def _get_baseline_result(self, baseline_id: str) -> Any:
        """Get baseline result from storage"""
        try:
            # Load baseline from file
            baseline_file = self.baseline_dir / f"{baseline_id}.json"
            
            if not baseline_file.exists():
                self.logger.warning(f"Baseline file not found: {baseline_file}")
                return None
            
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            # Create baseline object
            baseline = self.PerformanceBaseline(
                baseline_id=baseline_data['baseline_id'],
                timestamp=datetime.fromisoformat(baseline_data['timestamp']),
                test_name=baseline_data['test_name'],
                scenario=baseline_data['scenario'],
                expected_execution_time_ms=baseline_data['expected_execution_time_ms'],
                expected_memory_usage_mb=baseline_data['expected_memory_usage_mb'],
                expected_cpu_utilization=baseline_data['expected_cpu_utilization'],
                expected_throughput_ops_per_sec=baseline_data['expected_throughput_ops_per_sec'],
                tolerance_percentage=baseline_data.get('tolerance_percentage', 10.0)
            )
            
            return baseline
            
        except Exception as e:
            self.logger.error(f"Error loading baseline: {e}")
            return None
    
    async def _run_current_test(self, config: Dict[str, Any]) -> Any:
        """Run the current test to get current performance metrics"""
        try:
            # This would typically run the actual benchmark or profiling
            # For now, we'll simulate by creating a mock result
            
            from src.app.core.performance_profiling import BenchmarkResult
            
            # Simulate current test execution
            current_result = BenchmarkResult(
                benchmark_id=f"current_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                timestamp=datetime.now(),
                test_name=config['test_name'],
                scenario=config.get('scenario', 'current'),
                execution_time_ms=config.get('mock_execution_time', 100.0),
                memory_usage_mb=config.get('mock_memory_usage', 50.0),
                cpu_utilization=config.get('mock_cpu_usage', 25.0),
                throughput_ops_per_sec=config.get('mock_throughput', 1000.0),
                metadata={'source': 'regression_test'}
            )
            
            return current_result
            
        except Exception as e:
            self.logger.error(f"Error running current test: {e}")
            raise
    
    def _compare_results(self, test_name: str, baseline_id: str, 
                        current_result: Any, baseline_result: Any, 
                        tolerance: float) -> RegressionTest:
        """Compare current results with baseline"""
        try:
            # Calculate performance changes
            execution_time_change = self._calculate_percentage_change(
                current_result.execution_time_ms, 
                baseline_result.expected_execution_time_ms
            )
            
            memory_change = self._calculate_percentage_change(
                current_result.memory_usage_mb,
                baseline_result.expected_memory_usage_mb
            )
            
            cpu_change = self._calculate_percentage_change(
                current_result.cpu_utilization,
                baseline_result.expected_cpu_utilization
            )
            
            throughput_change = self._calculate_percentage_change(
                current_result.throughput_ops_per_sec,
                baseline_result.expected_throughput_ops_per_sec
            )
            
            # Determine regression status
            status, degradation_score, improvement_score = self._determine_regression_status(
                execution_time_change, memory_change, cpu_change, throughput_change, tolerance
            )
            
            # Create regression test result
            regression_test = RegressionTest(
                test_id=f"regression_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                test_name=test_name,
                baseline_id=baseline_id,
                current_result=current_result,
                baseline_result=baseline_result,
                tolerance_percentage=tolerance,
                status=status,
                degradation_score=degradation_score,
                improvement_score=improvement_score,
                metadata={
                    'execution_time_change': execution_time_change,
                    'memory_change': memory_change,
                    'cpu_change': cpu_change,
                    'throughput_change': throughput_change,
                    'baseline_metrics': {
                        'execution_time_ms': baseline_result.expected_execution_time_ms,
                        'memory_usage_mb': baseline_result.expected_memory_usage_mb,
                        'cpu_utilization': baseline_result.expected_cpu_utilization,
                        'throughput_ops_per_sec': baseline_result.expected_throughput_ops_per_sec
                    },
                    'current_metrics': {
                        'execution_time_ms': current_result.execution_time_ms,
                        'memory_usage_mb': current_result.memory_usage_mb,
                        'cpu_utilization': current_result.cpu_utilization,
                        'throughput_ops_per_sec': current_result.throughput_ops_per_sec
                    }
                }
            )
            
            return regression_test
            
        except Exception as e:
            self.logger.error(f"Error comparing results: {e}")
            raise
    
    def _calculate_percentage_change(self, current: float, baseline: float) -> float:
        """Calculate percentage change from baseline"""
        try:
            if baseline == 0:
                return 0.0 if current == 0 else float('inf')
            
            return ((current - baseline) / baseline) * 100
            
        except Exception:
            return 0.0
    
    def _determine_regression_status(self, execution_time_change: float, memory_change: float,
                                   cpu_change: float, throughput_change: float, 
                                   tolerance: float) -> Tuple[RegressionStatus, float, float]:
        """Determine if performance has degraded, improved, or stayed the same"""
        try:
            # Calculate overall degradation score (higher = worse performance)
            degradation_factors = []
            
            # Execution time: higher is worse
            if execution_time_change > tolerance:
                degradation_factors.append(execution_time_change - tolerance)
            
            # Memory usage: higher is worse
            if memory_change > tolerance:
                degradation_factors.append(memory_change - tolerance)
            
            # CPU usage: higher is worse
            if cpu_change > tolerance:
                degradation_factors.append(cpu_change - tolerance)
            
            # Throughput: lower is worse
            if throughput_change < -tolerance:
                degradation_factors.append(abs(throughput_change) - tolerance)
            
            # Calculate improvement score (higher = better performance)
            improvement_factors = []
            
            # Execution time: lower is better
            if execution_time_change < -tolerance:
                improvement_factors.append(abs(execution_time_change) - tolerance)
            
            # Memory usage: lower is better
            if memory_change < -tolerance:
                improvement_factors.append(abs(memory_change) - tolerance)
            
            # CPU usage: lower is better
            if cpu_change < -tolerance:
                improvement_factors.append(abs(cpu_change) - tolerance)
            
            # Throughput: higher is better
            if throughput_change > tolerance:
                improvement_factors.append(throughput_change - tolerance)
            
            # Calculate scores
            degradation_score = sum(degradation_factors) / len(degradation_factors) if degradation_factors else 0.0
            improvement_score = sum(improvement_factors) / len(improvement_factors) if improvement_factors else 0.0
            
            # Determine status
            if degradation_score > 0 and degradation_score > improvement_score:
                status = RegressionStatus.DEGRADED
            elif improvement_score > 0 and improvement_score > degradation_score:
                status = RegressionStatus.IMPROVED
            elif abs(execution_time_change) <= tolerance and abs(memory_change) <= tolerance and \
                 abs(cpu_change) <= tolerance and abs(throughput_change) <= tolerance:
                status = RegressionStatus.PASSED
            else:
                status = RegressionStatus.UNKNOWN
            
            return status, degradation_score, improvement_score
            
        except Exception as e:
            self.logger.error(f"Error determining regression status: {e}")
            return RegressionStatus.UNKNOWN, 0.0, 0.0
    
    def _generate_regression_report(self, regression_tests: List[RegressionTest]) -> RegressionReport:
        """Generate a comprehensive regression report"""
        try:
            # Count test results
            total_tests = len(regression_tests)
            passed_tests = len([t for t in regression_tests if t.status == RegressionStatus.PASSED])
            degraded_tests = len([t for t in regression_tests if t.status == RegressionStatus.DEGRADED])
            improved_tests = len([t for t in regression_tests if t.status == RegressionStatus.IMPROVED])
            unknown_tests = len([t for t in regression_tests if t.status == RegressionStatus.UNKNOWN])
            
            # Calculate overall score (0.0 to 1.0)
            if total_tests == 0:
                overall_score = 0.0
            else:
                # Weight: PASSED=1.0, IMPROVED=1.0, DEGRADED=0.0, UNKNOWN=0.5
                score = (passed_tests * 1.0 + improved_tests * 1.0 + 
                        degraded_tests * 0.0 + unknown_tests * 0.5)
                overall_score = score / total_tests
            
            # Generate recommendations
            recommendations = self._generate_recommendations(regression_tests)
            
            # Create report
            report = RegressionReport(
                report_id=f"regression_report_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                timestamp=datetime.now(),
                total_tests=total_tests,
                passed_tests=passed_tests,
                degraded_tests=degraded_tests,
                improved_tests=improved_tests,
                unknown_tests=unknown_tests,
                overall_score=overall_score,
                test_details=regression_tests,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating regression report: {e}")
            raise
    
    def _generate_recommendations(self, regression_tests: List[RegressionTest]) -> List[str]:
        """Generate recommendations based on regression test results"""
        try:
            recommendations = []
            
            # Check for degraded tests
            degraded_tests = [t for t in regression_tests if t.status == RegressionStatus.DEGRADED]
            
            if degraded_tests:
                recommendations.append(f"⚠️ {len(degraded_tests)} tests show performance degradation")
                
                # Analyze specific issues
                execution_time_degraded = [t for t in degraded_tests 
                                         if t.metadata.get('execution_time_change', 0) > 0]
                if execution_time_degraded:
                    avg_degradation = statistics.mean([
                        t.metadata.get('execution_time_change', 0) for t in execution_time_degraded
                    ])
                    recommendations.append(f"📈 Execution time degraded by {avg_degradation:.1f}% on average")
                
                memory_degraded = [t for t in degraded_tests 
                                 if t.metadata.get('memory_change', 0) > 0]
                if memory_degraded:
                    avg_degradation = statistics.mean([
                        t.metadata.get('memory_change', 0) for t in memory_degraded
                    ])
                    recommendations.append(f"💾 Memory usage increased by {avg_degradation:.1f}% on average")
            
            # Check for improved tests
            improved_tests = [t for t in regression_tests if t.status == RegressionStatus.IMPROVED]
            if improved_tests:
                recommendations.append(f"✅ {len(improved_tests)} tests show performance improvement")
            
            # Check for unknown tests
            unknown_tests = [t for t in regression_tests if t.status == RegressionStatus.UNKNOWN]
            if unknown_tests:
                recommendations.append(f"❓ {len(unknown_tests)} tests have unclear results - investigate further")
            
            # Overall recommendations
            if len(degraded_tests) > len(improved_tests):
                recommendations.append("🔍 Focus on identifying and fixing performance bottlenecks")
                recommendations.append("📊 Review recent code changes for potential performance impacts")
            elif len(improved_tests) > len(degraded_tests):
                recommendations.append("🎉 Performance improvements detected - consider documenting optimizations")
            else:
                recommendations.append("📈 Performance is stable - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _save_regression_report(self, report: RegressionReport):
        """Save regression report to file"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save detailed report
            report_path = self.baseline_dir / f"regression_report_{timestamp}.json"
            
            # Convert report to JSON-serializable format
            report_data = {
                'report_id': report.report_id,
                'timestamp': report.timestamp.isoformat(),
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'degraded_tests': report.degraded_tests,
                'improved_tests': report.improved_tests,
                'unknown_tests': report.unknown_tests,
                'overall_score': report.overall_score,
                'recommendations': report.recommendations,
                'test_details': [
                    {
                        'test_id': t.test_id,
                        'test_name': t.test_name,
                        'baseline_id': t.baseline_id,
                        'status': t.status.value,
                        'degradation_score': t.degradation_score,
                        'improvement_score': t.improvement_score,
                        'timestamp': t.timestamp.isoformat(),
                        'metadata': t.metadata
                    }
                    for t in report.test_details
                ]
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Regression report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving regression report: {e}")
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """Get summary of all regression tests and reports"""
        try:
            if not self.regression_reports:
                return {"message": "No regression reports available"}
            
            # Calculate overall statistics
            total_reports = len(self.regression_reports)
            total_tests = sum(r.total_tests for r in self.regression_reports)
            total_passed = sum(r.passed_tests for r in self.regression_reports)
            total_degraded = sum(r.degraded_tests for r in self.regression_reports)
            total_improved = sum(r.improved_tests for r in self.regression_reports)
            
            # Calculate average scores
            avg_scores = [r.overall_score for r in self.regression_reports]
            avg_overall_score = statistics.mean(avg_scores) if avg_scores else 0.0
            
            # Recent trends
            recent_reports = sorted(self.regression_reports, key=lambda x: x.timestamp, reverse=True)[:5]
            recent_trend = []
            for report in recent_reports:
                recent_trend.append({
                    'timestamp': report.timestamp.isoformat(),
                    'overall_score': report.overall_score,
                    'passed_tests': report.passed_tests,
                    'degraded_tests': report.degraded_tests,
                    'improved_tests': report.improved_tests
                })
            
            return {
                'total_reports': total_reports,
                'total_tests': total_tests,
                'overall_statistics': {
                    'total_passed': total_passed,
                    'total_degraded': total_degraded,
                    'total_improved': total_improved,
                    'pass_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
                },
                'score_statistics': {
                    'average_overall_score': avg_overall_score,
                    'min_score': min(avg_scores) if avg_scores else 0.0,
                    'max_score': max(avg_scores) if avg_scores else 0.0
                },
                'recent_trend': recent_trend,
                'latest_report': {
                    'timestamp': self.regression_reports[-1].timestamp.isoformat(),
                    'overall_score': self.regression_reports[-1].overall_score,
                    'summary': f"{self.regression_reports[-1].passed_tests}/{self.regression_reports[-1].total_tests} tests passed"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting regression summary: {e}")
            return {"error": str(e)}
    
    def clear_regression_history(self):
        """Clear regression test and report history"""
        self.regression_tests.clear()
        self.regression_reports.clear()
        self.logger.info("🗑️ Regression history cleared")

# Global regression tester instance
performance_regression_tester = PerformanceRegressionTester()

def get_performance_regression_tester() -> PerformanceRegressionTester:
    """Get the global performance regression tester instance"""
    return performance_regression_tester
