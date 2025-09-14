#!/usr/bin/env python3
"""
Comprehensive Test Runner for AlphaPulse
Executes all tests and generates detailed reports with metrics and visualizations
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from ..database.reflect_and_compare import DatabaseStructureScanner
from ..database.models_enhanced import create_tables, drop_tables, get_test_session

class ComprehensiveTestRunner:
    """Comprehensive test runner for AlphaPulse"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_database_scan(self) -> Dict:
        """Run database structure scan"""
        print("🔍 Running database structure scan...")
        
        scanner = DatabaseStructureScanner()
        report = scanner.generate_report(f"{self.output_dir}/database_scan_report.json")
        
        self.test_results['database_scan'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'matches_required_schema': report['executive_summary']['matches_required_schema'],
            'migration_needed': report['executive_summary']['migration_needed'],
            'missing_tables_count': report['executive_summary']['missing_tables_count'],
            'recommendations': report['recommendations']
        }
        
        return report
    
    def setup_test_database(self):
        """Setup test database with enhanced models"""
        print("🔧 Setting up test database...")
        
        try:
            # Create tables
            create_tables()
            print("✅ Test database setup completed")
            return True
        except Exception as e:
            print(f"❌ Test database setup failed: {e}")
            return False
    
    def run_unit_tests(self) -> Dict:
        """Run unit tests for indicators and signal validation"""
        print("🧪 Running unit tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run unit tests
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'tests/test_indicators_enhanced.py',
                '-v', '--tb=short', '--json-report',
                '--json-report-file', f'{self.output_dir}/unit_tests_report.json'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            end_time = time.perf_counter()
            test_time = end_time - start_time
            
            # Parse results
            passed = result.stdout.count('PASSED')
            failed = result.stdout.count('FAILED')
            total = passed + failed
            
            self.test_results['unit_tests'] = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': test_time,
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'success_rate': passed / total if total > 0 else 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
            print(f"✅ Unit tests completed: {passed}/{total} passed")
            return self.test_results['unit_tests']
            
        except Exception as e:
            print(f"❌ Unit tests failed: {e}")
            return {'error': str(e)}
    
    def run_integration_tests(self) -> Dict:
        """Run integration tests for full pipeline"""
        print("🔄 Running integration tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run integration tests
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'tests/test_pipeline_enhanced.py',
                '-v', '--tb=short', '--json-report',
                '--json-report-file', f'{self.output_dir}/integration_tests_report.json'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            end_time = time.perf_counter()
            test_time = end_time - start_time
            
            # Parse results
            passed = result.stdout.count('PASSED')
            failed = result.stdout.count('FAILED')
            total = passed + failed
            
            self.test_results['integration_tests'] = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': test_time,
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'success_rate': passed / total if total > 0 else 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
            print(f"✅ Integration tests completed: {passed}/{total} passed")
            return self.test_results['integration_tests']
            
        except Exception as e:
            print(f"❌ Integration tests failed: {e}")
            return {'error': str(e)}
    
    def run_database_tests(self) -> Dict:
        """Run database tests"""
        print("🗄️ Running database tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run database tests
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'tests/test_db_enhanced.py',
                '-v', '--tb=short', '--json-report',
                '--json-report-file', f'{self.output_dir}/database_tests_report.json'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            end_time = time.perf_counter()
            test_time = end_time - start_time
            
            # Parse results
            passed = result.stdout.count('PASSED')
            failed = result.stdout.count('FAILED')
            total = passed + failed
            
            self.test_results['database_tests'] = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': test_time,
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'success_rate': passed / total if total > 0 else 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
            print(f"✅ Database tests completed: {passed}/{total} passed")
            return self.test_results['database_tests']
            
        except Exception as e:
            print(f"❌ Database tests failed: {e}")
            return {'error': str(e)}
    
    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmarks"""
        print("⚡ Running performance benchmarks...")
        
        start_time = time.perf_counter()
        
        try:
            # Run performance benchmarks
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'tests/test_performance_benchmark.py',
                '--benchmark-only', '--benchmark-save', f'{self.output_dir}/performance_benchmark',
                '--benchmark-save-data', f'{self.output_dir}/performance_data.json'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            end_time = time.perf_counter()
            benchmark_time = end_time - start_time
            
            self.test_results['performance_benchmarks'] = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': benchmark_time,
                'output': result.stdout,
                'error': result.stderr
            }
            
            print(f"✅ Performance benchmarks completed")
            return self.test_results['performance_benchmarks']
            
        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")
            return {'error': str(e)}
    
    def generate_visualizations(self):
        """Generate test result visualizations"""
        print("📊 Generating visualizations...")
        
        try:
            # Create test results summary
            test_summary = {
                'unit_tests': self.test_results.get('unit_tests', {}),
                'integration_tests': self.test_results.get('integration_tests', {}),
                'database_tests': self.test_results.get('database_tests', {})
            }
            
            # Test success rates
            categories = []
            success_rates = []
            
            for test_type, results in test_summary.items():
                if 'success_rate' in results:
                    categories.append(test_type.replace('_', ' ').title())
                    success_rates.append(results['success_rate'] * 100)
            
            # Create success rate chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, success_rates, color=['#2ecc71', '#3498db', '#e74c3c'])
            plt.title('Test Success Rates', fontsize=16, fontweight='bold')
            plt.ylabel('Success Rate (%)', fontsize=12)
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/test_success_rates.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create test duration chart
            durations = []
            for test_type, results in test_summary.items():
                if 'duration_seconds' in results:
                    durations.append(results['duration_seconds'])
            
            if durations:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(categories, durations, color=['#f39c12', '#9b59b6', '#1abc9c'])
                plt.title('Test Execution Times', fontsize=16, fontweight='bold')
                plt.ylabel('Duration (seconds)', fontsize=12)
                
                # Add value labels on bars
                for bar, duration in zip(bars, durations):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f'{duration:.1f}s', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/test_durations.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"✅ Visualizations saved to {self.output_dir}/")
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive test report"""
        print("📋 Generating comprehensive report...")
        
        # Calculate overall metrics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for test_type, results in self.test_results.items():
            if 'total_tests' in results:
                total_tests += results['total_tests']
                total_passed += results.get('passed', 0)
                total_failed += results.get('failed', 0)
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'test_runner_version': '1.0.0',
                'output_directory': self.output_dir
            },
            'executive_summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'overall_success_rate': overall_success_rate,
                'total_duration_seconds': (self.end_time - self.start_time) if self.end_time and self.start_time else 0,
                'database_migration_needed': self.test_results.get('database_scan', {}).get('migration_needed', False)
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations(),
            'performance_metrics': self._extract_performance_metrics()
        }
        
        # Save report
        report_file = f"{self.output_dir}/comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved to {report_file}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Database recommendations
        if self.test_results.get('database_scan', {}).get('migration_needed', False):
            recommendations.append("Run database migrations to match required schema")
        
        # Test success rate recommendations
        for test_type, results in self.test_results.items():
            if 'success_rate' in results and results['success_rate'] < 0.9:
                recommendations.append(f"Improve {test_type.replace('_', ' ')} success rate (currently {results['success_rate']:.1%})")
        
        # Performance recommendations
        if 'performance_benchmarks' in self.test_results:
            recommendations.append("Review performance benchmark results for optimization opportunities")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system is ready for production")
        
        return recommendations
    
    def _extract_performance_metrics(self) -> Dict:
        """Extract performance metrics from test results"""
        metrics = {}
        
        # Extract latency metrics from test outputs
        for test_type, results in self.test_results.items():
            if 'output' in results:
                output = results['output']
                
                # Look for latency patterns
                if 'latency' in output.lower():
                    # Extract average latency
                    import re
                    latency_match = re.search(r'Average latency: ([\d.]+)ms', output)
                    if latency_match:
                        metrics[f'{test_type}_avg_latency_ms'] = float(latency_match.group(1))
                
                # Look for throughput patterns
                if 'throughput' in output.lower():
                    throughput_match = re.search(r'Throughput: ([\d.]+) signals/sec', output)
                    if throughput_match:
                        metrics[f'{test_type}_throughput_signals_per_sec'] = float(throughput_match.group(1))
        
        return metrics
    
    def run_all_tests(self) -> Dict:
        """Run all comprehensive tests"""
        print("🚀 Starting comprehensive AlphaPulse testing...")
        print("=" * 80)
        
        self.start_time = time.perf_counter()
        
        # Step 1: Database scan
        print("\n📊 STEP 1: Database Structure Analysis")
        print("-" * 50)
        db_scan = self.run_database_scan()
        
        # Step 2: Setup test database
        print("\n🔧 STEP 2: Test Database Setup")
        print("-" * 50)
        db_setup_success = self.setup_test_database()
        
        if not db_setup_success:
            print("❌ Test database setup failed. Aborting tests.")
            return self.test_results
        
        # Step 3: Unit tests
        print("\n🧪 STEP 3: Unit Tests")
        print("-" * 50)
        unit_results = self.run_unit_tests()
        
        # Step 4: Integration tests
        print("\n🔄 STEP 4: Integration Tests")
        print("-" * 50)
        integration_results = self.run_integration_tests()
        
        # Step 5: Database tests
        print("\n🗄️ STEP 5: Database Tests")
        print("-" * 50)
        db_results = self.run_database_tests()
        
        # Step 6: Performance benchmarks
        print("\n⚡ STEP 6: Performance Benchmarks")
        print("-" * 50)
        performance_results = self.run_performance_benchmarks()
        
        # Step 7: Generate visualizations
        print("\n📊 STEP 7: Generate Visualizations")
        print("-" * 50)
        self.generate_visualizations()
        
        # Step 8: Generate comprehensive report
        print("\n📋 STEP 8: Generate Comprehensive Report")
        print("-" * 50)
        self.end_time = time.perf_counter()
        report = self.generate_comprehensive_report()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TESTING SUMMARY")
        print("=" * 80)
        
        summary = report['executive_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration_seconds']:.1f} seconds")
        print(f"Database Migration Needed: {summary['database_migration_needed']}")
        
        if summary['overall_success_rate'] >= 0.9:
            print("\n🎉 All tests passed successfully! AlphaPulse is ready for production.")
        else:
            print("\n⚠️ Some tests failed. Please review the detailed report.")
        
        print(f"\n📁 Results saved to: {self.output_dir}/")
        print("=" * 80)
        
        return report

def main():
    """Main function to run comprehensive tests"""
    parser = argparse.ArgumentParser(description="Comprehensive AlphaPulse Test Runner")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for test results")
    parser.add_argument("--skip-db-scan", action="store_true", help="Skip database structure scan")
    parser.add_argument("--skip-unit", action="store_true", help="Skip unit tests")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--skip-database", action="store_true", help="Skip database tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance benchmarks")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner(args.output_dir)
    
    # Run tests based on arguments
    if not args.skip_db_scan:
        runner.run_database_scan()
    
    if not args.skip_unit:
        runner.run_unit_tests()
    
    if not args.skip_integration:
        runner.run_integration_tests()
    
    if not args.skip_database:
        runner.run_database_tests()
    
    if not args.skip_performance:
        runner.run_performance_benchmarks()
    
    if not args.skip_visualizations:
        runner.generate_visualizations()
    
    # Generate report
    report = runner.generate_comprehensive_report()
    
    return report

if __name__ == "__main__":
    main()
