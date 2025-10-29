#!/usr/bin/env python3
"""
Unified Test Runner for AlphaPulse

This script consolidates and runs all tests (test_integration.py, test_indicators.py,
test_performance.py, test_database.py, test_edge_cases.py, test_utils.py, conftest.py)
with specific fixtures (SQLite DB, fakeredis, CSV mock, datetime.UTC) and test types
(Unit, Integration, Database, Performance, Edge Cases).

Usage:
    python scripts/run_tests.py [options]

Options:
    --test-type: Specific test type to run (unit, integration, database, performance, edge-cases, all)
    --verbose: Verbose output
    --benchmark: Run performance benchmarks
    --coverage: Generate coverage report
    --parallel: Run tests in parallel
    --html-report: Generate HTML test report
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from ..utils.utils import setup_logging, save_json_file, load_json_file

logger = logging.getLogger(__name__)


class TestRunner:
    """Unified test runner for AlphaPulse."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize test runner.
        
        Args:
            config: Test configuration
        """
        self.config = config or self._load_default_config()
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        setup_logging(
            level=self.config.get('log_level', 'INFO'),
            log_file=self.config.get('log_file', 'logs/test_runner.log')
        )
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default test configuration."""
        return {
            'test_files': [
                'tests/test_integration.py',
                'tests/test_indicators.py',
                'tests/test_performance.py',
                'tests/test_database.py',
                'tests/test_edge_cases.py',
                'tests/test_utils.py',
                'tests/conftest.py'
            ],
            'test_types': {
                'unit': ['tests/test_utils.py', 'tests/test_indicators.py'],
                'integration': ['tests/test_integration.py'],
                'database': ['tests/test_database.py'],
                'performance': ['tests/test_performance.py'],
                'edge-cases': ['tests/test_edge_cases.py']
            },
            'fixtures': {
                'sqlite_db': 'test.db',
                'fakeredis': True,
                'csv_mock': 'tests/data/mock_candles.csv',
                'datetime_utc': True
            },
            'targets': {
                'success_rate': 100.0,  # 100% success target
                'latency_target': 50,   # 50ms latency target
                'accuracy_target': 0.75, # 75% accuracy target
                'filter_rate_target': 0.65  # 65% filter rate target
            },
            'log_level': 'INFO',
            'log_file': 'logs/test_runner.log',
            'reports_dir': 'reports',
            'benchmark_save': 'alphapulse'
        }
    
    def setup_test_environment(self):
        """Setup test environment with required fixtures."""
        logger.info("Setting up test environment...")
        
        # Create necessary directories
        Path('logs').mkdir(exist_ok=True)
        Path('reports').mkdir(exist_ok=True)
        Path('tests/data').mkdir(exist_ok=True)
        
        # Create mock CSV data if it doesn't exist
        self._create_mock_csv_data()
        
        # Setup environment variables
        os.environ['ALPHAPULSE_TEST_MODE'] = 'true'
        os.environ['ALPHAPULSE_DATABASE_URL'] = 'sqlite:///test.db'
        os.environ['ALPHAPULSE_REDIS_URL'] = 'redis://localhost:6379'
        os.environ['ALPHAPULSE_ENVIRONMENT'] = 'test'
        
        logger.info("Test environment setup complete")
    
    def _create_mock_csv_data(self):
        """Create mock CSV data for testing."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        csv_path = self.config['fixtures']['csv_mock']
        
        if not Path(csv_path).exists():
            logger.info(f"Creating mock CSV data at {csv_path}")
            
            # Generate realistic test data
            np.random.seed(42)
            n_candles = 1000
            
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(hours=n_candles),
                periods=n_candles,
                freq='1min'
            )
            
            # Generate price data
            base_price = 50000
            returns = np.random.normal(0, 0.001, n_candles)
            prices = base_price * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices * (1 + np.random.normal(0, 0.0005, n_candles)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_candles))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_candles))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, n_candles)
            })
            
            data.to_csv(csv_path, index=False)
            logger.info(f"Mock CSV data created with {len(data)} candles")
    
    def run_tests(self, test_type: str = 'all', options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run tests based on type and options.
        
        Args:
            test_type: Type of tests to run
            options: Additional options
            
        Returns:
            Test results summary
        """
        self.start_time = datetime.now()
        logger.info(f"Starting test run for type: {test_type}")
        
        options = options or {}
        
        # Build pytest command
        cmd = self._build_pytest_command(test_type, options)
        
        # Run tests
        result = self._execute_pytest(cmd)
        
        # Process results
        self.end_time = datetime.now()
        test_summary = self._process_test_results(result, test_type)
        
        # Save results
        self._save_test_results(test_summary)
        
        return test_summary
    
    def _build_pytest_command(self, test_type: str, options: Dict[str, Any]) -> List[str]:
        """Build pytest command with appropriate arguments."""
        cmd = ['python', '-m', 'pytest']
        
        # Add test files based on type
        if test_type == 'all':
            cmd.extend(self.config['test_files'])
        elif test_type in self.config['test_types']:
            cmd.extend(self.config['test_types'][test_type])
        else:
            logger.warning(f"Unknown test type: {test_type}, running all tests")
            cmd.extend(self.config['test_files'])
        
        # Add common options
        cmd.extend([
            '-v',  # Verbose output
            '--tb=short',  # Short traceback format
            '--strict-markers',  # Strict marker checking
            '--disable-warnings',  # Disable warnings
            '--color=yes'  # Colored output
        ])
        
        # Add benchmark options
        if options.get('benchmark', False):
            cmd.extend([
                '--benchmark-only',
                f'--benchmark-save={self.config["benchmark_save"]}',
                '--benchmark-sort=mean'
            ])
        
        # Add coverage options
        if options.get('coverage', False):
            cmd.extend([
                '--cov=core',
                '--cov=services',
                '--cov=utils',
                '--cov=database',
                '--cov-report=html:reports/coverage',
                '--cov-report=term-missing'
            ])
        
        # Add parallel options
        if options.get('parallel', False):
            cmd.extend(['-n', 'auto'])
        
        # Add HTML report
        if options.get('html_report', False):
            cmd.extend([
                '--html=reports/test_report.html',
                '--self-contained-html'
            ])
        
        # Add custom markers
        cmd.extend([
            '-m', f'not slow',  # Skip slow tests by default
            '--asyncio-mode=auto'  # Auto-detect async tests
        ])
        
        return cmd
    
    def _execute_pytest(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute pytest command and capture results."""
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out")
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test execution timed out',
                'success': False
            }
        except Exception as e:
            logger.error(f"Error executing tests: {e}")
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    def _process_test_results(self, result: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Process pytest results and extract metrics."""
        summary = {
            'test_type': test_type,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': None,
            'success': result['success'],
            'returncode': result['returncode'],
            'metrics': {},
            'errors': [],
            'warnings': []
        }
        
        # Calculate duration
        if self.start_time and self.end_time:
            summary['duration'] = (self.end_time - self.start_time).total_seconds()
        
        # Parse stdout for test results
        if result['stdout']:
            summary.update(self._parse_pytest_output(result['stdout']))
        
        # Parse stderr for errors
        if result['stderr']:
            summary['errors'].append(result['stderr'])
        
        # Calculate success rate
        if 'total_tests' in summary and summary['total_tests'] > 0:
            summary['success_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
        else:
            summary['success_rate'] = 0.0
        
        # Check if targets are met
        summary['targets_met'] = self._check_targets(summary)
        
        return summary
    
    def _parse_pytest_output(self, stdout: str) -> Dict[str, Any]:
        """Parse pytest output to extract test metrics."""
        metrics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'error_tests': 0,
            'xfailed_tests': 0,
            'xpassed_tests': 0
        }
        
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for test summary
            if 'passed' in line and 'failed' in line:
                # Parse summary line like "10 passed, 2 failed, 1 skipped"
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if 'passed' in part and 'xpassed' not in part:
                        metrics['passed_tests'] = int(part.split()[0])
                    elif 'failed' in part and 'xfailed' not in part:
                        metrics['failed_tests'] = int(part.split()[0])
                    elif 'skipped' in part:
                        metrics['skipped_tests'] = int(part.split()[0])
                    elif 'error' in part:
                        metrics['error_tests'] = int(part.split()[0])
                    elif 'xfailed' in part:
                        metrics['xfailed_tests'] = int(part.split()[0])
                    elif 'xpassed' in part:
                        metrics['xpassed_tests'] = int(part.split()[0])
                
                # Calculate total
                metrics['total_tests'] = (
                    metrics['passed_tests'] + 
                    metrics['failed_tests'] + 
                    metrics['skipped_tests'] + 
                    metrics['error_tests'] + 
                    metrics['xfailed_tests'] + 
                    metrics['xpassed_tests']
                )
                break
        
        return metrics
    
    def _check_targets(self, summary: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance targets are met."""
        targets = self.config['targets']
        results = {}
        
        # Check success rate
        success_rate = summary.get('success_rate', 0.0)
        results['success_rate'] = success_rate >= targets['success_rate']
        
        # Check duration (if available)
        if summary.get('duration'):
            # Assume latency target applies to average test duration
            avg_test_duration = summary['duration'] / summary.get('total_tests', 1)
            results['latency'] = avg_test_duration * 1000 < targets['latency_target']
        
        return results
    
    def _save_test_results(self, summary: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/test_results_{timestamp}.json"
        
        # Add configuration to results
        summary['config'] = self.config
        
        save_json_file(summary, filename)
        logger.info(f"Test results saved to {filename}")
    
    def run_all_test_types(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all test types and generate comprehensive report."""
        logger.info("Running all test types...")
        
        all_results = {}
        overall_summary = {
            'start_time': datetime.now().isoformat(),
            'test_types': {},
            'overall_success': True,
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0
        }
        
        # Run each test type
        for test_type in self.config['test_types'].keys():
            logger.info(f"Running {test_type} tests...")
            
            try:
                result = self.run_tests(test_type, options)
                all_results[test_type] = result
                overall_summary['test_types'][test_type] = result
                
                # Update overall metrics
                overall_summary['total_tests'] += result.get('total_tests', 0)
                overall_summary['total_passed'] += result.get('passed_tests', 0)
                overall_summary['total_failed'] += result.get('failed_tests', 0)
                
                if not result.get('success', False):
                    overall_summary['overall_success'] = False
                
            except Exception as e:
                logger.error(f"Error running {test_type} tests: {e}")
                all_results[test_type] = {
                    'success': False,
                    'error': str(e)
                }
                overall_summary['overall_success'] = False
        
        # Calculate overall success rate
        if overall_summary['total_tests'] > 0:
            overall_summary['overall_success_rate'] = (
                overall_summary['total_passed'] / overall_summary['total_tests']
            ) * 100
        else:
            overall_summary['overall_success_rate'] = 0.0
        
        overall_summary['end_time'] = datetime.now().isoformat()
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/comprehensive_test_report_{timestamp}.json"
        save_json_file(overall_summary, filename)
        
        logger.info(f"Comprehensive test report saved to {filename}")
        logger.info(f"Overall success rate: {overall_summary['overall_success_rate']:.2f}%")
        
        return overall_summary


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='AlphaPulse Test Runner')
    parser.add_argument(
        '--test-type',
        choices=['unit', 'integration', 'database', 'performance', 'edge-cases', 'all'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarks'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--html-report',
        action='store_true',
        help='Generate HTML test report'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to test configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        config = load_json_file(args.config)
    
    # Create test runner
    runner = TestRunner(config)
    
    # Setup test environment
    runner.setup_test_environment()
    
    # Build options
    options = {
        'benchmark': args.benchmark,
        'coverage': args.coverage,
        'parallel': args.parallel,
        'html_report': args.html_report,
        'verbose': args.verbose
    }
    
    # Run tests
    if args.test_type == 'all':
        results = runner.run_all_test_types(options)
    else:
        results = runner.run_tests(args.test_type, options)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RUN SUMMARY")
    print("="*60)
    print(f"Test Type: {args.test_type}")
    print(f"Success: {results.get('success', False)}")
    print(f"Success Rate: {results.get('success_rate', 0):.2f}%")
    print(f"Total Tests: {results.get('total_tests', 0)}")
    print(f"Passed: {results.get('passed_tests', 0)}")
    print(f"Failed: {results.get('failed_tests', 0)}")
    print(f"Duration: {results.get('duration', 0):.2f}s")
    
    if 'targets_met' in results:
        print("\nTargets Met:")
        for target, met in results['targets_met'].items():
            status = "✓" if met else "✗"
            print(f"  {target}: {status}")
    
    # Exit with appropriate code
    if results.get('success', False):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
