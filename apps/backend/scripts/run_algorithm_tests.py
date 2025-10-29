#!/usr/bin/env python3
"""
Comprehensive Test Runner for AlphaPlus Algorithm Integration
Runs all tests for the enhanced algorithm implementations
"""

import pytest
import asyncio
import sys
import os
import time
from datetime import datetime
import subprocess

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_test_suite(test_file: str, test_name: str = None) -> dict:
    """Run a specific test suite and return results"""
    print(f"\n🧪 Running {test_name or test_file}...")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            'test_file': test_file,
            'test_name': test_name or test_file,
            'success': result.returncode == 0,
            'elapsed_time': elapsed_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'test_file': test_file,
            'test_name': test_name or test_file,
            'success': False,
            'elapsed_time': 300,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'return_code': -1
        }
    except Exception as e:
        return {
            'test_file': test_file,
            'test_name': test_name or test_file,
            'success': False,
            'elapsed_time': 0,
            'stdout': '',
            'stderr': str(e),
            'return_code': -1
        }

def run_all_tests():
    """Run all test suites"""
    print("🚀 Starting Comprehensive Test Suite for AlphaPlus Algorithm Integration")
    print("=" * 80)
    
    # Define test suites
    test_suites = [
        {
            'file': 'tests/test_historical_data_preloader.py',
            'name': 'Historical Data Preloader Tests'
        },
        {
            'file': 'tests/test_enhanced_orderbook_integration.py',
            'name': 'Enhanced Order Book Integration Tests'
        },
        {
            'file': 'tests/test_standalone_psychological_levels_analyzer.py',
            'name': 'Standalone Psychological Levels Analyzer Tests'
        },
        {
            'file': 'tests/test_enhanced_volume_weighted_levels_analyzer.py',
            'name': 'Enhanced Volume-Weighted Levels Analyzer Tests'
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_suite in test_suites:
        result = run_test_suite(test_suite['file'], test_suite['name'])
        results.append(result)
        
        # Print immediate results
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} {result['test_name']} ({result['elapsed_time']:.2f}s)")
        
        if not result['success']:
            print(f"   Error: {result['stderr'][:200]}...")
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = len(results) - passed_tests
    
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total Time: {total_elapsed_time:.2f}s")
    print(f"Success Rate: {(passed_tests/len(results)*100):.1f}%")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS")
    print("-" * 80)
    
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} {result['test_name']}")
        print(f"   Time: {result['elapsed_time']:.2f}s")
        
        if not result['success']:
            print(f"   Error: {result['stderr']}")
            if result['stdout']:
                print(f"   Output: {result['stdout'][:300]}...")
        print()
    
    # Return results for further processing
    return {
        'total_tests': len(results),
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'total_time': total_elapsed_time,
        'success_rate': passed_tests/len(results)*100,
        'results': results
    }

def run_specific_test(test_name: str):
    """Run a specific test by name"""
    test_mapping = {
        'preloader': 'tests/test_historical_data_preloader.py',
        'orderbook': 'tests/test_enhanced_orderbook_integration.py',
        'psychological': 'tests/test_standalone_psychological_levels_analyzer.py',
        'volume': 'tests/test_enhanced_volume_weighted_levels_analyzer.py'
    }
    
    if test_name.lower() in test_mapping:
        test_file = test_mapping[test_name.lower()]
        result = run_test_suite(test_file, f"{test_name.title()} Tests")
        
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"\n{status} {result['test_name']} ({result['elapsed_time']:.2f}s)")
        
        if not result['success']:
            print(f"Error: {result['stderr']}")
        
        return result
    else:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_mapping.keys())}")
        return None

def run_performance_tests():
    """Run performance-focused tests"""
    print("⚡ Running Performance Tests...")
    
    performance_tests = [
        'tests/test_historical_data_preloader.py::TestHistoricalDataPreloaderPerformance',
        'tests/test_enhanced_orderbook_integration.py::TestEnhancedOrderBookIntegrationPerformance',
        'tests/test_standalone_psychological_levels_analyzer.py::TestStandalonePsychologicalLevelsAnalyzerPerformance',
        'tests/test_enhanced_volume_weighted_levels_analyzer.py::TestEnhancedVolumeWeightedLevelsAnalyzerPerformance'
    ]
    
    results = []
    
    for test_class in performance_tests:
        print(f"\n🧪 Running {test_class}...")
        
        start_time = time.time()
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_class, '-v'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout for performance tests
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_class} ({elapsed_time:.2f}s)")
        
        results.append({
            'test_class': test_class,
            'success': success,
            'elapsed_time': elapsed_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    
    return results

def generate_test_report(results: dict):
    """Generate a detailed test report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("AlphaPlus Algorithm Integration Test Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {results['total_tests']}\n")
        f.write(f"Passed: {results['passed_tests']}\n")
        f.write(f"Failed: {results['failed_tests']}\n")
        f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
        f.write(f"Total Time: {results['total_time']:.2f}s\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results['results']:
            status = "PASSED" if result['success'] else "FAILED"
            f.write(f"{status}: {result['test_name']}\n")
            f.write(f"  Time: {result['elapsed_time']:.2f}s\n")
            
            if not result['success']:
                f.write(f"  Error: {result['stderr']}\n")
            
            f.write("\n")
    
    print(f"📄 Test report saved to: {report_file}")
    return report_file

def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaPlus Algorithm Integration Test Runner')
    parser.add_argument('--test', help='Run specific test (preloader, orderbook, psychological, volume)')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--report', action='store_true', help='Generate detailed test report')
    
    args = parser.parse_args()
    
    if args.test:
        run_specific_test(args.test)
    elif args.performance:
        run_performance_tests()
    else:
        results = run_all_tests()
        
        if args.report:
            generate_test_report(results)
        
        # Exit with appropriate code
        if results['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()
