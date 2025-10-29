#!/usr/bin/env python3
"""
SDE Framework Integration Test for AlphaPlus
Tests the complete SDE framework integration with database and signal generation
"""

import asyncio
import logging
import time
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import json
import sys
import os

# Add backend to path
sys.path.append('backend')

from src.data.mock_database import MockDataPipeline, MockSDEIntegration
from src.strategies.dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer
from src.data.volume_analyzer import VolumeAnalyzer
from src.core.websocket_binance import BinanceWebSocketClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SDETestResult:
    """SDE test result"""
    test_name: str
    success: bool
    execution_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class SDEFrameworkTester:
    """Test SDE framework integration"""
    
    def __init__(self):
        self.data_pipeline = MockDataPipeline()
        self.sde_integration = MockSDEIntegration()
        self.sr_analyzer = DynamicSupportResistanceAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.websocket_client = BinanceWebSocketClient()
        
        self.test_results = []
        self.symbol = "BTCUSDT"
        self.timeframe = "1m"
    
    async def initialize(self):
        """Initialize all components"""
        await self.data_pipeline.initialize()
        await self.sde_integration.initialize()
        logger.info("✅ SDE Framework Tester initialized")
    
    async def close(self):
        """Close all components"""
        await self.data_pipeline.close()
        await self.sde_integration.close()
    
    async def run_test(self, test_name: str, test_func) -> SDETestResult:
        """Run a single test"""
        try:
            logger.info(f"🧪 Running test: {test_name}")
            start_time = time.time()
            
            result = await test_func()
            execution_time_ms = (time.time() - start_time) * 1000
            
            test_result = SDETestResult(
                test_name=test_name,
                success=True,
                execution_time_ms=execution_time_ms,
                details=result
            )
            
            logger.info(f"✅ Test {test_name} completed in {execution_time_ms:.2f} ms")
            return test_result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Test {test_name} failed: {e}")
            
            test_result = SDETestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=execution_time_ms,
                details={},
                error_message=str(e)
            )
            
            return test_result
    
    async def test_data_pipeline(self) -> Dict[str, Any]:
        """Test data pipeline functionality"""
        # Generate sample data
        sample_data = []
        base_price = 45000.0
        
        for i in range(100):
            timestamp = datetime.now(timezone.utc) - timedelta(minutes=100-i)
            price_change = (i % 10 - 5) * 10  # Random price movement
            price = base_price + price_change
            
            sample_data.append({
                'type': 'kline',
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'timestamp': timestamp,
                'open': price - 5,
                'high': price + 10,
                'low': price - 10,
                'close': price,
                'volume': 100 + (i % 50),
                'quote_volume': price * (100 + (i % 50)),
                'trades': 50 + (i % 20),
                'source': 'test'
            })
        
        # Process data through pipeline
        processed_count = 0
        for data in sample_data:
            success = await self.data_pipeline.process_websocket_message(data)
            if success:
                processed_count += 1
        
        # Calculate indicators
        await self.data_pipeline.calculate_technical_indicators(self.symbol, self.timeframe)
        
        # Get pipeline stats
        stats = self.data_pipeline.get_pipeline_stats()
        
        return {
            'sample_data_count': len(sample_data),
            'processed_count': processed_count,
            'processing_success_rate': (processed_count / len(sample_data)) * 100,
            'pipeline_stats': stats
        }
    
    async def test_sde_signal_generation(self) -> Dict[str, Any]:
        """Test SDE signal generation"""
        # Create mock signal generation request
        request = type('SignalGenerationRequest', (), {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_data': {
                'current_price': 45000.0,
                'indicators': {
                    'sma_20': 44800.0,
                    'sma_50': 44500.0,
                    'rsi_14': 35.2,
                    'macd': 0.85,
                    'atr_14': 1200.0
                }
            },
            'analysis_results': {
                'sentiment_analysis': {
                    'overall_sentiment': 0.3,
                    'confidence': 0.8
                },
                'volume_analysis': {
                    'volume_trend': 'increasing',
                    'volume_ratio': 1.5
                },
                'technical_analysis': {
                    'trend_direction': 'bullish',
                    'trend_strength': 0.7
                }
            },
            'timestamp': datetime.now(timezone.utc)
        })()
        
        # Generate signal
        signal_result = await self.sde_integration.generate_signal(request)
        
        # Get integration stats
        stats = self.sde_integration.get_integration_stats()
        
        return {
            'signal_generated': signal_result is not None,
            'signal_details': {
                'signal_id': signal_result.signal_id if signal_result else None,
                'direction': signal_result.direction if signal_result else None,
                'confidence': signal_result.confidence if signal_result else None,
                'strength': signal_result.strength if signal_result else None
            } if signal_result else None,
            'integration_stats': stats
        }
    
    async def test_perfect_calculations(self) -> Dict[str, Any]:
        """Test perfect calculations integration"""
        # Get sample data from pipeline
        ohlcv_data = await self.data_pipeline.get_latest_ohlcv_data(self.symbol, self.timeframe, 100)
        
        if not ohlcv_data:
            raise Exception("No OHLCV data available for testing")
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Test psychological levels
        psychological_levels = self.sr_analyzer.detect_psychological_levels(df, self.symbol)
        
        # Test support/resistance
        sr_levels = self.sr_analyzer.detect_dynamic_levels(df, self.symbol, self.timeframe)
        
        # Test volume patterns
        volume_patterns = await self.volume_analyzer.analyze_volume_patterns(df, self.symbol, self.timeframe)
        
        return {
            'data_points': len(df),
            'psychological_levels_count': len(psychological_levels),
            'support_levels_count': len([l for l in sr_levels if l.level_type.value == 'support']),
            'resistance_levels_count': len([l for l in sr_levels if l.level_type.value == 'resistance']),
            'volume_patterns_count': len(volume_patterns),
            'perfect_calculations_active': True
        }
    
    async def test_multi_timeframe_analysis(self) -> Dict[str, Any]:
        """Test multi-timeframe analysis"""
        # Generate data for multiple timeframes
        timeframes = ['1m', '5m', '15m', '1h']
        timeframe_results = {}
        
        for tf in timeframes:
            # Generate sample data for this timeframe
            sample_data = []
            base_price = 45000.0
            
            for i in range(50):
                timestamp = datetime.now(timezone.utc) - timedelta(minutes=50-i)
                price_change = (i % 10 - 5) * 10
                price = base_price + price_change
                
                sample_data.append({
                    'type': 'kline',
                    'symbol': self.symbol,
                    'timeframe': tf,
                    'timestamp': timestamp,
                    'open': price - 5,
                    'high': price + 10,
                    'low': price - 10,
                    'close': price,
                    'volume': 100 + (i % 50),
                    'source': 'test'
                })
            
            # Process data
            processed_count = 0
            for data in sample_data:
                success = await self.data_pipeline.process_websocket_message(data)
                if success:
                    processed_count += 1
            
            # Calculate indicators
            await self.data_pipeline.calculate_technical_indicators(self.symbol, tf)
            
            timeframe_results[tf] = {
                'data_points': len(sample_data),
                'processed_count': processed_count,
                'indicators_calculated': True
            }
        
        return {
            'timeframes_tested': len(timeframes),
            'timeframe_results': timeframe_results,
            'multi_timeframe_active': True
        }
    
    async def test_signal_quality_assessment(self) -> Dict[str, Any]:
        """Test signal quality assessment"""
        # Generate multiple signals with different qualities
        signal_requests = []
        
        # High quality signal
        signal_requests.append(type('SignalGenerationRequest', (), {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_data': {
                'current_price': 45000.0,
                'indicators': {
                    'sma_20': 44800.0,
                    'sma_50': 44500.0,
                    'rsi_14': 25.0,  # Oversold
                    'macd': 1.5,     # Strong bullish
                    'atr_14': 1200.0
                }
            },
            'analysis_results': {
                'sentiment_analysis': {'overall_sentiment': 0.7, 'confidence': 0.9},
                'volume_analysis': {'volume_trend': 'increasing', 'volume_ratio': 2.0},
                'technical_analysis': {'trend_direction': 'bullish', 'trend_strength': 0.9}
            },
            'timestamp': datetime.now(timezone.utc)
        })())
        
        # Medium quality signal
        signal_requests.append(type('SignalGenerationRequest', (), {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_data': {
                'current_price': 45000.0,
                'indicators': {
                    'sma_20': 45000.0,
                    'sma_50': 45000.0,
                    'rsi_14': 50.0,  # Neutral
                    'macd': 0.0,     # Neutral
                    'atr_14': 1200.0
                }
            },
            'analysis_results': {
                'sentiment_analysis': {'overall_sentiment': 0.0, 'confidence': 0.5},
                'volume_analysis': {'volume_trend': 'normal', 'volume_ratio': 1.0},
                'technical_analysis': {'trend_direction': 'neutral', 'trend_strength': 0.5}
            },
            'timestamp': datetime.now(timezone.utc)
        })())
        
        # Low quality signal
        signal_requests.append(type('SignalGenerationRequest', (), {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_data': {
                'current_price': 45000.0,
                'indicators': {
                    'sma_20': 45200.0,
                    'sma_50': 45500.0,
                    'rsi_14': 75.0,  # Overbought
                    'macd': -1.0,    # Bearish
                    'atr_14': 1200.0
                }
            },
            'analysis_results': {
                'sentiment_analysis': {'overall_sentiment': -0.3, 'confidence': 0.3},
                'volume_analysis': {'volume_trend': 'decreasing', 'volume_ratio': 0.5},
                'technical_analysis': {'trend_direction': 'bearish', 'trend_strength': 0.3}
            },
            'timestamp': datetime.now(timezone.utc)
        })())
        
        # Generate signals
        signal_results = []
        for i, request in enumerate(signal_requests):
            result = await self.sde_integration.generate_signal(request)
            signal_results.append({
                'signal_quality': ['high', 'medium', 'low'][i],
                'signal_generated': result is not None,
                'confidence': result.confidence if result else 0,
                'strength': result.strength if result else 0
            })
        
        return {
            'signals_tested': len(signal_requests),
            'signal_results': signal_results,
            'quality_assessment_active': True
        }
    
    async def run_all_tests(self) -> List[SDETestResult]:
        """Run all SDE framework tests"""
        logger.info("🚀 Starting SDE Framework Integration Tests")
        logger.info("=" * 60)
        
        # Define tests
        tests = [
            ("Data Pipeline Test", self.test_data_pipeline),
            ("SDE Signal Generation Test", self.test_sde_signal_generation),
            ("Perfect Calculations Test", self.test_perfect_calculations),
            ("Multi-Timeframe Analysis Test", self.test_multi_timeframe_analysis),
            ("Signal Quality Assessment Test", self.test_signal_quality_assessment)
        ]
        
        # Run tests
        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            self.test_results.append(result)
        
        return self.test_results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        avg_execution_time = sum(r.execution_time_ms for r in self.test_results) / total_tests
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': (successful_tests / total_tests) * 100,
                'avg_execution_time_ms': avg_execution_time
            },
            'test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'execution_time_ms': r.execution_time_ms,
                    'details': r.details,
                    'error_message': r.error_message
                }
                for r in self.test_results
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if not r.success]
        slow_tests = [r for r in self.test_results if r.execution_time_ms > 1000]
        
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests to ensure system reliability.")
        
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow tests for better performance.")
        
        successful_tests = [r for r in self.test_results if r.success]
        if len(successful_tests) == len(self.test_results):
            recommendations.append("All tests passed! SDE framework integration is working correctly.")
        
        return recommendations
    
    def save_test_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sde_test_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Test report saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving test report: {e}")

async def main():
    """Main function for SDE framework testing"""
    tester = SDEFrameworkTester()
    
    try:
        await tester.initialize()
        
        # Run all tests
        test_results = await tester.run_all_tests()
        
        # Generate report
        report = tester.generate_test_report()
        
        # Display results
        logger.info("=" * 60)
        logger.info("📊 SDE FRAMEWORK TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {report['summary']['total_tests']}")
        logger.info(f"Successful Tests: {report['summary']['successful_tests']}")
        logger.info(f"Failed Tests: {report['summary']['failed_tests']}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        logger.info(f"Average Execution Time: {report['summary']['avg_execution_time_ms']:.2f} ms")
        
        logger.info("\nTest Results:")
        for i, result in enumerate(report['test_results'], 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"  {i}. {result['test_name']}: {status} ({result['execution_time_ms']:.2f} ms)")
            if not result['success']:
                logger.info(f"     Error: {result['error_message']}")
        
        logger.info("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        
        # Save report
        tester.save_test_report(report)
        
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())
