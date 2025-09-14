#!/usr/bin/env python3
"""
Integration Test for AlphaPulse Trading System
Tests complete data flow from WebSocket to signal generation
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import os

# Update import paths for new structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from alphapulse_core import AlphaPulse, TradingSignal, SignalDirection, MarketRegime
from websocket_binance import BinanceWebSocketClient
from indicators_engine import TechnicalIndicators, IndicatorValues
from ml_signal_generator import MLSignalGenerator, CandlestickPattern, MarketRegime as MLMarketRegime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlphaPulseIntegrationTest:
    """Integration test for AlphaPulse system"""
    
    def __init__(self):
        """Initialize integration test"""
        self.test_results = {
            'websocket_tests': [],
            'indicator_tests': [],
            'ml_signal_tests': [],
            'integration_tests': [],
            'performance_tests': []
        }
        
        # Test data
        self.sample_candlesticks = self._generate_sample_data()
        
        logger.info("AlphaPulse Integration Test initialized")
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample candlestick data for testing"""
        base_price = 50000
        data = []
        
        for i in range(100):
            # Generate realistic price movements
            price_change = (i % 20 - 10) * 50  # Oscillating pattern
            volume_base = 1000 + (i % 10) * 100
            
            candle = {
                'timestamp': datetime.now() - timedelta(minutes=100-i),
                'open': base_price + price_change,
                'high': base_price + price_change + 100,
                'low': base_price + price_change - 100,
                'close': base_price + price_change + 50,
                'volume': volume_base + (i % 5) * 50,
                'is_complete': True
            }
            data.append(candle)
            base_price = candle['close']
        
        return data
    
    async def test_websocket_connection(self):
        """Test WebSocket connection and data parsing"""
        logger.info("🧪 Testing WebSocket Connection...")
        
        try:
            # Initialize WebSocket client
            client = BinanceWebSocketClient(
                symbols=["BTCUSDT"],
                timeframes=["1m"]
            )
            
            # Test connection
            connection_success = await client.connect()
            self.test_results['websocket_tests'].append({
                'test': 'connection',
                'status': 'PASS' if connection_success else 'FAIL',
                'message': 'WebSocket connection established' if connection_success else 'Connection failed'
            })
            
            if connection_success:
                # Test data parsing
                message_count = 0
                start_time = time.time()
                
                async for message in client.listen():
                    if message and message.get('type') == 'kline':
                        message_count += 1
                        
                        # Validate message structure
                        required_fields = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
                        if all(field in message for field in required_fields):
                            self.test_results['websocket_tests'].append({
                                'test': 'data_parsing',
                                'status': 'PASS',
                                'message': f'Successfully parsed {message_count} messages'
                            })
                        else:
                            self.test_results['websocket_tests'].append({
                                'test': 'data_parsing',
                                'status': 'FAIL',
                                'message': 'Invalid message structure'
                            })
                        
                        # Stop after 5 messages or 10 seconds
                        if message_count >= 5 or (time.time() - start_time) > 10:
                            break
                
                await client.disconnect()
            
        except Exception as e:
            self.test_results['websocket_tests'].append({
                'test': 'websocket',
                'status': 'FAIL',
                'message': f'WebSocket test error: {e}'
            })
    
    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        logger.info("🧪 Testing Technical Indicators...")
        
        try:
            # Initialize indicators
            indicators = TechnicalIndicators()
            
            # Test with sample data
            close_prices = [candle['close'] for candle in self.sample_candlesticks]
            
            for i, candle in enumerate(self.sample_candlesticks[50:60]):  # Test last 10 candles
                # Calculate indicators
                result = indicators.calculate_all_indicators(
                    open_price=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume'],
                    close_prices=close_prices[:50+i+1]  # Use available data
                )
                
                # Validate indicator values
                valid_rsi = 0 <= result.rsi <= 100
                valid_macd = isinstance(result.macd_line, (int, float))
                valid_bb = result.bb_upper > result.bb_middle > result.bb_lower
                valid_fib = (result.fib_236 < result.fib_382 < result.fib_500 < result.fib_618)
                
                if all([valid_rsi, valid_macd, valid_bb, valid_fib]):
                    self.test_results['indicator_tests'].append({
                        'test': f'indicator_calculation_{i}',
                        'status': 'PASS',
                        'message': f'All indicators calculated correctly for candle {i}'
                    })
                else:
                    self.test_results['indicator_tests'].append({
                        'test': f'indicator_calculation_{i}',
                        'status': 'FAIL',
                        'message': f'Invalid indicator values for candle {i}'
                    })
                
                # Test specific indicators
                self._test_rsi_calculation(indicators, close_prices[:50+i+1])
                self._test_macd_calculation(indicators, close_prices[:50+i+1])
                self._test_bollinger_bands(indicators, close_prices[:50+i+1])
            
        except Exception as e:
            self.test_results['indicator_tests'].append({
                'test': 'indicators',
                'status': 'FAIL',
                'message': f'Indicator test error: {e}'
            })
    
    def _test_rsi_calculation(self, indicators: TechnicalIndicators, close_prices: List[float]):
        """Test RSI calculation specifically"""
        try:
            rsi = indicators.calculate_rsi(close_prices)
            
            if 0 <= rsi <= 100:
                self.test_results['indicator_tests'].append({
                    'test': 'rsi_calculation',
                    'status': 'PASS',
                    'message': f'RSI calculated correctly: {rsi:.2f}'
                })
            else:
                self.test_results['indicator_tests'].append({
                    'test': 'rsi_calculation',
                    'status': 'FAIL',
                    'message': f'Invalid RSI value: {rsi}'
                })
        except Exception as e:
            self.test_results['indicator_tests'].append({
                'test': 'rsi_calculation',
                'status': 'FAIL',
                'message': f'RSI calculation error: {e}'
            })
    
    def _test_macd_calculation(self, indicators: TechnicalIndicators, close_prices: List[float]):
        """Test MACD calculation specifically"""
        try:
            macd_line, macd_signal, macd_histogram = indicators.calculate_macd(close_prices)
            
            if all(isinstance(x, (int, float)) for x in [macd_line, macd_signal, macd_histogram]):
                self.test_results['indicator_tests'].append({
                    'test': 'macd_calculation',
                    'status': 'PASS',
                    'message': f'MACD calculated correctly: {macd_line:.2f}, {macd_signal:.2f}, {macd_histogram:.2f}'
                })
            else:
                self.test_results['indicator_tests'].append({
                    'test': 'macd_calculation',
                    'status': 'FAIL',
                    'message': 'Invalid MACD values'
                })
        except Exception as e:
            self.test_results['indicator_tests'].append({
                'test': 'macd_calculation',
                'status': 'FAIL',
                'message': f'MACD calculation error: {e}'
            })
    
    def _test_bollinger_bands(self, indicators: TechnicalIndicators, close_prices: List[float]):
        """Test Bollinger Bands calculation specifically"""
        try:
            bb_upper, bb_middle, bb_lower = indicators.calculate_bollinger_bands(close_prices)
            
            if bb_upper > bb_middle > bb_lower:
                self.test_results['indicator_tests'].append({
                    'test': 'bollinger_bands',
                    'status': 'PASS',
                    'message': f'Bollinger Bands calculated correctly: {bb_upper:.2f}, {bb_middle:.2f}, {bb_lower:.2f}'
                })
            else:
                self.test_results['indicator_tests'].append({
                    'test': 'bollinger_bands',
                    'status': 'FAIL',
                    'message': 'Invalid Bollinger Bands values'
                })
        except Exception as e:
            self.test_results['indicator_tests'].append({
                'test': 'bollinger_bands',
                'status': 'FAIL',
                'message': f'Bollinger Bands calculation error: {e}'
            })
    
    async def test_ml_signal_generation(self):
        """Test ML signal generation"""
        logger.info("🧪 Testing ML Signal Generation...")
        
        try:
            # Initialize ML signal generator
            generator = MLSignalGenerator()
            
            # Test with sample data
            for i, candle in enumerate(self.sample_candlesticks[80:90]):  # Test last 10 candles
                # Create candlestick pattern
                candlestick_pattern = CandlestickPattern(
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume'],
                    timestamp=candle['timestamp']
                )
                
                # Create indicator values
                indicators = IndicatorValues(
                    rsi=45.0 + (i % 20),  # Varying RSI
                    macd_line=0.5,
                    macd_signal=0.3,
                    macd_histogram=0.2,
                    bb_upper=candle['close'] * 1.02,
                    bb_middle=candle['close'],
                    bb_lower=candle['close'] * 0.98,
                    pivot=(candle['high'] + candle['low'] + candle['close']) / 3,
                    s1=candle['low'] * 0.99,
                    r1=candle['high'] * 1.01,
                    fib_236=candle['low'] + (candle['high'] - candle['low']) * 0.236,
                    fib_382=candle['low'] + (candle['high'] - candle['low']) * 0.382,
                    fib_500=candle['low'] + (candle['high'] - candle['low']) * 0.5,
                    fib_618=candle['low'] + (candle['high'] - candle['low']) * 0.618,
                    breakout_strength=1.5 + (i % 10) * 0.1,
                    adx=25.0 + (i % 15),
                    atr=100.0 + (i % 50),
                    volume_sma=candle['volume'] * 0.8
                )
                
                # Generate signal
                signal = await generator.generate_signal(
                    candlestick=candlestick_pattern,
                    indicators=indicators,
                    market_regime=MLMarketRegime.TRENDING,
                    symbol="BTC/USDT",
                    timeframe="1m"
                )
                
                if signal:
                    # Validate signal structure
                    valid_signal = (
                        hasattr(signal, 'pattern_type') and
                        hasattr(signal, 'direction') and
                        hasattr(signal, 'confidence') and
                        0 <= signal.confidence <= 1
                    )
                    
                    if valid_signal:
                        self.test_results['ml_signal_tests'].append({
                            'test': f'signal_generation_{i}',
                            'status': 'PASS',
                            'message': f'Signal generated: {signal.pattern_type.value} - {signal.direction.value} (confidence: {signal.confidence:.2f})'
                        })
                    else:
                        self.test_results['ml_signal_tests'].append({
                            'test': f'signal_generation_{i}',
                            'status': 'FAIL',
                            'message': 'Invalid signal structure'
                        })
                else:
                    self.test_results['ml_signal_tests'].append({
                        'test': f'signal_generation_{i}',
                        'status': 'PASS',
                        'message': 'No signal generated (filtered out)'
                    })
            
            # Test performance stats
            stats = generator.get_performance_stats()
            if 'signals_generated' in stats and 'signals_filtered' in stats:
                self.test_results['ml_signal_tests'].append({
                    'test': 'performance_stats',
                    'status': 'PASS',
                    'message': f'Performance stats: {stats["signals_generated"]} generated, {stats["signals_filtered"]} filtered'
                })
            
        except Exception as e:
            self.test_results['ml_signal_tests'].append({
                'test': 'ml_signal_generation',
                'status': 'FAIL',
                'message': f'ML signal generation error: {e}'
            })
    
    async def test_full_integration(self):
        """Test complete AlphaPulse integration"""
        logger.info("🧪 Testing Full AlphaPulse Integration...")
        
        try:
            # Initialize AlphaPulse with mock data
            ap = AlphaPulse(
                symbols=["BTC/USDT"],
                timeframes=["1m"],
                redis_url="redis://localhost:6379"
            )
            
            # Add signal callback for testing
            signals_received = []
            
            async def signal_handler(signal: TradingSignal):
                signals_received.append(signal)
                logger.info(f"📊 Signal received: {signal.symbol} {signal.timeframe} {signal.direction.value}")
            
            ap.add_signal_callback(signal_handler)
            
            # Start AlphaPulse
            await ap.start()
            
            # Let it run for a short time
            await asyncio.sleep(5)
            
            # Check system status
            status = ap.get_system_status()
            if status['is_running']:
                self.test_results['integration_tests'].append({
                    'test': 'system_startup',
                    'status': 'PASS',
                    'message': 'AlphaPulse system started successfully'
                })
            else:
                self.test_results['integration_tests'].append({
                    'test': 'system_startup',
                    'status': 'FAIL',
                    'message': 'AlphaPulse system failed to start'
                })
            
            # Check performance stats
            perf_stats = ap.get_performance_stats()
            if 'total_ticks_processed' in perf_stats:
                self.test_results['integration_tests'].append({
                    'test': 'performance_tracking',
                    'status': 'PASS',
                    'message': f'Performance tracking working: {perf_stats["total_ticks_processed"]} ticks processed'
                })
            
            # Stop AlphaPulse
            await ap.stop()
            
            self.test_results['integration_tests'].append({
                'test': 'system_shutdown',
                'status': 'PASS',
                'message': 'AlphaPulse system stopped successfully'
            })
            
        except Exception as e:
            self.test_results['integration_tests'].append({
                'test': 'full_integration',
                'status': 'FAIL',
                'message': f'Integration test error: {e}'
            })
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("🧪 Testing Performance Benchmarks...")
        
        try:
            # Test indicator calculation speed
            indicators = TechnicalIndicators()
            close_prices = [candle['close'] for candle in self.sample_candlesticks]
            
            start_time = time.time()
            for i in range(100):  # 100 calculations
                result = indicators.calculate_all_indicators(
                    open_price=self.sample_candlesticks[i]['open'],
                    high=self.sample_candlesticks[i]['high'],
                    low=self.sample_candlesticks[i]['low'],
                    close=self.sample_candlesticks[i]['close'],
                    volume=self.sample_candlesticks[i]['volume'],
                    close_prices=close_prices[:i+1]
                )
            
            calculation_time = time.time() - start_time
            avg_time_per_calculation = calculation_time / 100 * 1000  # Convert to milliseconds
            
            if avg_time_per_calculation < 50:  # Target: <50ms
                self.test_results['performance_tests'].append({
                    'test': 'indicator_calculation_speed',
                    'status': 'PASS',
                    'message': f'Indicator calculation speed: {avg_time_per_calculation:.2f}ms per calculation'
                })
            else:
                self.test_results['performance_tests'].append({
                    'test': 'indicator_calculation_speed',
                    'status': 'FAIL',
                    'message': f'Indicator calculation too slow: {avg_time_per_calculation:.2f}ms per calculation'
                })
            
            # Test ML signal generation speed
            generator = MLSignalGenerator()
            
            start_time = time.time()
            for i in range(50):  # 50 signal generations
                candlestick_pattern = CandlestickPattern(
                    open=self.sample_candlesticks[i]['open'],
                    high=self.sample_candlesticks[i]['high'],
                    low=self.sample_candlesticks[i]['low'],
                    close=self.sample_candlesticks[i]['close'],
                    volume=self.sample_candlesticks[i]['volume'],
                    timestamp=self.sample_candlesticks[i]['timestamp']
                )
                
                indicators = IndicatorValues(
                    rsi=50.0,
                    macd_line=0.0,
                    macd_signal=0.0,
                    macd_histogram=0.0,
                    bb_upper=self.sample_candlesticks[i]['close'] * 1.02,
                    bb_middle=self.sample_candlesticks[i]['close'],
                    bb_lower=self.sample_candlesticks[i]['close'] * 0.98,
                    pivot=(self.sample_candlesticks[i]['high'] + self.sample_candlesticks[i]['low'] + self.sample_candlesticks[i]['close']) / 3,
                    s1=self.sample_candlesticks[i]['low'] * 0.99,
                    r1=self.sample_candlesticks[i]['high'] * 1.01,
                    fib_236=self.sample_candlesticks[i]['low'] + (self.sample_candlesticks[i]['high'] - self.sample_candlesticks[i]['low']) * 0.236,
                    fib_382=self.sample_candlesticks[i]['low'] + (self.sample_candlesticks[i]['high'] - self.sample_candlesticks[i]['low']) * 0.382,
                    fib_500=self.sample_candlesticks[i]['low'] + (self.sample_candlesticks[i]['high'] - self.sample_candlesticks[i]['low']) * 0.5,
                    fib_618=self.sample_candlesticks[i]['low'] + (self.sample_candlesticks[i]['high'] - self.sample_candlesticks[i]['low']) * 0.618,
                    breakout_strength=1.5,
                    adx=25.0,
                    atr=100.0,
                    volume_sma=self.sample_candlesticks[i]['volume'] * 0.8
                )
                
                await generator.generate_signal(
                    candlestick=candlestick_pattern,
                    indicators=indicators,
                    market_regime=MLMarketRegime.TRENDING,
                    symbol="BTC/USDT",
                    timeframe="1m"
                )
            
            signal_time = time.time() - start_time
            avg_time_per_signal = signal_time / 50 * 1000  # Convert to milliseconds
            
            if avg_time_per_signal < 100:  # Target: <100ms
                self.test_results['performance_tests'].append({
                    'test': 'signal_generation_speed',
                    'status': 'PASS',
                    'message': f'Signal generation speed: {avg_time_per_signal:.2f}ms per signal'
                })
            else:
                self.test_results['performance_tests'].append({
                    'test': 'signal_generation_speed',
                    'status': 'FAIL',
                    'message': f'Signal generation too slow: {avg_time_per_signal:.2f}ms per signal'
                })
            
        except Exception as e:
            self.test_results['performance_tests'].append({
                'test': 'performance_benchmarks',
                'status': 'FAIL',
                'message': f'Performance benchmark error: {e}'
            })
    
    def print_test_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("🧪 ALPHAPULSE INTEGRATION TEST RESULTS")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_category, tests in self.test_results.items():
            print(f"\n📋 {test_category.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            for test in tests:
                status_icon = "✅" if test['status'] == 'PASS' else "❌"
                print(f"{status_icon} {test['test']}: {test['status']}")
                print(f"   {test['message']}")
                
                total_tests += 1
                if test['status'] == 'PASS':
                    passed_tests += 1
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "   Success Rate: 0%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! AlphaPulse system is ready for production.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please review and fix issues.")
        
        print("="*80)


async def main():
    """Run all integration tests"""
    logger.info("🚀 Starting AlphaPulse Integration Tests...")
    
    # Initialize test suite
    test_suite = AlphaPulseIntegrationTest()
    
    # Run tests
    await test_suite.test_websocket_connection()
    test_suite.test_technical_indicators()
    await test_suite.test_ml_signal_generation()
    await test_suite.test_full_integration()
    await test_suite.test_performance_benchmarks()
    
    # Print results
    test_suite.print_test_results()


if __name__ == "__main__":
    asyncio.run(main())
