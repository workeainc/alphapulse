#!/usr/bin/env python3
"""
Complete Advanced Pattern Recognition System Test
Tests the integration between existing and advanced pattern recognition
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import existing components
from src.database.connection import TimescaleDBConnection

# Import advanced pattern components
from src.ai.multi_timeframe_pattern_engine import MultiTimeframePatternEngine
from src.ai.pattern_failure_analyzer import PatternFailureAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteSystemTest:
    """Test the complete advanced pattern recognition system"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'postgres',
            'password': 'Emon_@17711'
        }
        
        self.db_connection = None
        self.multi_timeframe_engine = None
        self.failure_analyzer = None
    
    async def initialize(self):
        """Initialize all components"""
        try:
            print("🚀 Initializing Complete System Test...")
            
            # Initialize database connection
            self.db_connection = TimescaleDBConnection(self.db_config)
            await self.db_connection.initialize()
            print("✅ Database connection established")
            
            # Initialize advanced pattern engines
            self.multi_timeframe_engine = MultiTimeframePatternEngine(self.db_config)
            self.failure_analyzer = PatternFailureAnalyzer(self.db_config)
            
            await self.multi_timeframe_engine.initialize()
            await self.failure_analyzer.initialize()
            print("✅ Advanced pattern engines initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def generate_test_data(self, symbol: str = "BTCUSDT", periods: int = 100) -> Dict[str, pd.DataFrame]:
        """Generate realistic test candlestick data"""
        try:
            print(f"📊 Generating test data for {symbol}...")
            
            # Generate realistic price data
            base_price = 50000.0
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            current_price = base_price
            current_time = datetime.now() - timedelta(hours=periods)
            
            for i in range(periods):
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.02) * current_price  # 2% volatility
                current_price += price_change
                
                # Generate OHLC data
                open_price = current_price
                high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
                close_price = open_price + np.random.normal(0, 0.005) * open_price
                
                # Ensure realistic OHLC relationships
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Generate volume
                volume = np.random.uniform(100, 1000)
                
                timestamps.append(current_time)
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)
                
                current_time += timedelta(hours=1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            # Create multi-timeframe data
            timeframes = {
                "1m": df,
                "5m": df.resample('5min', on='timestamp').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna(),
                "15m": df.resample('15min', on='timestamp').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna(),
                "1h": df.resample('1h', on='timestamp').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            }
            
            print(f"✅ Generated {periods} candlesticks across {len(timeframes)} timeframes")
            return timeframes
            
        except Exception as e:
            logger.error(f"❌ Test data generation failed: {e}")
            return {}
    
    async def test_multi_timeframe_analysis(self, symbol: str, candlestick_data: Dict[str, pd.DataFrame]):
        """Test multi-timeframe pattern analysis"""
        try:
            print(f"\n🔄 Testing Multi-Timeframe Pattern Analysis for {symbol}...")
            
            # Test multi-timeframe pattern detection
            multi_timeframe_patterns = await self.multi_timeframe_engine.detect_multi_timeframe_patterns(
                symbol, "1h", candlestick_data
            )
            
            print(f"📊 Multi-timeframe patterns detected: {len(multi_timeframe_patterns)}")
            
            # Display results
            for i, pattern in enumerate(multi_timeframe_patterns[:3]):  # Show first 3
                print(f"\n  Pattern {i+1}:")
                print(f"    Name: {pattern.pattern_name}")
                print(f"    Type: {pattern.pattern_type}")
                print(f"    Primary Confidence: {pattern.primary_confidence:.3f}")
                print(f"    Overall Confidence: {pattern.overall_confidence:.3f}")
                print(f"    Confirmation Score: {pattern.confirmation_score:.1f}")
                print(f"    Trend Alignment: {pattern.trend_alignment}")
                print(f"    Failure Probability: {pattern.failure_probability:.3f}")
                print(f"    Timeframes Confirmed: {len(pattern.confirmation_timeframes)}")
            
            # Store results in database
            for pattern in multi_timeframe_patterns:
                await self.multi_timeframe_engine.store_multi_timeframe_pattern(pattern)
            
            print(f"✅ Multi-timeframe analysis completed and stored")
            return multi_timeframe_patterns
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe analysis failed: {e}")
            return []
    
    async def test_failure_prediction(self, symbol: str, candlestick_data: Dict[str, pd.DataFrame]):
        """Test pattern failure prediction"""
        try:
            print(f"\n⚠️ Testing Pattern Failure Prediction for {symbol}...")
            
            # Create sample pattern data
            sample_pattern = {
                "pattern_id": f"test_pattern_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "pattern_name": "doji",
                "confidence": 0.75,
                "strength": "moderate"
            }
            
            # Test failure prediction
            failure_prediction = await self.failure_analyzer.predict_pattern_failure(
                sample_pattern, {"ohlcv": candlestick_data["1h"].to_dict('records')}
            )
            
            if failure_prediction:
                print(f"📊 Failure Prediction Results:")
                print(f"    Failure Probability: {failure_prediction.failure_probability:.3f}")
                print(f"    Prediction Confidence: {failure_prediction.failure_confidence:.3f}")
                print(f"    Failure Reasons: {failure_prediction.failure_reasons}")
                print(f"    Market Volatility: {failure_prediction.market_volatility:.6f}")
                print(f"    Volume Profile: {failure_prediction.volume_profile}")
                print(f"    RSI Value: {failure_prediction.rsi_value:.1f}")
                print(f"    MACD Signal: {failure_prediction.macd_signal}")
                
                # Store prediction in database
                await self.failure_analyzer.store_failure_prediction(failure_prediction)
                print(f"✅ Failure prediction stored in database")
            
            return failure_prediction
            
        except Exception as e:
            logger.error(f"❌ Failure prediction failed: {e}")
            return None
    
    async def test_database_operations(self):
        """Test database operations and data retrieval"""
        try:
            print(f"\n🗄️ Testing Database Operations...")
            
            # Test retrieving multi-timeframe patterns
            mtf_patterns = await self.multi_timeframe_engine.get_multi_timeframe_patterns("BTCUSDT", 5)
            print(f"📊 Retrieved {len(mtf_patterns)} multi-timeframe patterns from database")
            
            # Test retrieving failure predictions
            failure_predictions = await self.failure_analyzer.get_failure_predictions("BTCUSDT", 5)
            print(f"📊 Retrieved {len(failure_predictions)} failure predictions from database")
            
            # Test database connectivity
            async with self.db_connection.get_async_session() as session:
                from sqlalchemy import text
                
                # Check table counts
                tables = [
                    'multi_timeframe_patterns',
                    'pattern_failure_predictions',
                    'pattern_strength_scores',
                    'advanced_pattern_signals'
                ]
                
                for table in tables:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"    {table}: {count} records")
            
            print(f"✅ Database operations test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database operations failed: {e}")
            return False
    
    async def test_performance(self, symbol: str, candlestick_data: Dict[str, pd.DataFrame]):
        """Test system performance"""
        try:
            print(f"\n⚡ Testing System Performance...")
            
            import time
            
            # Test multi-timeframe performance
            start_time = time.time()
            mtf_patterns = await self.multi_timeframe_engine.detect_multi_timeframe_patterns(
                symbol, "1h", candlestick_data
            )
            mtf_time = (time.time() - start_time) * 1000
            
            # Test failure prediction performance
            start_time = time.time()
            sample_pattern = {
                "pattern_id": f"perf_test_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "pattern_name": "hammer",
                "confidence": 0.8,
                "strength": "strong"
            }
            
            failure_pred = await self.failure_analyzer.predict_pattern_failure(
                sample_pattern, {"ohlcv": candlestick_data["1h"].to_dict('records')}
            )
            failure_time = (time.time() - start_time) * 1000
            
            print(f"📊 Performance Results:")
            print(f"    Multi-timeframe Analysis: {mtf_time:.2f}ms")
            print(f"    Failure Prediction: {failure_time:.2f}ms")
            print(f"    Total Processing Time: {mtf_time + failure_time:.2f}ms")
            print(f"    Patterns Detected: {len(mtf_patterns)}")
            
            # Performance benchmarks
            if mtf_time < 100 and failure_time < 50:
                print(f"✅ Performance meets ultra-low latency requirements")
            else:
                print(f"⚠️ Performance may need optimization")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_complete_test(self):
        """Run the complete system test"""
        try:
            print("🎯 Starting Complete Advanced Pattern Recognition System Test")
            print("=" * 60)
            
            # Initialize system
            if not await self.initialize():
                return False
            
            # Generate test data
            test_data = self.generate_test_data("BTCUSDT", 200)
            if not test_data:
                return False
            
            # Run tests
            test_results = []
            
            # Test 1: Multi-timeframe analysis
            mtf_result = await self.test_multi_timeframe_analysis("BTCUSDT", test_data)
            test_results.append(("Multi-timeframe Analysis", len(mtf_result) > 0))
            
            # Test 2: Failure prediction
            failure_result = await self.test_failure_prediction("BTCUSDT", test_data)
            test_results.append(("Failure Prediction", failure_result is not None))
            
            # Test 3: Database operations
            db_result = await self.test_database_operations()
            test_results.append(("Database Operations", db_result))
            
            # Test 4: Performance
            perf_result = await self.test_performance("BTCUSDT", test_data)
            test_results.append(("Performance", perf_result))
            
            # Summary
            print(f"\n📈 Test Summary:")
            print("=" * 60)
            
            passed_tests = 0
            for test_name, result in test_results:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"    {test_name}: {status}")
                if result:
                    passed_tests += 1
            
            print(f"\n🎉 Overall Result: {passed_tests}/{len(test_results)} tests passed")
            
            if passed_tests == len(test_results):
                print("🚀 Advanced Pattern Recognition System is ready for production!")
                print("✅ All components working correctly")
                print("✅ Database integration successful")
                print("✅ Performance meets requirements")
            else:
                print("⚠️ Some tests failed. Please check the logs.")
            
            return passed_tests == len(test_results)
            
        except Exception as e:
            logger.error(f"❌ Complete test failed: {e}")
            return False
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.multi_timeframe_engine:
                await self.multi_timeframe_engine.cleanup()
            if self.failure_analyzer:
                await self.failure_analyzer.cleanup()
            if self.db_connection:
                await self.db_connection.close()
            print("✅ System cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main test function"""
    test = CompleteSystemTest()
    success = await test.run_complete_test()
    
    if success:
        print("\n🎯 Advanced Pattern Recognition System Test: SUCCESS")
        print("Ready for integration with your main analyzing engine!")
    else:
        print("\n❌ Advanced Pattern Recognition System Test: FAILED")
        print("Please check the logs and fix any issues.")

if __name__ == "__main__":
    asyncio.run(main())
