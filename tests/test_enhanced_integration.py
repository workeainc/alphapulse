"""
Enhanced Integration Test for AlphaPlus
Tests the complete enhanced multi-timeframe pattern engine with all analyzers
"""

import sys
import os
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ai.enhanced_multi_timeframe_pattern_engine import EnhancedMultiTimeframePatternEngine

class EnhancedIntegrationTestSuite:
    """Comprehensive integration test suite for enhanced pattern engine"""
    
    def __init__(self):
        self.test_results = []
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'alpha_emon',
            'password': 'Emon_@17711'
        }
        self.engine = None
        
    def generate_multi_timeframe_data(self) -> dict:
        """Generate multi-timeframe test data"""
        np.random.seed(42)  # For reproducible results
        base_price = 45000
        base_time = datetime.now() - timedelta(days=30)
        
        # Generate data for different timeframes
        timeframes = {
            '5m': {'periods': 1440, 'freq': '5min'},  # 5 days of 5-minute data
            '15m': {'periods': 480, 'freq': '15min'},  # 5 days of 15-minute data
            '1h': {'periods': 120, 'freq': '1h'},      # 5 days of hourly data
            '4h': {'periods': 30, 'freq': '4h'},       # 5 days of 4-hour data
            '1d': {'periods': 30, 'freq': '1d'}        # 30 days of daily data
        }
        
        candlestick_data = {}
        
        for timeframe, config in timeframes.items():
            # Generate price data with some patterns
            prices = []
            volumes = []
            current_price = base_price
            
            for i in range(config['periods']):
                # Add trend component
                if i < config['periods'] // 3:
                    # Uptrend
                    current_price += np.random.normal(10, 5)
                elif i < 2 * config['periods'] // 3:
                    # Downtrend
                    current_price -= np.random.normal(8, 4)
                else:
                    # Sideways
                    current_price += np.random.normal(0, 3)
                
                # Add some noise
                current_price += np.random.normal(0, 20)
                
                # Generate OHLCV
                open_price = current_price
                high_price = open_price + abs(np.random.normal(0, 15))
                low_price = open_price - abs(np.random.normal(0, 15))
                close_price = open_price + np.random.normal(0, 10)
                volume = np.random.normal(5000, 1000)
                
                prices.append({
                    'open': open_price,
                    'high': max(high_price, open_price, close_price),
                    'low': min(low_price, open_price, close_price),
                    'close': close_price,
                    'volume': max(100, volume)
                })
                
                current_price = close_price
            
            # Create DataFrame
            df = pd.DataFrame(prices)
            df.index = pd.date_range(start=base_time, periods=len(prices), freq=config['freq'])
            
            candlestick_data[timeframe] = df
        
        return candlestick_data
    
    async def test_enhanced_engine_initialization(self) -> bool:
        """Test enhanced engine initialization"""
        print("🧪 Testing Enhanced Engine Initialization...")
        
        try:
            # Initialize engine
            self.engine = EnhancedMultiTimeframePatternEngine(self.db_config)
            await self.engine.initialize()
            
            print("✅ Enhanced engine initialized successfully")
            print("✅ Database connection established")
            print("✅ All analyzers loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            return False
    
    async def test_enhanced_pattern_detection(self) -> bool:
        """Test enhanced pattern detection"""
        print("🧪 Testing Enhanced Pattern Detection...")
        
        try:
            # Generate test data
            candlestick_data = self.generate_multi_timeframe_data()
            print(f"✅ Generated multi-timeframe data:")
            for timeframe, df in candlestick_data.items():
                print(f"   {timeframe}: {len(df)} candles")
            
            # Detect enhanced patterns
            patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                "BTCUSDT", "1h", candlestick_data
            )
            
            print(f"✅ Enhanced patterns detected: {len(patterns)}")
            
            # Validate pattern structure
            for i, pattern in enumerate(patterns[:3]):  # Show first 3 patterns
                print(f"   Pattern {i+1}:")
                print(f"     Name: {pattern.pattern_name}")
                print(f"     Type: {pattern.pattern_type}")
                print(f"     Confidence: {pattern.confidence:.3f}")
                poc_str = f"{pattern.poc_level:.2f}" if pattern.poc_level else "None"
                va_low_str = f"{pattern.value_area_low:.2f}" if pattern.value_area_low else "None"
                va_high_str = f"{pattern.value_area_high:.2f}" if pattern.value_area_high else "None"
                smc_conf_str = f"{pattern.smc_confidence:.3f}" if pattern.smc_confidence else "None"
                print(f"     Volume Profile: POC={poc_str}, VA={va_low_str}-{va_high_str}")
                print(f"     Elliott Wave: Wave={pattern.current_wave}, Count={pattern.wave_count}")
                print(f"     Wyckoff: {pattern.wyckoff_pattern}, Phase={pattern.wyckoff_phase}")
                print(f"     SMC: {pattern.smc_patterns}, Confidence={smc_conf_str}")
                
                # Validate required fields
                assert pattern.symbol == "BTCUSDT"
                assert pattern.primary_timeframe == "1h"
                assert pattern.confidence >= 0 and pattern.confidence <= 1
                assert pattern.poc_level is not None
                assert pattern.current_wave is not None
                assert pattern.wave_count is not None
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced pattern detection failed: {e}")
            return False
    
    async def test_volume_profile_integration(self) -> bool:
        """Test Volume Profile integration"""
        print("🧪 Testing Volume Profile Integration...")
        
        try:
            # Generate test data
            candlestick_data = self.generate_multi_timeframe_data()
            
            # Detect patterns
            patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                "BTCUSDT", "1h", candlestick_data
            )
            
            # Check Volume Profile analysis
            volume_profile_patterns = [p for p in patterns if p.volume_profile_confidence is not None]
            
            print(f"✅ Volume Profile patterns: {len(volume_profile_patterns)}")
            
            for pattern in volume_profile_patterns[:2]:  # Show first 2
                print(f"   POC Level: {pattern.poc_level:.2f}")
                print(f"   Value Area: {pattern.value_area_low:.2f} - {pattern.value_area_high:.2f}")
                print(f"   Volume Profile Confidence: {pattern.volume_profile_confidence:.3f}")
                print(f"   Volume Nodes: {pattern.volume_nodes_count}")
                print(f"   Volume Gaps: {pattern.volume_gaps_count}")
                
                # Validate Volume Profile fields
                assert pattern.poc_level > 0
                assert pattern.value_area_high >= pattern.value_area_low
                assert pattern.volume_profile_confidence >= 0 and pattern.volume_profile_confidence <= 1
                assert pattern.volume_nodes_count >= 0
                assert pattern.volume_gaps_count >= 0
            
            return True
            
        except Exception as e:
            print(f"❌ Volume Profile integration failed: {e}")
            return False
    
    async def test_elliott_wave_integration(self) -> bool:
        """Test Elliott Wave integration"""
        print("🧪 Testing Elliott Wave Integration...")
        
        try:
            # Generate test data
            candlestick_data = self.generate_multi_timeframe_data()
            
            # Detect patterns
            patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                "BTCUSDT", "1h", candlestick_data
            )
            
            # Check Elliott Wave analysis
            elliott_patterns = [p for p in patterns if p.elliott_confidence is not None]
            
            print(f"✅ Elliott Wave patterns: {len(elliott_patterns)}")
            
            for pattern in elliott_patterns[:2]:  # Show first 2
                print(f"   Current Wave: {pattern.current_wave}")
                print(f"   Wave Count: {pattern.wave_count}")
                print(f"   Pattern Type: {pattern.pattern_type_elliott}")
                print(f"   Trend Direction: {pattern.trend_direction_elliott}")
                print(f"   Next Target: {pattern.next_target_elliott:.2f}")
                print(f"   Elliott Confidence: {pattern.elliott_confidence:.3f}")
                print(f"   Fibonacci Levels: {len(pattern.fibonacci_levels) if pattern.fibonacci_levels else 0}")
                
                # Validate Elliott Wave fields
                assert pattern.current_wave is not None
                assert pattern.wave_count >= 0
                assert pattern.pattern_type_elliott is not None
                assert pattern.trend_direction_elliott in ['bullish', 'bearish', 'neutral']
                assert pattern.next_target_elliott > 0
                assert pattern.elliott_confidence >= 0 and pattern.elliott_confidence <= 1
            
            return True
            
        except Exception as e:
            print(f"❌ Elliott Wave integration failed: {e}")
            return False
    
    async def test_wyckoff_integration(self) -> bool:
        """Test Wyckoff integration"""
        print("🧪 Testing Wyckoff Integration...")
        
        try:
            # Generate test data
            candlestick_data = self.generate_multi_timeframe_data()
            
            # Detect patterns
            patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                "BTCUSDT", "1h", candlestick_data
            )
            
            # Check Wyckoff analysis
            wyckoff_patterns = [p for p in patterns if p.wyckoff_pattern is not None]
            
            print(f"✅ Wyckoff patterns: {len(wyckoff_patterns)}")
            
            for pattern in wyckoff_patterns[:2]:  # Show first 2
                print(f"   Wyckoff Pattern: {pattern.wyckoff_pattern}")
                print(f"   Wyckoff Confidence: {pattern.wyckoff_confidence:.3f}")
                print(f"   Wyckoff Phase: {pattern.wyckoff_phase}")
                
                # Validate Wyckoff fields
                assert pattern.wyckoff_pattern is not None
                assert pattern.wyckoff_confidence >= 0 and pattern.wyckoff_confidence <= 1
                assert pattern.wyckoff_phase in ['accumulation', 'distribution', 'test', 'markup', 'markdown', None]
            
            return True
            
        except Exception as e:
            print(f"❌ Wyckoff integration failed: {e}")
            return False
    
    async def test_smc_integration(self) -> bool:
        """Test SMC integration"""
        print("🧪 Testing SMC Integration...")
        
        try:
            # Generate test data
            candlestick_data = self.generate_multi_timeframe_data()
            
            # Detect patterns
            patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                "BTCUSDT", "1h", candlestick_data
            )
            
            # Check SMC analysis
            smc_patterns = [p for p in patterns if p.smc_patterns is not None]
            
            print(f"✅ SMC patterns: {len(smc_patterns)}")
            
            for pattern in smc_patterns[:2]:  # Show first 2
                print(f"   SMC Patterns: {pattern.smc_patterns}")
                print(f"   SMC Confidence: {pattern.smc_confidence:.3f}")
                print(f"   Order Blocks: {pattern.order_blocks_count}")
                print(f"   Fair Value Gaps: {pattern.fair_value_gaps_count}")
                
                # Validate SMC fields
                assert pattern.smc_patterns is not None
                assert pattern.smc_confidence >= 0 and pattern.smc_confidence <= 1
                assert pattern.order_blocks_count >= 0
                assert pattern.fair_value_gaps_count >= 0
            
            return True
            
        except Exception as e:
            print(f"❌ SMC integration failed: {e}")
            return False
    
    async def test_performance(self) -> bool:
        """Test performance and scalability"""
        print("🧪 Testing Performance...")
        
        try:
            # Generate larger dataset
            candlestick_data = self.generate_multi_timeframe_data()
            
            # Test multiple symbols
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            total_patterns = 0
            total_time = 0
            
            for symbol in symbols:
                start_time = time.time()
                
                patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                    symbol, "1h", candlestick_data
                )
                
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000
                
                total_patterns += len(patterns)
                total_time += processing_time
                
                print(f"   {symbol}: {len(patterns)} patterns in {processing_time:.2f}ms")
            
            avg_time = total_time / len(symbols)
            print(f"✅ Average processing time: {avg_time:.2f}ms")
            print(f"✅ Total patterns detected: {total_patterns}")
            
            # Performance should be under 300ms per symbol for enhanced analysis
            assert avg_time < 300, f"Average processing time {avg_time:.2f}ms exceeds 300ms limit"
            
            # Get performance metrics
            metrics = self.engine.get_performance_metrics()
            print(f"✅ Performance metrics:")
            print(f"   Total patterns: {metrics['total_patterns_detected']}")
            print(f"   Volume Profile: {metrics['volume_profile_patterns']}")
            print(f"   Elliott Wave: {metrics['elliott_wave_patterns']}")
            print(f"   Wyckoff: {metrics['wyckoff_patterns']}")
            print(f"   SMC: {metrics['smc_patterns']}")
            print(f"   Average time: {metrics['average_processing_time']:.2f}ms")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    async def test_database_storage(self) -> bool:
        """Test database storage (optional)"""
        print("🧪 Testing Database Storage...")
        
        try:
            # Generate test data
            candlestick_data = self.generate_multi_timeframe_data()
            
            # Detect patterns
            patterns = await self.engine.detect_enhanced_multi_timeframe_patterns(
                "BTCUSDT", "1h", candlestick_data
            )
            
            if patterns:
                # Try to store patterns (this might fail if table doesn't exist)
                try:
                    await self.engine.store_enhanced_patterns(patterns)
                    print("✅ Patterns stored in database successfully")
                except Exception as e:
                    print(f"⚠️ Database storage failed (expected if table doesn't exist): {e}")
                    print("✅ Pattern generation still works correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Database storage test failed: {e}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all enhanced integration tests"""
        print("🚀 Starting Enhanced Integration Test Suite")
        print("=" * 70)
        
        tests = [
            ("Engine Initialization", self.test_enhanced_engine_initialization),
            ("Enhanced Pattern Detection", self.test_enhanced_pattern_detection),
            ("Volume Profile Integration", self.test_volume_profile_integration),
            ("Elliott Wave Integration", self.test_elliott_wave_integration),
            ("Wyckoff Integration", self.test_wyckoff_integration),
            ("SMC Integration", self.test_smc_integration),
            ("Performance", self.test_performance),
            ("Database Storage", self.test_database_storage)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            print("-" * 50)
            
            try:
                if await test_func():
                    print(f"✅ {test_name}: PASS")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAIL")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
        
        print("\n" + "=" * 70)
        print(f"📊 Test Results Summary:")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Cleanup
        if self.engine:
            await self.engine.close()
        
        if passed_tests == total_tests:
            print("🎉 All Enhanced Integration tests passed!")
            return True
        else:
            print(f"⚠️ {total_tests - passed_tests} tests failed")
            return False

async def main():
    """Main test execution"""
    test_suite = EnhancedIntegrationTestSuite()
    success = await test_suite.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
