#!/usr/bin/env python3
"""
Test Wyckoff Enhancement Implementation
Validates Wyckoff pattern detection and integration with existing systems
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from src.data.volume_analyzer import VolumeAnalyzer, VolumePatternType
from src.ai.multi_timeframe_pattern_engine import MultiTimeframePatternEngine
from src.database.connection import TimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WyckoffEnhancementTester:
    """Test suite for Wyckoff pattern detection enhancement"""
    
    def __init__(self):
        self.volume_analyzer = VolumeAnalyzer()
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        }
        self.multi_timeframe_engine = MultiTimeframePatternEngine(self.db_config)
        
    async def initialize(self):
        """Initialize test environment"""
        try:
            await self.multi_timeframe_engine.initialize()
            logger.info("✅ Test environment initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize test environment: {e}")
            raise
    
    def generate_test_data(self, pattern_type: str) -> pd.DataFrame:
        """Generate test data for different Wyckoff patterns"""
        np.random.seed(42)  # For reproducible results
        
        # Base price data
        base_price = 45000
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        
        if pattern_type == 'spring':
            # Generate Spring pattern: breakdown below support then recovery
            prices = []
            support_level = base_price - 1000  # Clear support level
            
            for i in range(100):
                if i < 75:
                    # Normal trading around support
                    price = support_level + np.random.normal(0, 200)
                elif i < 80:
                    # Breakdown below support
                    price = support_level - 300 + np.random.normal(0, 50)
                elif i < 85:
                    # Quick recovery above support
                    price = support_level + 200 + np.random.normal(0, 50)
                else:
                    # Continued recovery
                    price = support_level + 400 + np.random.normal(0, 100)
                prices.append(price)
                
        elif pattern_type == 'upthrust':
            # Generate Upthrust pattern: breakout above resistance then rejection
            prices = []
            resistance_level = base_price + 1000  # Clear resistance level
            
            for i in range(100):
                if i < 75:
                    # Normal trading around resistance
                    price = resistance_level + np.random.normal(0, 200)
                elif i < 80:
                    # Breakout above resistance
                    price = resistance_level + 300 + np.random.normal(0, 50)
                elif i < 85:
                    # Quick rejection below resistance
                    price = resistance_level - 200 + np.random.normal(0, 50)
                else:
                    # Continued rejection
                    price = resistance_level - 400 + np.random.normal(0, 100)
                prices.append(price)
                
        elif pattern_type == 'accumulation':
            # Generate Accumulation pattern: price stability near support
            prices = []
            for i in range(100):
                if i < 70:
                    # Downtrend to support
                    price = base_price - (100 - i) * 10 + np.random.normal(0, 50)
                else:
                    # Stable trading near support
                    price = base_price - 1000 + np.random.normal(0, 30)
                prices.append(price)
                
        elif pattern_type == 'distribution':
            # Generate Distribution pattern: price stability near resistance
            prices = []
            for i in range(100):
                if i < 70:
                    # Uptrend to resistance
                    price = base_price + (100 - i) * 10 + np.random.normal(0, 50)
                else:
                    # Stable trading near resistance
                    price = base_price + 1000 + np.random.normal(0, 30)
                prices.append(price)
                
        else:
            # Default: random walk
            prices = [base_price + np.random.normal(0, 100) for _ in range(100)]
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 20)) for p in prices],
            'low': [p - abs(np.random.normal(0, 20)) for p in prices],
            'close': [p + np.random.normal(0, 10) for p in prices],
            'volume': [np.random.randint(1000, 10000) for _ in prices]
        })
        
        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    async def test_wyckoff_spring_detection(self):
        """Test Wyckoff Spring pattern detection"""
        logger.info("🧪 Testing Wyckoff Spring pattern detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('spring')
            
            # Detect Wyckoff patterns
            patterns = self.volume_analyzer.detect_wyckoff_patterns(df, 'BTCUSDT', '1h')
            
            # Check for Spring pattern
            spring_patterns = [p for p in patterns if p.pattern_type == VolumePatternType.WYCKOFF_SPRING]
            
            if spring_patterns:
                spring_pattern = spring_patterns[0]
                logger.info(f"✅ Spring pattern detected with confidence: {spring_pattern.confidence:.2f}")
                logger.info(f"   Support level: {spring_pattern.pattern_data.get('support_level', 'N/A')}")
                logger.info(f"   Breakdown depth: {spring_pattern.pattern_data.get('breakdown_depth', 'N/A'):.4f}")
                return True
            else:
                logger.warning("⚠️ No Spring pattern detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Spring pattern test failed: {e}")
            return False
    
    async def test_wyckoff_upthrust_detection(self):
        """Test Wyckoff Upthrust pattern detection"""
        logger.info("🧪 Testing Wyckoff Upthrust pattern detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('upthrust')
            
            # Detect Wyckoff patterns
            patterns = self.volume_analyzer.detect_wyckoff_patterns(df, 'BTCUSDT', '1h')
            
            # Check for Upthrust pattern
            upthrust_patterns = [p for p in patterns if p.pattern_type == VolumePatternType.WYCKOFF_UPTHRUST]
            
            if upthrust_patterns:
                upthrust_pattern = upthrust_patterns[0]
                logger.info(f"✅ Upthrust pattern detected with confidence: {upthrust_pattern.confidence:.2f}")
                logger.info(f"   Resistance level: {upthrust_pattern.pattern_data.get('resistance_level', 'N/A')}")
                logger.info(f"   Breakout height: {upthrust_pattern.pattern_data.get('breakout_height', 'N/A'):.4f}")
                return True
            else:
                logger.warning("⚠️ No Upthrust pattern detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Upthrust pattern test failed: {e}")
            return False
    
    async def test_wyckoff_accumulation_detection(self):
        """Test Wyckoff Accumulation pattern detection"""
        logger.info("🧪 Testing Wyckoff Accumulation pattern detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('accumulation')
            
            # Detect Wyckoff patterns
            patterns = self.volume_analyzer.detect_wyckoff_patterns(df, 'BTCUSDT', '1h')
            
            # Check for Accumulation pattern
            accumulation_patterns = [p for p in patterns if p.pattern_type == VolumePatternType.WYCKOFF_ACCUMULATION]
            
            if accumulation_patterns:
                accumulation_pattern = accumulation_patterns[0]
                logger.info(f"✅ Accumulation pattern detected with confidence: {accumulation_pattern.confidence:.2f}")
                logger.info(f"   Support level: {accumulation_pattern.pattern_data.get('support_level', 'N/A')}")
                logger.info(f"   Support touches: {accumulation_pattern.pattern_data.get('support_touches', 'N/A')}")
                return True
            else:
                logger.warning("⚠️ No Accumulation pattern detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Accumulation pattern test failed: {e}")
            return False
    
    async def test_wyckoff_distribution_detection(self):
        """Test Wyckoff Distribution pattern detection"""
        logger.info("🧪 Testing Wyckoff Distribution pattern detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('distribution')
            
            # Detect Wyckoff patterns
            patterns = self.volume_analyzer.detect_wyckoff_patterns(df, 'BTCUSDT', '1h')
            
            # Check for Distribution pattern
            distribution_patterns = [p for p in patterns if p.pattern_type == VolumePatternType.WYCKOFF_DISTRIBUTION]
            
            if distribution_patterns:
                distribution_pattern = distribution_patterns[0]
                logger.info(f"✅ Distribution pattern detected with confidence: {distribution_pattern.confidence:.2f}")
                logger.info(f"   Resistance level: {distribution_pattern.pattern_data.get('resistance_level', 'N/A')}")
                logger.info(f"   Resistance touches: {distribution_pattern.pattern_data.get('resistance_touches', 'N/A')}")
                return True
            else:
                logger.warning("⚠️ No Distribution pattern detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Distribution pattern test failed: {e}")
            return False
    
    async def test_multi_timeframe_integration(self):
        """Test Wyckoff patterns integration with multi-timeframe engine"""
        logger.info("🧪 Testing Wyckoff patterns integration with multi-timeframe engine...")
        
        try:
            # Generate test data for multiple timeframes
            candlestick_data = {}
            for timeframe in ['1h', '4h', '1d']:
                df = self.generate_test_data('spring')  # Use spring pattern for testing
                candlestick_data[timeframe] = df
            
            # Detect multi-timeframe patterns
            patterns = await self.multi_timeframe_engine.detect_multi_timeframe_patterns(
                'BTCUSDT', '1h', candlestick_data
            )
            
            # Check for Wyckoff patterns in results
            wyckoff_patterns = [p for p in patterns if 'wyckoff' in p.pattern_name.lower()]
            
            if wyckoff_patterns:
                wyckoff_pattern = wyckoff_patterns[0]
                logger.info(f"✅ Wyckoff pattern detected in multi-timeframe analysis")
                logger.info(f"   Pattern: {wyckoff_pattern.pattern_name}")
                logger.info(f"   Overall confidence: {wyckoff_pattern.overall_confidence:.2f}")
                logger.info(f"   Confirmation score: {wyckoff_pattern.confirmation_score:.2f}")
                logger.info(f"   Confirmation timeframes: {wyckoff_pattern.confirmation_timeframes}")
                return True
            else:
                logger.warning("⚠️ No Wyckoff patterns detected in multi-timeframe analysis")
                return False
                
        except Exception as e:
            logger.error(f"❌ Multi-timeframe integration test failed: {e}")
            return False
    
    async def test_performance(self):
        """Test performance of Wyckoff pattern detection"""
        logger.info("🧪 Testing Wyckoff pattern detection performance...")
        
        try:
            start_time = datetime.now()
            
            # Generate test data
            df = self.generate_test_data('spring')
            
            # Run pattern detection multiple times
            for _ in range(10):
                patterns = self.volume_analyzer.detect_wyckoff_patterns(df, 'BTCUSDT', '1h')
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"✅ Performance test completed")
            logger.info(f"   Average processing time: {processing_time / 10:.2f}ms per detection")
            
            if processing_time / 10 < 100:  # Should be under 100ms
                logger.info("✅ Performance meets requirements (<100ms)")
                return True
            else:
                logger.warning(f"⚠️ Performance exceeds requirements: {processing_time / 10:.2f}ms")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Wyckoff enhancement tests"""
        logger.info("🚀 Starting Wyckoff Enhancement Test Suite")
        logger.info("=" * 60)
        
        test_results = []
        
        # Initialize test environment
        await self.initialize()
        
        # Run individual pattern tests
        test_results.append(await self.test_wyckoff_spring_detection())
        test_results.append(await self.test_wyckoff_upthrust_detection())
        test_results.append(await self.test_wyckoff_accumulation_detection())
        test_results.append(await self.test_wyckoff_distribution_detection())
        
        # Run integration tests
        test_results.append(await self.test_multi_timeframe_integration())
        test_results.append(await self.test_performance())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        logger.info("=" * 60)
        logger.info(f"📊 Test Results Summary:")
        logger.info(f"   Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 All Wyckoff enhancement tests passed!")
            return True
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
            return False

async def main():
    """Main test execution"""
    tester = WyckoffEnhancementTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("✅ Wyckoff Enhancement Implementation: SUCCESS")
        return 0
    else:
        logger.error("❌ Wyckoff Enhancement Implementation: FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
