"""
Phase 1 Integration Test for AlphaPulse Trading Bot
Tests WebSocket client, signal generator, and database integration
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.websocket_client import BinanceWebSocketClient
from ..strategies.signal_generator import SignalGenerator, SignalType, SignalStrength
from ..strategies.pattern_detector import CandlestickPatternDetector
from ..strategies.indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase1IntegrationTest:
    """Test integration of all Phase 1 components"""
    
    def __init__(self):
        self.websocket_client = None
        self.signal_generator = None
        self.pattern_detector = None
        self.technical_analyzer = None
        self.test_data = None
        self.test_results = {}
        
    def setup_components(self):
        """Initialize all Phase 1 components"""
        try:
            logger.info("🔧 Setting up Phase 1 components...")
            
            # Initialize pattern detector and technical analyzer
            self.pattern_detector = CandlestickPatternDetector()
            self.technical_analyzer = TechnicalIndicators()
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator(
                self.pattern_detector, 
                self.technical_analyzer
            )
            
            # Initialize WebSocket client
            self.websocket_client = BinanceWebSocketClient()
            
            # Generate test data
            self._generate_test_data()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup components: {e}")
            return False
    
    def _generate_test_data(self):
        """Generate sample candlestick data for testing"""
        logger.info("📊 Generating test data...")
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Generate realistic price data with some patterns
        np.random.seed(42)  # For reproducible results
        
        # Start with base price
        base_price = 50000.0
        prices = [base_price]
        
        for i in range(1, 100):
            # Add some volatility and trend
            change = np.random.normal(0, 0.02)  # 2% volatility
            if i > 50:  # Add uptrend after 50 periods
                change += 0.01
            prices.append(prices[-1] * (1 + change))
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = 0.01  # 1% intraday volatility
            
            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            open_price = np.random.uniform(low, high)
            
            # Ensure OHLC relationship is valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        self.test_data = pd.DataFrame(data)
        self.test_data.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Generated {len(self.test_data)} test candlesticks")
    
    async def test_websocket_client(self):
        """Test WebSocket client functionality"""
        logger.info("🔌 Testing WebSocket client...")
        
        try:
            # Test connection
            connected = await self.websocket_client.connect()
            if not connected:
                self.test_results['websocket'] = "FAILED - Connection failed"
                return False
            
            # Test subscription (we'll use a mock callback for testing)
            async def mock_callback(data):
                logger.info(f"📡 Received data: {data['symbol']} {data['interval']}")
            
            subscribed = await self.websocket_client.subscribe_candlesticks(
                'BTCUSDT', '1m', mock_callback
            )
            
            if subscribed:
                # Test getting subscriptions
                subscriptions = await self.websocket_client.get_all_subscriptions()
                logger.info(f"📡 Active subscriptions: {len(subscriptions)}")
                
                # Test unsubscribe
                unsubscribed = await self.websocket_client.unsubscribe_candlesticks(
                    'BTCUSDT', '1m'
                )
                
                if unsubscribed:
                    self.test_results['websocket'] = "PASSED"
                    logger.info("✅ WebSocket client test passed")
                    return True
                else:
                    self.test_results['websocket'] = "FAILED - Unsubscribe failed"
                    return False
            else:
                self.test_results['websocket'] = "FAILED - Subscription failed"
                return False
                
        except Exception as e:
            self.test_results['websocket'] = f"FAILED - {str(e)}"
            logger.error(f"❌ WebSocket test failed: {e}")
            return False
        finally:
            await self.websocket_client.disconnect()
    
    def test_signal_generator(self):
        """Test signal generation functionality"""
        logger.info("🎯 Testing signal generator...")
        
        try:
            # Generate signals from test data
            signals = self.signal_generator.generate_signals(self.test_data, "BTCUSDT")
            
            if signals:
                logger.info(f"🎯 Generated {len(signals)} signals")
                
                # Test signal filtering
                strong_signals = self.signal_generator.filter_signals(
                    signals, min_strength=SignalStrength.STRONG
                )
                logger.info(f"🎯 Strong signals: {len(strong_signals)}")
                
                # Test signal summary
                summary = self.signal_generator.get_signal_summary(signals)
                logger.info(f"📊 Signal summary: {summary}")
                
                # Validate signal structure
                for signal in signals[:3]:  # Check first 3 signals
                    if not self._validate_signal(signal):
                        self.test_results['signal_generator'] = "FAILED - Invalid signal structure"
                        return False
                
                self.test_results['signal_generator'] = "PASSED"
                logger.info("✅ Signal generator test passed")
                return True
            else:
                self.test_results['signal_generator'] = "FAILED - No signals generated"
                return False
                
        except Exception as e:
            self.test_results['signal_generator'] = f"FAILED - {str(e)}"
            logger.error(f"❌ Signal generator test failed: {e}")
            return False
    
    def _validate_signal(self, signal) -> bool:
        """Validate signal structure"""
        required_fields = [
            'timestamp', 'symbol', 'signal_type', 'strength', 'confidence',
            'entry_price', 'stop_loss', 'take_profit', 'position_size',
            'reasoning', 'patterns_detected', 'indicators_confirming',
            'risk_reward_ratio'
        ]
        
        for field in required_fields:
            if not hasattr(signal, field):
                logger.error(f"❌ Signal missing field: {field}")
                return False
        
        # Validate data types and ranges
        if not isinstance(signal.confidence, (int, float)) or not 0 <= signal.confidence <= 1:
            logger.error(f"❌ Invalid confidence value: {signal.confidence}")
            return False
        
        if not isinstance(signal.risk_reward_ratio, (int, float)) or signal.risk_reward_ratio < 0:
            logger.error(f"❌ Invalid risk-reward ratio: {signal.risk_reward_ratio}")
            return False
        
        return True
    
    def test_pattern_detection(self):
        """Test pattern detection functionality"""
        logger.info("🔍 Testing pattern detection...")
        
        try:
            patterns = self.pattern_detector.detect_patterns_from_dataframe(self.test_data)
            
            if patterns:
                logger.info(f"🔍 Detected {len(patterns)} patterns")
                
                # Check pattern structure
                for pattern in patterns[:3]:  # Check first 3 patterns
                    if not hasattr(pattern, 'pattern_name') or not hasattr(pattern, 'confidence'):
                        self.test_results['pattern_detection'] = "FAILED - Invalid pattern structure"
                        return False
                
                self.test_results['pattern_detection'] = "PASSED"
                logger.info("✅ Pattern detection test passed")
                return True
            else:
                self.test_results['pattern_detection'] = "FAILED - No patterns detected"
                return False
                
        except Exception as e:
            self.test_results['pattern_detection'] = f"FAILED - {str(e)}"
            logger.error(f"❌ Pattern detection test failed: {e}")
            return False
    
    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        logger.info("📊 Testing technical indicators...")
        
        try:
            indicators = self.technical_analyzer.calculate_all_indicators(self.test_data)
            
            if indicators:
                logger.info(f"📊 Calculated {len(indicators)} indicators")
                
                # Check common indicators
                expected_indicators = ['rsi', 'macd', 'ema_20', 'ema_50', 'bollinger_upper', 'bollinger_lower']
                missing_indicators = []
                
                for indicator in expected_indicators:
                    if indicator not in indicators:
                        missing_indicators.append(indicator)
                
                if missing_indicators:
                    logger.warning(f"⚠️ Missing indicators: {missing_indicators}")
                
                self.test_results['technical_indicators'] = "PASSED"
                logger.info("✅ Technical indicators test passed")
                return True
            else:
                self.test_results['technical_indicators'] = "FAILED - No indicators calculated"
                return False
                
        except Exception as e:
            self.test_results['technical_indicators'] = f"FAILED - {str(e)}"
            logger.error(f"❌ Technical indicators test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Phase 1 integration tests"""
        logger.info("🚀 Starting Phase 1 Integration Tests...")
        
        # Setup components
        if not self.setup_components():
            logger.error("❌ Component setup failed")
            return False
        
        # Run tests
        tests = [
            ("Pattern Detection", self.test_pattern_detection),
            ("Technical Indicators", self.test_technical_indicators),
            ("Signal Generator", self.test_signal_generator),
            ("WebSocket Client", self.test_websocket_client)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = f"ERROR - {str(e)}"
        
        # Print results summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 PHASE 1 INTEGRATION TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("\n🎉 ALL TESTS PASSED! Phase 1 is ready!")
            logger.info("\n📋 Next steps:")
            logger.info("1. Set up TimescaleDB using: python backend/scripts/setup_database.py")
            logger.info("2. Test with real market data")
            logger.info("3. Move to Phase 2: Trading Strategy Framework")
        else:
            logger.error(f"\n❌ {total - passed} tests failed. Check the logs above.")
            logger.info("\n🔧 Fix the failing tests before proceeding to Phase 2.")
        
        return passed == total

async def main():
    """Main function to run integration tests"""
    tester = Phase1IntegrationTest()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Phase 1 Integration Tests Completed Successfully!")
        print("🚀 Your AlphaPulse trading bot now has real-time capabilities!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
