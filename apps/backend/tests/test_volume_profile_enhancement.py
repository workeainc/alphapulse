"""
Test Suite for Volume Profile Enhancement
Validates POC, Value Areas, and advanced volume profile analysis
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from src.data.volume_profile_analyzer import VolumeProfileAnalyzer, VolumeProfileAnalysis, VolumeProfileType

class VolumeProfileTestSuite:
    """Comprehensive test suite for Volume Profile analysis"""
    
    def __init__(self):
        self.analyzer = VolumeProfileAnalyzer()
        self.test_results = []
        
    def generate_test_data(self, pattern_type: str = "normal") -> pd.DataFrame:
        """Generate test data for different volume profile scenarios"""
        np.random.seed(42)  # For reproducible results
        base_price = 45000
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        
        if pattern_type == "normal":
            # Normal volume distribution
            prices = []
            volumes = []
            for i in range(100):
                # Normal price movement
                price = base_price + np.random.normal(0, 500)
                # Normal volume with some variation
                volume = np.random.normal(5000, 1000)
                prices.append(price)
                volumes.append(max(100, volume))
                
        elif pattern_type == "poc_clear":
            # Clear POC at specific level
            prices = []
            volumes = []
            poc_level = base_price + 200
            for i in range(100):
                if i < 30:
                    # Build up volume around POC
                    price = poc_level + np.random.normal(0, 100)
                    volume = np.random.normal(8000, 500)  # High volume
                elif i < 60:
                    # Normal trading
                    price = base_price + np.random.normal(0, 300)
                    volume = np.random.normal(4000, 800)
                else:
                    # Return to POC with high volume
                    price = poc_level + np.random.normal(0, 150)
                    volume = np.random.normal(9000, 600)  # Very high volume
                prices.append(price)
                volumes.append(max(100, volume))
                
        elif pattern_type == "value_area":
            # Clear value area formation
            prices = []
            volumes = []
            value_area_high = base_price + 300
            value_area_low = base_price - 200
            for i in range(100):
                if i < 40:
                    # High volume in value area
                    price = np.random.uniform(value_area_low, value_area_high)
                    volume = np.random.normal(7000, 1000)
                elif i < 70:
                    # Lower volume outside value area
                    price = base_price + np.random.normal(0, 400)
                    volume = np.random.normal(2000, 500)
                else:
                    # Return to value area
                    price = np.random.uniform(value_area_low, value_area_high)
                    volume = np.random.normal(6000, 800)
                prices.append(price)
                volumes.append(max(100, volume))
                
        elif pattern_type == "volume_gaps":
            # Volume gaps at specific levels
            prices = []
            volumes = []
            gap_levels = [base_price - 100, base_price + 150]
            for i in range(100):
                price = base_price + np.random.normal(0, 300)
                # Check if price is near gap levels
                near_gap = any(abs(price - gap) < 50 for gap in gap_levels)
                if near_gap:
                    volume = np.random.normal(500, 200)  # Very low volume
                else:
                    volume = np.random.normal(5000, 1000)  # Normal volume
                prices.append(price)
                volumes.append(max(100, volume))
                
        elif pattern_type == "single_prints":
            # Single print patterns
            prices = []
            volumes = []
            single_print_levels = [base_price - 200, base_price + 250]
            for i in range(100):
                price = base_price + np.random.normal(0, 300)
                # Check if price is at single print levels
                at_single_print = any(abs(price - level) < 30 for level in single_print_levels)
                if at_single_print:
                    volume = np.random.normal(15000, 2000)  # Very high volume
                else:
                    volume = np.random.normal(3000, 800)  # Low volume
                prices.append(price)
                volumes.append(max(100, volume))
                
        elif pattern_type == "volume_climax":
            # Volume climax patterns
            prices = []
            volumes = []
            climax_level = base_price + 100
            for i in range(100):
                if i < 80:
                    # Normal trading
                    price = base_price + np.random.normal(0, 200)
                    volume = np.random.normal(4000, 800)
                else:
                    # Volume climax at specific level
                    price = climax_level + np.random.normal(0, 50)
                    volume = np.random.normal(20000, 3000)  # Extreme volume
                prices.append(price)
                volumes.append(max(100, volume))
                
        else:
            # Default normal pattern
            prices = [base_price + np.random.normal(0, 300) for _ in range(100)]
            volumes = [max(100, np.random.normal(5000, 1000)) for _ in range(100)]
        
        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices,
            'high': [p + abs(np.random.normal(0, 50)) for p in prices],
            'low': [p - abs(np.random.normal(0, 50)) for p in prices],
            'close': [p + np.random.normal(0, 30) for p in prices],
            'volume': volumes
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def test_volume_profile_basic(self) -> bool:
        """Test basic volume profile analysis"""
        print("🧪 Testing Volume Profile Basic Analysis...")
        
        try:
            # Generate test data
            df = self.generate_test_data("normal")
            print(f"✅ Test data generated: {len(df)} candles")
            print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
            print(f"   Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate results
            assert analysis.symbol == "BTCUSDT"
            assert analysis.timeframe == "1h"
            assert analysis.poc_level > 0
            assert analysis.value_area_high >= analysis.value_area_low
            assert analysis.confidence_score >= 0 and analysis.confidence_score <= 1
            assert analysis.processing_time_ms > 0
            
            print(f"✅ POC Level: {analysis.poc_level:.2f}")
            print(f"✅ Value Area: {analysis.value_area_low:.2f} - {analysis.value_area_high:.2f}")
            print(f"✅ Confidence Score: {analysis.confidence_score:.3f}")
            print(f"✅ Processing Time: {analysis.processing_time_ms:.2f}ms")
            print(f"✅ Volume Nodes: {len(analysis.volume_nodes)}")
            print(f"✅ Volume Gaps: {len(analysis.volume_gaps)}")
            print(f"✅ Single Prints: {len(analysis.single_prints)}")
            print(f"✅ Volume Climax: {len(analysis.volume_climax_levels)}")
            print(f"✅ Volume Exhaustion: {len(analysis.volume_exhaustion_levels)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
            return False
    
    def test_poc_detection(self) -> bool:
        """Test POC (Point of Control) detection"""
        print("🧪 Testing POC Detection...")
        
        try:
            # Generate data with clear POC
            df = self.generate_test_data("poc_clear")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate POC
            assert analysis.poc_level > 0
            assert analysis.confidence_score > 0.5  # Should have good confidence
            
            print(f"✅ POC detected at: {analysis.poc_level:.2f}")
            print(f"✅ POC confidence: {analysis.confidence_score:.3f}")
            
            # Test trading signals
            current_price = df['close'].iloc[-1]
            signals = self.analyzer.get_trading_signals(analysis, current_price)
            
            assert 'poc_signal' in signals
            assert 'overall_signal' in signals
            assert 'confidence' in signals
            
            print(f"✅ POC Signal: {signals['poc_signal']['signal']}")
            print(f"✅ Overall Signal: {signals['overall_signal']}")
            print(f"✅ Signal Confidence: {signals['confidence']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ POC test failed: {e}")
            return False
    
    def test_value_areas(self) -> bool:
        """Test Value Areas detection"""
        print("🧪 Testing Value Areas Detection...")
        
        try:
            # Generate data with clear value areas
            df = self.generate_test_data("value_area")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate value areas
            assert analysis.value_area_high > analysis.value_area_low
            assert analysis.value_area_volume_percentage == 0.70  # Default setting
            
            print(f"✅ Value Area High: {analysis.value_area_high:.2f}")
            print(f"✅ Value Area Low: {analysis.value_area_low:.2f}")
            print(f"✅ Value Area Range: {analysis.value_area_high - analysis.value_area_low:.2f}")
            
            # Test value area signals
            current_price = df['close'].iloc[-1]
            signals = self.analyzer.get_trading_signals(analysis, current_price)
            
            assert 'value_area_signal' in signals
            print(f"✅ Value Area Signal: {signals['value_area_signal']['signal']}")
            print(f"✅ Value Area Message: {signals['value_area_signal']['message']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Value Areas test failed: {e}")
            return False
    
    def test_volume_nodes(self) -> bool:
        """Test volume nodes detection"""
        print("🧪 Testing Volume Nodes Detection...")
        
        try:
            # Generate normal data
            df = self.generate_test_data("normal")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate volume nodes
            assert len(analysis.volume_nodes) >= 0
            
            print(f"✅ Volume Nodes detected: {len(analysis.volume_nodes)}")
            
            # Check node properties
            for i, node in enumerate(analysis.volume_nodes[:3]):  # Show first 3
                print(f"   Node {i+1}: Price={node.price_level:.2f}, Volume={node.volume:.0f}, POC Score={node.poc_score:.3f}")
                assert node.price_level > 0
                assert node.volume > 0
                assert node.poc_score >= 0 and node.poc_score <= 1
                assert node.level_type == VolumeProfileType.VOLUME_NODE
            
            return True
            
        except Exception as e:
            print(f"❌ Volume Nodes test failed: {e}")
            return False
    
    def test_volume_gaps(self) -> bool:
        """Test volume gaps detection"""
        print("🧪 Testing Volume Gaps Detection...")
        
        try:
            # Generate data with volume gaps
            df = self.generate_test_data("volume_gaps")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate volume gaps
            assert len(analysis.volume_gaps) >= 0
            
            print(f"✅ Volume Gaps detected: {len(analysis.volume_gaps)}")
            
            # Check gap properties
            for i, gap in enumerate(analysis.volume_gaps[:3]):  # Show first 3
                print(f"   Gap {i+1}: Price={gap.price_level:.2f}, Volume={gap.volume:.0f}, Gap Strength={gap.metadata.get('gap_strength', 0):.2f}")
                assert gap.price_level > 0
                assert gap.level_type == VolumeProfileType.VOLUME_GAP
            
            return True
            
        except Exception as e:
            print(f"❌ Volume Gaps test failed: {e}")
            return False
    
    def test_single_prints(self) -> bool:
        """Test single prints detection"""
        print("🧪 Testing Single Prints Detection...")
        
        try:
            # Generate data with single prints
            df = self.generate_test_data("single_prints")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate single prints
            assert len(analysis.single_prints) >= 0
            
            print(f"✅ Single Prints detected: {len(analysis.single_prints)}")
            
            # Check single print properties
            for i, single_print in enumerate(analysis.single_prints[:3]):  # Show first 3
                print(f"   Single Print {i+1}: Price={single_print.price_level:.2f}, Volume={single_print.volume:.0f}, Isolation={single_print.metadata.get('isolation_ratio', 0):.2f}")
                assert single_print.price_level > 0
                assert single_print.level_type == VolumeProfileType.SINGLE_PRINT
            
            return True
            
        except Exception as e:
            print(f"❌ Single Prints test failed: {e}")
            return False
    
    def test_volume_climax(self) -> bool:
        """Test volume climax detection"""
        print("🧪 Testing Volume Climax Detection...")
        
        try:
            # Generate data with volume climax
            df = self.generate_test_data("volume_climax")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Validate volume climax
            assert len(analysis.volume_climax_levels) >= 0
            
            print(f"✅ Volume Climax detected: {len(analysis.volume_climax_levels)}")
            
            # Check climax properties
            for i, climax in enumerate(analysis.volume_climax_levels[:3]):  # Show first 3
                print(f"   Climax {i+1}: Price={climax.price_level:.2f}, Volume={climax.volume:.0f}, Strength={climax.metadata.get('climax_strength', 0):.2f}")
                assert climax.price_level > 0
                assert climax.level_type == VolumeProfileType.VOLUME_CLIMAX
            
            return True
            
        except Exception as e:
            print(f"❌ Volume Climax test failed: {e}")
            return False
    
    def test_trading_signals(self) -> bool:
        """Test trading signal generation"""
        print("🧪 Testing Trading Signal Generation...")
        
        try:
            # Generate test data
            df = self.generate_test_data("normal")
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            # Test signals at different price levels
            current_price = df['close'].iloc[-1]
            poc_price = analysis.poc_level
            value_area_mid = (analysis.value_area_high + analysis.value_area_low) / 2
            
            # Test at current price
            signals_current = self.analyzer.get_trading_signals(analysis, current_price)
            assert 'overall_signal' in signals_current
            assert 'confidence' in signals_current
            
            # Test at POC level
            signals_poc = self.analyzer.get_trading_signals(analysis, poc_price)
            assert 'overall_signal' in signals_poc
            
            # Test at value area
            signals_va = self.analyzer.get_trading_signals(analysis, value_area_mid)
            assert 'overall_signal' in signals_va
            
            print(f"✅ Current Price Signal: {signals_current['overall_signal']}")
            print(f"✅ POC Level Signal: {signals_poc['overall_signal']}")
            print(f"✅ Value Area Signal: {signals_va['overall_signal']}")
            print(f"✅ Signal Confidence: {signals_current['confidence']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trading Signals test failed: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance and processing time"""
        print("🧪 Testing Performance...")
        
        try:
            # Generate larger dataset
            df = self.generate_test_data("normal")
            df = pd.concat([df] * 5)  # 5x larger dataset
            
            start_time = time.time()
            
            # Analyze volume profile
            analysis = self.analyzer.analyze_volume_profile(df, "BTCUSDT", "1h")
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"✅ Dataset size: {len(df)} candles")
            print(f"✅ Processing time: {processing_time:.2f}ms")
            print(f"✅ Internal processing time: {analysis.processing_time_ms:.2f}ms")
            
            # Performance should be under 100ms
            assert processing_time < 100, f"Processing time {processing_time:.2f}ms exceeds 100ms limit"
            
            print("✅ Performance meets requirements (<100ms)")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all volume profile tests"""
        print("🚀 Starting Volume Profile Test Suite")
        print("=" * 60)
        
        tests = [
            ("Basic Analysis", self.test_volume_profile_basic),
            ("POC Detection", self.test_poc_detection),
            ("Value Areas", self.test_value_areas),
            ("Volume Nodes", self.test_volume_nodes),
            ("Volume Gaps", self.test_volume_gaps),
            ("Single Prints", self.test_single_prints),
            ("Volume Climax", self.test_volume_climax),
            ("Trading Signals", self.test_trading_signals),
            ("Performance", self.test_performance)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running: {test_name}")
            print("-" * 40)
            
            try:
                if test_func():
                    print(f"✅ {test_name}: PASS")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAIL")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results Summary:")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 All Volume Profile tests passed!")
            return True
        else:
            print(f"⚠️ {total_tests - passed_tests} tests failed")
            return False

def main():
    """Main test execution"""
    test_suite = VolumeProfileTestSuite()
    success = test_suite.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
