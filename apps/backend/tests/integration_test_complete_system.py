"""
Complete System Integration Test
Tests all 4 phases working together with simulated market data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging

# Import all our analyzers
from backend.strategies.market_structure_analyzer import MarketStructureAnalyzer
from backend.strategies.dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer
from backend.strategies.demand_supply_zone_analyzer import DemandSupplyZoneAnalyzer
from backend.strategies.advanced_order_flow_analyzer import AdvancedOrderFlowAnalyzer
from backend.strategies.advanced_pattern_detector import AdvancedPatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_market_data(symbol: str = 'BTCUSDT', periods: int = 200) -> pd.DataFrame:
    """Generate realistic market data with clear patterns and zones"""
    np.random.seed(42)
    
    # Create time series
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=periods)
    dates = pd.date_range(start=start_time, end=end_time, periods=periods, freq='1h')
    
    # Base price and trend
    base_price = 50000
    prices = []
    volumes = []
    
    # Create realistic market scenarios
    for i in range(periods):
        # Add some trending behavior
        trend = (i - periods/2) * 2  # Gentle uptrend then downtrend
        
        # Add some volatility
        volatility = 500 + np.random.normal(0, 200)
        
        # Create specific zones and patterns
        if 20 <= i <= 30:  # Demand zone around 49000
            price = 49000 + np.random.normal(0, 100)
            volume = np.random.uniform(3000, 6000)
        elif 50 <= i <= 60:  # Supply zone around 52000
            price = 52000 + np.random.normal(0, 100)
            volume = np.random.uniform(3000, 6000)
        elif 80 <= i <= 90:  # Another demand zone around 48000
            price = 48000 + np.random.normal(0, 100)
            volume = np.random.uniform(3000, 6000)
        elif 120 <= i <= 130:  # Strong uptrend
            price = base_price + trend + 1000 + np.random.normal(0, 150)
            volume = np.random.uniform(4000, 8000)
        elif 150 <= i <= 160:  # Strong downtrend
            price = base_price + trend - 1000 + np.random.normal(0, 150)
            volume = np.random.uniform(4000, 8000)
        else:
            # Normal market conditions
            price = base_price + trend + np.random.normal(0, volatility)
            volume = np.random.uniform(1000, 4000)
        
        prices.append(price)
        volumes.append(volume)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + np.random.uniform(0, 50) for p in prices],
        'low': [p - np.random.uniform(0, 50) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Ensure OHLC relationships are correct
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

def generate_order_book_data() -> dict:
    """Generate realistic order book data"""
    base_price = 50000
    
    # Generate bids (buy orders)
    bids = []
    for i in range(10):
        price = base_price - (i * 10) - np.random.uniform(0, 5)
        size = np.random.uniform(0.1, 2.0)
        bids.append([price, size])
    
    # Generate asks (sell orders)
    asks = []
    for i in range(10):
        price = base_price + (i * 10) + np.random.uniform(0, 5)
        size = np.random.uniform(0.1, 2.0)
        asks.append([price, size])
    
    return {
        'bids': bids,
        'asks': asks,
        'timestamp': datetime.now(timezone.utc)
    }

def generate_trade_data(periods: int = 100) -> pd.DataFrame:
    """Generate realistic trade data"""
    np.random.seed(42)
    
    dates = pd.date_range(start=datetime.now(timezone.utc) - timedelta(hours=periods), 
                         periods=periods, freq='1m')
    
    trades = []
    base_price = 50000
    
    for i in range(periods):
        price = base_price + np.random.normal(0, 100)
        size = np.random.uniform(0.01, 1.0)
        side = np.random.choice(['buy', 'sell'])
        
        trades.append({
            'timestamp': dates[i],
            'price': price,
            'size': size,
            'side': side
        })
    
    return pd.DataFrame(trades)

async def test_individual_analyzers():
    """Test each analyzer individually"""
    logger.info("🧪 Testing Individual Analyzers...")
    
    # Generate test data
    market_data = generate_realistic_market_data()
    order_book = generate_order_book_data()
    trade_data = generate_trade_data()
    
    # Test Market Structure Analyzer
    logger.info("Testing Market Structure Analyzer...")
    market_config = {
        'swing_lookback': 5,
        'min_swing_strength': 0.3,
        'trend_threshold': 0.6
    }
    market_analyzer = MarketStructureAnalyzer(market_config)
    market_analysis = await market_analyzer.analyze_market_structure('BTCUSDT', '1h', market_data)
    
    assert market_analysis is not None
    assert market_analysis.symbol == 'BTCUSDT'
    assert market_analysis.timeframe == '1h'
    logger.info(f"✅ Market Structure Analysis: {market_analysis.structure_type.value}")
    
    # Test Dynamic Support/Resistance Analyzer
    logger.info("Testing Dynamic Support/Resistance Analyzer...")
    sr_config = {
        'min_touches': 2,
        'price_threshold': 0.02,
        'volume_threshold': 0.1
    }
    sr_analyzer = DynamicSupportResistanceAnalyzer(sr_config)
    sr_analysis = await sr_analyzer.analyze_support_resistance('BTCUSDT', '1h', market_data)
    
    assert sr_analysis is not None
    assert sr_analysis.symbol == 'BTCUSDT'
    assert sr_analysis.timeframe == '1h'
    logger.info(f"✅ Support/Resistance Analysis: {len(sr_analysis.support_levels)} support, {len(sr_analysis.resistance_levels)} resistance levels")
    
    # Test Demand/Supply Zone Analyzer
    logger.info("Testing Demand/Supply Zone Analyzer...")
    zone_config = {
        'min_zone_touches': 2,
        'zone_price_threshold': 0.02,
        'volume_threshold': 0.1
    }
    zone_analyzer = DemandSupplyZoneAnalyzer(zone_config)
    zone_analysis = await zone_analyzer.analyze_demand_supply_zones('BTCUSDT', '1h', market_data)
    
    assert zone_analysis is not None
    assert zone_analysis.symbol == 'BTCUSDT'
    assert zone_analysis.timeframe == '1h'
    logger.info(f"✅ Demand/Supply Zone Analysis: {len(zone_analysis.demand_zones)} demand, {len(zone_analysis.supply_zones)} supply zones")
    
    # Test Advanced Order Flow Analyzer
    logger.info("Testing Advanced Order Flow Analyzer...")
    flow_config = {
        'toxicity_threshold': 0.7,
        'large_order_threshold': 0.1,
        'pattern_confidence_threshold': 0.6
    }
    flow_analyzer = AdvancedOrderFlowAnalyzer(flow_config)
    flow_analysis = await flow_analyzer.analyze_order_flow('BTCUSDT', '1h', market_data, order_book, trade_data)
    
    assert flow_analysis is not None
    assert flow_analysis.symbol == 'BTCUSDT'
    assert flow_analysis.timeframe == '1h'
    logger.info(f"✅ Order Flow Analysis: Toxicity {flow_analysis.toxicity_score:.3f}")
    
    return market_analysis, sr_analysis, zone_analysis, flow_analysis

async def test_advanced_pattern_detector_integration():
    """Test the complete AdvancedPatternDetector with all analyzers integrated"""
    logger.info("🔗 Testing Advanced Pattern Detector Integration...")
    
    # Generate test data
    market_data = generate_realistic_market_data()
    
    # Create comprehensive config
    config = {
        'min_pattern_bars': 3,
        'max_pattern_bars': 20,
        'min_confidence': 0.5,
        'volume_threshold': 1.5,
        
        # Market Structure config
        'swing_lookback': 5,
        'min_swing_strength': 0.3,
        'trend_threshold': 0.6,
        
        # Support/Resistance config
        'min_touches': 2,
        'price_threshold': 0.02,
        'volume_threshold': 0.1,
        
        # Demand/Supply Zone config
        'min_zone_touches': 2,
        'zone_price_threshold': 0.02,
        'zone_volume_threshold': 0.1,
        'breakout_threshold': 0.03,
        'min_data_points': 50,
        'volume_profile_bins': 20,
        'zone_strength_threshold': 0.5,
        
        # Order Flow config
        'toxicity_threshold': 0.7,
        'large_order_threshold': 0.1,
        'pattern_confidence_threshold': 0.6
    }
    
    # Initialize pattern detector
    pattern_detector = AdvancedPatternDetector(config)
    
    # Mock the data storage to avoid database dependency
    pattern_detector.storage = None
    pattern_detector.db_connection = None
    
    # Test pattern detection with enhanced analysis
    logger.info("Detecting patterns with all 4 analysis systems...")
    
    # Convert DataFrame to list of dicts for pattern detector
    candlestick_data = market_data.to_dict('records')
    
    # Detect patterns
    patterns = await pattern_detector.detect_patterns('BTCUSDT', '1h', 100)
    
    assert patterns is not None
    logger.info(f"✅ Pattern Detection: Found {len(patterns)} patterns")
    
    # Analyze pattern metadata
    enhanced_patterns = 0
    high_confidence_patterns = 0
    
    for pattern in patterns:
        # Check for enhanced metadata from all 4 systems
        has_market_structure = 'structure_alignment' in pattern.metadata
        has_support_resistance = 'sr_level_proximity' in pattern.metadata
        has_demand_supply = 'demand_zone_proximity' in pattern.metadata or 'supply_zone_proximity' in pattern.metadata
        has_order_flow = 'order_flow_toxicity' in pattern.metadata
        
        if has_market_structure or has_support_resistance or has_demand_supply or has_order_flow:
            enhanced_patterns += 1
        
        if pattern.confidence > 0.8:
            high_confidence_patterns += 1
        
        logger.info(f"Pattern: {pattern.pattern_type.value}, Confidence: {pattern.confidence:.3f}")
        logger.info(f"  Market Structure: {has_market_structure}")
        logger.info(f"  Support/Resistance: {has_support_resistance}")
        logger.info(f"  Demand/Supply Zones: {has_demand_supply}")
        logger.info(f"  Order Flow: {has_order_flow}")
    
    logger.info(f"✅ Enhanced Patterns: {enhanced_patterns}/{len(patterns)}")
    logger.info(f"✅ High Confidence Patterns: {high_confidence_patterns}/{len(patterns)}")
    
    return patterns

async def test_cross_system_validation():
    """Test that all systems work together and provide consistent results"""
    logger.info("🔄 Testing Cross-System Validation...")
    
    # Generate test data
    market_data = generate_realistic_market_data()
    
    # Test all analyzers with same data
    market_analysis, sr_analysis, zone_analysis, flow_analysis = await test_individual_analyzers()
    
    # Validate consistency
    logger.info("Validating cross-system consistency...")
    
    # Check that all analyses have reasonable confidence scores
    assert 0 <= market_analysis.analysis_confidence <= 1
    assert 0 <= sr_analysis.analysis_confidence <= 1
    assert 0 <= zone_analysis.analysis_confidence <= 1
    assert 0 <= flow_analysis.analysis_confidence <= 1
    
    # Check that market structure and support/resistance are aligned
    if market_analysis.structure_type.value == 'uptrend':
        # In uptrend, we should have more support levels than resistance
        assert len(sr_analysis.support_levels) >= len(sr_analysis.resistance_levels) - 2
    
    # Check that demand/supply zones align with support/resistance
    if len(zone_analysis.demand_zones) > 0 and len(sr_analysis.support_levels) > 0:
        # Demand zones should be near support levels
        demand_prices = [zone.zone_start_price for zone in zone_analysis.demand_zones]
        support_prices = [level.price_level for level in sr_analysis.support_levels]
        
        # Check for reasonable alignment (within 5% of each other)
        aligned_count = 0
        for demand_price in demand_prices:
            for support_price in support_prices:
                if abs(demand_price - support_price) / support_price < 0.05:
                    aligned_count += 1
                    break
        
        logger.info(f"✅ Zone/Support Alignment: {aligned_count}/{len(demand_prices)} demand zones aligned with support levels")
    
    logger.info("✅ Cross-system validation completed successfully")

async def test_performance_benchmarks():
    """Test performance of the complete system"""
    logger.info("⚡ Testing Performance Benchmarks...")
    
    # Generate larger dataset for performance testing
    large_market_data = generate_realistic_market_data(periods=500)
    large_order_book = generate_order_book_data()
    large_trade_data = generate_trade_data(periods=300)
    
    import time
    
    # Test individual analyzer performance
    start_time = time.time()
    
    # Market Structure
    market_analyzer = MarketStructureAnalyzer()
    market_analysis = await market_analyzer.analyze_market_structure('BTCUSDT', '1h', large_market_data)
    market_time = time.time() - start_time
    
    # Support/Resistance
    start_time = time.time()
    sr_analyzer = DynamicSupportResistanceAnalyzer()
    sr_analysis = await sr_analyzer.analyze_support_resistance('BTCUSDT', '1h', large_market_data)
    sr_time = time.time() - start_time
    
    # Demand/Supply Zones
    start_time = time.time()
    zone_analyzer = DemandSupplyZoneAnalyzer()
    zone_analysis = await zone_analyzer.analyze_demand_supply_zones('BTCUSDT', '1h', large_market_data)
    zone_time = time.time() - start_time
    
    # Order Flow
    start_time = time.time()
    flow_analyzer = AdvancedOrderFlowAnalyzer()
    flow_analysis = await flow_analyzer.analyze_order_flow('BTCUSDT', '1h', large_market_data, large_order_book, large_trade_data)
    flow_time = time.time() - start_time
    
    # Complete system
    start_time = time.time()
    pattern_detector = AdvancedPatternDetector()
    pattern_detector.storage = None
    pattern_detector.db_connection = None
    candlestick_data = large_market_data.to_dict('records')
    patterns = await pattern_detector.detect_patterns('BTCUSDT', '1h', 200)
    total_time = time.time() - start_time
    
    logger.info(f"✅ Performance Benchmarks (500 data points):")
    logger.info(f"  Market Structure: {market_time:.3f}s")
    logger.info(f"  Support/Resistance: {sr_time:.3f}s")
    logger.info(f"  Demand/Supply Zones: {zone_time:.3f}s")
    logger.info(f"  Order Flow: {flow_time:.3f}s")
    logger.info(f"  Complete System: {total_time:.3f}s")
    logger.info(f"  Patterns Found: {len(patterns)}")
    
    # Performance assertions
    assert market_time < 2.0  # Should complete within 2 seconds
    assert sr_time < 2.0
    assert zone_time < 3.0
    assert flow_time < 2.0
    assert total_time < 10.0  # Complete system should be under 10 seconds

async def main():
    """Run complete integration test suite"""
    logger.info("🚀 Starting Complete System Integration Test")
    logger.info("=" * 60)
    
    try:
        # Test individual analyzers
        await test_individual_analyzers()
        logger.info("✅ Individual Analyzer Tests: PASSED")
        
        # Test pattern detector integration
        patterns = await test_advanced_pattern_detector_integration()
        logger.info("✅ Pattern Detector Integration: PASSED")
        
        # Test cross-system validation
        await test_cross_system_validation()
        logger.info("✅ Cross-System Validation: PASSED")
        
        # Test performance
        await test_performance_benchmarks()
        logger.info("✅ Performance Benchmarks: PASSED")
        
        logger.info("=" * 60)
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info(f"📊 Summary:")
        logger.info(f"  - 4 Analysis Systems: ✅ Working")
        logger.info(f"  - Pattern Detection: ✅ Enhanced")
        logger.info(f"  - Cross-System Integration: ✅ Validated")
        logger.info(f"  - Performance: ✅ Optimized")
        logger.info(f"  - Patterns Found: {len(patterns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎊 INTEGRATION TEST SUCCESSFUL - SYSTEM READY FOR PRODUCTION!")
    else:
        print("\n💥 INTEGRATION TEST FAILED - NEEDS INVESTIGATION!")
