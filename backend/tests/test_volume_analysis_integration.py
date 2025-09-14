#!/usr/bin/env python3
"""
Test script for Volume Analysis Integration
Validates the enhanced volume analysis system with TimescaleDB integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncpg
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

def generate_test_ohlcv_data(symbol: str, timeframe: str, periods: int = 50) -> List[Dict]:
    """Generate test OHLCV data for testing"""
    base_price = 50000.0
    base_volume = 1000.0
    
    data = []
    current_time = datetime.now(timezone.utc) - timedelta(hours=periods)
    
    for i in range(periods):
        # Generate realistic price movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        volume_change = np.random.normal(1, 0.3)  # Volume variation
        
        open_price = base_price * (1 + price_change)
        close_price = open_price * (1 + np.random.normal(0, 0.01))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = base_volume * volume_change
        
        # Add some volume spikes for testing
        if i % 10 == 0:  # Every 10th candle
            volume *= np.random.uniform(2.0, 5.0)  # Volume spike
        
        data.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': current_time + timedelta(minutes=i),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
        base_volume = volume
    
    return data

async def test_database_connection():
    """Test database connection and table existence"""
    try:
        logger.info("🔍 Testing database connection...")
        
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Test if volume analysis tables exist
        tables = ['volume_analysis', 'volume_patterns', 'comprehensive_analysis']
        
        for table in tables:
            exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table}'
                );
            """)
            
            if exists:
                logger.info(f"✅ Table {table} exists")
            else:
                logger.warning(f"⚠️ Table {table} does not exist")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

async def test_volume_analysis_service():
    """Test the enhanced volume analyzer service"""
    try:
        logger.info("🔍 Testing Enhanced Volume Analyzer Service...")
        
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Import the service
        from app.services.enhanced_volume_analyzer_service import EnhancedVolumeAnalyzerService
        
        # Initialize service with connection pool mock
        class MockPool:
            async def acquire(self):
                return conn
            async def release(self, conn):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        service = EnhancedVolumeAnalyzerService(MockPool())
        
        # Generate test data
        test_data = generate_test_ohlcv_data('BTCUSDT', '1m', 50)
        
        # Test volume analysis
        result = await service.analyze_volume('BTCUSDT', '1m', test_data)
        
        logger.info(f"✅ Volume analysis completed:")
        logger.info(f"   - Symbol: {result.symbol}")
        logger.info(f"   - Timeframe: {result.timeframe}")
        logger.info(f"   - Volume Ratio: {result.volume_ratio:.3f}")
        logger.info(f"   - Volume Trend: {result.volume_trend}")
        logger.info(f"   - Volume Positioning Score: {result.volume_positioning_score:.3f}")
        logger.info(f"   - Order Book Imbalance: {result.order_book_imbalance:.3f}")
        logger.info(f"   - Volume Breakout: {result.volume_breakout}")
        logger.info(f"   - Volume Pattern Type: {result.volume_pattern_type}")
        logger.info(f"   - Analysis: {result.volume_analysis}")
        logger.info(f"   - Advanced Metrics:")
        logger.info(f"     * VWAP: {result.vwap:.2f}" if result.vwap else "     * VWAP: N/A")
        logger.info(f"     * CVD: {result.cumulative_volume_delta:.0f}" if result.cumulative_volume_delta else "     * CVD: N/A")
        logger.info(f"     * RVOL: {result.relative_volume:.3f}" if result.relative_volume else "     * RVOL: N/A")
        logger.info(f"     * VWP: {result.volume_weighted_price:.2f}" if result.volume_weighted_price else "     * VWP: N/A")
        logger.info(f"     * Flow Imbalance: {result.volume_flow_imbalance:.4f}" if result.volume_flow_imbalance else "     * Flow Imbalance: N/A")
        logger.info(f"   - Delta Profile Metrics:")
        logger.info(f"     * Support Levels: {len(result.support_levels)} found" if result.support_levels else "     * Support Levels: None")
        logger.info(f"     * Resistance Levels: {len(result.resistance_levels)} found" if result.resistance_levels else "     * Resistance Levels: None")
        logger.info(f"     * Volume Nodes: {len(result.volume_nodes)} detected" if result.volume_nodes else "     * Volume Nodes: None")
        logger.info(f"   - Liquidity Metrics:")
        logger.info(f"     * Liquidity Score: {result.liquidity_score:.3f}" if result.liquidity_score else "     * Liquidity Score: N/A")
        logger.info(f"     * Spoofing Detected: {result.spoofing_detected}" if result.spoofing_detected is not None else "     * Spoofing Detected: N/A")
        logger.info(f"     * Whale Activity: {result.whale_activity}" if result.whale_activity is not None else "     * Whale Activity: N/A")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Volume analyzer service test failed: {e}")
        return False

async def test_volume_pattern_integration():
    """Test the volume pattern integration service"""
    try:
        logger.info("🔍 Testing Volume Pattern Integration Service...")
        
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Import the service
        from app.services.volume_pattern_integration_service import VolumePatternIntegrationService
        
        # Initialize service with connection pool mock
        class MockPool:
            async def acquire(self):
                return conn
            async def release(self, conn):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        service = VolumePatternIntegrationService(MockPool())
        
        # Create test pattern data
        test_pattern = {
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'timestamp': datetime.now(timezone.utc),
            'pattern_type': 'bullish_engulfing',
            'pattern_strength': 'medium',
            'confidence': 0.75,
            'direction': 'bullish'
        }
        
        # Generate test OHLCV data
        test_data = generate_test_ohlcv_data('BTCUSDT', '1m', 50)
        
        # Test pattern enhancement with volume
        enhanced_pattern = await service.analyze_pattern_with_volume(test_pattern, test_data)
        
        logger.info(f"✅ Volume pattern integration completed:")
        logger.info(f"   - Original Confidence: {test_pattern['confidence']:.3f}")
        logger.info(f"   - Volume Enhanced Confidence: {enhanced_pattern.get('volume_enhanced_confidence', 0):.3f}")
        logger.info(f"   - Volume Confirmation: {enhanced_pattern.get('volume_confirmation', False)}")
        logger.info(f"   - Volume Ratio: {enhanced_pattern.get('volume_metrics', {}).get('volume_ratio', 0):.3f}")
        logger.info(f"   - Pattern Strength: {enhanced_pattern.get('pattern_strength', 'unknown')}")
        logger.info(f"   - Volume Analysis: {enhanced_pattern.get('volume_analysis', 'N/A')}")
        logger.info(f"   - Advanced Metrics:")
        advanced_metrics = enhanced_pattern.get('advanced_volume_metrics', {})
        logger.info(f"     * VWAP: {advanced_metrics.get('vwap', 0):.2f}" if advanced_metrics.get('vwap') else "     * VWAP: N/A")
        logger.info(f"     * CVD: {advanced_metrics.get('cumulative_volume_delta', 0):.0f}" if advanced_metrics.get('cumulative_volume_delta') else "     * CVD: N/A")
        logger.info(f"     * RVOL: {advanced_metrics.get('relative_volume', 0):.3f}" if advanced_metrics.get('relative_volume') else "     * RVOL: N/A")
        logger.info(f"     * VWP: {advanced_metrics.get('volume_weighted_price', 0):.2f}" if advanced_metrics.get('volume_weighted_price') else "     * VWP: N/A")
        logger.info(f"     * Flow Imbalance: {advanced_metrics.get('volume_flow_imbalance', 0):.4f}" if advanced_metrics.get('volume_flow_imbalance') else "     * Flow Imbalance: N/A")
        logger.info(f"   - Delta Profile Metrics:")
        delta_metrics = enhanced_pattern.get('delta_profile_metrics', {})
        logger.info(f"     * Support Levels: {len(delta_metrics.get('support_levels', []))} found")
        logger.info(f"     * Resistance Levels: {len(delta_metrics.get('resistance_levels', []))} found")
        logger.info(f"     * Volume Nodes: {len(delta_metrics.get('volume_nodes', []))} detected")
        logger.info(f"   - Liquidity Metrics:")
        liquidity_metrics = enhanced_pattern.get('liquidity_metrics', {})
        logger.info(f"     * Liquidity Score: {liquidity_metrics.get('liquidity_score', 0):.3f}")
        logger.info(f"     * Spoofing Detected: {liquidity_metrics.get('spoofing_detected', False)}")
        logger.info(f"     * Whale Activity: {liquidity_metrics.get('whale_activity', False)}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Volume pattern integration test failed: {e}")
        return False

async def test_database_queries():
    """Test database queries for volume analysis"""
    try:
        logger.info("🔍 Testing database queries...")
        
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Test volume analysis query
        volume_count = await conn.fetchval("""
            SELECT COUNT(*) FROM volume_analysis 
            WHERE symbol = 'BTCUSDT' AND timeframe = '1m'
        """)
        
        logger.info(f"✅ Volume analysis records: {volume_count}")
        
        # Test comprehensive analysis query
        comprehensive_count = await conn.fetchval("""
            SELECT COUNT(*) FROM comprehensive_analysis 
            WHERE symbol = 'BTCUSDT' AND timeframe = '1m'
        """)
        
        logger.info(f"✅ Comprehensive analysis records: {comprehensive_count}")
        
        # Test volume statistics
        stats = await conn.fetchrow("""
            SELECT 
                AVG(volume_ratio) as avg_volume_ratio,
                AVG(volume_positioning_score) as avg_positioning,
                COUNT(*) as total_records
            FROM volume_analysis 
            WHERE symbol = 'BTCUSDT' AND timeframe = '1m'
        """)
        
        if stats:
            logger.info(f"✅ Volume statistics:")
            avg_ratio = stats['avg_volume_ratio'] if stats['avg_volume_ratio'] is not None else 0.0
            avg_positioning = stats['avg_positioning'] if stats['avg_positioning'] is not None else 0.0
            logger.info(f"   - Average Volume Ratio: {avg_ratio:.3f}")
            logger.info(f"   - Average Positioning Score: {avg_positioning:.3f}")
            logger.info(f"   - Total Records: {stats['total_records']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database queries test failed: {e}")
        return False

async def run_migration_test():
    """Test the volume analysis migration"""
    try:
        logger.info("🔍 Testing volume analysis migration...")
        
        # Run the migration
        import importlib.util
        spec = importlib.util.spec_from_file_location("migration", "database/migrations/003_volume_analysis_tables.py")
        migration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration_module)
        create_volume_analysis_tables = migration_module.create_volume_analysis_tables
        
        success = await create_volume_analysis_tables()
        
        if success:
            logger.info("✅ Volume analysis migration test completed successfully")
            return True
        else:
            logger.error("❌ Volume analysis migration test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Migration test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Volume Analysis Integration Tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Volume Analysis Migration", run_migration_test),
        ("Enhanced Volume Analyzer Service", test_volume_analysis_service),
        ("Volume Pattern Integration", test_volume_pattern_integration),
        ("Database Queries", test_database_queries)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Volume analysis integration is working correctly.")
    else:
        logger.error("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
