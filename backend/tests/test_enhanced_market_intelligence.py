#!/usr/bin/env python3
"""
Test Enhanced Market Intelligence System
Comprehensive testing for enhanced market intelligence with inflow/outflow analysis
"""

import asyncio
import logging
import sys
import os
import asyncpg
import ccxt
from datetime import datetime, timedelta

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.enhanced_market_intelligence_collector import EnhancedMarketIntelligenceCollector
from database.migrations.004_enhanced_market_intelligence_tables import create_enhanced_market_intelligence_tables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

class EnhancedMarketIntelligenceTester:
    """Test suite for enhanced market intelligence system"""
    
    def __init__(self):
        self.db_pool = None
        self.exchange = None
        self.collector = None
        
    async def setup(self):
        """Setup test environment"""
        try:
            logger.info("🔄 Setting up test environment...")
            
            # Create database connection pool
            self.db_pool = await asyncpg.create_pool(**DB_CONFIG)
            logger.info("✅ Database connection pool created")
            
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': '',
                'secret': '',
                'sandbox': True,
                'enableRateLimit': True
            })
            logger.info("✅ Exchange initialized")
            
            # Initialize collector
            self.collector = EnhancedMarketIntelligenceCollector(self.db_pool, self.exchange)
            await self.collector.initialize()
            logger.info("✅ Enhanced Market Intelligence Collector initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up test environment: {e}")
            raise
    
    async def test_database_migration(self):
        """Test database migration"""
        try:
            logger.info("🔄 Testing database migration...")
            
            # Run migration
            success = await create_enhanced_market_intelligence_tables()
            
            if success:
                logger.info("✅ Database migration completed successfully")
                
                # Verify tables exist
                await self.verify_tables_exist()
            else:
                logger.error("❌ Database migration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing database migration: {e}")
            return False
        
        return True
    
    async def verify_tables_exist(self):
        """Verify that all required tables exist"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if tables exist
                tables_to_check = [
                    'enhanced_market_intelligence',
                    'inflow_outflow_analysis',
                    'whale_movement_tracking',
                    'correlation_analysis',
                    'predictive_market_regime',
                    'market_anomaly_detection'
                ]
                
                for table in tables_to_check:
                    result = await conn.fetchval(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = $1
                        );
                    """, table)
                    
                    if result:
                        logger.info(f"✅ Table {table} exists")
                    else:
                        logger.error(f"❌ Table {table} does not exist")
                        return False
                
                # Check if hypertables exist
                hypertables_to_check = [
                    'enhanced_market_intelligence',
                    'inflow_outflow_analysis',
                    'whale_movement_tracking',
                    'correlation_analysis',
                    'predictive_market_regime',
                    'market_anomaly_detection'
                ]
                
                for table in hypertables_to_check:
                    result = await conn.fetchval(f"""
                        SELECT EXISTS (
                            SELECT FROM timescaledb_information.hypertables 
                            WHERE hypertable_name = $1
                        );
                    """, table)
                    
                    if result:
                        logger.info(f"✅ Hypertable {table} exists")
                    else:
                        logger.warning(f"⚠️ Hypertable {table} does not exist")
                
                logger.info("✅ Table verification completed")
                
        except Exception as e:
            logger.error(f"❌ Error verifying tables: {e}")
            raise
    
    async def test_market_intelligence_collection(self):
        """Test market intelligence data collection"""
        try:
            logger.info("🔄 Testing market intelligence collection...")
            
            # Collect comprehensive market intelligence
            result = await self.collector.collect_comprehensive_market_intelligence()
            
            if result and 'market_intelligence' in result:
                market_intelligence = result['market_intelligence']
                logger.info(f"✅ Market intelligence collected successfully")
                logger.info(f"   - BTC Dominance: {market_intelligence.btc_dominance:.2f}%")
                logger.info(f"   - Total2 Value: ${market_intelligence.total2_value:,.0f}")
                logger.info(f"   - Total3 Value: ${market_intelligence.total3_value:,.0f}")
                logger.info(f"   - Market Regime: {market_intelligence.market_regime}")
                logger.info(f"   - Fear & Greed Index: {market_intelligence.fear_greed_index}")
                logger.info(f"   - Composite Market Strength: {market_intelligence.composite_market_strength:.3f}")
            else:
                logger.error("❌ Market intelligence collection failed")
                return False
            
            if result and 'inflow_outflow' in result:
                inflow_outflow = result['inflow_outflow']
                logger.info(f"✅ Inflow/outflow analysis collected for {len(inflow_outflow)} symbols")
                
                for data in inflow_outflow[:3]:  # Show first 3 symbols
                    logger.info(f"   - {data.symbol}: {data.flow_direction} ({data.flow_strength}) - Confidence: {data.flow_confidence:.3f}")
            else:
                logger.error("❌ Inflow/outflow analysis failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing market intelligence collection: {e}")
            return False
    
    async def test_data_storage(self):
        """Test data storage in database"""
        try:
            logger.info("🔄 Testing data storage...")
            
            async with self.db_pool.acquire() as conn:
                # Check enhanced market intelligence data
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM enhanced_market_intelligence 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)
                logger.info(f"✅ Enhanced market intelligence records in last hour: {count}")
                
                # Check inflow/outflow data
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM inflow_outflow_analysis 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)
                logger.info(f"✅ Inflow/outflow records in last hour: {count}")
                
                # Get latest market intelligence data
                latest_data = await conn.fetchrow("""
                    SELECT btc_dominance, market_regime, composite_market_strength, timestamp
                    FROM enhanced_market_intelligence 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                if latest_data:
                    logger.info(f"✅ Latest market intelligence data:")
                    logger.info(f"   - BTC Dominance: {latest_data['btc_dominance']:.2f}%")
                    logger.info(f"   - Market Regime: {latest_data['market_regime']}")
                    logger.info(f"   - Composite Strength: {latest_data['composite_market_strength']:.3f}")
                    logger.info(f"   - Timestamp: {latest_data['timestamp']}")
                
                # Get latest inflow/outflow data
                latest_flow = await conn.fetchrow("""
                    SELECT symbol, flow_direction, flow_strength, flow_confidence, timestamp
                    FROM inflow_outflow_analysis 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                if latest_flow:
                    logger.info(f"✅ Latest inflow/outflow data:")
                    logger.info(f"   - Symbol: {latest_flow['symbol']}")
                    logger.info(f"   - Flow Direction: {latest_flow['flow_direction']}")
                    logger.info(f"   - Flow Strength: {latest_flow['flow_strength']}")
                    logger.info(f"   - Confidence: {latest_flow['flow_confidence']:.3f}")
                    logger.info(f"   - Timestamp: {latest_flow['timestamp']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing data storage: {e}")
            return False
    
    async def test_continuous_aggregates(self):
        """Test continuous aggregates"""
        try:
            logger.info("🔄 Testing continuous aggregates...")
            
            async with self.db_pool.acquire() as conn:
                # Check if continuous aggregates exist
                aggregates = await conn.fetch("""
                    SELECT view_name FROM timescaledb_information.continuous_aggregates
                    WHERE view_name LIKE '%enhanced_market_intelligence%' 
                       OR view_name LIKE '%inflow_outflow%'
                       OR view_name LIKE '%correlation%'
                """)
                
                if aggregates:
                    logger.info(f"✅ Found {len(aggregates)} continuous aggregates:")
                    for agg in aggregates:
                        logger.info(f"   - {agg['view_name']}")
                else:
                    logger.warning("⚠️ No continuous aggregates found")
                
                # Test querying continuous aggregates
                try:
                    result = await conn.fetchrow("""
                        SELECT * FROM enhanced_market_intelligence_5m 
                        ORDER BY bucket DESC 
                        LIMIT 1
                    """)
                    
                    if result:
                        logger.info("✅ Continuous aggregate query successful")
                        logger.info(f"   - Bucket: {result['bucket']}")
                        logger.info(f"   - Avg BTC Dominance: {result['avg_btc_dominance']:.2f}%")
                        logger.info(f"   - Data Points: {result['data_points']}")
                    else:
                        logger.info("ℹ️ No data in continuous aggregate yet (normal for new setup)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Continuous aggregate query failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing continuous aggregates: {e}")
            return False
    
    async def test_performance_queries(self):
        """Test performance of key queries"""
        try:
            logger.info("🔄 Testing query performance...")
            
            async with self.db_pool.acquire() as conn:
                # Test market intelligence query performance
                start_time = datetime.now()
                result = await conn.fetch("""
                    SELECT btc_dominance, market_regime, composite_market_strength, timestamp
                    FROM enhanced_market_intelligence 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                """)
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Market intelligence query: {len(result)} records in {query_time:.3f}s")
                
                # Test inflow/outflow query performance
                start_time = datetime.now()
                result = await conn.fetch("""
                    SELECT symbol, flow_direction, flow_strength, net_exchange_flow, timestamp
                    FROM inflow_outflow_analysis 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                """)
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Inflow/outflow query: {len(result)} records in {query_time:.3f}s")
                
                # Test anomaly detection query
                start_time = datetime.now()
                result = await conn.fetch("""
                    SELECT symbol, anomaly_type, anomaly_severity, timestamp
                    FROM market_anomaly_detection 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                """)
                query_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Anomaly detection query: {len(result)} records in {query_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing query performance: {e}")
            return False
    
    async def test_data_quality(self):
        """Test data quality and validation"""
        try:
            logger.info("🔄 Testing data quality...")
            
            async with self.db_pool.acquire() as conn:
                # Check for null values in critical fields
                null_check = await conn.fetchval("""
                    SELECT COUNT(*) FROM enhanced_market_intelligence 
                    WHERE btc_dominance IS NULL 
                       OR total2_value IS NULL 
                       OR market_regime IS NULL
                """)
                
                if null_check == 0:
                    logger.info("✅ No null values in critical market intelligence fields")
                else:
                    logger.warning(f"⚠️ Found {null_check} records with null values in critical fields")
                
                # Check data freshness
                latest_timestamp = await conn.fetchval("""
                    SELECT MAX(timestamp) FROM enhanced_market_intelligence
                """)
                
                if latest_timestamp:
                    age_minutes = (datetime.now() - latest_timestamp).total_seconds() / 60
                    logger.info(f"✅ Latest market intelligence data is {age_minutes:.1f} minutes old")
                    
                    if age_minutes < 10:
                        logger.info("✅ Data is fresh (less than 10 minutes old)")
                    else:
                        logger.warning(f"⚠️ Data may be stale ({age_minutes:.1f} minutes old)")
                
                # Check data consistency
                btc_dominance_range = await conn.fetchrow("""
                    SELECT MIN(btc_dominance), MAX(btc_dominance), AVG(btc_dominance)
                    FROM enhanced_market_intelligence 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)
                
                if btc_dominance_range:
                    min_val, max_val, avg_val = btc_dominance_range
                    logger.info(f"✅ BTC Dominance range in last hour: {min_val:.2f}% - {max_val:.2f}% (avg: {avg_val:.2f}%)")
                    
                    if min_val >= 0 and max_val <= 100:
                        logger.info("✅ BTC Dominance values are within valid range")
                    else:
                        logger.warning("⚠️ BTC Dominance values outside valid range")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing data quality: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        try:
            logger.info("🚀 Starting Enhanced Market Intelligence Comprehensive Test Suite")
            
            # Setup
            await self.setup()
            
            # Test database migration
            if not await self.test_database_migration():
                logger.error("❌ Database migration test failed")
                return False
            
            # Test data collection
            if not await self.test_market_intelligence_collection():
                logger.error("❌ Data collection test failed")
                return False
            
            # Test data storage
            if not await self.test_data_storage():
                logger.error("❌ Data storage test failed")
                return False
            
            # Test continuous aggregates
            await self.test_continuous_aggregates()
            
            # Test query performance
            if not await self.test_performance_queries():
                logger.error("❌ Query performance test failed")
                return False
            
            # Test data quality
            if not await self.test_data_quality():
                logger.error("❌ Data quality test failed")
                return False
            
            logger.info("✅ All tests completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return False
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup test environment"""
        try:
            if self.collector:
                await self.collector.close()
                logger.info("✅ Collector closed")
            
            if self.db_pool:
                await self.db_pool.close()
                logger.info("✅ Database connection pool closed")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Main test function"""
    tester = EnhancedMarketIntelligenceTester()
    
    try:
        success = await tester.run_comprehensive_test()
        
        if success:
            logger.info("🎉 Enhanced Market Intelligence System Test Suite PASSED!")
            return 0
        else:
            logger.error("💥 Enhanced Market Intelligence System Test Suite FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Test suite crashed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
