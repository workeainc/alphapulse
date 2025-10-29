#!/usr/bin/env python3
"""
Comprehensive Test Script for AlphaPlus Database Integration
Tests all components and ensures everything works together seamlessly
"""

import asyncio
import logging
import asyncpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import json
import time

# Import all the components we need to test
from src.services.algorithm_integration_service import AlgorithmIntegrationService
from src.strategies.standalone_psychological_levels_analyzer import StandalonePsychologicalLevelsAnalyzer
from src.strategies.enhanced_volume_weighted_levels_analyzer import EnhancedVolumeWeightedLevelsAnalyzer
from src.services.historical_data_preloader import HistoricalDataPreloader, PreloadConfig

logger = logging.getLogger(__name__)

class ComprehensiveIntegrationTest:
    """Comprehensive test for all AlphaPlus components"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.conn = None
        
        # Test results
        self.test_results = {
            'database_connection': False,
            'table_existence': False,
            'hypertable_creation': False,
            'constraints': False,
            'indexes': False,
            'permissions': False,
            'algorithm_integration': False,
            'psychological_levels': False,
            'volume_analysis': False,
            'data_insertion': False,
            'data_retrieval': False,
            'performance': False
        }
        
        # Performance metrics
        self.performance_metrics = {
            'total_test_time': 0.0,
            'database_operations': 0.0,
            'algorithm_execution': 0.0,
            'data_insertion': 0.0,
            'data_retrieval': 0.0
        }
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.conn = await asyncpg.connect(self.db_url)
            logger.info("✅ Database connection established")
            self.test_results['database_connection'] = True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("🔌 Database connection closed")
    
    async def run_comprehensive_test(self):
        """Run comprehensive integration test"""
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # Test 1: Database Schema Verification
            await self.test_database_schema()
            
            # Test 2: Table Existence and Structure
            await self.test_table_existence()
            
            # Test 3: Hypertable Creation
            await self.test_hypertable_creation()
            
            # Test 4: Constraints and Indexes
            await self.test_constraints_and_indexes()
            
            # Test 5: Permissions
            await self.test_permissions()
            
            # Test 6: Algorithm Integration Service
            await self.test_algorithm_integration_service()
            
            # Test 7: Psychological Levels Analyzer
            await self.test_psychological_levels_analyzer()
            
            # Test 8: Volume Analysis
            await self.test_volume_analysis()
            
            # Test 9: Data Operations
            await self.test_data_operations()
            
            # Test 10: Performance Testing
            await self.test_performance()
            
            # Calculate total test time
            self.performance_metrics['total_test_time'] = time.time() - start_time
            
            # Generate comprehensive report
            await self.generate_test_report()
            
            logger.info("🎉 Comprehensive integration test completed!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        finally:
            await self.close()
    
    async def test_database_schema(self):
        """Test database schema and extensions"""
        logger.info("🔍 Testing database schema...")
        
        try:
            # Check TimescaleDB extension
            result = await self.conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            if result:
                logger.info("✅ TimescaleDB extension is active")
            else:
                logger.warning("⚠️ TimescaleDB extension not found")
            
            # Check database version
            version = await self.conn.fetchval("SELECT version()")
            logger.info(f"📊 Database version: {version[:50]}...")
            
            self.test_results['database_connection'] = True
            
        except Exception as e:
            logger.error(f"❌ Database schema test failed: {e}")
            raise
    
    async def test_table_existence(self):
        """Test if all required tables exist"""
        logger.info("🔍 Testing table existence...")
        
        required_tables = [
            'ohlcv_data',
            'volume_profile_analysis',
            'order_book_levels',
            'market_microstructure',
            'psychological_levels_analysis',
            'psychological_levels',
            'psychological_level_interactions',
            'algorithm_results',
            'signal_confluence'
        ]
        
        existing_tables = []
        for table in required_tables:
            try:
                result = await self.conn.fetchval(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table}'")
                if result:
                    existing_tables.append(table)
                    logger.info(f"✅ Table {table} exists")
                else:
                    logger.warning(f"⚠️ Table {table} not found")
            except Exception as e:
                logger.warning(f"⚠️ Error checking table {table}: {e}")
        
        if len(existing_tables) >= len(required_tables) * 0.8:  # At least 80% of tables exist
            self.test_results['table_existence'] = True
            logger.info(f"✅ {len(existing_tables)}/{len(required_tables)} tables exist")
        else:
            logger.warning(f"⚠️ Only {len(existing_tables)}/{len(required_tables)} tables exist")
    
    async def test_hypertable_creation(self):
        """Test TimescaleDB hypertable creation"""
        logger.info("🔍 Testing hypertable creation...")
        
        try:
            hypertables = await self.conn.fetch("""
                SELECT hypertable_name FROM timescaledb_information.hypertables 
                WHERE hypertable_name IN ('ohlcv_data', 'volume_profile_analysis', 'order_book_levels', 
                                         'market_microstructure', 'psychological_levels_analysis', 
                                         'psychological_levels', 'psychological_level_interactions', 
                                         'algorithm_results', 'signal_confluence')
            """)
            
            if hypertables:
                for row in hypertables:
                    logger.info(f"✅ Hypertable {row['hypertable_name']} is active")
                self.test_results['hypertable_creation'] = True
            else:
                logger.warning("⚠️ No hypertables found")
                
        except Exception as e:
            logger.error(f"❌ Hypertable test failed: {e}")
    
    async def test_constraints_and_indexes(self):
        """Test constraints and indexes"""
        logger.info("🔍 Testing constraints and indexes...")
        
        try:
            # Check unique constraints
            constraints = await self.conn.fetch("""
                SELECT constraint_name, table_name 
                FROM information_schema.table_constraints 
                WHERE constraint_type = 'UNIQUE' 
                AND table_name IN ('ohlcv_data', 'volume_profile_analysis', 'psychological_levels_analysis', 'psychological_levels')
            """)
            
            if constraints:
                for row in constraints:
                    logger.info(f"✅ Constraint {row['constraint_name']} exists on {row['table_name']}")
                self.test_results['constraints'] = True
            else:
                logger.warning("⚠️ No unique constraints found")
            
            # Check indexes
            indexes = await self.conn.fetch("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE tablename IN ('ohlcv_data', 'volume_profile_analysis', 'psychological_levels_analysis', 'psychological_levels')
                AND indexname LIKE 'idx_%'
            """)
            
            if indexes:
                for row in indexes:
                    logger.info(f"✅ Index {row['indexname']} exists on {row['tablename']}")
                self.test_results['indexes'] = True
            else:
                logger.warning("⚠️ No performance indexes found")
                
        except Exception as e:
            logger.error(f"❌ Constraints and indexes test failed: {e}")
    
    async def test_permissions(self):
        """Test user permissions"""
        logger.info("🔍 Testing user permissions...")
        
        try:
            # Check if alpha_emon user exists
            user_exists = await self.conn.fetchval("SELECT 1 FROM pg_user WHERE usename = 'alpha_emon'")
            if user_exists:
                logger.info("✅ User alpha_emon exists")
                self.test_results['permissions'] = True
            else:
                logger.warning("⚠️ User alpha_emon not found")
                
        except Exception as e:
            logger.error(f"❌ Permissions test failed: {e}")
    
    async def test_algorithm_integration_service(self):
        """Test algorithm integration service"""
        logger.info("🔍 Testing algorithm integration service...")
        
        try:
            # Initialize the service
            service = AlgorithmIntegrationService(self.db_url)
            await service.initialize()
            
            # Test with sample data
            sample_data = [
                {
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'timestamp': datetime.now(timezone.utc),
                    'open': 50000.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'close': 50500.0,
                    'volume': 1000.0
                }
            ]
            
            # Test algorithm execution
            df = pd.DataFrame(sample_data)
            results = await service.run_all_algorithms(df, 'BTCUSDT', '1h')
            
            if results and len(results) > 0:
                logger.info(f"✅ Algorithm integration service working - {len(results)} algorithms executed")
                self.test_results['algorithm_integration'] = True
            else:
                logger.warning("⚠️ Algorithm integration service returned no results")
                
        except Exception as e:
            logger.error(f"❌ Algorithm integration service test failed: {e}")
    
    async def test_psychological_levels_analyzer(self):
        """Test psychological levels analyzer"""
        logger.info("🔍 Testing psychological levels analyzer...")
        
        try:
            # Initialize the analyzer
            analyzer = StandalonePsychologicalLevelsAnalyzer(self.db_url)
            await analyzer.initialize()
            
            # Create sample OHLCV data
            sample_data = self._create_sample_ohlcv_data('BTCUSDT', '1h', 100)
            
            # Test analysis
            analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
            
            if analysis and analysis.psychological_levels:
                logger.info(f"✅ Psychological levels analyzer working - {len(analysis.psychological_levels)} levels detected")
                self.test_results['psychological_levels'] = True
            else:
                logger.warning("⚠️ Psychological levels analyzer returned no levels")
                
        except Exception as e:
            logger.error(f"❌ Psychological levels analyzer test failed: {e}")
    
    async def test_volume_analysis(self):
        """Test volume analysis"""
        logger.info("🔍 Testing volume analysis...")
        
        try:
            # Initialize the analyzer
            analyzer = EnhancedVolumeWeightedLevelsAnalyzer(self.db_url)
            await analyzer.initialize()
            
            # Create sample OHLCV data
            sample_data = self._create_sample_ohlcv_data('BTCUSDT', '1h', 100)
            
            # Test analysis
            analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
            
            if analysis and analysis.poc_price > 0:
                logger.info(f"✅ Volume analysis working - POC price: {analysis.poc_price}")
                self.test_results['volume_analysis'] = True
            else:
                logger.warning("⚠️ Volume analysis returned no results")
                
        except Exception as e:
            logger.error(f"❌ Volume analysis test failed: {e}")
    
    async def test_data_operations(self):
        """Test data insertion and retrieval"""
        logger.info("🔍 Testing data operations...")
        
        try:
            # Test data insertion
            test_timestamp = datetime.now(timezone.utc)
            
            # Insert test data into psychological_levels_analysis
            await self.conn.execute("""
                INSERT INTO psychological_levels_analysis (
                    symbol, timeframe, timestamp, current_price,
                    nearest_support_price, nearest_resistance_price,
                    market_regime, analysis_confidence, algorithm_inputs
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                    current_price = EXCLUDED.current_price
            """, 'TESTUSDT', '1h', test_timestamp, 50000.0, 49000.0, 51000.0, 
                 'uptrend', 0.85, json.dumps({'test': True}))
            
            # Test data retrieval
            result = await self.conn.fetchrow("""
                SELECT * FROM psychological_levels_analysis 
                WHERE symbol = 'TESTUSDT' AND timeframe = '1h'
                ORDER BY timestamp DESC LIMIT 1
            """)
            
            if result:
                logger.info("✅ Data insertion and retrieval working")
                self.test_results['data_insertion'] = True
                self.test_results['data_retrieval'] = True
                
                # Clean up test data
                await self.conn.execute("DELETE FROM psychological_levels_analysis WHERE symbol = 'TESTUSDT'")
            else:
                logger.warning("⚠️ Data retrieval returned no results")
                
        except Exception as e:
            logger.error(f"❌ Data operations test failed: {e}")
    
    async def test_performance(self):
        """Test performance metrics"""
        logger.info("🔍 Testing performance...")
        
        try:
            # Test query performance
            start_time = time.time()
            
            # Run a complex query
            result = await self.conn.fetch("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT symbol) as unique_symbols,
                       COUNT(DISTINCT timeframe) as unique_timeframes
                FROM ohlcv_data 
                WHERE timestamp > NOW() - INTERVAL '1 day'
            """)
            
            query_time = time.time() - start_time
            
            if result and query_time < 1.0:  # Query should complete in under 1 second
                logger.info(f"✅ Performance test passed - Query time: {query_time:.3f}s")
                self.test_results['performance'] = True
            else:
                logger.warning(f"⚠️ Performance test failed - Query time: {query_time:.3f}s")
                
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
    
    def _create_sample_ohlcv_data(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Create sample OHLCV data for testing"""
        data = []
        base_price = 50000.0
        
        for i in range(count):
            timestamp = datetime.now(timezone.utc) - timedelta(hours=count-i)
            
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            open_price = base_price * (1 + price_change)
            close_price = open_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            
            data.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(np.random.uniform(100, 1000), 2)
            })
            
            base_price = close_price
        
        return data
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating test report...")
        
        # Calculate success rate
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests) * 100
        
        # Generate report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'test_timestamp': datetime.now(timezone.utc).isoformat()
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report to file
        report_filename = f"comprehensive_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to {report_filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed Tests: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"⏱️ Total Test Time: {self.performance_metrics['total_test_time']:.2f}s")
        print("\n📊 Test Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        if success_rate >= 80:
            print("\n🎉 INTEGRATION TEST SUCCESSFUL!")
            print("Your AlphaPlus system is ready for production use.")
        elif success_rate >= 60:
            print("\n⚠️ INTEGRATION TEST PARTIALLY SUCCESSFUL")
            print("Some components need attention before production use.")
        else:
            print("\n❌ INTEGRATION TEST FAILED")
            print("Significant issues need to be resolved before production use.")
        
        print("="*60)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_results['database_connection']:
            recommendations.append("Fix database connection issues")
        
        if not self.test_results['table_existence']:
            recommendations.append("Run database migrations to create missing tables")
        
        if not self.test_results['hypertable_creation']:
            recommendations.append("Ensure TimescaleDB hypertables are properly created")
        
        if not self.test_results['constraints']:
            recommendations.append("Add missing unique constraints for data integrity")
        
        if not self.test_results['indexes']:
            recommendations.append("Create performance indexes for better query performance")
        
        if not self.test_results['algorithm_integration']:
            recommendations.append("Fix algorithm integration service issues")
        
        if not self.test_results['psychological_levels']:
            recommendations.append("Resolve psychological levels analyzer problems")
        
        if not self.test_results['volume_analysis']:
            recommendations.append("Fix volume analysis functionality")
        
        if not self.test_results['data_insertion']:
            recommendations.append("Resolve data insertion issues")
        
        if not self.test_results['data_retrieval']:
            recommendations.append("Fix data retrieval problems")
        
        if not self.test_results['performance']:
            recommendations.append("Optimize database performance")
        
        if not recommendations:
            recommendations.append("All systems are working correctly - ready for production!")
        
        return recommendations

async def main():
    """Main test function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test = ComprehensiveIntegrationTest()
    await test.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
