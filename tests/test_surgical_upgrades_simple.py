#!/usr/bin/env python3
"""
Simple Test for Surgical Upgrades Implementation
Tests core functionality of the surgical upgrades
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSurgicalUpgradesTester:
    """Simple tester for surgical upgrades"""
    
    def __init__(self):
        self.db_pool = None
        self.test_results = {}
        
    async def setup(self):
        """Setup database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='alpha_emon',
                password='Emon_@17711',
                database='alphapulse'
            )
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def test_database_tables(self):
        """Test that all surgical upgrade tables exist"""
        logger.info("🧪 Testing Database Tables...")
        
        try:
            # List of tables that should exist after surgical upgrades
            required_tables = [
                # Phase 1 tables
                'component_interface_registry',
                'interface_performance_metrics',
                'standardized_interface_results',
                'calibration_models',
                'calibration_history',
                'reliability_buckets',
                'signal_gates_config',
                'gate_validation_results',
                'gate_performance_metrics',
                'enhanced_signal_payloads',
                'signal_reasoning_chain',
                'mtf_agreement_details',
                'news_override_rules',
                'news_override_events',
                'real_time_news_feed',
                'quota_configuration',
                'quota_usage_tracking',
                'quota_replacement_events',
                'active_signal_mutex',
                'cooldown_tracking',
                'signal_lifecycle_events',
                'data_sanity_rules',
                'data_sanity_validation_results',
                'orderbook_health_metrics',
                
                # Phase 2 tables
                'calibrated_signal_generation',
                'confidence_fusion_components',
                'ensemble_voting_results',
                'calibration_training_data',
                'calibration_model_performance',
                'dynamic_threshold_adjustment',
                'gate_execution_engine',
                'gate_dependencies',
                'gate_performance_analytics',
                'signal_state_machine',
                'signal_expiry_management',
                'signal_cooldown_management',
                'priority_signal_queue',
                'quota_replacement_history',
                'dynamic_quota_adjustment',
                'real_time_validation_rules',
                'validation_rule_execution',
                'market_condition_validation',
                'enhanced_performance_metrics',
                'performance_alerting',
                'system_health_monitoring'
            ]
            
            missing_tables = []
            
            async with self.db_pool.acquire() as conn:
                for table in required_tables:
                    result = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)
                    if result['count'] == 0:
                        missing_tables.append(table)
            
            if missing_tables:
                logger.error(f"❌ Missing tables: {missing_tables}")
                return False
            else:
                logger.info(f"✅ All {len(required_tables)} required tables exist")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database tables test failed: {e}")
            return False
    
    async def test_default_configurations(self):
        """Test that default configurations are in place"""
        logger.info("🧪 Testing Default Configurations...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Test gate configurations
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM signal_gates_config 
                    WHERE is_enabled = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Gate configurations: {result['count']} active gates")
                else:
                    logger.error("❌ No active gate configurations")
                    return False
                
                # Test quota configurations
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM quota_configuration 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Quota configurations: {result['count']} active quotas")
                else:
                    logger.error("❌ No active quota configurations")
                    return False
                
                # Test dynamic thresholds
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM dynamic_threshold_adjustment
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Dynamic thresholds: {result['count']} configured")
                else:
                    logger.error("❌ No dynamic thresholds configured")
                    return False
                
                # Test validation rules
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM real_time_validation_rules 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Validation rules: {result['count']} active rules")
                else:
                    logger.warning("⚠️ No validation rules found")
                
                # Test news override rules
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM news_override_rules 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ News override rules: {result['count']} active rules")
                else:
                    logger.warning("⚠️ No news override rules found")
                
                # Test data sanity rules
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM data_sanity_rules 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Data sanity rules: {result['count']} active rules")
                else:
                    logger.warning("⚠️ No data sanity rules found")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Default configurations test failed: {e}")
            return False
    
    async def test_interface_registry(self):
        """Test interface registry functionality"""
        logger.info("🧪 Testing Interface Registry...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check interface registry entries
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM component_interface_registry
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Interface registry: {result['count']} entries")
                else:
                    logger.error("❌ No interface registry entries")
                    return False
                
                # Check performance metrics
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM interface_performance_metrics
                """)
                if result['count'] >= 0:  # Can be 0 initially
                    logger.info(f"✅ Interface performance metrics: {result['count']} entries")
                else:
                    logger.error("❌ Interface performance metrics table error")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Interface registry test failed: {e}")
            return False
    
    async def test_calibration_system(self):
        """Test calibration system"""
        logger.info("🧪 Testing Calibration System...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check calibration models
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM calibration_models
                """)
                if result['count'] >= 0:  # Can be 0 initially
                    logger.info(f"✅ Calibration models: {result['count']} models")
                else:
                    logger.error("❌ Calibration models table error")
                    return False
                
                # Check reliability buckets
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM reliability_buckets
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Reliability buckets: {result['count']} buckets")
                else:
                    logger.warning("⚠️ No reliability buckets found")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Calibration system test failed: {e}")
            return False
    
    async def test_performance_monitoring(self):
        """Test performance monitoring"""
        logger.info("🧪 Testing Performance Monitoring...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check system health monitoring
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM system_health_monitoring
                """)
                if result['count'] >= 0:  # Can be 0 initially
                    logger.info(f"✅ System health monitoring: {result['count']} components")
                else:
                    logger.error("❌ System health monitoring table error")
                    return False
                
                # Check enhanced performance metrics
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM enhanced_performance_metrics
                """)
                if result['count'] >= 0:  # Can be 0 initially
                    logger.info(f"✅ Enhanced performance metrics: {result['count']} entries")
                else:
                    logger.error("❌ Enhanced performance metrics table error")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Performance monitoring test failed: {e}")
            return False
    
    async def run_simple_test(self):
        """Run simple test suite"""
        logger.info("🚀 Starting Simple Surgical Upgrades Test...")
        
        if not await self.setup():
            return False
        
        test_functions = [
            self.test_database_tables,
            self.test_default_configurations,
            self.test_interface_registry,
            self.test_calibration_system,
            self.test_performance_monitoring
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_func in test_functions:
            try:
                if await test_func():
                    passed_tests += 1
                    self.test_results[test_func.__name__] = True
                else:
                    self.test_results[test_func.__name__] = False
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed with exception: {e}")
                self.test_results[test_func.__name__] = False
        
        # Generate test summary
        logger.info("\n" + "="*60)
        logger.info("📊 SIMPLE SURGICAL UPGRADES TEST SUMMARY")
        logger.info("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL SIMPLE SURGICAL UPGRADES TESTS PASSED!")
            return True
        else:
            logger.error(f"❌ {total_tests - passed_tests} tests failed")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()

async def main():
    """Main test function"""
    tester = SimpleSurgicalUpgradesTester()
    try:
        success = await tester.run_simple_test()
        return success
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
