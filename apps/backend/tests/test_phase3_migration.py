"""
Test script to verify Phase 3: Automated Model Retraining migration
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase3MigrationTest:
    """Test the Phase 3 migration results"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'alpha_emon',
            'password': 'Emon_@17711',
            'database': 'alphapulse'
        }
    
    async def create_connection_pool(self) -> asyncpg.Pool:
        """Create database connection pool"""
        try:
            pool = await asyncpg.create_pool(**self.db_config)
            logger.info("✅ Database connection pool created successfully")
            return pool
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            raise
    
    async def test_table_existence(self, conn: asyncpg.Connection) -> Dict[str, bool]:
        """Test if all Phase 3 tables exist"""
        tables_to_check = [
            'model_training_jobs',
            'training_data_management', 
            'model_performance_tracking',
            'model_deployment_pipeline',
            'ab_testing_framework',
            'model_versioning',
            'real_time_model_performance',
            'model_drift_detection',
            'automated_retraining_triggers'
        ]
        
        results = {}
        for table in tables_to_check:
            try:
                result = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = $1
                    )
                """, table)
                results[table] = result
                status = "✅" if result else "❌"
                logger.info(f"{status} Table {table}: {'EXISTS' if result else 'MISSING'}")
            except Exception as e:
                results[table] = False
                logger.error(f"❌ Error checking table {table}: {e}")
        
        return results
    
    async def test_default_configurations(self, conn: asyncpg.Connection) -> Dict[str, int]:
        """Test if default configurations were inserted"""
        configs_to_check = [
            ('automated_retraining_triggers', 'model_name'),
            ('quality_gates', 'gate_name'),
            ('ml_ops_alerts', 'alert_name')
        ]
        
        results = {}
        for table, column in configs_to_check:
            try:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                results[table] = count
                logger.info(f"📊 Table {table}: {count} records")
            except Exception as e:
                results[table] = 0
                logger.error(f"❌ Error checking {table}: {e}")
        
        return results
    
    async def test_index_existence(self, conn: asyncpg.Connection) -> Dict[str, bool]:
        """Test if key indexes exist"""
        indexes_to_check = [
            'idx_training_jobs_status',
            'idx_training_jobs_model_name',
            'idx_training_data_name',
            'idx_performance_model_id',
            'idx_deployment_status',
            'idx_realtime_model_id',
            'idx_drift_model_id',
            'idx_retraining_triggers_model_name'
        ]
        
        results = {}
        for index in indexes_to_check:
            try:
                result = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes 
                        WHERE indexname = $1
                    )
                """, index)
                results[index] = result
                status = "✅" if result else "❌"
                logger.info(f"{status} Index {index}: {'EXISTS' if result else 'MISSING'}")
            except Exception as e:
                results[index] = False
                logger.error(f"❌ Error checking index {index}: {e}")
        
        return results
    
    async def test_table_structure(self, conn: asyncpg.Connection) -> Dict[str, Dict]:
        """Test table structure for key tables"""
        tables_to_check = [
            'model_training_jobs',
            'automated_retraining_triggers'
        ]
        
        results = {}
        for table in tables_to_check:
            try:
                columns = await conn.fetch(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = $1
                    ORDER BY ordinal_position
                """, table)
                
                results[table] = {
                    'column_count': len(columns),
                    'columns': [dict(col) for col in columns]
                }
                logger.info(f"📋 Table {table}: {len(columns)} columns")
            except Exception as e:
                results[table] = {'error': str(e)}
                logger.error(f"❌ Error checking structure of {table}: {e}")
        
        return results
    
    async def run_tests(self):
        """Run all tests"""
        pool = None
        try:
            pool = await self.create_connection_pool()
            
            async with pool.acquire() as conn:
                logger.info("🔍 Testing Phase 3 Migration Results...")
                logger.info("=" * 50)
                
                # Test table existence
                logger.info("📋 Testing Table Existence:")
                table_results = await self.test_table_existence(conn)
                
                logger.info("")
                logger.info("📊 Testing Default Configurations:")
                config_results = await self.test_default_configurations(conn)
                
                logger.info("")
                logger.info("🔍 Testing Index Existence:")
                index_results = await self.test_index_existence(conn)
                
                logger.info("")
                logger.info("🏗️ Testing Table Structure:")
                structure_results = await self.test_table_structure(conn)
                
                # Summary
                logger.info("")
                logger.info("=" * 50)
                logger.info("📈 PHASE 3 MIGRATION TEST SUMMARY:")
                logger.info("=" * 50)
                
                tables_exist = all(table_results.values())
                configs_exist = any(config_results.values())
                indexes_exist = any(index_results.values())
                
                logger.info(f"Tables Created: {'✅' if tables_exist else '❌'} ({sum(table_results.values())}/{len(table_results)})")
                logger.info(f"Default Configs: {'✅' if configs_exist else '❌'} ({sum(config_results.values())} total records)")
                logger.info(f"Indexes Created: {'✅' if indexes_exist else '❌'} ({sum(index_results.values())}/{len(index_results)})")
                
                if tables_exist and configs_exist and indexes_exist:
                    logger.info("🎉 PHASE 3 MIGRATION: SUCCESSFUL!")
                else:
                    logger.warning("⚠️ PHASE 3 MIGRATION: PARTIAL SUCCESS - Some components may need attention")
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        finally:
            if pool:
                await pool.close()

async def main():
    """Main test function"""
    test = Phase3MigrationTest()
    await test.run_tests()

if __name__ == "__main__":
    asyncio.run(main())
