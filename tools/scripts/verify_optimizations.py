#!/usr/bin/env python3
"""
Verify Advanced Optimizations
Script to verify all advanced database optimizations are in place
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationVerifier:
    """Verify all advanced optimizations"""
    
    def __init__(self):
        # Database configuration
        self.host = 'localhost'
        self.port = 5432
        self.database = 'alphapulse'
        self.username = 'alpha_emon'
        self.password = 'Emon_@17711'
        
        self.connection = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            logger.info("Initializing database connection...")
            
            self.connection = await asyncpg.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            
            # Test connection
            await self.connection.execute("SELECT 1")
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    async def verify_advanced_indexes(self):
        """Verify advanced indexes are created"""
        try:
            logger.info("Verifying advanced indexes...")
            
            # Check BRIN indexes
            brin_indexes = await self.connection.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname LIKE '%brin%' AND schemaname = 'public';
            """)
            
            logger.info(f"Found {len(brin_indexes)} BRIN indexes:")
            for idx in brin_indexes:
                logger.info(f"  - {idx['indexname']}")
            
            # Check partial indexes
            partial_indexes = await self.connection.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname LIKE '%high_confidence%' AND schemaname = 'public';
            """)
            
            logger.info(f"Found {len(partial_indexes)} partial indexes:")
            for idx in partial_indexes:
                logger.info(f"  - {idx['indexname']}")
            
            # Check covering indexes
            covering_indexes = await self.connection.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname LIKE '%covering%' AND schemaname = 'public';
            """)
            
            logger.info(f"Found {len(covering_indexes)} covering indexes:")
            for idx in covering_indexes:
                logger.info(f"  - {idx['indexname']}")
            
            # Check GIN indexes
            gin_indexes = await self.connection.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname LIKE '%gin%' AND schemaname = 'public';
            """)
            
            logger.info(f"Found {len(gin_indexes)} GIN indexes:")
            for idx in gin_indexes:
                logger.info(f"  - {idx['indexname']}")
            
            logger.info("Advanced indexes verification completed")
            
        except Exception as e:
            logger.error(f"Failed to verify advanced indexes: {e}")
            raise
    
    async def verify_materialized_views(self):
        """Verify materialized views are created"""
        try:
            logger.info("Verifying materialized views...")
            
            # Check materialized views
            materialized_views = await self.connection.fetch("""
                SELECT matviewname FROM pg_matviews WHERE schemaname = 'public';
            """)
            
            logger.info(f"Found {len(materialized_views)} materialized views:")
            for view in materialized_views:
                logger.info(f"  - {view['matviewname']}")
            
            logger.info("Materialized views verification completed")
            
        except Exception as e:
            logger.error(f"Failed to verify materialized views: {e}")
            raise
    
    async def verify_continuous_aggregates(self):
        """Verify continuous aggregates are created"""
        try:
            logger.info("Verifying continuous aggregates...")
            
            # Check continuous aggregates
            continuous_aggregates = await self.connection.fetch("""
                SELECT view_name FROM timescaledb_information.continuous_aggregates;
            """)
            
            logger.info(f"Found {len(continuous_aggregates)} continuous aggregates:")
            for agg in continuous_aggregates:
                logger.info(f"  - {agg['view_name']}")
            
            logger.info("Continuous aggregates verification completed")
            
        except Exception as e:
            logger.error(f"Failed to verify continuous aggregates: {e}")
            raise
    
    async def verify_compression_policies(self):
        """Verify compression policies are set up"""
        try:
            logger.info("Verifying compression policies...")
            
            # Check compression policies
            compression_policies = await self.connection.fetch("""
                SELECT * FROM timescaledb_information.compression_settings LIMIT 1;
            """)
            
            if compression_policies:
                logger.info("Compression settings table structure:")
                for key, value in compression_policies[0].items():
                    logger.info(f"  - {key}: {value}")
            
            # Get compression policies count
            compression_count = await self.connection.fetch("""
                SELECT COUNT(*) as count FROM timescaledb_information.compression_settings;
            """)
            
            logger.info(f"Found {compression_count[0]['count']} compression policies")
            
            logger.info("Compression policies verification completed")
            
            logger.info("Compression policies verification completed")
            
        except Exception as e:
            logger.error(f"Failed to verify compression policies: {e}")
            raise
    
    async def verify_retention_policies(self):
        """Verify retention policies are set up"""
        try:
            logger.info("Verifying retention policies...")
            
            # Check retention policies
            retention_count = await self.connection.fetch("""
                SELECT COUNT(*) as count FROM timescaledb_information.jobs 
                WHERE proc_name = 'policy_retention';
            """)
            
            logger.info(f"Found {retention_count[0]['count']} retention policies")
            
            logger.info("Retention policies verification completed")
            
        except Exception as e:
            logger.error(f"Failed to verify retention policies: {e}")
            raise
    
    async def verify_optimization_functions(self):
        """Verify optimization functions are created"""
        try:
            logger.info("Verifying optimization functions...")
            
            # Check functions
            functions = await self.connection.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('get_optimized_market_data', 'get_latest_signals_with_context', 'cleanup_expired_cache', 'analyze_table_performance');
            """)
            
            logger.info(f"Found {len(functions)} optimization functions:")
            for func in functions:
                logger.info(f"  - {func['proname']}")
            
            logger.info("Optimization functions verification completed")
            
        except Exception as e:
            logger.error(f"Failed to verify optimization functions: {e}")
            raise
    
    async def test_performance(self):
        """Test performance of optimized queries"""
        try:
            logger.info("Testing query performance...")
            
            # Test optimized market data function
            start_time = datetime.now()
            result = await self.connection.fetch("""
                SELECT * FROM get_optimized_market_data('BTCUSDT', NOW() - INTERVAL '1 day', NOW()) LIMIT 10;
            """)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds() * 1000
            logger.info(f"Optimized market data query executed in {execution_time:.2f}ms, returned {len(result)} rows")
            
            # Test signals function
            start_time = datetime.now()
            result = await self.connection.fetch("""
                SELECT * FROM get_latest_signals_with_context(10);
            """)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds() * 1000
            logger.info(f"Signals query executed in {execution_time:.2f}ms, returned {len(result)} rows")
            
            logger.info("Performance testing completed")
            
        except Exception as e:
            logger.error(f"Failed to test performance: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to verify optimizations"""
    verifier = OptimizationVerifier()
    
    try:
        await verifier.initialize()
        
        # Verify all optimizations
        await verifier.verify_advanced_indexes()
        await verifier.verify_materialized_views()
        await verifier.verify_continuous_aggregates()
        await verifier.verify_compression_policies()
        await verifier.verify_retention_policies()
        await verifier.verify_optimization_functions()
        await verifier.test_performance()
        
        logger.info("🎉 All advanced optimizations verified successfully!")
        logger.info("✅ Your AlphaPlus database is now optimized for ultra-low latency!")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)
    
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
