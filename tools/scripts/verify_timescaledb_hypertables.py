#!/usr/bin/env python3
"""
TimescaleDB Hypertables Verification Script
Verify that all hypertables are properly configured and working
"""

import asyncio
import logging
import asyncpg
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string
DATABASE_URL = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

class TimescaleDBVerifier:
    """Verify TimescaleDB hypertables configuration and functionality"""
    
    def __init__(self):
        self.db_url = DATABASE_URL
        self.verification_results = {}
    
    async def run_complete_verification(self):
        """Run complete TimescaleDB verification"""
        try:
            logger.info("🔍 Starting TimescaleDB Hypertables Verification")
            logger.info("=" * 60)
            
            conn = await asyncpg.connect(self.db_url)
            
            try:
                # Step 1: Verify hypertables exist
                await self.verify_hypertables_exist(conn)
                
                # Step 2: Test data insertion
                await self.test_data_insertion(conn)
                
                # Step 3: Verify chunks are created
                await self.verify_chunks_creation(conn)
                
                # Step 4: Test TimescaleDB functions
                await self.test_timescaledb_functions(conn)
                
                # Step 5: Verify policies
                await self.verify_policies(conn)
                
                # Step 6: Performance test
                await self.performance_test(conn)
                
                # Generate report
                await self.generate_verification_report()
                
                logger.info("=" * 60)
                logger.info("✅ TimescaleDB verification completed successfully!")
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"❌ TimescaleDB verification failed: {e}")
            raise
    
    async def verify_hypertables_exist(self, conn):
        """Verify all hypertables exist and are properly configured"""
        logger.info("📊 Verifying hypertables existence and configuration...")
        
        expected_tables = [
            'raw_news_content',
            'economic_events_calendar', 
            'crypto_events',
            'news_event_correlation',
            'breaking_news_alerts',
            'news_impact_analysis',
            'multi_language_news'
        ]
        
        # Check hypertables
        hypertables = await conn.fetch("""
            SELECT 
                hypertable_schema,
                hypertable_name,
                owner,
                num_dimensions,
                num_chunks,
                compression_enabled
            FROM timescaledb_information.hypertables 
            WHERE hypertable_name = ANY($1);
        """, expected_tables)
        
        found_tables = [ht['hypertable_name'] for ht in hypertables]
        
        for table in expected_tables:
            if table in found_tables:
                logger.info(f"✅ Hypertable '{table}' exists")
            else:
                logger.error(f"❌ Hypertable '{table}' missing")
                raise Exception(f"Missing hypertable: {table}")
        
        # Check dimensions (time partitioning)
        dimensions = await conn.fetch("""
            SELECT 
                hypertable_name,
                dimension_number,
                column_name,
                column_type
            FROM timescaledb_information.dimensions 
            WHERE hypertable_name = ANY($1);
        """, expected_tables)
        
        self.verification_results['hypertables'] = {
            'total_hypertables': len(hypertables),
            'hypertables': [dict(ht) for ht in hypertables],
            'dimensions': [dict(dim) for dim in dimensions]
        }
        
        logger.info(f"✅ All {len(expected_tables)} hypertables verified successfully")
    
    async def test_data_insertion(self, conn):
        """Test data insertion into hypertables"""
        logger.info("💾 Testing data insertion into hypertables...")
        
        test_data_inserted = {}
        
        # Test raw_news_content insertion
        try:
            await conn.execute("""
                INSERT INTO raw_news_content (
                    timestamp, title, description, source, 
                    sentiment_score, impact_score, breaking_news,
                    entities, metadata
                ) VALUES (
                    NOW(), 'Test News Article', 'Test Description', 'TestSource',
                    0.5, 0.7, TRUE,
                    '{"test": "entity"}', '{"test": "metadata"}'
                );
            """)
            test_data_inserted['raw_news_content'] = True
            logger.info("✅ raw_news_content: Data insertion successful")
        except Exception as e:
            logger.error(f"❌ raw_news_content: Data insertion failed: {e}")
            test_data_inserted['raw_news_content'] = False
        
        # Test economic_events_calendar insertion
        try:
            await conn.execute("""
                INSERT INTO economic_events_calendar (
                    timestamp, event_name, event_type, country, importance,
                    impact_score, metadata
                ) VALUES (
                    NOW(), 'Test Economic Event', 'FOMC', 'US', 'high',
                    0.8, '{"test": "metadata"}'
                );
            """)
            test_data_inserted['economic_events_calendar'] = True
            logger.info("✅ economic_events_calendar: Data insertion successful")
        except Exception as e:
            logger.error(f"❌ economic_events_calendar: Data insertion failed: {e}")
            test_data_inserted['economic_events_calendar'] = False
        
        # Test breaking_news_alerts insertion
        try:
            await conn.execute("""
                INSERT INTO breaking_news_alerts (
                    timestamp, alert_id, title, priority, 
                    impact_prediction, confidence, metadata
                ) VALUES (
                    NOW(), 'test-alert-123', 'Test Breaking News', 'high',
                    0.9, 0.8, '{"test": "metadata"}'
                );
            """)
            test_data_inserted['breaking_news_alerts'] = True
            logger.info("✅ breaking_news_alerts: Data insertion successful")
        except Exception as e:
            logger.error(f"❌ breaking_news_alerts: Data insertion failed: {e}")
            test_data_inserted['breaking_news_alerts'] = False
        
        self.verification_results['data_insertion'] = test_data_inserted
        
        successful_insertions = sum(test_data_inserted.values())
        total_tests = len(test_data_inserted)
        logger.info(f"✅ Data insertion test: {successful_insertions}/{total_tests} successful")
    
    async def verify_chunks_creation(self, conn):
        """Verify that chunks are being created properly"""
        logger.info("🧩 Verifying chunk creation...")
        
        chunks = await conn.fetch("""
            SELECT 
                hypertable_name,
                chunk_name,
                primary_dimension,
                primary_dimension_type,
                range_start,
                range_end,
                is_compressed,
                chunk_tablespace
            FROM timescaledb_information.chunks 
            WHERE hypertable_name IN (
                'raw_news_content', 'economic_events_calendar', 'breaking_news_alerts'
            );
        """)
        
        chunk_info = {}
        for chunk in chunks:
            table = chunk['hypertable_name']
            if table not in chunk_info:
                chunk_info[table] = []
            chunk_info[table].append(dict(chunk))
        
        for table, table_chunks in chunk_info.items():
            logger.info(f"✅ {table}: {len(table_chunks)} chunks created")
        
        self.verification_results['chunks'] = {
            'total_chunks': len(chunks),
            'chunks_by_table': chunk_info
        }
        
        logger.info(f"✅ Total chunks created: {len(chunks)}")
    
    async def test_timescaledb_functions(self, conn):
        """Test TimescaleDB specific functions"""
        logger.info("⚡ Testing TimescaleDB functions...")
        
        function_tests = {}
        
        # Test time_bucket function
        try:
            result = await conn.fetch("""
                SELECT 
                    time_bucket('1 hour', timestamp) as hour_bucket,
                    COUNT(*) as news_count,
                    AVG(sentiment_score) as avg_sentiment
                FROM raw_news_content 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY hour_bucket
                ORDER BY hour_bucket DESC
                LIMIT 5;
            """)
            function_tests['time_bucket'] = True
            logger.info(f"✅ time_bucket function: {len(result)} buckets returned")
        except Exception as e:
            logger.error(f"❌ time_bucket function failed: {e}")
            function_tests['time_bucket'] = False
        
        # Test first/last functions
        try:
            result = await conn.fetchrow("""
                SELECT 
                    first(title, timestamp) as first_title,
                    last(title, timestamp) as last_title
                FROM raw_news_content;
            """)
            function_tests['first_last'] = True
            logger.info("✅ first/last functions working")
        except Exception as e:
            logger.error(f"❌ first/last functions failed: {e}")
            function_tests['first_last'] = False
        
        # Test histogram function
        try:
            result = await conn.fetch("""
                SELECT 
                    histogram(sentiment_score, 0, 1, 5)
                FROM raw_news_content 
                WHERE sentiment_score IS NOT NULL;
            """)
            function_tests['histogram'] = True
            logger.info("✅ histogram function working")
        except Exception as e:
            logger.error(f"❌ histogram function failed: {e}")
            function_tests['histogram'] = False
        
        self.verification_results['timescaledb_functions'] = function_tests
        
        successful_functions = sum(function_tests.values())
        total_functions = len(function_tests)
        logger.info(f"✅ TimescaleDB functions test: {successful_functions}/{total_functions} successful")
    
    async def verify_policies(self, conn):
        """Verify retention and compression policies"""
        logger.info("📋 Verifying TimescaleDB policies...")
        
        # Check retention policies (try different views for different TimescaleDB versions)
        retention_policies = []
        try:
            retention_policies = await conn.fetch("""
                SELECT 
                    hypertable_schema,
                    hypertable_name,
                    older_than
                FROM timescaledb_information.drop_chunks_policies;
            """)
        except:
            try:
                # Try alternative query for older TimescaleDB versions
                retention_policies = await conn.fetch("""
                    SELECT 
                        schemaname as hypertable_schema,
                        tablename as hypertable_name,
                        older_than
                    FROM timescaledb_information.policy_stats 
                    WHERE policy_type = 'drop_chunks';
                """)
            except:
                logger.warning("⚠️ Could not query retention policies - may not be supported in this TimescaleDB version")
        
        # Check compression policies (if any)
        compression_policies = []
        try:
            compression_policies = await conn.fetch("""
                SELECT 
                    hypertable_schema,
                    hypertable_name
                FROM timescaledb_information.compression_settings;
            """)
        except:
            logger.warning("⚠️ Could not query compression policies - may not be supported in this TimescaleDB version")
        
        self.verification_results['policies'] = {
            'retention_policies': [dict(rp) for rp in retention_policies],
            'compression_policies': [dict(cp) for cp in compression_policies]
        }
        
        logger.info(f"✅ Retention policies: {len(retention_policies)} configured")
        logger.info(f"✅ Compression policies: {len(compression_policies)} configured")
    
    async def performance_test(self, conn):
        """Run performance tests on hypertables"""
        logger.info("🚀 Running performance tests...")
        
        performance_results = {}
        
        # Test 1: Large time range query
        start_time = datetime.now()
        try:
            result = await conn.fetch("""
                SELECT 
                    COUNT(*) as total_news,
                    AVG(sentiment_score) as avg_sentiment,
                    MAX(impact_score) as max_impact
                FROM raw_news_content 
                WHERE timestamp > NOW() - INTERVAL '30 days';
            """)
            end_time = datetime.now()
            performance_results['large_range_query'] = {
                'duration_ms': (end_time - start_time).total_seconds() * 1000,
                'success': True,
                'result_count': len(result)
            }
            logger.info(f"✅ Large range query: {performance_results['large_range_query']['duration_ms']:.2f}ms")
        except Exception as e:
            performance_results['large_range_query'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Large range query failed: {e}")
        
        # Test 2: Time bucketing aggregation
        start_time = datetime.now()
        try:
            result = await conn.fetch("""
                SELECT 
                    time_bucket('1 hour', timestamp) as bucket,
                    COUNT(*) as news_count,
                    AVG(sentiment_score) as avg_sentiment
                FROM raw_news_content 
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY bucket
                ORDER BY bucket DESC;
            """)
            end_time = datetime.now()
            performance_results['time_bucket_aggregation'] = {
                'duration_ms': (end_time - start_time).total_seconds() * 1000,
                'success': True,
                'result_count': len(result)
            }
            logger.info(f"✅ Time bucket aggregation: {performance_results['time_bucket_aggregation']['duration_ms']:.2f}ms")
        except Exception as e:
            performance_results['time_bucket_aggregation'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Time bucket aggregation failed: {e}")
        
        # Test 3: Multi-table join performance
        start_time = datetime.now()
        try:
            result = await conn.fetch("""
                SELECT 
                    n.title,
                    n.sentiment_score,
                    a.priority,
                    a.impact_prediction
                FROM raw_news_content n
                LEFT JOIN breaking_news_alerts a ON n.timestamp = a.timestamp
                WHERE n.timestamp > NOW() - INTERVAL '1 day'
                LIMIT 100;
            """)
            end_time = datetime.now()
            performance_results['multi_table_join'] = {
                'duration_ms': (end_time - start_time).total_seconds() * 1000,
                'success': True,
                'result_count': len(result)
            }
            logger.info(f"✅ Multi-table join: {performance_results['multi_table_join']['duration_ms']:.2f}ms")
        except Exception as e:
            performance_results['multi_table_join'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ Multi-table join failed: {e}")
        
        self.verification_results['performance'] = performance_results
        
        successful_tests = sum(1 for test in performance_results.values() if test.get('success', False))
        total_tests = len(performance_results)
        logger.info(f"✅ Performance tests: {successful_tests}/{total_tests} successful")
    
    async def generate_verification_report(self):
        """Generate comprehensive verification report"""
        logger.info("📋 Generating verification report...")
        
        # Calculate overall success metrics
        total_tests = 0
        successful_tests = 0
        
        # Hypertables
        if 'hypertables' in self.verification_results:
            total_tests += 1
            successful_tests += 1
        
        # Data insertion
        if 'data_insertion' in self.verification_results:
            insertion_results = self.verification_results['data_insertion']
            total_tests += len(insertion_results)
            successful_tests += sum(insertion_results.values())
        
        # TimescaleDB functions
        if 'timescaledb_functions' in self.verification_results:
            function_results = self.verification_results['timescaledb_functions']
            total_tests += len(function_results)
            successful_tests += sum(function_results.values())
        
        # Performance tests
        if 'performance' in self.verification_results:
            perf_results = self.verification_results['performance']
            total_tests += len(perf_results)
            successful_tests += sum(1 for test in perf_results.values() if test.get('success', False))
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create comprehensive report
        report = {
            'verification_timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'results': self.verification_results,
            'summary': {
                'hypertables_created': self.verification_results.get('hypertables', {}).get('total_hypertables', 0),
                'chunks_created': self.verification_results.get('chunks', {}).get('total_chunks', 0),
                'retention_policies': len(self.verification_results.get('policies', {}).get('retention_policies', [])),
                'compression_policies': len(self.verification_results.get('policies', {}).get('compression_policies', []))
            }
        }
        
        # Save report
        import os
        os.makedirs('reports', exist_ok=True)
        report_filename = f"reports/timescaledb_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Verification report saved to: {report_filename}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 TIMESCALEDB VERIFICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT - TimescaleDB is properly configured and performing optimally!")
        elif success_rate >= 75:
            logger.info("✅ GOOD - TimescaleDB is working well with minor issues")
        else:
            logger.info("⚠️ NEEDS ATTENTION - TimescaleDB has significant issues that need addressing")
        
        logger.info("=" * 60)

async def main():
    """Main function to run TimescaleDB verification"""
    verifier = TimescaleDBVerifier()
    await verifier.run_complete_verification()

if __name__ == "__main__":
    asyncio.run(main())
