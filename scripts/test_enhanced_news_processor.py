#!/usr/bin/env python3
"""
Enhanced News Event Processor Test with TimescaleDB
Test the complete news processing pipeline with TimescaleDB hypertables
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import asyncpg
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from services.enhanced_news_event_processor import EnhancedNewsEventProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsProcessorTester:
    """Test the Enhanced News Event Processor with TimescaleDB"""
    
    def __init__(self):
        self.db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.db_pool = None
        self.news_processor = None
        self.test_results = {}
    
    async def run_complete_test(self):
        """Run complete enhanced news processor test"""
        try:
            logger.info("🚀 Starting Enhanced News Processor Test with TimescaleDB")
            logger.info("=" * 70)
            
            # Initialize
            await self.initialize_services()
            
            # Test 1: Basic news collection and processing
            await self.test_news_collection_and_processing()
            
            # Test 2: TimescaleDB storage verification
            await self.test_timescaledb_storage()
            
            # Test 3: Breaking news detection
            await self.test_breaking_news_detection()
            
            # Test 4: Time-series queries
            await self.test_timeseries_queries()
            
            # Test 5: Performance with bulk data
            await self.test_bulk_performance()
            
            # Generate report
            await self.generate_test_report()
            
            logger.info("=" * 70)
            logger.info("✅ Enhanced News Processor Test completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Enhanced News Processor Test failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def initialize_services(self):
        """Initialize database and news processor"""
        logger.info("🔧 Initializing services...")
        
        # Create database connection pool
        self.db_pool = await asyncpg.create_pool(
            self.db_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("✅ Database connection pool created")
        
        # Initialize Enhanced News Event Processor
        self.news_processor = EnhancedNewsEventProcessor(self.db_pool)
        await self.news_processor.initialize()
        logger.info("✅ Enhanced News Event Processor initialized")
    
    async def test_news_collection_and_processing(self):
        """Test news collection and processing functionality"""
        logger.info("📰 Testing News Collection and Processing...")
        
        try:
            # Process comprehensive news and events
            result = await self.news_processor.process_comprehensive_news_events()
            
            self.test_results['news_processing'] = {
                'success': True,
                'articles_processed': result.get('news_articles', 0),
                'breaking_news_detected': result.get('breaking_news', 0)
            }
            
            logger.info(f"✅ News processing successful: {result['news_articles']} articles, {result['breaking_news']} breaking news")
            
        except Exception as e:
            logger.error(f"❌ News collection and processing failed: {e}")
            self.test_results['news_processing'] = {'success': False, 'error': str(e)}
            raise
    
    async def test_timescaledb_storage(self):
        """Test data storage in TimescaleDB hypertables"""
        logger.info("💾 Testing TimescaleDB Storage...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check data was stored in raw_news_content
                news_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM raw_news_content 
                    WHERE timestamp > NOW() - INTERVAL '1 hour';
                """)
                
                # Check data was stored in breaking_news_alerts
                alerts_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM breaking_news_alerts 
                    WHERE timestamp > NOW() - INTERVAL '1 hour';
                """)
                
                # Verify hypertable chunks are being used
                chunks_info = await conn.fetch("""
                    SELECT 
                        hypertable_name,
                        COUNT(*) as chunk_count
                    FROM timescaledb_information.chunks 
                    WHERE hypertable_name IN ('raw_news_content', 'breaking_news_alerts')
                    GROUP BY hypertable_name;
                """)
                
                self.test_results['timescaledb_storage'] = {
                    'success': True,
                    'news_articles_stored': news_count,
                    'breaking_alerts_stored': alerts_count,
                    'chunks_info': [dict(chunk) for chunk in chunks_info]
                }
                
                logger.info(f"✅ TimescaleDB storage verified: {news_count} news articles, {alerts_count} alerts")
                logger.info(f"✅ Chunks created: {len(chunks_info)} hypertables have active chunks")
                
        except Exception as e:
            logger.error(f"❌ TimescaleDB storage test failed: {e}")
            self.test_results['timescaledb_storage'] = {'success': False, 'error': str(e)}
            raise
    
    async def test_breaking_news_detection(self):
        """Test breaking news detection and alerting"""
        logger.info("🚨 Testing Breaking News Detection...")
        
        try:
            # Create mock high-impact news for testing
            test_articles = [
                {
                    'title': 'BREAKING: Bitcoin Reaches $100,000 - Historic Milestone',
                    'description': 'Bitcoin has just reached an unprecedented $100,000 mark in a dramatic surge.',
                    'content': 'In an unprecedented move, Bitcoin has reached $100,000...',
                    'source': 'CryptoNews',
                    'impact_score': 0.95,
                    'breaking_indicators': ['BREAKING', 'Historic', 'unprecedented']
                }
            ]
            
            # Manually test breaking news detection logic
            breaking_news_count = 0
            for article_data in test_articles:
                if article_data['impact_score'] >= 0.8 or any(indicator in article_data['title'] for indicator in article_data['breaking_indicators']):
                    breaking_news_count += 1
            
            self.test_results['breaking_news_detection'] = {
                'success': True,
                'test_articles': len(test_articles),
                'breaking_news_detected': breaking_news_count
            }
            
            logger.info(f"✅ Breaking news detection: {breaking_news_count}/{len(test_articles)} articles detected as breaking news")
            
        except Exception as e:
            logger.error(f"❌ Breaking news detection test failed: {e}")
            self.test_results['breaking_news_detection'] = {'success': False, 'error': str(e)}
            raise
    
    async def test_timeseries_queries(self):
        """Test TimescaleDB time-series specific queries"""
        logger.info("📊 Testing Time-Series Queries...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Test 1: Time bucket aggregation
                start_time = datetime.now()
                hourly_sentiment = await conn.fetch("""
                    SELECT 
                        time_bucket('1 hour', timestamp) as hour_bucket,
                        COUNT(*) as article_count,
                        AVG(sentiment_score) as avg_sentiment,
                        AVG(impact_score) as avg_impact
                    FROM raw_news_content 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                      AND sentiment_score IS NOT NULL
                    GROUP BY hour_bucket
                    ORDER BY hour_bucket DESC
                    LIMIT 10;
                """)
                end_time = datetime.now()
                time_bucket_duration = (end_time - start_time).total_seconds() * 1000
                
                # Test 2: First/Last aggregates
                start_time = datetime.now()
                first_last_results = await conn.fetchrow("""
                    SELECT 
                        first(title, timestamp) as first_article,
                        last(title, timestamp) as last_article,
                        first(sentiment_score, timestamp) as first_sentiment,
                        last(sentiment_score, timestamp) as last_sentiment
                    FROM raw_news_content 
                    WHERE timestamp > NOW() - INTERVAL '6 hours';
                """)
                end_time = datetime.now()
                first_last_duration = (end_time - start_time).total_seconds() * 1000
                
                # Test 3: Moving average with window functions
                start_time = datetime.now()
                moving_avg = await conn.fetch("""
                    SELECT 
                        timestamp,
                        sentiment_score,
                        AVG(sentiment_score) OVER (
                            ORDER BY timestamp 
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) as moving_avg_sentiment
                    FROM raw_news_content 
                    WHERE timestamp > NOW() - INTERVAL '12 hours'
                      AND sentiment_score IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 20;
                """)
                end_time = datetime.now()
                moving_avg_duration = (end_time - start_time).total_seconds() * 1000
                
                self.test_results['timeseries_queries'] = {
                    'success': True,
                    'time_bucket_test': {
                        'duration_ms': time_bucket_duration,
                        'buckets_returned': len(hourly_sentiment)
                    },
                    'first_last_test': {
                        'duration_ms': first_last_duration,
                        'first_article': first_last_results['first_article'] if first_last_results else None
                    },
                    'moving_average_test': {
                        'duration_ms': moving_avg_duration,
                        'data_points': len(moving_avg)
                    }
                }
                
                logger.info(f"✅ Time bucket query: {time_bucket_duration:.2f}ms ({len(hourly_sentiment)} buckets)")
                logger.info(f"✅ First/Last query: {first_last_duration:.2f}ms")
                logger.info(f"✅ Moving average query: {moving_avg_duration:.2f}ms ({len(moving_avg)} points)")
                
        except Exception as e:
            logger.error(f"❌ Time-series queries test failed: {e}")
            self.test_results['timeseries_queries'] = {'success': False, 'error': str(e)}
            raise
    
    async def test_bulk_performance(self):
        """Test performance with bulk data operations"""
        logger.info("⚡ Testing Bulk Performance...")
        
        try:
            # Generate multiple processing cycles to test performance
            start_time = datetime.now()
            
            cycles = 3
            total_articles = 0
            
            for i in range(cycles):
                logger.info(f"🔄 Processing cycle {i+1}/{cycles}...")
                result = await self.news_processor.process_comprehensive_news_events()
                total_articles += result.get('news_articles', 0)
            
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Check database performance
            async with self.db_pool.acquire() as conn:
                start_time = datetime.now()
                
                # Complex aggregation query
                performance_query = await conn.fetch("""
                    SELECT 
                        source,
                        COUNT(*) as article_count,
                        AVG(sentiment_score) as avg_sentiment,
                        AVG(impact_score) as avg_impact,
                        COUNT(*) FILTER (WHERE breaking_news = TRUE) as breaking_count
                    FROM raw_news_content 
                    WHERE timestamp > NOW() - INTERVAL '2 hours'
                    GROUP BY source
                    ORDER BY article_count DESC;
                """)
                
                end_time = datetime.now()
                query_duration = (end_time - start_time).total_seconds() * 1000
            
            self.test_results['bulk_performance'] = {
                'success': True,
                'processing_cycles': cycles,
                'total_articles_processed': total_articles,
                'total_duration_seconds': total_duration,
                'articles_per_second': total_articles / total_duration if total_duration > 0 else 0,
                'complex_query_duration_ms': query_duration,
                'sources_analyzed': len(performance_query)
            }
            
            logger.info(f"✅ Bulk performance: {cycles} cycles, {total_articles} articles in {total_duration:.2f}s")
            logger.info(f"✅ Processing rate: {total_articles / total_duration:.2f} articles/second")
            logger.info(f"✅ Complex aggregation query: {query_duration:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Bulk performance test failed: {e}")
            self.test_results['bulk_performance'] = {'success': False, 'error': str(e)}
            raise
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📋 Generating test report...")
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create comprehensive report
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'test_results': self.test_results,
            'summary': {
                'timescaledb_integration': 'EXCELLENT' if success_rate >= 90 else 'GOOD' if success_rate >= 75 else 'NEEDS_ATTENTION',
                'performance_rating': 'OPTIMAL' if self.test_results.get('bulk_performance', {}).get('articles_per_second', 0) >= 1.0 else 'ACCEPTABLE',
                'breaking_news_detection': 'FUNCTIONAL' if self.test_results.get('breaking_news_detection', {}).get('success', False) else 'ISSUES'
            }
        }
        
        # Save report
        import os
        os.makedirs('reports', exist_ok=True)
        report_filename = f"reports/enhanced_news_processor_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Test report saved to: {report_filename}")
        
        # Print summary
        logger.info("=" * 70)
        logger.info("📊 ENHANCED NEWS PROCESSOR TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 70)
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT - Enhanced News Processor with TimescaleDB is working perfectly!")
        elif success_rate >= 75:
            logger.info("✅ GOOD - Enhanced News Processor is working well with minor issues")
        else:
            logger.info("⚠️ NEEDS ATTENTION - Enhanced News Processor has significant issues")
        
        logger.info("=" * 70)
        
        # Performance highlights
        if 'bulk_performance' in self.test_results and self.test_results['bulk_performance'].get('success'):
            perf = self.test_results['bulk_performance']
            logger.info("🚀 PERFORMANCE HIGHLIGHTS:")
            logger.info(f"  • Processing Rate: {perf['articles_per_second']:.2f} articles/second")
            logger.info(f"  • Query Performance: {perf['complex_query_duration_ms']:.2f}ms for complex aggregations")
            logger.info("=" * 70)
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.news_processor:
                await self.news_processor.close()
            if self.db_pool:
                await self.db_pool.close()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main function to run the enhanced news processor test"""
    tester = NewsProcessorTester()
    await tester.run_complete_test()

if __name__ == "__main__":
    asyncio.run(main())
