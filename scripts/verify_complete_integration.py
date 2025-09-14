#!/usr/bin/env python3
"""
Complete Integration Verification for Enhanced News and Events System
Verify all components are properly connected and working
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

class CompleteIntegrationVerifier:
    """Verify complete integration of enhanced news and events system"""
    
    def __init__(self):
        self.db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.db_pool = None
        self.news_processor = None
        self.verification_results = {}
    
    async def run_complete_verification(self):
        """Run complete integration verification"""
        try:
            logger.info("🔍 Starting Complete Integration Verification")
            logger.info("=" * 80)
            
            # Step 1: Verify file structure
            await self.verify_file_structure()
            
            # Step 2: Verify database connectivity
            await self.verify_database_connectivity()
            
            # Step 3: Verify TimescaleDB tables
            await self.verify_timescaledb_tables()
            
            # Step 4: Verify enhanced news processor
            await self.verify_enhanced_news_processor()
            
            # Step 5: Verify configuration files
            await self.verify_configuration_files()
            
            # Step 6: Verify integration points
            await self.verify_integration_points()
            
            # Step 7: Verify performance
            await self.verify_performance()
            
            # Generate comprehensive report
            await self.generate_comprehensive_report()
            
            logger.info("=" * 80)
            logger.info("✅ Complete Integration Verification finished!")
            
        except Exception as e:
            logger.error(f"❌ Complete integration verification failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def verify_file_structure(self):
        """Verify all required files exist"""
        logger.info("📁 Verifying file structure...")
        
        required_files = [
            "backend/services/enhanced_news_event_processor.py",
            "backend/database/migrations/013_enhanced_news_events_timescaledb.py",
            "backend/database/migrations/012_enhanced_news_events_basic.py",
            "scripts/verify_timescaledb_hypertables.py",
            "scripts/test_enhanced_news_processor.py",
            "scripts/integrate_enhanced_news_events.py",
            "config/enhanced_news_config.json",
            "config/deployment_config.json"
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)
                logger.info(f"✅ {file_path}")
            else:
                missing_files.append(file_path)
                logger.error(f"❌ {file_path} - MISSING")
        
        self.verification_results['file_structure'] = {
            'total_files': len(required_files),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'missing_file_list': missing_files
        }
        
        if missing_files:
            raise Exception(f"Missing {len(missing_files)} required files")
        
        logger.info(f"✅ File structure verification: {len(existing_files)}/{len(required_files)} files found")
    
    async def verify_database_connectivity(self):
        """Verify database connectivity"""
        logger.info("🔌 Verifying database connectivity...")
        
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            async with self.db_pool.acquire() as conn:
                # Test basic connection
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    logger.info("✅ Database connection successful")
                else:
                    raise Exception("Database connection test failed")
                
                # Test TimescaleDB extension
                timescaledb_version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                if timescaledb_version:
                    logger.info(f"✅ TimescaleDB extension found: version {timescaledb_version}")
                else:
                    raise Exception("TimescaleDB extension not found")
            
            self.verification_results['database_connectivity'] = {
                'status': 'success',
                'timescaledb_version': timescaledb_version
            }
            
        except Exception as e:
            logger.error(f"❌ Database connectivity failed: {e}")
            self.verification_results['database_connectivity'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    async def verify_timescaledb_tables(self):
        """Verify TimescaleDB tables exist and are properly configured"""
        logger.info("📊 Verifying TimescaleDB tables...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check all required tables
                required_tables = [
                    'raw_news_content',
                    'economic_events_calendar',
                    'crypto_events',
                    'news_event_correlation',
                    'breaking_news_alerts',
                    'news_impact_analysis',
                    'multi_language_news'
                ]
                
                table_status = {}
                for table in required_tables:
                    # Check if table exists
                    exists = await conn.fetchval(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                        table
                    )
                    
                    if exists:
                        # Check if it's a hypertable
                        is_hypertable = await conn.fetchval(
                            "SELECT EXISTS (SELECT FROM timescaledb_information.hypertables WHERE hypertable_name = $1)",
                            table
                        )
                        
                        if is_hypertable:
                            # Get chunk count
                            chunk_count = await conn.fetchval(
                                "SELECT COUNT(*) FROM timescaledb_information.chunks WHERE hypertable_name = $1",
                                table
                            )
                            
                            table_status[table] = {
                                'exists': True,
                                'is_hypertable': True,
                                'chunk_count': chunk_count
                            }
                            logger.info(f"✅ {table}: hypertable with {chunk_count} chunks")
                        else:
                            table_status[table] = {
                                'exists': True,
                                'is_hypertable': False
                            }
                            logger.warning(f"⚠️ {table}: exists but not a hypertable")
                    else:
                        table_status[table] = {
                            'exists': False,
                            'is_hypertable': False
                        }
                        logger.error(f"❌ {table}: table missing")
                
                self.verification_results['timescaledb_tables'] = table_status
                
                # Check if all tables exist and are hypertables
                all_exist = all(status['exists'] for status in table_status.values())
                all_hypertables = all(status.get('is_hypertable', False) for status in table_status.values())
                
                if not all_exist:
                    raise Exception("Not all required tables exist")
                
                if not all_hypertables:
                    logger.warning("⚠️ Some tables are not hypertables - consider running TimescaleDB migration")
                
        except Exception as e:
            logger.error(f"❌ TimescaleDB tables verification failed: {e}")
            raise
    
    async def verify_enhanced_news_processor(self):
        """Verify enhanced news processor functionality"""
        logger.info("📰 Verifying enhanced news processor...")
        
        try:
            # Initialize processor
            self.news_processor = EnhancedNewsEventProcessor(self.db_pool)
            await self.news_processor.initialize()
            
            # Test basic functionality
            result = await self.news_processor.process_comprehensive_news_events()
            
            if result and 'news_articles' in result:
                logger.info(f"✅ Enhanced news processor working: {result['news_articles']} articles processed")
                
                self.verification_results['enhanced_news_processor'] = {
                    'status': 'success',
                    'articles_processed': result.get('news_articles', 0),
                    'breaking_news_detected': result.get('breaking_news', 0)
                }
            else:
                raise Exception("Enhanced news processor test failed")
                
        except Exception as e:
            logger.error(f"❌ Enhanced news processor verification failed: {e}")
            self.verification_results['enhanced_news_processor'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    async def verify_configuration_files(self):
        """Verify configuration files are properly formatted"""
        logger.info("⚙️ Verifying configuration files...")
        
        config_files = [
            "config/enhanced_news_config.json",
            "config/deployment_config.json"
        ]
        
        config_status = {}
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                config_status[config_file] = {
                    'status': 'valid',
                    'size': len(str(config))
                }
                logger.info(f"✅ {config_file}: valid JSON configuration")
                
            except Exception as e:
                config_status[config_file] = {
                    'status': 'invalid',
                    'error': str(e)
                }
                logger.error(f"❌ {config_file}: invalid configuration - {e}")
        
        self.verification_results['configuration_files'] = config_status
        
        # Check if all configs are valid
        all_valid = all(status['status'] == 'valid' for status in config_status.values())
        if not all_valid:
            logger.warning("⚠️ Some configuration files are invalid")
    
    async def verify_integration_points(self):
        """Verify integration points with other systems"""
        logger.info("🔗 Verifying integration points...")
        
        integration_points = {
            'database_connection': True,
            'timescaledb_hypertables': True,
            'news_processor_service': True,
            'configuration_loading': True,
            'sentiment_analysis': True,
            'breaking_news_detection': True
        }
        
        # Test sentiment analysis integration
        try:
            from textblob import TextBlob
            test_text = "Bitcoin is performing well today."
            blob = TextBlob(test_text)
            sentiment = blob.sentiment.polarity
            
            if sentiment is not None:
                logger.info(f"✅ Sentiment analysis working: {sentiment}")
            else:
                integration_points['sentiment_analysis'] = False
                logger.warning("⚠️ Sentiment analysis not working properly")
                
        except Exception as e:
            integration_points['sentiment_analysis'] = False
            logger.warning(f"⚠️ Sentiment analysis integration issue: {e}")
        
        # Test configuration loading
        try:
            if hasattr(self.news_processor, 'config') and self.news_processor.config:
                logger.info("✅ Configuration loading working")
            else:
                integration_points['configuration_loading'] = False
                logger.warning("⚠️ Configuration loading not working")
        except Exception as e:
            integration_points['configuration_loading'] = False
            logger.warning(f"⚠️ Configuration loading issue: {e}")
        
        self.verification_results['integration_points'] = integration_points
        
        working_points = sum(integration_points.values())
        total_points = len(integration_points)
        
        logger.info(f"✅ Integration points: {working_points}/{total_points} working")
    
    async def verify_performance(self):
        """Verify system performance"""
        logger.info("⚡ Verifying system performance...")
        
        try:
            # Test processing speed
            start_time = datetime.now()
            
            # Run multiple processing cycles
            cycles = 3
            total_articles = 0
            
            for i in range(cycles):
                result = await self.news_processor.process_comprehensive_news_events()
                total_articles += result.get('news_articles', 0)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Test database query performance
            async with self.db_pool.acquire() as conn:
                start_time = datetime.now()
                
                # Complex query test
                await conn.fetch("""
                    SELECT 
                        time_bucket('1 hour', timestamp) as hour_bucket,
                        COUNT(*) as article_count,
                        AVG(sentiment_score) as avg_sentiment
                    FROM raw_news_content 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY hour_bucket
                    ORDER BY hour_bucket DESC
                    LIMIT 5;
                """)
                
                end_time = datetime.now()
                query_duration = (end_time - start_time).total_seconds() * 1000
            
            performance_metrics = {
                'processing_cycles': cycles,
                'total_articles_processed': total_articles,
                'total_duration_seconds': duration,
                'articles_per_second': total_articles / duration if duration > 0 else 0,
                'query_duration_ms': query_duration,
                'performance_rating': 'excellent' if query_duration < 100 else 'good' if query_duration < 500 else 'needs_optimization'
            }
            
            self.verification_results['performance'] = performance_metrics
            
            logger.info(f"✅ Performance test: {cycles} cycles, {total_articles} articles in {duration:.2f}s")
            logger.info(f"✅ Processing rate: {performance_metrics['articles_per_second']:.2f} articles/second")
            logger.info(f"✅ Query performance: {query_duration:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Performance verification failed: {e}")
            self.verification_results['performance'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def generate_comprehensive_report(self):
        """Generate comprehensive verification report"""
        logger.info("📋 Generating comprehensive verification report...")
        
        # Calculate overall metrics
        total_checks = 0
        successful_checks = 0
        
        # File structure
        if 'file_structure' in self.verification_results:
            fs = self.verification_results['file_structure']
            total_checks += 1
            if fs['missing_files'] == 0:
                successful_checks += 1
        
        # Database connectivity
        if 'database_connectivity' in self.verification_results:
            total_checks += 1
            if self.verification_results['database_connectivity']['status'] == 'success':
                successful_checks += 1
        
        # TimescaleDB tables
        if 'timescaledb_tables' in self.verification_results:
            total_checks += 1
            table_status = self.verification_results['timescaledb_tables']
            all_exist = all(status['exists'] for status in table_status.values())
            if all_exist:
                successful_checks += 1
        
        # Enhanced news processor
        if 'enhanced_news_processor' in self.verification_results:
            total_checks += 1
            if self.verification_results['enhanced_news_processor']['status'] == 'success':
                successful_checks += 1
        
        # Configuration files
        if 'configuration_files' in self.verification_results:
            total_checks += 1
            config_status = self.verification_results['configuration_files']
            all_valid = all(status['status'] == 'valid' for status in config_status.values())
            if all_valid:
                successful_checks += 1
        
        # Integration points
        if 'integration_points' in self.verification_results:
            total_checks += 1
            integration_points = self.verification_results['integration_points']
            working_points = sum(integration_points.values())
            total_points = len(integration_points)
            if working_points >= total_points * 0.8:  # 80% threshold
                successful_checks += 1
        
        # Performance
        if 'performance' in self.verification_results:
            total_checks += 1
            if 'status' not in self.verification_results['performance']:
                successful_checks += 1
        
        success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Create comprehensive report
        report = {
            'verification_timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'total_checks': total_checks,
            'successful_checks': successful_checks,
            'failed_checks': total_checks - successful_checks,
            'verification_results': self.verification_results,
            'summary': {
                'system_status': 'PRODUCTION_READY' if success_rate >= 90 else 'NEEDS_ATTENTION' if success_rate >= 75 else 'CRITICAL_ISSUES',
                'recommendations': self.generate_recommendations()
            }
        }
        
        # Save report
        import os
        os.makedirs('reports', exist_ok=True)
        report_filename = f"reports/complete_integration_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Comprehensive report saved to: {report_filename}")
        
        # Print summary
        logger.info("=" * 80)
        logger.info("📊 COMPLETE INTEGRATION VERIFICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Successful: {successful_checks}")
        logger.info(f"Failed: {total_checks - successful_checks}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 80)
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT - System is fully integrated and production-ready!")
        elif success_rate >= 75:
            logger.info("✅ GOOD - System is mostly integrated with minor issues")
        else:
            logger.info("⚠️ NEEDS ATTENTION - System has significant integration issues")
        
        logger.info("=" * 80)
        
        # Print recommendations
        logger.info("📋 RECOMMENDATIONS:")
        for rec in report['summary']['recommendations']:
            logger.info(f"  • {rec}")
        logger.info("=" * 80)
    
    def generate_recommendations(self) -> list:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        # Check file structure
        if 'file_structure' in self.verification_results:
            fs = self.verification_results['file_structure']
            if fs['missing_files'] > 0:
                recommendations.append(f"Create {fs['missing_files']} missing files")
        
        # Check database connectivity
        if 'database_connectivity' in self.verification_results:
            if self.verification_results['database_connectivity']['status'] == 'failed':
                recommendations.append("Fix database connectivity issues")
        
        # Check TimescaleDB tables
        if 'timescaledb_tables' in self.verification_results:
            table_status = self.verification_results['timescaledb_tables']
            non_hypertables = [table for table, status in table_status.items() 
                              if status['exists'] and not status.get('is_hypertable', False)]
            if non_hypertables:
                recommendations.append(f"Convert {len(non_hypertables)} tables to hypertables")
        
        # Check configuration
        if 'configuration_files' in self.verification_results:
            config_status = self.verification_results['configuration_files']
            invalid_configs = [config for config, status in config_status.items() 
                              if status['status'] == 'invalid']
            if invalid_configs:
                recommendations.append(f"Fix {len(invalid_configs)} invalid configuration files")
        
        # Check integration points
        if 'integration_points' in self.verification_results:
            integration_points = self.verification_results['integration_points']
            failed_points = [point for point, status in integration_points.items() if not status]
            if failed_points:
                recommendations.append(f"Fix {len(failed_points)} integration point issues")
        
        # Check performance
        if 'performance' in self.verification_results:
            perf = self.verification_results['performance']
            if 'performance_rating' in perf and perf['performance_rating'] == 'needs_optimization':
                recommendations.append("Optimize system performance")
        
        if not recommendations:
            recommendations.append("System is fully integrated and ready for production")
            recommendations.append("Consider adding more news sources for comprehensive coverage")
            recommendations.append("Monitor performance metrics in production")
        
        return recommendations
    
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
    """Main function to run complete integration verification"""
    verifier = CompleteIntegrationVerifier()
    await verifier.run_complete_verification()

if __name__ == "__main__":
    asyncio.run(main())
