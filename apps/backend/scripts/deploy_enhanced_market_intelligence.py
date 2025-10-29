#!/usr/bin/env python3
"""
Deploy Enhanced Market Intelligence System
Comprehensive deployment script for enhanced market intelligence with inflow/outflow analysis
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from src.database.migrations.004_enhanced_market_intelligence_tables import create_enhanced_market_intelligence_tables
    from tests.test_enhanced_market_intelligence import EnhancedMarketIntelligenceTester
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure you're running from the backend directory")
    sys.exit(1)

class EnhancedMarketIntelligenceDeployer:
    """Deployment manager for enhanced market intelligence system"""
    
    def __init__(self):
        self.deployment_start = datetime.now()
        self.deployment_report = {
            'start_time': self.deployment_start,
            'steps': [],
            'status': 'running',
            'errors': [],
            'warnings': []
        }
    
    async def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        try:
            logger.info("🔍 Checking system prerequisites...")
            
            # Check Python version
            if sys.version_info < (3, 8):
                logger.error("❌ Python 3.8+ required")
                return False
            
            # Check required packages
            required_packages = ['asyncpg', 'aiohttp', 'ccxt', 'numpy']
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.error(f"❌ Missing required packages: {missing_packages}")
                return False
            
            logger.info("✅ Prerequisites check passed")
            self.deployment_report['steps'].append({
                'step': 'prerequisites_check',
                'status': 'passed',
                'timestamp': datetime.now()
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ Prerequisites check failed: {e}")
            self.deployment_report['errors'].append(f"Prerequisites check failed: {e}")
            return False
    
    async def run_database_migration(self) -> bool:
        """Run database migration"""
        try:
            logger.info("🗄️ Running database migration...")
            
            success = await create_enhanced_market_intelligence_tables()
            
            if success:
                logger.info("✅ Database migration completed successfully")
                self.deployment_report['steps'].append({
                    'step': 'database_migration',
                    'status': 'passed',
                    'timestamp': datetime.now()
                })
                return True
            else:
                logger.error("❌ Database migration failed")
                self.deployment_report['errors'].append("Database migration failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database migration error: {e}")
            self.deployment_report['errors'].append(f"Database migration error: {e}")
            return False
    
    async def run_system_tests(self) -> bool:
        """Run comprehensive system tests"""
        try:
            logger.info("🧪 Running system tests...")
            
            tester = EnhancedMarketIntelligenceTester()
            success = await tester.run_comprehensive_test()
            
            if success:
                logger.info("✅ System tests passed")
                self.deployment_report['steps'].append({
                    'step': 'system_tests',
                    'status': 'passed',
                    'timestamp': datetime.now()
                })
                return True
            else:
                logger.error("❌ System tests failed")
                self.deployment_report['errors'].append("System tests failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ System tests error: {e}")
            self.deployment_report['errors'].append(f"System tests error: {e}")
            return False
    
    async def validate_deployment(self) -> bool:
        """Validate deployment"""
        try:
            logger.info("✅ Validating deployment...")
            
            # Check if tables exist
            import asyncpg
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'alphapulse',
                'user': 'alpha_emon',
                'password': 'Emon_@17711'
            }
            
            db_pool = await asyncpg.create_pool(**db_config)
            
            required_tables = [
                'enhanced_market_intelligence',
                'inflow_outflow_analysis',
                'whale_movement_tracking',
                'correlation_analysis',
                'predictive_market_regime',
                'market_anomaly_detection'
            ]
            
            async with db_pool.acquire() as conn:
                for table in required_tables:
                    exists = await conn.fetchval(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = '{table}'
                        );
                    """)
                    
                    if not exists:
                        logger.error(f"❌ Table {table} not found")
                        return False
            
            await db_pool.close()
            logger.info("✅ Deployment validation passed")
            
            self.deployment_report['steps'].append({
                'step': 'deployment_validation',
                'status': 'passed',
                'timestamp': datetime.now()
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment validation failed: {e}")
            self.deployment_report['errors'].append(f"Deployment validation failed: {e}")
            return False
    
    def generate_deployment_report(self):
        """Generate deployment report"""
        try:
            deployment_end = datetime.now()
            duration = deployment_end - self.deployment_start
            
            self.deployment_report.update({
                'end_time': deployment_end,
                'duration_seconds': duration.total_seconds(),
                'status': 'completed' if not self.deployment_report['errors'] else 'failed'
            })
            
            # Save report
            report_filename = f"enhanced_market_intelligence_deployment_report_{deployment_end.strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(report_filename, 'w') as f:
                json.dump(self.deployment_report, f, default=str, indent=2)
            
            logger.info(f"📊 Deployment report saved: {report_filename}")
            
            # Print summary
            logger.info("=" * 60)
            logger.info("📋 DEPLOYMENT SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Status: {self.deployment_report['status'].upper()}")
            logger.info(f"Duration: {duration.total_seconds():.2f} seconds")
            logger.info(f"Steps completed: {len(self.deployment_report['steps'])}")
            logger.info(f"Errors: {len(self.deployment_report['errors'])}")
            logger.info(f"Warnings: {len(self.deployment_report['warnings'])}")
            
            if self.deployment_report['errors']:
                logger.info("\n❌ ERRORS:")
                for error in self.deployment_report['errors']:
                    logger.info(f"  - {error}")
            
            if self.deployment_report['warnings']:
                logger.info("\n⚠️ WARNINGS:")
                for warning in self.deployment_report['warnings']:
                    logger.info(f"  - {warning}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error generating deployment report: {e}")
    
    async def deploy(self) -> bool:
        """Main deployment process"""
        try:
            logger.info("🚀 Starting Enhanced Market Intelligence System Deployment")
            logger.info("=" * 60)
            
            # Step 1: Check prerequisites
            if not await self.check_prerequisites():
                return False
            
            # Step 2: Run database migration
            if not await self.run_database_migration():
                return False
            
            # Step 3: Run system tests
            if not await self.run_system_tests():
                return False
            
            # Step 4: Validate deployment
            if not await self.validate_deployment():
                return False
            
            # Generate report
            self.generate_deployment_report()
            
            if self.deployment_report['status'] == 'completed':
                logger.info("🎉 Enhanced Market Intelligence System Deployment COMPLETED SUCCESSFULLY!")
                return True
            else:
                logger.error("💥 Enhanced Market Intelligence System Deployment FAILED!")
                return False
                
        except Exception as e:
            logger.error(f"💥 Deployment crashed: {e}")
            self.deployment_report['errors'].append(f"Deployment crashed: {e}")
            self.generate_deployment_report()
            return False

async def main():
    """Main function"""
    try:
        deployer = EnhancedMarketIntelligenceDeployer()
        success = await deployer.deploy()
        
        if success:
            logger.info("🎉 Deployment completed successfully!")
            return 0
        else:
            logger.error("💥 Deployment failed!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Deployment script crashed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
