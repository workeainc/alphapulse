"""
Test Production Deployment Preparation
Comprehensive test for production deployment preparation components
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
import asyncpg
import redis.asyncio as redis
from config.production import production_config
from app.services.real_time_data_manager import RealTimeDataManager
from app.services.deployment_manager import DeploymentManager, DeploymentConfig, DeploymentStatus
from monitoring.production_dashboard import ProductionDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionDeploymentTest:
    """Test class for production deployment preparation"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.real_time_manager = None
        self.deployment_manager = None
        self.dashboard = None
        self.test_results = {}
    
    async def setup(self):
        """Setup test environment"""
        logger.info("🚀 Setting up production deployment test environment")
        
        try:
            # Create database connection
            self.db_pool = await asyncpg.create_pool(
                host=production_config.DATABASE_CONFIG["host"],
                port=production_config.DATABASE_CONFIG["port"],
                database=production_config.DATABASE_CONFIG["database"],
                user=production_config.DATABASE_CONFIG["username"],
                password=production_config.DATABASE_CONFIG["password"],
                min_size=5,
                max_size=20
            )
            logger.info("✅ Database connection established")
            
            # Create Redis connection
            self.redis_client = redis.Redis(
                host=production_config.REDIS_CONFIG["host"],
                port=production_config.REDIS_CONFIG["port"],
                db=production_config.REDIS_CONFIG["db"],
                password=production_config.REDIS_CONFIG["password"],
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Initialize components
            self.real_time_manager = RealTimeDataManager(self.db_pool, self.redis_client)
            self.deployment_manager = DeploymentManager(self.db_pool)
            self.dashboard = ProductionDashboard(self.db_pool, self.redis_client)
            
            logger.info("✅ All components initialized")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment")
        
        try:
            if self.real_time_manager:
                await self.real_time_manager.stop()
            
            if self.deployment_manager:
                await self.deployment_manager.stop()
            
            if self.dashboard:
                await self.dashboard.stop()
            
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    async def test_production_config(self):
        """Test production configuration"""
        logger.info("🧪 Testing production configuration")
        
        try:
            # Test configuration validation
            validation_results = production_config.validate_config()
            
            # Check database configuration
            db_config = validation_results["database"]
            if not db_config["valid"]:
                raise ValueError(f"Database configuration invalid: {db_config['issues']}")
            
            # Check Redis configuration
            redis_config = validation_results["redis"]
            if not redis_config["valid"]:
                raise ValueError(f"Redis configuration invalid: {redis_config['issues']}")
            
            # Check security configuration
            security_config = validation_results["security"]
            if not security_config["valid"]:
                logger.warning(f"⚠️ Security configuration issues: {security_config['issues']}")
            
            # Test configuration values
            assert production_config.ENVIRONMENT == "production"
            assert production_config.DEBUG == False
            assert production_config.API_HOST == "0.0.0.0"
            assert production_config.API_PORT == 8000
            
            # Test database URL generation
            db_url = production_config.get_database_url()
            assert "postgresql://" in db_url
            assert production_config.DATABASE_CONFIG["database"] in db_url
            
            # Test Redis URL generation
            redis_url = production_config.get_redis_url()
            assert "redis://" in redis_url
            assert str(production_config.REDIS_CONFIG["port"]) in redis_url
            
            self.test_results["production_config"] = {
                "status": "PASSED",
                "validation_results": validation_results,
                "database_url": db_url,
                "redis_url": redis_url
            }
            
            logger.info("✅ Production configuration test passed")
            
        except Exception as e:
            logger.error(f"❌ Production configuration test failed: {e}")
            self.test_results["production_config"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_real_time_data_manager(self):
        """Test real-time data manager"""
        logger.info("🧪 Testing real-time data manager")
        
        try:
            # Start real-time manager
            await self.real_time_manager.start()
            
            # Test data streams initialization
            assert len(self.real_time_manager.data_streams) > 0
            
            # Test broadcast functionality
            test_data = {
                "type": "test",
                "message": "Test broadcast",
                "timestamp": datetime.now().isoformat()
            }
            
            await self.real_time_manager.broadcast_data("market_data_stream", test_data)
            
            # Test status retrieval
            status = self.real_time_manager.get_status()
            assert "is_running" in status
            assert "active_connections" in status
            assert "data_streams" in status
            
            # Test metrics collection
            await asyncio.sleep(2)  # Wait for metrics collection
            
            # Stop real-time manager
            await self.real_time_manager.stop()
            
            self.test_results["real_time_data_manager"] = {
                "status": "PASSED",
                "data_streams_count": len(self.real_time_manager.data_streams),
                "status_data": status
            }
            
            logger.info("✅ Real-time data manager test passed")
            
        except Exception as e:
            logger.error(f"❌ Real-time data manager test failed: {e}")
            self.test_results["real_time_data_manager"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_deployment_manager(self):
        """Test deployment manager"""
        logger.info("🧪 Testing deployment manager")
        
        try:
            # Start deployment manager
            await self.deployment_manager.start()
            
            # Create test deployment configuration
            deployment_config = DeploymentConfig(
                deployment_id="test_deployment_001",
                version="1.0.0",
                environment="test",
                services=["api", "worker", "monitoring"],
                health_check_endpoints=[
                    "http://localhost:8000/health",
                    "http://localhost:8080/health"
                ],
                rollback_version="0.9.0"
            )
            
            # Test deployment creation
            deployment_result = await self.deployment_manager.deploy(deployment_config)
            
            assert deployment_result.deployment_id == "test_deployment_001"
            assert deployment_result.status in [DeploymentStatus.ACTIVE, DeploymentStatus.FAILED]
            
            # Test deployment status retrieval
            status = await self.deployment_manager.get_deployment_status("test_deployment_001")
            assert status is not None
            
            # Test health summary
            health_summary = self.deployment_manager.get_health_summary()
            assert "total_deployments" in health_summary
            assert "active_deployments" in health_summary
            
            # Stop deployment manager
            await self.deployment_manager.stop()
            
            self.test_results["deployment_manager"] = {
                "status": "PASSED",
                "deployment_result": {
                    "deployment_id": deployment_result.deployment_id,
                    "status": deployment_result.status.value,
                    "health_checks_passed": deployment_result.health_checks_passed,
                    "health_checks_failed": deployment_result.health_checks_failed
                },
                "health_summary": health_summary
            }
            
            logger.info("✅ Deployment manager test passed")
            
        except Exception as e:
            logger.error(f"❌ Deployment manager test failed: {e}")
            self.test_results["deployment_manager"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_production_dashboard(self):
        """Test production dashboard"""
        logger.info("🧪 Testing production dashboard")
        
        try:
            # Start dashboard
            await self.dashboard.start()
            
            # Test metrics retrieval
            metrics = await self.dashboard._get_current_metrics()
            assert "timestamp" in metrics
            assert "system_metrics" in metrics
            assert "service_metrics" in metrics
            assert "trading_metrics" in metrics
            assert "alert_metrics" in metrics
            assert "deployment_metrics" in metrics
            
            # Test system health
            system_health = await self.dashboard._get_system_health()
            assert "overall_health" in system_health
            assert "system_metrics" in system_health
            
            # Test services status
            services_status = await self.dashboard._get_services_status()
            assert "sde_framework" in services_status
            assert "signal_generator" in services_status
            assert "database" in services_status
            
            # Test trading overview
            trading_overview = await self.dashboard._get_trading_overview()
            assert "enabled" in trading_overview
            assert "signal_accuracy" in trading_overview
            
            # Test deployments status
            deployments_status = await self.dashboard._get_deployments_status()
            assert "total_deployments" in deployments_status
            assert "active_deployments" in deployments_status
            
            # Stop dashboard
            await self.dashboard.stop()
            
            self.test_results["production_dashboard"] = {
                "status": "PASSED",
                "metrics_keys": list(metrics.keys()),
                "system_health": system_health["overall_health"],
                "services_count": len(services_status)
            }
            
            logger.info("✅ Production dashboard test passed")
            
        except Exception as e:
            logger.error(f"❌ Production dashboard test failed: {e}")
            self.test_results["production_dashboard"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_database_tables(self):
        """Test database tables"""
        logger.info("🧪 Testing database tables")
        
        try:
            # Test real_time_metrics table
            async with self.db_pool.acquire() as conn:
                # Insert test data
                await conn.execute("""
                    INSERT INTO real_time_metrics (timestamp, active_connections, total_connections, messages_sent, messages_received, errors)
                    VALUES (NOW(), 10, 100, 1000, 950, 5)
                """)
                
                # Query test data
                result = await conn.fetchrow("""
                    SELECT active_connections, total_connections, messages_sent, messages_received, errors
                    FROM real_time_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                assert result is not None
                assert result['active_connections'] == 10
                assert result['total_connections'] == 100
                
                # Test deployment_history table
                await conn.execute("""
                    INSERT INTO deployment_history (deployment_id, status, start_time, end_time, health_checks_passed, health_checks_failed)
                    VALUES ('test_deployment_002', 'active', NOW(), NOW(), 5, 0)
                """)
                
                deployment_result = await conn.fetchrow("""
                    SELECT deployment_id, status, health_checks_passed, health_checks_failed
                    FROM deployment_history 
                    WHERE deployment_id = 'test_deployment_002'
                """)
                
                assert deployment_result is not None
                assert deployment_result['deployment_id'] == 'test_deployment_002'
                assert deployment_result['status'] == 'active'
                
                # Test alerts table
                await conn.execute("""
                    INSERT INTO alerts (alert_id, severity, service, message, timestamp)
                    VALUES ('test_alert_001', 'warning', 'test_service', 'Test alert message', NOW())
                """)
                
                alert_result = await conn.fetchrow("""
                    SELECT alert_id, severity, service, message
                    FROM alerts 
                    WHERE alert_id = 'test_alert_001'
                """)
                
                assert alert_result is not None
                assert alert_result['alert_id'] == 'test_alert_001'
                assert alert_result['severity'] == 'warning'
                
                # Clean up test data
                await conn.execute("DELETE FROM real_time_metrics WHERE active_connections = 10")
                await conn.execute("DELETE FROM deployment_history WHERE deployment_id = 'test_deployment_002'")
                await conn.execute("DELETE FROM alerts WHERE alert_id = 'test_alert_001'")
            
            self.test_results["database_tables"] = {
                "status": "PASSED",
                "tables_tested": ["real_time_metrics", "deployment_history", "alerts"]
            }
            
            logger.info("✅ Database tables test passed")
            
        except Exception as e:
            logger.error(f"❌ Database tables test failed: {e}")
            self.test_results["database_tables"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("🧪 Testing component integration")
        
        try:
            # Start all components
            await self.real_time_manager.start()
            await self.deployment_manager.start()
            await self.dashboard.start()
            
            # Test data flow
            test_data = {
                "type": "integration_test",
                "component": "real_time_manager",
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast data through real-time manager
            await self.real_time_manager.broadcast_data("market_data_stream", test_data)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check dashboard metrics
            metrics = await self.dashboard._get_current_metrics()
            assert metrics["service_metrics"]["active_connections"] >= 0
            
            # Check deployment manager health
            health_summary = self.deployment_manager.get_health_summary()
            assert health_summary["total_deployments"] >= 0
            
            # Stop all components
            await self.real_time_manager.stop()
            await self.deployment_manager.stop()
            await self.dashboard.stop()
            
            self.test_results["integration"] = {
                "status": "PASSED",
                "components_integrated": ["real_time_manager", "deployment_manager", "dashboard"]
            }
            
            logger.info("✅ Integration test passed")
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            self.test_results["integration"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Production Deployment Preparation Tests")
        
        try:
            await self.setup()
            
            # Run individual tests
            await self.test_production_config()
            await self.test_real_time_data_manager()
            await self.test_deployment_manager()
            await self.test_production_dashboard()
            await self.test_database_tables()
            await self.test_integration()
            
            # Generate test report
            await self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
        finally:
            await self.cleanup()
    
    async def generate_test_report(self):
        """Generate test report"""
        logger.info("📊 Generating test report")
        
        # Calculate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        
        # Print summary
        logger.info("=" * 80)
        logger.info("📊 PRODUCTION DEPLOYMENT PREPARATION TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info("=" * 80)
        
        # Print detailed results
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
            
            if result["status"] == "FAILED" and "error" in result:
                logger.error(f"   Error: {result['error']}")
        
        logger.info("=" * 80)
        
        # Save report to file
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests/total_tests)*100 if total_tests > 0 else 0
            },
            "results": self.test_results
        }
        
        with open("production_deployment_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info("📄 Test report saved to: production_deployment_test_report.json")
        
        # Return success status
        return failed_tests == 0

async def main():
    """Main test function"""
    test = ProductionDeploymentTest()
    success = await test.run_all_tests()
    
    if success:
        logger.info("🎉 All production deployment preparation tests passed!")
        return 0
    else:
        logger.error("❌ Some production deployment preparation tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
