#!/usr/bin/env python3
"""
Production Deployment System Test
Comprehensive test suite for production deployment functionality
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
import asyncpg
import redis.asyncio as redis
import importlib.util

# Import production config using importlib.util to avoid module cache issues
spec = importlib.util.spec_from_file_location('production', 'config/production.py')
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)
production_config = production_module.production_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeploymentSystemTest:
    """Test suite for production deployment system"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.test_results = []
    
    async def setup(self):
        """Setup test environment"""
        try:
            # Create database connection pool
            self.db_pool = await asyncpg.create_pool(
                host=production_config.DATABASE_CONFIG['host'],
                port=production_config.DATABASE_CONFIG['port'],
                database=production_config.DATABASE_CONFIG['database'],
                user=production_config.DATABASE_CONFIG['username'],
                password=production_config.DATABASE_CONFIG['password'],
                min_size=2,
                max_size=10
            )
            logger.info("Database connection pool created")
            
            # Create Redis client
            self.redis_client = redis.Redis(
                host=production_config.REDIS_CONFIG['host'],
                port=production_config.REDIS_CONFIG['port'],
                db=production_config.REDIS_CONFIG['db'],
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    async def test_database_tables(self):
        """Test production deployment database tables"""
        try:
            logger.info("Testing Production Deployment Database Tables...")
            
            # Test production deployment tables
            async with self.db_pool.acquire() as conn:
                # Check deployment_metrics table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'deployment_metrics')"
                )
                assert result is True, "deployment_metrics table not found"
                
                # Check service_health table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'service_health')"
                )
                assert result is True, "service_health table not found"
                
                # Check deployment_configs table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'deployment_configs')"
                )
                assert result is True, "deployment_configs table not found"
                
                # Check deployment_events table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'deployment_events')"
                )
                assert result is True, "deployment_events table not found"
                
                # Check system_health_metrics table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'system_health_metrics')"
                )
                assert result is True, "system_health_metrics table not found"
                
                # Check deployment_alerts table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'deployment_alerts')"
                )
                assert result is True, "deployment_alerts table not found"
                
                # Check performance_metrics table
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'performance_metrics')"
                )
                assert result is True, "performance_metrics table not found"
            
            logger.info("Production Deployment Database Tables test passed")
            self.test_results.append({"test": "database_tables", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"Production Deployment Database Tables test failed: {e}")
            self.test_results.append({"test": "database_tables", "status": "failed", "error": str(e)})
            return False
    
    async def test_deployment_views(self):
        """Test production deployment database views"""
        try:
            logger.info("Testing Production Deployment Database Views...")
            
            # Test deployment views
            async with self.db_pool.acquire() as conn:
                # Check deployment_summary view
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.views WHERE table_name = 'deployment_summary')"
                )
                assert result is True, "deployment_summary view not found"
                
                # Check service_health_summary view
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.views WHERE table_name = 'service_health_summary')"
                )
                assert result is True, "service_health_summary view not found"
                
                # Check active_alerts view
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.views WHERE table_name = 'active_alerts')"
                )
                assert result is True, "active_alerts view not found"
                
                # Test view queries
                deployment_summary = await conn.fetch("SELECT * FROM deployment_summary LIMIT 1")
                assert deployment_summary is not None, "deployment_summary view query failed"
                
                service_health_summary = await conn.fetch("SELECT * FROM service_health_summary LIMIT 1")
                assert service_health_summary is not None, "service_health_summary view query failed"
                
                active_alerts = await conn.fetch("SELECT * FROM active_alerts LIMIT 1")
                assert active_alerts is not None, "active_alerts view query failed"
            
            logger.info("Production Deployment Database Views test passed")
            self.test_results.append({"test": "database_views", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"Production Deployment Database Views test failed: {e}")
            self.test_results.append({"test": "database_views", "status": "failed", "error": str(e)})
            return False
    
    async def test_deployment_metrics_storage(self):
        """Test deployment metrics storage functionality"""
        try:
            logger.info("Testing Deployment Metrics Storage...")
            
            # Test storing deployment metrics
            async with self.db_pool.acquire() as conn:
                # Insert test deployment metrics
                test_metrics = {
                    "deployment_id": "test-deployment-001",
                    "start_time": datetime.now(),
                    "end_time": datetime.now() + timedelta(minutes=5),
                    "total_services": 3,
                    "deployed_services": 3,
                    "failed_services": 0,
                    "health_checks_passed": 3,
                    "health_checks_failed": 0,
                    "rollback_triggered": False,
                    "deployment_duration": 300.0,
                    "error_message": None
                }
                
                await conn.execute("""
                    INSERT INTO deployment_metrics (
                        deployment_id, start_time, end_time, total_services,
                        deployed_services, failed_services, health_checks_passed,
                        health_checks_failed, rollback_triggered, deployment_duration,
                        error_message
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                test_metrics["deployment_id"],
                test_metrics["start_time"],
                test_metrics["end_time"],
                test_metrics["total_services"],
                test_metrics["deployed_services"],
                test_metrics["failed_services"],
                test_metrics["health_checks_passed"],
                test_metrics["health_checks_failed"],
                test_metrics["rollback_triggered"],
                test_metrics["deployment_duration"],
                test_metrics["error_message"]
                )
                
                # Verify insertion
                stored_metrics = await conn.fetchrow(
                    "SELECT * FROM deployment_metrics WHERE deployment_id = $1",
                    test_metrics["deployment_id"]
                )
                assert stored_metrics is not None, "Deployment metrics not stored"
                assert stored_metrics["deployment_id"] == test_metrics["deployment_id"]
                assert stored_metrics["total_services"] == test_metrics["total_services"]
                
                # Clean up test data
                await conn.execute(
                    "DELETE FROM deployment_metrics WHERE deployment_id = $1",
                    test_metrics["deployment_id"]
                )
            
            logger.info("Deployment Metrics Storage test passed")
            self.test_results.append({"test": "deployment_metrics_storage", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"Deployment Metrics Storage test failed: {e}")
            self.test_results.append({"test": "deployment_metrics_storage", "status": "failed", "error": str(e)})
            return False
    
    async def test_service_health_monitoring(self):
        """Test service health monitoring functionality"""
        try:
            logger.info("Testing Service Health Monitoring...")
            
            # Test storing service health data
            async with self.db_pool.acquire() as conn:
                # Insert test service health data
                test_health = {
                    "service_name": "test-service-001",
                    "status": "healthy",
                    "response_time_ms": 150.0,
                    "status_code": 200,
                    "last_check": datetime.now(),
                    "error_count": 0,
                    "consecutive_failures": 0,
                    "deployment_id": "test-deployment-001"
                }
                
                await conn.execute("""
                    INSERT INTO service_health (
                        service_name, status, response_time_ms, status_code,
                        last_check, error_count, consecutive_failures, deployment_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                test_health["service_name"],
                test_health["status"],
                test_health["response_time_ms"],
                test_health["status_code"],
                test_health["last_check"],
                test_health["error_count"],
                test_health["consecutive_failures"],
                test_health["deployment_id"]
                )
                
                # Verify insertion
                stored_health = await conn.fetchrow(
                    "SELECT * FROM service_health WHERE service_name = $1",
                    test_health["service_name"]
                )
                assert stored_health is not None, "Service health data not stored"
                assert stored_health["service_name"] == test_health["service_name"]
                assert stored_health["status"] == test_health["status"]
                
                # Test service health summary view
                health_summary = await conn.fetch(
                    "SELECT * FROM service_health_summary WHERE service_name = $1",
                    test_health["service_name"]
                )
                assert len(health_summary) > 0, "Service health summary view not working"
                
                # Clean up test data
                await conn.execute(
                    "DELETE FROM service_health WHERE service_name = $1",
                    test_health["service_name"]
                )
            
            logger.info("Service Health Monitoring test passed")
            self.test_results.append({"test": "service_health_monitoring", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"Service Health Monitoring test failed: {e}")
            self.test_results.append({"test": "service_health_monitoring", "status": "failed", "error": str(e)})
            return False
    
    async def test_deployment_alerts(self):
        """Test deployment alerts functionality"""
        try:
            logger.info("Testing Deployment Alerts...")
            
            # Test storing deployment alerts
            async with self.db_pool.acquire() as conn:
                # Insert test alert
                test_alert = {
                    "alert_type": "deployment_failure",
                    "alert_message": "Test deployment failed",
                    "deployment_id": "test-deployment-001",
                    "severity": "error",
                    "status": "active"
                }
                
                await conn.execute("""
                    INSERT INTO deployment_alerts (
                        alert_type, alert_message, deployment_id, severity, status
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                test_alert["alert_type"],
                test_alert["alert_message"],
                test_alert["deployment_id"],
                test_alert["severity"],
                test_alert["status"]
                )
                
                # Verify insertion
                stored_alert = await conn.fetchrow(
                    "SELECT * FROM deployment_alerts WHERE deployment_id = $1",
                    test_alert["deployment_id"]
                )
                assert stored_alert is not None, "Deployment alert not stored"
                assert stored_alert["alert_type"] == test_alert["alert_type"]
                assert stored_alert["severity"] == test_alert["severity"]
                
                # Test active alerts view
                active_alerts = await conn.fetch(
                    "SELECT * FROM active_alerts WHERE deployment_id = $1",
                    test_alert["deployment_id"]
                )
                assert len(active_alerts) > 0, "Active alerts view not working"
                
                # Clean up test data
                await conn.execute(
                    "DELETE FROM deployment_alerts WHERE deployment_id = $1",
                    test_alert["deployment_id"]
                )
            
            logger.info("Deployment Alerts test passed")
            self.test_results.append({"test": "deployment_alerts", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"Deployment Alerts test failed: {e}")
            self.test_results.append({"test": "deployment_alerts", "status": "failed", "error": str(e)})
            return False
    
    async def test_system_health_metrics(self):
        """Test system health metrics functionality"""
        try:
            logger.info("Testing System Health Metrics...")
            
            # Test storing system health metrics
            async with self.db_pool.acquire() as conn:
                # Insert test system metrics
                test_metrics = {
                    "cpu_percent": 45.2,
                    "memory_percent": 62.1,
                    "disk_percent": 75.8,
                    "network_io_in": 1024.0,
                    "network_io_out": 2048.0,
                    "active_connections": 10,
                    "total_requests": 1000,
                    "error_rate": 0.01,
                    "response_time_avg": 150.0
                }
                
                await conn.execute("""
                    INSERT INTO system_health_metrics (
                        cpu_percent, memory_percent, disk_percent, network_io_in,
                        network_io_out, active_connections, total_requests,
                        error_rate, response_time_avg
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                test_metrics["cpu_percent"],
                test_metrics["memory_percent"],
                test_metrics["disk_percent"],
                test_metrics["network_io_in"],
                test_metrics["network_io_out"],
                test_metrics["active_connections"],
                test_metrics["total_requests"],
                test_metrics["error_rate"],
                test_metrics["response_time_avg"]
                )
                
                # Verify insertion
                stored_metrics = await conn.fetchrow(
                    "SELECT * FROM system_health_metrics ORDER BY timestamp DESC LIMIT 1"
                )
                assert stored_metrics is not None, "System health metrics not stored"
                assert stored_metrics["cpu_percent"] == test_metrics["cpu_percent"]
                assert stored_metrics["memory_percent"] == test_metrics["memory_percent"]
                
                # Clean up test data (keep recent data for monitoring)
                await conn.execute(
                    "DELETE FROM system_health_metrics WHERE cpu_percent = $1 AND memory_percent = $2",
                    test_metrics["cpu_percent"], test_metrics["memory_percent"]
                )
            
            logger.info("System Health Metrics test passed")
            self.test_results.append({"test": "system_health_metrics", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"System Health Metrics test failed: {e}")
            self.test_results.append({"test": "system_health_metrics", "status": "failed", "error": str(e)})
            return False
    
    async def test_performance_metrics(self):
        """Test performance metrics functionality"""
        try:
            logger.info("Testing Performance Metrics...")
            
            # Test storing performance metrics
            async with self.db_pool.acquire() as conn:
                # Insert test performance metrics
                test_metrics = {
                    "deployment_id": "test-deployment-001",
                    "service_name": "test-service-001",
                    "metric_name": "response_time",
                    "metric_value": 150.0,
                    "metric_unit": "ms"
                }
                
                await conn.execute("""
                    INSERT INTO performance_metrics (
                        deployment_id, service_name, metric_name, metric_value, metric_unit
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                test_metrics["deployment_id"],
                test_metrics["service_name"],
                test_metrics["metric_name"],
                test_metrics["metric_value"],
                test_metrics["metric_unit"]
                )
                
                # Verify insertion
                stored_metrics = await conn.fetchrow(
                    "SELECT * FROM performance_metrics WHERE deployment_id = $1 AND service_name = $2",
                    test_metrics["deployment_id"], test_metrics["service_name"]
                )
                assert stored_metrics is not None, "Performance metrics not stored"
                assert stored_metrics["metric_name"] == test_metrics["metric_name"]
                assert stored_metrics["metric_value"] == test_metrics["metric_value"]
                
                # Clean up test data
                await conn.execute(
                    "DELETE FROM performance_metrics WHERE deployment_id = $1",
                    test_metrics["deployment_id"]
                )
            
            logger.info("Performance Metrics test passed")
            self.test_results.append({"test": "performance_metrics", "status": "passed"})
            return True
            
        except Exception as e:
            logger.error(f"Performance Metrics test failed: {e}")
            self.test_results.append({"test": "performance_metrics", "status": "failed", "error": str(e)})
            return False
    
    async def cleanup(self):
        """Cleanup test environment"""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Test environment cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def print_results(self):
        """Print test results"""
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION DEPLOYMENT SYSTEM TEST RESULTS")
        logger.info("="*60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "passed")
        failed = sum(1 for result in self.test_results if result["status"] == "failed")
        total = len(self.test_results)
        
        for result in self.test_results:
            status_icon = "PASS" if result["status"] == "passed" else "FAIL"
            logger.info(f"{status_icon} {result['test']}: {result['status'].upper()}")
            if result["status"] == "failed" and "error" in result:
                logger.error(f"   Error: {result['error']}")
        
        logger.info(f"\nSummary: {passed}/{total} tests passed")
        
        if failed == 0:
            logger.info("All tests passed! Production deployment system is ready.")
        else:
            logger.error(f"{failed} tests failed. Please fix the issues before proceeding.")
        
        logger.info("="*60)

async def main():
    """Main test function"""
    test = ProductionDeploymentSystemTest()
    
    try:
        # Setup
        if not await test.setup():
            logger.error("Test setup failed")
            return
        
        # Run tests
        await test.test_database_tables()
        await test.test_deployment_views()
        await test.test_deployment_metrics_storage()
        await test.test_service_health_monitoring()
        await test.test_deployment_alerts()
        await test.test_system_health_metrics()
        await test.test_performance_metrics()
        
        # Print results
        test.print_results()
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    finally:
        await test.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
