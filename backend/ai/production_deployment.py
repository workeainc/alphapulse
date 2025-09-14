"""
Production Deployment for AlphaPulse
Phase 5: Production deployment with health checks and monitoring
"""

import asyncio
import logging
import time
import json
import os
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import subprocess
import socket

logger = logging.getLogger(__name__)

@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str = 'production'
    port: int = 8000
    host: str = '0.0.0.0'
    workers: int = 4
    max_connections: int = 1000
    timeout_seconds: int = 30
    health_check_interval: int = 30
    graceful_shutdown_timeout: int = 60

class HealthChecker:
    """System health checker"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.is_running = False
        self.health_history = deque(maxlen=100)
        self.health_callbacks = []
        
        # Health check endpoints
        self.health_endpoints = {
            'api': '/health',
            'database': '/health/db',
            'cache': '/health/cache',
            'streaming': '/health/streaming',
            'monitoring': '/health/monitoring'
        }
        
        logger.info("Health Checker initialized")
    
    async def start(self):
        """Start health checking"""
        self.is_running = True
        asyncio.create_task(self._health_check_loop())
        logger.info("🚀 Health Checker started")
    
    async def stop(self):
        """Stop health checking"""
        self.is_running = False
        logger.info("🛑 Health Checker stopped")
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.is_running:
            try:
                health_results = await self._perform_health_checks()
                self.health_history.append(health_results)
                
                # Check overall system health
                overall_status = self._evaluate_overall_health(health_results)
                
                # Notify callbacks
                for callback in self.health_callbacks:
                    try:
                        await callback(overall_status, health_results)
                    except Exception as e:
                        logger.error(f"Error in health callback: {e}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self) -> Dict[str, HealthCheck]:
        """Perform all health checks"""
        results = {}
        
        # API health check
        results['api'] = await self._check_api_health()
        
        # Database health check
        results['database'] = await self._check_database_health()
        
        # Cache health check
        results['cache'] = await self._check_cache_health()
        
        # Streaming health check
        results['streaming'] = await self._check_streaming_health()
        
        # Monitoring health check
        results['monitoring'] = await self._check_monitoring_health()
        
        return results
    
    async def _check_api_health(self) -> HealthCheck:
        """Check API health"""
        start_time = time.time()
        
        try:
            # Simulate API health check
            await asyncio.sleep(0.1)  # Simulate network call
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='api',
                status='healthy',
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='api',
                status='unhealthy',
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    async def _check_database_health(self) -> HealthCheck:
        """Check database health"""
        start_time = time.time()
        
        try:
            # Simulate database health check
            await asyncio.sleep(0.05)  # Simulate DB query
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='database',
                status='healthy',
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='database',
                status='unhealthy',
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    async def _check_cache_health(self) -> HealthCheck:
        """Check cache health"""
        start_time = time.time()
        
        try:
            # Simulate cache health check
            await asyncio.sleep(0.02)  # Simulate cache access
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='cache',
                status='healthy',
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='cache',
                status='unhealthy',
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    async def _check_streaming_health(self) -> HealthCheck:
        """Check streaming health"""
        start_time = time.time()
        
        try:
            # Simulate streaming health check
            await asyncio.sleep(0.03)  # Simulate streaming check
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='streaming',
                status='healthy',
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='streaming',
                status='unhealthy',
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    async def _check_monitoring_health(self) -> HealthCheck:
        """Check monitoring health"""
        start_time = time.time()
        
        try:
            # Simulate monitoring health check
            await asyncio.sleep(0.01)  # Simulate monitoring check
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='monitoring',
                status='healthy',
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component='monitoring',
                status='unhealthy',
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def _evaluate_overall_health(self, health_results: Dict[str, HealthCheck]) -> str:
        """Evaluate overall system health"""
        unhealthy_count = sum(1 for check in health_results.values() 
                             if check.status == 'unhealthy')
        degraded_count = sum(1 for check in health_results.values() 
                            if check.status == 'degraded')
        
        if unhealthy_count > 0:
            return 'unhealthy'
        elif degraded_count > 0:
            return 'degraded'
        else:
            return 'healthy'
    
    def add_health_callback(self, callback: Callable[[str, Dict[str, HealthCheck]], None]):
        """Add health check callback"""
        self.health_callbacks.append(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary"""
        if not self.health_history:
            return {'status': 'unknown', 'last_check': None}
        
        latest_checks = self.health_history[-1]
        overall_status = self._evaluate_overall_health(latest_checks)
        
        return {
            'overall_status': overall_status,
            'component_status': {name: check.status for name, check in latest_checks.items()},
            'last_check_time': latest_checks[list(latest_checks.keys())[0]].timestamp.isoformat(),
            'total_checks': len(self.health_history)
        }

class GracefulShutdown:
    """Graceful shutdown handler"""
    
    def __init__(self, shutdown_timeout: int = 60):
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_callbacks = []
        self.is_shutting_down = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Graceful Shutdown handler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Perform graceful shutdown"""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info("🔄 Starting graceful shutdown...")
        
        try:
            # Execute shutdown callbacks
            for callback in self.shutdown_callbacks:
                try:
                    await asyncio.wait_for(callback(), timeout=self.shutdown_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Shutdown callback timed out")
                except Exception as e:
                    logger.error(f"Error in shutdown callback: {e}")
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    def add_shutdown_callback(self, callback: Callable[[], None]):
        """Add shutdown callback"""
        self.shutdown_callbacks.append(callback)

class ProductionDeployment:
    """Production deployment manager"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.health_checker = HealthChecker(config.health_check_interval)
        self.graceful_shutdown = GracefulShutdown(config.graceful_shutdown_timeout)
        
        self.is_running = False
        self.start_time = None
        
        logger.info("Production Deployment initialized")
    
    async def start(self):
        """Start production deployment"""
        try:
            # Start health checker
            await self.health_checker.start()
            
            # Register shutdown callbacks
            self.graceful_shutdown.add_shutdown_callback(self.stop)
            
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("🚀 Production deployment started")
            logger.info(f"Environment: {self.config.environment}")
            logger.info(f"Host: {self.config.host}:{self.config.port}")
            logger.info(f"Workers: {self.config.workers}")
            
        except Exception as e:
            logger.error(f"Error starting production deployment: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop production deployment"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping production deployment...")
        
        self.is_running = False
        
        # Stop health checker
        await self.health_checker.stop()
        
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        logger.info(f"Production deployment stopped. Uptime: {uptime}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'deployment_running': self.is_running,
            'environment': self.config.environment,
            'host': self.config.host,
            'port': self.config.port,
            'workers': self.config.workers,
            'uptime_seconds': uptime.total_seconds(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'health_status': self.health_checker.get_health_summary()
        }
    
    def get_health_endpoint(self) -> Dict[str, Any]:
        """Get health endpoint response"""
        health_summary = self.health_checker.get_health_summary()
        
        return {
            'status': health_summary['overall_status'],
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'environment': self.config.environment,
            'components': health_summary.get('component_status', {}),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }

# Default production configuration
default_config = DeploymentConfig(
    environment='production',
    port=8000,
    host='0.0.0.0',
    workers=4,
    max_connections=1000,
    timeout_seconds=30,
    health_check_interval=30,
    graceful_shutdown_timeout=60
)

# Global production deployment instance
production_deployment = ProductionDeployment(default_config)
