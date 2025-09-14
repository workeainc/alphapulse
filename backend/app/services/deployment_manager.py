"""
Production Deployment Manager for AlphaPulse
Handles deployment, health checks, rollbacks, and deployment status tracking
"""

import asyncio
import logging
import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncpg
import aiohttp
import importlib.util
spec = importlib.util.spec_from_file_location('production', 'config/production.py')
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)
production_config = production_module.production_config

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    version: str
    environment: str
    services: List[str]
    health_check_endpoints: List[str]
    rollback_version: Optional[str] = None
    deployment_timeout: int = 300  # 5 minutes
    health_check_timeout: int = 30  # 30 seconds
    max_retries: int = 3

@dataclass
class HealthCheckResult:
    """Health check result"""
    endpoint: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    status_code: int
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DeploymentResult:
    """Deployment result"""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    error_message: Optional[str] = None
    rollback_triggered: bool = False

class DeploymentManager:
    """
    Production deployment manager for handling deployments, health checks, and rollbacks
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.is_running = False
        
        # Active deployments
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # Health check cache
        self.health_check_cache: Dict[str, HealthCheckResult] = {}
        self.health_check_interval = 30  # seconds
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Deployment configuration
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        
        logger.info("Production Deployment Manager initialized")
    
    async def start(self):
        """Start the deployment manager"""
        if self.is_running:
            logger.warning("⚠️ Deployment manager already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting deployment manager...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._deployment_monitoring_loop()),
            asyncio.create_task(self._cleanup_old_deployments())
        ]
        
        logger.info("✅ Deployment manager started")
    
    async def stop(self):
        """Stop the deployment manager"""
        if not self.is_running:
            logger.warning("⚠️ Deployment manager not running")
            return
        
        self.is_running = False
        logger.info("🛑 Stopping deployment manager...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("✅ Deployment manager stopped")
    
    async def deploy(self, deployment_config: DeploymentConfig) -> DeploymentResult:
        """Deploy a new version"""
        deployment_id = deployment_config.deployment_id
        
        # Check if deployment already exists
        if deployment_id in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} already exists")
        
        # Create deployment result
        deployment_result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now()
        )
        
        self.active_deployments[deployment_id] = deployment_result
        self.deployment_configs[deployment_id] = deployment_config
        
        logger.info(f"Starting deployment {deployment_id} (version {deployment_config.version})")
        
        try:
            # Update status to deploying
            deployment_result.status = DeploymentStatus.DEPLOYING
            
            # Run deployment steps
            await self._run_deployment_steps(deployment_config, deployment_result)
            
            # Perform health checks
            await self._perform_health_checks(deployment_config, deployment_result)
            
            # Check if deployment was successful
            if deployment_result.health_checks_failed == 0:
                deployment_result.status = DeploymentStatus.ACTIVE
                logger.info(f"✅ Deployment {deployment_id} successful")
            else:
                deployment_result.status = DeploymentStatus.FAILED
                deployment_result.error_message = f"Health checks failed: {deployment_result.health_checks_failed}/{len(deployment_config.health_check_endpoints)}"
                logger.error(f"Deployment {deployment_id} failed")
                
                # Trigger rollback if configured
                if deployment_config.rollback_version:
                    await self._trigger_rollback(deployment_config, deployment_result)
            
        except Exception as e:
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.error_message = str(e)
            logger.error(f"❌ Deployment {deployment_id} failed: {e}")
            
            # Trigger rollback if configured
            if deployment_config.rollback_version:
                await self._trigger_rollback(deployment_config, deployment_result)
        
        finally:
            # Finalize deployment
            deployment_result.end_time = datetime.now()
            await self._store_deployment_result(deployment_result)
            
            # Add to history
            self.deployment_history.append(deployment_result)
            
            # Remove from active deployments if not active
            if deployment_result.status != DeploymentStatus.ACTIVE:
                del self.active_deployments[deployment_id]
        
        return deployment_result
    
    async def _run_deployment_steps(self, config: DeploymentConfig, result: DeploymentResult):
        """Run deployment steps"""
        logger.info(f"Running deployment steps for {config.deployment_id}")
        
        # Step 1: Backup current version
        await self._backup_current_version(config)
        
        # Step 2: Deploy new version
        await self._deploy_new_version(config)
        
        # Step 3: Update configuration
        await self._update_configuration(config)
        
        logger.info(f"Deployment steps completed for {config.deployment_id}")
    
    async def _backup_current_version(self, config: DeploymentConfig):
        """Backup current version"""
        try:
            # Create backup directory
            backup_dir = f"backups/{config.deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup configuration files
            config_files = [
                "config/production.py",
                "core/config.py",
                "requirements.prod.txt"
            ]
            
            for file_path in config_files:
                if os.path.exists(file_path):
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                        dst.write(src.read())
            
            logger.info(f"Backup created: {backup_dir}")
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise
    
    async def _deploy_new_version(self, config: DeploymentConfig):
        """Deploy new version"""
        try:
            # Update version files
            version_file = "VERSION"
            with open(version_file, 'w') as f:
                f.write(config.version)
            
            # Update deployment timestamp
            deployment_file = f"deployments/{config.deployment_id}.json"
            os.makedirs("deployments", exist_ok=True)
            
            deployment_info = {
                "deployment_id": config.deployment_id,
                "version": config.version,
                "environment": config.environment,
                "deployed_at": datetime.now().isoformat(),
                "services": config.services
            }
            
            with open(deployment_file, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            logger.info(f"New version {config.version} deployed")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    async def _update_configuration(self, config: DeploymentConfig):
        """Update configuration"""
        try:
            # Update environment variables
            env_file = ".env.production"
            if os.path.exists(env_file):
                with open(env_file, 'a') as f:
                    f.write(f"\nDEPLOYMENT_VERSION={config.version}\n")
                    f.write(f"DEPLOYMENT_ID={config.deployment_id}\n")
                    f.write(f"DEPLOYMENT_TIMESTAMP={datetime.now().isoformat()}\n")
            
            logger.info(f"Configuration updated for {config.deployment_id}")
            
        except Exception as e:
            logger.error(f"❌ Configuration update failed: {e}")
            raise
    
    async def _perform_health_checks(self, config: DeploymentConfig, result: DeploymentResult):
        """Perform health checks"""
        logger.info(f"Performing health checks for {config.deployment_id}")
        
        for endpoint in config.health_check_endpoints:
            try:
                health_result = await self._check_endpoint_health(endpoint)
                
                if health_result.status == "healthy":
                    result.health_checks_passed += 1
                else:
                    result.health_checks_failed += 1
                
                # Store health check result
                self.health_check_cache[endpoint] = health_result
                
                logger.info(f"Health check {endpoint}: {health_result.status}")
                
            except Exception as e:
                result.health_checks_failed += 1
                logger.error(f"❌ Health check failed for {endpoint}: {e}")
    
    async def _check_endpoint_health(self, endpoint: str) -> HealthCheckResult:
        """Check endpoint health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=self.health_check_interval) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = "healthy"
                    elif response.status < 500:
                        status = "degraded"
                    else:
                        status = "unhealthy"
                    
                    return HealthCheckResult(
                        endpoint=endpoint,
                        status=status,
                        response_time_ms=response_time,
                        status_code=response.status
                    )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                endpoint=endpoint,
                status="unhealthy",
                response_time_ms=response_time,
                status_code=0,
                error_message=str(e)
            )
    
    async def _trigger_rollback(self, config: DeploymentConfig, result: DeploymentResult):
        """Trigger rollback"""
        logger.warning(f"Triggering rollback for {config.deployment_id}")
        
        result.status = DeploymentStatus.ROLLING_BACK
        result.rollback_triggered = True
        
        try:
            # Create rollback config
            rollback_config = DeploymentConfig(
                deployment_id=f"{config.deployment_id}_rollback",
                version=config.rollback_version,
                environment=config.environment,
                services=config.services,
                health_check_endpoints=config.health_check_endpoints
            )
            
            # Perform rollback
            await self._run_deployment_steps(rollback_config, result)
            await self._perform_health_checks(rollback_config, result)
            
            if result.health_checks_failed == 0:
                result.status = DeploymentStatus.ROLLED_BACK
                logger.info(f"✅ Rollback successful for {config.deployment_id}")
            else:
                result.status = DeploymentStatus.FAILED
                result.error_message = "Rollback failed"
                logger.error(f"Rollback failed for {config.deployment_id}")
        
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = f"Rollback failed: {e}"
            logger.error(f"❌ Rollback failed for {config.deployment_id}: {e}")
    
    async def rollback(self, deployment_id: str) -> DeploymentResult:
        """Manually trigger rollback"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment_result = self.active_deployments[deployment_id]
        config = self.deployment_configs[deployment_id]
        
        if not config.rollback_version:
            raise ValueError(f"No rollback version configured for {deployment_id}")
        
        logger.info(f"🔄 Manual rollback triggered for {deployment_id}")
        
        # Trigger rollback
        await self._trigger_rollback(config, deployment_result)
        
        return deployment_result
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status"""
        return self.active_deployments.get(deployment_id)
    
    async def get_all_deployments(self) -> List[DeploymentResult]:
        """Get all deployments"""
        return list(self.active_deployments.values()) + self.deployment_history
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self.is_running:
            try:
                # Check health of active deployments
                for deployment_id, deployment_result in self.active_deployments.items():
                    if deployment_result.status == DeploymentStatus.ACTIVE:
                        config = self.deployment_configs.get(deployment_id)
                        if config:
                            await self._perform_health_checks(config, deployment_result)
                            
                            # Update deployment result
                            await self._store_deployment_result(deployment_result)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _deployment_monitoring_loop(self):
        """Monitor deployment status"""
        while self.is_running:
            try:
                # Check for stuck deployments
                current_time = datetime.now()
                timeout = timedelta(minutes=30)  # 30 minutes timeout
                
                stuck_deployments = []
                for deployment_id, deployment_result in self.active_deployments.items():
                    if (deployment_result.status == DeploymentStatus.DEPLOYING and
                        current_time - deployment_result.start_time > timeout):
                        stuck_deployments.append(deployment_id)
                
                for deployment_id in stuck_deployments:
                    logger.warning(f"⚠️ Deployment {deployment_id} appears to be stuck")
                    deployment_result = self.active_deployments[deployment_id]
                    deployment_result.status = DeploymentStatus.FAILED
                    deployment_result.error_message = "Deployment timeout"
                    deployment_result.end_time = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error in deployment monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_deployments(self):
        """Clean up old deployment records"""
        while self.is_running:
            try:
                # Keep only last 100 deployments in history
                if len(self.deployment_history) > 100:
                    self.deployment_history = self.deployment_history[-100:]
                
                # Clean up old health check cache
                current_time = datetime.now()
                timeout = timedelta(hours=1)  # 1 hour
                
                expired_keys = []
                for endpoint, result in self.health_check_cache.items():
                    if current_time - result.timestamp > timeout:
                        expired_keys.append(endpoint)
                
                for key in expired_keys:
                    del self.health_check_cache[key]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"❌ Error in cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _store_deployment_result(self, result: DeploymentResult):
        """Store deployment result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO deployment_history (
                        deployment_id, status, start_time, end_time,
                        health_checks_passed, health_checks_failed,
                        error_message, rollback_triggered
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (deployment_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        end_time = EXCLUDED.end_time,
                        health_checks_passed = EXCLUDED.health_checks_passed,
                        health_checks_failed = EXCLUDED.health_checks_failed,
                        error_message = EXCLUDED.error_message,
                        rollback_triggered = EXCLUDED.rollback_triggered
                """,
                result.deployment_id,
                result.status.value,
                result.start_time,
                result.end_time,
                result.health_checks_passed,
                result.health_checks_failed,
                result.error_message,
                result.rollback_triggered
                )
        except Exception as e:
            logger.error(f"❌ Error storing deployment result: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        total_deployments = len(self.active_deployments)
        active_deployments = sum(1 for d in self.active_deployments.values() 
                               if d.status == DeploymentStatus.ACTIVE)
        failed_deployments = sum(1 for d in self.active_deployments.values() 
                               if d.status == DeploymentStatus.FAILED)
        
        return {
            "total_deployments": total_deployments,
            "active_deployments": active_deployments,
            "failed_deployments": failed_deployments,
            "health_check_cache_size": len(self.health_check_cache),
            "deployment_history_size": len(self.deployment_history)
        }
