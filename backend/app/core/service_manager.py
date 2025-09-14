"""
Service Lifecycle Manager for AlphaPlus
Handles proper initialization, dependency injection, and graceful shutdown of all services
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ServiceInfo:
    """Service information"""
    name: str
    service: Any
    dependencies: List[str] = field(default_factory=list)
    status: ServiceStatus = ServiceStatus.UNINITIALIZED
    start_time: Optional[datetime] = None
    error_message: Optional[str] = None
    health_check: Optional[Callable] = None

class ServiceManager:
    """
    Service lifecycle manager for AlphaPlus
    Handles initialization order, dependency injection, and graceful shutdown
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.initialization_order: List[str] = []
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        
    def register_service(
        self, 
        name: str, 
        service: Any, 
        dependencies: Optional[List[str]] = None,
        health_check: Optional[Callable] = None
    ):
        """
        Register a service with the manager
        
        Args:
            name: Service name
            service: Service instance
            dependencies: List of service dependencies
            health_check: Optional health check function
        """
        self.services[name] = ServiceInfo(
            name=name,
            service=service,
            dependencies=dependencies or [],
            health_check=health_check
        )
        logger.info(f"Registered service: {name}")
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate the correct initialization order based on dependencies"""
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            service_info = self.services.get(service_name)
            if service_info:
                for dep in service_info.dependencies:
                    if dep not in self.services:
                        raise ValueError(f"Service {service_name} depends on unknown service: {dep}")
                    visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        for service_name in self.services:
            if service_name not in visited:
                visit(service_name)
        
        return order
    
    async def initialize_services(self) -> bool:
        """
        Initialize all services in the correct order
        
        Returns:
            bool: True if all services initialized successfully
        """
        try:
            logger.info("🚀 Starting service initialization...")
            
            # Calculate initialization order
            self.initialization_order = self._calculate_initialization_order()
            logger.info(f"Service initialization order: {self.initialization_order}")
            
            # Initialize services in order
            for service_name in self.initialization_order:
                service_info = self.services[service_name]
                
                try:
                    logger.info(f"Initializing service: {service_name}")
                    service_info.status = ServiceStatus.INITIALIZING
                    
                    # Initialize the service
                    if hasattr(service_info.service, 'initialize'):
                        await service_info.service.initialize()
                    elif hasattr(service_info.service, 'start'):
                        await service_info.service.start()
                    
                    service_info.status = ServiceStatus.RUNNING
                    service_info.start_time = datetime.now()
                    logger.info(f"✅ Service initialized: {service_name}")
                    
                except Exception as e:
                    service_info.status = ServiceStatus.ERROR
                    service_info.error_message = str(e)
                    logger.error(f"❌ Failed to initialize service {service_name}: {e}")
                    logger.error(traceback.format_exc())
                    return False
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("🎉 All services initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def shutdown_services(self):
        """Gracefully shutdown all services"""
        try:
            logger.info("🛑 Starting service shutdown...")
            self._shutdown_event.set()
            
            # Stop health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown services in reverse order
            for service_name in reversed(self.initialization_order):
                service_info = self.services[service_name]
                
                try:
                    logger.info(f"Shutting down service: {service_name}")
                    service_info.status = ServiceStatus.STOPPING
                    
                    # Shutdown the service
                    if hasattr(service_info.service, 'shutdown'):
                        await service_info.service.shutdown()
                    elif hasattr(service_info.service, 'stop'):
                        await service_info.service.stop()
                    elif hasattr(service_info.service, 'close'):
                        await service_info.service.close()
                    
                    service_info.status = ServiceStatus.STOPPED
                    logger.info(f"✅ Service shutdown: {service_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to shutdown service {service_name}: {e}")
            
            logger.info("🎉 All services shutdown successfully!")
            
        except Exception as e:
            logger.error(f"❌ Service shutdown failed: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks for all services"""
        for service_name, service_info in self.services.items():
            if service_info.status == ServiceStatus.RUNNING and service_info.health_check:
                try:
                    is_healthy = await service_info.health_check()
                    if not is_healthy:
                        logger.warning(f"Service health check failed: {service_name}")
                        service_info.status = ServiceStatus.ERROR
                except Exception as e:
                    logger.error(f"Health check error for {service_name}: {e}")
                    service_info.status = ServiceStatus.ERROR
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service by name"""
        service_info = self.services.get(name)
        return service_info.service if service_info else None
    
    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Get service status by name"""
        service_info = self.services.get(name)
        return service_info.status if service_info else None
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        status = {}
        for name, service_info in self.services.items():
            status[name] = {
                'status': service_info.status.value,
                'start_time': service_info.start_time.isoformat() if service_info.start_time else None,
                'error_message': service_info.error_message,
                'dependencies': service_info.dependencies
            }
        return status
    
    def is_all_services_healthy(self) -> bool:
        """Check if all services are healthy"""
        return all(
            service_info.status == ServiceStatus.RUNNING 
            for service_info in self.services.values()
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all services"""
        try:
            health_status = {
                'overall_status': 'healthy' if self.is_all_services_healthy() else 'unhealthy',
                'services': {},
                'total_services': len(self.services),
                'healthy_services': 0,
                'unhealthy_services': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            for name, service_info in self.services.items():
                is_healthy = service_info.status == ServiceStatus.RUNNING
                if is_healthy:
                    health_status['healthy_services'] += 1
                else:
                    health_status['unhealthy_services'] += 1
                
                health_status['services'][name] = {
                    'status': service_info.status.value,
                    'healthy': is_healthy,
                    'start_time': service_info.start_time.isoformat() if service_info.start_time else None,
                    'error_message': service_info.error_message
                }
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Error getting health status: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global service manager instance
service_manager = ServiceManager()

# Convenience functions
def get_service_manager() -> ServiceManager:
    """Get the global service manager instance"""
    return service_manager

async def initialize_all_services() -> bool:
    """Initialize all registered services"""
    return await service_manager.initialize_services()

async def shutdown_all_services():
    """Shutdown all registered services"""
    await service_manager.shutdown_services()

async def stop_all_services():
    """Stop all registered services (alias for shutdown)"""
    await service_manager.shutdown_services()

def register_service(name: str, service: Any, dependencies: Optional[List[str]] = None, health_check: Optional[Callable] = None):
    """Register a service with the global manager"""
    service_manager.register_service(name, service, dependencies, health_check)

def get_service(name: str) -> Optional[Any]:
    """Get a service from the global manager"""
    return service_manager.get_service(name)
