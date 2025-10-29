#!/usr/bin/env python3
"""
Multi-Region Resilience Framework for AlphaPulse
Provides cross-region failover, load balancing, and disaster recovery
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.app.core.resilience import get_resilience_manager
from src.app.database.enhanced_connection import get_enhanced_connection

logger = logging.getLogger(__name__)

class RegionStatus(Enum):
    """Region health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class FailoverStrategy(Enum):
    """Failover strategies"""
    ACTIVE_PASSIVE = "active_passive"      # One primary, others standby
    ACTIVE_ACTIVE = "active_active"        # All regions active, load balanced
    ROUND_ROBIN = "round_robin"           # Rotate through regions
    LATENCY_BASED = "latency_based"       # Choose region with lowest latency
    HEALTH_BASED = "health_based"         # Choose healthiest region

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LATENCY_BASED = "latency_based"
    HEALTH_BASED = "health_based"

@dataclass
class RegionConfig:
    """Configuration for a region"""
    name: str
    endpoint: str
    database_url: str
    weight: float = 1.0
    max_connections: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    health_check_interval: float = 30.0
    failover_priority: int = 0  # Lower number = higher priority

@dataclass
class RegionHealth:
    """Health status of a region"""
    region_name: str
    status: RegionStatus
    last_check: datetime
    response_time: float
    error_rate: float
    connection_count: int
    is_available: bool
    last_failure: Optional[datetime] = None
    failure_reason: Optional[str] = None

@dataclass
class FailoverEvent:
    """Record of a failover event"""
    timestamp: datetime
    from_region: str
    to_region: str
    reason: str
    duration: float
    success: bool

class RegionHealthMonitor:
    """Monitors health of all regions"""
    
    def __init__(self, regions: List[RegionConfig]):
        self.regions = regions
        self.health_status: Dict[str, RegionHealth] = {}
        self.lock = threading.Lock()
        self.monitoring_task = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize health status
        for region in regions:
            self.health_status[region.name] = RegionHealth(
                region_name=region.name,
                status=RegionStatus.UNHEALTHY,
                last_check=datetime.now(timezone.utc),
                response_time=0.0,
                error_rate=0.0,
                connection_count=0,
                is_available=False
            )
        
        # Don't start monitoring during initialization
        # It will be started when needed
    
    def _start_monitoring(self):
        """Start background health monitoring"""
        # Don't start monitoring during initialization to avoid event loop issues
        # Monitoring will be started when needed
        pass
    
    async def start_monitoring(self):
        """Start background health monitoring (call this when event loop is available)"""
        if self.monitoring_task is None:
            async def monitoring_loop():
                while True:
                    try:
                        await self._check_all_regions()
                        await asyncio.sleep(30)  # Check every 30 seconds
                    except Exception as e:
                        self.logger.error(f"❌ Error in health monitoring: {e}")
                        await asyncio.sleep(60)
            
            self.monitoring_task = asyncio.create_task(monitoring_loop())
            self.logger.info("✅ Health monitoring started")
    
    async def _check_all_regions(self):
        """Check health of all regions"""
        # Skip health checks if monitoring hasn't started
        if self.monitoring_task is None:
            return
            
        tasks = []
        for region in self.regions:
            task = asyncio.create_task(self._check_region_health(region))
            tasks.append(task)
        
        # Wait for all health checks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_region_health(self, region: RegionConfig):
        """Check health of a specific region"""
        try:
            start_time = time.time()
            
            # Test database connection
            connection = get_enhanced_connection(region.database_url)
            await connection.test_connection()
            
            response_time = time.time() - start_time
            
            # Update health status
            with self.lock:
                health = self.health_status[region.name]
                health.status = RegionStatus.HEALTHY
                health.last_check = datetime.now(timezone.utc)
                health.response_time = response_time
                health.error_rate = 0.0
                health.is_available = True
                health.last_failure = None
                health.failure_reason = None
            
            self.logger.debug(f"✅ Region {region.name} healthy (response: {response_time:.3f}s)")
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Update health status
            with self.lock:
                health = self.health_status[region.name]
                health.status = RegionStatus.UNHEALTHY
                health.last_check = datetime.now(timezone.utc)
                health.response_time = response_time
                health.error_rate = 1.0
                health.is_available = False
                health.last_failure = datetime.now(timezone.utc)
                health.failure_reason = str(e)
            
            self.logger.warning(f"⚠️ Region {region.name} unhealthy: {e}")
    
    def get_region_health(self, region_name: str) -> Optional[RegionHealth]:
        """Get health status of a specific region"""
        with self.lock:
            return self.health_status.get(region_name)
    
    def get_all_health_status(self) -> List[RegionHealth]:
        """Get health status of all regions"""
        with self.lock:
            return list(self.health_status.values())
    
    def get_healthy_regions(self) -> List[str]:
        """Get list of healthy regions"""
        with self.lock:
            return [
                name for name, health in self.health_status.items()
                if health.is_available and health.status == RegionStatus.HEALTHY
            ]
    
    def get_best_region(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED) -> Optional[str]:
        """Get the best region based on strategy"""
        healthy_regions = self.get_healthy_regions()
        
        if not healthy_regions:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin
            return random.choice(healthy_regions)
        
        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            # Choose healthiest region
            best_region = None
            best_score = -1
            
            for region_name in healthy_regions:
                health = self.health_status[region_name]
                score = self._calculate_health_score(health)
                if score > best_score:
                    best_score = score
                    best_region = region_name
            
            return best_region
        
        elif strategy == LoadBalancingStrategy.LATENCY_BASED:
            # Choose region with lowest latency
            best_region = None
            best_latency = float('inf')
            
            for region_name in healthy_regions:
                health = self.health_status[region_name]
                if health.response_time < best_latency:
                    best_latency = health.response_time
                    best_region = region_name
            
            return best_region
        
        else:
            # Default to first healthy region
            return healthy_regions[0]
    
    def _calculate_health_score(self, health: RegionHealth) -> float:
        """Calculate health score for a region"""
        score = 100.0
        
        # Deduct points for response time
        if health.response_time > 1.0:
            score -= 20
        elif health.response_time > 0.5:
            score -= 10
        
        # Deduct points for error rate
        score -= health.error_rate * 50
        
        # Deduct points for connection count
        if health.connection_count > 80:
            score -= 20
        elif health.connection_count > 50:
            score -= 10
        
        return max(0.0, score)

class MultiRegionConnectionManager:
    """Manages connections across multiple regions"""
    
    def __init__(self, regions: List[RegionConfig], failover_strategy: FailoverStrategy = FailoverStrategy.ACTIVE_PASSIVE):
        self.regions = regions
        self.failover_strategy = failover_strategy
        self.health_monitor = RegionHealthMonitor(regions)
        self.current_primary_region = None
        self.failover_history: List[FailoverEvent] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize primary region
        self._initialize_primary_region()
    
    def _initialize_primary_region(self):
        """Initialize the primary region"""
        # Sort regions by failover priority
        sorted_regions = sorted(self.regions, key=lambda r: r.failover_priority)
        
        # Find first healthy region
        for region in sorted_regions:
            if self.health_monitor.get_region_health(region.name).is_available:
                self.current_primary_region = region.name
                self.logger.info(f"🏆 Primary region set to: {region.name}")
                break
        
        if not self.current_primary_region:
            self.logger.warning("⚠️ No healthy regions available for primary")
    
    async def get_connection(self, region_name: Optional[str] = None) -> Any:
        """Get a database connection for the specified or best region"""
        try:
            if region_name:
                # Use specified region if available
                if self._is_region_available(region_name):
                    return await self._create_connection(region_name)
                else:
                    self.logger.warning(f"⚠️ Specified region {region_name} not available")
            
            # Get best available region
            best_region = self._get_best_available_region()
            if not best_region:
                raise Exception("No healthy regions available")
            
            return await self._create_connection(best_region)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting connection: {e}")
            raise
    
    async def _create_connection(self, region_name: str) -> Any:
        """Create a connection for the specified region"""
        region_config = next(r for r in self.regions if r.name == region_name)
        
        # Get enhanced connection
        connection = get_enhanced_connection(region_config.database_url)
        
        # Test connection before returning
        await connection.test_connection()
        
        return connection
    
    def _is_region_available(self, region_name: str) -> bool:
        """Check if a region is available"""
        health = self.health_monitor.get_region_health(region_name)
        return health and health.is_available
    
    def _get_best_available_region(self) -> Optional[str]:
        """Get the best available region based on strategy"""
        if self.failover_strategy == FailoverStrategy.ACTIVE_PASSIVE:
            # Use primary region if available, otherwise failover
            if self.current_primary_region and self._is_region_available(self.current_primary_region):
                return self.current_primary_region
            
            # Find next best region
            return self._find_failover_region()
        
        elif self.failover_strategy == FailoverStrategy.ACTIVE_ACTIVE:
            # Use load balancing strategy
            return self.health_monitor.get_best_region(LoadBalancingStrategy.HEALTH_BASED)
        
        elif self.failover_strategy == FailoverStrategy.ROUND_ROBIN:
            # Round-robin through healthy regions
            healthy_regions = self.health_monitor.get_healthy_regions()
            if healthy_regions:
                return random.choice(healthy_regions)
        
        elif self.failover_strategy == FailoverStrategy.LATENCY_BASED:
            # Choose region with lowest latency
            return self.health_monitor.get_best_region(LoadBalancingStrategy.LATENCY_BASED)
        
        elif self.failover_strategy == FailoverStrategy.HEALTH_BASED:
            # Choose healthiest region
            return self.health_monitor.get_best_region(LoadBalancingStrategy.HEALTH_BASED)
        
        return None
    
    def _find_failover_region(self) -> Optional[str]:
        """Find the next best region for failover"""
        # Sort regions by failover priority
        sorted_regions = sorted(self.regions, key=lambda r: r.failover_priority)
        
        for region in sorted_regions:
            if region.name != self.current_primary_region and self._is_region_available(region.name):
                return region.name
        
        return None
    
    async def execute_with_failover(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute an operation with automatic failover"""
        last_error = None
        
        # Try primary region first
        if self.current_primary_region and self._is_region_available(self.current_primary_region):
            try:
                connection = await self.get_connection(self.current_primary_region)
                result = await operation(connection, *args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                self.logger.warning(f"⚠️ Primary region {self.current_primary_region} failed: {e}")
        
        # Try failover regions
        failover_regions = self._get_failover_candidates()
        
        for region_name in failover_regions:
            try:
                self.logger.info(f"🔄 Failing over to region: {region_name}")
                
                connection = await self.get_connection(region_name)
                result = await operation(connection, *args, **kwargs)
                
                # Record successful failover
                await self._record_failover_event(
                    from_region=self.current_primary_region or "unknown",
                    to_region=region_name,
                    reason="Primary region failure",
                    success=True
                )
                
                # Update primary region
                self.current_primary_region = region_name
                self.logger.info(f"✅ Failover successful to {region_name}")
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"⚠️ Failover region {region_name} failed: {e}")
        
        # All regions failed
        await self._record_failover_event(
            from_region=self.current_primary_region or "unknown",
            to_region="none",
            reason="All regions failed",
            success=False
        )
        
        raise Exception(f"All regions failed. Last error: {last_error}")
    
    def _get_failover_candidates(self) -> List[str]:
        """Get list of failover candidate regions"""
        candidates = []
        
        # Sort regions by failover priority
        sorted_regions = sorted(self.regions, key=lambda r: r.failover_priority)
        
        for region in sorted_regions:
            if region.name != self.current_primary_region and self._is_region_available(region.name):
                candidates.append(region.name)
        
        return candidates
    
    async def _record_failover_event(self, from_region: str, to_region: str, reason: str, success: bool):
        """Record a failover event"""
        event = FailoverEvent(
            timestamp=datetime.now(timezone.utc),
            from_region=from_region,
            to_region=to_region,
            reason=reason,
            duration=0.0,  # Could be calculated if needed
            success=success
        )
        
        with self.lock:
            self.failover_history.append(event)
            
            # Keep only last 100 events
            if len(self.failover_history) > 100:
                self.failover_history = self.failover_history[-100:]
    
    def get_failover_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get failover history"""
        with self.lock:
            return [
                asdict(event) for event in self.failover_history[-limit:]
            ]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current multi-region status"""
        return {
            "current_primary": self.current_primary_region,
            "failover_strategy": self.failover_strategy.value,
            "total_regions": len(self.regions),
            "healthy_regions": len(self.health_monitor.get_healthy_regions()),
            "region_health": [
                asdict(health) for health in self.health_monitor.get_all_health_status()
            ],
            "recent_failovers": self.get_failover_history(5)
        }
    
    async def force_failover(self, target_region: str) -> bool:
        """Force a failover to a specific region"""
        try:
            if not self._is_region_available(target_region):
                self.logger.error(f"❌ Target region {target_region} not available")
                return False
            
            old_primary = self.current_primary_region
            
            # Update primary region
            self.current_primary_region = target_region
            
            # Record forced failover
            await self._record_failover_event(
                from_region=old_primary or "unknown",
                to_region=target_region,
                reason="Manual failover",
                success=True
            )
            
            self.logger.info(f"✅ Manual failover to {target_region} successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Manual failover failed: {e}")
            return False

# Global multi-region manager instance
_multi_region_manager = None

def get_multi_region_manager() -> MultiRegionConnectionManager:
    """Get the global multi-region manager instance"""
    global _multi_region_manager
    if _multi_region_manager is None:
        # Default regions - should be configured based on environment
        default_regions = [
            RegionConfig(
                name="primary",
                endpoint="https://primary.alphapulse.com",
                database_url="postgresql://user:pass@primary-db:5432/alphapulse",
                weight=1.0,
                failover_priority=0
            ),
            RegionConfig(
                name="secondary",
                endpoint="https://secondary.alphapulse.com", 
                database_url="postgresql://user:pass@secondary-db:5432/alphapulse",
                weight=0.8,
                failover_priority=1
            ),
            RegionConfig(
                name="disaster_recovery",
                endpoint="https://dr.alphapulse.com",
                database_url="postgresql://user:pass@dr-db:5432/alphapulse", 
                weight=0.6,
                failover_priority=2
            )
        ]
        
        _multi_region_manager = MultiRegionConnectionManager(
            regions=default_regions,
            failover_strategy=FailoverStrategy.ACTIVE_PASSIVE
        )
    
    return _multi_region_manager

async def execute_with_multi_region_failover(operation: Callable, *args, **kwargs) -> Any:
    """Execute operation with multi-region failover"""
    manager = get_multi_region_manager()
    return await manager.execute_with_failover(operation, *args, **kwargs)
