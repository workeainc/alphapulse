#!/usr/bin/env python3
"""
Resilience utilities for AlphaPulse
Provides retry logic, circuit breaker, timeout management, and other production resilience features
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import functools
import contextlib

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class RetryConfig:
    """Configuration for retry operations"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: tuple = (Exception,)
    backoff_factor: float = 1.0

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    monitor_interval: float = 10.0  # seconds

class RetryManager:
    """Advanced retry logic with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_retry(
        self, 
        operation: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """Execute operation with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success - log and return
                if attempt > 0:
                    self.logger.info(f"✅ Operation succeeded on attempt {attempt + 1}")
                return result
                
            except self.config.retry_on_exceptions as e:
                last_exception = e
                
                # Check if we should retry
                if attempt == self.config.max_attempts - 1:
                    self.logger.error(f"❌ Operation failed after {self.config.max_attempts} attempts: {e}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = self._calculate_delay(attempt)
                
                self.logger.warning(
                    f"⚠️ Operation failed (attempt {attempt + 1}/{self.config.max_attempts}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Apply backoff factor
        delay *= self.config.backoff_factor
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            delay += jitter
        
        return delay

class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures"""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now(timezone.utc)
        self.logger = logging.getLogger(__name__)
        self.monitor_task = None
        
        # Start monitoring only if we're in an event loop
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start monitoring task if we're in an event loop"""
        try:
            loop = asyncio.get_running_loop()
            self.monitor_task = asyncio.create_task(self._monitor_circuit())
        except RuntimeError:
            # No running event loop, will start when execute is called
            pass
    
    async def execute(
        self, 
        operation: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """Execute operation with circuit breaker protection"""
        
        # Start monitoring if not already started
        if self.monitor_task is None:
            self.monitor_task = asyncio.create_task(self._monitor_circuit())
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.logger.info("🔄 Attempting to reset circuit breaker")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Success - reset circuit
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            # Failure - update circuit state
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.logger.info("✅ Circuit breaker reset - service recovered")
            self.state = CircuitState.CLOSED
        
        self.failure_count = 0
        self.last_state_change = datetime.now(timezone.utc)
    
    def _on_failure(self, exception: Exception):
        """Handle operation failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.logger.warning(
                    f"⚠️ Circuit breaker OPENED after {self.failure_count} failures. "
                    f"Last error: {exception}"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now(timezone.utc)
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    async def _monitor_circuit(self):
        """Background monitoring of circuit breaker state"""
        while True:
            try:
                # Log circuit state periodically
                if self.state == CircuitState.OPEN:
                    time_open = (datetime.now(timezone.utc) - self.last_state_change).total_seconds()
                    self.logger.info(
                        f"🔌 Circuit breaker OPEN for {time_open:.1f}s. "
                        f"Will attempt reset in {self.config.recovery_timeout - time_open:.1f}s"
                    )
                
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in circuit breaker monitor: {e}")
                await asyncio.sleep(self.config.monitor_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_state_change': self.last_state_change.isoformat(),
            'failure_threshold': self.config.failure_threshold,
            'recovery_timeout': self.config.recovery_timeout
        }

class TimeoutManager:
    """Timeout management for operations"""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_timeout(
        self, 
        operation: Callable[..., T], 
        timeout: float = None, 
        *args, 
        **kwargs
    ) -> T:
        """Execute operation with timeout"""
        timeout = timeout or self.default_timeout
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)
            else:
                # For sync operations, run in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, operation, *args, **kwargs),
                    timeout=timeout
                )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Operation timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")
    
    @contextlib.asynccontextmanager
    async def timeout_context(self, timeout: float = None):
        """Context manager for timeout operations"""
        timeout = timeout or self.default_timeout
        
        try:
            yield
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Operation timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout}s")

class DeadLetterQueue:
    """Dead letter queue for failed operations"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.failed_operations = []
        self.logger = logging.getLogger(__name__)
    
    async def add_failed_operation(
        self, 
        operation_name: str, 
        data: Any, 
        error: Exception, 
        context: Dict[str, Any] = None
    ):
        """Add a failed operation to the dead letter queue"""
        try:
            failed_op = {
                'id': f"dlq_{int(time.time())}_{len(self.failed_operations)}",
                'operation_name': operation_name,
                'data': data,
                'error': str(error),
                'error_type': type(error).__name__,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'context': context or {},
                'retry_count': 0
            }
            
            # Add to queue
            self.failed_operations.append(failed_op)
            
            # Maintain queue size
            if len(self.failed_operations) > self.max_queue_size:
                removed = self.failed_operations.pop(0)
                self.logger.warning(f"⚠️ Dead letter queue full, removed oldest entry: {removed['id']}")
            
            self.logger.info(f"📥 Added failed operation to DLQ: {operation_name} - {error}")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding to dead letter queue: {e}")
    
    async def retry_failed_operation(self, operation_id: str, retry_func: Callable) -> bool:
        """Retry a failed operation from the dead letter queue"""
        try:
            # Find the operation
            failed_op = next((op for op in self.failed_operations if op['id'] == operation_id), None)
            
            if not failed_op:
                self.logger.warning(f"⚠️ Failed operation {operation_id} not found in DLQ")
                return False
            
            # Increment retry count
            failed_op['retry_count'] += 1
            
            # Attempt retry
            try:
                await retry_func(failed_op['data'])
                
                # Success - remove from DLQ
                self.failed_operations.remove(failed_op)
                self.logger.info(f"✅ Successfully retried operation {operation_id}")
                return True
                
            except Exception as e:
                failed_op['last_retry_error'] = str(e)
                failed_op['last_retry_time'] = datetime.now(timezone.utc).isoformat()
                self.logger.warning(f"⚠️ Retry failed for operation {operation_id}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error retrying operation {operation_id}: {e}")
            return False
    
    def get_failed_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed operations from the dead letter queue"""
        return self.failed_operations[-limit:] if limit else self.failed_operations.copy()
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get dead letter queue statistics"""
        return {
            'total_failed_operations': len(self.failed_operations),
            'max_queue_size': self.max_queue_size,
            'queue_utilization': len(self.failed_operations) / self.max_queue_size * 100,
            'oldest_failure': self.failed_operations[0]['timestamp'] if self.failed_operations else None,
            'newest_failure': self.failed_operations[-1]['timestamp'] if self.failed_operations else None
        }

class ResilienceManager:
    """Main resilience manager combining all resilience features"""
    
    def __init__(self):
        self.retry_manager = RetryManager()
        self.circuit_breaker = CircuitBreaker()
        self.timeout_manager = TimeoutManager()
        self.dead_letter_queue = DeadLetterQueue()
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_resilience(
        self,
        operation: Callable[..., T],
        operation_name: str,
        retry_config: RetryConfig = None,
        timeout: float = None,
        *args,
        **kwargs
    ) -> T:
        """Execute operation with full resilience protection"""
        try:
            # Execute with timeout
            result = await self.timeout_manager.execute_with_timeout(
                operation, timeout, *args, **kwargs
            )
            return result
            
        except Exception as e:
            # Add to dead letter queue
            await self.dead_letter_queue.add_failed_operation(
                operation_name, 
                {'args': args, 'kwargs': kwargs}, 
                e, 
                {'retry_config': retry_config, 'timeout': timeout}
            )
            
            # Re-raise the exception
            raise
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status"""
        return {
            'circuit_breaker': self.circuit_breaker.get_status(),
            'dead_letter_queue': self.dead_letter_queue.get_queue_stats(),
            'retry_manager': {
                'config': {
                    'max_attempts': self.retry_manager.config.max_attempts,
                    'base_delay': self.retry_manager.config.base_delay,
                    'max_delay': self.retry_manager.config.max_delay
                }
            },
            'timeout_manager': {
                'default_timeout': self.timeout_manager.default_timeout
            }
        }

# Global resilience manager instance - created lazily
_resilience_manager = None

def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager instance (lazy initialization)"""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager

# Convenience functions
async def execute_with_retry(operation: Callable[..., T], *args, **kwargs) -> T:
    """Execute operation with retry logic"""
    return await get_resilience_manager().retry_manager.execute_with_retry(operation, *args, **kwargs)

async def execute_with_circuit_breaker(operation: Callable[..., T], *args, **kwargs) -> T:
    """Execute operation with circuit breaker protection"""
    return await get_resilience_manager().circuit_breaker.execute(operation, *args, **kwargs)

async def execute_with_timeout(operation: Callable[..., T], timeout: float = None, *args, **kwargs) -> T:
    """Execute operation with timeout"""
    return await get_resilience_manager().timeout_manager.execute_with_timeout(operation, timeout, *args, **kwargs)

async def execute_with_resilience(operation: Callable[..., T], operation_name: str, *args, **kwargs) -> T:
    """Execute operation with full resilience protection"""
    return await get_resilience_manager().execute_with_resilience(operation, operation_name, *args, **kwargs)
