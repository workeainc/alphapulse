#!/usr/bin/env python3
"""
Rate Limiting and Backpressure Management for AlphaPulse
Provides intelligent rate limiting, backpressure, and throttling mechanisms
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

class BackpressureStrategy(Enum):
    """Backpressure strategies"""
    DROP = "drop"                    # Drop new requests
    QUEUE = "queue"                  # Queue requests with timeout
    THROTTLE = "throttle"            # Throttle request rate
    CIRCUIT_BREAKER = "circuit_breaker"  # Open circuit breaker

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    max_requests: int = 100          # Max requests per window
    window_size: float = 60.0        # Window size in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: int = 10             # Allow burst of requests
    jitter: bool = True              # Add jitter to prevent thundering herd

@dataclass
class BackpressureConfig:
    """Configuration for backpressure"""
    max_queue_size: int = 1000       # Max queued requests
    queue_timeout: float = 30.0      # Queue timeout in seconds
    strategy: BackpressureStrategy = BackpressureStrategy.THROTTLE
    throttle_factor: float = 0.5     # Throttle to 50% of normal rate
    circuit_breaker_threshold: int = 100  # Requests before opening circuit

@dataclass
class RateLimitStats:
    """Rate limiting statistics"""
    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    current_rate: float = 0.0
    window_start: datetime = None
    last_request_time: datetime = None

class FixedWindowRateLimiter:
    """Fixed window rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.window_start = datetime.now(timezone.utc)
        self.request_count = 0
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = datetime.now(timezone.utc)
            
            # Check if window has expired
            if (now - self.window_start).total_seconds() >= self.config.window_size:
                self.window_start = now
                self.request_count = 0
            
            # Check if under limit
            if self.request_count < self.config.max_requests:
                self.request_count += 1
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            now = datetime.now(timezone.utc)
            window_progress = (now - self.window_start).total_seconds() / self.config.window_size
            
            return {
                'strategy': self.config.strategy.value,
                'window_start': self.window_start.isoformat(),
                'window_progress': min(window_progress, 1.0),
                'request_count': self.request_count,
                'max_requests': self.config.max_requests,
                'remaining_requests': max(0, self.config.max_requests - self.request_count)
            }

class SlidingWindowRateLimiter:
    """Sliding window rate limiter with precision"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.requests = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = datetime.now(timezone.utc)
            
            # Remove expired requests
            cutoff_time = now - timedelta(seconds=self.config.window_size)
            while self.requests and self.requests[0] < cutoff_time:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.config.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            now = datetime.now(timezone.utc)
            cutoff_time = now - timedelta(seconds=self.config.window_size)
            
            # Count requests in current window
            current_requests = sum(1 for req_time in self.requests if req_time >= cutoff_time)
            
            return {
                'strategy': self.config.strategy.value,
                'current_requests': current_requests,
                'max_requests': self.config.max_requests,
                'remaining_requests': max(0, self.config.max_requests - current_requests),
                'window_size': self.config.window_size
            }

class TokenBucketRateLimiter:
    """Token bucket rate limiter for burst handling"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokens = config.max_requests
        self.last_refill = datetime.now(timezone.utc)
        self.lock = threading.Lock()
        
        # Calculate refill rate
        self.refill_rate = config.max_requests / config.window_size
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = datetime.now(timezone.utc)
            
            # Refill tokens
            time_passed = (now - self.last_refill).total_seconds()
            tokens_to_add = time_passed * self.refill_rate
            self.tokens = min(self.config.max_requests, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if tokens available
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            now = datetime.now(timezone.utc)
            time_passed = (now - self.last_refill).total_seconds()
            tokens_to_add = time_passed * self.refill_rate
            projected_tokens = min(self.config.max_requests, self.tokens + tokens_to_add)
            
            return {
                'strategy': self.config.strategy.value,
                'current_tokens': self.tokens,
                'projected_tokens': projected_tokens,
                'max_tokens': self.config.max_requests,
                'refill_rate': self.refill_rate,
                'last_refill': self.last_refill.isoformat()
            }

class RateLimiter:
    """Main rate limiter with strategy selection"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create appropriate limiter based on strategy
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            self.limiter = FixedWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.limiter = SlidingWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.limiter = TokenBucketRateLimiter(config)
        else:
            self.limiter = SlidingWindowRateLimiter(config)
        
        # Statistics
        self.stats = RateLimitStats()
        self.stats.window_start = datetime.now(timezone.utc)
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        is_allowed = self.limiter.is_allowed()
        
        # Update statistics
        self.stats.total_requests += 1
        if is_allowed:
            self.stats.allowed_requests += 1
        else:
            self.stats.blocked_requests += 1
        
        self.stats.last_request_time = datetime.now(timezone.utc)
        
        return is_allowed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        limiter_stats = self.limiter.get_stats()
        
        # Calculate current rate
        if self.stats.last_request_time:
            time_diff = (datetime.now(timezone.utc) - self.stats.last_request_time).total_seconds()
            if time_diff > 0:
                self.stats.current_rate = self.stats.total_requests / time_diff
        
        return {
            **limiter_stats,
            'total_requests': self.stats.total_requests,
            'allowed_requests': self.stats.allowed_requests,
            'blocked_requests': self.stats.blocked_requests,
            'current_rate': self.stats.current_rate,
            'block_rate': self.stats.blocked_requests / max(1, self.stats.total_requests)
        }

class RequestQueue:
    """Queue for handling backpressure"""
    
    def __init__(self, config: BackpressureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.queue = queue.Queue(maxsize=config.max_queue_size)
        self.stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_dropped': 0,
            'total_timeout': 0,
            'current_queue_size': 0
        }
    
    async def enqueue(self, request_data: Any, timeout: float = None) -> bool:
        """Enqueue a request"""
        timeout = timeout or self.config.queue_timeout
        
        try:
            # Try to put in queue with timeout
            self.queue.put(request_data, timeout=timeout)
            
            # Update statistics
            self.stats['total_queued'] += 1
            self.stats['current_queue_size'] = self.queue.qsize()
            
            return True
            
        except queue.Full:
            # Queue is full, handle based on strategy
            if self.config.strategy == BackpressureStrategy.DROP:
                self.stats['total_dropped'] += 1
                self.logger.warning("⚠️ Request dropped due to full queue")
                return False
            else:
                # For other strategies, try to wait
                try:
                    self.queue.put(request_data, timeout=timeout)
                    self.stats['total_queued'] += 1
                    self.stats['current_queue_size'] = self.queue.qsize()
                    return True
                except queue.Full:
                    self.stats['total_dropped'] += 1
                    self.logger.warning("⚠️ Request dropped after timeout")
                    return False
    
    def dequeue(self) -> Optional[Any]:
        """Dequeue a request"""
        try:
            request = self.queue.get_nowait()
            self.stats['total_processed'] += 1
            self.stats['current_queue_size'] = self.queue.qsize()
            return request
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            'queue_size': self.queue.qsize(),
            'max_queue_size': self.config.max_queue_size,
            'queue_utilization': (self.stats['current_queue_size'] / self.config.max_queue_size) * 100
        }

class BackpressureManager:
    """Manages backpressure strategies"""
    
    def __init__(self, config: BackpressureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.request_queue = RequestQueue(config)
        self.circuit_breaker_open = False
        self.circuit_breaker_count = 0
        self.last_circuit_breaker_reset = datetime.now(timezone.utc)
        
        # Throttling
        self.throttle_start = datetime.now(timezone.utc)
        self.throttle_request_count = 0
    
    async def handle_request(self, request_data: Any, process_func: Callable) -> Any:
        """Handle request with backpressure management"""
        
        # Check circuit breaker
        if self.circuit_breaker_open:
            if self._should_reset_circuit_breaker():
                self._reset_circuit_breaker()
            else:
                raise Exception("Circuit breaker is open")
        
        # Handle based on strategy
        if self.config.strategy == BackpressureStrategy.QUEUE:
            return await self._handle_queued_request(request_data, process_func)
        elif self.config.strategy == BackpressureStrategy.THROTTLE:
            return await self._handle_throttled_request(request_data, process_func)
        elif self.config.strategy == BackpressureStrategy.CIRCUIT_BREAKER:
            return await self._handle_circuit_breaker_request(request_data, process_func)
        else:
            # Default to drop
            return await self._handle_dropped_request(request_data, process_func)
    
    async def _handle_queued_request(self, request_data: Any, process_func: Callable) -> Any:
        """Handle request with queuing"""
        # Enqueue request
        if not await self.request_queue.enqueue(request_data):
            raise Exception("Failed to enqueue request")
        
        # Process request
        try:
            result = await process_func(request_data)
            return result
        except Exception as e:
            self.circuit_breaker_count += 1
            if self.circuit_breaker_count >= self.config.circuit_breaker_threshold:
                self._open_circuit_breaker()
            raise
    
    async def _handle_throttled_request(self, request_data: Any, process_func: Callable) -> Any:
        """Handle request with throttling"""
        now = datetime.now(timezone.utc)
        
        # Check if we should throttle
        if self._should_throttle():
            # Calculate delay
            delay = self._calculate_throttle_delay()
            await asyncio.sleep(delay)
        
        # Process request
        try:
            result = await process_func(request_data)
            self.throttle_request_count += 1
            return result
        except Exception as e:
            self.circuit_breaker_count += 1
            if self.circuit_breaker_count >= self.config.circuit_breaker_threshold:
                self._open_circuit_breaker()
            raise
    
    async def _handle_circuit_breaker_request(self, request_data: Any, process_func: Callable) -> Any:
        """Handle request with circuit breaker"""
        # Process request
        try:
            result = await process_func(request_data)
            self.circuit_breaker_count = 0  # Reset on success
            return result
        except Exception as e:
            self.circuit_breaker_count += 1
            if self.circuit_breaker_count >= self.config.circuit_breaker_threshold:
                self._open_circuit_breaker()
            raise
    
    async def _handle_dropped_request(self, request_data: Any, process_func: Callable) -> Any:
        """Handle request with drop strategy"""
        # Check if we should drop
        if self.request_queue.queue.qsize() >= self.config.max_queue_size * 0.8:
            self.stats['total_dropped'] += 1
            raise Exception("Request dropped due to high load")
        
        # Process request
        try:
            result = await process_func(request_data)
            return result
        except Exception as e:
            self.circuit_breaker_count += 1
            if self.circuit_breaker_count >= self.config.circuit_breaker_threshold:
                self._open_circuit_breaker()
            raise
    
    def _should_throttle(self) -> bool:
        """Check if we should throttle requests"""
        now = datetime.now(timezone.utc)
        time_since_start = (now - self.throttle_start).total_seconds()
        
        # Throttle if we've exceeded normal rate
        expected_requests = time_since_start * (1.0 / self.config.throttle_factor)
        return self.throttle_request_count > expected_requests
    
    def _calculate_throttle_delay(self) -> float:
        """Calculate delay for throttling"""
        base_delay = 0.1  # Base delay in seconds
        return base_delay * (1.0 / self.config.throttle_factor)
    
    def _should_reset_circuit_breaker(self) -> bool:
        """Check if circuit breaker should reset"""
        now = datetime.now(timezone.utc)
        time_since_open = (now - self.last_circuit_breaker_reset).total_seconds()
        return time_since_open >= 60.0  # Reset after 1 minute
    
    def _open_circuit_breaker(self):
        """Open circuit breaker"""
        self.circuit_breaker_open = True
        self.last_circuit_breaker_reset = datetime.now(timezone.utc)
        self.logger.warning("⚠️ Circuit breaker opened due to high error rate")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_open = False
        self.circuit_breaker_count = 0
        self.logger.info("✅ Circuit breaker reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backpressure statistics"""
        queue_stats = self.request_queue.get_stats()
        
        return {
            **queue_stats,
            'strategy': self.config.strategy.value,
            'circuit_breaker_open': self.circuit_breaker_open,
            'circuit_breaker_count': self.circuit_breaker_count,
            'throttle_request_count': self.throttle_request_count,
            'throttle_factor': self.config.throttle_factor
        }

class RateLimitManager:
    """Main rate limiting and backpressure manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.backpressure_managers: Dict[str, BackpressureManager] = {}
        self.global_stats = {
            'total_requests': 0,
            'total_allowed': 0,
            'total_blocked': 0,
            'total_queued': 0,
            'total_dropped': 0
        }
    
    def add_rate_limiter(self, name: str, config: RateLimitConfig):
        """Add a rate limiter"""
        self.rate_limiters[name] = RateLimiter(config)
        self.logger.info(f"✅ Added rate limiter: {name}")
    
    def add_backpressure_manager(self, name: str, config: BackpressureConfig):
        """Add a backpressure manager"""
        self.backpressure_managers[name] = BackpressureManager(config)
        self.logger.info(f"✅ Added backpressure manager: {name}")
    
    async def execute_with_rate_limit(
        self, 
        limiter_name: str, 
        operation: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """Execute operation with rate limiting"""
        
        if limiter_name not in self.rate_limiters:
            raise Exception(f"Rate limiter '{limiter_name}' not found")
        
        limiter = self.rate_limiters[limiter_name]
        
        # Check rate limit
        if not limiter.is_allowed():
            self.global_stats['total_blocked'] += 1
            raise Exception(f"Rate limit exceeded for '{limiter_name}'")
        
        # Execute operation
        try:
            result = await operation(*args, **kwargs)
            self.global_stats['total_allowed'] += 1
            return result
        except Exception as e:
            raise
    
    async def execute_with_backpressure(
        self, 
        manager_name: str, 
        operation: Callable[..., T], 
        request_data: Any, 
        *args, 
        **kwargs
    ) -> T:
        """Execute operation with backpressure management"""
        
        if manager_name not in self.backpressure_managers:
            raise Exception(f"Backpressure manager '{manager_name}' not found")
        
        manager = self.backpressure_managers[manager_name]
        
        # Handle with backpressure
        try:
            result = await manager.handle_request(request_data, operation)
            return result
        except Exception as e:
            raise
    
    def get_rate_limit_stats(self, limiter_name: str) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        if limiter_name not in self.rate_limiters:
            return {}
        
        return self.rate_limiters[limiter_name].get_stats()
    
    def get_backpressure_stats(self, manager_name: str) -> Dict[str, Any]:
        """Get backpressure manager statistics"""
        if manager_name not in self.backpressure_managers:
            return {}
        
        return self.backpressure_managers[manager_name].get_stats()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics"""
        return {
            **self.global_stats,
            'rate_limiters': list(self.rate_limiters.keys()),
            'backpressure_managers': list(self.backpressure_managers.keys())
        }

# Global rate limit manager instance
_rate_limit_manager = None

def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager instance"""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager

# Convenience functions
async def execute_with_rate_limit(
    limiter_name: str, 
    operation: Callable[..., T], 
    *args, 
    **kwargs
) -> T:
    """Execute operation with rate limiting"""
    return await get_rate_limit_manager().execute_with_rate_limit(
        limiter_name, operation, *args, **kwargs
    )

async def execute_with_backpressure(
    manager_name: str, 
    operation: Callable[..., T], 
    request_data: Any, 
    *args, 
    **kwargs
) -> T:
    """Execute operation with backpressure management"""
    return await get_rate_limit_manager().execute_with_backpressure(
        manager_name, operation, request_data, *args, **kwargs
    )
