"""
Error Recovery System for AlphaPulse
Phase 5C: Production Features & Monitoring

Implements:
1. Circuit breaker pattern for fault tolerance
2. Retry mechanisms with exponential backoff
3. Fallback strategies and graceful degradation
4. Automated error recovery procedures
5. Error classification and handling strategies
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import functools
import traceback

# Local imports
from ..core.prefect_config import prefect_settings
from .advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class ErrorContext:
    """Error context information"""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    description: str
    timeout: float
    max_attempts: int
    backoff_factor: float = 2.0
    jitter: bool = True

class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance
    Prevents cascading failures by temporarily stopping requests to failing services
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
        logger.info(f"🔌 Circuit Breaker initialized: threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("🔄 Circuit breaker attempting reset to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("✅ Circuit breaker reset to CLOSED")
        
        self.failure_count = 0
        self.success_count += 1
    
    def _on_failure(self):
        """Handle execution failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"🚨 Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout
        }

class RetryMechanism:
    """
    Retry mechanism with exponential backoff and jitter
    Provides intelligent retry logic for transient failures
    """
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        
        logger.info(f"🔄 Retry Mechanism initialized: max_attempts={max_attempts}, base_delay={base_delay}s")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                if attempt > 1:
                    logger.info(f"✅ Operation succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"⚠️ Attempt {attempt} failed: {e}")
                
                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"⏳ Waiting {delay:.2f}s before retry {attempt + 1}")
                    await asyncio.sleep(delay)
        
        # All attempts failed
        logger.error(f"❌ Operation failed after {self.max_attempts} attempts")
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor
        
        return delay

class ErrorRecoverySystem:
    """
    Comprehensive error recovery system for the model retraining pipeline
    Implements multiple recovery strategies and automated error handling
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # Initialize default recovery strategies
        self._initialize_default_strategies()
        
        # Recovery configuration
        self.recovery_config = {
            'max_error_history': 1000,
            'auto_recovery_enabled': True,
            'notification_enabled': True,
            'escalation_threshold': 5,  # Escalate after 5 critical errors
            'escalation_cooldown': 3600,  # 1 hour cooldown
        }
        
        # Error patterns for classification
        self.error_patterns = {
            'transient': [
                'Connection refused',
                'Timeout',
                'Temporary failure',
                'Service unavailable',
                'Rate limit exceeded'
            ],
            'permanent': [
                'Authentication failed',
                'Invalid credentials',
                'Permission denied',
                'Resource not found',
                'Invalid configuration'
            ],
            'system': [
                'Out of memory',
                'Disk full',
                'CPU overload',
                'Network failure',
                'Database connection lost'
            ]
        }
        
        logger.info("🚀 Error Recovery System initialized")
    
    def _initialize_default_strategies(self):
        """Initialize default recovery strategies for common operations"""
        # Database operations
        self.recovery_actions['database_query'] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Retry database query with exponential backoff",
            timeout=30.0,
            max_attempts=3,
            backoff_factor=2.0
        )
        
        # External API calls
        self.recovery_actions['external_api'] = RecoveryAction(
            strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            description="Use circuit breaker for external API calls",
            timeout=60.0,
            max_attempts=5,
            backoff_factor=1.5
        )
        
        # Model training
        self.recovery_actions['model_training'] = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            description="Fallback to previous model version on training failure",
            timeout=300.0,
            max_attempts=2,
            backoff_factor=3.0
        )
        
        # Data processing
        self.recovery_actions['data_processing'] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Retry data processing with reduced batch size",
            timeout=120.0,
            max_attempts=3,
            backoff_factor=2.0
        )
    
    def register_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Register a circuit breaker for a specific service"""
        if name in self.circuit_breakers:
            logger.warning(f"Circuit breaker '{name}' already exists, updating configuration")
        
        self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        logger.info(f"🔌 Registered circuit breaker: {name}")
        return self.circuit_breakers[name]
    
    def register_retry_mechanism(self, name: str, **kwargs) -> RetryMechanism:
        """Register a retry mechanism for a specific operation"""
        if name in self.retry_mechanisms:
            logger.warning(f"Retry mechanism '{name}' already exists, updating configuration")
        
        self.retry_mechanisms[name] = RetryMechanism(**kwargs)
        logger.info(f"🔄 Registered retry mechanism: {name}")
        return self.retry_mechanisms[name]
    
    async def execute_with_recovery(self, 
                                  operation_name: str,
                                  func: Callable,
                                  *args,
                                  **kwargs) -> Any:
        """Execute function with appropriate recovery strategy"""
        try:
            # Get recovery strategy for operation
            strategy = self.recovery_actions.get(operation_name)
            if not strategy:
                logger.warning(f"No recovery strategy found for '{operation_name}', using default retry")
                strategy = RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Default retry strategy",
                    timeout=30.0,
                    max_attempts=3
                )
            
            # Execute based on strategy
            if strategy.strategy == RecoveryStrategy.RETRY:
                retry_mech = self.retry_mechanisms.get(operation_name)
                if not retry_mech:
                    retry_mech = RetryMechanism(
                        max_attempts=strategy.max_attempts,
                        base_delay=1.0,
                        backoff_factor=strategy.backoff_factor
                    )
                
                return await retry_mech.execute(func, *args, **kwargs)
            
            elif strategy.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                circuit_breaker = self.circuit_breakers.get(operation_name)
                if not circuit_breaker:
                    circuit_breaker = CircuitBreaker(
                        failure_threshold=5,
                        recovery_timeout=strategy.timeout
                    )
                
                return await circuit_breaker.call(func, *args, **kwargs)
            
            elif strategy.strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_with_fallback(func, *args, **kwargs)
            
            else:
                # Default to direct execution
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
        except Exception as e:
            await self._handle_error(e, operation_name, func, args, kwargs)
            raise
    
    async def _execute_with_fallback(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with fallback strategy"""
        try:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary operation failed, attempting fallback: {e}")
            
            # Try fallback operation
            try:
                fallback_result = await self._get_fallback_result(func, *args, **kwargs)
                logger.info("✅ Fallback operation succeeded")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"❌ Fallback operation also failed: {fallback_error}")
                raise e  # Re-raise original error
    
    async def _get_fallback_result(self, func: Callable, *args, **kwargs) -> Any:
        """Get fallback result for failed operation"""
        # This is a placeholder - implement specific fallback logic based on operation type
        # For example, return cached data, use previous model version, etc.
        
        if 'model_training' in str(func):
            # Return previous model version
            return {'success': False, 'fallback': True, 'message': 'Using previous model version'}
        elif 'database' in str(func):
            # Return cached data
            return {'success': False, 'fallback': True, 'message': 'Using cached data'}
        else:
            # Generic fallback
            return {'success': False, 'fallback': True, 'message': 'Generic fallback response'}
    
    async def _handle_error(self, 
                           error: Exception,
                           operation_name: str,
                           func: Callable,
                           args: tuple,
                           kwargs: dict):
        """Handle error and determine recovery action"""
        # Classify error
        error_type = self._classify_error(error)
        severity = self._determine_severity(error, error_type)
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            component=operation_name,
            operation=func.__name__,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            additional_context={
                'args': str(args),
                'kwargs': str(kwargs),
                'function_name': func.__name__
            }
        )
        
        # Record error
        self._record_error(error_context)
        
        # Log error
        await self._log_error(error_context)
        
        # Check for escalation
        if severity == ErrorSeverity.CRITICAL:
            await self._check_escalation(error_context)
        
        # Attempt auto-recovery if enabled
        if self.recovery_config['auto_recovery_enabled']:
            await self._attempt_auto_recovery(error_context)
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error based on patterns"""
        error_message = str(error).lower()
        
        for pattern_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message:
                    return pattern_type
        
        return "unknown"
    
    def _determine_severity(self, error: Exception, error_type: str) -> ErrorSeverity:
        """Determine error severity"""
        if error_type == "system":
            return ErrorSeverity.CRITICAL
        elif error_type == "permanent":
            return ErrorSeverity.HIGH
        elif error_type == "transient":
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _record_error(self, error_context: ErrorContext):
        """Record error in history"""
        self.error_history.append(error_context)
        
        # Maintain history size limit
        if len(self.error_history) > self.recovery_config['max_error_history']:
            self.error_history = self.error_history[-self.recovery_config['max_error_history']:]
    
    async def _log_error(self, error_context: ErrorContext):
        """Log error to monitoring system"""
        try:
            log_level = LogLevel.ERROR if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else LogLevel.WARNING
            
            await redis_logger.log(
                event_type=EventType.SYSTEM_ERROR,
                data={
                    'error_type': error_context.error_type,
                    'severity': error_context.severity.value,
                    'component': error_context.component,
                    'operation': error_context.operation,
                    'message': error_context.error_message,
                    'timestamp': error_context.timestamp.isoformat()
                },
                log_level=log_level
            )
        except Exception as e:
            logger.error(f"Failed to log error to monitoring system: {e}")
    
    async def _check_escalation(self, error_context: ErrorContext):
        """Check if error escalation is needed"""
        # Count critical errors in recent time window
        recent_critical_errors = [
            e for e in self.error_history
            if e.severity == ErrorSeverity.CRITICAL and
            e.timestamp > datetime.now() - timedelta(seconds=self.recovery_config['escalation_cooldown'])
        ]
        
        if len(recent_critical_errors) >= self.recovery_config['escalation_threshold']:
            await self._escalate_error(error_context)
    
    async def _escalate_error(self, error_context: ErrorContext):
        """Escalate critical error for manual intervention"""
        escalation_message = f"🚨 CRITICAL ERROR ESCALATION: {error_context.component} - {error_context.error_message}"
        logger.critical(escalation_message)
        
        # TODO: Implement escalation mechanisms (email, Slack, pager duty, etc.)
        # For now, log to Redis with critical level
        
        try:
            await redis_logger.log(
                event_type=EventType.SYSTEM_ALERT,
                data={
                    'level': 'critical',
                    'component': 'error_recovery_system',
                    'message': escalation_message,
                    'timestamp': datetime.now().isoformat(),
                    'details': {
                        'error_context': {
                            'type': error_context.error_type,
                            'severity': error_context.severity.value,
                            'component': error_context.component,
                            'operation': error_context.operation
                        }
                    }
                },
                log_level=LogLevel.ERROR
            )
        except Exception as e:
            logger.error(f"Failed to escalate error: {e}")
    
    async def _attempt_auto_recovery(self, error_context: ErrorContext):
        """Attempt automatic error recovery"""
        try:
            if error_context.error_type == "transient":
                logger.info(f"🔄 Attempting auto-recovery for transient error: {error_context.error_message}")
                # Transient errors might resolve themselves, just wait
                await asyncio.sleep(5)
                
            elif error_context.error_type == "system":
                logger.info(f"🔧 Attempting system-level recovery for: {error_context.error_message}")
                # System errors might need resource cleanup
                await self._cleanup_resources()
                
            # TODO: Implement more sophisticated auto-recovery strategies
            
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
    
    async def _cleanup_resources(self):
        """Clean up system resources"""
        try:
            # Clear any temporary files
            # Reset any stuck processes
            # Clear memory caches if needed
            logger.info("🧹 Performing resource cleanup")
            
            # This is a placeholder - implement actual cleanup logic
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            period_errors = [
                e for e in self.error_history
                if e.timestamp > cutoff_time
            ]
            
            if not period_errors:
                return {
                    'period_hours': hours,
                    'total_errors': 0,
                    'message': 'No errors in specified period',
                    'severity_distribution': {},
                    'component_distribution': {},
                    'error_type_distribution': {},
                    'recent_errors': []
                }
            
            # Count by severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = len([
                    e for e in period_errors if e.severity == severity
                ])
            
            # Count by component
            component_counts = {}
            for error in period_errors:
                component = error.component
                component_counts[component] = component_counts.get(component, 0) + 1
            
            # Count by error type
            type_counts = {}
            for error in period_errors:
                error_type = error.error_type
                type_counts[error_type] = type_counts.get(error_type, 0) + 1
            
            return {
                'period_hours': hours,
                'total_errors': len(period_errors),
                'severity_distribution': severity_counts,
                'component_distribution': component_counts,
                'error_type_distribution': type_counts,
                'recent_errors': [
                    {
                        'timestamp': e.timestamp.isoformat(),
                        'severity': e.severity.value,
                        'component': e.component,
                        'operation': e.operation,
                        'message': e.error_message
                    }
                    for e in period_errors[-10:]  # Last 10 errors
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating error summary: {e}")
            return {
                'period_hours': hours,
                'total_errors': 0,
                'error': str(e),
                'severity_distribution': {},
                'component_distribution': {},
                'error_type_distribution': {},
                'recent_errors': []
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Circuit breaker status
            circuit_breaker_status = {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            }
            
            # Retry mechanism status
            retry_mechanism_status = {
                name: {
                    'max_attempts': rm.max_attempts,
                    'base_delay': rm.base_delay,
                    'backoff_factor': rm.backoff_factor
                }
                for name, rm in self.retry_mechanisms.items()
            }
            
            # Error summary for last 24 hours
            error_summary = self.get_error_summary(24)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'circuit_breakers': circuit_breaker_status,
                'retry_mechanisms': retry_mechanism_status,
                'error_summary': error_summary,
                'recovery_config': self.recovery_config,
                'total_error_history': len(self.error_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'error': str(e)}

# Global error recovery system instance
error_recovery_system = ErrorRecoverySystem()
