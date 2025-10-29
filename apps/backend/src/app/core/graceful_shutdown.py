#!/usr/bin/env python3
"""
Graceful Shutdown Management for AlphaPulse
Provides proper cleanup, signal handling, and graceful service termination
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
import threading
import weakref
import contextlib

logger = logging.getLogger(__name__)

class ShutdownState(Enum):
    """Shutdown states"""
    RUNNING = "running"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    FORCE_SHUTDOWN = "force_shutdown"

@dataclass
class ShutdownConfig:
    """Configuration for graceful shutdown"""
    shutdown_timeout: float = 30.0        # Time to wait for graceful shutdown
    force_shutdown_timeout: float = 10.0  # Time to wait before force shutdown
    cleanup_timeout: float = 15.0         # Time to wait for cleanup operations
    signal_handling: bool = True          # Enable signal handling
    log_shutdown_progress: bool = True    # Log shutdown progress
    cleanup_order: List[str] = None       # Order of cleanup operations

@dataclass
class ShutdownTask:
    """Shutdown task information"""
    name: str
    cleanup_func: Callable
    priority: int = 0                     # Lower number = higher priority
    timeout: float = 10.0                 # Timeout for this task
    critical: bool = False                # Critical task that must complete
    description: str = ""

class GracefulShutdownManager:
    """Manages graceful shutdown of the application"""
    
    def __init__(self, config: ShutdownConfig = None):
        self.config = config or ShutdownConfig()
        self.logger = logging.getLogger(__name__)
        
        # Shutdown state
        self.state = ShutdownState.RUNNING
        self.shutdown_start_time = None
        self.shutdown_request_time = None
        
        # Shutdown tasks
        self.shutdown_tasks: Dict[str, ShutdownTask] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # Signal handling
        self.original_signal_handlers = {}
        self.signal_handlers_registered = False
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
        self.shutdown_lock = asyncio.Lock()
        
        # Statistics
        self.shutdown_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'shutdown_duration': 0.0,
            'cleanup_duration': 0.0
        }
        
        # Register default cleanup tasks
        self._register_default_tasks()
        
        # Register signal handlers if enabled
        if self.config.signal_handling:
            self._register_signal_handlers()
    
    def _register_default_tasks(self):
        """Register default cleanup tasks"""
        # Database connections
        self.register_shutdown_task(
            "database_connections",
            self._cleanup_database_connections,
            priority=1,
            critical=True,
            description="Close database connections and pools"
        )
        
        # Background tasks
        self.register_shutdown_task(
            "background_tasks",
            self._cleanup_background_tasks,
            priority=2,
            critical=False,
            description="Cancel and cleanup background tasks"
        )
        
        # File handles
        self.register_shutdown_task(
            "file_handles",
            self._cleanup_file_handles,
            priority=3,
            critical=False,
            description="Close open file handles"
        )
        
        # Network connections
        self.register_shutdown_task(
            "network_connections",
            self._cleanup_network_connections,
            priority=4,
            critical=False,
            description="Close network connections"
        )
        
        # Logging
        self.register_shutdown_task(
            "logging",
            self._cleanup_logging,
            priority=5,
            critical=False,
            description="Flush and cleanup logging"
        )
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        try:
            # Store original handlers
            self.original_signal_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
            self.original_signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Windows-specific signals
            if hasattr(signal, 'SIGBREAK'):
                self.original_signal_handlers[signal.SIGBREAK] = signal.signal(signal.SIGBREAK, self._signal_handler)
            
            self.signal_handlers_registered = True
            self.logger.info("✅ Signal handlers registered for graceful shutdown")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        self.logger.info(f"🔄 Received signal {signal_name}, initiating graceful shutdown...")
        
        # Request shutdown asynchronously
        asyncio.create_task(self.request_shutdown())
    
    def register_shutdown_task(
        self, 
        name: str, 
        cleanup_func: Callable, 
        priority: int = 0,
        timeout: float = 10.0,
        critical: bool = False,
        description: str = ""
    ):
        """Register a shutdown task"""
        if name in self.shutdown_tasks:
            self.logger.warning(f"⚠️ Shutdown task '{name}' already registered, overwriting")
        
        task = ShutdownTask(
            name=name,
            cleanup_func=cleanup_func,
            priority=priority,
            timeout=timeout,
            critical=critical,
            description=description
        )
        
        self.shutdown_tasks[name] = task
        self.shutdown_stats['total_tasks'] = len(self.shutdown_tasks)
        
        self.logger.info(f"✅ Registered shutdown task: {name} (priority: {priority})")
    
    def unregister_shutdown_task(self, name: str):
        """Unregister a shutdown task"""
        if name in self.shutdown_tasks:
            del self.shutdown_tasks[name]
            self.shutdown_stats['total_tasks'] = len(self.shutdown_tasks)
            self.logger.info(f"✅ Unregistered shutdown task: {name}")
    
    async def request_shutdown(self):
        """Request graceful shutdown"""
        async with self.shutdown_lock:
            if self.state != ShutdownState.RUNNING:
                return
            
            self.state = ShutdownState.SHUTDOWN_REQUESTED
            self.shutdown_request_time = datetime.now(timezone.utc)
            self.shutdown_event.set()
            
            self.logger.info("🔄 Graceful shutdown requested")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to be requested"""
        await self.shutdown_event.wait()
    
    async def shutdown(self, timeout: float = None):
        """Execute graceful shutdown"""
        async with self.shutdown_lock:
            if self.state in [ShutdownState.SHUTTING_DOWN, ShutdownState.SHUTDOWN_COMPLETE]:
                return
            
            timeout = timeout or self.config.shutdown_timeout
            self.state = ShutdownState.SHUTTING_DOWN
            self.shutdown_start_time = datetime.now(timezone.utc)
            
            self.logger.info(f"🔄 Starting graceful shutdown (timeout: {timeout}s)")
            
            try:
                # Execute shutdown tasks
                await self._execute_shutdown_tasks(timeout)
                
                # Mark shutdown complete
                self.state = ShutdownState.SHUTDOWN_COMPLETE
                self._update_shutdown_stats()
                
                self.logger.info("✅ Graceful shutdown completed successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Error during shutdown: {e}")
                
                # Force shutdown if critical tasks failed
                if self._has_critical_failures():
                    self.logger.warning("⚠️ Critical tasks failed, forcing shutdown")
                    await self._force_shutdown()
    
    async def _execute_shutdown_tasks(self, timeout: float):
        """Execute all shutdown tasks in priority order"""
        start_time = time.time()
        
        # Sort tasks by priority
        sorted_tasks = sorted(
            self.shutdown_tasks.values(),
            key=lambda task: task.priority
        )
        
        self.logger.info(f"🔄 Executing {len(sorted_tasks)} shutdown tasks")
        
        for task in sorted_tasks:
            if time.time() - start_time > timeout:
                self.logger.warning(f"⚠️ Shutdown timeout reached, stopping task execution")
                break
            
            try:
                await self._execute_single_task(task)
                
            except Exception as e:
                self.logger.error(f"❌ Task '{task.name}' failed: {e}")
                self.failed_tasks.append(task.name)
                
                if task.critical:
                    raise Exception(f"Critical task '{task.name}' failed: {e}")
        
        # Wait for remaining time if any
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time > 0:
            self.logger.info(f"⏳ Waiting {remaining_time:.1f}s for remaining operations...")
            await asyncio.sleep(remaining_time)
    
    async def _execute_single_task(self, task: ShutdownTask):
        """Execute a single shutdown task"""
        self.logger.info(f"🔄 Executing task: {task.name} - {task.description}")
        
        start_time = time.time()
        
        try:
            # Execute cleanup function with timeout
            if asyncio.iscoroutinefunction(task.cleanup_func):
                await asyncio.wait_for(task.cleanup_func(), timeout=task.timeout)
            else:
                # For sync functions, run in thread pool
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, task.cleanup_func),
                    timeout=task.timeout
                )
            
            execution_time = time.time() - start_time
            self.completed_tasks.append(task.name)
            self.shutdown_stats['completed_tasks'] += 1
            
            self.logger.info(f"✅ Task '{task.name}' completed in {execution_time:.3f}s")
            
        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Task '{task.name}' timed out after {task.timeout}s")
            raise
        except Exception as e:
            self.logger.error(f"❌ Task '{task.name}' failed: {e}")
            raise
    
    async def _force_shutdown(self):
        """Force shutdown when graceful shutdown fails"""
        self.state = ShutdownState.FORCE_SHUTDOWN
        self.logger.warning("⚠️ Force shutdown initiated")
        
        # Cancel all remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to cancel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("✅ Force shutdown completed")
    
    def _has_critical_failures(self) -> bool:
        """Check if any critical tasks failed"""
        for task_name in self.failed_tasks:
            task = self.shutdown_tasks.get(task_name)
            if task and task.critical:
                return True
        return False
    
    def _update_shutdown_stats(self):
        """Update shutdown statistics"""
        if self.shutdown_start_time:
            self.shutdown_stats['shutdown_duration'] = (
                datetime.now(timezone.utc) - self.shutdown_start_time
            ).total_seconds()
        
        self.shutdown_stats['failed_tasks'] = len(self.failed_tasks)
    
    # Default cleanup functions
    async def _cleanup_database_connections(self):
        """Cleanup database connections"""
        try:
            # Import and cleanup enhanced connection if available
            try:
                from src.app.database.enhanced_connection import get_enhanced_connection
                connection = get_enhanced_connection()
                await connection.close()
                self.logger.info("✅ Database connections closed")
            except ImportError:
                self.logger.info("ℹ️ No enhanced database connection to cleanup")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up database connections: {e}")
            raise
    
    async def _cleanup_background_tasks(self):
        """Cleanup background tasks"""
        try:
            # Cancel all background tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.logger.info(f"✅ Cancelled {len(tasks)} background tasks")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up background tasks: {e}")
            raise
    
    async def _cleanup_file_handles(self):
        """Cleanup file handles"""
        try:
            # This is a placeholder - in a real application, you'd track file handles
            self.logger.info("✅ File handles cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up file handles: {e}")
            raise
    
    async def _cleanup_network_connections(self):
        """Cleanup network connections"""
        try:
            # This is a placeholder - in a real application, you'd track network connections
            self.logger.info("✅ Network connections cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up network connections: {e}")
            raise
    
    async def _cleanup_logging(self):
        """Cleanup logging"""
        try:
            # Flush all loggers
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            
            self.logger.info("✅ Logging cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up logging: {e}")
            raise
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status"""
        return {
            'state': self.state.value,
            'shutdown_request_time': self.shutdown_request_time.isoformat() if self.shutdown_request_time else None,
            'shutdown_start_time': self.shutdown_start_time.isoformat() if self.shutdown_start_time else None,
            'total_tasks': self.shutdown_stats['total_tasks'],
            'completed_tasks': self.shutdown_stats['completed_tasks'],
            'failed_tasks': self.shutdown_stats['failed_tasks'],
            'shutdown_duration': self.shutdown_stats['shutdown_duration'],
            'registered_tasks': list(self.shutdown_tasks.keys()),
            'completed_task_names': self.completed_tasks,
            'failed_task_names': self.failed_tasks
        }
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self.state != ShutdownState.RUNNING
    
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown is complete"""
        return self.state == ShutdownState.SHUTDOWN_COMPLETE
    
    async def cleanup(self):
        """Cleanup the shutdown manager"""
        try:
            # Restore original signal handlers
            if self.signal_handlers_registered:
                for signum, handler in self.original_signal_handlers.items():
                    signal.signal(signum, handler)
                
                self.signal_handlers_registered = False
                self.logger.info("✅ Signal handlers restored")
            
            # Clear shutdown event
            self.shutdown_event.clear()
            
            self.logger.info("✅ Shutdown manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up shutdown manager: {e}")

# Global shutdown manager instance
_shutdown_manager = None

def get_shutdown_manager() -> GracefulShutdownManager:
    """Get the global shutdown manager instance"""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdownManager()
    return _shutdown_manager

async def request_shutdown():
    """Request graceful shutdown"""
    manager = get_shutdown_manager()
    await manager.request_shutdown()

async def wait_for_shutdown():
    """Wait for shutdown to be requested"""
    manager = get_shutdown_manager()
    await manager.wait_for_shutdown()

async def shutdown(timeout: float = None):
    """Execute graceful shutdown"""
    manager = get_shutdown_manager()
    await manager.shutdown(timeout)

def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested"""
    manager = get_shutdown_manager()
    return manager.is_shutdown_requested()

def is_shutdown_complete() -> bool:
    """Check if shutdown is complete"""
    manager = get_shutdown_manager()
    return manager.is_shutdown_complete()

# Context manager for graceful shutdown
@contextlib.asynccontextmanager
async def graceful_shutdown_context(timeout: float = None):
    """Context manager for graceful shutdown"""
    manager = get_shutdown_manager()
    
    try:
        yield manager
    finally:
        if manager.is_shutdown_requested():
            await manager.shutdown(timeout)
        await manager.cleanup()
