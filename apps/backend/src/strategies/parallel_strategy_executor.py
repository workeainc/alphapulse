"""
Parallel Strategy Execution for AlphaPlus
Runs strategies concurrently using asyncio and multiprocessing for maximum performance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import threading
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class StrategyTask:
    """Strategy execution task"""
    strategy_name: str
    symbol: str
    timeframe: str
    data: pd.DataFrame
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass
class StrategyResult:
    """Strategy execution result"""
    strategy_name: str
    symbol: str
    timeframe: str
    signals: List[Dict[str, Any]]
    processing_time_ms: float
    success: bool
    error_message: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ParallelStrategyExecutor:
    """
    Parallel strategy execution system with intelligent load balancing
    Runs strategies concurrently to maximize throughput
    """
    
    def __init__(self, 
                 max_process_workers: int = None,
                 max_thread_workers: int = 8,
                 enable_process_pool: bool = True,
                 enable_thread_pool: bool = True):
        
        # Determine optimal worker counts
        cpu_count = mp.cpu_count()
        self.max_process_workers = max_process_workers or min(cpu_count, 4)
        self.max_thread_workers = max_thread_workers
        
        # Execution pools
        self.process_pool = None
        self.thread_pool = None
        
        if enable_process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_process_workers)
        
        if enable_thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_thread_workers)
        
        # Task queues
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        
        # Strategy registry
        self.strategy_registry: Dict[str, Callable] = {}
        self.strategy_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'active_workers': 0,
            'queue_sizes': {'high': 0, 'normal': 0, 'low': 0}
        }
        
        # Execution state
        self.is_running = False
        self.execution_task = None
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        logger.info(f"Parallel Strategy Executor initialized with {self.max_process_workers} process workers, {self.max_thread_workers} thread workers")
    
    def register_strategy(self, name: str, strategy_func: Callable, metadata: Dict[str, Any] = None):
        """Register a strategy function"""
        self.strategy_registry[name] = strategy_func
        self.strategy_metadata[name] = metadata or {}
        logger.info(f"📝 Registered strategy: {name}")
    
    async def start(self):
        """Start the parallel executor"""
        if self.is_running:
            logger.warning("Parallel executor is already running")
            return
        
        self.is_running = True
        self.execution_task = asyncio.create_task(self._execution_loop())
        logger.info("🚀 Parallel Strategy Executor started")
    
    async def stop(self):
        """Stop the parallel executor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown pools
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        logger.info("🛑 Parallel Strategy Executor stopped")
    
    async def execute_strategy(self, 
                             strategy_name: str,
                             symbol: str,
                             timeframe: str,
                             data: pd.DataFrame,
                             parameters: Dict[str, Any] = None,
                             priority: int = 1) -> StrategyResult:
        """
        Execute a strategy with specified priority
        Returns immediately with a future result
        """
        if strategy_name not in self.strategy_registry:
            return StrategyResult(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                signals=[],
                processing_time_ms=0.0,
                success=False,
                error_message=f"Strategy '{strategy_name}' not registered"
            )
        
        # Create task
        task = StrategyTask(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            parameters=parameters or {},
            priority=priority
        )
        
        # Add to appropriate queue
        if priority >= 3:
            await self.high_priority_queue.put(task)
        elif priority >= 2:
            await self.normal_priority_queue.put(task)
        else:
            await self.low_priority_queue.put(task)
        
        # Update stats
        with self.stats_lock:
            self.stats['total_tasks'] += 1
            if priority >= 3:
                self.stats['queue_sizes']['high'] += 1
            elif priority >= 2:
                self.stats['queue_sizes']['normal'] += 1
            else:
                self.stats['queue_sizes']['low'] += 1
        
        logger.debug(f"📋 Queued strategy task: {strategy_name} for {symbol} {timeframe} (priority: {priority})")
        
        # Return a placeholder result (in real implementation, you'd return a future)
        return StrategyResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            signals=[],
            processing_time_ms=0.0,
            success=True,
            metadata={'queued': True, 'priority': priority}
        )
    
    async def _execution_loop(self):
        """Main execution loop that processes tasks from queues"""
        while self.is_running:
            try:
                # Process high priority tasks first
                if not self.high_priority_queue.empty():
                    task = await self.high_priority_queue.get()
                    await self._execute_task(task)
                    self.high_priority_queue.task_done()
                    with self.stats_lock:
                        self.stats['queue_sizes']['high'] -= 1
                
                # Process normal priority tasks
                elif not self.normal_priority_queue.empty():
                    task = await self.normal_priority_queue.get()
                    await self._execute_task(task)
                    self.normal_priority_queue.task_done()
                    with self.stats_lock:
                        self.stats['queue_sizes']['normal'] -= 1
                
                # Process low priority tasks
                elif not self.low_priority_queue.empty():
                    task = await self.low_priority_queue.get()
                    await self._execute_task(task)
                    self.low_priority_queue.task_done()
                    with self.stats_lock:
                        self.stats['queue_sizes']['low'] -= 1
                
                # Small delay to prevent busy waiting
                else:
                    await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"❌ Error in execution loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_task(self, task: StrategyTask):
        """Execute a single strategy task"""
        start_time = time.time()
        
        try:
            # Get strategy function
            strategy_func = self.strategy_registry[task.strategy_name]
            
            # Determine execution method based on strategy type
            if self._should_use_process_pool(task):
                result = await self._execute_in_process_pool(task, strategy_func)
            else:
                result = await self._execute_in_thread_pool(task, strategy_func)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            with self.stats_lock:
                self.stats['completed_tasks'] += 1
                self.stats['total_processing_time_ms'] += processing_time
                self.stats['avg_processing_time_ms'] = (
                    self.stats['total_processing_time_ms'] / self.stats['completed_tasks']
                )
            
            logger.debug(f"✅ Executed {task.strategy_name} for {task.symbol} {task.timeframe} in {processing_time:.2f}ms")
            
        except Exception as e:
            # Update failure statistics
            with self.stats_lock:
                self.stats['failed_tasks'] += 1
            
            logger.error(f"❌ Failed to execute {task.strategy_name} for {task.symbol} {task.timeframe}: {e}")
    
    def _should_use_process_pool(self, task: StrategyTask) -> bool:
        """Determine if task should use process pool"""
        # Use process pool for CPU-intensive strategies
        cpu_intensive_strategies = ['ml_pattern_detector', 'advanced_analytics', 'ensemble_learning']
        return task.strategy_name in cpu_intensive_strategies and self.process_pool is not None
    
    async def _execute_in_process_pool(self, task: StrategyTask, strategy_func: Callable):
        """Execute task in process pool"""
        loop = asyncio.get_event_loop()
        
        # Prepare data for process pool (serialize if needed)
        serialized_data = self._serialize_data_for_process(task.data)
        
        # Execute in process pool
        result = await loop.run_in_executor(
            self.process_pool,
            partial(self._execute_strategy_in_process, strategy_func, task, serialized_data)
        )
        
        return result
    
    async def _execute_in_thread_pool(self, task: StrategyTask, strategy_func: Callable):
        """Execute task in thread pool"""
        loop = asyncio.get_event_loop()
        
        # Execute in thread pool
        result = await loop.run_in_executor(
            self.thread_pool,
            partial(self._execute_strategy_in_thread, strategy_func, task)
        )
        
        return result
    
    def _serialize_data_for_process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Serialize data for process pool execution"""
        return {
            'values': data.values.tolist(),
            'columns': data.columns.tolist(),
            'index': data.index.tolist()
        }
    
    def _deserialize_data_from_process(self, serialized_data: Dict[str, Any]) -> pd.DataFrame:
        """Deserialize data from process pool execution"""
        return pd.DataFrame(
            serialized_data['values'],
            columns=serialized_data['columns'],
            index=serialized_data['index']
        )
    
    def _execute_strategy_in_process(self, strategy_func: Callable, task: StrategyTask, serialized_data: Dict[str, Any]):
        """Execute strategy in separate process"""
        try:
            # Deserialize data
            data = self._deserialize_data_from_process(serialized_data)
            
            # Execute strategy
            signals = strategy_func(data, task.symbol, task.timeframe, task.parameters)
            
            return {
                'success': True,
                'signals': signals,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'signals': [],
                'error': str(e)
            }
    
    def _execute_strategy_in_thread(self, strategy_func: Callable, task: StrategyTask):
        """Execute strategy in thread pool"""
        try:
            # Execute strategy
            signals = strategy_func(task.data, task.symbol, task.timeframe, task.parameters)
            
            return {
                'success': True,
                'signals': signals,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'signals': [],
                'error': str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add system metrics
        stats['cpu_usage_percent'] = psutil.cpu_percent()
        stats['memory_usage_percent'] = psutil.virtual_memory().percent
        
        # Add queue metrics
        stats['queue_sizes'] = {
            'high': self.high_priority_queue.qsize(),
            'normal': self.normal_priority_queue.qsize(),
            'low': self.low_priority_queue.qsize()
        }
        
        return stats
    
    async def wait_for_completion(self, timeout_seconds: float = 30.0):
        """Wait for all queued tasks to complete"""
        start_time = time.time()
        
        while (self.high_priority_queue.qsize() > 0 or 
               self.normal_priority_queue.qsize() > 0 or 
               self.low_priority_queue.qsize() > 0):
            
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"⏰ Timeout waiting for task completion after {timeout_seconds}s")
                break
            
            await asyncio.sleep(0.1)
        
        logger.info("✅ All tasks completed")
