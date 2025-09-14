"""
Backpressure Handler for AlphaPulse
Manages stream backpressure and flow control
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class BackpressureMetrics:
    """Backpressure performance metrics"""
    messages_queued: int = 0
    messages_dropped: int = 0
    messages_processed: int = 0
    avg_queue_time_ms: float = 0.0
    max_queue_size: int = 0
    backpressure_threshold_hit: int = 0

class BackpressureHandler:
    """
    Backpressure handler for managing stream flow control
    
    Features:
    - Queue management with configurable limits
    - Message dropping when backpressure threshold is exceeded
    - Flow control and rate limiting
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Backpressure settings
        self.max_queue_size = self.config.get('max_queue_size', 10000)
        self.backpressure_threshold = self.config.get('backpressure_threshold', 0.8)  # 80% of max queue
        self.drop_threshold = self.config.get('drop_threshold', 0.95)  # 95% of max queue
        self.rate_limit_mps = self.config.get('rate_limit_mps', 1000)  # messages per second
        
        # Performance settings
        self.processing_timeout = self.config.get('processing_timeout', 5.0)  # seconds
        self.cleanup_interval = self.config.get('cleanup_interval', 60)  # seconds
        
        # State management
        self.is_running = False
        self.metrics = BackpressureMetrics()
        self.message_queue = deque(maxlen=self.max_queue_size)
        self.processing_semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
        self.last_processed_time = datetime.now(timezone.utc)
        
        # Processing callbacks
        self.process_callbacks = []
        self.drop_callbacks = []
        
        logger.info("BackpressureHandler initialized")
    
    async def initialize(self):
        """Initialize the backpressure handler"""
        try:
            self.is_running = True
            logger.info("✅ BackpressureHandler initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BackpressureHandler: {e}")
            raise
    
    async def enqueue_message(self, message: Any, priority: int = 0) -> bool:
        """
        Enqueue a message with backpressure handling
        
        Args:
            message: Message to enqueue
            priority: Message priority (higher = higher priority)
            
        Returns:
            True if message was enqueued, False if dropped
        """
        try:
            current_queue_size = len(self.message_queue)
            
            # Check if we should drop messages due to backpressure
            if current_queue_size >= self.max_queue_size * self.drop_threshold:
                self.metrics.messages_dropped += 1
                self.metrics.backpressure_threshold_hit += 1
                await self._trigger_drop_callbacks(message, "queue_full")
                logger.warning(f"Message dropped due to queue full: {current_queue_size}/{self.max_queue_size}")
                return False
            
            # Check rate limiting
            current_time = datetime.now(timezone.utc)
            time_diff = (current_time - self.last_processed_time).total_seconds()
            
            if time_diff < 1.0 / self.rate_limit_mps:
                self.metrics.messages_dropped += 1
                await self._trigger_drop_callbacks(message, "rate_limit")
                logger.warning(f"Message dropped due to rate limit: {self.rate_limit_mps} mps")
                return False
            
            # Enqueue message
            self.message_queue.append({
                'message': message,
                'priority': priority,
                'timestamp': current_time
            })
            
            self.metrics.messages_queued += 1
            self.metrics.max_queue_size = max(self.metrics.max_queue_size, len(self.message_queue))
            
            # Check backpressure threshold
            if current_queue_size >= self.max_queue_size * self.backpressure_threshold:
                self.metrics.backpressure_threshold_hit += 1
                logger.warning(f"Backpressure threshold hit: {current_queue_size}/{self.max_queue_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error enqueueing message: {e}")
            return False
    
    async def process_messages(self, batch_size: int = 100) -> List[Any]:
        """
        Process messages from the queue
        
        Args:
            batch_size: Number of messages to process
            
        Returns:
            List of processed messages
        """
        processed_messages = []
        
        try:
            async with self.processing_semaphore:
                # Get messages to process
                messages_to_process = []
                for _ in range(min(batch_size, len(self.message_queue))):
                    if self.message_queue:
                        messages_to_process.append(self.message_queue.popleft())
                
                # Process messages
                for msg_data in messages_to_process:
                    message = msg_data['message']
                    enqueue_time = msg_data['timestamp']
                    
                    # Calculate queue time
                    queue_time = (datetime.now(timezone.utc) - enqueue_time).total_seconds() * 1000
                    self.metrics.avg_queue_time_ms = (
                        (self.metrics.avg_queue_time_ms * (self.metrics.messages_processed) + queue_time) /
                        (self.metrics.messages_processed + 1)
                    )
                    
                    # Process message
                    try:
                        await self._process_message(message)
                        processed_messages.append(message)
                        self.metrics.messages_processed += 1
                        self.last_processed_time = datetime.now(timezone.utc)
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        self.metrics.messages_dropped += 1
                        await self._trigger_drop_callbacks(message, "processing_error")
            
        except Exception as e:
            logger.error(f"Error in process_messages: {e}")
        
        return processed_messages
    
    async def _process_message(self, message: Any):
        """Process a single message"""
        for callback in self.process_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Error in process callback: {e}")
    
    async def _trigger_drop_callbacks(self, message: Any, reason: str):
        """Trigger drop callbacks"""
        for callback in self.drop_callbacks:
            try:
                await callback(message, reason)
            except Exception as e:
                logger.error(f"Error in drop callback: {e}")
    
    def add_process_callback(self, callback: Callable):
        """Add message processing callback"""
        self.process_callbacks.append(callback)
    
    def add_drop_callback(self, callback: Callable):
        """Add message drop callback"""
        self.drop_callbacks.append(callback)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'queue_size': len(self.message_queue),
            'max_queue_size': self.max_queue_size,
            'backpressure_threshold': self.backpressure_threshold,
            'drop_threshold': self.drop_threshold,
            'rate_limit_mps': self.rate_limit_mps,
            'is_backpressure_active': len(self.message_queue) >= self.max_queue_size * self.backpressure_threshold
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get backpressure metrics"""
        return {
            'is_running': self.is_running,
            'messages_queued': self.metrics.messages_queued,
            'messages_dropped': self.metrics.messages_dropped,
            'messages_processed': self.metrics.messages_processed,
            'avg_queue_time_ms': self.metrics.avg_queue_time_ms,
            'max_queue_size': self.metrics.max_queue_size,
            'backpressure_threshold_hit': self.metrics.backpressure_threshold_hit,
            'current_queue_size': len(self.message_queue)
        }
    
    async def shutdown(self):
        """Shutdown the backpressure handler"""
        self.is_running = False
        
        # Process remaining messages
        remaining_messages = await self.process_messages(batch_size=len(self.message_queue))
        logger.info(f"Processed {len(remaining_messages)} remaining messages during shutdown")
        
        logger.info("🛑 BackpressureHandler shutdown complete")

# Global instance
backpressure_handler = BackpressureHandler()
