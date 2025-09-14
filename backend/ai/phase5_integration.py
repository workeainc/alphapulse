"""
Phase 5 Integration for AlphaPulse
System Integration: Low-latency streaming, monitoring, and production deployment
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from .streaming_pipeline import streaming_pipeline
from .monitoring_autoscaling import monitoring_autoscaling

logger = logging.getLogger(__name__)

class Phase5Integration:
    """Main integration class for Phase 5: System Integration"""
    
    def __init__(self):
        self.is_running = False
        self.streaming_task = None
        self.monitoring_task = None
        
        logger.info("Phase 5 Integration initialized")
    
    async def start(self):
        """Start the Phase 5 system"""
        if self.is_running:
            return
        
        try:
            # Start streaming pipeline
            await streaming_pipeline.start()
            
            # Start monitoring and auto-scaling
            await monitoring_autoscaling.start()
            
            # Start background tasks
            self.streaming_task = asyncio.create_task(self._streaming_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.is_running = True
            logger.info("🚀 Phase 5 System Integration started")
            
        except Exception as e:
            logger.error(f"Error starting Phase 5 system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the Phase 5 system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop background tasks
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await streaming_pipeline.stop()
        await monitoring_autoscaling.stop()
        
        logger.info("🛑 Phase 5 System Integration stopped")
    
    async def publish_signal(self, signal_data: Dict[str, Any]) -> str:
        """Publish trading signal to streaming pipeline"""
        try:
            message_id = await streaming_pipeline.publish_signal(signal_data)
            logger.info(f"Signal published to streaming pipeline: {message_id}")
            return message_id
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
            return ""
    
    async def _streaming_loop(self):
        """Background task for streaming operations"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get streaming stats
                stats = streaming_pipeline.get_pipeline_stats()
                logger.debug(f"Streaming stats: {stats}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitoring_loop(self):
        """Background task for monitoring operations"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get monitoring status
                status = monitoring_autoscaling.get_system_status()
                logger.debug(f"Monitoring status: {status}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive Phase 5 system statistics"""
        stats = {
            'phase5_running': self.is_running,
            'streaming_pipeline': streaming_pipeline.get_pipeline_stats(),
            'monitoring_autoscaling': monitoring_autoscaling.get_system_status()
        }
        
        return stats

# Global Phase 5 integration instance
phase5_integration = Phase5Integration()
