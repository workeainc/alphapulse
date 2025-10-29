"""
Phase 4 Integration for AlphaPulse
Advanced Logging & Feedback System Integration
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from .advanced_logging_system import redis_logger, LogLevel, EventType
from .ensemble_analyzer import ensemble_analyzer
from .walk_forward_optimizer import walk_forward_optimizer

logger = logging.getLogger(__name__)

class Phase4Integration:
    """Main integration class for Phase 4: Advanced Logging & Feedback"""
    
    def __init__(self):
        self.is_running = False
        self.analysis_task = None
        self.optimization_task = None
        
        logger.info("Phase 4 Integration initialized")
    
    async def start(self):
        """Start the Phase 4 system"""
        if self.is_running:
            return
        
        try:
            # Start Redis logger
            await redis_logger.start()
            
            # Start background tasks
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            self.is_running = True
            logger.info("🚀 Phase 4 Advanced Logging & Feedback System started")
            
        except Exception as e:
            logger.error(f"Error starting Phase 4 system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the Phase 4 system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop background tasks
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        await redis_logger.stop()
        logger.info("🛑 Phase 4 Advanced Logging & Feedback System stopped")
    
    async def log_event(self,
                       event_type: EventType,
                       data: Dict[str, Any],
                       log_level: LogLevel = LogLevel.INFO,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log an event with full Phase 4 analysis pipeline"""
        try:
            # Log to Redis
            entry_id = await redis_logger.log(event_type, data, log_level, metadata)
            
            # Create log data for analysis
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type.value,
                'log_level': log_level.value,
                'data': data,
                'metadata': metadata or {}
            }
            
            # Perform ensemble analysis
            anomaly_result = await ensemble_analyzer.detect_anomaly(log_data)
            classification_result = await ensemble_analyzer.classify_event(log_data)
            
            # Add analysis results to metadata
            analysis_metadata = {
                'anomaly_detection': {
                    'is_anomaly': anomaly_result.is_anomaly,
                    'anomaly_score': anomaly_result.anomaly_score,
                    'confidence': anomaly_result.confidence
                },
                'ensemble_classification': {
                    'prediction': classification_result.prediction,
                    'confidence': classification_result.confidence,
                    'ensemble_method': classification_result.ensemble_method
                }
            }
            
            # Update metadata with analysis results
            if metadata is None:
                metadata = {}
            metadata.update(analysis_metadata)
            
            # Add to training data if we have labels
            if 'label' in metadata:
                ensemble_analyzer.add_training_data(log_data, metadata['label'])
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Error in Phase 4 log_event: {e}")
            return ""
    
    async def _analysis_loop(self):
        """Background task for periodic model analysis and retraining"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Retrain ensemble models
                retrain_result = await ensemble_analyzer.retrain_models()
                logger.info(f"Phase 4 Model retraining result: {retrain_result}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Phase 4 analysis loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _optimization_loop(self):
        """Background task for periodic walk-forward optimization"""
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Get training data
                training_data = ensemble_analyzer.training_data
                
                if len(training_data) >= 1000:
                    # Perform walk-forward optimization
                    optimization_result = await walk_forward_optimizer.walk_forward_optimization(training_data)
                    logger.info(f"Phase 4 Walk-forward optimization result: {optimization_result}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Phase 4 optimization loop: {e}")
                await asyncio.sleep(3600)  # Wait before retrying
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive Phase 4 system statistics"""
        stats = {
            'phase4_running': self.is_running,
            'redis_logger': redis_logger.get_stats(),
            'ensemble_analyzer': ensemble_analyzer.get_analyzer_stats(),
            'walk_forward_optimizer': walk_forward_optimizer.get_optimization_summary()
        }
        
        return stats

# Global Phase 4 integration instance
phase4_integration = Phase4Integration()
