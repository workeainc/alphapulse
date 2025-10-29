#!/usr/bin/env python3
"""
Hard Example Integration Service for AlphaPulse
Phase 5C: Integration with Existing Infrastructure

Integrates the hard example buffer with:
1. Existing retraining pipeline
2. Model registry and monitoring
3. Performance tracking and alerts
4. Automated scheduling
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import time

# Local imports
from .hard_example_buffer_service import (
    HardExampleBufferService, 
    TradeOutcome, 
    BufferStats,
    hard_example_buffer_service
)
from ..src.database.data_versioning_dao import DataVersioningDAO
from ..src.database.connection import get_enhanced_connection

# AI imports
try:
    from ..src.ai.retraining import RetrainingOrchestrator
    from ..src.ai.model_registry import ModelRegistry
    from ..src.ai.advanced_logging_system import redis_logger, EventType, LogLevel
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Prefect imports
try:
    from prefect import task, flow, get_run_logger
    from prefect.schedules import CronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

logger = logging.getLogger(__name__)

class HardExampleIntegrationService:
    """
    Service for integrating hard example buffer with existing infrastructure
    Manages the complete workflow from outcome computation to retraining
    """
    
    def __init__(self):
        self.buffer_service = hard_example_buffer_service
        self.db_connection = get_enhanced_connection()
        # DAO will be created with session when needed
        
        # AI services (if available)
        self.model_orchestrator = None
        self.model_registry = None
        
        if AI_AVAILABLE:
            try:
                self.model_orchestrator = RetrainingOrchestrator()
                self.model_registry = ModelRegistry()
                logger.info("✅ AI services integrated")
            except Exception as e:
                logger.warning(f"⚠️ AI services not available: {e}")
        
        # Integration configuration
        self.integration_config = {
            'auto_trigger_retraining': True,
            'min_hard_examples_for_retrain': 100,
            'retrain_threshold_ratio': 0.15,  # 15% of buffer size
            'performance_alert_threshold': 5.0,  # 5 seconds
            'buffer_imbalance_alert_threshold': 0.1,  # 10% deviation
        }
        
        # Performance tracking
        self.integration_metrics = {
            'total_workflows_executed': 0,
            'total_retraining_triggers': 0,
            'avg_workflow_time': 0.0,
            'last_workflow_time': 0.0,
            'last_workflow_status': 'unknown'
        }
        
        logger.info("🚀 Hard Example Integration Service initialized")
    
    async def execute_complete_workflow(self, 
                                      symbols: List[str] = None,
                                      force_retrain: bool = False) -> Dict[str, Any]:
        """
        Execute the complete hard example workflow
        From outcome computation to retraining integration
        """
        workflow_start = time.time()
        logger.info("🚀 Starting complete hard example workflow")
        
        try:
            workflow_results = {
                'workflow_id': f"workflow_{int(time.time())}",
                'start_time': datetime.now(),
                'symbols': symbols,
                'steps_completed': [],
                'errors': [],
                'metrics': {}
            }
            
            # Step 1: Compute trade outcomes
            logger.info("📊 Step 1: Computing trade outcomes...")
            try:
                outcomes = await self.buffer_service.compute_trade_outcomes(symbols)
                workflow_results['steps_completed'].append('outcome_computation')
                workflow_results['metrics']['outcomes_computed'] = len(outcomes)
                logger.info(f"✅ Computed {len(outcomes)} outcomes")
            except Exception as e:
                error_msg = f"Outcome computation failed: {e}"
                workflow_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                raise
            
            if not outcomes:
                logger.info("✅ No outcomes to process, workflow complete")
                return workflow_results
            
            # Step 2: Categorize hard examples
            logger.info("🏷️ Step 2: Categorizing hard examples...")
            try:
                categorized_outcomes = await self.buffer_service.categorize_hard_examples(outcomes)
                workflow_results['steps_completed'].append('hard_example_categorization')
                workflow_results['metrics']['hard_examples_captured'] = len([o for o in categorized_outcomes if o.buffer_type])
                logger.info(f"✅ Categorized hard examples")
            except Exception as e:
                error_msg = f"Hard example categorization failed: {e}"
                workflow_results['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                raise
            
            # Step 3: Maintain buffer balance
            logger.info("⚖️ Step 3: Maintaining buffer balance...")
            try:
                await self.buffer_service.maintain_buffer_balance()
                workflow_results['steps_completed'].append('buffer_balance_maintenance')
                logger.info("✅ Buffer balance maintained")
            except Exception as e:
                error_msg = f"Buffer balance maintenance failed: {e}"
                workflow_results['errors'].append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
                # Don't fail workflow for balance issues
            
            # Step 4: Check if retraining should be triggered
            logger.info("🔍 Step 4: Checking retraining triggers...")
            try:
                retrain_triggered = await self._check_retraining_triggers(force_retrain)
                workflow_results['steps_completed'].append('retraining_trigger_check')
                workflow_results['metrics']['retraining_triggered'] = retrain_triggered
                
                if retrain_triggered:
                    logger.info("🚀 Retraining triggered")
                else:
                    logger.info("⏸️ Retraining not triggered")
                    
            except Exception as e:
                error_msg = f"Retraining trigger check failed: {e}"
                workflow_results['errors'].append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
            
            # Step 5: Cleanup old examples (if needed)
            if datetime.now().weekday() == 0:  # Monday
                logger.info("🧹 Step 5: Weekly cleanup...")
                try:
                    await self.buffer_service.cleanup_old_examples()
                    workflow_results['steps_completed'].append('weekly_cleanup')
                    logger.info("✅ Weekly cleanup completed")
                except Exception as e:
                    error_msg = f"Weekly cleanup failed: {e}"
                    workflow_results['errors'].append(error_msg)
                    logger.warning(f"⚠️ {error_msg}")
            
            # Step 6: Log workflow completion
            workflow_time = time.time() - workflow_start
            workflow_results['end_time'] = datetime.now()
            workflow_results['total_time_seconds'] = workflow_time
            workflow_results['status'] = 'completed' if not workflow_results['errors'] else 'completed_with_errors'
            
            # Update integration metrics
            self._update_integration_metrics(workflow_time, len(workflow_results['errors']) == 0)
            
            # Log to monitoring system if available
            if AI_AVAILABLE:
                await self._log_workflow_to_monitoring(workflow_results)
            
            logger.info(f"✅ Complete workflow finished in {workflow_time:.2f}s")
            return workflow_results
            
        except Exception as e:
            workflow_time = time.time() - workflow_start
            workflow_results['end_time'] = datetime.now()
            workflow_results['total_time_seconds'] = workflow_time
            workflow_results['status'] = 'failed'
            workflow_results['errors'].append(f"Workflow failed: {e}")
            
            # Update integration metrics
            self._update_integration_metrics(workflow_time, False)
            
            logger.error(f"❌ Complete workflow failed after {workflow_time:.2f}s: {e}")
            raise
    
    async def _check_retraining_triggers(self, force_retrain: bool = False) -> bool:
        """Check if retraining should be triggered based on hard examples"""
        try:
            if force_retrain:
                logger.info("🚀 Force retrain requested")
                return True
            
            # Get current buffer statistics
            buffer_stats = await self.buffer_service.get_buffer_statistics()
            
            # Check if we have enough hard examples
            if buffer_stats.total_examples < self.integration_config['min_hard_examples_for_retrain']:
                logger.info(f"⏸️ Insufficient hard examples: {buffer_stats.total_examples} < {self.integration_config['min_hard_examples_for_retrain']}")
                return False
            
            # Check if buffer has grown significantly
            buffer_growth_ratio = buffer_stats.total_examples / self.integration_config['min_hard_examples_for_retrain']
            if buffer_growth_ratio > self.integration_config['retrain_threshold_ratio']:
                logger.info(f"🚀 Buffer growth threshold met: {buffer_growth_ratio:.2f} > {self.integration_config['retrain_threshold_ratio']}")
                return True
            
            # Check buffer balance
            hard_negative_deviation = abs(
                buffer_stats.hard_negative_ratio - 0.60
            )
            near_positive_deviation = abs(
                buffer_stats.near_positive_ratio - 0.40
            )
            
            if (hard_negative_deviation > self.integration_config['buffer_imbalance_alert_threshold'] or
                near_positive_deviation > self.integration_config['buffer_imbalance_alert_threshold']):
                logger.info(f"🚀 Buffer imbalance detected, triggering retrain")
                return True
            
            logger.info("⏸️ No retraining triggers met")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking retraining triggers: {e}")
            return False
    
    async def _log_workflow_to_monitoring(self, workflow_results: Dict[str, Any]):
        """Log workflow results to monitoring systems"""
        try:
            if not AI_AVAILABLE:
                return
            
            # Log to Redis logger
            await redis_logger.log_event(
                event_type=EventType.WORKFLOW_COMPLETED,
                level=LogLevel.INFO,
                message=f"Hard example workflow completed: {workflow_results['status']}",
                metadata={
                    'workflow_id': workflow_results['workflow_id'],
                    'total_time': workflow_results['total_time_seconds'],
                    'outcomes_computed': workflow_results['metrics'].get('outcomes_computed', 0),
                    'hard_examples_captured': workflow_results['metrics'].get('hard_examples_captured', 0),
                    'retraining_triggered': workflow_results['metrics'].get('retraining_triggered', False),
                    'errors': workflow_results['errors']
                }
            )
            
            # Log to model registry if available
            if self.model_registry:
                await self.model_registry.log_experiment(
                    experiment_name="hard_example_buffer_workflow",
                    parameters={
                        'workflow_id': workflow_results['workflow_id'],
                        'symbols': workflow_results['symbols'],
                        'force_retrain': False
                    },
                    metrics={
                        'total_time_seconds': workflow_results['total_time_seconds'],
                        'outcomes_computed': workflow_results['metrics'].get('outcomes_computed', 0),
                        'hard_examples_captured': workflow_results['metrics'].get('hard_examples_captured', 0),
                        'retraining_triggered': workflow_results['metrics'].get('retraining_triggered', False)
                    },
                    tags=['hard_example_buffer', 'workflow', workflow_results['status']]
                )
            
            logger.info("✅ Workflow logged to monitoring systems")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log workflow to monitoring: {e}")
    
    def _update_integration_metrics(self, workflow_time: float, success: bool):
        """Update integration performance metrics"""
        try:
            self.integration_metrics['total_workflows_executed'] += 1
            self.integration_metrics['last_workflow_time'] = workflow_time
            self.integration_metrics['last_workflow_status'] = 'success' if success else 'failure'
            
            # Update average workflow time
            current_avg = self.integration_metrics['avg_workflow_time']
            total_workflows = self.integration_metrics['total_workflows_executed']
            self.integration_metrics['avg_workflow_time'] = (
                (current_avg * (total_workflows - 1) + workflow_time) / total_workflows
            )
            
            # Check performance alerts
            if workflow_time > self.integration_config['performance_alert_threshold']:
                logger.warning(f"⚠️ Workflow performance alert: {workflow_time:.2f}s > {self.integration_config['performance_alert_threshold']}s")
            
        except Exception as e:
            logger.error(f"❌ Error updating integration metrics: {e}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and health"""
        try:
            # Get buffer statistics
            buffer_stats = await self.buffer_service.get_buffer_statistics()
            
            # Get performance metrics
            buffer_metrics = await self.buffer_service.get_performance_metrics()
            
            # Check database health
            db_health = await self.db_connection.health_check()
            
            # Determine overall health
            overall_health = 'healthy'
            if not db_health['healthy']:
                overall_health = 'unhealthy'
            elif buffer_stats.total_examples > self.buffer_service.buffer_config['max_size']:
                overall_health = 'warning'
            elif (abs(buffer_stats.hard_negative_ratio - 0.60) > 0.1 or
                  abs(buffer_stats.near_positive_ratio - 0.40) > 0.1):
                overall_health = 'warning'
            
            return {
                'overall_health': overall_health,
                'timestamp': datetime.now(),
                'database_health': db_health,
                'buffer_statistics': {
                    'total_examples': buffer_stats.total_examples,
                    'hard_negatives': buffer_stats.hard_negatives,
                    'near_positives': buffer_stats.near_positives,
                    'hard_negative_ratio': buffer_stats.hard_negative_ratio,
                    'near_positive_ratio': buffer_stats.near_positive_ratio,
                    'buffer_size_mb': buffer_stats.buffer_size_mb
                },
                'performance_metrics': {
                    'outcome_computation_time': buffer_metrics['outcome_computation_time'],
                    'buffer_update_time': buffer_metrics['buffer_update_time'],
                    'total_trades_processed': buffer_metrics['total_trades_processed'],
                    'total_hard_examples_captured': buffer_metrics['total_hard_examples_captured']
                },
                'integration_metrics': self.integration_metrics,
                'configuration': {
                    **self.buffer_service.buffer_config,
                    **self.integration_config
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting integration status: {e}")
            return {
                'overall_health': 'unknown',
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def trigger_manual_retraining(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Manually trigger retraining with current hard examples"""
        try:
            logger.info("🚀 Manual retraining triggered")
            
            # Execute workflow with force retrain
            workflow_results = await self.execute_complete_workflow(
                symbols=symbols,
                force_retrain=True
            )
            
            # If retraining was triggered, log it
            if workflow_results['metrics'].get('retraining_triggered', False):
                self.integration_metrics['total_retraining_triggers'] += 1
                logger.info("✅ Manual retraining completed successfully")
            else:
                logger.warning("⚠️ Manual retraining requested but not triggered")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"❌ Manual retraining failed: {e}")
            raise
    
    async def reset_integration_metrics(self):
        """Reset integration performance metrics"""
        try:
            self.integration_metrics = {
                'total_workflows_executed': 0,
                'total_retraining_triggers': 0,
                'avg_workflow_time': 0.0,
                'last_workflow_time': 0.0,
                'last_workflow_status': 'unknown'
            }
            
            # Also reset buffer service metrics
            await self.buffer_service.reset_performance_metrics()
            
            logger.info("🔄 Integration metrics reset")
            
        except Exception as e:
            logger.error(f"❌ Error resetting integration metrics: {e}")

# Prefect integration tasks
if PREFECT_AVAILABLE:
    @task(name="hard_example_integration_workflow")
    async def hard_example_integration_workflow_task(symbols: List[str] = None):
        """Prefect task for hard example integration workflow"""
        service = HardExampleIntegrationService()
        return await service.execute_complete_workflow(symbols)
    
    @task(name="check_integration_status")
    async def check_integration_status_task():
        """Prefect task for checking integration status"""
        service = HardExampleIntegrationService()
        return await service.get_integration_status()
    
    @task(name="trigger_manual_retraining")
    async def trigger_manual_retraining_task(symbols: List[str] = None):
        """Prefect task for manual retraining trigger"""
        service = HardExampleIntegrationService()
        return await service.trigger_manual_retraining(symbols)
    
    @flow(name="hard_example_integration_flow")
    async def hard_example_integration_flow(symbols: List[str] = None):
        """Complete hard example integration flow"""
        logger = get_run_logger()
        
        try:
            logger.info("🚀 Starting hard example integration flow")
            
            # Execute integration workflow
            workflow_results = await hard_example_integration_workflow_task(symbols)
            
            # Check integration status
            status = await check_integration_status_task()
            
            logger.info(f"✅ Integration flow completed: {workflow_results['status']}")
            logger.info(f"📊 Integration health: {status['overall_health']}")
            
            return {
                'workflow_results': workflow_results,
                'integration_status': status
            }
            
        except Exception as e:
            logger.error(f"❌ Integration flow failed: {e}")
            raise

# Global service instance
hard_example_integration_service = HardExampleIntegrationService()

# Export for use in other modules
__all__ = [
    'HardExampleIntegrationService',
    'hard_example_integration_service'
]
