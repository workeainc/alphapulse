#!/usr/bin/env python3
"""
Test Suite for Closed-Loop Monitoring System
Tests the integration between monitoring alerts and automated retraining
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.monitoring_service import MonitoringService, MonitoringAlert, ClosedLoopAction
from app.services.enhanced_auto_retraining_pipeline import EnhancedAutoRetrainingPipeline
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClosedLoopMonitoringTestSuite:
    """Comprehensive test suite for closed-loop monitoring"""
    
    def __init__(self):
        self.database_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.engine = create_engine(self.database_url)
        
        # Test results
        self.test_results = {
            'test_suite': 'Closed-Loop Monitoring System',
            'tests': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        }
        
        # Services
        self.monitoring_service = None
        self.retraining_pipeline = None
        
    async def run_all_tests(self):
        """Run all closed-loop monitoring tests"""
        logger.info("🚀 Starting Closed-Loop Monitoring Test Suite")
        
        try:
            # Initialize services
            await self._initialize_services()
            
            # Run test categories
            await self._test_monitoring_alert_creation()
            await self._test_closed_loop_actions()
            await self._test_automated_response_rules()
            await self._test_feedback_loop_metrics()
            await self._test_monitoring_retraining_integration()
            await self._test_alert_trigger_workflow()
            await self._test_retraining_trigger_from_alerts()
            await self._test_feedback_loop_analysis()
            
            # Calculate summary
            self._calculate_summary()
            
            # Save results
            await self._save_test_results()
            
            logger.info("✅ Closed-Loop Monitoring Test Suite completed")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise
    
    async def _initialize_services(self):
        """Initialize monitoring and retraining services"""
        try:
            # Initialize monitoring service
            self.monitoring_service = MonitoringService()
            await self.monitoring_service.start()
            
            # Initialize retraining pipeline
            self.retraining_pipeline = EnhancedAutoRetrainingPipeline()
            await self.retraining_pipeline.initialize()
            
            # Integrate services
            await self.retraining_pipeline.integrate_with_monitoring(self.monitoring_service)
            
            logger.info("✅ Services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            raise
    
    async def _test_monitoring_alert_creation(self):
        """Test monitoring alert creation and storage"""
        logger.info("📊 Testing Monitoring Alert Creation")
        
        test_cases = [
            {
                'name': 'Drift Alert Creation',
                'alert_type': 'drift',
                'severity': 'high',
                'current_value': 0.35,
                'threshold': 0.25
            },
            {
                'name': 'Performance Alert Creation',
                'alert_type': 'performance',
                'severity': 'medium',
                'current_value': 0.15,
                'threshold': 0.10
            },
            {
                'name': 'Risk Alert Creation',
                'alert_type': 'risk',
                'severity': 'critical',
                'current_value': 90.0,
                'threshold': 80.0
            }
        ]
        
        for test_case in test_cases:
            await self._run_test(
                f"Monitoring Alert - {test_case['name']}",
                self._create_monitoring_alert_test,
                test_case
            )
    
    async def _test_closed_loop_actions(self):
        """Test closed-loop action creation and execution"""
        logger.info("🔄 Testing Closed-Loop Actions")
        
        test_cases = [
            {
                'name': 'Retraining Action',
                'action_type': 'trigger_retraining',
                'trigger_source': 'drift_detection'
            },
            {
                'name': 'Shadow Deployment Action',
                'action_type': 'deploy_shadow',
                'trigger_source': 'performance_degradation'
            },
            {
                'name': 'Rollback Action',
                'action_type': 'rollback',
                'trigger_source': 'risk_alert'
            }
        ]
        
        for test_case in test_cases:
            await self._run_test(
                f"Closed-Loop Action - {test_case['name']}",
                self._create_closed_loop_action_test,
                test_case
            )
    
    async def _test_automated_response_rules(self):
        """Test automated response rules"""
        logger.info("⚙️ Testing Automated Response Rules")
        
        test_cases = [
            {
                'name': 'Drift Retraining Rule',
                'rule_type': 'drift_retraining',
                'model_id': 'lightgbm_ensemble',
                'current_value': 0.30
            },
            {
                'name': 'Performance Degradation Rule',
                'rule_type': 'performance_retraining',
                'model_id': 'lstm_time_series',
                'current_value': 0.12
            },
            {
                'name': 'Risk Alert Rule',
                'rule_type': 'risk_retraining',
                'model_id': 'ensemble_system',
                'current_value': 85.0
            }
        ]
        
        for test_case in test_cases:
            await self._run_test(
                f"Automated Response Rule - {test_case['name']}",
                self._check_automated_response_rules_test,
                test_case
            )
    
    async def _test_feedback_loop_metrics(self):
        """Test feedback loop metrics logging"""
        logger.info("📈 Testing Feedback Loop Metrics")
        
        test_cases = [
            {
                'name': 'Drift Loop Metrics',
                'loop_type': 'drift_retraining',
                'model_id': 'lightgbm_ensemble'
            },
            {
                'name': 'Performance Loop Metrics',
                'loop_type': 'performance_retraining',
                'model_id': 'lstm_time_series'
            }
        ]
        
        for test_case in test_cases:
            await self._run_test(
                f"Feedback Loop Metrics - {test_case['name']}",
                self._log_feedback_loop_metrics_test,
                test_case
            )
    
    async def _test_monitoring_retraining_integration(self):
        """Test monitoring-retraining integration"""
        logger.info("🔗 Testing Monitoring-Retraining Integration")
        
        await self._run_test(
            "Integration Setup",
            self._setup_monitoring_retraining_integration_test
        )
        
        await self._run_test(
            "Integration Activation",
            self._activate_monitoring_retraining_integration_test
        )
    
    async def _test_alert_trigger_workflow(self):
        """Test complete alert trigger workflow"""
        logger.info("🚨 Testing Alert Trigger Workflow")
        
        await self._run_test(
            "Alert Creation and Trigger",
            self._alert_trigger_workflow_test
        )
    
    async def _test_retraining_trigger_from_alerts(self):
        """Test retraining trigger from monitoring alerts"""
        logger.info("🔄 Testing Retraining Trigger from Alerts")
        
        test_cases = [
            {
                'name': 'High Severity Drift Alert',
                'alert_type': 'drift',
                'severity': 'high',
                'current_value': 0.40,
                'expected_trigger': True
            },
            {
                'name': 'Low Severity Performance Alert',
                'alert_type': 'performance',
                'severity': 'low',
                'current_value': 0.05,
                'expected_trigger': False
            },
            {
                'name': 'Critical Risk Alert',
                'alert_type': 'risk',
                'severity': 'critical',
                'current_value': 95.0,
                'expected_trigger': True
            }
        ]
        
        for test_case in test_cases:
            await self._run_test(
                f"Retraining Trigger - {test_case['name']}",
                self._retraining_trigger_from_alert_test,
                test_case
            )
    
    async def _test_feedback_loop_analysis(self):
        """Test feedback loop analysis and metrics"""
        logger.info("📊 Testing Feedback Loop Analysis")
        
        await self._run_test(
            "Feedback Loop Performance Analysis",
            self._feedback_loop_analysis_test
        )
    
    # Individual test methods
    async def _create_monitoring_alert_test(self, test_case: Dict[str, Any]):
        """Test monitoring alert creation"""
        try:
            alert_id = f"test_alert_{uuid.uuid4().hex[:8]}"
            model_id = "test_model"
            
            alert = MonitoringAlert(
                alert_id=alert_id,
                model_id=model_id,
                alert_type=test_case['alert_type'],
                severity_level=test_case['severity'],
                trigger_condition={'threshold': test_case['threshold']},
                current_value=test_case['current_value'],
                threshold_value=test_case['threshold'],
                is_triggered=test_case['current_value'] > test_case['threshold'],
                triggered_at=datetime.now() if test_case['current_value'] > test_case['threshold'] else None,
                alert_metadata={'test_case': test_case['name']}
            )
            
            # Create alert
            success = await self.monitoring_service.create_monitoring_alert(alert)
            assert success, f"Failed to create monitoring alert: {alert_id}"
            
            # Verify alert was stored
            stored_alert = await self._get_stored_alert(alert_id)
            assert stored_alert is not None, f"Alert not found in database: {alert_id}"
            assert stored_alert['alert_type'] == test_case['alert_type'], "Alert type mismatch"
            assert stored_alert['severity_level'] == test_case['severity'], "Severity level mismatch"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in monitoring alert test: {e}")
            return False
    
    async def _create_closed_loop_action_test(self, test_case: Dict[str, Any]):
        """Test closed-loop action creation"""
        try:
            action_id = f"test_action_{uuid.uuid4().hex[:8]}"
            alert_id = f"test_alert_{uuid.uuid4().hex[:8]}"
            model_id = "test_model"
            
            action = ClosedLoopAction(
                action_id=action_id,
                alert_id=alert_id,
                model_id=model_id,
                action_type=test_case['action_type'],
                action_status='pending',
                trigger_source=test_case['trigger_source'],
                action_config={'priority': 'high'},
                execution_start=datetime.now(),
                success=None,
                error_message=None,
                action_metadata={'test_case': test_case['name']}
            )
            
            # Trigger action
            success = await self.monitoring_service.trigger_closed_loop_action(action)
            assert success, f"Failed to trigger closed-loop action: {action_id}"
            
            # Verify action was stored
            stored_action = await self._get_stored_action(action_id)
            assert stored_action is not None, f"Action not found in database: {action_id}"
            assert stored_action['action_type'] == test_case['action_type'], "Action type mismatch"
            assert stored_action['trigger_source'] == test_case['trigger_source'], "Trigger source mismatch"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in closed-loop action test: {e}")
            return False
    
    async def _check_automated_response_rules_test(self, test_case: Dict[str, Any]):
        """Test automated response rules checking"""
        try:
            model_id = test_case['model_id']
            alert_type = test_case['rule_type'].split('_')[0]  # Extract alert type from rule type
            current_value = test_case['current_value']
            
            # Check response rules
            triggered_rules = await self.monitoring_service.check_automated_response_rules(
                model_id, alert_type, current_value
            )
            
            # Verify rules were found
            assert len(triggered_rules) > 0, f"No response rules found for {alert_type}"
            
            # Check if any rules should be triggered based on current value
            should_trigger = any(
                rule['rule_type'] == test_case['rule_type'] 
                for rule in triggered_rules
            )
            
            # For high values, we expect rules to be triggered
            if current_value > 0.25:  # High threshold
                assert should_trigger, f"Expected rules to be triggered for high value: {current_value}"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in automated response rules test: {e}")
            return False
    
    async def _log_feedback_loop_metrics_test(self, test_case: Dict[str, Any]):
        """Test feedback loop metrics logging"""
        try:
            model_id = test_case['model_id']
            loop_type = test_case['loop_type']
            
            # Create sample metrics
            metrics = {
                'trigger_to_action_latency_seconds': 15.5,
                'action_success_rate': 0.95,
                'performance_improvement': 0.12,
                'drift_reduction': 0.08,
                'false_positive_rate': 0.05,
                'false_negative_rate': 0.03,
                'total_triggers': 10,
                'successful_actions': 9,
                'failed_actions': 1,
                'metadata': {'test_case': test_case['name']}
            }
            
            # Log metrics
            success = await self.monitoring_service.log_feedback_loop_metrics(
                model_id, loop_type, metrics
            )
            assert success, f"Failed to log feedback loop metrics for {model_id}"
            
            # Verify metrics were stored
            stored_metrics = await self._get_stored_feedback_metrics(model_id, loop_type)
            assert stored_metrics is not None, f"Feedback metrics not found for {model_id}"
            assert stored_metrics['loop_type'] == loop_type, "Loop type mismatch"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in feedback loop metrics test: {e}")
            return False
    
    async def _setup_monitoring_retraining_integration_test(self):
        """Test monitoring-retraining integration setup"""
        try:
            integration_id = f"test_integration_{uuid.uuid4().hex[:8]}"
            model_id = "test_model"
            rule_id = "test_rule"
            job_id = "test_job"
            
            # Create integration record
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO monitoring_retraining_integration (
                        integration_id, model_id, monitoring_rule_id, retraining_job_id,
                        integration_type, is_active, trigger_conditions, action_sequence
                    ) VALUES (
                        :integration_id, :model_id, :rule_id, :job_id,
                        'drift_retraining', TRUE, '{"drift_threshold": 0.25}', '{"action": "retrain"}'
                    )
                """), {
                    'integration_id': integration_id,
                    'model_id': model_id,
                    'rule_id': rule_id,
                    'job_id': job_id
                })
                conn.commit()
            
            # Verify integration was created
            stored_integration = await self._get_stored_integration(integration_id)
            assert stored_integration is not None, f"Integration not found: {integration_id}"
            assert stored_integration['is_active'] is True, "Integration should be active"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in integration setup test: {e}")
            return False
    
    async def _activate_monitoring_retraining_integration_test(self):
        """Test monitoring-retraining integration activation"""
        try:
            # This would test the actual integration between monitoring and retraining
            # For now, we'll verify the services are properly connected
            
            assert self.monitoring_service is not None, "Monitoring service not initialized"
            assert self.retraining_pipeline is not None, "Retraining pipeline not initialized"
            assert hasattr(self.retraining_pipeline, 'monitoring_service'), "Retraining pipeline not integrated with monitoring"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in integration activation test: {e}")
            return False
    
    async def _alert_trigger_workflow_test(self):
        """Test complete alert trigger workflow"""
        try:
            alert_id = f"workflow_alert_{uuid.uuid4().hex[:8]}"
            model_id = "workflow_model"
            
            # Create alert
            alert = MonitoringAlert(
                alert_id=alert_id,
                model_id=model_id,
                alert_type='drift',
                severity_level='high',
                trigger_condition={'threshold': 0.25},
                current_value=0.35,
                threshold_value=0.25,
                is_triggered=True,
                triggered_at=datetime.now(),
                alert_metadata={'workflow_test': True}
            )
            
            # Create alert
            success = await self.monitoring_service.create_monitoring_alert(alert)
            assert success, "Failed to create workflow alert"
            
            # Update trigger status
            success = await self.monitoring_service.update_alert_trigger_status(
                alert_id, True, "test_job_id"
            )
            assert success, "Failed to update alert trigger status"
            
            # Verify trigger status was updated
            stored_alert = await self._get_stored_alert(alert_id)
            assert stored_alert['is_triggered'] is True, "Alert should be triggered"
            assert stored_alert['retraining_job_id'] == "test_job_id", "Retraining job ID should be set"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in alert trigger workflow test: {e}")
            return False
    
    async def _retraining_trigger_from_alert_test(self, test_case: Dict[str, Any]):
        """Test retraining trigger from monitoring alert"""
        try:
            alert_data = {
                'alert_id': f"trigger_alert_{uuid.uuid4().hex[:8]}",
                'model_id': 'test_model',
                'alert_type': test_case['alert_type'],
                'severity_level': test_case['severity'],
                'current_value': test_case['current_value']
            }
            
            # Handle monitoring alert
            triggered = await self.retraining_pipeline.handle_monitoring_alert(alert_data)
            
            # Verify trigger matches expectation
            assert triggered == test_case['expected_trigger'], \
                f"Trigger expectation mismatch: expected {test_case['expected_trigger']}, got {triggered}"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in retraining trigger test: {e}")
            return False
    
    async def _feedback_loop_analysis_test(self):
        """Test feedback loop analysis"""
        try:
            # Get feedback loop metrics
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        model_id,
                        loop_type,
                        AVG(action_success_rate) as avg_success_rate,
                        AVG(performance_improvement) as avg_performance_improvement,
                        AVG(drift_reduction) as avg_drift_reduction,
                        COUNT(*) as total_metrics
                    FROM feedback_loop_metrics
                    GROUP BY model_id, loop_type
                """))
                
                metrics = result.fetchall()
                
                # Verify we have some metrics
                assert len(metrics) > 0, "No feedback loop metrics found"
                
                # Check metrics structure
                for metric in metrics:
                    assert metric.avg_success_rate is not None, "Success rate should not be null"
                    assert metric.total_metrics > 0, "Total metrics should be positive"
            
            return True
            
        except Exception as e:
            logger.error(f"Error in feedback loop analysis test: {e}")
            return False
    
    # Helper methods
    async def _run_test(self, test_name: str, test_func, test_case: Dict[str, Any] = None):
        """Run a single test"""
        try:
            logger.info(f"🧪 Running test: {test_name}")
            
            if test_case:
                result = await test_func(test_case)
            else:
                result = await test_func()
            
            if result:
                logger.info(f"✅ Test passed: {test_name}")
                self.test_results['tests'].append({
                    'name': test_name,
                    'status': 'passed',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.error(f"❌ Test failed: {test_name}")
                self.test_results['tests'].append({
                    'name': test_name,
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"❌ Test error: {test_name} - {e}")
            self.test_results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    async def _get_stored_alert(self, alert_id: str) -> Dict[str, Any]:
        """Get stored alert from database"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT alert_id, model_id, alert_type, severity_level, 
                       current_value, threshold_value, is_triggered, retraining_job_id
                FROM monitoring_alert_triggers
                WHERE alert_id = :alert_id
            """), {'alert_id': alert_id})
            
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    
    async def _get_stored_action(self, action_id: str) -> Dict[str, Any]:
        """Get stored action from database"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT action_id, alert_id, model_id, action_type, 
                       action_status, trigger_source
                FROM closed_loop_actions
                WHERE action_id = :action_id
            """), {'action_id': action_id})
            
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    
    async def _get_stored_feedback_metrics(self, model_id: str, loop_type: str) -> Dict[str, Any]:
        """Get stored feedback metrics from database"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT metric_id, model_id, loop_type, action_success_rate,
                       performance_improvement, drift_reduction
                FROM feedback_loop_metrics
                WHERE model_id = :model_id AND loop_type = :loop_type
                ORDER BY timestamp DESC
                LIMIT 1
            """), {'model_id': model_id, 'loop_type': loop_type})
            
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    
    async def _get_stored_integration(self, integration_id: str) -> Dict[str, Any]:
        """Get stored integration from database"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT integration_id, model_id, integration_type, is_active
                FROM monitoring_retraining_integration
                WHERE integration_id = :integration_id
            """), {'integration_id': integration_id})
            
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    
    def _calculate_summary(self):
        """Calculate test summary"""
        total_tests = len(self.test_results['tests'])
        passed_tests = len([t for t in self.test_results['tests'] if t['status'] == 'passed'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': success_rate
        }
        
        logger.info(f"📊 Test Summary: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
    
    async def _save_test_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"closed_loop_monitoring_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"💾 Test results saved to: {filename}")
        
        # Also save pipeline stats
        if self.retraining_pipeline:
            pipeline_stats = self.retraining_pipeline.get_stats()
            pipeline_filename = f"closed_loop_monitoring_pipeline_stats_{timestamp}.json"
            
            with open(pipeline_filename, 'w') as f:
                json.dump(pipeline_stats, f, indent=2, default=str)
            
            logger.info(f"💾 Pipeline stats saved to: {pipeline_filename}")

async def main():
    """Main test runner"""
    test_suite = ClosedLoopMonitoringTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
