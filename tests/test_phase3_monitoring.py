#!/usr/bin/env python3
"""
Test script for Phase 3 - Priority 8: Grafana/Metabase Dashboards & Alerts
Tests the monitoring service, Prometheus metrics, and dashboard configurations.
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the monitoring service
try:
    from prometheus_client import REGISTRY
    MONITORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Monitoring service not available: {e}")
    MONITORING_AVAILABLE = False

# Import database components
try:
    from ..database.connection_simple import SimpleTimescaleDBConnection
    from ..database.data_versioning_dao import DataVersioningDAO
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database components not available: {e}")
    DATABASE_AVAILABLE = False


def clear_prometheus_registry():
    """Clear the Prometheus registry to avoid duplicate metrics"""
    if MONITORING_AVAILABLE:
        try:
            # Clear all collectors from the registry
            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                REGISTRY.unregister(collector)
        except Exception as e:
            logger.warning(f"Could not clear Prometheus registry: {e}")

def create_monitoring_service():
    """Create a monitoring service with a fresh registry"""
    if not MONITORING_AVAILABLE:
        return None
    
    try:
        # Clear the global registry first
        clear_prometheus_registry()
        
        # Create a custom registry for this test
        from prometheus_client import CollectorRegistry
        custom_registry = CollectorRegistry()
        
        # Create a new service instance with custom registry
        from app.services.monitoring_service import MonitoringService
        service = MonitoringService(registry=custom_registry)
        
        return service
    except Exception as e:
        logger.error(f"Failed to create monitoring service: {e}")
        return None

def create_inference_timer(service):
    """Create an inference timer for the given service"""
    if not MONITORING_AVAILABLE:
        return None
    
    try:
        from app.services.monitoring_service import InferenceTimer
        return InferenceTimer(service)
    except Exception as e:
        logger.error(f"Failed to create inference timer: {e}")
        return None


def check_grafana_dashboard():
    """Check if Grafana dashboard configuration exists"""
    logger.info("🧪 Checking Grafana dashboard configuration...")
    
    dashboard_path = "grafana/alphapulse_dashboard.json"
    
    if os.path.exists(dashboard_path):
        try:
            with open(dashboard_path, 'r') as f:
                dashboard_config = json.load(f)
            
            # Validate dashboard structure
            required_fields = ['panels', 'time', 'refresh']
            for field in required_fields:
                if field not in dashboard_config.get('dashboard', {}):
                    logger.error(f"❌ Missing required field: {field}")
                    return False
            
            # Check for required panels
            panels = dashboard_config['dashboard'].get('panels', [])
            panel_titles = [panel.get('title', '') for panel in panels]
            
            required_panels = [
                "Trading Performance Overview",
                "Model Drift Detection", 
                "Inference Latency (p95)",
                "Active Learning Queue Status"
            ]
            
            for required_panel in required_panels:
                if required_panel not in panel_titles:
                    logger.error(f"❌ Missing required panel: {required_panel}")
                    return False
            
            logger.info(f"✅ Grafana dashboard configuration valid with {len(panels)} panels")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading Grafana dashboard: {e}")
            return False
    else:
        logger.error(f"❌ Grafana dashboard file not found: {dashboard_path}")
        return False


def check_prometheus_alerts():
    """Check if Prometheus alerts configuration exists"""
    logger.info("🧪 Checking Prometheus alerts configuration...")
    
    alerts_path = "grafana/alphapulse_alerts.yml"
    
    if os.path.exists(alerts_path):
        try:
            with open(alerts_path, 'r') as f:
                alerts_content = f.read()
            
            # Check for required alert groups
            required_groups = [
                "alphapulse_trading_alerts",
                "alphapulse_drift_alerts", 
                "alphapulse_latency_alerts",
                "alphapulse_system_alerts"
            ]
            
            for group in required_groups:
                if group not in alerts_content:
                    logger.error(f"❌ Missing alert group: {group}")
                    return False
            
            # Check for specific alerts
            required_alerts = [
                "LowPrecision",
                "HighPSIDrift",
                "HighP95Latency",
                "HighCPUUsage"
            ]
            
            for alert in required_alerts:
                if alert not in alerts_content:
                    logger.error(f"❌ Missing alert: {alert}")
                    return False
            
            logger.info("✅ Prometheus alerts configuration valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading Prometheus alerts: {e}")
            return False
    else:
        logger.error(f"❌ Prometheus alerts file not found: {alerts_path}")
        return False


def check_metabase_config():
    """Check if Metabase configuration exists"""
    logger.info("🧪 Checking Metabase configuration...")
    
    config_path = "metabase/alphapulse_config.yml"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Check for required sections
            required_sections = [
                "database:",
                "dashboards:",
                "alerts:",
                "scheduled_reports:"
            ]
            
            for section in required_sections:
                if section not in config_content:
                    logger.error(f"❌ Missing section: {section}")
                    return False
            
            # Check for required dashboards
            required_dashboards = [
                "AlphaPulse Trading Overview",
                "AlphaPulse Model Analytics",
                "AlphaPulse Trading Signals"
            ]
            
            for dashboard in required_dashboards:
                if dashboard not in config_content:
                    logger.error(f"❌ Missing dashboard: {dashboard}")
                    return False
            
            logger.info("✅ Metabase configuration valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading Metabase config: {e}")
            return False
    else:
        logger.error(f"❌ Metabase config file not found: {config_path}")
        return False


async def test_monitoring_service_initialization():
    """Test monitoring service initialization"""
    logger.info("🧪 Testing monitoring service initialization...")
    
    if not MONITORING_AVAILABLE:
        logger.warning("❌ Monitoring service not available, skipping test")
        return False
    
    try:
        service = create_monitoring_service()
        if service is None:
            logger.error("❌ Failed to create monitoring service")
            return False
        
        # Check if service initialized properly
        if hasattr(service, 'win_rate') and hasattr(service, 'precision'):
            logger.info("✅ Monitoring service initialized successfully")
            return True
        else:
            logger.error("❌ Monitoring service not properly initialized")
            return False
        
    except Exception as e:
        logger.error(f"❌ Monitoring service initialization failed: {e}")
        return False


async def test_metrics_recording():
    """Test metrics recording functionality"""
    logger.info("🧪 Testing metrics recording functionality...")
    
    if not MONITORING_AVAILABLE:
        logger.warning("❌ Monitoring service not available, skipping test")
        return False
    
    try:
        service = create_monitoring_service()
        if service is None:
            logger.error("❌ Failed to create monitoring service")
            return False
        
        # Test signal recording
        service.record_signal_generated("BTCUSDT", "xgboost_v1")
        service.record_signal_executed("BTCUSDT", "xgboost_v1")
        service.record_signal_rejected("ETHUSDT", "lightgbm_v1")
        
        # Test inference timing
        service.record_inference_duration(0.1)
        service.record_inference_duration(0.05)
        service.record_inference_duration(0.2)
        
        # Test model training failure
        service.record_model_training_failure()
        
        # Test active learning
        service.record_active_learning_processed()
        
        logger.info("✅ Metrics recording functionality working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Metrics recording test failed: {e}")
        return False


async def test_inference_timer():
    """Test inference timer context manager"""
    logger.info("🧪 Testing inference timer context manager...")
    
    if not MONITORING_AVAILABLE:
        logger.warning("❌ Monitoring service not available, skipping test")
        return False
    
    try:
        service = create_monitoring_service()
        if service is None:
            logger.error("❌ Failed to create monitoring service")
            return False
        
        # Test inference timer
        timer = create_inference_timer(service)
        if timer is None:
            logger.error("❌ Failed to create inference timer")
            return False
        
        with timer:
            time.sleep(0.1)  # Simulate inference
        
        logger.info("✅ Inference timer context manager working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference timer test failed: {e}")
        return False


async def test_metrics_generation():
    """Test Prometheus metrics generation"""
    logger.info("🧪 Testing Prometheus metrics generation...")
    
    if not MONITORING_AVAILABLE:
        logger.warning("❌ Monitoring service not available, skipping test")
        return False
    
    try:
        service = create_monitoring_service()
        if service is None:
            logger.error("❌ Failed to create monitoring service")
            return False
        
        # Record some test metrics
        service.record_signal_generated("BTCUSDT", "xgboost_v1")
        service.record_inference_duration(0.1)
        service.record_model_training_failure()
        
        # Generate metrics
        metrics = service.get_metrics()
        
        # Check if metrics contain expected content
        if "alphapulse_signals_generated_total" in metrics:
            logger.info("✅ Prometheus metrics generation working")
            return True
        else:
            logger.error("❌ Metrics generation failed - missing expected content")
            return False
        
    except Exception as e:
        logger.error(f"❌ Metrics generation test failed: {e}")
        return False


async def test_system_metrics():
    """Test system metrics collection"""
    logger.info("🧪 Testing system metrics collection...")
    
    if not MONITORING_AVAILABLE:
        logger.warning("❌ Monitoring service not available, skipping test")
        return False
    
    try:
        service = create_monitoring_service()
        if service is None:
            logger.error("❌ Failed to create monitoring service")
            return False
        
        # Get system metrics
        system_metrics = service.get_system_metrics()
        
        # Validate metrics structure
        required_fields = ['cpu_usage', 'memory_usage', 'disk_usage', 'database_health']
        for field in required_fields:
            if not hasattr(system_metrics, field):
                logger.error(f"❌ Missing system metric field: {field}")
                return False
        
        # Check if metrics are reasonable
        if 0 <= system_metrics.cpu_usage <= 100:
            logger.info("✅ System metrics collection working")
            return True
        else:
            logger.error("❌ System metrics out of reasonable range")
            return False
        
    except Exception as e:
        logger.error(f"❌ System metrics test failed: {e}")
        return False


async def test_service_stats():
    """Test service statistics"""
    logger.info("🧪 Testing service statistics...")
    
    if not MONITORING_AVAILABLE:
        logger.warning("❌ Monitoring service not available, skipping test")
        return False
    
    try:
        service = create_monitoring_service()
        if service is None:
            logger.error("❌ Failed to create monitoring service")
            return False
        
        # Get service stats
        stats = service.get_service_stats()
        
        # Validate stats structure
        required_fields = ['service_running', 'prometheus_available', 'database_available']
        for field in required_fields:
            if field not in stats:
                logger.error(f"❌ Missing service stat field: {field}")
                return False
        
        logger.info("✅ Service statistics working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service stats test failed: {e}")
        return False


async def test_dashboard_integration():
    """Test dashboard integration with database"""
    logger.info("🧪 Testing dashboard integration with database...")
    
    if not DATABASE_AVAILABLE:
        logger.warning("❌ Database not available, skipping test")
        return False
    
    try:
        db_connection = SimpleTimescaleDBConnection()
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            # Test basic database connectivity
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row[0] == 1:
                logger.info("✅ Database connectivity for dashboard working")
                return True
            else:
                logger.error("❌ Database connectivity test failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Dashboard integration test failed: {e}")
        return False


async def test_alert_configuration():
    """Test alert configuration validation"""
    logger.info("🧪 Testing alert configuration validation...")
    
    # This would test the alert rules and thresholds
    # For now, we'll just check if the files exist and are valid
    
    alert_tests = [
        ("Grafana Dashboard", check_grafana_dashboard),
        ("Prometheus Alerts", check_prometheus_alerts),
        ("Metabase Config", check_metabase_config)
    ]
    
    all_passed = True
    
    for test_name, test_func in alert_tests:
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} configuration valid")
            else:
                logger.error(f"❌ {test_name} configuration invalid")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {test_name} test failed: {e}")
            all_passed = False
    
    return all_passed


async def main():
    """Main test function"""
    logger.info("🚀 Starting Phase 3 - Priority 8: Grafana/Metabase Dashboards & Alerts Tests")
    logger.info("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run tests
    tests = [
        ("Configuration Files", test_alert_configuration),
        ("Service Initialization", test_monitoring_service_initialization),
        ("Metrics Recording", test_metrics_recording),
        ("Inference Timer", test_inference_timer),
        ("Metrics Generation", test_metrics_generation),
        ("System Metrics", test_system_metrics),
        ("Service Stats", test_service_stats),
        ("Dashboard Integration", test_dashboard_integration),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All Dashboard & Alerts tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
