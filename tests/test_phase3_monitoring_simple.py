#!/usr/bin/env python3
"""
Simplified Test script for Phase 3 - Priority 8: Grafana/Metabase Dashboards & Alerts
Tests the configuration files and basic functionality without complex Prometheus metrics.
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

# Import database components
try:
    from ..database.connection_simple import SimpleTimescaleDBConnection
    from ..database.data_versioning_dao import DataVersioningDAO
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database components not available: {e}")
    DATABASE_AVAILABLE = False


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
            
            # Check for specific dashboard names
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


def test_prometheus_metrics_structure():
    """Test that Prometheus metrics structure is correct"""
    logger.info("🧪 Testing Prometheus metrics structure...")
    
    try:
        # Check if prometheus_client is available
        import prometheus_client
        logger.info("✅ Prometheus client available")
        
        # Test basic metric creation
        from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry
        
        # Create a test registry
        registry = CollectorRegistry()
        
        # Test creating basic metrics
        test_gauge = Gauge('test_gauge', 'Test gauge', registry=registry)
        test_counter = Counter('test_counter', 'Test counter', registry=registry)
        test_histogram = Histogram('test_histogram', 'Test histogram', registry=registry)
        
        # Test metric operations
        test_gauge.set(1.0)
        test_counter.inc()
        test_histogram.observe(0.5)
        
        # Test metrics generation
        from prometheus_client import generate_latest
        metrics = generate_latest(registry).decode('utf-8')
        
        if 'test_gauge' in metrics and 'test_counter' in metrics and 'test_histogram' in metrics:
            logger.info("✅ Prometheus metrics structure working correctly")
            return True
        else:
            logger.error("❌ Prometheus metrics generation failed")
            return False
            
    except ImportError:
        logger.warning("❌ Prometheus client not available, skipping test")
        return False
    except Exception as e:
        logger.error(f"❌ Prometheus metrics test failed: {e}")
        return False


def test_alert_configuration():
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
    logger.info("🚀 Starting Phase 3 - Priority 8: Grafana/Metabase Dashboards & Alerts Tests (Simplified)")
    logger.info("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run tests
    tests = [
        ("Configuration Files", test_alert_configuration),
        ("Prometheus Metrics Structure", test_prometheus_metrics_structure),
        ("Dashboard Integration", test_dashboard_integration),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'=' * 60}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            test_results[test_name] = result
            
            if result:
                logger.info(f"✅ PASSED: {test_name}")
            else:
                logger.error(f"❌ FAILED: {test_name}")
                
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 80)
    
    passed_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    
    for test_name, result in test_results.items():
        if result:
            logger.info(f"✅ PASSED: {test_name}")
        else:
            logger.error(f"❌ FAILED: {test_name}")
    
    logger.info(f"\n📊 Overall: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    
    if passed_count < total_count:
        logger.warning(f"⚠️ {total_count - passed_count} tests failed")
    else:
        logger.info("🎉 All tests passed!")
    
    return passed_count == total_count


if __name__ == "__main__":
    asyncio.run(main())
