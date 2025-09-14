#!/usr/bin/env python3
"""
Phase 2 Integration Validation for Streaming Infrastructure
Validates API contracts and data formats for Phase 2 compatibility
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from streaming.stream_processor import StreamProcessor
from streaming.stream_metrics import StreamMetrics
from core.config import STREAMING_CONFIG
from app.main_ai_system_simple import app
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2IntegrationValidator:
    def __init__(self):
        self.stream_processor = None
        self.stream_metrics = None
        self.test_client = None
        self.validation_results = {
            'api_contracts_validated': 0,
            'api_contracts_failed': 0,
            'data_formats_validated': 0,
            'data_formats_failed': 0,
            'integration_points_tested': 0,
            'integration_points_failed': 0
        }

    async def initialize_components(self):
        """Initialize streaming components for validation"""
        logger.info("🔧 Initializing streaming components for Phase 2 integration validation...")
        try:
            self.stream_processor = StreamProcessor(STREAMING_CONFIG)
            self.stream_metrics = StreamMetrics(STREAMING_CONFIG)
            
            await self.stream_processor.initialize()
            await self.stream_metrics.initialize()
            
            self.test_client = TestClient(app)
            
            logger.info("✅ All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False

    async def validate_api_contracts(self):
        """Validate API contracts for Phase 2 compatibility"""
        logger.info("🔍 Validating API contracts for Phase 2 compatibility...")
        
        # Test streaming processor API
        test_message = {
            'message_id': f"test_{int(datetime.now().timestamp())}",
            'symbol': 'BTCUSDT',
            'data_type': 'tick',
            'source': 'phase2_validation',
            'data': {'price': 50000.0, 'volume': 1.0},
            'timestamp': datetime.now()
        }
        
        try:
            result = await self.stream_processor.process_message(test_message)
            if result and isinstance(result, dict):
                self.validation_results['api_contracts_validated'] += 1
                logger.info("✅ Stream processor API contract validation passed")
            else:
                self.validation_results['api_contracts_failed'] += 1
                logger.error("❌ Stream processor API contract validation failed")
        except Exception as e:
            self.validation_results['api_contracts_failed'] += 1
            logger.error(f"❌ Stream processor API contract validation failed: {e}")

        # Test stream metrics API
        try:
            metrics = await self.stream_metrics.get_current_metrics()
            if metrics and isinstance(metrics, dict):
                self.validation_results['api_contracts_validated'] += 1
                logger.info("✅ Stream metrics API contract validation passed")
            else:
                self.validation_results['api_contracts_failed'] += 1
                logger.error("❌ Stream metrics API contract validation failed")
        except Exception as e:
            self.validation_results['api_contracts_failed'] += 1
            logger.error(f"❌ Stream metrics API contract validation failed: {e}")

    async def validate_data_formats(self):
        """Validate data formats for Phase 2 compatibility"""
        logger.info("🔍 Validating data formats for Phase 2 compatibility...")
        
        # Test market data format
        market_data = {
            'message_id': 'test_001',
            'symbol': 'BTCUSDT',
            'data_type': 'tick',
            'source': 'test',
            'data': {
                'price': 50000.0,
                'volume': 1.0,
                'bid': 49999.0,
                'ask': 50001.0,
                'timestamp': datetime.now().isoformat()
            },
            'timestamp': datetime.now()
        }
        
        required_fields = ['message_id', 'symbol', 'data_type', 'source', 'data', 'timestamp']
        missing_fields = [field for field in required_fields if field not in market_data]
        
        if not missing_fields:
            self.validation_results['data_formats_validated'] += 1
            logger.info("✅ Market data format validation passed")
        else:
            self.validation_results['data_formats_failed'] += 1
            logger.error(f"❌ Market data format validation failed: missing {missing_fields}")

    async def validate_integration_points(self):
        """Validate integration points for Phase 2"""
        logger.info("🔍 Validating integration points for Phase 2...")
        
        # Test API endpoints
        endpoints = [
            '/api/streaming/status',
            '/api/streaming/metrics',
            '/api/streaming/data/BTCUSDT'
        ]
        
        for endpoint in endpoints:
            try:
                response = self.test_client.get(endpoint)
                if response.status_code == 200:
                    self.validation_results['integration_points_tested'] += 1
                    logger.info(f"✅ {endpoint} integration test passed")
                else:
                    self.validation_results['integration_points_failed'] += 1
                    logger.error(f"❌ {endpoint} integration test failed: {response.status_code}")
            except Exception as e:
                self.validation_results['integration_points_failed'] += 1
                logger.error(f"❌ {endpoint} integration test failed: {e}")

    def generate_validation_report(self):
        """Generate validation report"""
        logger.info("📋 Generating Phase 2 integration validation report...")
        
        total_tests = sum(self.validation_results.values())
        passed_tests = (
            self.validation_results['api_contracts_validated'] +
            self.validation_results['data_formats_validated'] +
            self.validation_results['integration_points_tested']
        )
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'validation_summary': {
                'test_name': 'Phase 2 Integration Validation',
                'test_date': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate_percent': success_rate
            },
            'detailed_results': self.validation_results,
            'recommendations': []
        }
        
        if self.validation_results['api_contracts_failed'] > 0:
            report['recommendations'].append("❌ API contract validation failures - review API contracts")
        
        if self.validation_results['data_formats_failed'] > 0:
            report['recommendations'].append("❌ Data format validation failures - ensure data format compatibility")
        
        if self.validation_results['integration_points_failed'] > 0:
            report['recommendations'].append("❌ Integration point failures - verify integration compatibility")
        
        if not report['recommendations']:
            report['recommendations'].append("✅ All validation tests passed - Phase 2 integration ready")
        
        return report

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up validation resources...")
        try:
            if self.stream_processor:
                await self.stream_processor.shutdown()
            if self.stream_metrics:
                await self.stream_metrics.shutdown()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main validation execution"""
    logger.info("=" * 80)
    logger.info("🔍 PHASE 2 INTEGRATION VALIDATION")
    logger.info("=" * 80)
    
    validator = Phase2IntegrationValidator()
    
    try:
        if not await validator.initialize_components():
            return False
        
        await validator.validate_api_contracts()
        await validator.validate_data_formats()
        await validator.validate_integration_points()
        
        report = validator.generate_validation_report()
        
        logger.info("=" * 80)
        logger.info("📊 VALIDATION RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"Total Tests: {report['validation_summary']['total_tests']}")
        logger.info(f"Passed Tests: {report['validation_summary']['passed_tests']}")
        logger.info(f"Failed Tests: {report['validation_summary']['failed_tests']}")
        logger.info(f"Success Rate: {report['validation_summary']['success_rate_percent']:.1f}%")
        
        logger.info("\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            logger.info(f"  {rec}")
        
        report_file = backend_path / "phase2_integration_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📄 Report saved to: {report_file}")
        
        success = report['validation_summary']['success_rate_percent'] >= 90
        
        if success:
            logger.info("🎉 PHASE 2 INTEGRATION VALIDATION PASSED!")
        else:
            logger.error("❌ PHASE 2 INTEGRATION VALIDATION FAILED")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False
    finally:
        await validator.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
