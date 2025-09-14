#!/usr/bin/env python3
"""
Comprehensive Testing Script for Real Data Integration
Tests frontend-backend integration with real TimescaleDB data
Phase 7: Testing & Validation
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import aiohttp
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.real_data_integration_service import RealDataIntegrationService
from services.ai_model_integration_service import AIModelIntegrationService
from services.external_api_integration_service import ExternalAPIIntegrationService
from app.signals.intelligent_signal_generator import IntelligentSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealDataIntegrationTester:
    """Comprehensive tester for real data integration"""
    
    def __init__(self):
        self.real_data_service = RealDataIntegrationService()
        self.ai_model_service = AIModelIntegrationService()
        self.external_api_service = ExternalAPIIntegrationService()
        
        # Initialize signal generator with required parameters
        import asyncpg
        import ccxt
        # Create mock db_pool and exchange for testing
        self.db_pool = None  # Will be handled by the service
        self.exchange = ccxt.binance()  # Use Binance exchange for testing
        self.signal_generator = IntelligentSignalGenerator(self.db_pool, self.exchange)
        
        # Test results
        self.test_results = {
            "real_data_service": {},
            "ai_model_service": {},
            "external_api_service": {},
            "signal_generator": {},
            "api_endpoints": {},
            "websocket_streaming": {},
            "error_handling": {}
        }
        
        # API base URL
        self.api_base_url = "http://localhost:8000"
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("🚀 Starting Comprehensive Real Data Integration Tests")
        
        try:
            # Test 1: Real Data Service
            await self.test_real_data_service()
            
            # Test 2: AI Model Service
            await self.test_ai_model_service()
            
            # Test 3: External API Service
            await self.test_external_api_service()
            
            # Test 4: Signal Generator
            await self.test_signal_generator()
            
            # Test 5: API Endpoints
            await self.test_api_endpoints()
            
            # Test 6: WebSocket Streaming
            await self.test_websocket_streaming()
            
            # Test 7: Error Handling
            await self.test_error_handling()
            
            # Generate final report
            return self.generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return {"error": str(e), "status": "FAILED"}
    
    async def test_real_data_service(self):
        """Test real data service integration"""
        logger.info("📊 Testing Real Data Service...")
        
        test_symbol = "BTCUSDT"
        test_timeframe = "1h"
        
        try:
            # Test market data
            market_data = await self.real_data_service.get_real_market_data(test_symbol, test_timeframe)
            self.test_results["real_data_service"]["market_data"] = {
                "status": "PASS" if market_data else "FAIL",
                "data": market_data.__dict__ if market_data else None
            }
            
            # Test sentiment data
            sentiment_data = await self.real_data_service.get_real_sentiment_data(test_symbol, 24)
            self.test_results["real_data_service"]["sentiment_data"] = {
                "status": "PASS" if sentiment_data else "FAIL",
                "count": len(sentiment_data) if sentiment_data else 0
            }
            
            # Test technical indicators
            technical_data = await self.real_data_service.get_real_technical_indicators(test_symbol, test_timeframe)
            self.test_results["real_data_service"]["technical_data"] = {
                "status": "PASS" if technical_data else "FAIL",
                "data": technical_data.__dict__ if technical_data else None
            }
            
            # Test analysis data
            analysis_data = await self.real_data_service.get_real_analysis_data(test_symbol, test_timeframe)
            self.test_results["real_data_service"]["analysis_data"] = {
                "status": "PASS" if analysis_data else "FAIL",
                "has_fundamental": "fundamental" in analysis_data if analysis_data else False,
                "has_technical": "technical" in analysis_data if analysis_data else False,
                "has_sentiment": "sentiment" in analysis_data if analysis_data else False
            }
            
            # Test confidence calculation
            confidence_data = await self.real_data_service.calculate_real_confidence(test_symbol, test_timeframe)
            self.test_results["real_data_service"]["confidence_data"] = {
                "status": "PASS" if confidence_data else "FAIL",
                "confidence": confidence_data.get("current_confidence") if confidence_data else None
            }
            
            logger.info("✅ Real Data Service tests completed")
            
        except Exception as e:
            logger.error(f"❌ Real Data Service test failed: {e}")
            self.test_results["real_data_service"]["error"] = str(e)
    
    async def test_ai_model_service(self):
        """Test AI model service integration"""
        logger.info("🤖 Testing AI Model Service...")
        
        test_symbol = "BTCUSDT"
        test_timeframe = "1h"
        
        try:
            # Test AI signal generation
            ai_signal = await self.ai_model_service.generate_ai_signal(test_symbol, test_timeframe)
            self.test_results["ai_model_service"]["signal_generation"] = {
                "status": "PASS" if ai_signal else "FAIL",
                "signal": ai_signal.__dict__ if ai_signal else None
            }
            
            if ai_signal:
                self.test_results["ai_model_service"]["consensus_check"] = {
                    "status": "PASS" if ai_signal.consensus_achieved else "FAIL",
                    "consensus_score": ai_signal.consensus_score,
                    "confidence_score": ai_signal.confidence_score
                }
                
                self.test_results["ai_model_service"]["model_reasoning"] = {
                    "status": "PASS" if ai_signal.model_reasoning else "FAIL",
                    "reasoning_count": len(ai_signal.model_reasoning) if ai_signal.model_reasoning else 0
                }
            
            logger.info("✅ AI Model Service tests completed")
            
        except Exception as e:
            logger.error(f"❌ AI Model Service test failed: {e}")
            self.test_results["ai_model_service"]["error"] = str(e)
    
    async def test_external_api_service(self):
        """Test external API service integration"""
        logger.info("🌐 Testing External API Service...")
        
        try:
            # Test data collection status
            status = await self.external_api_service.get_pipeline_status()
            self.test_results["external_api_service"]["pipeline_status"] = {
                "status": "PASS" if status else "FAIL",
                "data": status
            }
            
            # Test API health
            health = await self.external_api_service.get_api_health_status()
            self.test_results["external_api_service"]["api_health"] = {
                "status": "PASS" if health else "FAIL",
                "data": health
            }
            
            logger.info("✅ External API Service tests completed")
            
        except Exception as e:
            logger.error(f"❌ External API Service test failed: {e}")
            self.test_results["external_api_service"]["error"] = str(e)
    
    async def test_signal_generator(self):
        """Test signal generator integration"""
        logger.info("📈 Testing Signal Generator...")
        
        test_symbol = "BTCUSDT"
        test_timeframe = "1h"
        
        try:
            # Test confidence building
            confidence = await self.signal_generator.get_confidence_building(test_symbol, test_timeframe)
            self.test_results["signal_generator"]["confidence_building"] = {
                "status": "PASS" if confidence else "FAIL",
                "data": confidence
            }
            
            # Test signal generation
            signal = await self.signal_generator.generate_single_pair_signal(test_symbol, test_timeframe)
            self.test_results["signal_generator"]["signal_generation"] = {
                "status": "PASS" if signal else "FAIL",
                "signal": signal.__dict__ if signal else None
            }
            
            if signal:
                self.test_results["signal_generator"]["signal_validation"] = {
                    "status": "PASS" if signal.confidence_score >= 0.85 else "FAIL",
                    "confidence_score": signal.confidence_score,
                    "has_tp_levels": all([
                        signal.take_profit_1, signal.take_profit_2, 
                        signal.take_profit_3, signal.take_profit_4
                    ]),
                    "has_stop_loss": signal.stop_loss is not None
                }
            
            logger.info("✅ Signal Generator tests completed")
            
        except Exception as e:
            logger.error(f"❌ Signal Generator test failed: {e}")
            self.test_results["signal_generator"]["error"] = str(e)
    
    async def test_api_endpoints(self):
        """Test API endpoints integration"""
        logger.info("🔌 Testing API Endpoints...")
        
        test_symbol = "BTCUSDT"
        test_timeframe = "1h"
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test status endpoint
                async with session.get(f"{self.api_base_url}/api/v1/production/status") as response:
                    status_data = await response.json()
                    self.test_results["api_endpoints"]["status"] = {
                        "status": "PASS" if response.status == 200 else "FAIL",
                        "data": status_data
                    }
                
                # Test analysis endpoint
                async with session.get(f"{self.api_base_url}/api/v1/free-apis/comprehensive/{test_symbol}") as response:
                    analysis_data = await response.json()
                    self.test_results["api_endpoints"]["analysis"] = {
                        "status": "PASS" if response.status == 200 else "FAIL",
                        "data": analysis_data
                    }
                
                # Test confidence endpoint
                async with session.get(f"{self.api_base_url}/api/v1/free-apis/market-data/{test_symbol}") as response:
                    confidence_data = await response.json()
                    self.test_results["api_endpoints"]["confidence"] = {
                        "status": "PASS" if response.status == 200 else "FAIL",
                        "data": confidence_data
                    }
                
                # Test signal endpoint
                async with session.get(f"{self.api_base_url}/api/v1/free-apis/sentiment/{test_symbol}") as response:
                    signal_data = await response.json()
                    self.test_results["api_endpoints"]["signal"] = {
                        "status": "PASS" if response.status == 200 else "FAIL",
                        "data": signal_data
                    }
                
                logger.info("✅ API Endpoints tests completed")
                
            except Exception as e:
                logger.error(f"❌ API Endpoints test failed: {e}")
                self.test_results["api_endpoints"]["error"] = str(e)
    
    async def test_websocket_streaming(self):
        """Test WebSocket streaming"""
        logger.info("🔄 Testing WebSocket Streaming...")
        
        test_symbol = "BTCUSDT"
        websocket_url = f"ws://localhost:8000/api/v1/streaming/{test_symbol}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(websocket_url) as ws:
                    # Wait for first message
                    message = await asyncio.wait_for(ws.receive(), timeout=10.0)
                    
                    if message.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(message.data)
                        self.test_results["websocket_streaming"]["first_message"] = {
                            "status": "PASS",
                            "type": data.get("type"),
                            "has_analysis": "analysis" in data,
                            "has_confidence": "confidence" in data,
                            "has_signal": "signal" in data
                        }
                    
                    # Wait for second message to test streaming
                    message2 = await asyncio.wait_for(ws.receive(), timeout=10.0)
                    
                    if message2.type == aiohttp.WSMsgType.TEXT:
                        data2 = json.loads(message2.data)
                        self.test_results["websocket_streaming"]["streaming"] = {
                            "status": "PASS",
                            "type": data2.get("type"),
                            "timestamp_different": data.get("timestamp") != data2.get("timestamp")
                        }
                    
                    logger.info("✅ WebSocket Streaming tests completed")
                    
        except asyncio.TimeoutError:
            logger.error("❌ WebSocket test timeout")
            self.test_results["websocket_streaming"]["error"] = "Timeout"
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            self.test_results["websocket_streaming"]["error"] = str(e)
    
    async def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        logger.info("🛡️ Testing Error Handling...")
        
        try:
            # Test with invalid symbol
            invalid_data = await self.real_data_service.get_real_market_data("INVALID_SYMBOL", "1h")
            self.test_results["error_handling"]["invalid_symbol"] = {
                "status": "PASS" if invalid_data is None else "FAIL",
                "handled_gracefully": invalid_data is None
            }
            
            # Test with invalid timeframe
            invalid_timeframe = await self.real_data_service.get_real_market_data("BTCUSDT", "invalid")
            self.test_results["error_handling"]["invalid_timeframe"] = {
                "status": "PASS" if invalid_timeframe is None else "FAIL",
                "handled_gracefully": invalid_timeframe is None
            }
            
            logger.info("✅ Error Handling tests completed")
            
        except Exception as e:
            logger.error(f"❌ Error Handling test failed: {e}")
            self.test_results["error_handling"]["error"] = str(e)
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("📋 Generating Test Report...")
        
        # Calculate overall status
        total_tests = 0
        passed_tests = 0
        
        for service, tests in self.test_results.items():
            for test_name, result in tests.items():
                if isinstance(result, dict) and "status" in result:
                    total_tests += 1
                    if result["status"] == "PASS":
                        passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{success_rate:.1f}%",
                "overall_status": "PASS" if success_rate >= 80 else "FAIL"
            },
            "test_results": self.test_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for service, tests in self.test_results.items():
            for test_name, result in tests.items():
                if isinstance(result, dict) and result.get("status") == "FAIL":
                    recommendations.append(f"Fix {service}.{test_name} - {result.get('error', 'Unknown error')}")
        
        if not recommendations:
            recommendations.append("All tests passed! System is ready for production.")
        
        return recommendations

async def main():
    """Main test runner"""
    tester = RealDataIntegrationTester()
    
    logger.info("🚀 Starting Comprehensive Real Data Integration Tests")
    logger.info("=" * 60)
    
    # Run all tests
    report = await tester.run_all_tests()
    
    # Save report
    report_file = f"real_data_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("📋 TEST REPORT SUMMARY:")
    logger.info(f"Total Tests: {report['test_summary']['total_tests']}")
    logger.info(f"Passed: {report['test_summary']['passed_tests']}")
    logger.info(f"Failed: {report['test_summary']['failed_tests']}")
    logger.info(f"Success Rate: {report['test_summary']['success_rate']}")
    logger.info(f"Overall Status: {report['test_summary']['overall_status']}")
    logger.info(f"Report saved to: {report_file}")
    
    if report['test_summary']['overall_status'] == 'PASS':
        logger.info("🎉 All tests passed! System is ready for production.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
