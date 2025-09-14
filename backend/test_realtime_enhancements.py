#!/usr/bin/env python3
"""
Real-Time Enhancements Test Suite
Tests all the enhanced real-time functionality
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any

# Import enhanced components
from core.websocket_binance import BinanceWebSocketClient
from services.free_api_manager import FreeAPIManager
from data.volume_analyzer import VolumeAnalyzer
from services.news_sentiment_service import NewsSentimentService

logger = logging.getLogger(__name__)

class RealTimeEnhancementsTest:
    """Test suite for real-time enhancements"""
    
    def __init__(self):
        self.test_results = {}
        self.websocket_client = None
        self.free_api_manager = None
        self.volume_analyzer = None
        self.news_service = None
        
    async def setup(self):
        """Setup test components"""
        try:
            logger.info("🚀 Setting up real-time enhancement tests...")
            
            # Initialize WebSocket client with enhanced features
            self.websocket_client = BinanceWebSocketClient(
                symbols=["BTCUSDT", "ETHUSDT"],
                timeframes=["1m", "5m"],
                enable_liquidations=True,
                enable_orderbook=True,
                enable_trades=True
            )
            
            # Initialize other components
            self.free_api_manager = FreeAPIManager()
            self.volume_analyzer = VolumeAnalyzer()
            self.news_service = NewsSentimentService()
            
            logger.info("✅ Test components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_websocket_enhancements(self) -> Dict[str, Any]:
        """Test enhanced WebSocket functionality"""
        logger.info("🔌 Testing WebSocket enhancements...")
        
        try:
            # Test connection
            connected = await self.websocket_client.connect()
            if not connected:
                return {"status": "failed", "error": "Failed to connect to WebSocket"}
            
            # Test message listening for 10 seconds
            messages_received = 0
            liquidation_events = 0
            orderbook_updates = 0
            trade_events = 0
            
            start_time = datetime.now()
            timeout = 10  # seconds
            
            async for message in self.websocket_client.listen():
                if message:
                    messages_received += 1
                    
                    if message.get('type') == 'liquidation':
                        liquidation_events += 1
                    elif message.get('type') == 'depth':
                        orderbook_updates += 1
                    elif message.get('type') == 'trade':
                        trade_events += 1
                
                # Check timeout
                if (datetime.now() - start_time).seconds >= timeout:
                    break
            
            # Get real-time stats
            stats = self.websocket_client.get_real_time_stats()
            
            # Test buffer access
            recent_liquidations = self.websocket_client.get_recent_liquidations(5)
            recent_trades = self.websocket_client.get_recent_trades(10)
            orderbook_snapshots = self.websocket_client.get_all_orderbook_snapshots()
            
            await self.websocket_client.disconnect()
            
            return {
                "status": "success",
                "messages_received": messages_received,
                "liquidation_events": liquidation_events,
                "orderbook_updates": orderbook_updates,
                "trade_events": trade_events,
                "stats": stats,
                "recent_liquidations_count": len(recent_liquidations),
                "recent_trades_count": len(recent_trades),
                "orderbook_symbols": len(orderbook_snapshots)
            }
            
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_liquidation_enhancements(self) -> Dict[str, Any]:
        """Test enhanced liquidation data processing"""
        logger.info("💥 Testing liquidation enhancements...")
        
        try:
            # Test liquidation data retrieval
            liquidation_data = await self.free_api_manager.get_liquidation_data("BTC")
            
            if not liquidation_data.get('success'):
                return {"status": "failed", "error": "Failed to get liquidation data"}
            
            events = liquidation_data.get('events', [])
            statistics = liquidation_data.get('statistics', {})
            
            return {
                "status": "success",
                "events_count": len(events),
                "statistics": statistics,
                "has_events": len(events) > 0,
                "has_statistics": bool(statistics)
            }
            
        except Exception as e:
            logger.error(f"❌ Liquidation test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_volume_analysis_enhancements(self) -> Dict[str, Any]:
        """Test enhanced volume analysis"""
        logger.info("📊 Testing volume analysis enhancements...")
        
        try:
            # Simulate real-time volume data
            test_symbol = "BTCUSDT"
            
            # Add some test volume data
            for i in range(20):
                volume_data = {
                    'timestamp': datetime.now(),
                    'volume': 1000 + i * 100,
                    'price': 50000 + i * 10,
                    'side': 'BUY' if i % 2 == 0 else 'SELL'
                }
                self.volume_analyzer.update_real_time_volume(test_symbol, volume_data)
            
            # Get real-time analysis
            volume_analysis = self.volume_analyzer.get_real_time_volume_analysis(test_symbol)
            volume_profile = self.volume_analyzer.get_volume_profile_realtime(test_symbol)
            
            return {
                "status": "success",
                "volume_analysis": volume_analysis,
                "volume_profile": volume_profile,
                "has_analysis": bool(volume_analysis),
                "has_profile": bool(volume_profile)
            }
            
        except Exception as e:
            logger.error(f"❌ Volume analysis test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_news_sentiment_enhancements(self) -> Dict[str, Any]:
        """Test enhanced news sentiment analysis"""
        logger.info("📰 Testing news sentiment enhancements...")
        
        try:
            # Test real-time news retrieval
            news_articles = await self.news_service.get_crypto_news_realtime("BTC", 10)
            
            # Test sentiment analysis
            sentiment_summary = self.news_service.get_sentiment_summary("BTC")
            
            # Test breaking news detection
            breaking_alerts = self.news_service.get_breaking_news_alerts(5)
            
            return {
                "status": "success",
                "news_count": len(news_articles),
                "sentiment_summary": sentiment_summary,
                "breaking_alerts_count": len(breaking_alerts),
                "has_news": len(news_articles) > 0,
                "has_sentiment": bool(sentiment_summary)
            }
            
        except Exception as e:
            logger.error(f"❌ News sentiment test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration of all real-time components"""
        logger.info("🔗 Testing real-time integration...")
        
        try:
            # Connect WebSocket
            await self.websocket_client.connect()
            
            # Collect data for 5 seconds
            integration_data = {
                'websocket_messages': 0,
                'liquidation_events': 0,
                'orderbook_updates': 0,
                'trade_events': 0,
                'volume_updates': 0
            }
            
            start_time = datetime.now()
            timeout = 5  # seconds
            
            async for message in self.websocket_client.listen():
                if message:
                    integration_data['websocket_messages'] += 1
                    
                    # Process different message types
                    if message.get('type') == 'liquidation':
                        integration_data['liquidation_events'] += 1
                        
                        # Update volume analyzer with liquidation data
                        volume_data = {
                            'timestamp': message.get('timestamp'),
                            'volume': message.get('quantity', 0),
                            'price': message.get('price', 0),
                            'side': message.get('side', 'unknown')
                        }
                        self.volume_analyzer.update_real_time_volume(
                            message.get('symbol', 'BTCUSDT'), 
                            volume_data
                        )
                        integration_data['volume_updates'] += 1
                        
                    elif message.get('type') == 'depth':
                        integration_data['orderbook_updates'] += 1
                        
                    elif message.get('type') == 'trade':
                        integration_data['trade_events'] += 1
                        
                        # Update volume analyzer with trade data
                        volume_data = {
                            'timestamp': message.get('timestamp'),
                            'volume': message.get('quantity', 0),
                            'price': message.get('price', 0),
                            'side': 'BUY' if message.get('is_buyer_maker', False) else 'SELL'
                        }
                        self.volume_analyzer.update_real_time_volume(
                            message.get('symbol', 'BTCUSDT'), 
                            volume_data
                        )
                        integration_data['volume_updates'] += 1
                
                # Check timeout
                if (datetime.now() - start_time).seconds >= timeout:
                    break
            
            await self.websocket_client.disconnect()
            
            # Get final analysis
            volume_analysis = self.volume_analyzer.get_real_time_volume_analysis("BTCUSDT")
            news_sentiment = self.news_service.get_sentiment_summary("BTC")
            
            return {
                "status": "success",
                "integration_data": integration_data,
                "volume_analysis": volume_analysis,
                "news_sentiment": news_sentiment,
                "integration_successful": integration_data['websocket_messages'] > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all real-time enhancement tests"""
        logger.info("🧪 Starting comprehensive real-time enhancement tests...")
        
        # Setup
        setup_success = await self.setup()
        if not setup_success:
            return {"status": "failed", "error": "Setup failed"}
        
        # Run individual tests
        tests = [
            ("websocket_enhancements", self.test_websocket_enhancements),
            ("liquidation_enhancements", self.test_liquidation_enhancements),
            ("volume_analysis_enhancements", self.test_volume_analysis_enhancements),
            ("news_sentiment_enhancements", self.test_news_sentiment_enhancements),
            ("integration", self.test_integration)
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"🧪 Running {test_name} test...")
            try:
                result = await test_func()
                results[test_name] = result
                
                if result.get('status') == 'success':
                    passed_tests += 1
                    logger.info(f"✅ {test_name} test passed")
                else:
                    logger.error(f"❌ {test_name} test failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} test error: {e}")
                results[test_name] = {"status": "failed", "error": str(e)}
        
        # Calculate overall results
        success_rate = (passed_tests / total_tests) * 100
        
        return {
            "status": "completed",
            "success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """Main test execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = RealTimeEnhancementsTest()
    results = await test_suite.run_all_tests()
    
    print("\n" + "="*80)
    print("🧪 REAL-TIME ENHANCEMENTS TEST RESULTS")
    print("="*80)
    print(f"Overall Status: {results['status']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Passed Tests: {results['passed_tests']}/{results['total_tests']}")
    print(f"Timestamp: {results['timestamp']}")
    print("\nDetailed Results:")
    
    for test_name, result in results['results'].items():
        status_emoji = "✅" if result.get('status') == 'success' else "❌"
        print(f"{status_emoji} {test_name}: {result.get('status', 'unknown')}")
        if result.get('error'):
            print(f"   Error: {result['error']}")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
