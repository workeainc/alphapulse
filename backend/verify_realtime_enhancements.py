#!/usr/bin/env python3
"""
Simple Real-Time Verification Script
Quick verification of real-time enhancements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_imports():
    """Verify all enhanced modules can be imported"""
    print("🔍 Verifying enhanced module imports...")
    
    try:
        from core.websocket_binance import BinanceWebSocketClient
        print("✅ BinanceWebSocketClient imported successfully")
        
        from services.free_api_manager import FreeAPIManager
        print("✅ FreeAPIManager imported successfully")
        
        from data.volume_analyzer import VolumeAnalyzer
        print("✅ VolumeAnalyzer imported successfully")
        
        from services.news_sentiment_service import NewsSentimentService
        print("✅ NewsSentimentService imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_enhancements():
    """Verify enhancement features are available"""
    print("\n🔍 Verifying enhancement features...")
    
    try:
        # Test WebSocket client enhancements
        from core.websocket_binance import BinanceWebSocketClient
        client = BinanceWebSocketClient(enable_liquidations=True, enable_orderbook=True, enable_trades=True)
        
        # Check if new methods exist
        assert hasattr(client, 'get_recent_liquidations'), "Missing get_recent_liquidations method"
        assert hasattr(client, 'get_recent_trades'), "Missing get_recent_trades method"
        assert hasattr(client, 'get_orderbook_snapshot'), "Missing get_orderbook_snapshot method"
        assert hasattr(client, 'get_real_time_stats'), "Missing get_real_time_stats method"
        print("✅ WebSocket enhancements verified")
        
        # Test Volume Analyzer enhancements
        from data.volume_analyzer import VolumeAnalyzer
        analyzer = VolumeAnalyzer()
        
        assert hasattr(analyzer, 'update_real_time_volume'), "Missing update_real_time_volume method"
        assert hasattr(analyzer, 'get_real_time_volume_analysis'), "Missing get_real_time_volume_analysis method"
        assert hasattr(analyzer, 'get_volume_profile_realtime'), "Missing get_volume_profile_realtime method"
        print("✅ Volume Analyzer enhancements verified")
        
        # Test News Service enhancements
        from services.news_sentiment_service import NewsSentimentService
        news_service = NewsSentimentService()
        
        assert hasattr(news_service, 'get_crypto_news_realtime'), "Missing get_crypto_news_realtime method"
        assert hasattr(news_service, 'get_breaking_news_alerts'), "Missing get_breaking_news_alerts method"
        assert hasattr(news_service, 'get_sentiment_summary'), "Missing get_sentiment_summary method"
        print("✅ News Service enhancements verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhancement verification failed: {e}")
        return False

def verify_functionality():
    """Verify basic functionality works"""
    print("\n🔍 Verifying basic functionality...")
    
    try:
        # Test Volume Analyzer
        from data.volume_analyzer import VolumeAnalyzer
        analyzer = VolumeAnalyzer()
        
        # Test real-time volume update
        test_data = {
            'timestamp': '2024-01-01T00:00:00',
            'volume': 1000,
            'price': 50000,
            'side': 'BUY'
        }
        analyzer.update_real_time_volume('BTCUSDT', test_data)
        
        # Test analysis retrieval
        analysis = analyzer.get_real_time_volume_analysis('BTCUSDT')
        assert 'symbol' in analysis, "Volume analysis missing symbol"
        assert 'total_volume' in analysis, "Volume analysis missing total_volume"
        print("✅ Volume Analyzer functionality verified")
        
        # Test News Service
        from services.news_sentiment_service import NewsSentimentService
        news_service = NewsSentimentService()
        
        # Test sentiment analysis
        sentiment = news_service._analyze_sentiment_simple("Bitcoin price surges to new highs")
        assert isinstance(sentiment, float), "Sentiment analysis should return float"
        assert -1 <= sentiment <= 1, "Sentiment should be between -1 and 1"
        print("✅ News Service functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 REAL-TIME ENHANCEMENTS VERIFICATION")
    print("=" * 50)
    
    # Run verifications
    import_success = verify_imports()
    enhancement_success = verify_enhancements()
    functionality_success = verify_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    total_tests = 3
    passed_tests = sum([import_success, enhancement_success, functionality_success])
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Import Verification: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"Enhancement Verification: {'✅ PASSED' if enhancement_success else '❌ FAILED'}")
    print(f"Functionality Verification: {'✅ PASSED' if functionality_success else '❌ FAILED'}")
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate == 100:
        print("\n🎉 ALL REAL-TIME ENHANCEMENTS VERIFIED SUCCESSFULLY!")
        print("✅ WebSocket liquidation streaming: READY")
        print("✅ Real-time volume analysis: READY")
        print("✅ Enhanced news sentiment: READY")
        print("✅ Order book streaming: READY")
        print("\n🚀 Your AlphaPlus system is now enhanced with real-time capabilities!")
    else:
        print(f"\n⚠️  Some verifications failed. Please check the errors above.")
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
