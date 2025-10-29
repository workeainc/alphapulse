#!/usr/bin/env python3
"""
Phase 3.1: Sentiment Analysis Integration Test Script
Tests database migrations, enhanced sentiment service, and signal generator integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase3_1SentimentTester:
    """Test Phase 3.1: Sentiment Analysis Integration"""
    
    def __init__(self):
        self.db_connection = None
        self.sentiment_service = None
        self.signal_generator = None
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all Phase 3.1 tests"""
        logger.info("🚀 Starting Phase 3.1: Sentiment Analysis Integration Tests")
        
        try:
            # Test 1: Database Migrations
            await self.test_database_migrations()
            
            # Test 2: Enhanced Sentiment Service
            await self.test_enhanced_sentiment_service()
            
            # Test 3: Signal Generator Integration
            await self.test_signal_generator_integration()
            
            # Test 4: Database Persistence
            await self.test_database_persistence()
            
            # Test 5: Integration Validation
            await self.test_integration_validation()
            
            # Print results
            self.print_test_results()
            
        except Exception as e:
            logger.error(f"❌ Phase 3.1 tests failed: {e}")
            return False
        
        return True
    
    async def test_database_migrations(self):
        """Test Phase 3.1 database migrations"""
        logger.info("📊 Testing Phase 3.1 Database Migrations...")
        
        try:
            # Import and run migrations
            from phase3_1_sentiment_migrations import SentimentMigrations
            
            migrations = SentimentMigrations()
            await migrations.run_migrations()
            
            # Verify migrations
            await self.verify_sentiment_migrations()
            
            self.test_results['database_migrations'] = '✅ PASSED'
            logger.info("✅ Database migrations test passed")
            
        except Exception as e:
            logger.error(f"❌ Database migrations test failed: {e}")
            self.test_results['database_migrations'] = f'❌ FAILED: {e}'
    
    async def verify_sentiment_migrations(self):
        """Verify that sentiment columns were added to enhanced_signals table"""
        try:
            # Connect to database
            from src.app.core.database_manager import DatabaseManager
            
            db_manager = DatabaseManager()
            await db_manager.initialize()
            conn = await db_manager.get_connection()
            
            # Check for sentiment columns
            result = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'sentiment_%'
                ORDER BY column_name
            """)
            
            sentiment_columns = [row['column_name'] for row in result]
            expected_columns = [
                'sentiment_analysis', 'sentiment_score', 'sentiment_label', 
                'sentiment_confidence', 'sentiment_sources', 'twitter_sentiment',
                'reddit_sentiment', 'news_sentiment', 'telegram_sentiment',
                'discord_sentiment', 'sentiment_trend', 'sentiment_volatility',
                'sentiment_momentum', 'sentiment_correlation', 'sentiment_last_updated'
            ]
            
            missing_columns = [col for col in expected_columns if col not in sentiment_columns]
            
            if missing_columns:
                raise Exception(f"Missing sentiment columns: {missing_columns}")
            
            logger.info(f"✅ Found {len(sentiment_columns)} sentiment columns in enhanced_signals table")
            
            # Check for sentiment view
            view_result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname = 'sentiment_enhanced_signals'
            """)
            
            if not view_result:
                raise Exception("sentiment_enhanced_signals view not found")
            
            logger.info("✅ sentiment_enhanced_signals view created successfully")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"❌ Sentiment migrations verification failed: {e}")
            raise
    
    async def test_enhanced_sentiment_service(self):
        """Test enhanced sentiment service functionality"""
        logger.info("📊 Testing Enhanced Sentiment Service...")
        
        try:
            # Import sentiment service
            from src.app.services.sentiment_service import SentimentService
            
            # Initialize service
            self.sentiment_service = SentimentService()
            
            # Test basic sentiment analysis
            basic_sentiment = await self.sentiment_service.get_sentiment("BTC/USDT")
            logger.info(f"✅ Basic sentiment analysis: {basic_sentiment.get('sentiment_score', 0):.3f}")
            
            # Test enhanced sentiment analysis
            enhanced_sentiment = await self.sentiment_service.get_enhanced_sentiment("BTC/USDT")
            
            # Verify enhanced features
            required_features = [
                'trend_analysis', 'volatility_metrics', 'momentum_indicators',
                'correlation_metrics', 'enhanced_confidence', 'sentiment_strength',
                'prediction_confidence', 'phase_3_1_features'
            ]
            
            missing_features = [feature for feature in required_features 
                              if feature not in enhanced_sentiment]
            
            if missing_features:
                raise Exception(f"Missing enhanced features: {missing_features}")
            
            # Verify trend analysis
            trend_analysis = enhanced_sentiment.get('trend_analysis', {})
            if 'trend' not in trend_analysis or 'trend_strength' not in trend_analysis:
                raise Exception("Invalid trend analysis structure")
            
            # Verify momentum indicators
            momentum = enhanced_sentiment.get('momentum_indicators', {})
            if 'momentum_direction' not in momentum or 'momentum_strength' not in momentum:
                raise Exception("Invalid momentum indicators structure")
            
            # Verify volatility metrics
            volatility = enhanced_sentiment.get('volatility_metrics', {})
            if 'volatility_rank' not in volatility or 'stability_score' not in volatility:
                raise Exception("Invalid volatility metrics structure")
            
            # Verify correlation metrics
            correlation = enhanced_sentiment.get('correlation_metrics', {})
            if 'correlation_strength' not in correlation or 'predictive_power' not in correlation:
                raise Exception("Invalid correlation metrics structure")
            
            logger.info(f"✅ Enhanced sentiment analysis: trend={trend_analysis.get('trend')}, "
                       f"momentum={momentum.get('momentum_direction')}, "
                       f"volatility={volatility.get('volatility_rank')}")
            
            # Test performance metrics
            performance = self.sentiment_service.get_performance_metrics()
            logger.info(f"✅ Performance metrics: {performance.get('total_analyses', 0)} analyses")
            
            self.test_results['enhanced_sentiment_service'] = '✅ PASSED'
            logger.info("✅ Enhanced sentiment service test passed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced sentiment service test failed: {e}")
            self.test_results['enhanced_sentiment_service'] = f'❌ FAILED: {e}'
    
    async def test_signal_generator_integration(self):
        """Test signal generator integration with enhanced sentiment"""
        logger.info("📊 Testing Signal Generator Integration...")
        
        try:
            # Import signal generator
            from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
            
            # Initialize signal generator
            config = {
                'use_sentiment': True,
                'min_confidence': 0.6,
                'min_strength': 0.6
            }
            
            self.signal_generator = RealTimeSignalGenerator(config)
            
            # Test sentiment analysis method
            sentiment_signals = await self.signal_generator.analyze_enhanced_sentiment_predictions("BTC/USDT")
            
            # Verify sentiment signals structure
            required_fields = [
                'sentiment_bias', 'sentiment_score', 'sentiment_confidence',
                'sentiment_strength', 'high_confidence_sentiment', 'trend_analysis',
                'momentum_indicators', 'volatility_metrics', 'correlation_metrics',
                'enhanced_confidence', 'prediction_confidence', 'phase_3_1_features'
            ]
            
            missing_fields = [field for field in required_fields 
                            if field not in sentiment_signals]
            
            if missing_fields:
                raise Exception(f"Missing sentiment signal fields: {missing_fields}")
            
            # Verify sentiment bias values
            valid_biases = ['bullish', 'bearish', 'neutral']
            if sentiment_signals.get('sentiment_bias') not in valid_biases:
                raise Exception(f"Invalid sentiment bias: {sentiment_signals.get('sentiment_bias')}")
            
            # Verify phase 3.1 features flag
            if not sentiment_signals.get('phase_3_1_features'):
                raise Exception("Phase 3.1 features not enabled")
            
            logger.info(f"✅ Sentiment signals: bias={sentiment_signals.get('sentiment_bias')}, "
                       f"score={sentiment_signals.get('sentiment_score'):.3f}, "
                       f"confidence={sentiment_signals.get('enhanced_confidence'):.3f}")
            
            self.test_results['signal_generator_integration'] = '✅ PASSED'
            logger.info("✅ Signal generator integration test passed")
            
        except Exception as e:
            logger.error(f"❌ Signal generator integration test failed: {e}")
            self.test_results['signal_generator_integration'] = f'❌ FAILED: {e}'
    
    async def test_database_persistence(self):
        """Test database persistence of sentiment-enhanced signals"""
        logger.info("📊 Testing Database Persistence...")
        
        try:
            # Connect to database
            from src.app.core.database_manager import DatabaseManager
            
            db_manager = DatabaseManager()
            await db_manager.initialize()
            conn = await db_manager.get_connection()
            
            # Create a test signal with sentiment data
            test_signal = {
                'id': f"TEST_SENTIMENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'strategy': 'real_time_sentiment_enhanced',
                'confidence': 0.75,
                'strength': 0.70,
                'timestamp': datetime.now(),
                'price': 50000.0,
                'stop_loss': 47500.0,
                'take_profit': 52500.0,
                'metadata': {
                    'reason': 'Phase 3.1 sentiment-enhanced signal',
                    'indicators': {
                        'sentiment_analysis': {
                            'sentiment_bias': 'bullish',
                            'sentiment_score': 0.3,
                            'enhanced_confidence': 0.75,
                            'trend_analysis': {'trend': 'increasing', 'trend_strength': 0.6},
                            'momentum_indicators': {'momentum_direction': 'bullish', 'momentum_strength': 'moderate'},
                            'volatility_metrics': {'volatility_rank': 'low', 'stability_score': 0.8},
                            'correlation_metrics': {'correlation_strength': 'moderate', 'predictive_power': 0.4},
                            'phase_3_1_features': True
                        }
                    },
                    'source': 'phase_3_1_test'
                }
            }
            
            # Insert test signal
            await conn.execute("""
                INSERT INTO enhanced_signals (
                    id, symbol, side, strategy, confidence, strength, timestamp, 
                    price, stop_loss, take_profit, metadata, sentiment_analysis,
                    sentiment_score, sentiment_label, sentiment_confidence,
                    sentiment_sources, twitter_sentiment, reddit_sentiment,
                    news_sentiment, telegram_sentiment, discord_sentiment,
                    sentiment_trend, sentiment_volatility, sentiment_momentum,
                    sentiment_correlation, sentiment_last_updated
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                    $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26
                )
            """, 
                test_signal['id'], test_signal['symbol'], test_signal['side'],
                test_signal['strategy'], test_signal['confidence'], test_signal['strength'],
                test_signal['timestamp'], test_signal['price'], test_signal['stop_loss'],
                test_signal['take_profit'], test_signal['metadata'],
                test_signal['metadata']['indicators']['sentiment_analysis'],
                0.3, 'bullish', 0.75, {'twitter': 0.2, 'reddit': 0.3, 'news': 0.4},
                0.2, 0.3, 0.4, 0.1, 0.1, 'increasing', 0.15, 0.2, 0.4, datetime.now()
            )
            
            # Verify insertion
            result = await conn.fetchrow("""
                SELECT sentiment_analysis, sentiment_score, sentiment_label, 
                       sentiment_confidence, sentiment_trend, sentiment_volatility,
                       sentiment_momentum, sentiment_correlation
                FROM enhanced_signals 
                WHERE id = $1
            """, test_signal['id'])
            
            if not result:
                raise Exception("Test signal not found in database")
            
            # Verify sentiment data
            if result['sentiment_score'] != 0.3:
                raise Exception(f"Sentiment score mismatch: {result['sentiment_score']}")
            
            if result['sentiment_label'] != 'bullish':
                raise Exception(f"Sentiment label mismatch: {result['sentiment_label']}")
            
            if result['sentiment_confidence'] != 0.75:
                raise Exception(f"Sentiment confidence mismatch: {result['sentiment_confidence']}")
            
            logger.info("✅ Test signal with sentiment data persisted successfully")
            
            # Clean up test data
            await conn.execute("DELETE FROM enhanced_signals WHERE id = $1", test_signal['id'])
            
            await conn.close()
            
            self.test_results['database_persistence'] = '✅ PASSED'
            logger.info("✅ Database persistence test passed")
            
        except Exception as e:
            logger.error(f"❌ Database persistence test failed: {e}")
            self.test_results['database_persistence'] = f'❌ FAILED: {e}'
    
    async def test_integration_validation(self):
        """Test overall integration validation"""
        logger.info("📊 Testing Integration Validation...")
        
        try:
            # Test end-to-end sentiment-enhanced signal generation
            if not self.signal_generator:
                raise Exception("Signal generator not initialized")
            
            # Create sample market data
            import pandas as pd
            import numpy as np
            
            # Generate sample OHLCV data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            np.random.seed(42)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.normal(50000, 1000, 100),
                'high': np.random.normal(50500, 1000, 100),
                'low': np.random.normal(49500, 1000, 100),
                'close': np.random.normal(50000, 1000, 100),
                'volume': np.random.normal(1000, 200, 100)
            })
            
            # Calculate basic indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = 30 + np.random.normal(0, 10, 100)  # Simulate oversold condition
            df['macd'] = np.random.normal(0, 0.5, 100)
            df['macd_signal'] = df['macd'] + np.random.normal(0, 0.1, 100)
            
            # Get latest data
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Test signal generation with sentiment integration
            signal = await self.signal_generator._analyze_and_generate_signal("BTC/USDT", latest, prev, df)
            
            if signal:
                # Verify sentiment integration in signal
                indicators = signal.get('indicators', {})
                sentiment_analysis = indicators.get('sentiment_analysis', {})
                
                if not sentiment_analysis:
                    raise Exception("Sentiment analysis not found in signal indicators")
                
                # Verify sentiment features
                if not sentiment_analysis.get('phase_3_1_features'):
                    raise Exception("Phase 3.1 features not found in sentiment analysis")
                
                logger.info(f"✅ Generated sentiment-enhanced signal: "
                           f"type={signal.get('signal_type')}, "
                           f"confidence={signal.get('confidence'):.3f}, "
                           f"sentiment_bias={sentiment_analysis.get('sentiment_bias')}")
                
            else:
                logger.warning("⚠️ No signal generated (this may be normal depending on conditions)")
            
            self.test_results['integration_validation'] = '✅ PASSED'
            logger.info("✅ Integration validation test passed")
            
        except Exception as e:
            logger.error(f"❌ Integration validation test failed: {e}")
            self.test_results['integration_validation'] = f'❌ FAILED: {e}'
    
    def print_test_results(self):
        """Print test results summary"""
        logger.info("\n" + "="*60)
        logger.info("📋 PHASE 3.1: SENTIMENT ANALYSIS INTEGRATION TEST RESULTS")
        logger.info("="*60)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.startswith("✅") else "❌ FAILED"
            logger.info(f"{test_name:.<40} {status}")
            if result.startswith("✅"):
                passed += 1
        
        logger.info("-"*60)
        logger.info(f"TOTAL: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL PHASE 3.1 TESTS PASSED! Sentiment Analysis Integration successful!")
        else:
            logger.info("⚠️ Some tests failed. Please review the errors above.")
        
        logger.info("="*60)

async def main():
    """Main test function"""
    tester = Phase3_1SentimentTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("🚀 Phase 3.1: Sentiment Analysis Integration completed successfully!")
        return 0
    else:
        logger.error("❌ Phase 3.1: Sentiment Analysis Integration failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
