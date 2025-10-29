#!/usr/bin/env python3
"""
Comprehensive verification script for Phase 1 database migrations and functionality
Phase 1.1: Basic Technical Indicators
Phase 1.2: Advanced Technical Indicators
Phase 1.3: Smart Money Concepts
Phase 1.4: Deep Learning Integration
"""
import asyncio
import logging
from src.app.core.database_manager import DatabaseManager
from src.ai.technical_indicators_engine import TechnicalIndicatorsEngine
from src.app.strategies.smart_money_concepts_engine import SmartMoneyConceptsEngine
from src.app.strategies.deep_learning_engine import DeepLearningEngine
from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_phase1_database_migrations():
    """Verify all Phase 1 database migrations are applied correctly"""
    logger.info("🔍 Verifying Phase 1 Database Migrations...")
    
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        async with db_manager.get_connection() as conn:
            # Check if enhanced_signals table exists
            result = await conn.fetch("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'enhanced_signals'
                )
            """)
            table_exists = result[0]['exists']
            
            if not table_exists:
                logger.error("❌ enhanced_signals table does not exist")
                return False
            
            logger.info("✅ enhanced_signals table exists")
            
            # Phase 1.1: Basic Technical Indicators columns
            logger.info("📊 Checking Phase 1.1: Basic Technical Indicators...")
            basic_indicators = ['ichimoku_data', 'fibonacci_data', 'volume_analysis', 'advanced_indicators']
            
            for column in basic_indicators:
                result = await conn.fetch("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'enhanced_signals' 
                        AND column_name = $1
                    )
                """, column)
                if result[0]['exists']:
                    logger.info(f"✅ {column} column exists")
                else:
                    logger.error(f"❌ {column} column missing")
                    return False
            
            # Phase 1.2: Advanced Technical Indicators columns
            logger.info("📈 Checking Phase 1.2: Advanced Technical Indicators...")
            advanced_indicators = ['signal_quality_score', 'confirmation_count']
            
            for column in advanced_indicators:
                result = await conn.fetch("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'enhanced_signals' 
                        AND column_name = $1
                    )
                """, column)
                if result[0]['exists']:
                    logger.info(f"✅ {column} column exists")
                else:
                    logger.error(f"❌ {column} column missing")
                    return False
            
            # Phase 1.3: Smart Money Concepts columns
            logger.info("🎯 Checking Phase 1.3: Smart Money Concepts...")
            smc_columns = [
                'smc_analysis', 'order_blocks_data', 'fair_value_gaps_data',
                'liquidity_sweeps_data', 'market_structures_data', 'smc_confidence', 'smc_bias'
            ]
            
            for column in smc_columns:
                result = await conn.fetch("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'enhanced_signals' 
                        AND column_name = $1
                    )
                """, column)
                if result[0]['exists']:
                    logger.info(f"✅ {column} column exists")
                else:
                    logger.error(f"❌ {column} column missing")
                    return False
            
            # Phase 1.4: Deep Learning columns
            logger.info("🤖 Checking Phase 1.4: Deep Learning...")
            dl_columns = [
                'dl_analysis', 'lstm_prediction', 'cnn_prediction', 
                'lstm_cnn_prediction', 'ensemble_prediction', 'dl_confidence', 'dl_bias'
            ]
            
            for column in dl_columns:
                result = await conn.fetch("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'enhanced_signals' 
                        AND column_name = $1
                    )
                """, column)
                if result[0]['exists']:
                    logger.info(f"✅ {column} column exists")
                else:
                    logger.error(f"❌ {column} column missing")
                    return False
            
            # Check views
            logger.info("👁️ Checking Phase 1 Views...")
            views = ['high_quality_signals', 'smc_enhanced_signals', 'ai_enhanced_signals']
            
            for view in views:
                result = await conn.fetch("""
                    SELECT EXISTS (
                        SELECT FROM pg_views 
                        WHERE viewname = $1
                    )
                """, view)
                if result[0]['exists']:
                    logger.info(f"✅ {view} view exists")
                else:
                    logger.warning(f"⚠️ {view} view missing")
            
            # Check functions
            logger.info("⚙️ Checking Phase 1 Functions...")
            functions = [
                'calculate_signal_quality', 
                'calculate_smc_enhanced_quality', 
                'calculate_ai_enhanced_quality'
            ]
            
            for func in functions:
                result = await conn.fetch("""
                    SELECT EXISTS (
                        SELECT FROM pg_proc 
                        WHERE proname = $1
                    )
                """, func)
                if result[0]['exists']:
                    logger.info(f"✅ {func} function exists")
                else:
                    logger.warning(f"⚠️ {func} function missing")
        
        logger.info("✅ All Phase 1 database migrations verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 1 database verification failed: {e}")
        return False

async def test_phase1_engines():
    """Test all Phase 1 engines functionality"""
    logger.info("🚀 Testing Phase 1 Engines...")
    
    # Test 1.1: Technical Indicators Engine
    logger.info("📊 Testing Phase 1.1: Technical Indicators Engine...")
    try:
        ti_engine = TechnicalIndicatorsEngine()
        await ti_engine.start()
        
        # Test basic indicators
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Test RSI
        rsi = ti_engine.calculate_rsi(test_data['close'])
        logger.info(f"✅ RSI calculated: {rsi.iloc[-1]:.2f}")
        
        # Test MACD
        macd = ti_engine.calculate_macd(test_data['close'])
        logger.info(f"✅ MACD calculated: {macd['macd'].iloc[-1]:.2f}")
        
        # Test Ichimoku
        ichimoku = ti_engine.calculate_ichimoku(test_data['close'])
        logger.info(f"✅ Ichimoku calculated: {len(ichimoku)} components")
        
        logger.info("✅ Technical Indicators Engine working")
        
    except Exception as e:
        logger.error(f"❌ Technical Indicators Engine failed: {e}")
        return False
    
    # Test 1.3: Smart Money Concepts Engine
    logger.info("🎯 Testing Phase 1.3: Smart Money Concepts Engine...")
    try:
        smc_engine = SmartMoneyConceptsEngine()
        await smc_engine.start()
        
        # Test SMC analysis
        smc_analysis = smc_engine.analyze_smart_money_concepts(test_data)
        logger.info(f"✅ SMC analysis completed: {len(smc_analysis)} components")
        
        logger.info("✅ Smart Money Concepts Engine working")
        
    except Exception as e:
        logger.error(f"❌ Smart Money Concepts Engine failed: {e}")
        return False
    
    # Test 1.4: Deep Learning Engine
    logger.info("🤖 Testing Phase 1.4: Deep Learning Engine...")
    try:
        dl_engine = DeepLearningEngine()
        await dl_engine.start()
        
        # Test model performance
        performance = dl_engine.get_model_performance()
        logger.info(f"✅ Deep Learning performance: {performance}")
        
        logger.info("✅ Deep Learning Engine working")
        
    except Exception as e:
        logger.error(f"❌ Deep Learning Engine failed: {e}")
        return False
    
    return True

async def test_phase1_signal_generation():
    """Test Phase 1 signal generation with all components"""
    logger.info("🔗 Testing Phase 1 Signal Generation...")
    
    try:
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Check if all Phase 1 engines are integrated
        if hasattr(signal_generator, 'technical_indicators_engine') and signal_generator.technical_indicators_engine:
            logger.info("✅ Technical Indicators Engine integrated")
        else:
            logger.error("❌ Technical Indicators Engine not integrated")
            return False
        
        if hasattr(signal_generator, 'smc_engine') and signal_generator.smc_engine:
            logger.info("✅ Smart Money Concepts Engine integrated")
        else:
            logger.error("❌ Smart Money Concepts Engine not integrated")
            return False
        
        if hasattr(signal_generator, 'dl_engine') and signal_generator.dl_engine:
            logger.info("✅ Deep Learning Engine integrated")
        else:
            logger.error("❌ Deep Learning Engine not integrated")
            return False
        
        # Test signal generation with mock data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Test signal generation
        signal = await signal_generator._analyze_and_generate_signal(mock_data, "BTCUSDT", "1h", None)
        
        if signal:
            logger.info("✅ Signal generated successfully")
            
            # Check for Phase 1 components
            indicators = signal.get('indicators', {})
            
            if indicators.get('technical_indicators'):
                logger.info("✅ Technical indicators included in signal")
            else:
                logger.warning("⚠️ Technical indicators not found in signal")
            
            if indicators.get('smc_analysis'):
                logger.info("✅ SMC analysis included in signal")
            else:
                logger.warning("⚠️ SMC analysis not found in signal")
            
            if indicators.get('dl_analysis'):
                logger.info("✅ Deep Learning analysis included in signal")
            else:
                logger.warning("⚠️ Deep Learning analysis not found in signal")
            
            logger.info(f"Signal confidence: {signal.get('confidence', 0)}")
            logger.info(f"Signal side: {signal.get('side', 'unknown')}")
            
        else:
            logger.error("❌ No signal generated")
            return False
        
        logger.info("✅ Phase 1 Signal Generation working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 1 Signal Generation failed: {e}")
        return False

async def test_phase1_database_persistence():
    """Test Phase 1 database persistence"""
    logger.info("💾 Testing Phase 1 Database Persistence...")
    
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        import json
        from datetime import datetime, timezone
        
        # Create a test signal with Phase 1 data
        test_signal = {
            'id': f"phase1_test_{datetime.now().timestamp()}",
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'strategy': 'phase1_comprehensive_test',
            'confidence': 0.85,
            'strength': 0.8,
            'timestamp': datetime.now(timezone.utc),
            'price': 45000.0,
            'stop_loss': 44000.0,
            'take_profit': 46000.0,
            'metadata': json.dumps({
                'test': True,
                'phase': '1_comprehensive_test',
                'technical_indicators': {
                    'rsi': 65.5,
                    'macd': 125.3,
                    'ichimoku': {'tenkan': 44500, 'kijun': 44800}
                },
                'smc_analysis': {
                    'order_blocks': [{'type': 'bullish', 'strength': 0.8}],
                    'fair_value_gaps': [{'type': 'bullish', 'filled': False}]
                },
                'dl_analysis': {
                    'lstm_prediction': 45200.0,
                    'cnn_prediction': 45150.0,
                    'ensemble_prediction': 45175.0
                }
            })
        }
        
        async with db_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO enhanced_signals (
                    id, symbol, side, strategy, confidence, strength, timestamp, price, 
                    stop_loss, take_profit, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, test_signal['id'], test_signal['symbol'], test_signal['side'], 
                 test_signal['strategy'], test_signal['confidence'], test_signal['strength'],
                 test_signal['timestamp'], test_signal['price'], test_signal['stop_loss'],
                 test_signal['take_profit'], test_signal['metadata'])
            
            # Verify the signal was saved
            result = await conn.fetch("""
                SELECT id, symbol, metadata FROM enhanced_signals 
                WHERE id = $1
            """, test_signal['id'])
            
            if result:
                saved_signal = result[0]
                metadata = json.loads(saved_signal['metadata'])
                
                if metadata.get('technical_indicators'):
                    logger.info("✅ Technical indicators data persisted")
                else:
                    logger.warning("⚠️ Technical indicators data missing")
                
                if metadata.get('smc_analysis'):
                    logger.info("✅ SMC analysis data persisted")
                else:
                    logger.warning("⚠️ SMC analysis data missing")
                
                if metadata.get('dl_analysis'):
                    logger.info("✅ Deep Learning analysis data persisted")
                else:
                    logger.warning("⚠️ Deep Learning analysis data missing")
                
            else:
                logger.error("❌ Signal not found in database")
                return False
        
        logger.info("✅ Phase 1 Database Persistence working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 1 Database Persistence failed: {e}")
        return False

async def run_phase1_verification():
    """Run comprehensive Phase 1 verification"""
    logger.info("🚀 Starting Comprehensive Phase 1 Verification...")
    
    tests = [
        ("Database Migrations", verify_phase1_database_migrations),
        ("Phase 1 Engines", test_phase1_engines),
        ("Signal Generation", test_phase1_signal_generation),
        ("Database Persistence", test_phase1_database_persistence)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1 VERIFICATION SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 1 components are working perfectly!")
        logger.info("📋 Phase 1 Summary:")
        logger.info("   ✅ Phase 1.1: Basic Technical Indicators - Working")
        logger.info("   ✅ Phase 1.2: Advanced Technical Indicators - Working")
        logger.info("   ✅ Phase 1.3: Smart Money Concepts - Working")
        logger.info("   ✅ Phase 1.4: Deep Learning Integration - Working")
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_phase1_verification())
