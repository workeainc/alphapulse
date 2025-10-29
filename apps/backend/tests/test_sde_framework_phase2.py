"""
Test Script for SDE Framework Phase 2 Implementation
Tests enhanced execution quality, news blackout, signal limits, and TP structure
"""

import asyncio
import logging
import asyncpg
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

async def test_phase2_database():
    """Test Phase 2 database tables and configurations"""
    logger.info("📋 Test 1: Phase 2 Database Integration")
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        
        # Check Phase 2 tables
        async with pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'sde_%'
                ORDER BY table_name
            """)
            
            phase2_tables = [
                'sde_news_blackout',
                'sde_signal_limits', 
                'sde_tp_structure',
                'sde_enhanced_execution',
                'sde_signal_queue',
                'sde_enhanced_performance'
            ]
            
            existing_tables = [table['table_name'] for table in tables]
            missing_tables = [table for table in phase2_tables if table not in existing_tables]
            
            if missing_tables:
                logger.error(f"❌ Missing Phase 2 tables: {missing_tables}")
                return False
            else:
                logger.info(f"✅ All Phase 2 tables exist: {len(phase2_tables)} tables")
            
            # Check Phase 2 configurations
            configs = await conn.fetch("""
                SELECT config_name, config_type 
                FROM sde_config 
                WHERE config_name LIKE 'sde_%_default'
                ORDER BY config_name
            """)
            
            phase2_configs = [
                'sde_news_blackout_default',
                'sde_signal_limits_default',
                'sde_tp_structure_default',
                'sde_enhanced_execution_default'
            ]
            
            existing_configs = [config['config_name'] for config in configs]
            missing_configs = [config for config in phase2_configs if config not in existing_configs]
            
            if missing_configs:
                logger.error(f"❌ Missing Phase 2 configurations: {missing_configs}")
                return False
            else:
                logger.info(f"✅ All Phase 2 configurations exist: {len(phase2_configs)} configs")
        
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 2 database test failed: {e}")
        return False

async def test_enhanced_execution_quality():
    """Test enhanced execution quality assessment"""
    logger.info("📋 Test 2: Enhanced Execution Quality Assessment")
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        from src.ai.sde_framework import SDEFramework
        
        sde_framework = SDEFramework(pool)
        
        # Test case 1: Good execution quality
        market_data_good = {
            'spread_atr_ratio': 0.08,
            'spread_percentage': 0.03,
            'atr_percentile': 50.0,
            'volatility_regime': 'normal',
            'impact_cost': 0.05,
            'orderbook_depth': 2000
        }
        
        execution_result = await sde_framework.assess_execution_quality(market_data_good)
        logger.info(f"Enhanced Execution Test 1 - Quality Score: {execution_result.quality_score:.2f}/10.0, All Gates Passed: {execution_result.all_gates_passed}")
        logger.info(f"  Breakdown: {execution_result.breakdown}")
        
        # Test case 2: Poor execution quality
        market_data_poor = {
            'spread_atr_ratio': 0.20,
            'spread_percentage': 0.08,
            'atr_percentile': 90.0,
            'volatility_regime': 'extreme',
            'impact_cost': 0.25,
            'orderbook_depth': 500
        }
        
        execution_result_2 = await sde_framework.assess_execution_quality(market_data_poor)
        logger.info(f"Enhanced Execution Test 2 - Quality Score: {execution_result_2.quality_score:.2f}/10.0, All Gates Passed: {execution_result_2.all_gates_passed}")
        logger.info(f"  Breakdown: {execution_result_2.breakdown}")
        
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced execution quality test failed: {e}")
        return False

async def test_news_blackout():
    """Test news blackout functionality"""
    logger.info("📋 Test 3: News Blackout Check")
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        from src.ai.sde_framework import SDEFramework
        
        sde_framework = SDEFramework(pool)
        
        # Insert test blackout data
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sde_news_blackout 
                (symbol, event_type, event_impact, event_title, event_description, start_time, end_time, blackout_active)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, 'BTCUSDT', 'news', 'high', 'Test High Impact News', 'Test news event', 
                 datetime.now() - timedelta(minutes=5), datetime.now() + timedelta(minutes=10), True)
        
        # Test blackout check
        current_time = datetime.now()
        blackout_result = await sde_framework.check_news_blackout('BTCUSDT', current_time)
        
        logger.info(f"News Blackout Test - Active: {blackout_result.blackout_active}")
        logger.info(f"  Event Type: {blackout_result.event_type}")
        logger.info(f"  Event Impact: {blackout_result.event_impact}")
        logger.info(f"  Reason: {blackout_result.blackout_reason}")
        
        # Test no blackout
        blackout_result_2 = await sde_framework.check_news_blackout('ETHUSDT', current_time)
        logger.info(f"News Blackout Test 2 - Active: {blackout_result_2.blackout_active}")
        
        # Clean up test data
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sde_news_blackout WHERE event_title = 'Test High Impact News'")
        
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ News blackout test failed: {e}")
        return False

async def test_signal_limits():
    """Test signal limits and quota management"""
    logger.info("📋 Test 4: Signal Limits Check")
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        from src.ai.sde_framework import SDEFramework
        
        sde_framework = SDEFramework(pool)
        
        # Test signal limits check
        limit_result = await sde_framework.check_signal_limits('test_account', 'BTCUSDT', '1h')
        
        logger.info(f"Signal Limits Test - Limit Exceeded: {limit_result.limit_exceeded}")
        logger.info(f"  Limit Type: {limit_result.limit_type}")
        logger.info(f"  Current Count: {limit_result.current_count}")
        logger.info(f"  Max Count: {limit_result.max_count}")
        
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal limits test failed: {e}")
        return False

async def test_tp_structure():
    """Test four TP structure calculation"""
    logger.info("📋 Test 5: TP Structure Calculation")
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        from src.ai.sde_framework import SDEFramework
        
        sde_framework = SDEFramework(pool)
        
        # Test LONG position
        entry_price = 50000.0
        stop_loss = 48000.0
        atr_value = 1000.0
        
        tp_result_long = await sde_framework.calculate_tp_structure(entry_price, stop_loss, atr_value, 'LONG')
        
        logger.info(f"TP Structure Test - LONG Position:")
        logger.info(f"  Entry: {entry_price}")
        logger.info(f"  Stop Loss: {tp_result_long.stop_loss}")
        logger.info(f"  TP1: {tp_result_long.tp1_price:.2f}")
        logger.info(f"  TP2: {tp_result_long.tp2_price:.2f}")
        logger.info(f"  TP3: {tp_result_long.tp3_price:.2f}")
        logger.info(f"  TP4: {tp_result_long.tp4_price:.2f}")
        logger.info(f"  Position Size: {tp_result_long.position_size:.4f}")
        logger.info(f"  Risk/Reward: {tp_result_long.risk_reward_ratio:.2f}")
        
        # Test SHORT position
        entry_price = 50000.0
        stop_loss = 52000.0
        
        tp_result_short = await sde_framework.calculate_tp_structure(entry_price, stop_loss, atr_value, 'SHORT')
        
        logger.info(f"TP Structure Test - SHORT Position:")
        logger.info(f"  Entry: {entry_price}")
        logger.info(f"  Stop Loss: {tp_result_short.stop_loss}")
        logger.info(f"  TP1: {tp_result_short.tp1_price:.2f}")
        logger.info(f"  TP2: {tp_result_short.tp2_price:.2f}")
        logger.info(f"  TP3: {tp_result_short.tp3_price:.2f}")
        logger.info(f"  TP4: {tp_result_short.tp4_price:.2f}")
        logger.info(f"  Position Size: {tp_result_short.position_size:.4f}")
        logger.info(f"  Risk/Reward: {tp_result_short.risk_reward_ratio:.2f}")
        
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ TP structure test failed: {e}")
        return False

async def test_end_to_end_phase2():
    """Test end-to-end Phase 2 functionality"""
    logger.info("📋 Test 6: End-to-End Phase 2 Processing")
    
    try:
        pool = await asyncpg.create_pool(**db_config)
        from src.ai.sde_framework import SDEFramework, ModelHeadResult, SignalDirection
        
        sde_framework = SDEFramework(pool)
        
        # Simulate complete Phase 2 signal processing
        symbol = "BTCUSDT"
        timeframe = "1h"
        account_id = "test_account"
        current_time = datetime.now()
        
        # Step 1: Model Consensus (Phase 1)
        head_results = {
            'head_a': ModelHeadResult(SignalDirection.LONG, 0.85, 0.85),
            'head_b': ModelHeadResult(SignalDirection.LONG, 0.82, 0.82),
            'head_c': ModelHeadResult(SignalDirection.LONG, 0.75, 0.75),
            'head_d': ModelHeadResult(SignalDirection.LONG, 0.80, 0.80)
        }
        
        consensus_result = await sde_framework.check_model_consensus(head_results)
        logger.info(f"Phase 2 E2E - Consensus: {'✅' if consensus_result.achieved else '❌'}")
        
        # Step 2: Confluence Score (Phase 1)
        analysis_results = {
            'support_resistance_quality': 0.8,
            'volume_confirmation': True,
            'htf_trend_strength': 0.75,
            'trend_alignment': True,
            'pattern_strength': 0.8,
            'breakout_confirmed': True
        }
        
        confluence_result = await sde_framework.calculate_confluence_score(analysis_results)
        logger.info(f"Phase 2 E2E - Confluence: {'✅' if confluence_result.gate_passed else '❌'} ({confluence_result.total_score:.2f}/8.0)")
        
        # Step 3: Enhanced Execution Quality (Phase 2)
        market_data = {
            'spread_atr_ratio': 0.10,
            'spread_percentage': 0.04,
            'atr_percentile': 60.0,
            'volatility_regime': 'normal',
            'impact_cost': 0.08,
            'orderbook_depth': 1500
        }
        
        execution_result = await sde_framework.assess_execution_quality(market_data)
        logger.info(f"Phase 2 E2E - Execution: {'✅' if execution_result.all_gates_passed else '❌'} ({execution_result.quality_score:.2f}/10.0)")
        
        # Step 4: News Blackout Check (Phase 2)
        blackout_result = await sde_framework.check_news_blackout(symbol, current_time)
        logger.info(f"Phase 2 E2E - News Blackout: {'❌' if blackout_result.blackout_active else '✅'}")
        
        # Step 5: Signal Limits Check (Phase 2)
        limit_result = await sde_framework.check_signal_limits(account_id, symbol, timeframe)
        logger.info(f"Phase 2 E2E - Signal Limits: {'❌' if limit_result.limit_exceeded else '✅'}")
        
        # Step 6: TP Structure Calculation (Phase 2)
        entry_price = 50000.0
        stop_loss = 48000.0
        atr_value = 1000.0
        
        tp_result = await sde_framework.calculate_tp_structure(entry_price, stop_loss, atr_value, 'LONG')
        logger.info(f"Phase 2 E2E - TP Structure: ✅ Calculated (RR: {tp_result.risk_reward_ratio:.2f})")
        
        # Final decision
        all_gates_passed = (
            consensus_result.achieved and
            confluence_result.gate_passed and
            execution_result.all_gates_passed and
            not blackout_result.blackout_active and
            not limit_result.limit_exceeded
        )
        
        logger.info(f"🎯 Phase 2 Final Decision: {'✅ EMIT SIGNAL' if all_gates_passed else '❌ REJECT SIGNAL'}")
        
        if all_gates_passed:
            logger.info(f"📊 Signal Details:")
            logger.info(f"  - Direction: LONG")
            logger.info(f"  - Entry: {entry_price}")
            logger.info(f"  - Stop Loss: {tp_result.stop_loss}")
            logger.info(f"  - TP1: {tp_result.tp1_price:.2f}")
            logger.info(f"  - TP2: {tp_result.tp2_price:.2f}")
            logger.info(f"  - TP3: {tp_result.tp3_price:.2f}")
            logger.info(f"  - TP4: {tp_result.tp4_price:.2f}")
            logger.info(f"  - Position Size: {tp_result.position_size:.4f}")
            logger.info(f"  - Risk/Reward: {tp_result.risk_reward_ratio:.2f}")
        
        await pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end Phase 2 test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("SDE FRAMEWORK PHASE 2 TESTING")
    logger.info("=" * 60)
    
    # Test 1: Phase 2 Database Integration
    test1_result = await test_phase2_database()
    
    # Test 2: Enhanced Execution Quality Assessment
    test2_result = await test_enhanced_execution_quality()
    
    # Test 3: News Blackout Check
    test3_result = await test_news_blackout()
    
    # Test 4: Signal Limits Check
    test4_result = await test_signal_limits()
    
    # Test 5: TP Structure Calculation
    test5_result = await test_tp_structure()
    
    # Test 6: End-to-End Phase 2 Processing
    test6_result = await test_end_to_end_phase2()
    
    # Summary
    logger.info("=" * 60)
    logger.info("PHASE 2 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Database Integration: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    logger.info(f"Enhanced Execution Quality: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    logger.info(f"News Blackout Check: {'✅ PASSED' if test3_result else '❌ FAILED'}")
    logger.info(f"Signal Limits Check: {'✅ PASSED' if test4_result else '❌ FAILED'}")
    logger.info(f"TP Structure Calculation: {'✅ PASSED' if test5_result else '❌ FAILED'}")
    logger.info(f"End-to-End Processing: {'✅ PASSED' if test6_result else '❌ FAILED'}")
    
    overall_result = all([test1_result, test2_result, test3_result, test4_result, test5_result, test6_result])
    logger.info(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_result else '❌ SOME TESTS FAILED'}")
    
    return overall_result

if __name__ == "__main__":
    asyncio.run(main())
