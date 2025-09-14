#!/usr/bin/env python3
"""
Test SMC (Smart Money Concepts) Enhancement Implementation
Validates SMC pattern detection including Order Blocks, Fair Value Gaps, and Liquidity Sweeps
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from data.enhanced_real_time_pipeline import (
    EnhancedRealTimePipeline, 
    SMCOrderBlock, 
    SMCFairValueGap, 
    SMCLiquiditySweep, 
    SMCMarketStructure
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SMCEnhancementTester:
    """Test suite for SMC pattern detection enhancement"""
    
    def __init__(self):
        self.pipeline = EnhancedRealTimePipeline()
        
    def generate_test_data(self, pattern_type: str) -> pd.DataFrame:
        """Generate test data for different SMC patterns"""
        np.random.seed(42)  # For reproducible results
        
        # Base price data
        base_price = 45000
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        
        if pattern_type == 'order_block_bullish':
            # Generate bullish order block: strong move up followed by consolidation
            prices = []
            for i in range(100):
                if i < 80:
                    # Normal trading
                    price = base_price + np.random.normal(0, 100)
                elif i < 85:
                    # Strong bullish move (order block)
                    price = base_price + 500 + np.random.normal(0, 50)
                else:
                    # Consolidation after the move
                    price = base_price + 400 + np.random.normal(0, 30)
                prices.append(price)
                
        elif pattern_type == 'order_block_bearish':
            # Generate bearish order block: strong move down followed by consolidation
            prices = []
            for i in range(100):
                if i < 80:
                    # Normal trading
                    price = base_price + np.random.normal(0, 100)
                elif i < 85:
                    # Strong bearish move (order block)
                    price = base_price - 500 + np.random.normal(0, 50)
                else:
                    # Consolidation after the move
                    price = base_price - 400 + np.random.normal(0, 30)
                prices.append(price)
                
        elif pattern_type == 'fair_value_gap':
            # Generate fair value gap: price gap between candles
            prices = []
            for i in range(100):
                if i < 80:
                    # Normal trading
                    price = base_price + np.random.normal(0, 100)
                elif i == 80:
                    # First candle
                    price = base_price - 200
                elif i == 81:
                    # Second candle with gap
                    price = base_price + 200
                else:
                    # Continue trading
                    price = base_price + np.random.normal(0, 100)
                prices.append(price)
                
        elif pattern_type == 'liquidity_sweep':
            # Generate liquidity sweep: move beyond support/resistance then reversal
            prices = []
            support_level = base_price - 1000
            for i in range(100):
                if i < 80:
                    # Trading near support
                    price = support_level + np.random.normal(0, 200)
                elif i < 85:
                    # Sweep below support
                    price = support_level - 300 + np.random.normal(0, 50)
                else:
                    # Reversal above support
                    price = support_level + 200 + np.random.normal(0, 100)
                prices.append(price)
                
        elif pattern_type == 'market_structure':
            # Generate market structure: break of structure
            prices = []
            swing_high = base_price + 500
            for i in range(100):
                if i < 80:
                    # Trading below swing high
                    price = swing_high - 200 + np.random.normal(0, 100)
                elif i < 85:
                    # Break above swing high (BOS)
                    price = swing_high + 200 + np.random.normal(0, 50)
                else:
                    # Continue above swing high
                    price = swing_high + 300 + np.random.normal(0, 100)
                prices.append(price)
                
        else:
            # Default: random walk
            prices = [base_price + np.random.normal(0, 100) for _ in range(100)]
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 20)) for p in prices],
            'low': [p - abs(np.random.normal(0, 20)) for p in prices],
            'close': [p + np.random.normal(0, 10) for p in prices],
            'volume': [np.random.randint(1000, 10000) for _ in prices]
        })
        
        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    async def test_smc_order_blocks(self):
        """Test SMC Order Block detection"""
        logger.info("🧪 Testing SMC Order Block detection...")
        
        try:
            # Test bullish order block
            df_bullish = self.generate_test_data('order_block_bullish')
            results_bullish = self.pipeline.analyze_smc_patterns(df_bullish, 'BTCUSDT', '1h')
            
            # Test bearish order block
            df_bearish = self.generate_test_data('order_block_bearish')
            results_bearish = self.pipeline.analyze_smc_patterns(df_bearish, 'BTCUSDT', '1h')
            
            # Check results
            bullish_blocks = results_bullish['order_blocks']
            bearish_blocks = results_bearish['order_blocks']
            
            if bullish_blocks:
                bullish_block = bullish_blocks[0]
                logger.info(f"✅ Bullish Order Block detected with confidence: {bullish_block.confidence:.2f}")
                logger.info(f"   Block type: {bullish_block.block_type}")
                logger.info(f"   Strength: {bullish_block.strength:.2f}")
                return True
            elif bearish_blocks:
                bearish_block = bearish_blocks[0]
                logger.info(f"✅ Bearish Order Block detected with confidence: {bearish_block.confidence:.2f}")
                logger.info(f"   Block type: {bearish_block.block_type}")
                logger.info(f"   Strength: {bearish_block.strength:.2f}")
                return True
            else:
                logger.warning("⚠️ No Order Blocks detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Order Block test failed: {e}")
            return False
    
    async def test_smc_fair_value_gaps(self):
        """Test SMC Fair Value Gap detection"""
        logger.info("🧪 Testing SMC Fair Value Gap detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('fair_value_gap')
            results = self.pipeline.analyze_smc_patterns(df, 'BTCUSDT', '1h')
            
            # Check results
            fair_value_gaps = results['fair_value_gaps']
            
            if fair_value_gaps:
                fvg = fair_value_gaps[0]
                logger.info(f"✅ Fair Value Gap detected with confidence: {fvg.strength:.2f}")
                logger.info(f"   Gap type: {fvg.gap_type}")
                logger.info(f"   Gap size: {fvg.gap_size:.2f}")
                logger.info(f"   Fill probability: {fvg.fill_probability:.2f}")
                return True
            else:
                logger.warning("⚠️ No Fair Value Gaps detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fair Value Gap test failed: {e}")
            return False
    
    async def test_smc_liquidity_sweeps(self):
        """Test SMC Liquidity Sweep detection"""
        logger.info("🧪 Testing SMC Liquidity Sweep detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('liquidity_sweep')
            results = self.pipeline.analyze_smc_patterns(df, 'BTCUSDT', '1h')
            
            # Check results
            liquidity_sweeps = results['liquidity_sweeps']
            
            if liquidity_sweeps:
                sweep = liquidity_sweeps[0]
                logger.info(f"✅ Liquidity Sweep detected with confidence: {sweep.sweep_strength:.2f}")
                logger.info(f"   Sweep type: {sweep.sweep_type}")
                logger.info(f"   Price level: {sweep.price_level:.2f}")
                logger.info(f"   Reversal probability: {sweep.reversal_probability:.2f}")
                return True
            else:
                logger.warning("⚠️ No Liquidity Sweeps detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Liquidity Sweep test failed: {e}")
            return False
    
    async def test_smc_market_structure(self):
        """Test SMC Market Structure detection"""
        logger.info("🧪 Testing SMC Market Structure detection...")
        
        try:
            # Generate test data
            df = self.generate_test_data('market_structure')
            results = self.pipeline.analyze_smc_patterns(df, 'BTCUSDT', '1h')
            
            # Check results
            market_structures = results['market_structure']
            
            if market_structures:
                structure = market_structures[0]
                logger.info(f"✅ Market Structure detected with confidence: {structure.confidence:.2f}")
                logger.info(f"   Structure type: {structure.structure_type}")
                logger.info(f"   Direction: {structure.direction}")
                logger.info(f"   Strength: {structure.strength:.2f}")
                return True
            else:
                logger.warning("⚠️ No Market Structure detected in test data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market Structure test failed: {e}")
            return False
    
    async def test_smc_integration(self):
        """Test SMC patterns integration with overall analysis"""
        logger.info("🧪 Testing SMC patterns integration...")
        
        try:
            # Generate test data
            df = self.generate_test_data('order_block_bullish')
            results = self.pipeline.analyze_smc_patterns(df, 'BTCUSDT', '1h')
            
            # Check overall results
            total_patterns = (
                len(results['order_blocks']) + 
                len(results['fair_value_gaps']) + 
                len(results['liquidity_sweeps']) + 
                len(results['market_structure'])
            )
            
            overall_confidence = results['confidence']
            
            logger.info(f"✅ SMC Integration test completed")
            logger.info(f"   Total patterns detected: {total_patterns}")
            logger.info(f"   Overall confidence: {overall_confidence:.2f}")
            
            if total_patterns > 0 and overall_confidence > 0:
                return True
            else:
                logger.warning("⚠️ No SMC patterns detected in integration test")
                return False
                
        except Exception as e:
            logger.error(f"❌ SMC Integration test failed: {e}")
            return False
    
    async def test_performance(self):
        """Test performance of SMC pattern detection"""
        logger.info("🧪 Testing SMC pattern detection performance...")
        
        try:
            start_time = datetime.now()
            
            # Generate test data
            df = self.generate_test_data('order_block_bullish')
            
            # Run pattern detection multiple times
            for _ in range(10):
                results = self.pipeline.analyze_smc_patterns(df, 'BTCUSDT', '1h')
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"✅ Performance test completed")
            logger.info(f"   Average processing time: {processing_time / 10:.2f}ms per detection")
            
            if processing_time / 10 < 100:  # Should be under 100ms
                logger.info("✅ Performance meets requirements (<100ms)")
                return True
            else:
                logger.warning(f"⚠️ Performance exceeds requirements: {processing_time / 10:.2f}ms")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all SMC enhancement tests"""
        logger.info("🚀 Starting SMC Enhancement Test Suite")
        logger.info("=" * 60)
        
        test_results = []
        
        # Run individual pattern tests
        test_results.append(await self.test_smc_order_blocks())
        test_results.append(await self.test_smc_fair_value_gaps())
        test_results.append(await self.test_smc_liquidity_sweeps())
        test_results.append(await self.test_smc_market_structure())
        
        # Run integration tests
        test_results.append(await self.test_smc_integration())
        test_results.append(await self.test_performance())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        logger.info("=" * 60)
        logger.info(f"📊 Test Results Summary:")
        logger.info(f"   Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 All SMC enhancement tests passed!")
            return True
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
            return False

async def main():
    """Main test execution"""
    tester = SMCEnhancementTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("✅ SMC Enhancement Implementation: SUCCESS")
        return 0
    else:
        logger.error("❌ SMC Enhancement Implementation: FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
