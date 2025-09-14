#!/usr/bin/env python3
"""
Enhanced Indicators Integration Examples
Show how to integrate enhanced indicators into existing AlphaPlus code
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced indicators
from services.enhanced_indicators_integration import EnhancedIndicatorsIntegration
from database.connection import TimescaleDBConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIndicatorsExample:
    """Example class showing enhanced indicators integration"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'pool_size': 10,
            'max_overflow': 20
        }
        self.db_connection = None
        self.indicators_integration = None
    
    async def initialize(self):
        """Initialize database connection and indicators integration"""
        logger.info("🚀 Initializing Enhanced Indicators Example...")
        
        # Initialize database connection
        self.db_connection = TimescaleDBConnection(self.db_config)
        await self.db_connection.initialize()
        
        # Initialize indicators integration
        async with self.db_connection.async_session() as db_session:
            self.indicators_integration = EnhancedIndicatorsIntegration(
                db_session=db_session,
                redis_client=None,  # Optional: add Redis for caching
                enable_enhanced=True
            )
        
        logger.info("✅ Enhanced Indicators Example initialized")
    
    async def example_1_basic_indicators(self):
        """Example 1: Basic indicator calculation"""
        logger.info("📊 Example 1: Basic Indicator Calculation")
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Calculate indicators using enhanced engine
        async with self.db_connection.async_session() as db_session:
            self.indicators_integration.db_session = db_session
            
            result = await self.indicators_integration.calculate_indicators(
                df=market_data,
                symbol="BTC/USDT",
                timeframe="1h"
            )
            
            # Display results
            logger.info(f"📈 RSI: {result.rsi:.2f}")
            logger.info(f"📈 MACD: {result.macd:.6f}")
            logger.info(f"📈 MACD Signal: {result.macd_signal:.6f}")
            logger.info(f"📈 Bollinger Upper: {result.bb_upper:.2f}")
            logger.info(f"📈 Bollinger Middle: {result.bb_middle:.2f}")
            logger.info(f"📈 Bollinger Lower: {result.bb_lower:.2f}")
            logger.info(f"📈 ATR: {result.atr:.4f}")
            logger.info(f"📈 VWAP: {result.vwap:.2f}")
            logger.info(f"📈 OBV: {result.obv:.0f}")
            logger.info(f"📈 Breakout Strength: {result.breakout_strength:.2f}")
            logger.info(f"📈 Trend Confidence: {result.trend_confidence:.2f}")
    
    async def example_2_historical_analysis(self):
        """Example 2: Historical data analysis with continuous aggregates"""
        logger.info("📚 Example 2: Historical Data Analysis")
        
        async with self.db_connection.async_session() as db_session:
            self.indicators_integration.db_session = db_session
            
            # Get historical data using continuous aggregates (ultra-fast)
            historical_data = await self.indicators_integration.get_indicators_from_timescaledb(
                symbol="BTC/USDT",
                timeframe="1h",
                hours_back=24,
                use_aggregates=True  # Uses pre-computed aggregates
            )
            
            if not historical_data.empty:
                logger.info(f"📊 Retrieved {len(historical_data)} historical data points")
                logger.info(f"📊 Latest RSI: {historical_data.iloc[0]['rsi']:.2f}")
                logger.info(f"📊 Latest MACD: {historical_data.iloc[0]['macd']:.6f}")
            else:
                logger.info("📊 No historical data found (normal for new system)")
    
    async def example_3_comprehensive_analysis(self):
        """Example 3: Comprehensive market analysis"""
        logger.info("🎯 Example 3: Comprehensive Market Analysis")
        
        async with self.db_connection.async_session() as db_session:
            self.indicators_integration.db_session = db_session
            
            # Get comprehensive analysis summary
            analysis = await self.indicators_integration.get_analysis_summary(
                symbol="BTC/USDT",
                timeframe="1h"
            )
            
            if analysis:
                logger.info("📊 Comprehensive Analysis Results:")
                logger.info(f"   Overall Signal: {analysis['overall_signal']['signal']}")
                logger.info(f"   Signal Strength: {analysis['overall_signal']['strength']:.2f}")
                logger.info(f"   Confidence: {analysis['overall_signal']['confidence']:.2f}")
                
                # RSI Analysis
                rsi_analysis = analysis['rsi_analysis']
                logger.info(f"   RSI: {rsi_analysis['current_rsi']:.2f} ({rsi_analysis['regime']})")
                
                # MACD Analysis
                macd_analysis = analysis['macd_analysis']
                logger.info(f"   MACD: {macd_analysis['macd']:.6f} ({macd_analysis['signal_direction']})")
                
                # Bollinger Bands Analysis
                bb_analysis = analysis['bollinger_analysis']
                logger.info(f"   BB Position: {bb_analysis['position_type']} ({bb_analysis['position']:.2f})")
                
                # Sentiment Analysis
                sentiment_analysis = analysis['sentiment_analysis']
                logger.info(f"   Sentiment: {sentiment_analysis['sentiment']:.2f} ({sentiment_analysis['regime']})")
                
                # Volatility Analysis
                volatility_analysis = analysis['volatility_analysis']
                logger.info(f"   Volatility: {volatility_analysis['volatility']:.4f} ({volatility_analysis['regime']})")
    
    async def example_4_performance_comparison(self):
        """Example 4: Performance comparison between enhanced and legacy"""
        logger.info("⚡ Example 4: Performance Comparison")
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1h'),
            'open': np.random.uniform(100, 110, 1000),
            'high': np.random.uniform(105, 115, 1000),
            'low': np.random.uniform(95, 105, 1000),
            'close': np.random.uniform(100, 110, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        })
        
        async with self.db_connection.async_session() as db_session:
            self.indicators_integration.db_session = db_session
            
            # Test enhanced engine
            import time
            start_time = time.time()
            
            enhanced_result = await self.indicators_integration.calculate_indicators(
                df=test_data,
                symbol="BTC/USDT",
                timeframe="1h",
                force_legacy=False
            )
            
            enhanced_time = (time.time() - start_time) * 1000
            
            # Test legacy engine
            start_time = time.time()
            
            legacy_result = await self.indicators_integration.calculate_indicators(
                df=test_data,
                symbol="BTC/USDT",
                timeframe="1h",
                force_legacy=True
            )
            
            legacy_time = (time.time() - start_time) * 1000
            
            # Display performance comparison
            logger.info("⚡ Performance Comparison:")
            logger.info(f"   Enhanced Engine: {enhanced_time:.2f}ms")
            logger.info(f"   Legacy Engine: {legacy_time:.2f}ms")
            logger.info(f"   Speed Improvement: {legacy_time/enhanced_time:.1f}x faster")
            
            # Get performance statistics
            stats = self.indicators_integration.get_performance_stats()
            logger.info(f"   Enhanced Usage Rate: {stats['enhanced_usage_rate']:.2%}")
            logger.info(f"   Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    
    async def example_5_integration_with_existing_code(self):
        """Example 5: How to integrate with existing AlphaPlus code"""
        logger.info("🔗 Example 5: Integration with Existing Code")
        
        # This shows how to replace existing indicator calls
        logger.info("📝 Before (Legacy):")
        logger.info("""
        from core.indicators_engine import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        result = indicators.calculate_all_indicators(
            open_price=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            close_prices=[100, 101, 102, 103, 104]
        )
        """)
        
        logger.info("📝 After (Enhanced):")
        logger.info("""
        from services.enhanced_indicators_integration import EnhancedIndicatorsIntegration
        
        # Initialize once (in your main application)
        indicators_integration = EnhancedIndicatorsIntegration(
            db_session=db_session,
            redis_client=redis_client,
            enable_enhanced=True
        )
        
        # Use in your existing code
        result = await indicators_integration.calculate_indicators(
            df=market_data_df,  # Your existing pandas DataFrame
            symbol="BTC/USDT",
            timeframe="1h"
        )
        
        # Access enhanced indicators
        rsi = result.rsi
        macd = result.macd
        vwap = result.vwap  # New!
        obv = result.obv    # New!
        """)
    
    async def run_all_examples(self):
        """Run all examples"""
        logger.info("🎯 Running Enhanced Indicators Examples")
        logger.info("=" * 60)
        
        await self.initialize()
        
        try:
            await self.example_1_basic_indicators()
            logger.info("-" * 40)
            
            await self.example_2_historical_analysis()
            logger.info("-" * 40)
            
            await self.example_3_comprehensive_analysis()
            logger.info("-" * 40)
            
            await self.example_4_performance_comparison()
            logger.info("-" * 40)
            
            await self.example_5_integration_with_existing_code()
            
        except Exception as e:
            logger.error(f"❌ Example failed: {e}")
        
        logger.info("=" * 60)
        logger.info("🎉 Enhanced Indicators Examples completed!")

async def main():
    """Main function to run examples"""
    example = EnhancedIndicatorsExample()
    await example.run_all_examples()

if __name__ == "__main__":
    asyncio.run(main())
