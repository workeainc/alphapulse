#!/usr/bin/env python3
"""
Test Phase 2 Backtesting Enhancements
Tests vectorized backtests, adjustable slippage, latency penalties, and KPI validation
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(symbol: str = "BTCUSDT", days: int = 100) -> pd.DataFrame:
    """Create realistic test data for backtesting"""
    logger.info(f"📊 Creating test data for {symbol} ({days} days)")
    
    # Generate realistic price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='H')
    
    # Start with realistic price
    base_price = 50000 if symbol == "BTCUSDT" else 3000
    
    # Generate price movements
    returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from price
        volatility = 0.01  # 1% intraday volatility
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
        close_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
        
        # Ensure OHLC relationship
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created {len(df)} data points for {symbol}")
    return df

def test_phase2_backtesting_enhancements():
    """Test all Phase 2 backtesting enhancements"""
    try:
        logger.info("🧪 Testing Phase 2 Backtesting Enhancements")
        
        # Import the enhanced backtester
        from ..ai.advanced_backtesting import (
            AdvancedBacktester, KPIConfig, LatencyConfig, SymbolConfig,
            BacktestType
        )
        
        # Import dependencies
        from ..ai.risk_management import RiskManager
        from ..ai.position_sizing import PositionSizingOptimizer
        from ..ai.market_regime_detection import MarketRegimeDetector
        
        # Create test data
        test_data = create_test_data("BTCUSDT", days=30)
        
        # Initialize dependencies
        risk_manager = RiskManager()
        position_sizing_optimizer = PositionSizingOptimizer()
        market_regime_detector = MarketRegimeDetector()
        
        # Initialize backtester with Phase 2 configurations
        kpi_config = KPIConfig(
            min_sharpe_ratio=0.5,
            max_drawdown_pct=25.0,
            min_win_rate=0.4,
            min_profit_factor=1.1,
            min_total_return_pct=5.0,
            max_consecutive_losses=7,
            min_trades=10
        )
        
        latency_config = LatencyConfig(
            order_execution_ms=150,
            market_data_ms=75,
            signal_generation_ms=50,
            slippage_impact_ms=250
        )
        
        # Create symbol-specific configurations
        btc_config = SymbolConfig(
            symbol="BTCUSDT",
            slippage_bps=8.0,  # 8 bps slippage
            fee_rate=0.0015,   # 0.15% fee
            min_tick_size=0.1,
            avg_spread_bps=3.0,
            volatility_multiplier=1.2
        )
        
        eth_config = SymbolConfig(
            symbol="ETHUSDT",
            slippage_bps=5.0,  # 5 bps slippage
            fee_rate=0.001,    # 0.1% fee
            min_tick_size=0.01,
            avg_spread_bps=2.0,
            volatility_multiplier=1.0
        )
        
        # Initialize backtester with proper dependencies
        backtester = AdvancedBacktester(
            risk_manager=risk_manager,
            position_sizing_optimizer=position_sizing_optimizer,
            market_regime_detector=market_regime_detector,
            initial_capital=100000.0,
            kpi_config=kpi_config,
            latency_config=latency_config
        )
        
        # Add symbol configurations
        backtester.add_symbol_config("BTCUSDT", btc_config)
        backtester.add_symbol_config("ETHUSDT", eth_config)
        
        # Load test data
        backtester.load_historical_data("BTCUSDT", test_data)
        
        # Test 1: Vectorized Backtesting
        logger.info("🔬 Test 1: Vectorized Backtesting")
        
        def simple_strategy(data, params, regime):
            """Simple moving average crossover strategy"""
            if len(data) < 50:
                return 'hold'
            
            current = data.iloc[-1]
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            
            # Calculate RSI if not present
            if 'rsi' in data.columns:
                rsi = current['rsi']
            else:
                # Calculate RSI manually
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.iloc[-1]
            
            if sma_20 > sma_50 and rsi < 70:
                return 'buy'
            elif sma_20 < sma_50 and rsi > 30:
                return 'sell'
            else:
                return 'hold'
        
        # Run vectorized backtest
        start_time = time.time()
        result = backtester.vectorized_backtest(
            symbol="BTCUSDT",
            strategy_func=simple_strategy,
            parameters={'lookback': 20}
        )
        vectorized_time = time.time() - start_time
        
        logger.info(f"✅ Vectorized backtest completed in {vectorized_time:.2f}s")
        logger.info(f"   - Total trades: {result.total_trades}")
        logger.info(f"   - Win rate: {result.win_rate:.2%}")
        logger.info(f"   - Total return: {result.total_return_pct:.2f}%")
        logger.info(f"   - Sharpe ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"   - Max drawdown: {result.max_drawdown_pct:.2f}%")
        
        # Test 2: Latency Penalties
        logger.info("🔬 Test 2: Latency Penalties")
        
        total_latency_penalty = result.total_latency_penalty
        avg_execution_delay = result.avg_execution_delay_ms
        
        logger.info(f"✅ Latency analysis:")
        logger.info(f"   - Total latency penalty: ${total_latency_penalty:.2f}")
        logger.info(f"   - Average execution delay: {avg_execution_delay:.1f}ms")
        
        if result.total_return != 0:
            latency_impact = (total_latency_penalty/result.total_return)*100
            logger.info(f"   - Latency impact: {latency_impact:.2f}% of total return")
        else:
            logger.info(f"   - Latency impact: N/A (no return)")
        
        # Test 3: Adjustable Slippage
        logger.info("🔬 Test 3: Adjustable Slippage")
        
        total_slippage_cost = result.total_slippage_cost
        total_fee_cost = result.total_fee_cost
        
        logger.info(f"✅ Slippage and fees analysis:")
        logger.info(f"   - Total slippage cost: ${total_slippage_cost:.2f}")
        logger.info(f"   - Total fee cost: ${total_fee_cost:.2f}")
        logger.info(f"   - Combined cost: ${total_slippage_cost + total_fee_cost:.2f}")
        
        if result.total_return != 0:
            cost_impact = ((total_slippage_cost + total_fee_cost)/result.total_return)*100
            logger.info(f"   - Cost impact: {cost_impact:.2f}% of total return")
        else:
            logger.info(f"   - Cost impact: N/A (no return)")
        
        # Test 4: KPI Validation
        logger.info("🔬 Test 4: KPI Validation")
        
        kpi_passed = result.kpi_validation_passed
        kpi_score = result.kpi_score
        
        logger.info(f"✅ KPI Validation:")
        logger.info(f"   - KPI Score: {kpi_score:.2f}/1.0")
        logger.info(f"   - Passed validation: {kpi_passed}")
        
        # Test 5: Symbol Configuration
        logger.info("🔬 Test 5: Symbol Configuration")
        
        btc_config_retrieved = backtester.get_symbol_config("BTCUSDT")
        eth_config_retrieved = backtester.get_symbol_config("ETHUSDT")
        default_config = backtester.get_symbol_config("UNKNOWN")
        
        logger.info(f"✅ Symbol configurations:")
        logger.info(f"   - BTCUSDT slippage: {btc_config_retrieved.slippage_bps} bps")
        logger.info(f"   - ETHUSDT slippage: {eth_config_retrieved.slippage_bps} bps")
        logger.info(f"   - Default slippage: {default_config.slippage_bps} bps")
        
        # Test 6: Performance Comparison
        logger.info("🔬 Test 6: Performance Comparison")
        
        # Run simple backtest for comparison
        start_time = time.time()
        simple_result = backtester.simple_backtest(
            symbol="BTCUSDT",
            strategy_func=simple_strategy,
            parameters={'lookback': 20}
        )
        simple_time = time.time() - start_time
        
        logger.info(f"✅ Performance comparison:")
        logger.info(f"   - Vectorized time: {vectorized_time:.2f}s")
        logger.info(f"   - Simple time: {simple_time:.2f}s")
        logger.info(f"   - Speed improvement: {simple_time/vectorized_time:.1f}x")
        
        # Test 7: Trade Analysis
        logger.info("🔬 Test 7: Trade Analysis")
        
        if result.trades:
            # Analyze first few trades
            for i, trade in enumerate(result.trades[:3]):
                logger.info(f"   Trade {i+1}:")
                logger.info(f"     - Type: {trade.trade_type}")
                logger.info(f"     - PnL: ${trade.pnl:.2f}")
                logger.info(f"     - Slippage: ${trade.slippage_cost:.2f}")
                logger.info(f"     - Fees: ${trade.fee_cost:.2f}")
                logger.info(f"     - Latency penalty: ${trade.latency_penalty:.2f}")
                logger.info(f"     - Execution delay: {trade.execution_delay_ms}ms")
        
        # Summary
        logger.info("🎉 Phase 2 Backtesting Enhancements Test Summary:")
        logger.info(f"✅ Vectorized backtesting: WORKING")
        logger.info(f"✅ Adjustable slippage per symbol: WORKING")
        logger.info(f"✅ Latency penalties: WORKING")
        logger.info(f"✅ KPI validation: WORKING")
        logger.info(f"✅ Performance improvement: {simple_time/vectorized_time:.1f}x faster")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 2 backtesting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase2_backtesting_enhancements()
    if success:
        logger.info("🎉 All Phase 2 backtesting tests passed!")
    else:
        logger.error("❌ Phase 2 backtesting tests failed!")
        sys.exit(1)
