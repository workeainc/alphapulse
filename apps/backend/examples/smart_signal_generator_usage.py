#!/usr/bin/env python3
"""
Smart Signal Generator - Usage Example
Demonstrates how to use the new Smart Tiered Intelligence Architecture
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai.smart_signal_generator import SmartSignalGenerator

async def main():
    """Demonstrate Smart Signal Generator usage"""
    
    print("=" * 80)
    print("SMART SIGNAL GENERATOR - USAGE EXAMPLE")
    print("=" * 80)
    
    # =====================================================================
    # STEP 1: Initialize Smart Signal Generator
    # =====================================================================
    print("\n📊 Step 1: Initializing Smart Signal Generator...")
    
    config = {
        'adaptive': {
            'target_min_signals': 3,
            'target_max_signals': 8,
            'adjustment_interval_hours': 6
        },
        'context_filter': {
            # Optional: configure context filter
        }
    }
    
    generator = SmartSignalGenerator(config)
    print("✅ Smart Signal Generator initialized")
    
    # =====================================================================
    # STEP 2: Prepare Sample Market Data
    # =====================================================================
    print("\n📊 Step 2: Preparing sample market data...")
    
    # Create sample OHLCV DataFrame
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(49000, 51000, 100),
        'high': np.random.uniform(50000, 52000, 100),
        'low': np.random.uniform(48000, 50000, 100),
        'close': np.random.uniform(49000, 51000, 100),
        'volume': np.random.uniform(1000000, 5000000, 100)
    })
    
    # Calculate some basic indicators for demonstration
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['rsi'] = 50 + np.random.uniform(-20, 20, 100)  # Simplified
    df['obv'] = (df['volume'] * np.where(df['close'].diff() > 0, 1, -1)).cumsum()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    print(f"✅ Created sample data: {len(df)} candles")
    
    # =====================================================================
    # STEP 3: Prepare Market Data Dict
    # =====================================================================
    print("\n📊 Step 3: Preparing market data dict...")
    
    market_data = {
        'current_price': float(df['close'].iloc[-1]),
        'volume': float(df['volume'].iloc[-1]),
        'indicators': {
            'sma_20': float(df['sma_20'].iloc[-1]),
            'sma_50': float(df['sma_50'].iloc[-1]),
            'ema_12': float(df['ema_12'].iloc[-1]),
            'ema_26': float(df['ema_26'].iloc[-1]),
            'macd': float(df['macd'].iloc[-1]),
            'rsi_14': float(df['rsi'].iloc[-1]),
            'obv': float(df['obv'].iloc[-1]),
            'vwap': float(df['vwap'].iloc[-1]),
            # Add more calculated indicators here
            'supertrend': {'signal': 0.6},  # Example
            'tsi': {'signal': 0.55},
            'cmo': {'signal': 0.58},
        }
    }
    
    print(f"✅ Market data prepared with {len(market_data['indicators'])} indicators")
    
    # =====================================================================
    # STEP 4: Prepare Analysis Results
    # =====================================================================
    print("\n📊 Step 4: Preparing analysis results...")
    
    analysis_results = {
        'dataframe': df,
        'volume_analysis': {
            'volume_trend': 'increasing',
            'volume_strength': 'strong'
        },
        'sentiment_analysis': {
            'overall_sentiment': 0.3,  # Slightly bullish
            'confidence': 0.65
        },
        'market_regime': {
            'regime': 'trending'  # or 'ranging', 'volatile', etc.
        }
    }
    
    print("✅ Analysis results prepared")
    
    # =====================================================================
    # STEP 5: Generate Signal
    # =====================================================================
    print("\n📊 Step 5: Generating smart signal...")
    print("-" * 80)
    
    result = await generator.generate_signal(
        symbol='BTC/USDT',
        timeframe='1h',
        market_data=market_data,
        analysis_results=analysis_results
    )
    
    # =====================================================================
    # STEP 6: Process Result
    # =====================================================================
    print("\n📊 Step 6: Processing result...")
    print("=" * 80)
    
    if result and result.signal_generated:
        print("✅ SIGNAL GENERATED!")
        print("=" * 80)
        print(f"\n🎯 SIGNAL DETAILS:")
        print(f"   Symbol:           {result.symbol}")
        print(f"   Timeframe:        {result.timeframe}")
        print(f"   Direction:        {result.direction.upper()} {'🚀' if result.direction == 'long' else '📉'}")
        print(f"   Confidence:       {result.confidence:.3f} ({result.confidence * 100:.1f}%)")
        print(f"   Quality Score:    {result.quality_score:.3f} ({result.quality_score * 100:.1f}%)")
        print(f"   Consensus Score:  {result.consensus_score:.3f}")
        print(f"   Market Regime:    {result.market_regime.upper()}")
        print(f"   Timestamp:        {result.timestamp}")
        
        print(f"\n📊 CONSENSUS INFO:")
        print(f"   Contributing Heads: {len(result.contributing_heads)}/{result.metadata['total_heads']}")
        print(f"   Heads:              {', '.join(result.contributing_heads)}")
        print(f"   Required:           {result.metadata['min_heads_required']}")
        
        print(f"\n⚙️ ADAPTIVE THRESHOLDS:")
        thresholds = result.metadata['adaptive_thresholds']
        print(f"   Min Confidence:     {thresholds['min_confidence']:.3f}")
        print(f"   Min Consensus:      {thresholds['min_consensus_heads']}")
        print(f"   Duplicate Window:   {thresholds['duplicate_window_hours']}h")
        print(f"   Market Regime:      {thresholds['market_regime']}")
        
        print(f"\n🧠 CONTEXT ANALYSIS:")
        print(f"   Priority Score:     {result.metadata['priority_score']:.3f}")
        print(f"   Confidence Adj:     {result.metadata['confidence_adjustment']:.2f}x")
        print(f"   Calc Time:          {result.metadata['calculation_time_ms']:.2f}ms")
        
        print(f"\n💡 REASONING:")
        print(f"   {result.reasoning}")
        
    else:
        print("❌ NO SIGNAL GENERATED")
        print("=" * 80)
        print("\nPossible reasons:")
        print("  - Consensus not achieved")
        print("  - Context filter blocked")
        print("  - Quality score too low")
        print("  - Duplicate signal detected")
    
    # =====================================================================
    # STEP 7: Show Statistics
    # =====================================================================
    print("\n" + "=" * 80)
    print("📈 GENERATOR STATISTICS")
    print("=" * 80)
    
    stats = generator.get_stats()
    
    print(f"\n📊 SIGNAL METRICS:")
    print(f"   Total Evaluations:   {stats['total_evaluations']}")
    print(f"   Signals Generated:   {stats['signals_generated']}")
    print(f"   Signal Rate:         {stats['signal_rate']:.2%}")
    print(f"   Signals Last 24h:    {stats['signals_last_24h']}")
    
    print(f"\n📊 QUALITY METRICS:")
    print(f"   Avg Confidence:      {stats['avg_confidence']:.3f}")
    print(f"   Avg Quality Score:   {stats['avg_quality_score']:.3f}")
    
    print(f"\n🚫 BLOCKING RATES:")
    print(f"   Consensus:           {stats['consensus_block_rate']:.2%}")
    print(f"   Context Filter:      {stats['context_block_rate']:.2%}")
    print(f"   Quality:             {stats['quality_block_rate']:.2%}")
    print(f"   Duplicate:           {stats['duplicate_block_rate']:.2%}")
    
    print(f"\n⚙️ ADAPTIVE SYSTEM:")
    print(f"   Threshold Adjustments: {stats['threshold_adjustments']}")
    
    # =====================================================================
    # STEP 8: Simulate Multiple Signals
    # =====================================================================
    print("\n" + "=" * 80)
    print("🔄 SIMULATING MULTIPLE SIGNAL GENERATIONS")
    print("=" * 80)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Testing {symbol}...")
        
        # Slightly modify market data for each symbol
        test_market_data = market_data.copy()
        test_market_data['current_price'] *= np.random.uniform(0.95, 1.05)
        
        result = await generator.generate_signal(
            symbol=symbol,
            timeframe='1h',
            market_data=test_market_data,
            analysis_results=analysis_results
        )
        
        if result and result.signal_generated:
            print(f"   ✅ Signal: {result.direction.upper()} | "
                  f"Confidence: {result.confidence:.3f} | "
                  f"Quality: {result.quality_score:.3f}")
        else:
            print(f"   ❌ No signal")
    
    # =====================================================================
    # FINAL STATISTICS
    # =====================================================================
    print("\n" + "=" * 80)
    print("📈 FINAL STATISTICS")
    print("=" * 80)
    
    final_stats = generator.get_stats()
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Evaluations:   {final_stats['total_evaluations']}")
    print(f"   Signals Generated:   {final_stats['signals_generated']}")
    print(f"   Success Rate:        {final_stats['signal_rate']:.2%}")
    print(f"   Avg Confidence:      {final_stats['avg_confidence']:.3f}")
    print(f"   Avg Quality:         {final_stats['avg_quality_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

