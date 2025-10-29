#!/usr/bin/env python3
"""
AlphaPulse Example - Real-time Trading Signal System
Demonstrates how to use AlphaPulse for live trading signals
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List

# Import AlphaPulse components
from alphapulse_core import AlphaPulse, TradingSignal, SignalDirection, MarketRegime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlphaPulseExample:
    """Example AlphaPulse implementation with signal handling"""
    
    def __init__(self):
        """Initialize AlphaPulse example"""
        self.signal_count = 0
        self.total_pnl = 0.0
        self.winning_signals = 0
        self.losing_signals = 0
        
        # Signal tracking
        self.active_signals = {}
        self.signal_history = []
        
        logger.info("AlphaPulse Example initialized")
    
    async def start_alphapulse(self):
        """Start AlphaPulse system with example configuration"""
        
        # Initialize AlphaPulse with popular trading pairs
        ap = AlphaPulse(
            symbols=["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"],
            timeframes=["1m", "5m", "15m"],
            redis_url="redis://localhost:6379",
            max_workers=4
        )
        
        # Add signal callback
        ap.add_signal_callback(self.handle_signal)
        
        # Add performance monitoring callback
        ap.add_signal_callback(self.monitor_performance)
        
        logger.info("🚀 Starting AlphaPulse Trading System...")
        logger.info("📊 Monitoring: BTC/USDT, ETH/USDT, ADA/USDT, SOL/USDT")
        logger.info("⏱️  Timeframes: 1m, 5m, 15m")
        
        try:
            # Start the system
            await ap.start()
            
            # Keep running and monitor performance
            await self.run_performance_monitor(ap)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down AlphaPulse...")
            await ap.stop()
            self.print_final_stats()
        except Exception as e:
            logger.error(f"❌ Error running AlphaPulse: {e}")
            await ap.stop()
    
    async def handle_signal(self, signal: TradingSignal):
        """Handle incoming trading signals"""
        self.signal_count += 1
        
        # Create signal summary
        signal_summary = {
            'id': signal.signal_id,
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'timeframe': signal.timeframe,
            'direction': signal.direction.value,
            'pattern': signal.pattern,
            'confidence': signal.confidence,
            'entry_price': signal.entry_price,
            'stop_loss': signal.SL,
            'take_profits': [signal.TP1, signal.TP2, signal.TP3, signal.TP4],
            'volume_confirmed': signal.volume_confirmed,
            'trend_aligned': signal.trend_aligned,
            'market_regime': signal.market_regime.value
        }
        
        # Store signal
        self.active_signals[signal.signal_id] = signal_summary
        self.signal_history.append(signal_summary)
        
        # Print signal details
        self.print_signal(signal_summary)
        
        # Simulate signal outcome (in real implementation, this would track actual trades)
        await self.simulate_signal_outcome(signal_summary)
    
    async def monitor_performance(self, signal: TradingSignal):
        """Monitor system performance"""
        # Get performance stats every 10 signals
        if self.signal_count % 10 == 0:
            logger.info(f"📈 Performance Update - Signals: {self.signal_count}")
    
    def print_signal(self, signal: Dict):
        """Print formatted signal information"""
        direction_emoji = "🟢" if signal['direction'] == 'buy' else "🔴"
        confidence_color = "🟢" if signal['confidence'] > 0.8 else "🟡" if signal['confidence'] > 0.6 else "🔴"
        
        print(f"\n{direction_emoji} SIGNAL #{self.signal_count}")
        print("=" * 60)
        print(f"📊 {signal['symbol']} ({signal['timeframe']}) - {signal['direction'].upper()}")
        print(f"🎯 Pattern: {signal['pattern']}")
        print(f"{confidence_color} Confidence: {signal['confidence']:.2%}")
        print(f"💰 Entry: ${signal['entry_price']:,.2f}")
        print(f"🛑 Stop Loss: ${signal['stop_loss']:,.2f}")
        print(f"📈 Take Profits:")
        for i, tp in enumerate(signal['take_profits'], 1):
            print(f"   TP{i}: ${tp:,.2f}")
        
        # Validation status
        volume_status = "✅" if signal['volume_confirmed'] else "❌"
        trend_status = "✅" if signal['trend_aligned'] else "❌"
        print(f"📊 Volume Confirmed: {volume_status}")
        print(f"📈 Trend Aligned: {trend_status}")
        print(f"🌍 Market Regime: {signal['market_regime']}")
        print(f"⏰ Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    async def simulate_signal_outcome(self, signal: Dict):
        """Simulate signal outcome for demonstration"""
        # In a real implementation, this would track actual price movements
        # For demonstration, we'll simulate outcomes based on confidence
        
        import random
        
        # Simulate outcome after 5 minutes
        await asyncio.sleep(5)
        
        # Generate random outcome based on confidence
        success_probability = signal['confidence']
        is_successful = random.random() < success_probability
        
        if is_successful:
            # Simulate profit (1-3% gain)
            profit_pct = random.uniform(0.01, 0.03)
            profit = signal['entry_price'] * profit_pct
            self.total_pnl += profit
            self.winning_signals += 1
            
            logger.info(f"✅ Signal #{self.signal_count} PROFIT: +${profit:.2f} (+{profit_pct:.2%})")
        else:
            # Simulate loss (0.5-1.5% loss)
            loss_pct = random.uniform(0.005, 0.015)
            loss = signal['entry_price'] * loss_pct
            self.total_pnl -= loss
            self.losing_signals += 1
            
            logger.info(f"❌ Signal #{self.signal_count} LOSS: -${loss:.2f} (-{loss_pct:.2%})")
        
        # Remove from active signals
        if signal['id'] in self.active_signals:
            del self.active_signals[signal['id']]
    
    async def run_performance_monitor(self, ap: AlphaPulse):
        """Run performance monitoring loop"""
        start_time = time.time()
        
        while True:
            try:
                # Get system stats
                stats = ap.get_performance_stats()
                system_status = ap.get_system_status()
                
                # Calculate runtime
                runtime = time.time() - start_time
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)
                seconds = int(runtime % 60)
                
                # Print performance summary
                print(f"\n📊 ALPHAPULSE PERFORMANCE SUMMARY")
                print("=" * 50)
                print(f"⏱️  Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
                print(f"📡 System Status: {'🟢 RUNNING' if system_status['is_running'] else '🔴 STOPPED'}")
                print(f"📊 Total Signals: {self.signal_count}")
                print(f"✅ Winning Signals: {self.winning_signals}")
                print(f"❌ Losing Signals: {self.losing_signals}")
                
                if self.signal_count > 0:
                    win_rate = (self.winning_signals / self.signal_count) * 100
                    print(f"📈 Win Rate: {win_rate:.1f}%")
                
                print(f"💰 Total P&L: ${self.total_pnl:,.2f}")
                print(f"⚡ Avg Latency: {stats.get('avg_latency_ms', 0):.2f}ms")
                print(f"🔄 Ticks Processed: {stats.get('total_ticks_processed', 0)}")
                print(f"🎯 Signals Generated: {stats.get('signals_generated', 0)}")
                print(f"🚫 Signals Filtered: {stats.get('signals_filtered', 0)}")
                print("=" * 50)
                
                # Wait 60 seconds before next update
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    def print_final_stats(self):
        """Print final statistics"""
        print(f"\n🎯 ALPHAPULSE FINAL STATISTICS")
        print("=" * 50)
        print(f"📊 Total Signals: {self.signal_count}")
        print(f"✅ Winning Signals: {self.winning_signals}")
        print(f"❌ Losing Signals: {self.losing_signals}")
        
        if self.signal_count > 0:
            win_rate = (self.winning_signals / self.signal_count) * 100
            print(f"📈 Win Rate: {win_rate:.1f}%")
        
        print(f"💰 Total P&L: ${self.total_pnl:,.2f}")
        print(f"📈 Average P&L per Signal: ${self.total_pnl / max(self.signal_count, 1):,.2f}")
        print("=" * 50)
        
        # Print recent signals
        if self.signal_history:
            print(f"\n📋 Last 5 Signals:")
            for signal in self.signal_history[-5:]:
                direction_emoji = "🟢" if signal['direction'] == 'buy' else "🔴"
                print(f"  {direction_emoji} {signal['symbol']} {signal['direction'].upper()} "
                      f"({signal['confidence']:.1%} confidence) - {signal['pattern']}")


async def main():
    """Main function to run AlphaPulse example"""
    print("🚀 ALPHAPULSE TRADING SIGNAL SYSTEM")
    print("=" * 60)
    print("High-Frequency Trading Signals with <100ms Latency")
    print("Inspired by TradingView's GG-Shot Indicator")
    print("=" * 60)
    
    # Check if Redis is available
    try:
        import redis
        r = redis.from_url("redis://localhost:6379")
        r.ping()
        print("✅ Redis connection established")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   Signals will not be stored. Install Redis for full functionality.")
    
    # Create and run AlphaPulse example
    example = AlphaPulseExample()
    await example.start_alphapulse()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 AlphaPulse stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your configuration and try again.")
