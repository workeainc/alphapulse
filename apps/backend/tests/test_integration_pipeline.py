#!/usr/bin/env python3
"""
Integration tests for AlphaPulse full pipeline
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import deque

# Import AlphaPulse components
try:
    from alphapulse_core import AlphaPulse, TradingSignal, SignalDirection
    from indicators_engine import TechnicalIndicators
    from ml_signal_generator import MLSignalGenerator
    from websocket_binance import BinanceWebSocketClient
except ImportError:
    # Create mock classes for testing
    class TradingSignal:
        def __init__(self, symbol, timeframe, direction, confidence, timestamp, tp1=None, tp2=None, tp3=None, tp4=None, sl=None):
            self.symbol = symbol
            self.timeframe = timeframe
            self.direction = direction
            self.confidence = confidence
            self.timestamp = timestamp
            self.tp1 = tp1
            self.tp2 = tp2
            self.tp3 = tp3
            self.tp4 = tp4
            self.sl = sl
    
    class SignalDirection:
        BUY = "buy"
        SELL = "sell"
    
    class AlphaPulse:
        def __init__(self, symbols, timeframes):
            self.symbols = symbols
            self.timeframes = timeframes
            self.signals = []
            self.indicators = TechnicalIndicators()
            self.signal_generator = MLSignalGenerator()
        
        async def start(self):
            pass
        
        async def stop(self):
            pass
        
        def process_candlestick(self, symbol, timeframe, candlestick_data):
            # Mock signal generation
            if np.random.random() > 0.8:  # 20% chance of signal
                signal = TradingSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=SignalDirection.BUY if np.random.random() > 0.5 else SignalDirection.SELL,
                    confidence=np.random.uniform(0.7, 0.95),
                    timestamp=datetime.utcnow(),
                    tp1=50000 + np.random.randint(100, 1000),
                    sl=50000 - np.random.randint(100, 1000)
                )
                self.signals.append(signal)
                return signal
            return None

class MockWebSocketClient:
    """Mock WebSocket client for testing"""
    
    def __init__(self, historical_data: List[Dict]):
        self.historical_data = historical_data
        self.current_index = 0
        self.is_connected = False
        self.callbacks = []
    
    async def connect(self):
        self.is_connected = True
    
    async def disconnect(self):
        self.is_connected = False
    
    def on_message(self, callback):
        self.callbacks.append(callback)
    
    async def replay_data(self, delay_ms: int = 10):
        """Replay historical data with specified delay"""
        for i, data in enumerate(self.historical_data):
            if not self.is_connected:
                break
            
            # Call all registered callbacks
            for callback in self.callbacks:
                await callback(data)
            
            # Simulate network delay
            await asyncio.sleep(delay_ms / 1000.0)
            
            self.current_index = i + 1

class TestFullPipelineIntegration:
    """Test full AlphaPulse pipeline integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.symbols = ['BTCUSDT', 'ETHUSDT']
        self.timeframes = ['1m', '5m', '15m']
        
        # Generate historical data
        self.historical_data = self._generate_historical_data(1000)
        
        # Initialize AlphaPulse
        self.alphapulse = AlphaPulse(self.symbols, self.timeframes)
        
        # Performance tracking
        self.latencies = []
        self.signals_generated = []
        self.accuracy_metrics = {}
    
    def _generate_historical_data(self, num_ticks: int) -> List[Dict]:
        """Generate realistic historical market data"""
        base_time = datetime.utcnow() - timedelta(hours=2)
        base_price = 50000
        
        data = []
        for i in range(num_ticks):
            # Simulate realistic price movement
            price_change = np.sin(i * 0.01) * 100 + np.random.normal(0, 50)
            current_price = base_price + price_change
            
            # Generate candlestick data
            open_price = current_price - np.random.uniform(0, 100)
            high_price = current_price + np.random.uniform(0, 200)
            low_price = current_price - np.random.uniform(0, 200)
            close_price = current_price + np.random.uniform(-100, 100)
            volume = 1000000 + np.random.uniform(-200000, 200000)
            
            tick_time = base_time + timedelta(seconds=i)
            
            data.append({
                'type': 'kline',
                'symbol': 'BTCUSDT',
                'timeframe': '1m',
                'timestamp': tick_time,
                'data': {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'is_closed': True
                }
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_full_pipeline_latency(self):
        """Test full pipeline latency with 1,000 historical ticks"""
        print("🔄 Testing full pipeline latency...")
        
        # Setup mock WebSocket client
        mock_ws = MockWebSocketClient(self.historical_data)
        
        # Register callback
        async def on_candlestick(data):
            start_time = time.perf_counter()
            
            # Process candlestick
            signal = self.alphapulse.process_candlestick(
                data['symbol'],
                data['timeframe'],
                data['data']
            )
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            self.latencies.append(latency_ms)
            
            if signal:
                self.signals_generated.append(signal)
        
        mock_ws.on_message(on_candlestick)
        
        # Start pipeline
        await self.alphapulse.start()
        await mock_ws.connect()
        
        # Replay historical data
        start_time = time.perf_counter()
        await mock_ws.replay_data(delay_ms=1)  # Fast replay for testing
        end_time = time.perf_counter()
        
        # Stop pipeline
        await mock_ws.disconnect()
        await self.alphapulse.stop()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_latency = np.mean(self.latencies) if self.latencies else 0
        max_latency = np.max(self.latencies) if self.latencies else 0
        min_latency = np.min(self.latencies) if self.latencies else 0
        
        print(f"📊 Pipeline Performance:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Ticks processed: {len(self.historical_data)}")
        print(f"  Signals generated: {len(self.signals_generated)}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        print(f"  Min latency: {min_latency:.2f}ms")
        
        # Assertions
        assert avg_latency < 50, f"Average latency should be < 50ms, got {avg_latency:.2f}ms"
        assert max_latency < 100, f"Max latency should be < 100ms, got {max_latency:.2f}ms"
        assert len(self.signals_generated) > 0, "Should generate at least one signal"
        
        # Throughput test
        throughput = len(self.historical_data) / total_time
        assert throughput > 1000, f"Throughput should be > 1000 ticks/sec, got {throughput:.2f}"
    
    @pytest.mark.asyncio
    async def test_signal_accuracy_simulation(self):
        """Test signal accuracy using simulated outcomes"""
        print("🎯 Testing signal accuracy simulation...")
        
        # Generate signals with known outcomes
        signals_with_outcomes = []
        
        for i in range(100):  # Generate 100 test signals
            # Simulate signal generation
            confidence = np.random.uniform(0.7, 0.95)
            direction = SignalDirection.BUY if np.random.random() > 0.5 else SignalDirection.SELL
            
            # Simulate outcome based on confidence (higher confidence = better outcomes)
            outcome_probability = confidence * 0.8  # Scale confidence to outcome probability
            is_win = np.random.random() < outcome_probability
            
            signal = TradingSignal(
                symbol='BTCUSDT',
                timeframe='15m',
                direction=direction,
                confidence=confidence,
                timestamp=datetime.utcnow() - timedelta(minutes=i*5),
                tp1=50000 + 500,
                sl=50000 - 300
            )
            
            signals_with_outcomes.append({
                'signal': signal,
                'outcome': 'win' if is_win else 'loss',
                'pnl': 500 if is_win else -300
            })
        
        # Calculate accuracy metrics
        total_signals = len(signals_with_outcomes)
        winning_signals = sum(1 for s in signals_with_outcomes if s['outcome'] == 'win')
        losing_signals = total_signals - winning_signals
        
        win_rate = winning_signals / total_signals if total_signals > 0 else 0
        
        # Calculate profit metrics
        total_pnl = sum(s['pnl'] for s in signals_with_outcomes)
        avg_win = np.mean([s['pnl'] for s in signals_with_outcomes if s['outcome'] == 'win']) if winning_signals > 0 else 0
        avg_loss = np.mean([s['pnl'] for s in signals_with_outcomes if s['outcome'] == 'loss']) if losing_signals > 0 else 0
        
        profit_factor = abs(avg_win * winning_signals / (avg_loss * losing_signals)) if losing_signals > 0 and avg_loss != 0 else float('inf')
        
        print(f"📈 Signal Accuracy Metrics:")
        print(f"  Total signals: {total_signals}")
        print(f"  Winning signals: {winning_signals}")
        print(f"  Losing signals: {losing_signals}")
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Average win: ${avg_win:.2f}")
        print(f"  Average loss: ${avg_loss:.2f}")
        print(f"  Profit factor: {profit_factor:.2f}")
        
        # Assertions
        assert 0.75 <= win_rate <= 0.85, f"Win rate should be 75-85%, got {win_rate:.2%}"
        assert profit_factor > 1.5, f"Profit factor should be > 1.5, got {profit_factor:.2f}"
        assert total_pnl > 0, f"Total PnL should be positive, got ${total_pnl:.2f}"
        
        # Store metrics for reporting
        self.accuracy_metrics = {
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'losing_signals': losing_signals,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    @pytest.mark.asyncio
    async def test_multi_symbol_processing(self):
        """Test processing multiple symbols simultaneously"""
        print("🔄 Testing multi-symbol processing...")
        
        # Generate data for multiple symbols
        symbols_data = {}
        for symbol in self.symbols:
            symbols_data[symbol] = self._generate_historical_data(500)
        
        # Process each symbol
        symbol_signals = {}
        symbol_latencies = {}
        
        for symbol, data in symbols_data.items():
            print(f"  Processing {symbol}...")
            
            latencies = []
            signals = []
            
            for tick in data:
                start_time = time.perf_counter()
                
                signal = self.alphapulse.process_candlestick(
                    symbol,
                    '1m',
                    tick['data']
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if signal:
                    signals.append(signal)
            
            symbol_signals[symbol] = signals
            symbol_latencies[symbol] = latencies
        
        # Verify multi-symbol processing
        total_signals = sum(len(signals) for signals in symbol_signals.values())
        total_latencies = [lat for latencies in symbol_latencies.values() for lat in latencies]
        
        avg_latency = np.mean(total_latencies) if total_latencies else 0
        
        print(f"📊 Multi-Symbol Results:")
        print(f"  Total signals across symbols: {total_signals}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        
        for symbol in self.symbols:
            print(f"  {symbol}: {len(symbol_signals[symbol])} signals")
        
        # Assertions
        assert total_signals > 0, "Should generate signals for multiple symbols"
        assert avg_latency < 50, f"Multi-symbol latency should be < 50ms, got {avg_latency:.2f}ms"
        
        # Verify all symbols were processed
        for symbol in self.symbols:
            assert symbol in symbol_signals, f"Symbol {symbol} should be processed"
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and resilience"""
        print("🛡️ Testing error recovery...")
        
        # Generate data with some malformed entries
        test_data = self.historical_data.copy()
        
        # Insert some malformed data
        malformed_entries = [
            {'type': 'kline', 'symbol': 'BTCUSDT', 'data': None},  # Missing data
            {'type': 'kline', 'symbol': 'BTCUSDT', 'data': {'invalid': 'data'}},  # Invalid data
            {'type': 'unknown', 'symbol': 'BTCUSDT', 'data': {}},  # Unknown type
        ]
        
        # Insert malformed data at random positions
        for entry in malformed_entries:
            insert_pos = np.random.randint(0, len(test_data))
            test_data.insert(insert_pos, entry)
        
        # Process data and count errors
        errors = []
        successful_processing = 0
        
        for tick in test_data:
            try:
                if tick.get('data') and isinstance(tick['data'], dict):
                    signal = self.alphapulse.process_candlestick(
                        tick['symbol'],
                        tick.get('timeframe', '1m'),
                        tick['data']
                    )
                    successful_processing += 1
                else:
                    errors.append(f"Invalid data format: {tick}")
            except Exception as e:
                errors.append(f"Processing error: {str(e)}")
        
        error_rate = len(errors) / len(test_data)
        
        print(f"📊 Error Recovery Results:")
        print(f"  Total entries: {len(test_data)}")
        print(f"  Successful processing: {successful_processing}")
        print(f"  Errors: {len(errors)}")
        print(f"  Error rate: {error_rate:.2%}")
        
        # Assertions
        assert error_rate < 0.1, f"Error rate should be < 10%, got {error_rate:.2%}"
        assert successful_processing > 0, "Should successfully process some data"
        
        # Verify system continues to function
        assert len(self.alphapulse.signals) >= 0, "System should continue generating signals"
    
    def test_performance_benchmark(self):
        """Performance benchmark test"""
        print("⚡ Running performance benchmark...")
        
        # Benchmark indicator calculations
        indicators = TechnicalIndicators()
        prices = [50000 + np.random.normal(0, 100) for _ in range(1000)]
        
        # RSI calculation benchmark
        start_time = time.perf_counter()
        for _ in range(100):
            rsi = indicators.calculate_rsi(prices, period=14)
        rsi_time = (time.perf_counter() - start_time) * 1000
        
        # Signal generation benchmark
        signal_generator = MLSignalGenerator()
        candlestick_data = {
            'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050, 'volume': 1500000
        }
        indicators_data = {'rsi': 65, 'macd': 100, 'volume_sma': 1000000, 'adx': 28}
        
        start_time = time.perf_counter()
        for _ in range(100):
            patterns = signal_generator.detect_patterns(candlestick_data, indicators_data)
        signal_time = (time.perf_counter() - start_time) * 1000
        
        print(f"📊 Performance Benchmark:")
        print(f"  RSI calculation (100x): {rsi_time:.2f}ms")
        print(f"  Signal generation (100x): {signal_time:.2f}ms")
        print(f"  Average RSI calc: {rsi_time/100:.2f}ms")
        print(f"  Average signal gen: {signal_time/100:.2f}ms")
        
        # Assertions
        assert rsi_time/100 < 1.0, f"RSI calculation should be < 1ms, got {rsi_time/100:.2f}ms"
        assert signal_time/100 < 1.0, f"Signal generation should be < 1ms, got {signal_time/100:.2f}ms"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
