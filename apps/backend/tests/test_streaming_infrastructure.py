"""
Test Suite for Phase 1 Streaming Infrastructure
Comprehensive testing of all streaming components with TimescaleDB integration
"""

import asyncio
import logging
import pytest
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

# Import streaming components
from src.streaming.stream_buffer import StreamMessage, StreamBuffer
from src.streaming.stream_normalizer import StreamNormalizer
from src.streaming.candle_builder import CandleBuilder
from src.streaming.rolling_state_manager import RollingStateManager
from src.streaming.stream_processor import StreamProcessor
from src.streaming.stream_metrics import StreamMetrics

# Import configuration
from src.core.config import STREAMING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStreamingInfrastructure:
    """Test suite for streaming infrastructure"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test environment"""
        self.config = STREAMING_CONFIG.copy()
        
        # Initialize components
        self.stream_buffer = StreamBuffer(self.config)
        self.stream_normalizer = StreamNormalizer(self.config)
        self.candle_builder = CandleBuilder(self.config)
        self.rolling_state_manager = RollingStateManager(self.config)
        self.stream_processor = StreamProcessor(self.config)
        self.stream_metrics = StreamMetrics(self.config)
        
        # Initialize all components
        await self.stream_buffer.initialize()
        await self.stream_normalizer.initialize()
        await self.candle_builder.initialize()
        await self.rolling_state_manager.initialize()
        await self.stream_processor.initialize()
        await self.stream_metrics.initialize()
        
        yield
        
        # Cleanup
        await self.stream_metrics.shutdown()
        await self.stream_processor.shutdown()
        await self.rolling_state_manager.shutdown()
        await self.candle_builder.shutdown()
        await self.stream_normalizer.shutdown()
        await self.stream_buffer.shutdown()
    
    @pytest.mark.asyncio
    async def test_stream_buffer_initialization(self):
        """Test stream buffer initialization"""
        assert self.stream_buffer.is_connected
        assert self.stream_buffer.is_running
        logger.info("✅ Stream buffer initialization test passed")
    
    @pytest.mark.asyncio
    async def test_stream_normalizer_initialization(self):
        """Test stream normalizer initialization"""
        assert self.stream_normalizer.is_running
        logger.info("✅ Stream normalizer initialization test passed")
    
    @pytest.mark.asyncio
    async def test_candle_builder_initialization(self):
        """Test candle builder initialization"""
        assert self.candle_builder.is_running
        assert len(self.candle_builder.timeframes) > 0
        logger.info("✅ Candle builder initialization test passed")
    
    @pytest.mark.asyncio
    async def test_rolling_state_manager_initialization(self):
        """Test rolling state manager initialization"""
        assert self.rolling_state_manager.is_running
        logger.info("✅ Rolling state manager initialization test passed")
    
    @pytest.mark.asyncio
    async def test_stream_processor_initialization(self):
        """Test stream processor initialization"""
        assert self.stream_processor.is_running
        logger.info("✅ Stream processor initialization test passed")
    
    @pytest.mark.asyncio
    async def test_stream_metrics_initialization(self):
        """Test stream metrics initialization"""
        assert self.stream_metrics.is_running
        logger.info("✅ Stream metrics initialization test passed")
    
    @pytest.mark.asyncio
    async def test_message_publishing(self):
        """Test message publishing to stream buffer"""
        # Create test message
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            data_type="tick",
            data={
                "price": 50000.0,
                "volume": 100.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            source="test"
        )
        
        # Publish message
        message_id = await self.stream_buffer.publish_message(message)
        assert message_id is not None
        
        # Check metrics
        metrics = self.stream_buffer.get_metrics()
        assert metrics['messages_received'] > 0
        
        logger.info("✅ Message publishing test passed")
    
    @pytest.mark.asyncio
    async def test_data_normalization(self):
        """Test data normalization"""
        # Create test message
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            data_type="tick",
            data={
                "price": 50000.0,
                "volume": 100.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            source="test"
        )
        
        # Normalize message
        normalized_data = await self.stream_normalizer.normalize_message(message)
        assert normalized_data is not None
        assert normalized_data.validation_status == 'valid'
        assert normalized_data.confidence_score > 0
        
        # Check metrics
        metrics = self.stream_normalizer.get_metrics()
        assert metrics['messages_processed'] > 0
        
        logger.info("✅ Data normalization test passed")
    
    @pytest.mark.asyncio
    async def test_candle_building(self):
        """Test candle building"""
        # Create test normalized data
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            data_type="tick",
            data={
                "price": 50000.0,
                "volume": 100.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            source="test"
        )
        
        normalized_data = await self.stream_normalizer.normalize_message(message)
        
        # Process tick for candle building
        candles = await self.candle_builder.process_tick(normalized_data)
        assert isinstance(candles, list)
        
        # Check metrics
        metrics = self.candle_builder.get_metrics()
        assert metrics['candles_updated'] > 0
        
        logger.info("✅ Candle building test passed")
    
    @pytest.mark.asyncio
    async def test_rolling_state_management(self):
        """Test rolling state management"""
        # Update rolling window
        window = await self.rolling_state_manager.update_rolling_window(
            "BTCUSDT", "1m", "price", 50000.0
        )
        assert window is not None
        assert len(window.data) > 0
        
        # Check metrics
        metrics = self.rolling_state_manager.get_metrics()
        assert metrics['windows_updated'] > 0
        
        logger.info("✅ Rolling state management test passed")
    
    @pytest.mark.asyncio
    async def test_stream_processing_pipeline(self):
        """Test complete stream processing pipeline"""
        # Create test message
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            data_type="tick",
            data={
                "price": 50000.0,
                "volume": 100.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            source="test"
        )
        
        # Process message through pipeline
        result = await self.stream_processor.process_message(message)
        assert result is not None
        assert result.success
        assert len(result.components_processed) > 0
        
        # Check metrics
        metrics = self.stream_processor.get_metrics()
        assert metrics['messages_processed'] > 0
        
        logger.info("✅ Stream processing pipeline test passed")
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection"""
        # Wait for metrics collection
        await asyncio.sleep(10)
        
        # Get current metrics
        metrics = self.stream_metrics.get_current_metrics()
        assert metrics is not None
        assert 'system_metrics' in metrics
        assert 'component_metrics' in metrics
        
        # Get metrics history
        history = self.stream_metrics.get_metrics_history(hours=1)
        assert isinstance(history, list)
        
        logger.info("✅ Metrics collection test passed")
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch message processing"""
        # Create batch of messages
        messages = []
        for i in range(10):
            message = StreamMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                data_type="tick",
                data={
                    "price": 50000.0 + i,
                    "volume": 100.0 + i,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                source="test"
            )
            messages.append(message)
        
        # Process batch
        results = await self.stream_processor.process_batch(messages)
        assert len(results) == 10
        
        # Check all results are successful
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 10
        
        logger.info("✅ Batch processing test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling"""
        # Create invalid message
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            data_type="tick",
            data={
                "price": -1.0,  # Invalid negative price
                "volume": -1.0,  # Invalid negative volume
                "timestamp": "invalid_timestamp"
            },
            source="test"
        )
        
        # Process message
        result = await self.stream_processor.process_message(message)
        assert result is not None
        # Should fail due to invalid data
        assert not result.success
        assert len(result.errors) > 0
        
        logger.info("✅ Error handling test passed")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test performance metrics"""
        # Process multiple messages to generate metrics
        for i in range(50):
            message = StreamMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                data_type="tick",
                data={
                    "price": 50000.0 + (i * 0.1),
                    "volume": 100.0 + i,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                source="test"
            )
            await self.stream_processor.process_message(message)
        
        # Wait for metrics collection
        await asyncio.sleep(5)
        
        # Check performance metrics
        metrics = self.stream_processor.get_metrics()
        assert metrics['avg_processing_time_ms'] > 0
        assert metrics['messages_processed'] >= 50
        
        logger.info("✅ Performance metrics test passed")
    
    @pytest.mark.asyncio
    async def test_component_integration(self):
        """Test component integration"""
        # Test that all components work together
        message = StreamMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            data_type="tick",
            data={
                "price": 50000.0,
                "volume": 100.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            source="test"
        )
        
        # Process through pipeline
        result = await self.stream_processor.process_message(message)
        assert result.success
        
        # Check that all components have processed data
        buffer_metrics = self.stream_buffer.get_metrics()
        normalizer_metrics = self.stream_normalizer.get_metrics()
        candle_metrics = self.candle_builder.get_metrics()
        rolling_metrics = self.rolling_state_manager.get_metrics()
        
        assert buffer_metrics['messages_received'] > 0
        assert normalizer_metrics['messages_processed'] > 0
        assert candle_metrics['candles_updated'] > 0
        assert rolling_metrics['windows_updated'] > 0
        
        logger.info("✅ Component integration test passed")

async def run_performance_test():
    """Run performance test"""
    logger.info("🚀 Starting performance test...")
    
    # Initialize components
    config = STREAMING_CONFIG.copy()
    stream_processor = StreamProcessor(config)
    await stream_processor.initialize()
    
    try:
        # Generate test messages
        start_time = time.time()
        message_count = 1000
        
        for i in range(message_count):
            message = StreamMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                data_type="tick",
                data={
                    "price": 50000.0 + (i * 0.01),
                    "volume": 100.0 + (i * 0.1),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                source="performance_test"
            )
            await stream_processor.process_message(message)
        
        end_time = time.time()
        processing_time = end_time - start_time
        throughput = message_count / processing_time
        
        # Get metrics
        metrics = stream_processor.get_metrics()
        
        logger.info(f"📊 Performance Test Results:")
        logger.info(f"   Messages processed: {message_count}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Throughput: {throughput:.2f} messages/second")
        logger.info(f"   Avg processing time: {metrics['avg_processing_time_ms']:.2f} ms")
        logger.info(f"   Success rate: {(metrics['messages_successful'] / metrics['messages_processed']) * 100:.2f}%")
        
        # Performance assertions
        assert throughput > 100, f"Throughput too low: {throughput} msg/s"
        assert metrics['avg_processing_time_ms'] < 100, f"Processing time too high: {metrics['avg_processing_time_ms']} ms"
        assert metrics['messages_successful'] / metrics['messages_processed'] > 0.95, "Success rate too low"
        
        logger.info("✅ Performance test passed!")
        
    finally:
        await stream_processor.shutdown()

async def run_integration_test():
    """Run integration test"""
    logger.info("🔗 Starting integration test...")
    
    # Initialize all components
    config = STREAMING_CONFIG.copy()
    stream_processor = StreamProcessor(config)
    stream_metrics = StreamMetrics(config)
    
    await stream_processor.initialize()
    await stream_metrics.initialize()
    
    try:
        # Test multiple symbols and timeframes
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        timeframes = ["1m", "5m", "15m"]
        
        for symbol in symbols:
            for i in range(100):
                message = StreamMessage(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    data_type="tick",
                    data={
                        "price": 50000.0 + (i * 0.1),
                        "volume": 100.0 + i,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    source="integration_test"
                )
                await stream_processor.process_message(message)
        
        # Wait for metrics collection
        await asyncio.sleep(10)
        
        # Check integration metrics
        metrics = stream_metrics.get_current_metrics()
        assert metrics is not None
        assert 'system_metrics' in metrics
        assert 'component_metrics' in metrics
        assert 'aggregated_metrics' in metrics
        
        logger.info("✅ Integration test passed!")
        
    finally:
        await stream_metrics.shutdown()
        await stream_processor.shutdown()

if __name__ == "__main__":
    """Run tests"""
    logger.info("🧪 Starting Streaming Infrastructure Tests...")
    
    # Run performance test
    asyncio.run(run_performance_test())
    
    # Run integration test
    asyncio.run(run_integration_test())
    
    logger.info("🎉 All tests completed successfully!")
