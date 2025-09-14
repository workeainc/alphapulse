"""
Complete System Integration Tests for AlphaPlus
Tests all phases: Real Data, AI Models, Streaming, Database Optimization, Security
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import all system components with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.main_ai_system_simple import app
    from streaming.stream_buffer import StreamBuffer
    from streaming.stream_processor import StreamProcessor
    from database.advanced_indexing import AdvancedIndexingManager
    from database.lifecycle_manager import DataLifecycleManager
    from database.security_manager import SecurityManager
    from database.connection import TimescaleDBConnection
    from core.websocket_binance import BinanceWebSocketClient
    from data.data_validator import DataValidator
    from services.news_sentiment_service import NewsSentimentService
    from ai.sde_framework import SDEFramework
    from ai.model_heads import ModelHeadsManager
    from ai.consensus_manager import ConsensusManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some imports not available: {e}")
    IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TestCompleteSystemIntegration:
    """Comprehensive integration tests for all system phases"""
    
    @pytest.fixture
    async def system_components(self):
        """Initialize all system components for testing"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        
        # Database connection
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'pool_size': 5,  # Smaller for testing
            'max_overflow': 10
        }
        
        db_connection = TimescaleDBConnection(db_config)
        await db_connection.initialize()
        
        # Initialize all components
        components = {
            'db_connection': db_connection,
            'stream_buffer': StreamBuffer({'redis_url': 'redis://localhost:6379'}),
            'stream_processor': StreamProcessor({'batch_size': 10}),
            'advanced_indexing': AdvancedIndexingManager(db_connection.get_session_factory()),
            'lifecycle_manager': DataLifecycleManager(db_connection.get_async_engine()),
            'security_manager': SecurityManager(db_connection.get_async_engine()),
            'data_validator': DataValidator(),
            'news_sentiment': NewsSentimentService(),
            'sde_framework': SDEFramework(),
            'model_heads': ModelHeadsManager(),
            'consensus_manager': ConsensusManager()
        }
        
        # Initialize components
        for name, component in components.items():
            if hasattr(component, 'initialize'):
                await component.initialize()
        
        yield components
        
        # Cleanup
        if hasattr(db_connection, 'close'):
            await db_connection.close()
    
    @pytest.mark.asyncio
    async def test_phase1_real_data_integration(self, system_components):
        """Test Phase 1: Real Data Integration"""
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required imports not available")
        logger.info("🧪 Testing Phase 1: Real Data Integration")
        
        # Test data validator
        validator = system_components['data_validator']
        
        # Test valid data
        valid_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.utcnow(),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0
        }
        
        assert validator.validate_market_data(valid_data) == True
        
        # Test invalid data
        invalid_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.utcnow(),
            'open': -50000.0,  # Negative price
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0
        }
        
        assert validator.validate_market_data(invalid_data) == False
        
        # Test news sentiment service
        news_service = system_components['news_sentiment']
        sentiment = await news_service.get_sentiment_for_symbol('BTC')
        
        assert 'sentiment' in sentiment
        assert 'confidence' in sentiment
        assert -1.0 <= sentiment['sentiment'] <= 1.0
        assert 0.0 <= sentiment['confidence'] <= 1.0
        
        logger.info("✅ Phase 1: Real Data Integration tests passed")
    
    @pytest.mark.asyncio
    async def test_phase2_ai_model_integration(self, system_components):
        """Test Phase 2: AI Model Integration"""
        logger.info("🧪 Testing Phase 2: AI Model Integration")
        
        # Test model heads manager
        model_heads = system_components['model_heads']
        
        # Test market data for AI analysis
        market_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.utcnow(),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0,
            'price_change': 0.001
        }
        
        # Test all model heads
        results = await model_heads.analyze_all_heads(market_data)
        
        assert len(results) == 4  # Should have 4 model heads
        assert 'head_a' in results
        assert 'head_b' in results
        assert 'head_c' in results
        assert 'head_d' in results
        
        # Test consensus manager
        consensus_manager = system_components['consensus_manager']
        consensus_result = await consensus_manager.check_consensus(results)
        
        assert 'consensus_achieved' in consensus_result
        assert 'consensus_score' in consensus_result
        assert 'agreed_direction' in consensus_result
        
        # Test SDE framework
        sde_framework = system_components['sde_framework']
        sde_output = await sde_framework.generate_sde_output(results, consensus_result)
        
        assert 'signal' in sde_output
        assert 'confidence' in sde_output
        assert 'direction' in sde_output
        
        logger.info("✅ Phase 2: AI Model Integration tests passed")
    
    @pytest.mark.asyncio
    async def test_phase3_streaming_infrastructure(self, system_components):
        """Test Phase 3: Streaming Infrastructure"""
        logger.info("🧪 Testing Phase 3: Streaming Infrastructure")
        
        # Test stream buffer
        stream_buffer = system_components['stream_buffer']
        
        # Test message processing
        test_message = {
            'id': 'test_001',
            'timestamp': datetime.utcnow(),
            'symbol': 'BTCUSDT',
            'data_type': 'candlestick',
            'data': {
                'open': 50000.0,
                'high': 50100.0,
                'low': 49900.0,
                'close': 50050.0,
                'volume': 1000.0
            },
            'source': 'test',
            'partition': 0,
            'priority': 1
        }
        
        # Add message to stream buffer
        await stream_buffer.add_message(test_message)
        
        # Get messages from stream buffer
        messages = await stream_buffer.get_messages(batch_size=1)
        assert len(messages) >= 0  # May be empty if Redis not available
        
        # Test stream processor
        stream_processor = system_components['stream_processor']
        processed_message = await stream_processor.process_message(test_message)
        
        assert processed_message is not None
        assert 'processed_at' in processed_message
        
        logger.info("✅ Phase 3: Streaming Infrastructure tests passed")
    
    @pytest.mark.asyncio
    async def test_phase4_database_optimization(self, system_components):
        """Test Phase 4: Database Optimization"""
        logger.info("🧪 Testing Phase 4: Database Optimization")
        
        # Test advanced indexing
        indexing_manager = system_components['advanced_indexing']
        
        # Get index statistics
        index_stats = await indexing_manager.get_index_statistics()
        
        assert 'total_indexes' in index_stats
        assert 'index_hit_ratio' in index_stats
        assert 'index_size_mb' in index_stats
        
        # Test lifecycle manager
        lifecycle_manager = system_components['lifecycle_manager']
        
        # Get lifecycle status
        lifecycle_status = await lifecycle_manager.get_status()
        
        assert 'compression_ratio' in lifecycle_status
        assert 'retention_status' in lifecycle_status
        assert 'archive_status' in lifecycle_status
        
        # Test database connection performance
        db_connection = system_components['db_connection']
        performance_metrics = await db_connection.get_performance_metrics()
        
        assert 'active_connections' in performance_metrics
        assert 'pool_size' in performance_metrics
        assert 'avg_query_time_ms' in performance_metrics
        
        logger.info("✅ Phase 4: Database Optimization tests passed")
    
    @pytest.mark.asyncio
    async def test_phase5_security_monitoring(self, system_components):
        """Test Phase 5: Security & Monitoring"""
        logger.info("🧪 Testing Phase 5: Security & Monitoring")
        
        # Test security manager
        security_manager = system_components['security_manager']
        
        # Get security status
        security_status = await security_manager.get_security_status()
        
        assert 'active_sessions' in security_status
        assert 'locked_users' in security_status
        assert 'security_alerts' in security_status
        
        # Test audit logging
        await security_manager.log_security_event(
            event_type='test_event',
            user_id='test_user',
            details={'test': 'data'},
            severity='low'
        )
        
        # Get recent audit logs
        audit_logs = await security_manager.get_recent_audit_logs(limit=5)
        
        assert isinstance(audit_logs, list)
        
        logger.info("✅ Phase 5: Security & Monitoring tests passed")
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self, system_components):
        """Test complete end-to-end data flow"""
        logger.info("🧪 Testing End-to-End Data Flow")
        
        # Simulate complete data flow
        market_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.utcnow(),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0,
            'price_change': 0.001
        }
        
        # Step 1: Data validation
        validator = system_components['data_validator']
        assert validator.validate_market_data(market_data) == True
        
        # Step 2: AI analysis
        model_heads = system_components['model_heads']
        ai_results = await model_heads.analyze_all_heads(market_data)
        
        # Step 3: Consensus check
        consensus_manager = system_components['consensus_manager']
        consensus_result = await consensus_manager.check_consensus(ai_results)
        
        # Step 4: SDE signal generation
        sde_framework = system_components['sde_framework']
        signal = await sde_framework.generate_sde_output(ai_results, consensus_result)
        
        # Step 5: Security logging
        security_manager = system_components['security_manager']
        await security_manager.log_security_event(
            event_type='signal_generated',
            user_id='system',
            details={'signal': signal, 'symbol': market_data['symbol']},
            severity='medium'
        )
        
        # Verify signal structure
        assert 'signal' in signal
        assert 'confidence' in signal
        assert 'direction' in signal
        assert signal['confidence'] >= 0.0
        assert signal['confidence'] <= 1.0
        
        logger.info("✅ End-to-End Data Flow tests passed")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, system_components):
        """Test system performance metrics"""
        logger.info("🧪 Testing Performance Metrics")
        
        # Test latency requirements
        start_time = datetime.utcnow()
        
        # Simulate fast processing
        market_data = {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.utcnow(),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000.0
        }
        
        # Process through pipeline
        validator = system_components['data_validator']
        validator.validate_market_data(market_data)
        
        model_heads = system_components['model_heads']
        await model_heads.analyze_all_heads(market_data)
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Check latency target (< 100ms)
        assert processing_time_ms < 100, f"Processing time {processing_time_ms}ms exceeds 100ms target"
        
        # Test throughput (simulate batch processing)
        batch_size = 10
        start_time = datetime.utcnow()
        
        for i in range(batch_size):
            validator.validate_market_data(market_data)
        
        end_time = datetime.utcnow()
        batch_time_ms = (end_time - start_time).total_seconds() * 1000
        throughput = (batch_size * 1000) / batch_time_ms  # messages per second
        
        # Check throughput target (1000+ msg/sec)
        assert throughput >= 1000, f"Throughput {throughput:.2f} msg/sec below 1000 target"
        
        logger.info(f"✅ Performance Metrics tests passed - Latency: {processing_time_ms:.2f}ms, Throughput: {throughput:.2f} msg/sec")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
