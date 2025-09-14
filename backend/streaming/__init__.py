"""
AlphaPulse Streaming Infrastructure
Phase 1: Real-time data streaming with Redis Streams and TimescaleDB integration

This module provides the core streaming infrastructure for real-time market data
processing, signal generation, and outcome tracking with enterprise-grade features.
"""

from .stream_buffer import StreamBuffer
from .stream_normalizer import StreamNormalizer
from .candle_builder import CandleBuilder
from .rolling_state_manager import RollingStateManager
from .stream_processor import StreamProcessor
from .stream_metrics import StreamMetrics
from .backpressure_handler import BackpressureHandler
from .failover_manager import FailoverManager
from .stream_encryption import StreamEncryption
from .stream_monitoring import StreamMonitoring
from .protocol_adapters import ProtocolAdapters
from .disaster_recovery import DisasterRecovery
from .capacity_planner import CapacityPlanner
from .api_protection import APIProtection

__version__ = "1.0.0"
__author__ = "AlphaPulse Team"

# Export main classes
__all__ = [
    'StreamBuffer',
    'StreamNormalizer', 
    'CandleBuilder',
    'RollingStateManager',
    'StreamProcessor',
    'StreamMetrics',
    'BackpressureHandler',
    'FailoverManager',
    'StreamEncryption',
    'StreamMonitoring',
    'ProtocolAdapters',
    'DisasterRecovery',
    'CapacityPlanner',
    'APIProtection'
]
