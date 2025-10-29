"""
Streaming Pipeline for AlphaPulse
Phase 5: Low-latency streaming with Kafka/Flink integration
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading

# Kafka imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Kafka not available - using in-memory fallback")

logger = logging.getLogger(__name__)

@dataclass
class StreamMessage:
    """Stream message for Kafka processing"""
    id: str
    timestamp: datetime
    topic: str
    key: str
    value: Dict[str, Any]
    partition: int = 0
    offset: int = 0

@dataclass
class StreamMetrics:
    """Streaming metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    avg_latency_ms: float = 0.0
    throughput_mps: float = 0.0
    error_count: int = 0

class KafkaStreamManager:
    """Kafka streaming manager"""
    
    def __init__(self, bootstrap_servers: List[str] = ['localhost:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumers = {}
        self.is_running = False
        self.metrics = StreamMetrics()
        self.message_buffer = deque(maxlen=10000)
        self.processing_callbacks = {}
        self.latency_history = deque(maxlen=1000)
        
        logger.info("Kafka Stream Manager initialized")
    
    async def start(self):
        """Start Kafka streaming"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - using simulation")
            self.is_running = True
            return
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8'),
                acks='all',
                retries=3
            )
            self.is_running = True
            logger.info("🚀 Kafka Stream Manager started")
        except Exception as e:
            logger.error(f"Error starting Kafka: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop Kafka streaming"""
        self.is_running = False
        if self.producer:
            self.producer.close()
        for consumer in self.consumers.values():
            consumer.close()
        logger.info("🛑 Kafka Stream Manager stopped")
    
    async def send_message(self, topic: str, key: str, value: Dict[str, Any]) -> str:
        """Send message to Kafka topic"""
        message_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            if not KAFKA_AVAILABLE:
                message = StreamMessage(
                    id=message_id,
                    timestamp=datetime.now(),
                    topic=topic,
                    key=key,
                    value=value
                )
                self.message_buffer.append(message)
                self._update_metrics('sent', time.time() - start_time)
                return message_id
            
            future = self.producer.send(topic=topic, key=key, value=value)
            record_metadata = future.get(timeout=10)
            
            latency = (time.time() - start_time) * 1000
            self._update_metrics('sent', latency)
            return message_id
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error sending message: {e}")
            raise
    
    def _update_metrics(self, direction: str, latency_ms: float):
        """Update metrics"""
        if direction == 'sent':
            self.metrics.messages_sent += 1
        else:
            self.metrics.messages_received += 1
        
        self.latency_history.append(latency_ms)
        if self.latency_history:
            self.metrics.avg_latency_ms = sum(self.latency_history) / len(self.latency_history)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        return {
            'messages_sent': self.metrics.messages_sent,
            'messages_received': self.metrics.messages_received,
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'error_count': self.metrics.error_count,
            'kafka_available': KAFKA_AVAILABLE,
            'is_running': self.is_running
        }

class StreamingPipeline:
    """Main streaming pipeline"""
    
    def __init__(self):
        self.kafka_manager = KafkaStreamManager()
        self.is_running = False
        
        logger.info("Streaming Pipeline initialized")
    
    async def start(self):
        """Start streaming pipeline"""
        await self.kafka_manager.start()
        self.is_running = True
        logger.info("🚀 Streaming Pipeline started")
    
    async def stop(self):
        """Stop streaming pipeline"""
        self.is_running = False
        await self.kafka_manager.stop()
        logger.info("🛑 Streaming Pipeline stopped")
    
    async def publish_signal(self, signal_data: Dict[str, Any]) -> str:
        """Publish trading signal"""
        return await self.kafka_manager.send_message(
            topic='trading_signals',
            key=f"signal_{signal_data.get('id', uuid.uuid4())}",
            value=signal_data
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'pipeline_running': self.is_running,
            'kafka_metrics': self.kafka_manager.get_metrics()
        }

# Global instance
streaming_pipeline = StreamingPipeline()
