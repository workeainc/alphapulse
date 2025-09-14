#!/usr/bin/env python3
"""
Week 7.2 Phase 3: Real-Time Anomaly Detection Pipeline

This pipeline integrates with the existing real-time data pipeline to provide
comprehensive anomaly detection for market data, including:
- Price anomalies (Z-score, IQR)
- Volume anomalies (spikes, unusual patterns)
- Order book anomalies (imbalances, liquidity gaps)
- Cross-market anomalies (correlations, regime shifts)

Author: AlphaPulse Team
Date: 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import json

# Import our statistical filters service
from .statistical_filters import StatisticalFilters, FilterConfig, AnomalyResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    """Anomaly alert for real-time notification"""
    timestamp: datetime
    symbol: str
    alert_type: str  # 'price', 'volume', 'order_book', 'cross_market'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    action_required: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the anomaly detection pipeline"""
    enable_price_anomalies: bool = True
    enable_volume_anomalies: bool = True
    enable_order_book_anomalies: bool = True
    enable_cross_market_anomalies: bool = True
    alert_threshold: str = 'medium'  # 'low', 'medium', 'high', 'critical'
    max_alerts_per_minute: int = 100
    enable_auto_trading_pause: bool = False
    trading_pause_threshold: str = 'critical'
    data_buffer_size: int = 1000
    correlation_window: int = 100  # Data points for correlation analysis


class RealTimeAnomalyPipeline:
    """
    Real-time anomaly detection pipeline for market data
    
    This pipeline integrates with the statistical filters service to provide
    comprehensive anomaly detection across multiple market data types.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the anomaly detection pipeline"""
        self.config = config or PipelineConfig()
        
        # Initialize statistical filters service
        filter_config = FilterConfig(
            z_score_threshold=2.5,
            iqr_multiplier=1.8,
            volume_threshold=2.5,
            order_book_imbalance_threshold=0.7
        )
        self.statistical_filters = StatisticalFilters(filter_config)
        
        # Data buffers for different market data types
        self.market_data_buffers: Dict[str, Dict[str, deque]] = {}
        self.correlation_data: Dict[str, Dict[str, deque]] = {}
        
        # Alert management
        self.alerts: deque = deque(maxlen=1000)
        self.alert_counters: Dict[str, int] = {}
        self.last_alert_reset = datetime.now()
        
        # Callbacks for different types of events
        self.anomaly_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.trading_pause_callbacks: List[Callable] = []
        
        # Pipeline statistics
        self.stats = {
            'total_anomalies_detected': 0,
            'price_anomalies': 0,
            'volume_anomalies': 0,
            'order_book_anomalies': 0,
            'cross_market_anomalies': 0,
            'alerts_generated': 0,
            'trading_pauses_triggered': 0,
            'last_anomaly': None,
            'pipeline_start_time': datetime.now()
        }
        
        logger.info("✅ Real-Time Anomaly Pipeline initialized")
    
    async def initialize(self):
        """Initialize the pipeline and all components"""
        logger.info("🚀 Initializing Real-Time Anomaly Pipeline...")
        
        # Initialize statistical filters
        await self.statistical_filters.initialize()
        
        # Set up anomaly callbacks
        self.statistical_filters.add_callback(self._on_anomaly_detected)
        
        # Initialize alert counter reset timer
        asyncio.create_task(self._reset_alert_counters())
        
        logger.info("✅ Real-Time Anomaly Pipeline ready")
    
    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a new symbol"""
        if symbol not in self.market_data_buffers:
            self.market_data_buffers[symbol] = {
                'price': deque(maxlen=self.config.data_buffer_size),
                'volume': deque(maxlen=self.config.data_buffer_size),
                'bid_ask_spread': deque(maxlen=self.config.data_buffer_size),
                'order_imbalance': deque(maxlen=self.config.data_buffer_size),
                'timestamp': deque(maxlen=self.config.data_buffer_size)
            }
        
        if symbol not in self.correlation_data:
            self.correlation_data[symbol] = {
                'price_changes': deque(maxlen=self.config.correlation_window),
                'volume_changes': deque(maxlen=self.config.correlation_window),
                'spread_changes': deque(maxlen=self.config.correlation_window)
            }
    
    async def process_market_data(self, symbol: str, market_data: Dict[str, Any]) -> List[AnomalyResult]:
        """
        Process incoming market data for anomaly detection
        
        Args:
            symbol: Trading symbol
            market_data: Dictionary containing market data
            
        Returns:
            List of detected anomalies
        """
        # Initialize buffers if needed
        self._initialize_symbol_buffers(symbol)
        
        # Extract data
        timestamp = market_data.get('timestamp', datetime.now())
        price = market_data.get('price', 0.0)
        volume = market_data.get('volume', 0.0)
        bid_ask_spread = market_data.get('bid_ask_spread', 0.0)
        order_imbalance = market_data.get('order_imbalance', 0.0)
        
        # Store in buffers
        self.market_data_buffers[symbol]['price'].append(price)
        self.market_data_buffers[symbol]['volume'].append(volume)
        self.market_data_buffers[symbol]['bid_ask_spread'].append(bid_ask_spread)
        self.market_data_buffers[symbol]['order_imbalance'].append(order_imbalance)
        self.market_data_buffers[symbol]['timestamp'].append(timestamp)
        
        # Calculate changes for correlation analysis
        if len(self.market_data_buffers[symbol]['price']) > 1:
            price_change = (price - list(self.market_data_buffers[symbol]['price'])[-2]) / list(self.market_data_buffers[symbol]['price'])[-2]
            self.correlation_data[symbol]['price_changes'].append(price_change)
        
        if len(self.market_data_buffers[symbol]['volume']) > 1:
            volume_change = (volume - list(self.market_data_buffers[symbol]['volume'])[-2]) / max(list(self.market_data_buffers[symbol]['volume'])[-2], 1.0)
            self.correlation_data[symbol]['volume_changes'].append(volume_change)
        
        # Detect anomalies
        anomalies = []
        
        # Price anomaly detection
        if self.config.enable_price_anomalies and price > 0:
            price_anomaly = await self.statistical_filters.detect_price_anomaly(symbol, price, timestamp)
            if price_anomaly:
                anomalies.append(price_anomaly)
                self.stats['price_anomalies'] += 1
        
        # Volume anomaly detection
        if self.config.enable_volume_anomalies and volume > 0:
            volume_anomaly = await self.statistical_filters.detect_volume_anomaly(symbol, volume, None, timestamp)
            if volume_anomaly:
                anomalies.append(volume_anomaly)
                self.stats['volume_anomalies'] += 1
        
        # Order book anomaly detection
        if self.config.enable_order_book_anomalies:
            # Simulate bid/ask volumes for order book imbalance detection
            bid_volume = volume * (1 - order_imbalance) / 2
            ask_volume = volume * (1 + order_imbalance) / 2
            
            order_book_anomaly = await self.statistical_filters.detect_order_book_anbalance(
                symbol, bid_volume, ask_volume, timestamp
            )
            if order_book_anomaly:
                anomalies.append(order_book_anomaly)
                self.stats['order_book_anomalies'] += 1
        
        # Cross-market anomaly detection
        if self.config.enable_cross_market_anomalies:
            cross_market_anomaly = await self._detect_cross_market_anomaly(symbol, market_data)
            if cross_market_anomaly:
                anomalies.append(cross_market_anomaly)
                self.stats['cross_market_anomalies'] += 1
        
        # Update statistics
        if anomalies:
            self.stats['total_anomalies_detected'] += len(anomalies)
            self.stats['last_anomaly'] = datetime.now()
        
        return anomalies
    
    async def _detect_cross_market_anomaly(self, symbol: str, market_data: Dict[str, Any]) -> Optional[AnomalyResult]:
        """Detect cross-market anomalies using correlation analysis"""
        if symbol not in self.correlation_data:
            return None
        
        # Check if we have enough data for correlation analysis
        if len(self.correlation_data[symbol]['price_changes']) < self.config.correlation_window // 2:
            return None
        
        # Calculate correlations between different metrics
        price_changes = list(self.correlation_data[symbol]['price_changes'])
        volume_changes = list(self.correlation_data[symbol]['volume_changes'])
        
        if len(price_changes) < 10 or len(volume_changes) < 10:
            return None
        
        # Calculate correlation between price and volume changes
        try:
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            
            # Check for unusual correlation patterns
            # Normal correlation should be between -0.8 and 0.8
            if abs(correlation) > 0.9:
                # This is an unusual correlation pattern
                severity = 'high' if abs(correlation) > 0.95 else 'medium'
                
                return AnomalyResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    anomaly_type='cross_market_correlation',
                    severity=severity,
                    confidence=min(abs(correlation), 1.0),
                    value=correlation,
                    threshold=0.9,
                    description=f"Unusual price-volume correlation: {correlation:.3f}",
                    metadata={
                        'correlation': correlation,
                        'price_changes_count': len(price_changes),
                        'volume_changes_count': len(volume_changes),
                        'method': 'correlation_analysis'
                    }
                )
        except Exception as e:
            logger.warning(f"Error calculating correlation for {symbol}: {e}")
        
        return None
    
    def _on_anomaly_detected(self, anomaly: AnomalyResult):
        """Callback when an anomaly is detected by the statistical filters"""
        # Check if we should generate an alert
        if self._should_generate_alert(anomaly):
            alert = self._create_anomaly_alert(anomaly)
            self.alerts.append(alert)
            self.stats['alerts_generated'] += 1
            
            # Trigger alert callbacks
            self._trigger_alert_callbacks(alert)
            
            # Check if we should pause trading
            if self.config.enable_auto_trading_pause and anomaly.severity == self.config.trading_pause_threshold:
                self._trigger_trading_pause(anomaly)
    
    def _should_generate_alert(self, anomaly: AnomalyResult) -> bool:
        """Determine if an alert should be generated based on severity and rate limits"""
        # Check severity threshold
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        alert_threshold_level = severity_levels.get(self.config.alert_threshold, 2)
        anomaly_level = severity_levels.get(anomaly.severity, 1)
        
        if anomaly_level < alert_threshold_level:
            return False
        
        # Check rate limiting
        current_minute = datetime.now().replace(second=0, microsecond=0)
        if current_minute not in self.alert_counters:
            self.alert_counters[current_minute] = 0
        
        if self.alert_counters[current_minute] >= self.config.max_alerts_per_minute:
            return False
        
        self.alert_counters[current_minute] += 1
        return True
    
    def _create_anomaly_alert(self, anomaly: AnomalyResult) -> AnomalyAlert:
        """Create an anomaly alert from an anomaly result"""
        return AnomalyAlert(
            timestamp=anomaly.timestamp,
            symbol=anomaly.symbol,
            alert_type=anomaly.anomaly_type,
            severity=anomaly.severity,
            message=anomaly.description,
            data=anomaly.metadata,
            action_required=anomaly.severity in ['high', 'critical']
        )
    
    def _trigger_alert_callbacks(self, alert: AnomalyAlert):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                asyncio.create_task(callback(alert))
            except Exception as e:
                logger.error(f"❌ Error in alert callback {callback.__name__}: {e}")
    
    def _trigger_trading_pause(self, anomaly: AnomalyResult):
        """Trigger trading pause callbacks for critical anomalies"""
        logger.warning(f"🚨 CRITICAL ANOMALY: Triggering trading pause for {anomaly.symbol}")
        
        for callback in self.trading_pause_callbacks:
            try:
                asyncio.create_task(callback(anomaly))
            except Exception as e:
                logger.error(f"❌ Error in trading pause callback {callback.__name__}: {e}")
        
        self.stats['trading_pauses_triggered'] += 1
    
    async def _reset_alert_counters(self):
        """Reset alert counters every minute"""
        while True:
            await asyncio.sleep(60)  # Wait 1 minute
            current_minute = datetime.now().replace(second=0, microsecond=0)
            
            # Clean up old counters
            old_counters = [k for k in self.alert_counters.keys() if k < current_minute]
            for old_counter in old_counters:
                del self.alert_counters[old_counter]
    
    def add_anomaly_callback(self, callback: Callable):
        """Add a callback for anomaly events"""
        if callback not in self.anomaly_callbacks:
            self.anomaly_callbacks.append(callback)
            logger.info(f"✅ Added anomaly callback: {callback.__name__}")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback for alert events"""
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            logger.info(f"✅ Added alert callback: {callback.__name__}")
    
    def add_trading_pause_callback(self, callback: Callable):
        """Add a callback for trading pause events"""
        if callback not in self.trading_pause_callbacks:
            self.trading_pause_callbacks.append(callback)
            logger.info(f"✅ Added trading pause callback: {callback.__name__}")
    
    async def get_recent_alerts(self, symbol: Optional[str] = None, 
                               alert_type: Optional[str] = None,
                               limit: int = 100) -> List[AnomalyAlert]:
        """Get recent alerts with optional filtering"""
        alerts = list(self.alerts)
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts[-limit:]
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'total_anomalies_detected': self.stats['total_anomalies_detected'],
            'price_anomalies': self.stats['price_anomalies'],
            'volume_anomalies': self.stats['volume_anomalies'],
            'order_book_anomalies': self.stats['order_book_anomalies'],
            'cross_market_anomalies': self.stats['cross_market_anomalies'],
            'alerts_generated': self.stats['alerts_generated'],
            'trading_pauses_triggered': self.stats['trading_pauses_triggered'],
            'last_anomaly': self.stats['last_anomaly'],
            'pipeline_start_time': self.stats['pipeline_start_time'],
            'active_symbols': len(self.market_data_buffers),
            'total_callbacks': len(self.anomaly_callbacks) + len(self.alert_callbacks) + len(self.trading_pause_callbacks)
        }
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Closing Real-Time Anomaly Pipeline...")
        
        # Close statistical filters
        await self.statistical_filters.close()
        
        # Clear buffers and data
        self.market_data_buffers.clear()
        self.correlation_data.clear()
        self.alerts.clear()
        self.alert_counters.clear()
        
        # Clear callbacks
        self.anomaly_callbacks.clear()
        self.alert_callbacks.clear()
        self.trading_pause_callbacks.clear()
        
        logger.info("✅ Real-Time Anomaly Pipeline closed")


# Example usage and testing
async def main():
    """Example usage of the Real-Time Anomaly Pipeline"""
    # Create pipeline with custom config
    config = PipelineConfig(
        enable_price_anomalies=True,
        enable_volume_anomalies=True,
        enable_order_book_anomalies=True,
        enable_cross_market_anomalies=True,
        alert_threshold='medium',
        max_alerts_per_minute=50
    )
    
    pipeline = RealTimeAnomalyPipeline(config)
    await pipeline.initialize()
    
    # Add callbacks
    def anomaly_callback(anomaly: AnomalyResult):
        print(f"🚨 ANOMALY: {anomaly.symbol} - {anomaly.description}")
    
    def alert_callback(alert: AnomalyAlert):
        print(f"📢 ALERT: {alert.symbol} - {alert.message}")
    
    def trading_pause_callback(anomaly: AnomalyResult):
        print(f"⏸️ TRADING PAUSE: {anomaly.symbol} - {anomaly.description}")
    
    pipeline.add_anomaly_callback(anomaly_callback)
    pipeline.add_alert_callback(alert_callback)
    pipeline.add_trading_pause_callback(trading_pause_callback)
    
    # Test with simulated market data
    print("🧪 Testing Real-Time Anomaly Pipeline...")
    
    # Simulate normal market data
    for i in range(100):
        market_data = {
            'timestamp': datetime.now(),
            'price': 50000 + (i * 10),
            'volume': 100000 + (i * 1000),
            'bid_ask_spread': 0.001 + (i * 0.0001),
            'order_imbalance': 0.1 + (i * 0.01)
        }
        
        anomalies = await pipeline.process_market_data("BTCUSDT", market_data)
        if anomalies:
            print(f"✅ Detected {len(anomalies)} anomalies")
    
    # Test with anomalous data
    print("\n🧪 Testing with anomalous data...")
    
    # Price anomaly
    anomaly_data = {
        'timestamp': datetime.now(),
        'price': 70000,  # Much higher than normal
        'volume': 200000,
        'bid_ask_spread': 0.002,
        'order_imbalance': 0.2
    }
    
    anomalies = await pipeline.process_market_data("BTCUSDT", anomaly_data)
    if anomalies:
        print(f"✅ Detected {len(anomalies)} anomalies in anomalous data")
    
    # Print statistics
    print(f"\n📊 Pipeline Statistics:")
    stats = pipeline.get_pipeline_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Print recent alerts
    print(f"\n📢 Recent Alerts:")
    alerts = await pipeline.get_recent_alerts(limit=5)
    for alert in alerts:
        print(f"  {alert.timestamp}: {alert.symbol} - {alert.message}")
    
    await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
