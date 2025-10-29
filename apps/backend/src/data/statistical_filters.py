#!/usr/bin/env python3
"""
Week 7.2 Phase 3: Real-Time Anomaly Detection with Statistical Filters

This service provides real-time anomaly detection using various statistical methods:
- Z-score filtering for standard deviation-based anomaly detection
- IQR (Interquartile Range) filtering for outlier detection
- Moving average-based anomaly detection
- Volume anomaly detection
- Order book imbalance anomaly detection

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis"""
    timestamp: datetime
    symbol: str
    anomaly_type: str  # 'price', 'volume', 'order_book', 'cross_chain'
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0.0 to 1.0
    value: float  # The anomalous value
    threshold: float  # The threshold that was exceeded
    description: str  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Configuration for statistical filters"""
    z_score_threshold: float = 3.0  # Standard deviations for Z-score
    iqr_multiplier: float = 1.5  # IQR multiplier for outlier detection
    moving_avg_window: int = 20  # Window for moving average calculations
    volume_threshold: float = 2.0  # Volume spike threshold
    order_book_imbalance_threshold: float = 0.7  # Order book imbalance threshold
    min_data_points: int = 30  # Minimum data points required for analysis
    enable_alerts: bool = True  # Enable real-time alerts
    alert_cooldown: int = 300  # Seconds between alerts for same anomaly type


class StatisticalFilters:
    """
    Real-time statistical filtering service for anomaly detection
    
    This service provides multiple statistical methods for detecting anomalies
    in real-time market data, including prices, volumes, and order book data.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """Initialize the statistical filters service"""
        self.config = config or FilterConfig()
        self.anomaly_history: Dict[str, deque] = {}  # Symbol -> recent anomalies
        self.data_buffers: Dict[str, Dict[str, deque]] = {}  # Symbol -> data type -> buffer
        self.callbacks: List[Callable] = []
        self.stats = {
            'anomalies_detected': 0,
            'total_checks': 0,
            'z_score_anomalies': 0,
            'iqr_anomalies': 0,
            'volume_anomalies': 0,
            'order_book_anomalies': 0,
            'last_anomaly': None
        }
        
        logger.info("✅ Statistical Filters service initialized")
    
    async def initialize(self):
        """Initialize the service"""
        logger.info("🚀 Initializing Statistical Filters service...")
        
        # Initialize data buffers for each data type
        self._initialize_buffers()
        
        logger.info("✅ Statistical Filters service ready")
    
    def _initialize_buffers(self):
        """Initialize data buffers for different data types"""
        # This method is called during initialization, but we'll initialize buffers
        # dynamically when symbols are first encountered
        pass
    
    def _ensure_symbol_buffers(self, symbol: str):
        """Ensure buffers exist for a symbol"""
        if symbol not in self.data_buffers:
            buffer_types = ['price', 'volume', 'bid_ask_spread', 'order_imbalance']
            max_buffer_size = max(self.config.min_data_points * 2, 100)
            
            self.data_buffers[symbol] = {}
            for data_type in buffer_types:
                self.data_buffers[symbol][data_type] = deque(maxlen=max_buffer_size)
    
    def add_callback(self, callback: Callable):
        """Add a callback function for anomaly alerts"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.info(f"✅ Added anomaly callback: {callback.__name__}")
    
    def _trigger_callbacks(self, anomaly: AnomalyResult):
        """Trigger all registered callbacks with anomaly data"""
        if not self.config.enable_alerts:
            return
        
        for callback in self.callbacks:
            try:
                asyncio.create_task(callback(anomaly))
            except Exception as e:
                logger.error(f"❌ Error in anomaly callback {callback.__name__}: {e}")
    
    def _check_alert_cooldown(self, symbol: str, anomaly_type: str) -> bool:
        """Check if enough time has passed since last alert for this anomaly type"""
        if symbol not in self.anomaly_history:
            return True
        
        # Check if we have a recent alert of the same type
        current_time = datetime.now()
        for anomaly in self.anomaly_history[symbol]:
            if (anomaly.anomaly_type == anomaly_type and 
                (current_time - anomaly.timestamp).seconds < self.config.alert_cooldown):
                return False
        
        return True
    
    async def detect_price_anomaly(self, symbol: str, price: float, 
                                 timestamp: Optional[datetime] = None) -> Optional[AnomalyResult]:
        """
        Detect price anomalies using Z-score and IQR methods
        
        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Timestamp of the price (defaults to now)
        
        Returns:
            AnomalyResult if anomaly detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.stats['total_checks'] += 1
        
        # Ensure we have enough data
        self._ensure_symbol_buffers(symbol)
        if len(self.data_buffers[symbol].get('price', [])) < self.config.min_data_points:
            # Add to buffer and continue
            self.data_buffers[symbol]['price'].append(price)
            return None
        
        price_buffer = self.data_buffers[symbol]['price']
        price_buffer.append(price)
        
        # Convert to numpy array for calculations
        prices = np.array(list(price_buffer))
        
        # Z-score anomaly detection
        z_score_anomaly = self._detect_z_score_anomaly(prices, price, symbol, 'price')
        if z_score_anomaly:
            return z_score_anomaly
        
        # IQR anomaly detection
        iqr_anomaly = self._detect_iqr_anomaly(prices, price, symbol, 'price')
        if iqr_anomaly:
            return iqr_anomaly
        
        return None
    
    def _detect_z_score_anomaly(self, data: np.ndarray, current_value: float, 
                               symbol: str, data_type: str) -> Optional[AnomalyResult]:
        """Detect anomalies using Z-score method"""
        if len(data) < 2:
            return None
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return None
        
        z_score = abs((current_value - mean) / std)
        
        if z_score > self.config.z_score_threshold:
            # Check alert cooldown
            if not self._check_alert_cooldown(symbol, f"{data_type}_z_score"):
                return None
            
            severity = self._calculate_severity(z_score, self.config.z_score_threshold)
            confidence = min(z_score / (self.config.z_score_threshold * 2), 1.0)
            
            anomaly = AnomalyResult(
                timestamp=datetime.now(),
                symbol=symbol,
                anomaly_type=f"{data_type}_z_score",
                severity=severity,
                confidence=confidence,
                value=current_value,
                threshold=mean + (self.config.z_score_threshold * std),
                description=f"Z-score anomaly: {z_score:.2f} std deviations from mean",
                metadata={
                    'z_score': z_score,
                    'mean': mean,
                    'std': std,
                    'method': 'z_score'
                }
            )
            
            self._record_anomaly(symbol, anomaly)
            self.stats['z_score_anomalies'] += 1
            return anomaly
        
        return None
    
    def _detect_iqr_anomaly(self, data: np.ndarray, current_value: float, 
                           symbol: str, data_type: str) -> Optional[AnomalyResult]:
        """Detect anomalies using IQR method"""
        if len(data) < 4:
            return None
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return None
        
        lower_bound = q1 - (self.config.iqr_multiplier * iqr)
        upper_bound = q3 + (self.config.iqr_multiplier * iqr)
        
        if current_value < lower_bound or current_value > upper_bound:
            # Check alert cooldown
            if not self._check_alert_cooldown(symbol, f"{data_type}_iqr"):
                return None
            
            # Calculate how far outside the bounds
            if current_value < lower_bound:
                distance = (lower_bound - current_value) / iqr
                threshold = lower_bound
            else:
                distance = (current_value - upper_bound) / iqr
                threshold = upper_bound
            
            severity = self._calculate_severity(distance, self.config.iqr_multiplier)
            confidence = min(distance / (self.config.iqr_multiplier * 2), 1.0)
            
            anomaly = AnomalyResult(
                timestamp=datetime.now(),
                symbol=symbol,
                anomaly_type=f"{data_type}_iqr",
                severity=severity,
                confidence=confidence,
                value=current_value,
                threshold=threshold,
                description=f"IQR anomaly: {distance:.2f} IQRs outside bounds",
                metadata={
                    'iqr': iqr,
                    'q1': q1,
                    'q3': q3,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'distance': distance,
                    'method': 'iqr'
                }
            )
            
            self._record_anomaly(symbol, anomaly)
            self.stats['iqr_anomalies'] += 1
            return anomaly
        
        return None
    
    def _calculate_severity(self, distance: float, base_threshold: float) -> str:
        """Calculate anomaly severity based on distance from threshold"""
        ratio = distance / base_threshold
        
        if ratio >= 3.0:
            return 'critical'
        elif ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    async def detect_volume_anomaly(self, symbol: str, volume: float, 
                                  avg_volume: Optional[float] = None,
                                  timestamp: Optional[datetime] = None) -> Optional[AnomalyResult]:
        """
        Detect volume anomalies using relative volume analysis
        
        Args:
            symbol: Trading symbol
            volume: Current volume
            avg_volume: Average volume (if not provided, calculated from buffer)
            timestamp: Timestamp of the volume (defaults to now)
        
        Returns:
            AnomalyResult if anomaly detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.stats['total_checks'] += 1
        
        # Ensure we have enough data
        self._ensure_symbol_buffers(symbol)
        if len(self.data_buffers[symbol].get('volume', [])) < self.config.min_data_points:
            # Add to buffer and continue
            self.data_buffers[symbol]['volume'].append(volume)
            return None
        
        volume_buffer = self.data_buffers[symbol]['volume']
        volume_buffer.append(volume)
        
        # Calculate average volume if not provided
        if avg_volume is None:
            avg_volume = np.mean(list(volume_buffer))
        
        # Check for volume spike
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > self.config.volume_threshold:
            # Check alert cooldown
            if not self._check_alert_cooldown(symbol, 'volume_spike'):
                return None
            
            severity = self._calculate_severity(volume_ratio, self.config.volume_threshold)
            confidence = min((volume_ratio - 1) / (self.config.volume_threshold - 1), 1.0)
            
            anomaly = AnomalyResult(
                timestamp=timestamp,
                symbol=symbol,
                anomaly_type='volume_spike',
                severity=severity,
                confidence=confidence,
                value=volume,
                threshold=avg_volume * self.config.volume_threshold,
                description=f"Volume spike: {volume_ratio:.2f}x average volume",
                metadata={
                    'volume_ratio': volume_ratio,
                    'average_volume': avg_volume,
                    'method': 'volume_ratio'
                }
            )
            
            self._record_anomaly(symbol, anomaly)
            self.stats['volume_anomalies'] += 1
            return anomaly
        
        return None
    
    async def detect_order_book_anbalance(self, symbol: str, bid_volume: float, ask_volume: float,
                                        timestamp: Optional[datetime] = None) -> Optional[AnomalyResult]:
        """
        Detect order book imbalance anomalies
        
        Args:
            symbol: Trading symbol
            bid_volume: Total bid volume
            ask_volume: Total ask volume
            timestamp: Timestamp of the data (defaults to now)
        
        Returns:
            AnomalyResult if anomaly detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.stats['total_checks'] += 1
        
        if bid_volume == 0 and ask_volume == 0:
            return None
        
        total_volume = bid_volume + ask_volume
        imbalance = abs(bid_volume - ask_volume) / total_volume
        
        if imbalance > self.config.order_book_imbalance_threshold:
            # Check alert cooldown
            if not self._check_alert_cooldown(symbol, 'order_book_imbalance'):
                return None
            
            severity = self._calculate_severity(imbalance, self.config.order_book_imbalance_threshold)
            confidence = min(imbalance, 1.0)
            
            # Determine imbalance direction
            direction = 'bid_heavy' if bid_volume > ask_volume else 'ask_heavy'
            
            anomaly = AnomalyResult(
                timestamp=timestamp,
                symbol=symbol,
                anomaly_type='order_book_imbalance',
                severity=severity,
                confidence=confidence,
                value=imbalance,
                threshold=self.config.order_book_imbalance_threshold,
                description=f"Order book imbalance: {imbalance:.2f} ({direction})",
                metadata={
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'total_volume': total_volume,
                    'imbalance': imbalance,
                    'direction': direction,
                    'method': 'order_book_imbalance'
                }
            )
            
            self._record_anomaly(symbol, anomaly)
            self.stats['order_book_anomalies'] += 1
            return anomaly
        
        return None
    
    def _record_anomaly(self, symbol: str, anomaly: AnomalyResult):
        """Record an anomaly in the history"""
        if symbol not in self.anomaly_history:
            self.anomaly_history[symbol] = deque(maxlen=100)
        
        self.anomaly_history[symbol].append(anomaly)
        self.stats['anomalies_detected'] += 1
        self.stats['last_anomaly'] = anomaly.timestamp
        
        # Trigger callbacks
        self._trigger_callbacks(anomaly)
        
        logger.info(f"🚨 Anomaly detected: {symbol} - {anomaly.anomaly_type} ({anomaly.severity})")
    
    async def get_anomaly_history(self, symbol: str, 
                                anomaly_type: Optional[str] = None,
                                limit: int = 50) -> List[AnomalyResult]:
        """Get anomaly history for a symbol"""
        if symbol not in self.anomaly_history:
            return []
        
        anomalies = list(self.anomaly_history[symbol])
        
        if anomaly_type:
            anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type]
        
        return anomalies[-limit:]
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'anomalies_detected': self.stats['anomalies_detected'],
            'total_checks': self.stats['total_checks'],
            'z_score_anomalies': self.stats['z_score_anomalies'],
            'iqr_anomalies': self.stats['iqr_anomalies'],
            'volume_anomalies': self.stats['volume_anomalies'],
            'order_book_anomalies': self.stats['order_book_anomalies'],
            'last_anomaly': self.stats['last_anomaly'],
            'active_symbols': len(self.anomaly_history),
            'total_callbacks': len(self.callbacks)
        }
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Closing Statistical Filters service...")
        
        # Clear buffers and history
        self.anomaly_history.clear()
        self.data_buffers.clear()
        self.callbacks.clear()
        
        logger.info("✅ Statistical Filters service closed")


# Example usage and testing
async def main():
    """Example usage of the Statistical Filters service"""
    # Create service with custom config
    config = FilterConfig(
        z_score_threshold=2.5,
        iqr_multiplier=1.8,
        volume_threshold=2.5
    )
    
    filters = StatisticalFilters(config)
    await filters.initialize()
    
    # Add a simple callback
    def anomaly_callback(anomaly: AnomalyResult):
        print(f"🚨 ALERT: {anomaly.symbol} - {anomaly.description}")
    
    filters.add_callback(anomaly_callback)
    
    # Test price anomaly detection
    print("🧪 Testing price anomaly detection...")
    
    # Simulate normal prices
    for i in range(50):
        await filters.detect_price_anomaly("BTCUSDT", 50000 + (i * 10))
    
    # Test anomaly
    anomaly = await filters.detect_price_anomaly("BTCUSDT", 60000)
    if anomaly:
        print(f"✅ Anomaly detected: {anomaly.description}")
    
    # Test volume anomaly
    print("\n🧪 Testing volume anomaly detection...")
    anomaly = await filters.detect_volume_anomaly("BTCUSDT", 1000000, 100000)
    if anomaly:
        print(f"✅ Volume anomaly detected: {anomaly.description}")
    
    # Test order book imbalance
    print("\n🧪 Testing order book imbalance detection...")
    anomaly = await filters.detect_order_book_anbalance("BTCUSDT", 800000, 200000)
    if anomaly:
        print(f"✅ Order book anomaly detected: {anomaly.description}")
    
    # Print statistics
    print(f"\n📊 Service Statistics:")
    stats = filters.get_service_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    await filters.close()


if __name__ == "__main__":
    asyncio.run(main())
