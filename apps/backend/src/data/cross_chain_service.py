#!/usr/bin/env python3
"""
Week 7.2 Phase 1: Real-Time Cross-Chain Correlation Engine

This service provides comprehensive cross-chain analysis:
- Multi-chain data aggregation (Ethereum, BSC, Polygon, Solana, etc.)
- Real-time correlation matrices and analysis
- Cross-chain market regime detection
- Multi-chain data streams and monitoring
- Correlation alert system for market shifts

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
import aiohttp
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChainType(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    SOLANA = "solana"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"


@dataclass
class ChainData:
    """Data structure for chain-specific information"""
    chain_id: str
    chain_name: str
    chain_type: ChainType
    native_token: str
    block_time: float  # Average block time in seconds
    gas_price: Optional[float] = None
    total_tx_count: Optional[int] = None
    active_addresses: Optional[int] = None
    tvl: Optional[float] = None  # Total Value Locked
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CrossChainMetric:
    """Cross-chain correlation metric"""
    timestamp: datetime
    metric_name: str
    chains: List[str]
    values: Dict[str, float]
    correlation_matrix: Optional[np.ndarray] = None
    correlation_score: Optional[float] = None
    regime_type: Optional[str] = None  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float = 0.0


@dataclass
class CorrelationAlert:
    """Alert for significant correlation changes"""
    timestamp: datetime
    alert_type: str  # 'correlation_shift', 'regime_change', 'anomaly'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    chains_involved: List[str]
    correlation_data: Dict[str, Any]
    action_required: bool = False


@dataclass
class CrossChainConfig:
    """Configuration for cross-chain service"""
    supported_chains: List[ChainType] = field(default_factory=lambda: [
        ChainType.ETHEREUM, ChainType.BSC, ChainType.POLYGON, 
        ChainType.SOLANA, ChainType.ARBITRUM
    ])
    update_interval: float = 30.0  # seconds
    correlation_window: int = 100  # data points for correlation
    correlation_threshold: float = 0.7  # minimum correlation for alerts
    regime_change_threshold: float = 0.3  # threshold for regime detection
    enable_alerts: bool = True
    max_correlation_pairs: int = 50  # maximum pairs to track
    data_retention_hours: int = 24


class ChainDataProvider:
    """Base class for chain-specific data providers"""
    
    def __init__(self, chain_type: ChainType, config: CrossChainConfig):
        self.chain_type = chain_type
        self.config = config
        self.last_update = None
        self.data_buffer: deque = deque(maxlen=1000)
        
    async def fetch_chain_data(self) -> Optional[ChainData]:
        """Fetch data for this specific chain"""
        raise NotImplementedError("Subclasses must implement fetch_chain_data")
    
    async def get_historical_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for correlation analysis"""
        return list(self.data_buffer)[-hours:]
    
    def update_data_buffer(self, data: Dict[str, Any]):
        """Update the data buffer with new information"""
        self.data_buffer.append({
            'timestamp': datetime.now(),
            'data': data
        })


class EthereumDataProvider(ChainDataProvider):
    """Ethereum-specific data provider"""
    
    def __init__(self, config: CrossChainConfig):
        super().__init__(ChainType.ETHEREUM, config)
        self.base_url = "https://api.etherscan.io/api"
        self.api_key = "demo"  # Use demo for testing
    
    async def fetch_chain_data(self) -> Optional[ChainData]:
        """Fetch Ethereum chain data"""
        try:
            # Simulate Ethereum data fetching
            # In production, this would use real API calls
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Mock data for demonstration
            data = {
                'gas_price': 25.5,  # Gwei
                'total_tx_count': 1500000,
                'active_addresses': 450000,
                'tvl': 45000000000,  # $45B
                'block_time': 12.0
            }
            
            self.update_data_buffer(data)
            
            return ChainData(
                chain_id="1",
                chain_name="Ethereum",
                chain_type=ChainType.ETHEREUM,
                native_token="ETH",
                block_time=12.0,
                gas_price=data['gas_price'],
                total_tx_count=data['total_tx_count'],
                active_addresses=data['active_addresses'],
                tvl=data['tvl']
            )
            
        except Exception as e:
            logger.error(f"Error fetching Ethereum data: {e}")
            return None


class BSCDataProvider(ChainDataProvider):
    """BSC-specific data provider"""
    
    def __init__(self, config: CrossChainConfig):
        super().__init__(ChainType.BSC, config)
        self.base_url = "https://api.bscscan.com/api"
        self.api_key = "demo"
    
    async def fetch_chain_data(self) -> Optional[ChainData]:
        """Fetch BSC chain data"""
        try:
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Mock BSC data
            data = {
                'gas_price': 5.2,  # Gwei
                'total_tx_count': 800000,
                'active_addresses': 280000,
                'tvl': 8500000000,  # $8.5B
                'block_time': 3.0
            }
            
            self.update_data_buffer(data)
            
            return ChainData(
                chain_id="56",
                chain_name="Binance Smart Chain",
                chain_type=ChainType.BSC,
                native_token="BNB",
                block_time=3.0,
                gas_price=data['gas_price'],
                total_tx_count=data['total_tx_count'],
                active_addresses=data['active_addresses'],
                tvl=data['tvl']
            )
            
        except Exception as e:
            logger.error(f"Error fetching BSC data: {e}")
            return None


class PolygonDataProvider(ChainDataProvider):
    """Polygon-specific data provider"""
    
    def __init__(self, config: CrossChainConfig):
        super().__init__(ChainType.POLYGON, config)
        self.base_url = "https://api.polygonscan.com/api"
        self.api_key = "demo"
    
    async def fetch_chain_data(self) -> Optional[ChainData]:
        """Fetch Polygon chain data"""
        try:
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Mock Polygon data
            data = {
                'gas_price': 30.0,  # Gwei
                'total_tx_count': 1200000,
                'active_addresses': 320000,
                'tvl': 12000000000,  # $12B
                'block_time': 2.0
            }
            
            self.update_data_buffer(data)
            
            return ChainData(
                chain_id="137",
                chain_name="Polygon",
                chain_type=ChainType.POLYGON,
                native_token="MATIC",
                block_time=2.0,
                gas_price=data['gas_price'],
                total_tx_count=data['total_tx_count'],
                active_addresses=data['active_addresses'],
                tvl=data['tvl']
            )
            
        except Exception as e:
            logger.error(f"Error fetching Polygon data: {e}")
            return None


class CrossChainService:
    """
    Main cross-chain correlation service
    
    This service aggregates data from multiple blockchain networks,
    calculates real-time correlations, and provides market regime analysis.
    """
    
    def __init__(self, config: Optional[CrossChainConfig] = None):
        """Initialize the cross-chain service"""
        self.config = config or CrossChainConfig()
        
        # Initialize chain data providers
        self.chain_providers: Dict[ChainType, ChainDataProvider] = {}
        self._initialize_providers()
        
        # Data storage and correlation analysis
        self.chain_data_history: Dict[ChainType, deque] = {}
        self.correlation_history: deque = deque(maxlen=1000)
        self.regime_history: deque = deque(maxlen=100)
        
        # Alert system
        self.alerts: deque = deque(maxlen=1000)
        self.callbacks: List[Callable] = []
        
        # Service statistics
        self.stats = {
            'total_updates': 0,
            'total_correlations': 0,
            'total_alerts': 0,
            'last_update': None,
            'active_chains': 0,
            'correlation_pairs': 0
        }
        
        # Performance monitoring
        self.update_times: deque = deque(maxlen=100)
        self.correlation_times: deque = deque(maxlen=100)
        
        logger.info("✅ Cross-Chain Service initialized")
    
    def _initialize_providers(self):
        """Initialize data providers for supported chains"""
        for chain_type in self.config.supported_chains:
            if chain_type == ChainType.ETHEREUM:
                self.chain_providers[chain_type] = EthereumDataProvider(self.config)
            elif chain_type == ChainType.BSC:
                self.chain_providers[chain_type] = BSCDataProvider(self.config)
            elif chain_type == ChainType.POLYGON:
                self.chain_providers[chain_type] = PolygonDataProvider(self.config)
            else:
                # Add more providers as needed
                logger.warning(f"Provider not implemented for {chain_type.value}")
        
        logger.info(f"✅ Initialized {len(self.chain_providers)} chain providers")
    
    async def initialize(self):
        """Initialize the service and start data collection"""
        logger.info("🚀 Initializing Cross-Chain Service...")
        
        # Start data collection loop
        asyncio.create_task(self._data_collection_loop())
        
        # Start correlation analysis loop
        asyncio.create_task(self._correlation_analysis_loop())
        
        logger.info("✅ Cross-Chain Service ready")
    
    async def _data_collection_loop(self):
        """Continuous data collection from all chains"""
        while True:
            try:
                start_time = time.time()
                
                # Fetch data from all chains
                await self._update_all_chain_data()
                
                # Record update time
                update_time = time.time() - start_time
                self.update_times.append(update_time)
                
                # Update statistics
                self.stats['total_updates'] += 1
                self.stats['last_update'] = datetime.now()
                self.stats['active_chains'] = len(self.chain_providers)
                
                logger.info(f"✅ Updated {len(self.chain_providers)} chains in {update_time:.2f}s")
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(self.config.update_interval)
    
    async def _correlation_analysis_loop(self):
        """Continuous correlation analysis"""
        while True:
            try:
                start_time = time.time()
                
                # Perform correlation analysis
                await self._analyze_cross_chain_correlations()
                
                # Record correlation time
                correlation_time = time.time() - start_time
                self.correlation_times.append(correlation_time)
                
                # Update statistics
                self.stats['total_correlations'] += 1
                
                logger.info(f"✅ Correlation analysis completed in {correlation_time:.2f}s")
                
                # Wait for next analysis
                await asyncio.sleep(self.config.update_interval * 2)
                
            except Exception as e:
                logger.error(f"Error in correlation analysis loop: {e}")
                await asyncio.sleep(self.config.update_interval * 2)
    
    async def _update_all_chain_data(self):
        """Update data from all supported chains"""
        tasks = []
        
        for chain_type, provider in self.chain_providers.items():
            task = asyncio.create_task(self._update_chain_data(chain_type, provider))
            tasks.append(task)
        
        # Wait for all updates to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and store data
        for i, (chain_type, provider) in enumerate(self.chain_providers.items()):
            if isinstance(results[i], Exception):
                logger.error(f"Error updating {chain_type.value}: {results[i]}")
            else:
                chain_data = results[i]
                if chain_data:
                    self._store_chain_data(chain_type, chain_data)
    
    async def _update_chain_data(self, chain_type: ChainType, provider: ChainDataProvider) -> Optional[ChainData]:
        """Update data for a specific chain"""
        try:
            return await provider.fetch_chain_data()
        except Exception as e:
            logger.error(f"Error updating {chain_type.value}: {e}")
            return None
    
    def _store_chain_data(self, chain_type: ChainType, chain_data: ChainData):
        """Store chain data in history"""
        if chain_type not in self.chain_data_history:
            self.chain_data_history[chain_type] = deque(maxlen=1000)
        
        self.chain_data_history[chain_type].append(chain_data)
    
    async def _analyze_cross_chain_correlations(self):
        """Analyze correlations between different chains"""
        if len(self.chain_data_history) < 2:
            return  # Need at least 2 chains for correlation
        
        try:
            # Get recent data from all chains
            chain_data = {}
            for chain_type, history in self.chain_data_history.items():
                if len(history) >= self.config.correlation_window:
                    chain_data[chain_type] = list(history)[-self.config.correlation_window:]
            
            if len(chain_data) < 2:
                return  # Not enough data for correlation
            
            # Calculate correlations for different metrics
            metrics = ['gas_price', 'total_tx_count', 'active_addresses', 'tvl']
            
            for metric in metrics:
                correlation_result = await self._calculate_metric_correlation(chain_data, metric)
                if correlation_result:
                    self.correlation_history.append(correlation_result)
                    
                    # Check for regime changes
                    regime_change = self._detect_regime_change(correlation_result)
                    if regime_change:
                        self.regime_history.append(regime_change)
                        
                        # Generate alert if significant
                        if self.config.enable_alerts:
                            await self._generate_correlation_alert(correlation_result, regime_change)
        
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
    
    async def _calculate_metric_correlation(self, chain_data: Dict[ChainType, List[ChainData]], 
                                          metric: str) -> Optional[CrossChainMetric]:
        """Calculate correlation for a specific metric across chains"""
        try:
            # Extract metric values for each chain
            metric_data = {}
            timestamps = []
            
            for chain_type, history in chain_data.items():
                values = []
                for data_point in history:
                    if hasattr(data_point, metric) and getattr(data_point, metric) is not None:
                        values.append(getattr(data_point, metric))
                        if not timestamps:
                            timestamps.append(data_point.last_updated)
                
                if values:
                    metric_data[chain_type.value] = values
            
            if len(metric_data) < 2:
                return None  # Need at least 2 chains
            
            # Ensure all chains have the same number of data points
            min_length = min(len(values) for values in metric_data.values())
            if min_length < 10:  # Need at least 10 data points
                return None
            
            # Truncate all arrays to the same length
            for chain in metric_data:
                metric_data[chain] = metric_data[chain][-min_length:]
            
            # Convert to numpy arrays for correlation calculation
            arrays = []
            chain_names = []
            for chain, values in metric_data.items():
                arrays.append(np.array(values[-min_length:]))
                chain_names.append(chain)
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(arrays)
            
            # Calculate average correlation
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_triangle = correlation_matrix[np.triu_indices(len(arrays), k=1)]
            avg_correlation = np.mean(upper_triangle)
            
            # Determine regime type
            regime_type = self._classify_regime(avg_correlation)
            
            # Calculate confidence based on data quality
            confidence = min(min_length / 100.0, 1.0)  # Higher confidence with more data
            
            return CrossChainMetric(
                timestamp=datetime.now(),
                metric_name=metric,
                chains=chain_names,
                values=metric_data,
                correlation_matrix=correlation_matrix,
                correlation_score=avg_correlation,
                regime_type=regime_type,
                confidence=confidence
            )
        
        except Exception as e:
            logger.error(f"Error calculating correlation for {metric}: {e}")
            return None
    
    def _classify_regime(self, correlation: float) -> str:
        """Classify market regime based on correlation"""
        if correlation > 0.7:
            return 'bull'  # High positive correlation
        elif correlation < -0.3:
            return 'bear'  # High negative correlation
        elif abs(correlation) < 0.3:
            return 'sideways'  # Low correlation
        else:
            return 'volatile'  # Moderate correlation
    
    def _detect_regime_change(self, correlation_result: CrossChainMetric) -> Optional[Dict[str, Any]]:
        """Detect significant regime changes"""
        if not self.regime_history:
            return {
                'timestamp': correlation_result.timestamp,
                'metric': correlation_result.metric_name,
                'old_regime': None,
                'new_regime': correlation_result.regime_type,
                'correlation_change': 0.0,
                'significance': 'high'
            }
        
        # Get the last regime for this metric
        last_regime = None
        for regime in reversed(self.regime_history):
            if regime.get('metric') == correlation_result.metric_name:
                last_regime = regime
                break
        
        if last_regime and last_regime['new_regime'] != correlation_result.regime_type:
            # Regime change detected
            correlation_change = abs(correlation_result.correlation_score or 0.0)
            
            return {
                'timestamp': correlation_result.timestamp,
                'metric': correlation_result.metric_name,
                'old_regime': last_regime['new_regime'],
                'new_regime': correlation_result.regime_type,
                'correlation_change': correlation_change,
                'significance': 'high' if correlation_change > self.config.regime_change_threshold else 'medium'
            }
        
        return None
    
    async def _generate_correlation_alert(self, correlation_result: CrossChainMetric, 
                                        regime_change: Dict[str, Any]):
        """Generate correlation alert"""
        if not self.config.enable_alerts:
            return
        
        # Determine alert type and severity
        if regime_change:
            alert_type = 'regime_change'
            severity = 'high' if regime_change['significance'] == 'high' else 'medium'
            message = f"Regime change detected: {regime_change['old_regime']} → {regime_change['new_regime']} for {correlation_result.metric_name}"
        else:
            alert_type = 'correlation_shift'
            severity = 'medium'
            message = f"Significant correlation shift: {correlation_result.correlation_score:.3f} for {correlation_result.metric_name}"
        
        # Create alert
        alert = CorrelationAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            chains_involved=correlation_result.chains,
            correlation_data={
                'metric': correlation_result.metric_name,
                'correlation_score': correlation_result.correlation_score,
                'regime_type': correlation_result.regime_type,
                'confidence': correlation_result.confidence
            },
            action_required=severity in ['high', 'critical']
        )
        
        # Store alert
        self.alerts.append(alert)
        self.stats['total_alerts'] += 1
        
        # Trigger callbacks
        self._trigger_callbacks(alert)
        
        logger.info(f"🚨 Correlation alert: {message}")
    
    def add_callback(self, callback: Callable):
        """Add a callback for correlation events"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.info(f"✅ Added correlation callback: {callback.__name__}")
    
    def _trigger_callbacks(self, alert: CorrelationAlert):
        """Trigger all registered callbacks with alert data"""
        for callback in self.callbacks:
            try:
                asyncio.create_task(callback(alert))
            except Exception as e:
                logger.error(f"❌ Error in correlation callback {callback.__name__}: {e}")
    
    def get_chain_data(self, chain_type: ChainType, limit: int = 100) -> List[ChainData]:
        """Get recent data for a specific chain"""
        if chain_type not in self.chain_data_history:
            return []
        
        return list(self.chain_data_history[chain_type])[-limit:]
    
    def get_correlation_history(self, limit: int = 100) -> List[CrossChainMetric]:
        """Get recent correlation analysis results"""
        return list(self.correlation_history)[-limit:]
    
    def get_regime_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent regime change history"""
        return list(self.regime_history)[-limit:]
    
    def get_recent_alerts(self, limit: int = 100) -> List[CorrelationAlert]:
        """Get recent correlation alerts"""
        return list(self.alerts)[-limit:]
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            'total_updates': self.stats['total_updates'],
            'total_correlations': self.stats['total_correlations'],
            'total_alerts': self.stats['total_alerts'],
            'last_update': self.stats['last_update'],
            'active_chains': self.stats['active_chains'],
            'correlation_pairs': self.stats['correlation_pairs'],
            'average_update_time': np.mean(self.update_times) if self.update_times else 0.0,
            'average_correlation_time': np.mean(self.correlation_times) if self.correlation_times else 0.0,
            'supported_chains': [chain.value for chain in self.config.supported_chains],
            'data_retention_hours': self.config.data_retention_hours
        }
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Closing Cross-Chain Service...")
        
        # Clear data and history
        self.chain_data_history.clear()
        self.correlation_history.clear()
        self.regime_history.clear()
        self.alerts.clear()
        self.callbacks.clear()
        
        logger.info("✅ Cross-Chain Service closed")


# Example usage and testing
async def main():
    """Example usage of the Cross-Chain Service"""
    # Create service with custom config
    config = CrossChainConfig(
        supported_chains=[ChainType.ETHEREUM, ChainType.BSC, ChainType.POLYGON],
        update_interval=10.0,  # 10 seconds for testing
        correlation_window=50,
        correlation_threshold=0.7,
        regime_change_threshold=0.3,
        enable_alerts=True
    )
    
    service = CrossChainService(config)
    await service.initialize()
    
    # Add a simple callback
    def correlation_callback(alert: CorrelationAlert):
        print(f"🚨 ALERT: {alert.message}")
        print(f"  Severity: {alert.severity}")
        print(f"  Chains: {', '.join(alert.chains_involved)}")
    
    service.add_callback(correlation_callback)
    
    # Let the service run for a while to collect data
    print("🧪 Testing Cross-Chain Service...")
    print("  Collecting data for 30 seconds...")
    
    await asyncio.sleep(30)
    
    # Print statistics
    print(f"\n📊 Service Statistics:")
    stats = service.get_service_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Print recent correlations
    print(f"\n📈 Recent Correlations:")
    correlations = service.get_correlation_history(limit=5)
    for i, corr in enumerate(correlations):
        print(f"  {i+1}. {corr.metric_name}: {corr.correlation_score:.3f} ({corr.regime_type})")
    
    # Print recent alerts
    print(f"\n🚨 Recent Alerts:")
    alerts = service.get_recent_alerts(limit=5)
    for i, alert in enumerate(alerts):
        print(f"  {i+1}. {alert.alert_type}: {alert.message}")
    
    # Print regime history
    print(f"\n🔄 Regime Changes:")
    regimes = service.get_regime_history(limit=5)
    for i, regime in enumerate(regimes):
        print(f"  {i+1}. {regime['metric']}: {regime['old_regime']} → {regime['new_regime']}")
    
    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
