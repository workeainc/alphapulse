"""
Enhanced Strategy Integration Layer for AlphaPlus
Seamlessly integrates all new enhancements with existing architecture
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import time
import json

# Import existing AlphaPlus components
try:
    from ..src.strategies.strategy_manager import StrategyManager
    from ..src.ai.multi_timeframe_fusion import MultiTimeframeFusion
    from ..src.database.connection import TimescaleDBConnection
    from ..src.app.services.signal_orchestrator import SignalOrchestrator
except ImportError:
    # Fallback for testing
    StrategyManager = None
    MultiTimeframeFusion = None
    TimescaleDBConnection = None
    TradingEngine = None

# Import new enhancement components
try:
    from ..src.core.in_memory_processor import InMemoryProcessor
    from ..src.strategies.parallel_strategy_executor import ParallelStrategyExecutor
    from ..src.strategies.ensemble_strategy_manager import EnsembleStrategyManager
except ImportError:
    InMemoryProcessor = None
    ParallelStrategyExecutor = None
    EnsembleStrategyManager = None

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for enhanced strategy integration"""
    enable_in_memory_processing: bool = True
    enable_parallel_execution: bool = True
    enable_ensemble_learning: bool = True
    enable_market_microstructure: bool = True
    enable_adaptive_tuning: bool = True
    max_buffer_size: int = 1000
    max_workers: int = 4
    ensemble_retrain_interval_hours: int = 24

@dataclass
class IntegratedSignal:
    """Integrated signal combining all enhancement layers"""
    symbol: str
    timeframe: str
    strategy_name: str
    direction: str
    confidence: float
    strength: str
    market_regime: str
    ensemble_boost: float
    processing_time_ms: float
    enhancement_metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

class EnhancedStrategyIntegration:
    """
    Integration layer that combines all enhancements with existing AlphaPlus architecture
    Provides seamless compatibility and optimal performance
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.logger = logger
        
        # Existing AlphaPlus components
        self.strategy_manager = None
        self.mtf_fusion = None
        self.db_connection = None
        self.trading_engine = None
        
        # New enhancement components
        self.in_memory_processor = None
        self.parallel_executor = None
        self.ensemble_manager = None
        
        # Integration state
        self.is_initialized = False
        self.enhancement_stats = {
            'in_memory_processed': 0,
            'parallel_executed': 0,
            'ensemble_predictions': 0,
            'total_integration_time_ms': 0.0
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'enhanced_signals': 0,
            'avg_processing_time_ms': 0.0,
            'enhancement_boost': 0.0
        }
        
        logger.info("Enhanced Strategy Integration initialized")
    
    async def initialize(self, 
                        strategy_manager: StrategyManager = None,
                        mtf_fusion: MultiTimeframeFusion = None,
                        db_connection: TimescaleDBConnection = None,
                        trading_engine: TradingEngine = None):
        """Initialize the integration layer with existing components"""
        try:
            self.logger.info("🚀 Initializing Enhanced Strategy Integration...")
            
            # Store existing components
            self.strategy_manager = strategy_manager
            self.mtf_fusion = mtf_fusion
            self.db_connection = db_connection
            self.trading_engine = trading_engine
            
            # Initialize enhancement components
            await self._initialize_enhancements()
            
            # Register strategies with enhancement layers
            await self._register_existing_strategies()
            
            self.is_initialized = True
            self.logger.info("✅ Enhanced Strategy Integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced Strategy Integration: {e}")
            raise
    
    async def _initialize_enhancements(self):
        """Initialize all enhancement components"""
        # Initialize in-memory processor
        if self.config.enable_in_memory_processing and InMemoryProcessor:
            self.in_memory_processor = InMemoryProcessor(
                max_buffer_size=self.config.max_buffer_size,
                max_workers=self.config.max_workers
            )
            self.logger.info("📦 In-memory processor initialized")
        
        # Initialize parallel executor
        if self.config.enable_parallel_execution and ParallelStrategyExecutor:
            self.parallel_executor = ParallelStrategyExecutor(
                max_process_workers=self.config.max_workers,
                max_thread_workers=self.config.max_workers * 2
            )
            await self.parallel_executor.start()
            self.logger.info("⚡ Parallel executor initialized")
        
        # Initialize ensemble manager
        if self.config.enable_ensemble_learning and EnsembleStrategyManager:
            self.ensemble_manager = EnsembleStrategyManager(
                retrain_interval_hours=self.config.ensemble_retrain_interval_hours
            )
            await self.ensemble_manager.load_models()
            self.logger.info("🧠 Ensemble manager initialized")
    
    async def _register_existing_strategies(self):
        """Register existing strategies with enhancement layers"""
        if not self.strategy_manager:
            return
        
        # Get existing strategies from strategy manager
        existing_strategies = getattr(self.strategy_manager, 'strategies', {})
        
        for strategy_name, strategy_config in existing_strategies.items():
            # Register with parallel executor
            if self.parallel_executor:
                self.parallel_executor.register_strategy(
                    strategy_name,
                    self._create_strategy_wrapper(strategy_name),
                    {'type': 'existing_strategy'}
                )
            
            # Register with ensemble manager
            if self.ensemble_manager:
                self.ensemble_manager.register_strategy(
                    strategy_name,
                    {'type': 'existing_strategy', 'config': strategy_config}
                )
        
        self.logger.info(f"📝 Registered {len(existing_strategies)} existing strategies")
    
    def _create_strategy_wrapper(self, strategy_name: str):
        """Create a wrapper function for existing strategies"""
        def strategy_wrapper(data: pd.DataFrame, symbol: str, timeframe: str, parameters: Dict[str, Any]):
            # This would call the actual strategy from strategy_manager
            # For now, return a placeholder
            return [{'strategy': strategy_name, 'confidence': 0.7, 'direction': 'neutral'}]
        
        return strategy_wrapper
    
    async def process_market_data_enhanced(self, 
                                         symbol: str,
                                         timeframe: str,
                                         market_data: Dict[str, Any]) -> Optional[IntegratedSignal]:
        """
        Process market data through all enhancement layers
        Returns integrated signal with all enhancements applied
        """
        if not self.is_initialized:
            self.logger.warning("Integration layer not initialized")
            return None
        
        start_time = time.time()
        
        try:
            # Step 1: In-memory processing (if enabled)
            in_memory_result = None
            if self.in_memory_processor:
                in_memory_result = await self.in_memory_processor.process_candle_in_memory(
                    symbol, timeframe, market_data
                )
                self.enhancement_stats['in_memory_processed'] += 1
            
            # Step 2: Get ensemble prediction (if enabled)
            ensemble_prediction = None
            if self.ensemble_manager:
                ensemble_prediction = await self.ensemble_manager.predict_best_strategy(
                    symbol, timeframe, market_data
                )
                if ensemble_prediction:
                    self.enhancement_stats['ensemble_predictions'] += 1
            
            # Step 3: Execute strategies in parallel (if enabled)
            parallel_results = []
            if self.parallel_executor and ensemble_prediction:
                # Execute the predicted best strategy with high priority
                result = await self.parallel_executor.execute_strategy(
                    ensemble_prediction.strategy_name,
                    symbol,
                    timeframe,
                    self._prepare_data_for_strategy(market_data),
                    priority=3
                )
                parallel_results.append(result)
                self.enhancement_stats['parallel_executed'] += 1
            
            # Step 4: Integrate results
            integrated_signal = await self._integrate_results(
                symbol, timeframe, in_memory_result, ensemble_prediction, parallel_results
            )
            
            # Step 5: Apply multi-timeframe fusion (if available)
            if self.mtf_fusion and integrated_signal:
                integrated_signal = await self._apply_mtf_fusion(integrated_signal, market_data)
            
            # Update performance statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time, integrated_signal is not None)
            
            return integrated_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced market data processing: {e}")
            return None
    
    async def _integrate_results(self, 
                               symbol: str,
                               timeframe: str,
                               in_memory_result: Optional[Dict[str, Any]],
                               ensemble_prediction: Optional[Any],
                               parallel_results: List[Any]) -> Optional[IntegratedSignal]:
        """Integrate results from all enhancement layers"""
        try:
            # Determine strategy and direction
            strategy_name = "default_strategy"
            direction = "neutral"
            confidence = 0.5
            strength = "weak"
            market_regime = "unknown"
            ensemble_boost = 0.0
            
            # Use ensemble prediction if available
            if ensemble_prediction:
                strategy_name = ensemble_prediction.strategy_name
                confidence = ensemble_prediction.confidence
                market_regime = ensemble_prediction.market_regime
                ensemble_boost = ensemble_prediction.predicted_performance
            
            # Use in-memory signals if available
            if in_memory_result and in_memory_result.get('signals'):
                signals = in_memory_result['signals']
                if signals:
                    # Get the strongest signal
                    strongest_signal = max(signals.values(), key=lambda x: x.get('confidence', 0))
                    direction = strongest_signal.get('direction', 'neutral')
                    confidence = max(confidence, strongest_signal.get('confidence', 0))
            
            # Determine signal strength
            if confidence >= 0.8:
                strength = "very_strong"
            elif confidence >= 0.6:
                strength = "strong"
            elif confidence >= 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            
            # Create integrated signal
            integrated_signal = IntegratedSignal(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                direction=direction,
                confidence=confidence,
                strength=strength,
                market_regime=market_regime,
                ensemble_boost=ensemble_boost,
                processing_time_ms=0.0,  # Will be set by caller
                enhancement_metadata={
                    'in_memory_processed': in_memory_result is not None,
                    'ensemble_prediction': ensemble_prediction is not None,
                    'parallel_executed': len(parallel_results) > 0,
                    'enhancement_layers': self._get_active_enhancements()
                }
            )
            
            return integrated_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error integrating results: {e}")
            return None
    
    async def _apply_mtf_fusion(self, integrated_signal: IntegratedSignal, market_data: Dict[str, Any]) -> IntegratedSignal:
        """Apply multi-timeframe fusion to integrated signal"""
        try:
            if not self.mtf_fusion:
                return integrated_signal
            
            # Create timeframe signal for MTF fusion
            from ..src.ai.multi_timeframe_fusion import TimeframeSignal, SignalDirection, SignalStrength
            
            tf_signal = TimeframeSignal(
                timeframe=integrated_signal.timeframe,
                direction=SignalDirection.BULLISH if integrated_signal.direction == 'buy' else SignalDirection.BEARISH,
                strength=getattr(SignalStrength, integrated_signal.strength.upper(), SignalStrength.MODERATE),
                confidence=integrated_signal.confidence,
                indicators=market_data.get('indicators', {}),
                patterns=market_data.get('patterns', [])
            )
            
            # Apply MTF fusion (simplified - in practice, you'd need signals from multiple timeframes)
            # For now, just boost confidence based on ensemble prediction
            if integrated_signal.ensemble_boost > 0:
                integrated_signal.confidence = min(1.0, integrated_signal.confidence + integrated_signal.ensemble_boost * 0.1)
            
            return integrated_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error applying MTF fusion: {e}")
            return integrated_signal
    
    def _prepare_data_for_strategy(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare market data for strategy execution"""
        try:
            # Convert market data to DataFrame format expected by strategies
            # This is a simplified conversion - adjust based on your actual data structure
            data = {
                'timestamp': [market_data.get('timestamp', datetime.now(timezone.utc))],
                'open': [market_data.get('open', 0.0)],
                'high': [market_data.get('high', 0.0)],
                'low': [market_data.get('low', 0.0)],
                'close': [market_data.get('close', 0.0)],
                'volume': [market_data.get('volume', 0.0)]
            }
            
            # Add indicators if available
            indicators = market_data.get('indicators', {})
            for key, value in indicators.items():
                data[key] = [value]
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing data for strategy: {e}")
            return pd.DataFrame()
    
    def _get_active_enhancements(self) -> List[str]:
        """Get list of active enhancement layers"""
        active_enhancements = []
        
        if self.in_memory_processor:
            active_enhancements.append('in_memory_processing')
        
        if self.parallel_executor:
            active_enhancements.append('parallel_execution')
        
        if self.ensemble_manager:
            active_enhancements.append('ensemble_learning')
        
        return active_enhancements
    
    def _update_performance_stats(self, processing_time_ms: float, signal_generated: bool):
        """Update performance statistics"""
        self.performance_stats['total_signals'] += 1
        if signal_generated:
            self.performance_stats['enhanced_signals'] += 1
        
        # Update average processing time
        total_signals = self.performance_stats['total_signals']
        current_avg = self.performance_stats['avg_processing_time_ms']
        self.performance_stats['avg_processing_time_ms'] = (
            (current_avg * (total_signals - 1) + processing_time_ms) / total_signals
        )
        
        # Calculate enhancement boost
        if total_signals > 0:
            self.performance_stats['enhancement_boost'] = (
                self.performance_stats['enhanced_signals'] / total_signals
            )
    
    async def record_strategy_performance(self, 
                                        strategy_name: str,
                                        symbol: str,
                                        timeframe: str,
                                        performance_metrics: Dict[str, Any]):
        """Record strategy performance for ensemble learning"""
        if self.ensemble_manager:
            from ..src.strategies.ensemble_strategy_manager import StrategyPerformance
            
            performance = StrategyPerformance(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                market_regime=performance_metrics.get('market_regime', 'unknown'),
                timestamp=datetime.now(timezone.utc),
                win_rate=performance_metrics.get('win_rate', 0.0),
                profit_factor=performance_metrics.get('profit_factor', 0.0),
                max_drawdown=performance_metrics.get('max_drawdown', 0.0),
                total_trades=performance_metrics.get('total_trades', 0),
                avg_profit=performance_metrics.get('avg_profit', 0.0),
                sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
                success=performance_metrics.get('success', True)
            )
            
            await self.ensemble_manager.record_strategy_performance(performance)
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            'is_initialized': self.is_initialized,
            'enhancement_stats': self.enhancement_stats,
            'performance_stats': self.performance_stats,
            'active_enhancements': self._get_active_enhancements(),
            'configuration': {
                'enable_in_memory_processing': self.config.enable_in_memory_processing,
                'enable_parallel_execution': self.config.enable_parallel_execution,
                'enable_ensemble_learning': self.config.enable_ensemble_learning,
                'max_buffer_size': self.config.max_buffer_size,
                'max_workers': self.config.max_workers
            }
        }
    
    async def shutdown(self):
        """Shutdown the integration layer"""
        try:
            self.logger.info("🛑 Shutting down Enhanced Strategy Integration...")
            
            # Shutdown enhancement components
            if self.in_memory_processor:
                self.in_memory_processor.shutdown()
            
            if self.parallel_executor:
                await self.parallel_executor.stop()
            
            self.is_initialized = False
            self.logger.info("✅ Enhanced Strategy Integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
