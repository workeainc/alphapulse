"""
Batch Predictor for AlphaPulse
Processes multiple symbols/timeframes in batches for optimal throughput
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time

from .model_registry import model_registry
from .feature_engineering import FeatureExtractor
from .model_accuracy_improvement import PatternType, MarketRegime

logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Batch prediction system that processes multiple symbols/timeframes
    simultaneously for optimal throughput and reduced latency.
    """
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time  # seconds
        
        # Batch queues for different prediction types
        self.pattern_batch_queue = deque()
        self.regime_batch_queue = deque()
        self.ensemble_batch_queue = deque()
        
        # Storage for processed items (to track results)
        self.processed_items = {}
        
        # Batch processing state
        self.is_processing = False
        self.last_batch_time = datetime.now()
        
        # Performance tracking
        self.total_batches_processed = 0
        self.total_predictions = 0
        self.avg_batch_time_ms = 0.0
        self.avg_predictions_per_batch = 0.0
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        
        # Start background batch processor
        self.batch_processor_task = None
        
        logger.info(f"BatchPredictor initialized with batch_size={batch_size}, max_wait_time={max_wait_time}s")
    
    async def start(self):
        """Start the batch prediction system"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            logger.info("BatchPredictor started")
    
    async def stop(self):
        """Stop the batch prediction system"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
            self.batch_processor_task = None
            logger.info("BatchPredictor stopped")
    
    async def add_to_pattern_batch(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                                  pattern_type: PatternType) -> str:
        """
        Add a pattern prediction request to the batch queue.
        Returns a request ID for tracking the result.
        """
        request_id = f"pattern_{symbol}_{timeframe}_{pattern_type.value}_{int(time.time() * 1000)}"
        
        batch_item = {
            'request_id': request_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data,
            'pattern_type': pattern_type,
            'timestamp': datetime.now(),
            'future': asyncio.Future()
        }
        
        self.pattern_batch_queue.append(batch_item)
        logger.debug(f"Added pattern prediction to batch: {request_id}")
        
        return request_id
    
    async def add_to_regime_batch(self, symbol: str, timeframe: str, data: pd.DataFrame,
                                 regime: MarketRegime) -> str:
        """
        Add a regime prediction request to the batch queue.
        Returns a request ID for tracking the result.
        """
        request_id = f"regime_{symbol}_{timeframe}_{regime.value}_{int(time.time() * 1000)}"
        
        batch_item = {
            'request_id': request_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data,
            'regime': regime,
            'timestamp': datetime.now(),
            'future': asyncio.Future()
        }
        
        self.regime_batch_queue.append(batch_item)
        logger.debug(f"Added regime prediction to batch: {request_id}")
        
        return request_id
    
    async def add_to_ensemble_batch(self, symbol: str, timeframe: str, data: pd.DataFrame,
                                   current_regime: MarketRegime) -> str:
        """
        Add an ensemble prediction request to the batch queue.
        Returns a request ID for tracking the result.
        """
        request_id = f"ensemble_{symbol}_{timeframe}_{current_regime.value}_{int(time.time() * 1000)}"
        
        batch_item = {
            'request_id': request_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data,
            'current_regime': current_regime,
            'timestamp': datetime.now(),
            'future': asyncio.Future()
        }
        
        self.ensemble_batch_queue.append(batch_item)
        logger.debug(f"Added ensemble prediction to batch: {request_id}")
        
        return request_id
    
    async def get_prediction_result(self, request_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Get the result of a prediction request.
        Returns None if the request is not found or timed out.
        """
        # First check if the item is already processed
        if request_id in self.processed_items:
            return self.processed_items[request_id]
        
        # Find the request in any of the batch queues
        for queue in [self.pattern_batch_queue, self.regime_batch_queue, self.ensemble_batch_queue]:
            for item in queue:
                if item['request_id'] == request_id:
                    try:
                        result = await asyncio.wait_for(item['future'], timeout=timeout)
                        # Store the result for future access
                        self.processed_items[request_id] = result
                        return result
                    except asyncio.TimeoutError:
                        logger.warning(f"Prediction request {request_id} timed out")
                        return None
        
        logger.warning(f"Prediction request {request_id} not found")
        return None
    
    async def _batch_processor_loop(self):
        """Main batch processing loop that runs continuously"""
        while True:
            try:
                # Process pattern predictions
                if len(self.pattern_batch_queue) >= self.batch_size or self._should_process_batch():
                    await self._process_pattern_batch()
                
                # Process regime predictions
                if len(self.regime_batch_queue) >= self.batch_size or self._should_process_batch():
                    await self._process_regime_batch()
                
                # Process ensemble predictions
                if len(self.ensemble_batch_queue) >= self.batch_size or self._should_process_batch():
                    await self._process_ensemble_batch()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)  # 1ms
                
            except asyncio.CancelledError:
                logger.info("Batch processor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                await asyncio.sleep(0.1)  # Wait before retrying
    
    def _should_process_batch(self) -> bool:
        """Check if we should process a batch based on time elapsed"""
        # If no batches have been processed yet, process immediately
        if self.total_batches_processed == 0:
            return True
        
        time_since_last_batch = (datetime.now() - self.last_batch_time).total_seconds()
        return time_since_last_batch >= self.max_wait_time
    
    async def _process_pattern_batch(self):
        """Process a batch of pattern predictions"""
        if not self.pattern_batch_queue:
            return
        
        start_time = datetime.now()
        batch_items = []
        
        # Collect items for this batch
        while self.pattern_batch_queue and len(batch_items) < self.batch_size:
            batch_items.append(self.pattern_batch_queue.popleft())
        
        if not batch_items:
            return
        
        logger.debug(f"Processing pattern batch with {len(batch_items)} items")
        
        try:
            # Group by pattern type for efficient processing
            pattern_groups = defaultdict(list)
            for item in batch_items:
                pattern_groups[item['pattern_type']].append(item)
            
            # Process each pattern type in parallel
            tasks = []
            for pattern_type, items in pattern_groups.items():
                task = self._process_pattern_group(pattern_type, items)
                tasks.append(task)
            
            # Wait for all pattern types to complete
            await asyncio.gather(*tasks)
            
            # Update performance stats
            batch_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(len(batch_items), batch_time_ms)
            
            logger.debug(f"Pattern batch processed in {batch_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing pattern batch: {e}")
            # Set error results for all items in the batch
            for item in batch_items:
                if not item['future'].done():
                    item['future'].set_result({
                        'prediction': 0.0,
                        'confidence': 0.0,
                        'error': str(e),
                        'request_id': item['request_id']
                    })
    
    async def _process_pattern_group(self, pattern_type: PatternType, items: List[Dict]) -> None:
        """Process a group of predictions for the same pattern type"""
        try:
            # Extract features for all items in the group
            feature_data = []
            for item in items:
                features_df, _ = self.feature_extractor.extract_features(item['data'], symbol=item['symbol'])
                # Convert DataFrame to numpy array and take the last row (most recent data)
                features_array = features_df.values
                if len(features_array) > 0:
                    feature_data.append(features_array[-1])  # Take the last row
                else:
                    feature_data.append(np.zeros(features_df.shape[1]))  # Fallback to zeros
            
            # Make batch prediction using model registry
            model_name = f"pattern_model_{pattern_type.value}"
            model = model_registry.get_model(model_name)
            
            if model is None:
                # Set error results
                for item in items:
                    if not item['future'].done():
                        item['future'].set_result({
                            'prediction': 0.0,
                            'confidence': 0.0,
                            'error': f'Model {model_name} not found',
                            'request_id': item['request_id']
                        })
                return
            
            # Make predictions for all items
            # Convert to 2D numpy array for RandomForestClassifier
            if len(feature_data) > 0:
                feature_data_2d = np.array(feature_data)
                predictions = model.predict_proba(feature_data_2d)
            else:
                predictions = np.array([])
            
            # Set results for each item
            for i, item in enumerate(items):
                if not item['future'].done():
                    if len(predictions) > i:
                        prediction = predictions[i][1]  # Probability of positive class
                    else:
                        prediction = 0.0
                    result = {
                        'prediction': prediction,
                        'confidence': prediction,
                        'model_name': model_name,
                        'request_id': item['request_id'],
                        'symbol': item['symbol'],
                        'timeframe': item['timeframe'],
                        'pattern_type': pattern_type.value
                    }
                    item['future'].set_result(result)
                    # Store in processed items for future access
                    self.processed_items[item['request_id']] = result
        
        except Exception as e:
            logger.error(f"Error processing pattern group {pattern_type}: {e}")
            # Set error results
            for item in items:
                if not item['future'].done():
                    item['future'].set_result({
                        'prediction': 0.0,
                        'confidence': 0.0,
                        'error': str(e),
                        'request_id': item['request_id']
                    })
    
    async def _process_regime_batch(self):
        """Process a batch of regime predictions"""
        if not self.regime_batch_queue:
            return
        
        start_time = datetime.now()
        batch_items = []
        
        # Collect items for this batch
        while self.regime_batch_queue and len(batch_items) < self.batch_size:
            batch_items.append(self.regime_batch_queue.popleft())
        
        if not batch_items:
            return
        
        logger.debug(f"Processing regime batch with {len(batch_items)} items")
        
        try:
            # Group by regime type for efficient processing
            regime_groups = defaultdict(list)
            for item in batch_items:
                regime_groups[item['regime']].append(item)
            
            # Process each regime type in parallel
            tasks = []
            for regime, items in regime_groups.items():
                task = self._process_regime_group(regime, items)
                tasks.append(task)
            
            # Wait for all regime types to complete
            await asyncio.gather(*tasks)
            
            # Update performance stats
            batch_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(len(batch_items), batch_time_ms)
            
            logger.debug(f"Regime batch processed in {batch_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing regime batch: {e}")
            # Set error results for all items in the batch
            for item in batch_items:
                if not item['future'].done():
                    item['future'].set_result({
                        'prediction': 0.0,
                        'confidence': 0.0,
                        'error': str(e),
                        'request_id': item['request_id']
                    })
    
    async def _process_regime_group(self, regime: MarketRegime, items: List[Dict]) -> None:
        """Process a group of predictions for the same regime type"""
        try:
            # Extract features for all items in the group
            feature_data = []
            for item in items:
                features_df, _ = self.feature_extractor.extract_features(item['data'], symbol=item['symbol'])
                # Convert DataFrame to numpy array and take the last row (most recent data)
                features_array = features_df.values
                if len(features_array) > 0:
                    feature_data.append(features_array[-1])  # Take the last row
                else:
                    feature_data.append(np.zeros(features_df.shape[1]))  # Fallback to zeros
            
            # Make batch prediction using model registry
            model_name = f"regime_model_{regime.value}"
            model = model_registry.get_model(model_name)
            
            if model is None:
                # Set error results
                for item in items:
                    if not item['future'].done():
                        item['future'].set_result({
                            'prediction': 0.0,
                            'confidence': 0.0,
                            'error': f'Model {model_name} not found',
                            'request_id': item['request_id']
                        })
                return
            
            # Make predictions for all items
            # Convert to 2D numpy array for RandomForestClassifier
            if len(feature_data) > 0:
                feature_data_2d = np.array(feature_data)
                predictions = model.predict_proba(feature_data_2d)
            else:
                predictions = np.array([])
            
                                               # Set results for each item
            for i, item in enumerate(items):
                if not item['future'].done():
                    if len(predictions) > i:
                        prediction = predictions[i][1]  # Probability of positive class
                    else:
                        prediction = 0.0
                    result = {
                        'prediction': prediction,
                        'confidence': prediction,
                        'model_name': model_name,
                        'request_id': item['request_id'],
                        'symbol': item['symbol'],
                        'timeframe': item['timeframe'],
                        'regime': regime.value
                    }
                    item['future'].set_result(result)
                    # Store in processed items for future access
                    self.processed_items[item['request_id']] = result
        
        except Exception as e:
            logger.error(f"Error processing regime group {regime}: {e}")
            # Set error results
            for item in items:
                if not item['future'].done():
                    item['future'].set_result({
                        'prediction': 0.0,
                        'confidence': 0.0,
                        'error': str(e),
                        'request_id': item['request_id']
                    })
    
    async def _process_ensemble_batch(self):
        """Process a batch of ensemble predictions"""
        if not self.ensemble_batch_queue:
            return
        
        start_time = datetime.now()
        batch_items = []
        
        # Collect items for this batch
        while self.ensemble_batch_queue and len(batch_items) < self.batch_size:
            batch_items.append(self.ensemble_batch_queue.popleft())
        
        if not batch_items:
            return
        
        logger.debug(f"Processing ensemble batch with {len(batch_items)} items")
        
        try:
            # Process ensemble predictions (more complex, process individually)
            tasks = []
            for item in batch_items:
                task = self._process_single_ensemble(item)
                tasks.append(task)
            
            # Wait for all ensemble predictions to complete
            await asyncio.gather(*tasks)
            
            # Update performance stats
            batch_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(len(batch_items), batch_time_ms)
            
            logger.debug(f"Ensemble batch processed in {batch_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing ensemble batch: {e}")
            # Set error results for all items in the batch
            for item in batch_items:
                if not item['future'].done():
                    item['future'].set_result({
                        'prediction': 0.0,
                        'confidence': 0.0,
                        'error': str(e),
                        'request_id': item['request_id']
                    })
    
    async def _process_single_ensemble(self, item: Dict) -> None:
        """Process a single ensemble prediction"""
        try:
            result = await model_registry.predict_ensemble(
                item['data'], 
                item['current_regime']
            )
            
            if not item['future'].done():
                final_result = {
                    **result,
                    'request_id': item['request_id'],
                    'symbol': item['symbol'],
                    'timeframe': item['timeframe']
                }
                item['future'].set_result(final_result)
                # Store in processed items for future access
                self.processed_items[item['request_id']] = final_result
        
        except Exception as e:
            logger.error(f"Error processing ensemble prediction: {e}")
            if not item['future'].done():
                item['future'].set_result({
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'error': str(e),
                    'request_id': item['request_id']
                })
    
    def _update_performance_stats(self, batch_size: int, batch_time_ms: float):
        """Update performance statistics"""
        self.total_batches_processed += 1
        self.total_predictions += batch_size
        self.last_batch_time = datetime.now()
        
        # Update averages
        if self.total_batches_processed > 0:
            self.avg_batch_time_ms = (
                (self.avg_batch_time_ms * (self.total_batches_processed - 1) + batch_time_ms) / 
                self.total_batches_processed
            )
            self.avg_predictions_per_batch = self.total_predictions / self.total_batches_processed
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_batches_processed': self.total_batches_processed,
            'total_predictions': self.total_predictions,
            'avg_batch_time_ms': self.avg_batch_time_ms,
            'avg_predictions_per_batch': self.avg_predictions_per_batch,
            'current_queue_sizes': {
                'pattern_queue': len(self.pattern_batch_queue),
                'regime_queue': len(self.regime_batch_queue),
                'ensemble_queue': len(self.ensemble_batch_queue)
            },
            'throughput_predictions_per_second': (
                self.total_predictions / 
                max(1, (datetime.now() - self.last_batch_time).total_seconds())
            ) if self.total_predictions > 0 else 0
        }
    
    async def predict_multiple_symbols(self, symbol_data: Dict[str, pd.DataFrame], 
                                     timeframes: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Convenience method to predict for multiple symbols at once.
        Returns predictions organized by symbol and timeframe.
        """
        if timeframes is None:
            timeframes = ['1h', '4h']
        
        results = {}
        
        for symbol, data in symbol_data.items():
            results[symbol] = {}
            
            for timeframe in timeframes:
                # Add to batch and get request ID
                request_id = await self.add_to_ensemble_batch(
                    symbol, timeframe, data, MarketRegime.BULL  # Default regime
                )
                
                # Get result
                result = await self.get_prediction_result(request_id)
                results[symbol][timeframe] = result
        
        return results


# Global batch predictor instance
batch_predictor = BatchPredictor()
