import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CandleStatus(Enum):
    COMPLETED = "completed"
    FORMING = "forming"
    INCOMPLETE = "incomplete"

@dataclass
class CandleInfo:
    """Information about a candle's status and timing"""
    status: CandleStatus
    open_time: datetime
    close_time: datetime
    current_time: datetime
    timeframe: str
    symbol: str
    is_ready_for_analysis: bool

class TimeSyncBatcher:
    """
    Time-Synchronized Batching System for MTF Analysis
    Only processes patterns on completed candles and tracks forming candles separately
    """
    
    def __init__(self):
        # Timeframe intervals in minutes
        self.timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        
        # Timeframe hierarchy (higher to lower)
        self.timeframe_hierarchy = ["1d", "4h", "1h", "15m", "5m", "1m"]
        
        # Track candle status for each symbol/timeframe
        self.candle_trackers: Dict[str, Dict[str, CandleInfo]] = {}
        
        # Callbacks for different events
        self.on_candle_complete_callbacks: List[Callable] = []
        self.on_candle_form_callbacks: List[Callable] = []
        
        # Batch processing queue
        self.processing_queue: List[Tuple[str, str, pd.DataFrame]] = []
        
        logger.info("🚀 Time-Synchronized Batcher initialized")
    
    def register_candle_complete_callback(self, callback: Callable):
        """Register callback for when a candle completes"""
        self.on_candle_complete_callbacks.append(callback)
    
    def register_candle_form_callback(self, callback: Callable):
        """Register callback for when a candle starts forming"""
        self.on_candle_form_callbacks.append(callback)
    
    def get_candle_info(self, symbol: str, timeframe: str, current_time: Optional[datetime] = None) -> CandleInfo:
        """
        Get information about the current candle for a symbol/timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "1h", "4h")
            current_time: Current time (defaults to UTC now)
            
        Returns:
            CandleInfo object with candle status and timing
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        if timeframe not in self.timeframe_minutes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Calculate candle boundaries
        minutes_since_epoch = int(current_time.timestamp() / 60)
        timeframe_minutes = self.timeframe_minutes[timeframe]
        
        # Find the start of the current candle
        candle_start_minutes = (minutes_since_epoch // timeframe_minutes) * timeframe_minutes
        candle_start = datetime.fromtimestamp(candle_start_minutes * 60)
        candle_end = candle_start + timedelta(minutes=timeframe_minutes)
        
        # Determine candle status
        if current_time >= candle_end:
            status = CandleStatus.COMPLETED
            is_ready_for_analysis = True
        elif current_time >= candle_start:
            status = CandleStatus.FORMING
            is_ready_for_analysis = False
        else:
            status = CandleStatus.INCOMPLETE
            is_ready_for_analysis = False
        
        candle_info = CandleInfo(
            status=status,
            open_time=candle_start,
            close_time=candle_end,
            current_time=current_time,
            timeframe=timeframe,
            symbol=symbol,
            is_ready_for_analysis=is_ready_for_analysis
        )
        
        # Update tracker
        if symbol not in self.candle_trackers:
            self.candle_trackers[symbol] = {}
        
        self.candle_trackers[symbol][timeframe] = candle_info
        
        return candle_info
    
    def is_candle_completed(self, symbol: str, timeframe: str, current_time: Optional[datetime] = None) -> bool:
        """Check if the current candle is completed"""
        candle_info = self.get_candle_info(symbol, timeframe, current_time)
        return candle_info.status == CandleStatus.COMPLETED
    
    def is_candle_forming(self, symbol: str, timeframe: str, current_time: Optional[datetime] = None) -> bool:
        """Check if the current candle is forming"""
        candle_info = self.get_candle_info(symbol, timeframe, current_time)
        return candle_info.status == CandleStatus.FORMING
    
    def get_next_candle_time(self, timeframe: str, current_time: Optional[datetime] = None) -> datetime:
        """Get the start time of the next candle"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        if timeframe not in self.timeframe_minutes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        minutes_since_epoch = int(current_time.timestamp() / 60)
        timeframe_minutes = self.timeframe_minutes[timeframe]
        
        # Find the start of the next candle
        next_candle_start_minutes = ((minutes_since_epoch // timeframe_minutes) + 1) * timeframe_minutes
        return datetime.fromtimestamp(next_candle_start_minutes * 60)
    
    def get_time_to_next_candle(self, timeframe: str, current_time: Optional[datetime] = None) -> timedelta:
        """Get time until the next candle starts"""
        next_candle_time = self.get_next_candle_time(timeframe, current_time)
        current = current_time or datetime.utcnow()
        return next_candle_time - current
    
    def get_time_to_candle_close(self, symbol: str, timeframe: str, current_time: Optional[datetime] = None) -> timedelta:
        """Get time until the current candle closes"""
        candle_info = self.get_candle_info(symbol, timeframe, current_time)
        current = current_time or datetime.utcnow()
        return candle_info.close_time - current
    
    async def process_completed_candles(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Process data only if the candle is completed
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data
            
        Returns:
            True if processed (candle was completed), False otherwise
        """
        candle_info = self.get_candle_info(symbol, timeframe)
        
        if candle_info.status == CandleStatus.COMPLETED:
            # Add to processing queue
            self.processing_queue.append((symbol, timeframe, data))
            
            # Trigger callbacks
            for callback in self.on_candle_complete_callbacks:
                try:
                    await callback(symbol, timeframe, data, candle_info)
                except Exception as e:
                    logger.error(f"Error in candle complete callback: {e}")
            
            logger.info(f"✅ Completed candle queued for processing: {symbol} {timeframe}")
            return True
        else:
            logger.debug(f"⏳ Candle not completed yet: {symbol} {timeframe} (status: {candle_info.status.value})")
            return False
    
    async def track_forming_candles(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Track forming candles for early signal detection
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data
            
        Returns:
            True if candle is forming, False otherwise
        """
        candle_info = self.get_candle_info(symbol, timeframe)
        
        if candle_info.status == CandleStatus.FORMING:
            # Trigger callbacks for forming candles
            for callback in self.on_candle_form_callbacks:
                try:
                    await callback(symbol, timeframe, data, candle_info)
                except Exception as e:
                    logger.error(f"Error in candle form callback: {e}")
            
            logger.debug(f"📊 Tracking forming candle: {symbol} {timeframe}")
            return True
        else:
            return False
    
    def get_processing_queue(self) -> List[Tuple[str, str, pd.DataFrame]]:
        """Get the current processing queue"""
        return self.processing_queue.copy()
    
    def clear_processing_queue(self):
        """Clear the processing queue"""
        self.processing_queue.clear()
        logger.info("🧹 Processing queue cleared")
    
    def get_higher_timeframe_completion_status(self, symbol: str, current_timeframe: str) -> Dict[str, bool]:
        """
        Get completion status of higher timeframes
        
        Args:
            symbol: Trading symbol
            current_timeframe: Current timeframe being analyzed
            
        Returns:
            Dictionary mapping higher timeframes to completion status
        """
        try:
            current_index = self.timeframe_hierarchy.index(current_timeframe)
            status = {}
            
            # Check higher timeframes
            for i in range(current_index):
                higher_timeframe = self.timeframe_hierarchy[i]
                is_completed = self.is_candle_completed(symbol, higher_timeframe)
                status[higher_timeframe] = is_completed
            
            return status
            
        except ValueError:
            logger.warning(f"Timeframe {current_timeframe} not in hierarchy")
            return {}
    
    def should_wait_for_higher_timeframes(self, symbol: str, current_timeframe: str) -> bool:
        """
        Determine if we should wait for higher timeframes to complete
        
        Args:
            symbol: Trading symbol
            current_timeframe: Current timeframe being analyzed
            
        Returns:
            True if we should wait for higher timeframes
        """
        higher_status = self.get_higher_timeframe_completion_status(symbol, current_timeframe)
        
        # Wait if any higher timeframe is still forming
        for timeframe, is_completed in higher_status.items():
            if not is_completed:
                logger.debug(f"⏳ Waiting for higher timeframe {timeframe} to complete")
                return True
        
        return False
    
    def get_optimal_processing_order(self, symbols: List[str], timeframes: List[str]) -> List[Tuple[str, str]]:
        """
        Get optimal processing order (higher timeframes first)
        
        Args:
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            
        Returns:
            List of (symbol, timeframe) tuples in optimal processing order
        """
        processing_order = []
        
        # Sort timeframes by hierarchy (higher first)
        sorted_timeframes = sorted(
            timeframes, 
            key=lambda tf: self.timeframe_hierarchy.index(tf) if tf in self.timeframe_hierarchy else 999
        )
        
        # Generate processing order
        for timeframe in sorted_timeframes:
            for symbol in symbols:
                processing_order.append((symbol, timeframe))
        
        return processing_order
    
    def get_candle_summary(self, symbol: str, timeframe: str) -> Dict:
        """Get summary of candle status and timing"""
        try:
            candle_info = self.get_candle_info(symbol, timeframe)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': candle_info.status.value,
                'open_time': candle_info.open_time.isoformat(),
                'close_time': candle_info.close_time.isoformat(),
                'current_time': candle_info.current_time.isoformat(),
                'is_ready_for_analysis': candle_info.is_ready_for_analysis,
                'time_to_close': str(self.get_time_to_candle_close(symbol, timeframe)),
                'time_to_next': str(self.get_time_to_next_candle(timeframe))
            }
        except Exception as e:
            logger.error(f"Error getting candle summary: {e}")
            return {}
    
    def get_all_candle_statuses(self) -> Dict[str, Dict[str, Dict]]:
        """Get status of all tracked candles"""
        statuses = {}
        
        for symbol, timeframes in self.candle_trackers.items():
            statuses[symbol] = {}
            for timeframe, candle_info in timeframes.items():
                statuses[symbol][timeframe] = {
                    'status': candle_info.status.value,
                    'is_ready_for_analysis': candle_info.is_ready_for_analysis,
                    'open_time': candle_info.open_time.isoformat(),
                    'close_time': candle_info.close_time.isoformat()
                }
        
        return statuses
    
    async def cleanup_old_trackers(self, max_age_hours: int = 24):
        """Clean up old candle trackers"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        for symbol in list(self.candle_trackers.keys()):
            for timeframe in list(self.candle_trackers[symbol].keys()):
                candle_info = self.candle_trackers[symbol][timeframe]
                if candle_info.current_time < cutoff_time:
                    del self.candle_trackers[symbol][timeframe]
                    cleaned_count += 1
            
            # Remove empty symbol entries
            if not self.candle_trackers[symbol]:
                del self.candle_trackers[symbol]
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} old candle trackers")
