import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import schedule
import time
from concurrent.futures import ThreadPoolExecutor

from .mtf_orchestrator import MTFOrchestrator
from .mtf_cache_manager import MTFCacheManager
from ..database.models import MTFContext
from ..database.database import get_db

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    symbol: str
    timeframe: str
    next_run: datetime
    interval_minutes: int
    is_active: bool

class MTFScheduler:
    """
    Scheduled processor for higher timeframe analysis
    Ensures MTF context is always up-to-date and processed at optimal times
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.mtf_orchestrator = MTFOrchestrator(redis_url)
        self.cache_manager = MTFCacheManager(redis_url)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Scheduling configuration
        self.timeframe_schedules = {
            "1d": {"interval": 1440, "offset_minutes": 0},  # Daily at 00:00
            "4h": {"interval": 240, "offset_minutes": 0},   # Every 4 hours
            "1h": {"interval": 60, "offset_minutes": 0},    # Every hour
            "15m": {"interval": 15, "offset_minutes": 0},   # Every 15 minutes
            "5m": {"interval": 5, "offset_minutes": 0},     # Every 5 minutes
        }
        
        # Active symbols and timeframes
        self.active_symbols: Set[str] = set()
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        
        # Performance tracking
        self.stats = {
            'total_scheduled_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run_times': {},
            'processing_times': []
        }
        
        logger.info("🚀 MTF Scheduler initialized")
    
    async def start(self, symbols: List[str] = None):
        """
        Start the MTF scheduler
        """
        if self.is_running:
            logger.warning("⚠️ MTF Scheduler is already running")
            return
        
        # Set active symbols
        if symbols:
            self.active_symbols = set(symbols)
        else:
            # Default symbols
            self.active_symbols = {"BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"}
        
        self.is_running = True
        
        # Initialize scheduled tasks
        await self._initialize_scheduled_tasks()
        
        # Start scheduler in background thread
        self.scheduler_thread = self.executor.submit(self._run_scheduler)
        
        logger.info(f"🚀 MTF Scheduler started with {len(self.active_symbols)} symbols")
    
    async def stop(self):
        """
        Stop the MTF scheduler
        """
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.cancel()
        
        # Cancel all pending tasks
        for task in self.scheduled_tasks.values():
            task.is_active = False
        
        logger.info("🛑 MTF Scheduler stopped")
    
    async def _initialize_scheduled_tasks(self):
        """
        Initialize scheduled tasks for all symbols and timeframes
        """
        current_time = datetime.now()
        
        for symbol in self.active_symbols:
            for timeframe, schedule_config in self.timeframe_schedules.items():
                task_key = f"{symbol}_{timeframe}"
                
                # Calculate next run time
                next_run = self._calculate_next_run_time(
                    current_time, 
                    schedule_config["interval"], 
                    schedule_config["offset_minutes"]
                )
                
                self.scheduled_tasks[task_key] = ScheduledTask(
                    symbol=symbol,
                    timeframe=timeframe,
                    next_run=next_run,
                    interval_minutes=schedule_config["interval"],
                    is_active=True
                )
        
        logger.info(f"📅 Initialized {len(self.scheduled_tasks)} scheduled tasks")
    
    def _calculate_next_run_time(
        self, 
        current_time: datetime, 
        interval_minutes: int, 
        offset_minutes: int
    ) -> datetime:
        """
        Calculate the next run time for a scheduled task
        """
        # Round down to the nearest interval
        total_minutes = current_time.hour * 60 + current_time.minute
        adjusted_minutes = (total_minutes // interval_minutes) * interval_minutes + offset_minutes
        
        # Calculate next run
        next_run = current_time.replace(
            hour=adjusted_minutes // 60,
            minute=adjusted_minutes % 60,
            second=0,
            microsecond=0
        )
        
        # If the calculated time is in the past, add interval
        if next_run <= current_time:
            next_run += timedelta(minutes=interval_minutes)
        
        return next_run
    
    def _run_scheduler(self):
        """
        Main scheduler loop (runs in background thread)
        """
        logger.info("🔄 Starting MTF scheduler loop")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for tasks that need to run
                tasks_to_run = []
                
                for task_key, task in self.scheduled_tasks.items():
                    if task.is_active and current_time >= task.next_run:
                        tasks_to_run.append(task)
                        
                        # Calculate next run time
                        task.next_run = self._calculate_next_run_time(
                            current_time, 
                            task.interval_minutes, 
                            0
                        )
                
                # Run tasks asynchronously
                if tasks_to_run:
                    asyncio.run(self._execute_tasks(tasks_to_run))
                
                # Sleep for a short interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    async def _execute_tasks(self, tasks: List[ScheduledTask]):
        """
        Execute scheduled tasks
        """
        for task in tasks:
            try:
                start_time = datetime.now()
                
                # Execute the task
                await self._process_timeframe(task.symbol, task.timeframe)
                
                # Update statistics
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_stats(task.symbol, task.timeframe, processing_time, True)
                
                logger.info(f"✅ Scheduled task completed: {task.symbol} {task.timeframe} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Scheduled task failed: {task.symbol} {task.timeframe} - {e}")
                self._update_stats(task.symbol, task.timeframe, 0.0, False)
    
    async def _process_timeframe(self, symbol: str, timeframe: str):
        """
        Process a specific symbol and timeframe
        """
        try:
            # Get market data for the timeframe
            data = await self._get_market_data(symbol, timeframe)
            
            if data is None or data.empty:
                logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                return
            
            # Process with MTF orchestrator
            result = await self.mtf_orchestrator.process_symbol_timeframe(symbol, timeframe, data)
            
            # Store context in database
            await self._store_mtf_context(symbol, timeframe, result)
            
            # Update cache
            self.cache_manager.cache_mtf_context(symbol, timeframe, result)
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol} {timeframe}: {e}")
            raise
    
    async def _get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get market data for a symbol and timeframe
        """
        try:
            # This would integrate with your existing market data service
            # For now, we'll create mock data for testing
            return self._create_mock_data(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return None
    
    def _create_mock_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Create mock market data for testing
        """
        # Generate realistic mock data
        import numpy as np
        
        # Number of candles based on timeframe
        candle_counts = {"1d": 30, "4h": 180, "1h": 720, "15m": 2880, "5m": 8640}
        num_candles = candle_counts.get(timeframe, 100)
        
        # Generate OHLCV data
        np.random.seed(hash(f"{symbol}_{timeframe}") % 2**32)
        
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        prices = []
        current_price = base_price
        
        for i in range(num_candles):
            # Random price movement
            change = np.random.normal(0, 0.02)  # 2% volatility
            current_price *= (1 + change)
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            close_price = current_price
            
            prices.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.uniform(1000, 10000)
            })
        
        df = pd.DataFrame(prices)
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=num_candles//24),
            periods=num_candles,
            freq='H'
        )
        
        return df
    
    async def _store_mtf_context(self, symbol: str, timeframe: str, context_data: Dict):
        """
        Store MTF context in database
        """
        try:
            db = next(get_db())
            
            # Check if context already exists
            existing_context = db.query(MTFContext).filter(
                MTFContext.symbol == symbol,
                MTFContext.timeframe == timeframe
            ).first()
            
            if existing_context:
                # Update existing context
                existing_context.trend_direction = context_data.get('trend_direction', 'neutral')
                existing_context.trend_strength = context_data.get('trend_strength', 0.0)
                existing_context.pattern_confirmed = context_data.get('pattern_confirmed', False)
                existing_context.market_regime = context_data.get('market_regime', 'neutral')
                existing_context.confidence_score = context_data.get('confidence_score', 0.0)
                existing_context.technical_indicators = context_data.get('technical_indicators', {})
                existing_context.market_conditions = context_data.get('market_conditions', {})
                existing_context.updated_at = datetime.now()
                existing_context.cache_ttl = self._calculate_cache_ttl(timeframe)
            else:
                # Create new context
                new_context = MTFContext(
                    symbol=symbol,
                    timeframe=timeframe,
                    trend_direction=context_data.get('trend_direction', 'neutral'),
                    trend_strength=context_data.get('trend_strength', 0.0),
                    pattern_confirmed=context_data.get('pattern_confirmed', False),
                    market_regime=context_data.get('market_regime', 'neutral'),
                    confidence_score=context_data.get('confidence_score', 0.0),
                    technical_indicators=context_data.get('technical_indicators', {}),
                    market_conditions=context_data.get('market_conditions', {}),
                    candle_status='completed',
                    last_candle_close=datetime.now(),
                    next_candle_open=self._calculate_next_candle_open(timeframe),
                    cache_ttl=self._calculate_cache_ttl(timeframe),
                    is_active=True
                )
                db.add(new_context)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"❌ Error storing MTF context: {e}")
            db.rollback()
    
    def _calculate_cache_ttl(self, timeframe: str) -> datetime:
        """
        Calculate cache TTL based on timeframe
        """
        ttl_minutes = {
            "1d": 1440, "4h": 240, "1h": 60, "15m": 15, "5m": 5
        }
        
        minutes = ttl_minutes.get(timeframe, 60)
        return datetime.now() + timedelta(minutes=minutes)
    
    def _calculate_next_candle_open(self, timeframe: str) -> datetime:
        """
        Calculate next candle open time
        """
        interval_minutes = {
            "1d": 1440, "4h": 240, "1h": 60, "15m": 15, "5m": 5
        }
        
        minutes = interval_minutes.get(timeframe, 60)
        return datetime.now() + timedelta(minutes=minutes)
    
    def _update_stats(self, symbol: str, timeframe: str, processing_time: float, success: bool):
        """
        Update performance statistics
        """
        self.stats['total_scheduled_runs'] += 1
        
        if success:
            self.stats['successful_runs'] += 1
            self.stats['processing_times'].append(processing_time)
        else:
            self.stats['failed_runs'] += 1
        
        # Keep only last 100 processing times
        if len(self.stats['processing_times']) > 100:
            self.stats['processing_times'] = self.stats['processing_times'][-100:]
        
        # Update last run time
        task_key = f"{symbol}_{timeframe}"
        self.stats['last_run_times'][task_key] = datetime.now()
    
    async def add_symbol(self, symbol: str):
        """
        Add a new symbol to the scheduler
        """
        if symbol in self.active_symbols:
            logger.warning(f"⚠️ Symbol {symbol} is already active")
            return
        
        self.active_symbols.add(symbol)
        
        # Create scheduled tasks for the new symbol
        current_time = datetime.now()
        
        for timeframe, schedule_config in self.timeframe_schedules.items():
            task_key = f"{symbol}_{timeframe}"
            
            next_run = self._calculate_next_run_time(
                current_time, 
                schedule_config["interval"], 
                schedule_config["offset_minutes"]
            )
            
            self.scheduled_tasks[task_key] = ScheduledTask(
                symbol=symbol,
                timeframe=timeframe,
                next_run=next_run,
                interval_minutes=schedule_config["interval"],
                is_active=True
            )
        
        logger.info(f"➕ Added symbol {symbol} to MTF scheduler")
    
    async def remove_symbol(self, symbol: str):
        """
        Remove a symbol from the scheduler
        """
        if symbol not in self.active_symbols:
            logger.warning(f"⚠️ Symbol {symbol} is not active")
            return
        
        self.active_symbols.remove(symbol)
        
        # Remove scheduled tasks for the symbol
        tasks_to_remove = [key for key in self.scheduled_tasks.keys() if key.startswith(f"{symbol}_")]
        
        for task_key in tasks_to_remove:
            self.scheduled_tasks[task_key].is_active = False
            del self.scheduled_tasks[task_key]
        
        logger.info(f"➖ Removed symbol {symbol} from MTF scheduler")
    
    async def get_scheduler_status(self) -> Dict:
        """
        Get current scheduler status
        """
        active_tasks = [task for task in self.scheduled_tasks.values() if task.is_active]
        
        return {
            'is_running': self.is_running,
            'active_symbols': list(self.active_symbols),
            'total_scheduled_tasks': len(self.scheduled_tasks),
            'active_tasks': len(active_tasks),
            'next_runs': {
                f"{task.symbol}_{task.timeframe}": task.next_run.isoformat()
                for task in active_tasks[:10]  # Show next 10 runs
            },
            'stats': self.stats
        }
    
    async def force_run(self, symbol: str, timeframe: str):
        """
        Force run a specific symbol and timeframe immediately
        """
        try:
            logger.info(f"🚀 Force running {symbol} {timeframe}")
            await self._process_timeframe(symbol, timeframe)
            logger.info(f"✅ Force run completed: {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"❌ Force run failed: {symbol} {timeframe} - {e}")
            raise
