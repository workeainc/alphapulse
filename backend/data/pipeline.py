#!/usr/bin/env python3
"""
Data Pipeline Orchestrator for AlphaPulse
Coordinates data fetching, validation, and storage
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import json

from .exchange_connector import ExchangeConnector, CandlestickData
from .validation import CandlestickValidator
from .storage import DataStorage
from ..strategies.indicators import TechnicalIndicators
from ..strategies.pattern_detector import CandlestickPatternDetector, PatternSignal

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    symbols: List[str]
    intervals: List[str]
    exchanges: List[str]
    batch_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    validation_enabled: bool = True
    storage_type: str = "postgresql"
    storage_path: str = "data"
    max_workers: int = 5
    update_frequency_minutes: int = 60
    analysis_enabled: bool = True
    pattern_detection_enabled: bool = True
    technical_indicators_enabled: bool = True

@dataclass
class PipelineStatus:
    """Status information for pipeline operations"""
    symbol: str
    interval: str
    exchange: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_stored: int = 0
    error_message: Optional[str] = None
    validation_report: Optional[Dict] = None
    analysis_report: Optional[Dict] = None
    patterns_detected: int = 0
    indicators_calculated: int = 0

class DataPipeline:
    """Orchestrates the complete data pipeline workflow"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize data pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        # Initialize with default exchange connector (will be overridden per exchange)
        self.connector = None
        self.validator = CandlestickValidator()
        self.storage = DataStorage(config.storage_path, config.storage_type)
        
        # Initialize analysis components
        if config.analysis_enabled:
            self.indicators_calc = TechnicalIndicators()
            self.pattern_detector = CandlestickPatternDetector()
            logger.info("Technical analysis components initialized")
        else:
            self.indicators_calc = None
            self.pattern_detector = None
        
        # Pipeline state
        self.status: Dict[str, PipelineStatus] = {}
        self.is_running = False
        self.last_update = {}
        
        # Initialize status for all symbol/interval/exchange combinations
        self._init_pipeline_status()
        
        logger.info(f"Data pipeline initialized with {len(config.symbols)} symbols, {len(config.intervals)} intervals")
    
    def _init_pipeline_status(self):
        """Initialize pipeline status for all combinations"""
        for symbol in self.config.symbols:
            for interval in self.config.intervals:
                for exchange in self.config.exchanges:
                    key = f"{symbol}_{interval}_{exchange}"
                    self.status[key] = PipelineStatus(
                        symbol=symbol,
                        interval=interval,
                        exchange=exchange,
                        status='pending'
                    )
    
    async def run_pipeline(self, force_update: bool = False) -> Dict[str, PipelineStatus]:
        """
        Run the complete data pipeline
        
        Args:
            force_update: Force update even if recent data exists
            
        Returns:
            Dictionary of pipeline status for each symbol/interval/exchange
        """
        if self.is_running:
            logger.warning("Pipeline is already running")
            return self.status
        
        self.is_running = True
        start_time = datetime.now()
        
        try:
            logger.info("Starting data pipeline execution")
            
            # Create tasks for all symbol/interval/exchange combinations
            tasks = []
            for symbol in self.config.symbols:
                for interval in self.config.intervals:
                    for exchange in self.config.exchanges:
                        task = self._process_symbol_interval(
                            symbol, interval, exchange, force_update
                        )
                        tasks.append(task)
            
            # Execute tasks with concurrency control
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def controlled_task(task):
                async with semaphore:
                    return await task
            
            controlled_tasks = [controlled_task(task) for task in tasks]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*controlled_tasks, return_exceptions=True)
            
            # Process results and update status
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                    # Update status for failed tasks
                    task_index = i
                    symbol_index = task_index // (len(self.config.intervals) * len(self.config.exchanges))
                    interval_index = (task_index % (len(self.config.intervals) * len(self.config.exchanges))) // len(self.config.exchanges)
                    exchange_index = task_index % len(self.config.exchanges)
                    
                    symbol = self.config.symbols[symbol_index]
                    interval = self.config.intervals[interval_index]
                    exchange = self.config.exchanges[exchange_index]
                    
                    key = f"{symbol}_{interval}_{exchange}"
                    if key in self.status:
                        self.status[key].status = 'failed'
                        self.status[key].error_message = str(result)
                        self.status[key].end_time = datetime.now()
            
            execution_time = datetime.now() - start_time
            logger.info(f"Pipeline execution completed in {execution_time}")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
        finally:
            self.is_running = False
        
        return self.status
    
    async def _process_symbol_interval(self, symbol: str, interval: str, exchange: str, force_update: bool) -> PipelineStatus:
        """
        Process a single symbol/interval/exchange combination
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            exchange: Exchange name
            force_update: Force update even if recent data exists
            
        Returns:
            Pipeline status
        """
        key = f"{symbol}_{interval}_{exchange}"
        status = self.status[key]
        
        try:
            # Check if update is needed
            if not force_update and self._should_skip_update(symbol, interval, exchange):
                status.status = 'completed'
                status.start_time = datetime.now()
                status.end_time = datetime.now()
                status.records_processed = 0
                status.records_stored = 0
                logger.info(f"Skipping update for {key} - recent data available")
                return status
            
            # Update status to running
            status.status = 'running'
            status.start_time = datetime.now()
            status.error_message = None
            
            logger.info(f"Processing {key}")
            
            # Step 1: Fetch data
            df = await self._fetch_data(symbol, interval, exchange)
            if df is None or df.empty:
                status.status = 'failed'
                status.error_message = "No data fetched"
                status.end_time = datetime.now()
                return status
            
            status.records_processed = len(df)
            
            # Step 2: Validate data
            if self.config.validation_enabled:
                df, validation_report = self.validator.validate_candlestick_data(df)
                status.validation_report = validation_report
                
                if df.empty:
                    status.status = 'failed'
                    status.error_message = "All data failed validation"
                    status.end_time = datetime.now()
                    return status
                
                logger.info(f"Validation completed for {key}: {len(df)} records passed")
            
            # Step 3: Technical Analysis (if enabled)
            if self.config.analysis_enabled and self.indicators_calc and self.pattern_detector:
                df, analysis_report = await self._perform_technical_analysis(df, symbol, interval, exchange)
                status.analysis_report = analysis_report
                status.patterns_detected = analysis_report.get('patterns_detected', 0)
                status.indicators_calculated = analysis_report.get('indicators_calculated', 0)
                logger.info(f"Technical analysis completed for {key}: {status.patterns_detected} patterns, {status.indicators_calculated} indicators")
            
            # Step 4: Store data
            storage_success = self.storage.store_candlestick_data(
                df, symbol, interval, exchange, overwrite=True
            )
            
            if storage_success:
                status.status = 'completed'
                status.records_stored = len(df)
                self.last_update[key] = datetime.now()
                logger.info(f"Successfully stored {len(df)} records for {key}")
            else:
                status.status = 'failed'
                status.error_message = "Failed to store data"
            
            status.end_time = datetime.now()
            return status
            
        except Exception as e:
            status.status = 'failed'
            status.error_message = str(e)
            status.end_time = datetime.now()
            logger.error(f"Error processing {key}: {e}")
            return status
    
    async def _fetch_data(self, symbol: str, interval: str, exchange: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a symbol/interval/exchange combination
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            exchange: Exchange name
            
        Returns:
            DataFrame with candlestick data or None
        """
        try:
            # Determine time range for data fetching
            end_time = datetime.now()
            
            # Check existing data to determine start time
            existing_data = self.storage.retrieve_candlestick_data(
                symbol, interval, exchange, limit=1
            )
            
            if existing_data is not None and not existing_data.empty:
                # Start from the last available timestamp
                start_time = existing_data.index.max() + self._get_interval_timedelta(interval)
            else:
                # No existing data, fetch last 30 days
                start_time = end_time - timedelta(days=30)
            
            # Ensure start time is not in the future
            if start_time >= end_time:
                logger.info(f"No new data needed for {symbol} {interval} {exchange}")
                return pd.DataFrame()
            
            # Fetch data from exchange
            logger.info(f"Fetching data for {symbol} {interval} {exchange} from {start_time} to {end_time}")
            
            # Convert interval to exchange format
            exchange_interval = self._convert_interval_format(interval)
            
            # Create exchange connector for this specific exchange
            from .exchange_connector import create_exchange_connector
            connector = create_exchange_connector(exchange)
            
            # Fetch data using connector
            data = await connector.fetch_ohlcv(
                symbol=symbol,
                timeframe=exchange_interval,
                limit=1000,
                start_time=start_time,
                end_time=end_time
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No data received for {symbol} {interval} {exchange}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(data)
            
            logger.info(f"Fetched {len(df)} records for {symbol} {interval} {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {interval} {exchange}: {e}")
            return None
    
    def _convert_to_dataframe(self, candlestick_data: List[CandlestickData]) -> pd.DataFrame:
        """Convert list of CandlestickData to DataFrame"""
        if not candlestick_data:
            return pd.DataFrame()
        
        data = []
        for candle in candlestick_data:
            row = {
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            }
            
            if candle.quote_volume is not None:
                row['quote_volume'] = candle.quote_volume
            if candle.trades_count is not None:
                row['trades_count'] = candle.trades_count
            if candle.taker_buy_base is not None:
                row['taker_buy_base'] = candle.taker_buy_base
            if candle.taker_buy_quote is not None:
                row['taker_buy_quote'] = candle.taker_buy_quote
            
            data.append(row)
        
        df = pd.DataFrame(data, index=[candle.timestamp for candle in candlestick_data])
        df.index.name = 'timestamp'
        
        return df
    
    def _convert_interval_format(self, interval: str) -> str:
        """Convert internal interval format to exchange format"""
        # Map internal intervals to exchange-specific formats
        interval_mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        
        return interval_mapping.get(interval, interval)
    
    def _get_interval_timedelta(self, interval: str) -> timedelta:
        """Get timedelta for an interval"""
        interval_mapping = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }
        
        return interval_mapping.get(interval, timedelta(hours=1))
    
    def _should_skip_update(self, symbol: str, interval: str, exchange: str) -> bool:
        """Check if update should be skipped based on recent data"""
        key = f"{symbol}_{interval}_{exchange}"
        
        if key not in self.last_update:
            return False
        
        last_update = self.last_update[key]
        update_frequency = timedelta(minutes=self.config.update_frequency_minutes)
        
        return datetime.now() - last_update < update_frequency
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline status"""
        summary = {
            'total_combinations': len(self.status),
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
            'is_running': self.is_running,
            'last_update': self.last_update,
            'status_breakdown': {}
        }
        
        for key, status in self.status.items():
            summary[status.status] += 1
            
            if status.status not in summary['status_breakdown']:
                summary['status_breakdown'][status.status] = []
            
            summary['status_breakdown'][status.status].append({
                'symbol': status.symbol,
                'interval': status.interval,
                'exchange': status.exchange,
                'records_processed': status.records_processed,
                'records_stored': status.records_stored,
                'error_message': status.error_message
            })
        
        return summary
    
    def get_symbol_status(self, symbol: str) -> Dict:
        """Get status for a specific symbol across all intervals and exchanges"""
        symbol_status = {}
        
        for key, status in self.status.items():
            if status.symbol == symbol:
                interval_exchange = f"{status.interval}_{status.exchange}"
                symbol_status[interval_exchange] = {
                    'status': status.status,
                    'records_processed': status.records_processed,
                    'records_stored': status.records_stored,
                    'last_update': self.last_update.get(key),
                    'error_message': status.error_message
                }
        
        return symbol_status
    
    def reset_pipeline_status(self):
        """Reset all pipeline status to pending"""
        for status in self.status.values():
            status.status = 'pending'
            status.start_time = None
            status.end_time = None
            status.records_processed = 0
            status.records_stored = 0
            status.error_message = None
            status.validation_report = None
        
        self.last_update.clear()
        logger.info("Pipeline status reset")
    
    def export_pipeline_report(self, filepath: str = None) -> str:
        """Export pipeline report to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"pipeline_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'symbols': self.config.symbols,
                'intervals': self.config.intervals,
                'exchanges': self.config.exchanges,
                'batch_size': self.config.batch_size,
                'retry_attempts': self.config.retry_attempts,
                'validation_enabled': self.config.validation_enabled,
                'storage_type': self.config.storage_type
            },
            'summary': self.get_pipeline_summary(),
            'detailed_status': {}
        }
        
        # Convert status objects to dictionaries
        for key, status in self.status.items():
            report['detailed_status'][key] = {
                'symbol': status.symbol,
                'interval': status.interval,
                'exchange': status.exchange,
                'status': status.status,
                'start_time': status.start_time.isoformat() if status.start_time else None,
                'end_time': status.end_time.isoformat() if status.end_time else None,
                'records_processed': status.records_processed,
                'records_stored': status.records_stored,
                'error_message': status.error_message,
                'validation_report': status.validation_report
            }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Pipeline report exported to {filepath}")
        return filepath
    
    async def _perform_technical_analysis(self, df: pd.DataFrame, symbol: str, 
                                        interval: str, exchange: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform technical analysis on candlestick data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            interval: Time interval
            exchange: Exchange name
            
        Returns:
            Tuple of (enhanced DataFrame, analysis report)
        """
        analysis_report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'interval': interval,
            'exchange': exchange,
            'patterns_detected': 0,
            'indicators_calculated': 0,
            'analysis_summary': {},
            'signals': {},
            'execution_time': None
        }
        
        start_time = time.time()
        
        try:
            # Ensure DataFrame has required columns
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                logger.error(f"DataFrame missing required columns for analysis: {df.columns}")
                return df, analysis_report
            
            # Calculate technical indicators
            if self.config.technical_indicators_enabled:
                indicators = self.indicators_calc.calculate_all_indicators(df)
                analysis_report['indicators_calculated'] = len(indicators)
                
                # Add indicators to DataFrame
                for name, values in indicators.items():
                    df[name] = values
                
                # Calculate signal strengths
                signals = self.indicators_calc.get_signal_strength(indicators)
                analysis_report['signals']['technical_indicators'] = signals
                
                logger.info(f"Calculated {len(indicators)} technical indicators for {symbol}")
            
            # Detect candlestick patterns
            if self.config.pattern_detection_enabled:
                pattern_signals = self.pattern_detector.detect_patterns_from_dataframe(df)
                analysis_report['patterns_detected'] = len(pattern_signals)
                
                # Add pattern information to DataFrame
                df['pattern_detected'] = False
                df['pattern_name'] = ''
                df['pattern_type'] = ''
                df['pattern_confidence'] = 0.0
                
                for signal in pattern_signals:
                    if signal.index < len(df):
                        df.iloc[signal.index, df.columns.get_loc('pattern_detected')] = True
                        df.iloc[signal.index, df.columns.get_loc('pattern_name')] = signal.pattern
                        df.iloc[signal.index, df.columns.get_loc('pattern_type')] = signal.type
                        df.iloc[signal.index, df.columns.get_loc('pattern_confidence')] = signal.confidence
                
                # Get pattern summary
                pattern_summary = self.pattern_detector.get_pattern_summary(pattern_signals)
                analysis_report['signals']['candlestick_patterns'] = pattern_summary
                
                logger.info(f"Detected {len(pattern_signals)} candlestick patterns for {symbol}")
            
            # Generate comprehensive analysis summary
            analysis_report['analysis_summary'] = self._generate_analysis_summary(df, analysis_report)
            
            execution_time = time.time() - start_time
            analysis_report['execution_time'] = execution_time
            
            logger.info(f"Technical analysis completed for {symbol} in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during technical analysis for {symbol}: {e}")
            analysis_report['error'] = str(e)
        
        return df, analysis_report
    
    def _generate_analysis_summary(self, df: pd.DataFrame, analysis_report: Dict) -> Dict:
        """
        Generate comprehensive analysis summary
        
        Args:
            df: Enhanced DataFrame with indicators and patterns
            analysis_report: Analysis report dictionary
            
        Returns:
            Analysis summary dictionary
        """
        summary = {
            'data_points': len(df),
            'time_range': {
                'start': df.index.min().isoformat() if not df.empty else None,
                'end': df.index.max().isoformat() if not df.empty else None
            },
            'price_summary': {},
            'volume_summary': {},
            'trend_analysis': {},
            'volatility_analysis': {},
            'signal_summary': {}
        }
        
        if not df.empty:
            # Price summary
            summary['price_summary'] = {
                'current_price': float(df['close'].iloc[-1]),
                'price_change': float(df['close'].iloc[-1] - df['close'].iloc[0]),
                'price_change_pct': float(((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100),
                'high': float(df['high'].max()),
                'low': float(df['low'].min()),
                'average_price': float(df['close'].mean())
            }
            
            # Volume summary
            if 'volume' in df.columns:
                summary['volume_summary'] = {
                    'current_volume': float(df['volume'].iloc[-1]),
                    'average_volume': float(df['volume'].mean()),
                    'volume_trend': 'increasing' if df['volume'].iloc[-1] > df['volume'].mean() else 'decreasing'
                }
            
            # Trend analysis
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                current_ema_12 = df['ema_12'].iloc[-1]
                current_ema_26 = df['ema_26'].iloc[-1]
                
                if not (pd.isna(current_ema_12) or pd.isna(current_ema_26)):
                    summary['trend_analysis'] = {
                        'trend': 'bullish' if current_ema_12 > current_ema_26 else 'bearish',
                        'ema_12': float(current_ema_12),
                        'ema_26': float(current_ema_26),
                        'trend_strength': abs(current_ema_12 - current_ema_26) / current_ema_26 * 100
                    }
            
            # Volatility analysis
            if 'atr' in df.columns:
                current_atr = df['atr'].iloc[-1]
                if not pd.isna(current_atr):
                    summary['volatility_analysis'] = {
                        'current_atr': float(current_atr),
                        'average_atr': float(df['atr'].mean()),
                        'volatility_trend': 'increasing' if current_atr > df['atr'].mean() else 'decreasing'
                    }
            
            # Signal summary
            signals = analysis_report.get('signals', {})
            summary['signal_summary'] = {
                'total_signals': len(signals),
                'signal_types': list(signals.keys()),
                'strongest_signals': []
            }
            
            # Find strongest signals
            all_signals = []
            for signal_type, signal_data in signals.items():
                if isinstance(signal_data, dict):
                    if 'strongest_patterns' in signal_data:
                        all_signals.extend(signal_data['strongest_patterns'])
                    elif isinstance(signal_data, dict):
                        for name, value in signal_data.items():
                            if isinstance(value, (int, float)):
                                all_signals.append({'name': name, 'value': value, 'type': signal_type})
            
            # Sort by value/confidence
            all_signals.sort(key=lambda x: abs(x.get('value', 0)), reverse=True)
            summary['signal_summary']['strongest_signals'] = all_signals[:5]
        
        return summary

# Example usage and testing
async def test_pipeline():
    """Test the data pipeline functionality"""
    # Create test configuration
    config = PipelineConfig(
        symbols=['BTCUSDT', 'ETHUSDT'],
        intervals=['1h', '4h'],
        exchanges=['binance'],
        batch_size=100,
        retry_attempts=2,
        validation_enabled=True,
        storage_type='postgresql',
        storage_path='test_data',
        max_workers=2,
        update_frequency_minutes=5
    )
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    # Run pipeline
    print("Starting pipeline...")
    status = await pipeline.run_pipeline(force_update=True)
    
    # Get summary
    summary = pipeline.get_pipeline_summary()
    print(f"Pipeline summary: {summary}")
    
    # Export report
    report_file = pipeline.export_pipeline_report()
    print(f"Report exported to: {report_file}")
    
    # Clean up
    import shutil
    if Path("test_data").exists():
        shutil.rmtree("test_data")
    
    return status, summary

if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_pipeline())
