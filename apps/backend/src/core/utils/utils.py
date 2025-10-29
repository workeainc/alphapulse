"""
General Utilities for AlphaPulse

This module provides common utility functions, configuration management,
and helper classes used throughout the AlphaPulse system.
"""

import os
import json
import yaml
import logging
import asyncio
import time
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from functools import wraps, lru_cache
import redis
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Config:
    """Configuration class for AlphaPulse."""
    # Database settings
    database_url: str = "postgresql://user:pass@localhost:5432/alphapulse"
    redis_url: str = "redis://localhost:6379"
    
    # Trading settings
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "15m", "1h"])
    max_positions: int = 10
    risk_per_trade: float = 0.02
    
    # Performance settings
    latency_target: int = 50  # milliseconds
    accuracy_target: float = 0.75
    filter_rate_target: float = 0.65
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 8
    macd_slow: int = 24
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    
    # ML settings
    model_path: str = "models/"
    feature_window: int = 100
    prediction_threshold: float = 0.7
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/alphapulse.log"
    
    # Environment
    environment: str = "production"
    debug: bool = False


class ConfigManager:
    """
    Configuration management system for AlphaPulse.
    
    Handles loading, validation, and dynamic updates of configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = Config()
        self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self):
        """Load configuration from file and environment variables."""
        # Load from file if exists
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                # Update config with file values
                for key, value in file_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration file: {e}")
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'ALPHAPULSE_DATABASE_URL': 'database_url',
            'ALPHAPULSE_REDIS_URL': 'redis_url',
            'ALPHAPULSE_SYMBOLS': 'symbols',
            'ALPHAPULSE_TIMEFRAMES': 'timeframes',
            'ALPHAPULSE_MAX_POSITIONS': 'max_positions',
            'ALPHAPULSE_RISK_PER_TRADE': 'risk_per_trade',
            'ALPHAPULSE_LATENCY_TARGET': 'latency_target',
            'ALPHAPULSE_ACCURACY_TARGET': 'accuracy_target',
            'ALPHAPULSE_FILTER_RATE_TARGET': 'filter_rate_target',
            'ALPHAPULSE_LOG_LEVEL': 'log_level',
            'ALPHAPULSE_ENVIRONMENT': 'environment',
            'ALPHAPULSE_DEBUG': 'debug'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                if config_key in ['symbols', 'timeframes']:
                    value = value.split(',')
                elif config_key in ['max_positions', 'latency_target']:
                    value = int(value)
                elif config_key in ['risk_per_trade', 'accuracy_target', 'filter_rate_target']:
                    value = float(value)
                elif config_key == 'debug':
                    value = value.lower() in ['true', '1', 'yes']
                
                setattr(self.config, config_key, value)
    
    def _validate_configuration(self):
        """Validate configuration values."""
        errors = []
        
        # Validate database URL
        if not self.config.database_url:
            errors.append("Database URL is required")
        
        # Validate Redis URL
        if not self.config.redis_url:
            errors.append("Redis URL is required")
        
        # Validate trading parameters
        if self.config.risk_per_trade <= 0 or self.config.risk_per_trade > 0.1:
            errors.append("Risk per trade must be between 0 and 0.1")
        
        if self.config.max_positions <= 0:
            errors.append("Max positions must be positive")
        
        # Validate performance targets
        if self.config.accuracy_target < 0 or self.config.accuracy_target > 1:
            errors.append("Accuracy target must be between 0 and 1")
        
        if self.config.filter_rate_target < 0 or self.config.filter_rate_target > 1:
            errors.append("Filter rate target must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_config(self) -> Config:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        self._validate_configuration()
        logger.info("Configuration updated")
    
    def save_config(self, file_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration (optional)
        """
        save_path = file_path or self.config_path
        
        # Convert config to dictionary
        config_dict = {}
        for key, value in self.config.__dict__.items():
            config_dict[key] = value
        
        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def start_timer(self, name: str):
        """Start a performance timer."""
        self.metrics[name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
    
    def end_timer(self, name: str) -> float:
        """
        End a performance timer and return duration.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds
        """
        if name not in self.metrics:
            logger.warning(f"Timer {name} not found")
            return 0.0
        
        with self.lock:
            self.metrics[name]['end_time'] = time.time()
            duration = self.metrics[name]['end_time'] - self.metrics[name]['start_time']
            self.metrics[name]['duration'] = duration
        
        return duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        with self.lock:
            return self.metrics.copy()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_free': disk.free
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}


class Cache:
    """
    Simple in-memory cache with TTL support.
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache = {}
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cache value.
        
        Args:
            key: Cache key
            value: Cache value
            ttl: Time-to-live in seconds (optional)
        """
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        with self.lock:
            self.cache[key] = {
                'value': value,
                'expiry': expiry
            }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            item = self.cache[key]
            if time.time() > item['expiry']:
                del self.cache[key]
                return None
            
            return item['value']
    
    def delete(self, key: str):
        """Delete cache entry."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def cleanup(self):
        """Remove expired entries."""
        current_time = time.time()
        with self.lock:
            expired_keys = [
                key for key, item in self.cache.items()
                if current_time > item['expiry']
            ]
            for key in expired_keys:
                del self.cache[key]


class RateLimiter:
    """
    Rate limiting utility for API calls and operations.
    """
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def can_call(self) -> bool:
        """
        Check if a call can be made.
        
        Returns:
            True if call is allowed
        """
        current_time = time.time()
        
        with self.lock:
            # Remove old calls
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_window]
            
            # Check if we can make another call
            if len(self.calls) < self.max_calls:
                self.calls.append(current_time)
                return True
            
            return False
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.can_call():
            time.sleep(0.1)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for functions that may fail.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            logger.error(f"All {max_attempts} attempts failed. Last error: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Async retry decorator for async functions that may fail.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            logger.error(f"All {max_attempts} attempts failed. Last error: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def hash_data(data: str) -> str:
    """Generate SHA-256 hash of data."""
    return hashlib.sha256(data.encode()).hexdigest()


def format_timestamp(timestamp: Union[datetime, float, int]) -> str:
    """Format timestamp for logging and display."""
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        dt = timestamp
    
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    except ValueError:
        # Try other formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Log file path (optional)
        format_string: Log format string (optional)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logger.error(f"Error setting up file logging: {e}")


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}


def save_json_file(data: Dict[str, Any], file_path: str):
    """Save data to JSON file safely."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load YAML file safely."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {e}")
        return {}


def save_yaml_file(data: Dict[str, Any], file_path: str):
    """Save data to YAML file safely."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {e}")


def run_in_threadpool(func: Callable, *args, **kwargs) -> Any:
    """
    Run function in thread pool.
    
    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    with ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result()


async def run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """
    Run function in executor (async wrapper).
    
    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


# Example usage
def example_usage():
    """Example usage of utility functions."""
    
    # Setup logging
    setup_logging(level="INFO", log_file="logs/utils.log")
    
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"Database URL: {config.database_url}")
    print(f"Symbols: {config.symbols}")
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_timer("example_operation")
    time.sleep(0.1)  # Simulate work
    duration = monitor.end_timer("example_operation")
    print(f"Operation took: {duration:.3f}s")
    
    # Cache usage
    cache = Cache(default_ttl=60)
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    print(f"Cached value: {value}")
    
    # Rate limiting
    limiter = RateLimiter(max_calls=5, time_window=60)
    for i in range(10):
        if limiter.can_call():
            print(f"Call {i+1}: Allowed")
        else:
            print(f"Call {i+1}: Rate limited")
    
    # Generate ID and hash
    unique_id = generate_id()
    data_hash = hash_data("test data")
    print(f"Generated ID: {unique_id}")
    print(f"Data hash: {data_hash}")
    
    # Timestamp formatting
    timestamp = format_timestamp(datetime.now())
    print(f"Formatted timestamp: {timestamp}")


if __name__ == "__main__":
    example_usage()
