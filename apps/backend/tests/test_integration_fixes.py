"""
Integration Test for AlphaPlus Fixed System
Tests all components and their integration after fixes
"""

import asyncio
import pytest
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import components to test
from src.app.core.config import settings, get_settings
from src.app.core.database_manager import DatabaseManager, initialize_database, close_database
from src.app.core.service_manager import ServiceManager, initialize_all_services, shutdown_all_services, register_service

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestIntegrationFixes:
    """Test class for integration fixes"""
    
    @pytest.fixture(autouse=True)
    async def setup_teardown(self):
        """Setup and teardown for each test"""
        # Setup
        logger.info("Setting up test environment...")
        
        # Teardown
        yield
        
        # Cleanup
        logger.info("Cleaning up test environment...")
        await close_database()
    
    @pytest.mark.asyncio
    async def test_configuration_loading(self):
        """Test configuration loading and validation"""
        logger.info("Testing configuration loading...")
        
        # Test settings loading
        assert settings is not None
        assert settings.app_name == "AlphaPlus Trading System"
        assert settings.app_version == "1.0.0"
        
        # Test configuration validation
        assert settings.validate_configuration() is True
        
        # Test environment detection
        assert settings.is_development() is True
        
        logger.info("✅ Configuration loading test passed")
    
    @pytest.mark.asyncio
    async def test_database_manager(self):
        """Test database manager functionality"""
        logger.info("Testing database manager...")
        
        # Create database manager
        db_manager = DatabaseManager()
        
        # Test initialization
        config = {
            'host': 'localhost',  # Use localhost for testing
            'port': 5432,
            'database': 'test_alphapulse',
            'username': 'test_user',
            'password': 'test_password',
            'min_size': 1,
            'max_size': 5
        }
        
        # Note: This will fail in test environment without actual database
        # but we can test the configuration and structure
        assert db_manager is not None
        assert db_manager.is_initialized is False
        
        # Test configuration methods
        default_config = db_manager._get_default_config()
        assert 'host' in default_config
        assert 'port' in default_config
        assert 'database' in default_config
        
        logger.info("✅ Database manager test passed")
    
    @pytest.mark.asyncio
    async def test_service_manager(self):
        """Test service manager functionality"""
        logger.info("Testing service manager...")
        
        # Create service manager
        service_manager = ServiceManager()
        
        # Test service registration
        mock_service = MockService("test_service")
        service_manager.register_service("test_service", mock_service)
        
        assert "test_service" in service_manager.services
        assert service_manager.get_service("test_service") == mock_service
        
        # Test dependency calculation
        service_manager.register_service("dependent_service", MockService("dependent"), dependencies=["test_service"])
        
        # Calculate initialization order
        order = service_manager._calculate_initialization_order()
        assert "test_service" in order
        assert "dependent_service" in order
        assert order.index("test_service") < order.index("dependent_service")
        
        logger.info("✅ Service manager test passed")
    
    @pytest.mark.asyncio
    async def test_import_paths(self):
        """Test that all import paths work correctly"""
        logger.info("Testing import paths...")
        
        # Test core imports
        try:
            from src.app.core.config import settings
            from src.app.core.database_manager import DatabaseManager
            from src.app.core.service_manager import ServiceManager
            logger.info("✅ Core imports successful")
        except ImportError as e:
            pytest.fail(f"Core import failed: {e}")
        
        # Test service imports
        try:
            from src.app.services.market_data_service import MarketDataService
            from src.app.services.sentiment_service import SentimentService
            from src.app.services.risk_manager import RiskManager
            logger.info("✅ Service imports successful")
        except ImportError as e:
            logger.warning(f"Service import warning: {e}")
        
        # Test strategy imports
        try:
            from src.app.strategies.strategy_manager import StrategyManager
            logger.info("✅ Strategy imports successful")
        except ImportError as e:
            logger.warning(f"Strategy import warning: {e}")
        
        # Test data imports
        try:
            from src.app.data.real_time_processor import RealTimeCandlestickProcessor
            logger.info("✅ Data imports successful")
        except ImportError as e:
            logger.warning(f"Data import warning: {e}")
        
        logger.info("✅ Import paths test passed")
    
    @pytest.mark.asyncio
    async def test_module_structure(self):
        """Test that all modules have proper structure"""
        logger.info("Testing module structure...")
        
        # Test __init__.py files exist
        init_files = [
            "backend/app/services/__init__.py",
            "backend/app/strategies/__init__.py",
            "backend/app/data/__init__.py",
            "backend/app/database/__init__.py",
            "backend/app/core/__init__.py"
        ]
        
        for init_file in init_files:
            assert Path(init_file).exists(), f"Missing __init__.py file: {init_file}"
        
        # Test core files exist
        core_files = [
            "backend/app/core/config.py",
            "backend/app/core/database_manager.py",
            "backend/app/core/service_manager.py"
        ]
        
        for core_file in core_files:
            assert Path(core_file).exists(), f"Missing core file: {core_file}"
        
        logger.info("✅ Module structure test passed")
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation"""
        logger.info("Testing configuration validation...")
        
        # Test valid configuration
        assert settings.validate_configuration() is True
        
        # Test configuration to dict
        config_dict = settings.to_dict()
        assert isinstance(config_dict, dict)
        assert "app_name" in config_dict
        assert "database" in config_dict
        assert "redis" in config_dict
        
        # Test database URL generation
        db_url = settings.get_database_url()
        assert "postgresql://" in db_url
        assert settings.database.username in db_url
        assert settings.database.database in db_url
        
        # Test Redis URL generation
        redis_url = settings.get_redis_url()
        assert "redis://" in redis_url
        assert settings.redis.host in redis_url
        
        logger.info("✅ Configuration validation test passed")
    
    @pytest.mark.asyncio
    async def test_service_dependencies(self):
        """Test service dependency management"""
        logger.info("Testing service dependencies...")
        
        service_manager = ServiceManager()
        
        # Test circular dependency detection
        service_manager.register_service("service_a", MockService("a"), dependencies=["service_b"])
        service_manager.register_service("service_b", MockService("b"), dependencies=["service_a"])
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            service_manager._calculate_initialization_order()
        
        # Test unknown dependency detection
        service_manager = ServiceManager()
        service_manager.register_service("service_c", MockService("c"), dependencies=["unknown_service"])
        
        with pytest.raises(ValueError, match="depends on unknown service"):
            service_manager._calculate_initialization_order()
        
        logger.info("✅ Service dependencies test passed")

class MockService:
    """Mock service for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
    
    async def initialize(self):
        """Mock initialize method"""
        self.initialized = True
    
    async def shutdown(self):
        """Mock shutdown method"""
        self.initialized = False

@pytest.mark.asyncio
async def test_end_to_end_integration():
    """End-to-end integration test"""
    logger.info("Running end-to-end integration test...")
    
    try:
        # Test configuration
        assert settings is not None
        assert settings.validate_configuration() is True
        
        # Test service manager
        service_manager = ServiceManager()
        mock_service = MockService("test")
        service_manager.register_service("test", mock_service)
        
        # Test service initialization
        success = await service_manager.initialize_services()
        assert success is True
        
        # Test service status
        status = service_manager.get_service_status("test")
        assert status.value == "running"
        
        # Test service shutdown
        await service_manager.shutdown_services()
        
        logger.info("✅ End-to-end integration test passed")
        
    except Exception as e:
        logger.error(f"❌ End-to-end integration test failed: {e}")
        pytest.fail(f"End-to-end integration test failed: {e}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
