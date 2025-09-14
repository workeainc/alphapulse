"""
Test configuration and fixtures for AlphaPlus Trading System.

This file consolidates all test configuration, fixtures, and setup
that was previously scattered across multiple test directories.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the backend directory to the Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Add the backend/app directory to the Python path
app_path = backend_path / "app"
sys.path.insert(0, str(app_path))

# Add the backend/ai directory to the Python path
ai_path = backend_path / "ai"
sys.path.insert(0, str(ai_path))

# Add the backend/strategies directory to the Python path
strategies_path = backend_path / "strategies"
sys.path.insert(0, str(strategies_path))

# Add the backend/services directory to the Python path
services_path = backend_path / "services"
sys.path.insert(0, str(services_path))

# Add the backend/core directory to the Python path
core_path = backend_path / "core"
sys.path.insert(0, str(core_path))

# Test configuration
pytest_plugins = [
    # "tests.fixtures.database",  # Commented out - module not found
    # "tests.fixtures.models",    # Commented out - module not found
    # "tests.fixtures.strategies", # Commented out - module not found
    # "tests.fixtures.data",      # Commented out - module not found
]

# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for system components"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "database: Tests that require database access"
    )

# Test session setup
@pytest.fixture(scope="session")
def test_session():
    """Session-level test setup."""
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="alphapulse_test_")
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["TEST_DB_PATH"] = str(Path(test_dir) / "test.db")
    os.environ["TEST_CACHE_PATH"] = str(Path(test_dir) / "cache")
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

# Test directory fixtures
@pytest.fixture
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def test_models_dir():
    """Provide test models directory."""
    return Path(__file__).parent / "models"

@pytest.fixture
def test_cache_dir():
    """Provide test cache directory."""
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for each test."""
    # Backup original environment
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    os.environ.update({
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "test",
        "DB_CONNECTION_TIMEOUT": "5",
        "CACHE_TTL": "60",
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    
    # Clean up any temporary files created during tests
    temp_files = [
        "test_*.db",
        "test_*.log",
        "test_*.json",
        "test_*.csv",
        "*.tmp"
    ]
    
    for pattern in temp_files:
        for file_path in Path(".").glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors

# Performance test configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle performance tests."""
    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if any(keyword in item.nodeid.lower() for keyword in ["benchmark", "stress", "load"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark database tests
        if any(keyword in item.nodeid.lower() for keyword in ["db", "database", "dao"]):
            item.add_marker(pytest.mark.database)

# Test timeout configuration
def pytest_runtest_setup(item):
    """Setup test timeouts."""
    if "slow" in item.keywords:
        # Slow tests get 5 minutes
        pytest.timeout = 300
    elif "performance" in item.keywords:
        # Performance tests get 2 minutes
        pytest.timeout = 120
    else:
        # Regular tests get 30 seconds
        pytest.timeout = 30

# Coverage configuration
def pytest_configure(config):
    """Configure test coverage."""
    if config.getoption("--cov"):
        config.option.cov_source = [
            "backend/app",
            "backend/ai",
            "backend/strategies",
            "backend/services",
            "backend/core"
        ]
        config.option.cov_report = ["term", "html", "xml"]
        config.option.cov_fail_under = 80

# Test result reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom test result summary."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "All tests passed! 🎉")
    else:
        terminalreporter.write_sep("=", "Some tests failed! ❌")
    
    # Show test statistics
    stats = terminalreporter.stats
    if stats:
        terminalreporter.write_line("\nTest Statistics:")
        for key, value in stats.items():
            terminalreporter.write_line(f"  {key}: {len(value)}")
