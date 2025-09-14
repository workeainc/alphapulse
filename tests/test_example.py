"""
Example test file for AlphaPulse backend
"""
import pytest
from fastapi import FastAPI


class TestExample:
    """Example test class"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        assert True is True
        assert 1 + 1 == 2
    
    def test_fastapi_import(self):
        """Test FastAPI import"""
        app = FastAPI()
        assert app is not None
        assert hasattr(app, 'get')
        assert hasattr(app, 'post')
    
    def test_string_operations(self):
        """Test string operations"""
        text = "AlphaPulse"
        assert len(text) == 10
        assert "Alpha" in text
        assert text.lower() == "alphapulse"


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality"""
    result = await async_operation()
    assert result == "success"


async def async_operation():
    """Simple async operation for testing"""
    return "success"


def test_list_operations():
    """Test list operations"""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5
    assert min(numbers) == 1
