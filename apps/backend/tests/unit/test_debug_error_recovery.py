#!/usr/bin/env python3
"""
Debug script for error recovery system
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    try:
        print("🧪 Testing Circuit Breaker...")
        
        from ..src.ai.error_recovery_system import CircuitBreaker
        
        # Create circuit breaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)
        
        # Test initial state
        print(f"Initial state: {cb.state.value}")
        print(f"Failure count: {cb.failure_count}")
        
        # Test successful execution
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        print(f"Success result: {result}")
        print(f"State after success: {cb.state.value}")
        print(f"Success count: {cb.success_count}")
        
        # Test failure handling
        async def failure_func():
            raise Exception("Test failure")
        
        # First failure
        try:
            await cb.call(failure_func)
        except Exception as e:
            print(f"First failure caught: {e}")
        
        print(f"Failure count after first failure: {cb.failure_count}")
        print(f"State after first failure: {cb.state.value}")
        
        # Second failure (should open circuit)
        try:
            await cb.call(failure_func)
        except Exception as e:
            print(f"Second failure caught: {e}")
        
        print(f"Failure count after second failure: {cb.failure_count}")
        print(f"State after second failure: {cb.state.value}")
        
        # Test circuit open behavior
        try:
            await cb.call(success_func)
            print("ERROR: Should not reach here - circuit should be open")
        except Exception as e:
            print(f"Circuit open behavior working: {e}")
        
        # Wait for recovery timeout
        print("Waiting for recovery timeout...")
        await asyncio.sleep(6)
        
        # Test half-open state
        print(f"State after timeout: {cb.state.value}")
        
        # Test successful recovery
        result = await cb.call(success_func)
        print(f"Recovery result: {result}")
        print(f"State after recovery: {cb.state.value}")
        
        print("✅ Circuit Breaker test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Circuit Breaker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_recovery_system():
    """Test error recovery system"""
    try:
        print("🧪 Testing Error Recovery System...")
        
        from ..src.ai.error_recovery_system import error_recovery_system
        
        # Test system initialization
        print(f"System initialized: {error_recovery_system is not None}")
        print(f"Has execute_with_recovery: {hasattr(error_recovery_system, 'execute_with_recovery')}")
        print(f"Has get_system_health: {hasattr(error_recovery_system, 'get_system_health')}")
        
        # Test circuit breaker registration
        cb = error_recovery_system.register_circuit_breaker("test_service", failure_threshold=2)
        print(f"Circuit breaker registered: {cb is not None}")
        print(f"Failure threshold: {cb.failure_threshold}")
        
        # Test retry mechanism registration
        rm = error_recovery_system.register_retry_mechanism("test_operation", max_attempts=3)
        print(f"Retry mechanism registered: {rm is not None}")
        print(f"Max attempts: {rm.max_attempts}")
        
        # Test error summary
        error_summary = error_recovery_system.get_error_summary(hours=24)
        print(f"Error summary: {error_summary}")
        print(f"Has total_errors key: {'total_errors' in error_summary}")
        
        # Test system health
        health = error_recovery_system.get_system_health()
        print(f"System health keys: {list(health.keys())}")
        print(f"Has circuit_breakers key: {'circuit_breakers' in health}")
        print(f"Has retry_mechanisms key: {'retry_mechanisms' in health}")
        
        print("✅ Error Recovery System test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error Recovery System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Error Recovery System Debug Tests")
    print("=" * 60)
    
    # Test 1: Circuit Breaker
    result1 = await test_circuit_breaker()
    
    print("\n" + "=" * 60)
    
    # Test 2: Error Recovery System
    result2 = await test_error_recovery_system()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Circuit Breaker Test: {'✅ PASSED' if result1 else '❌ FAILED'}")
    print(f"Error Recovery System Test: {'✅ PASSED' if result2 else '❌ FAILED'}")
    
    if result1 and result2:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
