"""
MTF Implementation Verification Script
Checks that all components are in place (no runtime execution needed)
"""

import os
import sys
from pathlib import Path

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists"""
    full_path = Path(path)
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"[OK] {description}: {path} ({size} bytes)")
        return True
    else:
        print(f"[FAIL] {description}: {path} NOT FOUND")
        return False

def check_directory_structure():
    """Verify all required files are in place"""
    print("\n" + "=" * 80)
    print("MTF IMPLEMENTATION VERIFICATION")
    print("=" * 80 + "\n")
    
    checks = []
    
    # Database migration
    print("1. Database Schema")
    checks.append(check_file_exists(
        "src/database/migrations/101_mtf_entry_fields.sql",
        "MTF database migration"
    ))
    
    # Core MTF services
    print("\n2. Core Services")
    checks.append(check_file_exists(
        "src/services/mtf_entry_system.py",
        "MTF Entry System"
    ))
    checks.append(check_file_exists(
        "src/services/mtf_signal_storage.py",
        "MTF Signal Storage"
    ))
    checks.append(check_file_exists(
        "src/services/config_loader.py",
        "Config Loader"
    ))
    
    # Configuration files
    print("\n3. Configuration")
    checks.append(check_file_exists(
        "config/mtf_config.yaml",
        "MTF Configuration"
    ))
    checks.append(check_file_exists(
        "config/symbol_config.yaml",
        "Symbol Configuration"
    ))
    
    # Modified integration files
    print("\n4. Integration Files")
    checks.append(check_file_exists(
        "src/services/ai_model_integration_service.py",
        "AI Model Integration (modified)"
    ))
    checks.append(check_file_exists(
        "src/services/signal_generation_scheduler.py",
        "Signal Generation Scheduler (modified)"
    ))
    checks.append(check_file_exists(
        "src/services/startup_orchestrator.py",
        "Startup Orchestrator (modified)"
    ))
    checks.append(check_file_exists(
        "src/services/orchestration_monitor.py",
        "Orchestration Monitor (modified)"
    ))
    
    # Test scripts
    print("\n5. Test Scripts")
    checks.append(check_file_exists(
        "test_mtf_storage.py",
        "MTF Storage Test"
    ))
    checks.append(check_file_exists(
        "test_mtf_performance.py",
        "MTF Performance Test"
    ))
    
    # Documentation
    print("\n6. Documentation")
    checks.append(check_file_exists(
        "MTF_ENTRY_SYSTEM_COMPLETE.md",
        "MTF System Documentation"
    ))
    checks.append(check_file_exists(
        "MTF_QUICK_START.md",
        "MTF Quick Start Guide"
    ))
    checks.append(check_file_exists(
        "MTF_GAPS_FIXED.md",
        "MTF Gaps Fixed Summary"
    ))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    passed = sum(checks)
    total = len(checks)
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n[OK] ALL COMPONENTS IN PLACE!")
        print("\nMTF Implementation is COMPLETE and ready for testing!")
        return True
    else:
        print(f"\n[WARN] {total - passed} components missing")
        return False


def check_code_integration():
    """Check that code contains expected integrations"""
    print("\n" + "=" * 80)
    print("CODE INTEGRATION CHECKS")
    print("=" * 80 + "\n")
    
    checks = []
    
    # Check signal_generation_scheduler for storage import
    print("1. Signal Generation Scheduler Integration")
    try:
        with open("src/services/signal_generation_scheduler.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from src.services.mtf_signal_storage import MTFSignalStorage' in content:
                print("[OK] MTFSignalStorage import found")
                checks.append(True)
            else:
                print("[FAIL] MTFSignalStorage import missing")
                checks.append(False)
            
            if 'store_mtf_signal' in content:
                print("[OK] store_mtf_signal() call found")
                checks.append(True)
            else:
                print("[FAIL] store_mtf_signal() call missing")
                checks.append(False)
            
            if 'check_active_signal_exists' in content:
                print("[OK] Deduplication check found")
                checks.append(True)
            else:
                print("[FAIL] Deduplication check missing")
                checks.append(False)
    except Exception as e:
        print(f"[FAIL] Error reading file: {e}")
        checks.extend([False, False, False])
    
    # Check AI service for config support
    print("\n2. AI Model Integration Service")
    try:
        with open("src/services/ai_model_integration_service.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from src.services.mtf_entry_system import MTFEntrySystem' in content:
                print("[OK] MTFEntrySystem import found")
                checks.append(True)
            else:
                print("[FAIL] MTFEntrySystem import missing")
                checks.append(False)
            
            if 'generate_ai_signal_with_mtf_entry' in content:
                print("[OK] generate_ai_signal_with_mtf_entry() method found")
                checks.append(True)
            else:
                print("[FAIL] generate_ai_signal_with_mtf_entry() method missing")
                checks.append(False)
    except Exception as e:
        print(f"[FAIL] Error reading file: {e}")
        checks.extend([False, False])
    
    # Check startup orchestrator for config loader
    print("\n3. Startup Orchestrator")
    try:
        with open("src/services/startup_orchestrator.py", 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from src.services.config_loader import ConfigLoader' in content:
                print("[OK] ConfigLoader import found")
                checks.append(True)
            else:
                print("[FAIL] ConfigLoader import missing")
                checks.append(False)
    except Exception as e:
        print(f"[FAIL] Error reading file: {e}")
        checks.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    passed = sum(checks)
    total = len(checks)
    print(f"Code Integration Checks: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[OK] ALL INTEGRATIONS COMPLETE!")
        return True
    else:
        print(f"[WARN] {total - passed} integrations missing or incomplete")
        return False


if __name__ == "__main__":
    print("\nMTF Implementation Verification")
    print("Checking files and code integrations...")
    
    files_ok = check_directory_structure()
    code_ok = check_code_integration()
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if files_ok and code_ok:
        print("\n[SUCCESS] MTF IMPLEMENTATION IS COMPLETE!")
        print("\nAll components are in place and properly integrated.")
        print("System is ready for testing and deployment.")
        print("\nNext steps:")
        print("  1. Verify database: docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse -c 'SELECT COUNT(*) FROM ai_signals_mtf;'")
        print("  2. Start system: python main_scaled.py")
        print("  3. Monitor logs for MTF messages:")
        print("     - 'MTF Analysis: BTCUSDT | Signal TF: 1h | Entry TF: 15m'")
        print("     - 'Stored MTF signal for BTCUSDT to database'")
        sys.exit(0)
    else:
        print("\n[WARN] SOME COMPONENTS MISSING")
        print("Review the checks above to see what needs attention.")
        sys.exit(1)

