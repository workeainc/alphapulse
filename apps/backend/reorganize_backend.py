"""
Backend Reorganization Script
Reorganizes apps/backend to follow proper monorepo structure
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple

# Get backend path
BACKEND_PATH = Path(__file__).parent

# Define source code folders to move to src/
SOURCE_FOLDERS = [
    'app',
    'ai', 
    'strategies',
    'services',
    'database',
    'data',
    'core',
    'routes',
    'monitoring',
    'risk',
    'streaming',
    'tracking',
    'utils',
    'outcome_tracking',
    'performance',
    'visualization'
]

# Test files at root to move to tests/
TEST_FILES_AT_ROOT = [
    'test_*.py',
    'check_*.py',
    'verify_*.py',
    'simple_*.py',
    'direct_*.py',
    'integration_*.py',
    'final_*.py',
    'real_dependencies_test.py',
    'show_results.py',
    'safe_exchange_config.py'
]

# Script files at root to move to scripts/
SCRIPT_FILES_AT_ROOT = [
    'run_*.py',
    'setup_*.py',
    'fix_*.py',
    'start_*.py',
    'monitor_*.py',
    'optimize_*.py',
    'integration_guide.py',
    'working_server.py'
]

# Folders to delete (archives, temp, cache)
FOLDERS_TO_DELETE = [
    'archive_redundant_files',
    'archived',
    'archives',
    'backup_before_reorganization',
    'cache',
    'catboost_info',
    'htmlcov',
    'logs',
    'results',
    'deployments',
    'artifacts',
    'ml_models'
]

# Files to delete (reports, temp files)
FILES_TO_DELETE = [
    '*.log',
    '*_report_*.json',
    'alphapulse.log'
]

# Documentation files to keep at root
DOCS_TO_KEEP = ['README.md']

# Documentation files to move to /docs
DOCS_TO_MOVE = [
    '*.md'  # All markdown files except README.md
]

def create_src_structure():
    """Create the new src/ directory structure"""
    print("Creating src/ directory structure...")
    src_path = BACKEND_PATH / 'src'
    src_path.mkdir(exist_ok=True)
    
    # Create __init__.py in src/
    (src_path / '__init__.py').write_text('')
    print("[OK] Created src/ directory")

def move_source_folders():
    """Move source code folders to src/"""
    print("\nMoving source code folders to src/...")
    src_path = BACKEND_PATH / 'src'
    
    for folder in SOURCE_FOLDERS:
        folder_path = BACKEND_PATH / folder
        if folder_path.exists() and folder_path.is_dir():
            dest_path = src_path / folder
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.move(str(folder_path), str(dest_path))
            print(f"  [OK] Moved {folder}/ to src/{folder}/")

def consolidate_tests():
    """Consolidate all test files into tests/ folder"""
    print("\nConsolidating tests...")
    tests_path = BACKEND_PATH / 'tests'
    
    # Tests folder already exists, but let's organize it better
    if not tests_path.exists():
        tests_path.mkdir()
    
    # Move test files from root
    for pattern in TEST_FILES_AT_ROOT:
        for file_path in BACKEND_PATH.glob(pattern):
            if file_path.is_file() and file_path.parent == BACKEND_PATH:
                dest_path = tests_path / file_path.name
                if dest_path.exists():
                    dest_path.unlink()
                shutil.move(str(file_path), str(dest_path))
                print(f"  [OK] Moved {file_path.name} to tests/")
    
    # Move test report JSON files to tests/reports
    reports_path = tests_path / 'reports'
    reports_path.mkdir(exist_ok=True)
    for file_path in BACKEND_PATH.glob('*_report_*.json'):
        if file_path.is_file():
            dest_path = reports_path / file_path.name
            if dest_path.exists():
                dest_path.unlink()
            shutil.move(str(file_path), str(dest_path))
            print(f"  [OK] Moved {file_path.name} to tests/reports/")

def consolidate_scripts():
    """Consolidate all script files into scripts/ folder"""
    print("\nConsolidating scripts...")
    scripts_path = BACKEND_PATH / 'scripts'
    scripts_path.mkdir(exist_ok=True)
    
    # Move script files from root
    for pattern in SCRIPT_FILES_AT_ROOT:
        for file_path in BACKEND_PATH.glob(pattern):
            if file_path.is_file() and file_path.parent == BACKEND_PATH:
                dest_path = scripts_path / file_path.name
                if dest_path.exists():
                    dest_path.unlink()
                shutil.move(str(file_path), str(dest_path))
                print(f"  [OK] Moved {file_path.name} to scripts/")

def delete_temporary_folders():
    """Delete temporary, cache, and archive folders"""
    print("\nDeleting temporary and archive folders...")
    
    for folder in FOLDERS_TO_DELETE:
        folder_path = BACKEND_PATH / folder
        if folder_path.exists() and folder_path.is_dir():
            try:
                shutil.rmtree(folder_path)
                print(f"  [OK] Deleted {folder}/")
            except Exception as e:
                print(f"  [ERROR] Could not delete {folder}/: {e}")
    
    # Delete log files
    for file_path in BACKEND_PATH.glob('*.log'):
        try:
            file_path.unlink()
            print(f"  [OK] Deleted {file_path.name}")
        except Exception as e:
            print(f"  [ERROR] Could not delete {file_path.name}: {e}")

def cleanup_docs():
    """Move documentation files to appropriate location"""
    print("\nCleaning up documentation...")
    
    # Get all .md files except README.md
    for md_file in BACKEND_PATH.glob('*.md'):
        if md_file.name != 'README.md':
            # These should be moved to root /docs or deleted
            print(f"  [INFO] Documentation file found: {md_file.name} (manual review needed)")

def update_imports_in_file(file_path: Path, dry_run: bool = False) -> int:
    """Update imports in a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        changes = 0
        
        # Pattern 1: from app. -> from src.app.
        # Pattern 2: from ai. -> from src.ai.
        # etc.
        for module in ['app', 'ai', 'strategies', 'services', 'database', 'data', 
                       'core', 'routes', 'monitoring', 'risk', 'streaming', 'tracking',
                       'utils', 'outcome_tracking', 'performance', 'visualization']:
            # Absolute imports
            pattern1 = f'from {module}\\.'
            replacement1 = f'from src.{module}.'
            new_content = re.sub(pattern1, replacement1, content)
            if new_content != content:
                changes += content.count(f'from {module}.')
                content = new_content
            
            # Import statements
            pattern2 = f'from {module} import'
            replacement2 = f'from src.{module} import'
            new_content = re.sub(pattern2, replacement2, content)
            if new_content != content:
                changes += content.count(f'from {module} import')
                content = new_content
            
            # Relative imports: from ..app -> from ..src.app
            pattern3 = f'from \\.\\.{module}'
            replacement3 = f'from ..src.{module}'
            new_content = re.sub(pattern3, replacement3, content)
            if new_content != content:
                changes += content.count(f'from ..{module}')
                content = new_content
            
            # Relative imports: from .app -> from .src.app (only in __init__ files)
            if file_path.name == '__init__.py':
                pattern4 = f'from \\.{module}'
                replacement4 = f'from .src.{module}'
                new_content = re.sub(pattern4, replacement4, content)
                if new_content != content:
                    changes += content.count(f'from .{module}')
                    content = new_content
        
        # Write back if changed
        if content != original_content and not dry_run:
            file_path.write_text(content, encoding='utf-8')
        
        return changes
    except Exception as e:
        print(f"    [ERROR] Error updating {file_path}: {e}")
        return 0

def update_all_imports():
    """Update all imports to use src. prefix"""
    print("\nUpdating imports to use src. prefix...")
    
    total_files = 0
    total_changes = 0
    
    # Update imports in src/ folder
    src_path = BACKEND_PATH / 'src'
    if src_path.exists():
        for py_file in src_path.rglob('*.py'):
            changes = update_imports_in_file(py_file)
            if changes > 0:
                total_files += 1
                total_changes += changes
                print(f"  [OK] Updated {py_file.relative_to(BACKEND_PATH)} ({changes} imports)")
    
    # Update imports in tests/ folder
    tests_path = BACKEND_PATH / 'tests'
    if tests_path.exists():
        for py_file in tests_path.rglob('*.py'):
            changes = update_imports_in_file(py_file)
            if changes > 0:
                total_files += 1
                total_changes += changes
                print(f"  [OK] Updated {py_file.relative_to(BACKEND_PATH)} ({changes} imports)")
    
    # Update imports in scripts/ folder  
    scripts_path = BACKEND_PATH / 'scripts'
    if scripts_path.exists():
        for py_file in scripts_path.rglob('*.py'):
            changes = update_imports_in_file(py_file)
            if changes > 0:
                total_files += 1
                total_changes += changes
                print(f"  [OK] Updated {py_file.relative_to(BACKEND_PATH)} ({changes} imports)")
    
    # Update main.py
    main_file = BACKEND_PATH / 'main.py'
    if main_file.exists():
        changes = update_imports_in_file(main_file)
        if changes > 0:
            total_files += 1
            total_changes += changes
            print(f"  [OK] Updated main.py ({changes} imports)")
    
    print(f"\n  Total: Updated {total_changes} imports in {total_files} files")

def update_config_files():
    """Update configuration files"""
    print("\nUpdating configuration files...")
    
    # Update pyproject.toml
    pyproject_path = BACKEND_PATH / 'pyproject.toml'
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        # Update packages line
        content = content.replace(
            'packages = ["app", "ai", "strategies", "database", "tracking", "data", "services"]',
            'packages = ["src"]'
        )
        # Update coverage paths
        content = content.replace(
            'addopts = "-v --cov=app --cov=ai --cov=strategies --cov-report=html --cov-report=term"',
            'addopts = "-v --cov=src --cov-report=html --cov-report=term"'
        )
        # Update lint paths
        content = content.replace(
            'lint = "flake8 app/ ai/ strategies/ database/"',
            'lint = "flake8 src/"'
        )
        content = content.replace(
            'lint:fix = "black app/ ai/ strategies/ database/ && isort app/ ai/ strategies/ database/"',
            'lint:fix = "black src/ && isort src/"'
        )
        pyproject_path.write_text(content)
        print("  [OK] Updated pyproject.toml")
    
    # Update package.json
    package_json_path = BACKEND_PATH / 'package.json'
    if package_json_path.exists():
        content = package_json_path.read_text()
        # Update lint and test paths
        content = content.replace(
            '"lint": "flake8 app/ ai/ strategies/ database/"',
            '"lint": "flake8 src/"'
        )
        content = content.replace(
            '"lint:fix": "black app/ ai/ strategies/ database/ && isort app/ ai/ strategies/ database/"',
            '"lint:fix": "black src/ && isort src/"'
        )
        content = content.replace(
            '"test:coverage": "pytest tests/ -v --cov=app --cov=ai --cov=strategies --cov-report=html --cov-report=term"',
            '"test:coverage": "pytest tests/ -v --cov=src --cov-report=html --cov-report=term"'
        )
        package_json_path.write_text(content)
        print("  [OK] Updated package.json")

def create_summary():
    """Create a summary of the reorganization"""
    print("\n" + "="*60)
    print("REORGANIZATION COMPLETE!")
    print("="*60)
    print("\nNew Structure:")
    print("  apps/backend/")
    print("  ├── src/              # All source code")
    print("  ├── tests/            # All tests")
    print("  ├── scripts/          # Development scripts")
    print("  ├── migrations/       # DB migrations")
    print("  ├── config/           # Configuration")
    print("  ├── main.py           # Entry point")
    print("  └── package.json      # Package config")
    print("\n[OK] Source folders moved to src/")
    print("[OK] Tests consolidated in tests/")
    print("[OK] Scripts consolidated in scripts/")
    print("[OK] Imports updated to use src. prefix")
    print("[OK] Configuration files updated")
    print("[OK] Temporary files cleaned up")
    print("\nNext steps:")
    print("1. Test the application: python main.py")
    print("2. Run tests: pytest tests/ -v")
    print("3. Review any remaining .md files")

def main():
    """Main reorganization function"""
    print("="*60)
    print("BACKEND REORGANIZATION SCRIPT")
    print("="*60)
    print(f"\nWorking directory: {BACKEND_PATH}")
    print("\nThis script will:")
    print("  1. Create src/ directory structure")
    print("  2. Move source code folders to src/")
    print("  3. Consolidate tests into tests/")
    print("  4. Consolidate scripts into scripts/")
    print("  5. Update all imports to use src. prefix")
    print("  6. Update configuration files")
    print("  7. Delete temporary/archive folders")
    print("  8. Cleanup documentation")
    
    response = input("\nProceed with reorganization? (yes/no): ")
    if response.lower() != 'yes':
        print("Reorganization cancelled.")
        return
    
    try:
        # Step 1: Create structure
        create_src_structure()
        
        # Step 2: Move source folders
        move_source_folders()
        
        # Step 3: Consolidate tests
        consolidate_tests()
        
        # Step 4: Consolidate scripts
        consolidate_scripts()
        
        # Step 5: Update imports
        update_all_imports()
        
        # Step 6: Update config files
        update_config_files()
        
        # Step 7: Delete temporary folders
        delete_temporary_folders()
        
        # Step 8: Cleanup docs
        cleanup_docs()
        
        # Summary
        create_summary()
        
    except Exception as e:
        print(f"\n[ERROR] Error during reorganization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

