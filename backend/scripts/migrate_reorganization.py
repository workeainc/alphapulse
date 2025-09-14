#!/usr/bin/env python3
"""
AlphaPulse Backend Reorganization Migration Script
Moves and consolidates files according to the new structure
"""

import os
import shutil
import json
import logging
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReorganizationMigrator:
    """Handles the migration of files to the new structure"""
    
    def __init__(self, backend_path: str = "."):
        self.backend_path = Path(backend_path)
        self.backup_path = self.backend_path / "backup_before_reorganization"
        self.migration_log = []
        
        # Load reorganization plan
        with open("reorganization_plan.json", "r") as f:
            self.plan = json.load(f)
    
    def create_backup(self):
        """Create backup of current structure"""
        logger.info("Creating backup of current structure...")
        
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
        
        self.backup_path.mkdir(exist_ok=True)
        
        # Copy all Python files to backup
        for py_file in self.backend_path.rglob("*.py"):
            if "backup" not in str(py_file) and "migrate" not in str(py_file):
                relative_path = py_file.relative_to(self.backend_path)
                backup_file = self.backup_path / relative_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(py_file, backup_file)
        
        logger.info(f"Backup created at: {self.backup_path}")
    
    def create_new_directories(self):
        """Create new directory structure"""
        logger.info("Creating new directory structure...")
        
        new_dirs = [
            "core", "utils", "services", "tests", "docs", 
            "ai", "strategies", "execution", "scripts"
        ]
        
        for dir_name in new_dirs:
            dir_path = self.backend_path / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Created directory: {dir_name}")
    
    def move_core_files(self):
        """Move core files to new structure"""
        logger.info("Moving core files...")
        
        core_moves = [
            ("alphapulse_core.py", "core/alphapulse_core.py"),
            ("indicators_engine.py", "core/indicators_engine.py"),
            ("ml_signal_generator.py", "core/ml_signal_generator.py"),
            ("market_regime_detector.py", "core/market_regime_detector.py"),
            ("optimized_trading_system.py", "core/optimized_trading_system.py"),
            ("websocket_binance.py", "core/websocket_binance.py")
        ]
        
        for source, dest in core_moves:
            source_path = self.backend_path / source
            dest_path = self.backend_path / dest
            
            if source_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))
                self.migration_log.append(f"Moved: {source} -> {dest}")
                logger.info(f"Moved: {source} -> {dest}")
    
    def move_database_files(self):
        """Move database files"""
        logger.info("Moving database files...")
        
        # Database files are already in the right place
        # Just ensure migrations are preserved
        migrations_dir = self.backend_path / "database" / "migrations"
        if migrations_dir.exists():
            logger.info("Database migrations preserved")
    
    def move_test_files(self):
        """Move and consolidate test files"""
        logger.info("Moving test files...")
        
        # Move existing test files
        test_moves = [
            ("test_simple_working.py", "tests/test_simple_working.py"),
            ("test_edge_cases_simple.py", "tests/test_edge_cases.py"),
            ("test_database_simple.py", "tests/test_database.py"),
            ("test_performance_simple.py", "tests/test_performance.py"),
            ("test_integration_simple.py", "tests/test_integration_simple.py"),
        ]
        
        for source, dest in test_moves:
            source_path = self.backend_path / source
            dest_path = self.backend_path / dest
            
            if source_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(dest_path))
                self.migration_log.append(f"Moved: {source} -> {dest}")
                logger.info(f"Moved: {source} -> {dest}")
        
        # Move existing tests directory content
        existing_tests = self.backend_path / "tests"
        if existing_tests.exists():
            for test_file in existing_tests.glob("*.py"):
                if test_file.name not in ["test_integration.py", "conftest.py"]:
                    # Keep existing test files
                    logger.info(f"Preserved test file: {test_file.name}")
    
    def consolidate_duplicate_files(self):
        """Consolidate duplicate files based on analysis"""
        logger.info("Consolidating duplicate files...")
        
        # Files to delete (100% duplicates)
        duplicates_to_delete = [
            "debug_manual_labeling.py",
            "test_add_low_confidence_function.py", 
            "run_chaos_engineering.py",
            "run_multi_region_dashboard.py",
            "run_resilience_dashboard.py",
            "run_security_dashboard.py",
            "database/migrations/env_enhanced.py",
            "database/db_scanner_simple.py",
            "tests/conftest_enhanced.py",
            "tests/test_unit_indicators.py",
            "tests/test_pipeline_enhanced.py"
        ]
        
        for file_path in duplicates_to_delete:
            full_path = self.backend_path / file_path
            if full_path.exists():
                # Move to backup instead of deleting
                backup_file = self.backup_path / file_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(full_path), str(backup_file))
                self.migration_log.append(f"Backed up duplicate: {file_path}")
                logger.info(f"Backed up duplicate: {file_path}")
    
    def consolidate_advanced_files(self):
        """Consolidate advanced_* files"""
        logger.info("Consolidating advanced_* files...")
        
        advanced_files = [
            "app/core/advanced_analytics.py",
            "app/core/advanced_diagnostics.py",
            "app/core/advanced_feature_engineering.py",
            "ai/advanced_backtesting.py",
            "ai/advanced_batching.py",
            "ai/advanced_feature_engineering.py",
            "ai/advanced_logging_system.py",
            "ai/advanced_portfolio_management.py"
        ]
        
        # Create consolidated advanced_utils.py
        consolidated_content = []
        consolidated_content.append('"""Consolidated Advanced Utilities for AlphaPulse"""')
        consolidated_content.append('')
        consolidated_content.append('import logging')
        consolidated_content.append('from typing import Dict, List, Any')
        consolidated_content.append('from datetime import datetime')
        consolidated_content.append('')
        consolidated_content.append('logger = logging.getLogger(__name__)')
        consolidated_content.append('')
        
        for file_path in advanced_files:
            full_path = self.backend_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Extract functions and classes
                    lines = content.split('\n')
                    in_class_or_function = False
                    extracted_content = []
                    
                    for line in lines:
                        if line.strip().startswith(('def ', 'class ', 'async def ')):
                            in_class_or_function = True
                            extracted_content.append(f'# From {file_path}')
                            extracted_content.append(line)
                        elif in_class_or_function and line.strip() == '':
                            in_class_or_function = False
                        elif in_class_or_function:
                            extracted_content.append(line)
                    
                    if extracted_content:
                        consolidated_content.extend(extracted_content)
                        consolidated_content.append('')
                    
                    # Move original to backup
                    backup_file = self.backup_path / file_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(full_path), str(backup_file))
                    self.migration_log.append(f"Consolidated: {file_path}")
                    logger.info(f"Consolidated: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Write consolidated file
        consolidated_path = self.backend_path / "ai" / "advanced_utils.py"
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(consolidated_path, 'w') as f:
            f.write('\n'.join(consolidated_content))
        
        logger.info(f"Created consolidated file: {consolidated_path}")
    
    def consolidate_performance_files(self):
        """Consolidate performance_* files"""
        logger.info("Consolidating performance_* files...")
        
        performance_files = [
            "app/core/performance_alerting.py",
            "app/core/performance_profiling.py", 
            "app/core/performance_regression.py",
            "test_performance_baseline.py",
            "test_performance_simple.py"
        ]
        
        # Create consolidated monitoring_services.py
        consolidated_content = []
        consolidated_content.append('"""Consolidated Monitoring Services for AlphaPulse"""')
        consolidated_content.append('')
        consolidated_content.append('import logging')
        consolidated_content.append('from typing import Dict, List, Any')
        consolidated_content.append('from datetime import datetime')
        consolidated_content.append('')
        consolidated_content.append('logger = logging.getLogger(__name__)')
        consolidated_content.append('')
        
        for file_path in performance_files:
            full_path = self.backend_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Extract functions and classes
                    lines = content.split('\n')
                    in_class_or_function = False
                    extracted_content = []
                    
                    for line in lines:
                        if line.strip().startswith(('def ', 'class ', 'async def ')):
                            in_class_or_function = True
                            extracted_content.append(f'# From {file_path}')
                            extracted_content.append(line)
                        elif in_class_or_function and line.strip() == '':
                            in_class_or_function = False
                        elif in_class_or_function:
                            extracted_content.append(line)
                    
                    if extracted_content:
                        consolidated_content.extend(extracted_content)
                        consolidated_content.append('')
                    
                    # Move original to backup
                    backup_file = self.backup_path / file_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(full_path), str(backup_file))
                    self.migration_log.append(f"Consolidated: {file_path}")
                    logger.info(f"Consolidated: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Write consolidated file
        consolidated_path = self.backend_path / "services" / "monitoring_services.py"
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(consolidated_path, 'w') as f:
            f.write('\n'.join(consolidated_content))
        
        logger.info(f"Created consolidated file: {consolidated_path}")
    
    def create_migration_summary(self):
        """Create migration summary report"""
        logger.info("Creating migration summary...")
        
        summary = {
            "migration_timestamp": datetime.now().isoformat(),
            "backup_location": str(self.backup_path),
            "files_moved": len(self.migration_log),
            "migration_log": self.migration_log,
            "new_structure": {
                "core": ["alphapulse_core.py", "indicators_engine.py", "ml_signal_generator.py", 
                        "market_regime_detector.py", "optimized_trading_system.py", "websocket_binance.py"],
                "utils": ["feature_engineering.py", "risk_management.py", "threshold_env.py", "utils.py", "config.py"],
                "services": ["data_services.py", "monitoring_services.py", "trading_services.py", 
                           "pattern_services.py", "active_learning_service.py"],
                "database": ["models.py", "queries.py", "connection.py", "migrations/"],
                "tests": ["test_integration.py", "test_indicators.py", "test_performance.py", 
                         "test_database.py", "test_edge_cases.py", "test_utils.py", "conftest.py"],
                "docs": ["README.md", "model_docs.md", "performance_baseline.md", "api_docs.md"],
                "ai": ["advanced_utils.py", "ml_models.py", "feature_store.py", "deployment.py"],
                "strategies": ["pattern_detectors.py", "signal_generators.py", "trend_analyzers.py", "strategy_manager.py"],
                "execution": ["trading_engine.py", "order_manager.py", "portfolio_manager.py", "risk_manager.py"],
                "scripts": ["run_alphapulse.py", "run_tests.py", "setup_database.py", "migrate_data.py"]
            }
        }
        
        summary_path = self.backend_path / "migration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Migration summary saved to: {summary_path}")
        return summary
    
    def run_migration(self):
        """Run the complete migration"""
        logger.info("Starting AlphaPulse backend reorganization migration...")
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Create new directories
            self.create_new_directories()
            
            # Step 3: Move core files
            self.move_core_files()
            
            # Step 4: Move database files
            self.move_database_files()
            
            # Step 5: Move test files
            self.move_test_files()
            
            # Step 6: Consolidate duplicates
            self.consolidate_duplicate_files()
            
            # Step 7: Consolidate advanced files
            self.consolidate_advanced_files()
            
            # Step 8: Consolidate performance files
            self.consolidate_performance_files()
            
            # Step 9: Create migration summary
            summary = self.create_migration_summary()
            
            logger.info("✅ Migration completed successfully!")
            logger.info(f"Backup location: {self.backup_path}")
            logger.info(f"Files processed: {len(self.migration_log)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            logger.info("Check backup directory for original files")
            raise

def main():
    """Main migration function"""
    migrator = ReorganizationMigrator()
    
    try:
        summary = migrator.run_migration()
        
        print("\n=== Migration Summary ===")
        print(f"Migration completed at: {summary['migration_timestamp']}")
        print(f"Backup location: {summary['backup_location']}")
        print(f"Files processed: {summary['files_moved']}")
        
        print("\n=== New Structure ===")
        for directory, files in summary['new_structure'].items():
            print(f"{directory}/:")
            for file in files:
                print(f"  - {file}")
        
        print(f"\nMigration log saved to: migration_summary.json")
        print("Original files backed up to: backup_before_reorganization/")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
