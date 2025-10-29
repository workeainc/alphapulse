#!/usr/bin/env python3
"""
AlphaPulse Backend Reorganization Plan
Consolidates duplicates and organizes files into logical folders
"""

import os
import shutil
import json
import logging
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileMove:
    """Represents a file move operation"""
    source: str
    destination: str
    operation: str  # 'move', 'merge', 'copy'
    reason: str

@dataclass
class ConsolidationPlan:
    """Complete reorganization plan"""
    new_structure: Dict[str, List[str]]
    file_moves: List[FileMove]
    duplicates_to_merge: List[Tuple[str, str]]
    files_to_delete: List[str]
    new_files_to_create: List[str]

class ReorganizationPlanner:
    """Plans and executes the reorganization"""
    
    def __init__(self, backend_path: str = "."):
        self.backend_path = Path(backend_path)
        self.plan = ConsolidationPlan(
            new_structure={},
            file_moves=[],
            duplicates_to_merge=[],
            files_to_delete=[],
            new_files_to_create=[]
        )
        
        # Load analysis results
        with open("reorganization_analysis_report.json", "r") as f:
            self.analysis = json.load(f)
    
    def create_new_structure(self):
        """Define the new directory structure"""
        self.plan.new_structure = {
            "core/": [
                "alphapulse_core.py",
                "indicators_engine.py", 
                "ml_signal_generator.py",
                "market_regime_detector.py",
                "optimized_trading_system.py",
                "websocket_binance.py"
            ],
            "utils/": [
                "feature_engineering.py",  # Merge all feature_*.py
                "risk_management.py",
                "threshold_env.py", 
                "utils.py",  # General helpers
                "config.py"  # Unified config
            ],
            "services/": [
                "data_services.py",  # Merge data_*.py
                "monitoring_services.py",  # Merge monitoring_*.py, performance_*.py
                "trading_services.py",  # Merge trading_*.py
                "pattern_services.py",  # Merge pattern_*.py
                "active_learning_service.py"
            ],
            "database/": [
                "models.py",  # Consolidated models with docstrings
                "queries.py",
                "connection.py",
                "migrations/"  # Keep all migrations
            ],
            "tests/": [
                "test_indicators.py",  # Merge indicator tests
                "test_integration.py",  # Merge phase*_integration.py
                "test_performance.py",
                "test_database.py", 
                "test_edge_cases.py",
                "test_utils.py",  # Common mocks
                "conftest.py"  # Unified fixtures
            ],
            "docs/": [
                "README.md",  # Overview
                "model_docs.md",  # Model docstrings
                "performance_baseline.md",  # Merge summaries
                "api_docs.md"
            ],
            "ai/": [
                "advanced_utils.py",  # Merge advanced_*.py
                "ml_models.py",  # Merge ml_*.py
                "feature_store.py",  # Merge feature_*.py
                "deployment.py"  # Merge deployment_*.py
            ],
            "strategies/": [
                "pattern_detectors.py",  # Merge pattern_*.py
                "signal_generators.py",  # Merge signal_*.py
                "trend_analyzers.py",  # Merge trend_*.py
                "strategy_manager.py"
            ],
            "execution/": [
                "trading_engine.py",  # Merge trading_*.py
                "order_manager.py",
                "portfolio_manager.py",
                "risk_manager.py"
            ],
            "scripts/": [
                "run_alphapulse.py",  # Unified runner
                "run_tests.py",  # Unified test runner
                "setup_database.py",
                "migrate_data.py"
            ]
        }
    
    def identify_duplicates_to_merge(self):
        """Identify files that should be merged based on similarity"""
        # High similarity files (100% similarity)
        high_similarity = [
            ("debug_manual_labeling.py", "test_add_low_confidence_function.py"),
            ("debug_manual_labeling.py", "test_phase3_active_learning.py"),
            ("run_analytics_dashboard.py", "run_chaos_engineering.py"),
            ("run_analytics_dashboard.py", "run_multi_region_dashboard.py"),
            ("run_analytics_dashboard.py", "run_resilience_dashboard.py"),
            ("run_analytics_dashboard.py", "run_security_dashboard.py"),
            ("test_add_low_confidence_function.py", "test_phase3_active_learning.py"),
            ("database/migrations/env.py", "database/migrations/env_enhanced.py"),
            ("database/db_scanner.py", "database/db_scanner_simple.py"),
            ("tests/conftest.py", "tests/conftest_enhanced.py"),
            ("tests/test_indicators_enhanced.py", "tests/test_unit_indicators.py"),
            ("tests/test_integration_pipeline.py", "tests/test_pipeline_enhanced.py")
        ]
        
        # Advanced_* files to consolidate
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
        
        # Performance_* files to consolidate
        performance_files = [
            "app/core/performance_alerting.py",
            "app/core/performance_profiling.py",
            "app/core/performance_regression.py",
            "test_performance_baseline.py",
            "test_performance_simple.py"
        ]
        
        # Test_* files to consolidate
        test_files = [
            "test_phase1_integration.py",
            "test_phase2_onnx_integration.py", 
            "test_phase2_simple.py",
            "test_phase2_verification.py",
            "test_phase3_active_learning.py",
            "test_phase3_monitoring.py",
            "test_phase3_monitoring_simple.py"
        ]
        
        self.plan.duplicates_to_merge.extend(high_similarity)
        
        # Add consolidation groups
        self.plan.duplicates_to_merge.extend([
            (advanced_files[0], f) for f in advanced_files[1:]
        ])
        self.plan.duplicates_to_merge.extend([
            (performance_files[0], f) for f in performance_files[1:]
        ])
        self.plan.duplicates_to_merge.extend([
            (test_files[0], f) for f in test_files[1:]
        ])
    
    def plan_file_moves(self):
        """Plan all file moves and consolidations"""
        
        # Core files
        core_moves = [
            ("alphapulse_core.py", "core/alphapulse_core.py"),
            ("indicators_engine.py", "core/indicators_engine.py"),
            ("ml_signal_generator.py", "core/ml_signal_generator.py"),
            ("market_regime_detector.py", "core/market_regime_detector.py"),
            ("optimized_trading_system.py", "core/optimized_trading_system.py"),
            ("websocket_binance.py", "core/websocket_binance.py")
        ]
        
        # Database files
        db_moves = [
            ("database/models.py", "database/models.py"),
            ("database/queries.py", "database/queries.py"),
            ("database/connection.py", "database/connection.py"),
            ("database/migrations/", "database/migrations/")
        ]
        
        # Strategy files
        strategy_moves = [
            ("strategies/strategy_manager.py", "strategies/strategy_manager.py"),
            ("strategies/pattern_detector.py", "strategies/pattern_detectors.py"),
            ("strategies/signal_generator.py", "strategies/signal_generators.py"),
            ("strategies/trend_following.py", "strategies/trend_analyzers.py")
        ]
        
        # Test files
        test_moves = [
            ("tests/test_unit_indicators.py", "tests/test_indicators.py"),
            ("test_simple_working.py", "tests/test_simple_working.py"),
            ("test_edge_cases_simple.py", "tests/test_edge_cases.py"),
            ("test_database_simple.py", "tests/test_database.py")
        ]
        
        # Add all moves
        for source, dest in core_moves + db_moves + strategy_moves + test_moves:
            self.plan.file_moves.append(FileMove(
                source=source,
                destination=dest,
                operation="move",
                reason="Reorganization"
            ))
    
    def identify_files_to_delete(self):
        """Identify duplicate files that should be deleted after merging"""
        # Files that are 100% similar to others
        duplicates_to_delete = [
            "debug_manual_labeling.py",  # Merged into test_phase3_active_learning.py
            "test_add_low_confidence_function.py",  # Merged into test_phase3_active_learning.py
            "run_chaos_engineering.py",  # Merged into run_analytics_dashboard.py
            "run_multi_region_dashboard.py",  # Merged into run_analytics_dashboard.py
            "run_resilience_dashboard.py",  # Merged into run_analytics_dashboard.py
            "run_security_dashboard.py",  # Merged into run_analytics_dashboard.py
            "database/migrations/env_enhanced.py",  # Merged into env.py
            "database/db_scanner_simple.py",  # Merged into db_scanner.py
            "tests/conftest_enhanced.py",  # Merged into conftest.py
            "tests/test_unit_indicators.py",  # Merged into test_indicators.py
            "tests/test_pipeline_enhanced.py",  # Merged into test_integration.py
        ]
        
        self.plan.files_to_delete.extend(duplicates_to_delete)
    
    def plan_new_files(self):
        """Plan new consolidated files to create"""
        new_files = [
            "utils/feature_engineering.py",  # Merge all feature_*.py
            "utils/utils.py",  # General utilities
            "services/data_services.py",  # Merge data_*.py
            "services/monitoring_services.py",  # Merge monitoring_*.py
            "services/trading_services.py",  # Merge trading_*.py
            "ai/advanced_utils.py",  # Merge advanced_*.py
            "ai/ml_models.py",  # Merge ml_*.py
            "strategies/pattern_detectors.py",  # Merge pattern_*.py
            "strategies/signal_generators.py",  # Merge signal_*.py
            "tests/test_indicators.py",  # Merge indicator tests
            "tests/test_integration.py",  # Merge integration tests
            "tests/test_utils.py",  # Common test utilities
            "docs/README.md",  # Main documentation
            "docs/model_docs.md",  # Model documentation
            "scripts/run_alphapulse.py",  # Unified runner
            "scripts/run_tests.py"  # Unified test runner
        ]
        
        self.plan.new_files_to_create.extend(new_files)
    
    def generate_plan(self) -> ConsolidationPlan:
        """Generate the complete reorganization plan"""
        logger.info("Generating reorganization plan...")
        
        self.create_new_structure()
        self.identify_duplicates_to_merge()
        self.plan_file_moves()
        self.identify_files_to_delete()
        self.plan_new_files()
        
        return self.plan
    
    def save_plan(self, filename: str = "reorganization_plan.json"):
        """Save the plan to a JSON file"""
        plan_dict = {
            "timestamp": datetime.now().isoformat(),
            "new_structure": self.plan.new_structure,
            "file_moves": [
                {
                    "source": move.source,
                    "destination": move.destination,
                    "operation": move.operation,
                    "reason": move.reason
                }
                for move in self.plan.file_moves
            ],
            "duplicates_to_merge": self.plan.duplicates_to_merge,
            "files_to_delete": self.plan.files_to_delete,
            "new_files_to_create": self.plan.new_files_to_create
        }
        
        with open(filename, "w") as f:
            json.dump(plan_dict, f, indent=2)
        
        logger.info(f"Plan saved to {filename}")
    
    def print_summary(self):
        """Print a summary of the reorganization plan"""
        print("\n=== AlphaPulse Reorganization Plan ===")
        print(f"New directories: {len(self.plan.new_structure)}")
        print(f"Files to move: {len(self.plan.file_moves)}")
        print(f"Duplicates to merge: {len(self.plan.duplicates_to_merge)}")
        print(f"Files to delete: {len(self.plan.files_to_delete)}")
        print(f"New files to create: {len(self.plan.new_files_to_create)}")
        
        print(f"\n=== New Directory Structure ===")
        for directory, files in self.plan.new_structure.items():
            print(f"{directory}:")
            for file in files:
                print(f"  - {file}")
        
        print(f"\n=== Files to Delete (Duplicates) ===")
        for file in self.plan.files_to_delete[:10]:  # Show first 10
            print(f"  - {file}")
        if len(self.plan.files_to_delete) > 10:
            print(f"  ... and {len(self.plan.files_to_delete) - 10} more")

def main():
    """Generate and save the reorganization plan"""
    planner = ReorganizationPlanner()
    plan = planner.generate_plan()
    planner.save_plan()
    planner.print_summary()

if __name__ == "__main__":
    main()
