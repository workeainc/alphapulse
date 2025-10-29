"""
Unified Results Configuration for AlphaPulse
Centralizes all output directory paths for performance testing and profiling
"""

from pathlib import Path
import os

class ResultsConfig:
    """Centralized configuration for all result output directories"""
    
    # Base results directory
    BASE_RESULTS_DIR = Path("results")
    
    # Performance testing directories
    PERFORMANCE_PROFILES = BASE_RESULTS_DIR / "performance_profiles"
    BENCHMARK_RESULTS = BASE_RESULTS_DIR / "benchmark_results"
    PERFORMANCE_BASELINES = BASE_RESULTS_DIR / "performance_baselines"
    
    # Additional result types can be added here
    LOGS = BASE_RESULTS_DIR / "logs"
    REPORTS = BASE_RESULTS_DIR / "reports"
    EXPORTS = BASE_RESULTS_DIR / "exports"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all result directories exist"""
        directories = [
            cls.BASE_RESULTS_DIR,
            cls.PERFORMANCE_PROFILES,
            cls.BENCHMARK_RESULTS,
            cls.PERFORMANCE_BASELINES,
            cls.LOGS,
            cls.REPORTS,
            cls.EXPORTS
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_performance_profiles_dir(cls) -> Path:
        """Get performance profiles directory path"""
        cls.PERFORMANCE_PROFILES.mkdir(parents=True, exist_ok=True)
        return cls.PERFORMANCE_PROFILES
    
    @classmethod
    def get_benchmark_results_dir(cls) -> Path:
        """Get benchmark results directory path"""
        cls.BENCHMARK_RESULTS.mkdir(parents=True, exist_ok=True)
        return cls.BENCHMARK_RESULTS
    
    @classmethod
    def get_performance_baselines_dir(cls) -> Path:
        """Get performance baselines directory path"""
        cls.PERFORMANCE_BASELINES.mkdir(parents=True, exist_ok=True)
        return cls.PERFORMANCE_BASELINES
    
    @classmethod
    def get_logs_dir(cls) -> Path:
        """Get logs directory path"""
        cls.LOGS.mkdir(parents=True, exist_ok=True)
        return cls.LOGS
    
    @classmethod
    def get_reports_dir(cls) -> Path:
        """Get reports directory path"""
        cls.REPORTS.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS
    
    @classmethod
    def get_exports_dir(cls) -> Path:
        """Get exports directory path"""
        cls.EXPORTS.mkdir(parents=True, exist_ok=True)
        return cls.EXPORTS
    
    @classmethod
    def cleanup_old_files(cls, days_to_keep: int = 30):
        """Clean up old result files"""
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for directory in [cls.PERFORMANCE_PROFILES, cls.BENCHMARK_RESULTS, cls.PERFORMANCE_BASELINES]:
            if directory.exists():
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        if file_path.stat().st_mtime < cutoff_time:
                            try:
                                file_path.unlink()
                                print(f"Cleaned up old file: {file_path}")
                            except Exception as e:
                                print(f"Failed to clean up {file_path}: {e}")

# Initialize directories on import
ResultsConfig.ensure_directories()
