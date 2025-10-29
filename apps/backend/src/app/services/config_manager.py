#!/usr/bin/env python3
"""
Configuration Manager for Ultra-Optimized Pattern Detection
Dynamically adjusts configuration based on performance metrics
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

@dataclass
class PatternDetectionConfig:
    """Configuration for ultra-optimized pattern detection"""
    # Performance settings
    max_workers: int = 8
    buffer_size: int = 1000
    cache_ttl_seconds: int = 300
    max_cache_size: int = 10000
    
    # Pattern detection thresholds
    min_confidence_threshold: float = 0.3
    min_pattern_strength: float = 0.2
    volume_confirmation_threshold: float = 1.5
    trend_alignment_weight: float = 0.1
    
    # Optimization settings
    enable_vectorization: bool = True
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    enable_sliding_windows: bool = True
    
    # Memory management
    max_memory_usage_mb: int = 2048
    cleanup_interval_seconds: int = 3600
    max_buffer_age_seconds: int = 7200
    
    # Performance monitoring
    monitor_interval_seconds: int = 5
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'processing_time_ms': 1000.0,
                'memory_usage_mb': 2048.0,
                'cpu_usage_percent': 80.0,
                'cache_hit_rate': 0.3,
                'error_rate': 0.05,
                'patterns_per_second': 1000.0
            }

class ConfigManager:
    """
    Dynamic configuration manager for ultra-optimized pattern detection
    """
    
    def __init__(self, config_file: str = "config/pattern_detection_config.json"):
        """Initialize configuration manager"""
        self.config_file = config_file
        self.config = PatternDetectionConfig()
        self.performance_history = []
        self.optimization_rules = self._load_optimization_rules()
        
        # Load initial configuration
        self.load_config()
        
        logger.info("⚙️ Configuration Manager initialized")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration with loaded data
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                logger.info(f"📁 Configuration loaded from {self.config_file}")
            else:
                logger.info("📁 No configuration file found, using defaults")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            config_data = asdict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules for dynamic configuration adjustment"""
        return {
            'high_processing_time': {
                'condition': lambda metrics: metrics.get('avg_processing_time_ms', 0) > 500,
                'actions': [
                    {'action': 'increase_workers', 'value': 2},
                    {'action': 'enable_parallel_processing', 'value': True},
                    {'action': 'reduce_buffer_size', 'value': 0.8}
                ]
            },
            'low_cache_hit_rate': {
                'condition': lambda metrics: metrics.get('cache_hit_rate', 1.0) < 0.4,
                'actions': [
                    {'action': 'increase_cache_ttl', 'value': 1.5},
                    {'action': 'increase_cache_size', 'value': 1.5}
                ]
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.get('memory_usage_mb', 0) > 1024,
                'actions': [
                    {'action': 'reduce_buffer_size', 'value': 0.7},
                    {'action': 'decrease_cache_size', 'value': 0.8},
                    {'action': 'enable_cleanup', 'value': True}
                ]
            },
            'high_error_rate': {
                'condition': lambda metrics: metrics.get('error_rate', 0) > 0.02,
                'actions': [
                    {'action': 'increase_confidence_threshold', 'value': 1.2},
                    {'action': 'decrease_workers', 'value': 0.8}
                ]
            },
            'low_patterns_per_second': {
                'condition': lambda metrics: metrics.get('patterns_per_second', 0) < 500,
                'actions': [
                    {'action': 'enable_vectorization', 'value': True},
                    {'action': 'enable_sliding_windows', 'value': True},
                    {'action': 'increase_workers', 'value': 1.5}
                ]
            }
        }
    
    def update_config_based_on_performance(self, performance_metrics: Dict[str, Any]):
        """Update configuration based on performance metrics"""
        try:
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': performance_metrics,
                'config': asdict(self.config)
            })
            
            # Keep only last 100 entries
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Apply optimization rules
            changes_made = []
            
            for rule_name, rule in self.optimization_rules.items():
                if rule['condition'](performance_metrics):
                    logger.info(f"🔧 Applying optimization rule: {rule_name}")
                    
                    for action in rule['actions']:
                        change = self._apply_config_action(action)
                        if change:
                            changes_made.append(change)
            
            # Save configuration if changes were made
            if changes_made:
                self.save_config()
                logger.info(f"✅ Configuration updated with {len(changes_made)} changes")
                
                # Log changes
                for change in changes_made:
                    logger.info(f"  - {change}")
            
            return changes_made
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return []
    
    def _apply_config_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Apply a configuration action"""
        try:
            action_type = action['action']
            value = action['value']
            
            if action_type == 'increase_workers':
                new_workers = min(self.config.max_workers + value, 16)
                if new_workers != self.config.max_workers:
                    self.config.max_workers = new_workers
                    return f"Increased max_workers to {new_workers}"
            
            elif action_type == 'decrease_workers':
                new_workers = max(int(self.config.max_workers * value), 2)
                if new_workers != self.config.max_workers:
                    self.config.max_workers = new_workers
                    return f"Decreased max_workers to {new_workers}"
            
            elif action_type == 'increase_cache_ttl':
                new_ttl = int(self.config.cache_ttl_seconds * value)
                if new_ttl != self.config.cache_ttl_seconds:
                    self.config.cache_ttl_seconds = new_ttl
                    return f"Increased cache TTL to {new_ttl} seconds"
            
            elif action_type == 'increase_cache_size':
                new_size = int(self.config.max_cache_size * value)
                if new_size != self.config.max_cache_size:
                    self.config.max_cache_size = new_size
                    return f"Increased cache size to {new_size}"
            
            elif action_type == 'decrease_cache_size':
                new_size = int(self.config.max_cache_size * value)
                if new_size != self.config.max_cache_size:
                    self.config.max_cache_size = new_size
                    return f"Decreased cache size to {new_size}"
            
            elif action_type == 'reduce_buffer_size':
                new_size = int(self.config.buffer_size * value)
                if new_size != self.config.buffer_size:
                    self.config.buffer_size = new_size
                    return f"Reduced buffer size to {new_size}"
            
            elif action_type == 'increase_confidence_threshold':
                new_threshold = self.config.min_confidence_threshold * value
                if new_threshold != self.config.min_confidence_threshold:
                    self.config.min_confidence_threshold = new_threshold
                    return f"Increased confidence threshold to {new_threshold:.3f}"
            
            elif action_type == 'enable_parallel_processing':
                if not self.config.enable_parallel_processing:
                    self.config.enable_parallel_processing = True
                    return "Enabled parallel processing"
            
            elif action_type == 'enable_vectorization':
                if not self.config.enable_vectorization:
                    self.config.enable_vectorization = True
                    return "Enabled vectorization"
            
            elif action_type == 'enable_sliding_windows':
                if not self.config.enable_sliding_windows:
                    self.config.enable_sliding_windows = True
                    return "Enabled sliding windows"
            
            elif action_type == 'enable_cleanup':
                # This would trigger immediate cleanup
                return "Triggered memory cleanup"
            
            return None
            
        except Exception as e:
            logger.error(f"Error applying config action {action}: {e}")
            return None
    
    def get_optimization_recommendations(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Get optimization recommendations based on current performance"""
        recommendations = []
        
        # Check processing time
        avg_processing_time = performance_metrics.get('avg_processing_time_ms', 0)
        if avg_processing_time > 500:
            recommendations.append("High processing time detected - consider increasing workers or optimizing algorithms")
        
        # Check cache performance
        cache_hit_rate = performance_metrics.get('cache_hit_rate', 1.0)
        if cache_hit_rate < 0.4:
            recommendations.append("Low cache hit rate - consider increasing cache TTL or size")
        
        # Check memory usage
        memory_usage = performance_metrics.get('memory_usage_mb', 0)
        if memory_usage > 1024:
            recommendations.append("High memory usage - consider reducing buffer sizes or enabling cleanup")
        
        # Check error rate
        error_rate = performance_metrics.get('error_rate', 0)
        if error_rate > 0.02:
            recommendations.append("High error rate - consider increasing confidence thresholds or reducing workers")
        
        # Check patterns per second
        patterns_per_second = performance_metrics.get('patterns_per_second', 0)
        if patterns_per_second < 500:
            recommendations.append("Low throughput - consider enabling vectorization or increasing workers")
        
        return recommendations
    
    def create_performance_profile(self, profile_name: str, description: str = ""):
        """Create a performance profile for different scenarios"""
        profile = {
            'name': profile_name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'performance_history': self.performance_history[-10:]  # Last 10 entries
        }
        
        profiles_dir = "config/profiles"
        os.makedirs(profiles_dir, exist_ok=True)
        
        profile_file = f"{profiles_dir}/{profile_name}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"📋 Performance profile '{profile_name}' saved to {profile_file}")
        return profile_file
    
    def load_performance_profile(self, profile_name: str):
        """Load a performance profile"""
        try:
            profile_file = f"config/profiles/{profile_name}.json"
            
            if os.path.exists(profile_file):
                with open(profile_file, 'r') as f:
                    profile = json.load(f)
                
                # Update configuration with profile data
                for key, value in profile['config'].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                self.save_config()
                logger.info(f"📋 Performance profile '{profile_name}' loaded")
                return True
            else:
                logger.error(f"Profile file not found: {profile_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading performance profile: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'current_config': asdict(self.config),
            'performance_history_count': len(self.performance_history),
            'optimization_rules_count': len(self.optimization_rules),
            'last_updated': datetime.now().isoformat()
        }
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = PatternDetectionConfig()
        self.save_config()
        logger.info("🔄 Configuration reset to defaults")
    
    def export_config(self, filename: str = None) -> str:
        """Export current configuration"""
        if not filename:
            filename = f"config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'performance_history': self.performance_history,
            'optimization_rules': self.optimization_rules
        }
        
        os.makedirs('exports', exist_ok=True)
        filepath = f"exports/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Configuration exported to {filepath}")
        return filepath

# Global configuration manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager
