#!/usr/bin/env python3
"""
Database Structure Scanner and Comparison Tool for AlphaPulse
Scans existing database structure and compares with required schema
"""

import os
import sys
import json
from sqlalchemy import create_engine, inspect, text, MetaData
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Update import paths for new structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..database.models_enhanced import Base, Signal, Log, Feedback, PerformanceMetrics

class DatabaseStructureScanner:
    """Comprehensive database structure scanner and comparison tool"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///test_alphapulse.db")
        self.engine = create_engine(self.database_url)
        self.inspector = inspect(self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        
    def scan_existing_structure(self) -> Dict[str, Any]:
        """Scan existing database structure using reflection"""
        print("🔍 Scanning existing database structure...")
        
        existing_structure = {
            'database_url': self.database_url,
            'scan_timestamp': datetime.now(timezone.utc).isoformat(),
            'tables': {},
            'summary': {
                'total_tables': 0,
                'total_columns': 0,
                'total_indexes': 0,
                'total_foreign_keys': 0
            }
        }
        
        for table_name in self.inspector.get_table_names():
            print(f"  📋 Analyzing table: {table_name}")
            
            table_info = {
                'name': table_name,
                'columns': [],
                'indexes': [],
                'foreign_keys': [],
                'primary_keys': []
            }
            
            # Get columns
            for column in self.inspector.get_columns(table_name):
                col_info = {
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': column['default'],
                    'primary_key': column['primary_key'],
                    'autoincrement': column.get('autoincrement', False)
                }
                table_info['columns'].append(col_info)
                existing_structure['summary']['total_columns'] += 1
            
            # Get indexes
            for index in self.inspector.get_indexes(table_name):
                index_info = {
                    'name': index['name'],
                    'columns': index['column_names'],
                    'unique': index['unique'],
                    'dialect_options': index.get('dialect_options', {})
                }
                table_info['indexes'].append(index_info)
                existing_structure['summary']['total_indexes'] += 1
            
            # Get foreign keys
            for fk in self.inspector.get_foreign_keys(table_name):
                fk_info = {
                    'constrained_columns': fk['constrained_columns'],
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns'],
                    'name': fk.get('name', '')
                }
                table_info['foreign_keys'].append(fk_info)
                existing_structure['summary']['total_foreign_keys'] += 1
            
            # Get primary keys
            pk_constraint = self.inspector.get_pk_constraint(table_name)
            if pk_constraint and pk_constraint['constrained_columns']:
                table_info['primary_keys'] = pk_constraint['constrained_columns']
            
            existing_structure['tables'][table_name] = table_info
            existing_structure['summary']['total_tables'] += 1
        
        return existing_structure
    
    def get_required_structure(self) -> Dict[str, Any]:
        """Define the required database structure for AlphaPulse testing"""
        return {
            'description': 'Required AlphaPulse testing database structure',
            'tables': {
                'signals': {
                    'description': 'Trading signals table for storing generated signals',
                    'columns': [
                        {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True, 'autoincrement': True},
                        {'name': 'signal_id', 'type': 'VARCHAR(20)', 'nullable': False, 'unique': True},
                        {'name': 'timestamp', 'type': 'DATETIME', 'nullable': False},
                        {'name': 'symbol', 'type': 'VARCHAR(20)', 'nullable': False},
                        {'name': 'timeframe', 'type': 'VARCHAR(10)', 'nullable': False},
                        {'name': 'direction', 'type': 'VARCHAR(10)', 'nullable': False},
                        {'name': 'confidence', 'type': 'FLOAT', 'nullable': False},
                        {'name': 'entry_price', 'type': 'FLOAT', 'nullable': False},
                        {'name': 'tp1', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'tp2', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'tp3', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'tp4', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'stop_loss', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'risk_reward_ratio', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'pattern_type', 'type': 'VARCHAR(50)', 'nullable': True},
                        {'name': 'volume_confirmation', 'type': 'BOOLEAN', 'nullable': True},
                        {'name': 'trend_alignment', 'type': 'BOOLEAN', 'nullable': True},
                        {'name': 'market_regime', 'type': 'VARCHAR(20)', 'nullable': True},
                        {'name': 'indicators', 'type': 'JSON', 'nullable': True},
                        {'name': 'validation_metrics', 'type': 'JSON', 'nullable': True},
                        {'name': 'metadata', 'type': 'JSON', 'nullable': True},
                        {'name': 'outcome', 'type': 'VARCHAR(20)', 'nullable': True, 'default': 'pending'},
                        {'name': 'created_at', 'type': 'DATETIME', 'nullable': True}
                    ],
                    'indexes': [
                        {'name': 'idx_signals_signal_id', 'columns': ['signal_id'], 'unique': True},
                        {'name': 'idx_signals_symbol_timeframe_timestamp', 'columns': ['symbol', 'timeframe', 'timestamp']},
                        {'name': 'idx_signals_confidence_outcome', 'columns': ['confidence', 'outcome']}
                    ]
                },
                'logs': {
                    'description': 'False positive logs for signal validation feedback',
                    'columns': [
                        {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True, 'autoincrement': True},
                        {'name': 'signal_id', 'type': 'VARCHAR(20)', 'nullable': False},
                        {'name': 'pattern_type', 'type': 'VARCHAR(50)', 'nullable': False},
                        {'name': 'confidence_score', 'type': 'FLOAT', 'nullable': False},
                        {'name': 'volume_context', 'type': 'JSON', 'nullable': True},
                        {'name': 'trend_context', 'type': 'JSON', 'nullable': True},
                        {'name': 'outcome', 'type': 'VARCHAR(20)', 'nullable': True},
                        {'name': 'timestamp', 'type': 'DATETIME', 'nullable': False},
                        {'name': 'created_at', 'type': 'DATETIME', 'nullable': True}
                    ],
                    'indexes': [
                        {'name': 'idx_logs_signal_id', 'columns': ['signal_id']},
                        {'name': 'idx_logs_timestamp', 'columns': ['timestamp']}
                    ],
                    'foreign_keys': [
                        {'constrained_columns': ['signal_id'], 'referred_table': 'signals', 'referred_columns': ['signal_id']}
                    ]
                },
                'feedback': {
                    'description': 'Signal feedback and outcomes',
                    'columns': [
                        {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True, 'autoincrement': True},
                        {'name': 'signal_id', 'type': 'VARCHAR(20)', 'nullable': False},
                        {'name': 'market_outcome', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'notes', 'type': 'TEXT', 'nullable': True},
                        {'name': 'timestamp', 'type': 'DATETIME', 'nullable': False},
                        {'name': 'created_at', 'type': 'DATETIME', 'nullable': True}
                    ],
                    'indexes': [
                        {'name': 'idx_feedback_signal_id', 'columns': ['signal_id']}
                    ],
                    'foreign_keys': [
                        {'constrained_columns': ['signal_id'], 'referred_table': 'signals', 'referred_columns': ['signal_id']}
                    ]
                },
                'performance_metrics': {
                    'description': 'Performance metrics for testing and monitoring',
                    'columns': [
                        {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True, 'autoincrement': True},
                        {'name': 'test_name', 'type': 'VARCHAR(100)', 'nullable': False},
                        {'name': 'test_timestamp', 'type': 'DATETIME', 'nullable': False},
                        {'name': 'avg_latency_ms', 'type': 'FLOAT', 'nullable': False},
                        {'name': 'max_latency_ms', 'type': 'FLOAT', 'nullable': False},
                        {'name': 'min_latency_ms', 'type': 'FLOAT', 'nullable': False},
                        {'name': 'p95_latency_ms', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'p99_latency_ms', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'win_rate', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'total_signals', 'type': 'INTEGER', 'nullable': False},
                        {'name': 'winning_signals', 'type': 'INTEGER', 'nullable': True},
                        {'name': 'losing_signals', 'type': 'INTEGER', 'nullable': True},
                        {'name': 'filtered_signals', 'type': 'INTEGER', 'nullable': True},
                        {'name': 'filter_rate', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'throughput_signals_per_sec', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'cpu_usage_percent', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'memory_usage_mb', 'type': 'FLOAT', 'nullable': True},
                        {'name': 'test_config', 'type': 'JSON', 'nullable': True},
                        {'name': 'test_results', 'type': 'JSON', 'nullable': True},
                        {'name': 'created_at', 'type': 'DATETIME', 'nullable': True}
                    ],
                    'indexes': [
                        {'name': 'idx_performance_metrics_test_timestamp', 'columns': ['test_name', 'test_timestamp']}
                    ]
                }
            }
        }
    
    def compare_structures(self, existing: Dict, required: Dict) -> Dict[str, Any]:
        """Compare existing and required database structures"""
        print("🔍 Comparing database structures...")
        
        comparison = {
            'comparison_timestamp': datetime.now(timezone.utc).isoformat(),
            'database_url': self.database_url,
            'matches_required': True,
            'missing_tables': [],
            'extra_tables': [],
            'table_comparisons': {},
            'recommendations': [],
            'migration_needed': False
        }
        
        existing_tables = set(existing['tables'].keys())
        required_tables = set(required['tables'].keys())
        
        # Find missing and extra tables
        missing_tables = required_tables - existing_tables
        extra_tables = existing_tables - required_tables
        
        comparison['missing_tables'] = list(missing_tables)
        comparison['extra_tables'] = list(extra_tables)
        
        if missing_tables:
            comparison['matches_required'] = False
            comparison['migration_needed'] = True
            comparison['recommendations'].append(f"Create missing tables: {', '.join(missing_tables)}")
        
        # Compare existing tables with required structure
        for table_name in existing_tables & required_tables:
            existing_table = existing['tables'][table_name]
            required_table = required['tables'][table_name]
            
            table_comparison = self._compare_table_structure(existing_table, required_table)
            comparison['table_comparisons'][table_name] = table_comparison
            
            if not table_comparison['matches']:
                comparison['matches_required'] = False
                comparison['migration_needed'] = True
        
        return comparison
    
    def _compare_table_structure(self, existing_table: Dict, required_table: Dict) -> Dict[str, Any]:
        """Compare individual table structures"""
        comparison = {
            'table_name': existing_table['name'],
            'matches': True,
            'missing_columns': [],
            'extra_columns': [],
            'column_mismatches': [],
            'missing_indexes': [],
            'extra_indexes': [],
            'missing_foreign_keys': [],
            'extra_foreign_keys': []
        }
        
        # Compare columns
        existing_columns = {col['name']: col for col in existing_table['columns']}
        required_columns = {col['name']: col for col in required_table['columns']}
        
        missing_columns = set(required_columns.keys()) - set(existing_columns.keys())
        extra_columns = set(existing_columns.keys()) - set(required_columns.keys())
        
        comparison['missing_columns'] = list(missing_columns)
        comparison['extra_columns'] = list(extra_columns)
        
        # Check column type mismatches
        for col_name in set(required_columns.keys()) & set(existing_columns.keys()):
            existing_col = existing_columns[col_name]
            required_col = required_columns[col_name]
            
            if not self._columns_match(existing_col, required_col):
                comparison['column_mismatches'].append({
                    'column': col_name,
                    'existing': existing_col,
                    'required': required_col
                })
        
        # Compare indexes
        existing_indexes = {idx['name']: idx for idx in existing_table['indexes']}
        required_indexes = {idx['name']: idx for idx in required_table['indexes']}
        
        missing_indexes = set(required_indexes.keys()) - set(existing_indexes.keys())
        extra_indexes = set(existing_indexes.keys()) - set(required_indexes.keys())
        
        comparison['missing_indexes'] = list(missing_indexes)
        comparison['extra_indexes'] = list(extra_indexes)
        
        # Compare foreign keys
        existing_fks = {(fk['constrained_columns'][0], fk['referred_table']) for fk in existing_table['foreign_keys']}
        required_fks = {(fk['constrained_columns'][0], fk['referred_table']) for fk in required_table.get('foreign_keys', [])}
        
        missing_fks = required_fks - existing_fks
        extra_fks = existing_fks - required_fks
        
        comparison['missing_foreign_keys'] = list(missing_fks)
        comparison['extra_foreign_keys'] = list(extra_fks)
        
        # Determine if table matches
        if (missing_columns or comparison['column_mismatches'] or 
            missing_indexes or missing_fks):
            comparison['matches'] = False
        
        return comparison
    
    def _columns_match(self, existing_col: Dict, required_col: Dict) -> bool:
        """Check if two column definitions match"""
        # Basic type compatibility check
        existing_type = existing_col['type'].upper()
        required_type = required_col['type'].upper()
        
        # Simple type mapping for compatibility
        type_mapping = {
            'INTEGER': ['INT', 'BIGINT', 'INTEGER'],
            'VARCHAR': ['VARCHAR', 'STRING', 'TEXT'],
            'FLOAT': ['FLOAT', 'DOUBLE', 'REAL', 'DECIMAL'],
            'DATETIME': ['DATETIME', 'TIMESTAMP'],
            'BOOLEAN': ['BOOLEAN', 'BOOL'],
            'JSON': ['JSON', 'JSONB', 'TEXT']  # JSON can be stored as TEXT in SQLite
        }
        
        # Check if types are compatible
        for base_type, compatible_types in type_mapping.items():
            if required_type.startswith(base_type):
                if any(existing_type.startswith(comp_type) for comp_type in compatible_types):
                    break
        else:
            return False
        
        # Check nullable constraint
        if required_col.get('nullable') is not None and existing_col.get('nullable') != required_col['nullable']:
            return False
        
        # Check primary key constraint
        if required_col.get('primary_key') is not None and existing_col.get('primary_key') != required_col['primary_key']:
            return False
        
        return True
    
    def generate_migration_recommendations(self, comparison: Dict) -> List[str]:
        """Generate specific migration recommendations"""
        recommendations = []
        
        if comparison['migration_needed']:
            recommendations.append("Database migration is required to match the required schema.")
            
            # Table creation recommendations
            for table_name in comparison['missing_tables']:
                recommendations.append(f"CREATE TABLE {table_name} with required columns and constraints")
            
            # Column addition recommendations
            for table_name, table_comp in comparison['table_comparisons'].items():
                for col_name in table_comp['missing_columns']:
                    recommendations.append(f"ADD COLUMN {col_name} to table {table_name}")
                
                for mismatch in table_comp['column_mismatches']:
                    recommendations.append(f"MODIFY COLUMN {mismatch['column']} in table {table_name}")
            
            # Index creation recommendations
            for table_name, table_comp in comparison['table_comparisons'].items():
                for idx_name in table_comp['missing_indexes']:
                    recommendations.append(f"CREATE INDEX {idx_name} on table {table_name}")
            
            # Foreign key recommendations
            for table_name, table_comp in comparison['table_comparisons'].items():
                for fk in table_comp['missing_foreign_keys']:
                    recommendations.append(f"ADD FOREIGN KEY constraint to table {table_name}")
        else:
            recommendations.append("Database structure matches required schema. No migration needed.")
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive database analysis report"""
        print("📊 Generating database analysis report...")
        
        # Scan existing structure
        existing_structure = self.scan_existing_structure()
        
        # Get required structure
        required_structure = self.get_required_structure()
        
        # Compare structures
        comparison = self.compare_structures(existing_structure, required_structure)
        
        # Generate recommendations
        recommendations = self.generate_migration_recommendations(comparison)
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'scanner_version': '1.0.0',
                'database_url': self.database_url
            },
            'executive_summary': {
                'database_exists': len(existing_structure['tables']) > 0,
                'matches_required_schema': comparison['matches_required'],
                'migration_needed': comparison['migration_needed'],
                'total_existing_tables': existing_structure['summary']['total_tables'],
                'total_required_tables': len(required_structure['tables']),
                'missing_tables_count': len(comparison['missing_tables']),
                'extra_tables_count': len(comparison['extra_tables'])
            },
            'existing_structure': existing_structure,
            'required_structure': required_structure,
            'comparison': comparison,
            'recommendations': recommendations,
            'migration_script_suggestions': self._generate_migration_suggestions(comparison)
        }
        
        # Save report to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Report saved to: {output_file}")
        
        return report
    
    def _generate_migration_suggestions(self, comparison: Dict) -> List[str]:
        """Generate specific migration script suggestions"""
        suggestions = []
        
        if not comparison['migration_needed']:
            suggestions.append("# No migration needed - database structure matches requirements")
            return suggestions
        
        suggestions.append("# Migration script suggestions for AlphaPulse database")
        suggestions.append("# Generated by DatabaseStructureScanner")
        suggestions.append("")
        
        # Create missing tables
        for table_name in comparison['missing_tables']:
            suggestions.append(f"# Create table: {table_name}")
            suggestions.append(f"CREATE TABLE {table_name} (")
            suggestions.append("    # Add required columns here")
            suggestions.append(");")
            suggestions.append("")
        
        # Add missing columns
        for table_name, table_comp in comparison['table_comparisons'].items():
            for col_name in table_comp['missing_columns']:
                suggestions.append(f"# Add column to {table_name}")
                suggestions.append(f"ALTER TABLE {table_name} ADD COLUMN {col_name} <type>;")
                suggestions.append("")
        
        # Create missing indexes
        for table_name, table_comp in comparison['table_comparisons'].items():
            for idx_name in table_comp['missing_indexes']:
                suggestions.append(f"# Create index on {table_name}")
                suggestions.append(f"CREATE INDEX {idx_name} ON {table_name} (<columns>);")
                suggestions.append("")
        
        return suggestions

def main():
    """Main function to run database structure analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Structure Scanner for AlphaPulse")
    parser.add_argument("--database-url", help="Database URL to scan")
    parser.add_argument("--output-file", help="Output file for the report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = DatabaseStructureScanner(args.database_url)
    
    # Generate report
    report = scanner.generate_report(args.output_file)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 DATABASE STRUCTURE ANALYSIS SUMMARY")
    print("="*80)
    
    summary = report['executive_summary']
    print(f"Database URL: {report['report_metadata']['database_url']}")
    print(f"Database exists: {summary['database_exists']}")
    print(f"Matches required schema: {summary['matches_required_schema']}")
    print(f"Migration needed: {summary['migration_needed']}")
    print(f"Existing tables: {summary['total_existing_tables']}")
    print(f"Required tables: {summary['total_required_tables']}")
    print(f"Missing tables: {summary['missing_tables_count']}")
    print(f"Extra tables: {summary['extra_tables_count']}")
    
    if summary['migration_needed']:
        print("\n🔧 MIGRATION RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print("\n" + "="*80)
    
    return report

if __name__ == "__main__":
    main()
