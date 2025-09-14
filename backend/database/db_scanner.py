#!/usr/bin/env python3
"""
Database Structure Scanner for AlphaPulse
Scans existing database structure and compares with required schema
"""

import os
import sys
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..database.models import Base, engine, SessionLocal
from ..database.connection import get_database_url

class DatabaseScanner:
    """Scans and analyzes database structure"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url()
        self.engine = create_engine(self.database_url)
        self.inspector = inspect(self.engine)
        self.session = SessionLocal()
        
    def scan_existing_structure(self) -> Dict:
        """Scan existing database structure"""
        print("🔍 Scanning existing database structure...")
        
        existing_tables = {}
        
        for table_name in self.inspector.get_table_names():
            print(f"  📋 Analyzing table: {table_name}")
            
            # Get columns
            columns = []
            for column in self.inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': column['default'],
                    'primary_key': column['primary_key']
                })
            
            # Get indexes
            indexes = []
            for index in self.inspector.get_indexes(table_name):
                indexes.append({
                    'name': index['name'],
                    'columns': index['column_names'],
                    'unique': index['unique']
                })
            
            # Get foreign keys
            foreign_keys = []
            for fk in self.inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    'constrained_columns': fk['constrained_columns'],
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns']
                })
            
            existing_tables[table_name] = {
                'columns': columns,
                'indexes': indexes,
                'foreign_keys': foreign_keys
            }
        
        return existing_tables
    
    def get_required_structure(self) -> Dict:
        """Define the required database structure for AlphaPulse testing"""
        return {
            'signals': {
                'description': 'Trading signals table for storing generated signals',
                'columns': [
                    {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                    {'name': 'symbol', 'type': 'VARCHAR(20)', 'nullable': False},
                    {'name': 'timeframe', 'type': 'VARCHAR(10)', 'nullable': False},
                    {'name': 'direction', 'type': 'VARCHAR(10)', 'nullable': False},  # 'buy'/'sell'
                    {'name': 'confidence', 'type': 'FLOAT', 'nullable': False},
                    {'name': 'tp1', 'type': 'FLOAT', 'nullable': True},
                    {'name': 'tp2', 'type': 'FLOAT', 'nullable': True},
                    {'name': 'tp3', 'type': 'FLOAT', 'nullable': True},
                    {'name': 'tp4', 'type': 'FLOAT', 'nullable': True},
                    {'name': 'sl', 'type': 'FLOAT', 'nullable': True},
                    {'name': 'timestamp', 'type': 'DATETIME', 'nullable': False},
                    {'name': 'outcome', 'type': 'VARCHAR(20)', 'nullable': True, 'default': 'pending'}  # 'win'/'loss'/'pending'
                ],
                'indexes': [
                    {'name': 'idx_signals_symbol_timeframe_timestamp', 'columns': ['symbol', 'timeframe', 'timestamp']}
                ]
            },
            'logs': {
                'description': 'False positive logs for signal validation feedback',
                'columns': [
                    {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                    {'name': 'pattern_type', 'type': 'VARCHAR(50)', 'nullable': False},
                    {'name': 'confidence_score', 'type': 'FLOAT', 'nullable': False},
                    {'name': 'volume_context', 'type': 'JSON', 'nullable': True},
                    {'name': 'trend_context', 'type': 'JSON', 'nullable': True},
                    {'name': 'outcome', 'type': 'VARCHAR(20)', 'nullable': True},
                    {'name': 'timestamp', 'type': 'DATETIME', 'nullable': False}
                ],
                'indexes': [
                    {'name': 'idx_logs_timestamp', 'columns': ['timestamp']}
                ]
            },
            'feedback': {
                'description': 'Signal feedback and outcomes',
                'columns': [
                    {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True},
                    {'name': 'signal_id', 'type': 'INTEGER', 'nullable': False},
                    {'name': 'market_outcome', 'type': 'FLOAT', 'nullable': True},  # PnL
                    {'name': 'notes', 'type': 'TEXT', 'nullable': True}
                ],
                'foreign_keys': [
                    {'constrained_columns': ['signal_id'], 'referred_table': 'signals', 'referred_columns': ['id']}
                ]
            }
        }
    
    def compare_structures(self, existing: Dict, required: Dict) -> Dict:
        """Compare existing vs required database structure"""
        print("\n📊 Comparing database structures...")
        
        comparison = {
            'missing_tables': [],
            'existing_tables': [],
            'mismatched_tables': [],
            'recommendations': []
        }
        
        # Check for missing tables
        for table_name, table_spec in required.items():
            if table_name not in existing:
                comparison['missing_tables'].append({
                    'table': table_name,
                    'description': table_spec['description']
                })
            else:
                comparison['existing_tables'].append(table_name)
                
                # Check for column mismatches
                existing_columns = {col['name']: col for col in existing[table_name]['columns']}
                required_columns = {col['name']: col for col in table_spec['columns']}
                
                missing_columns = []
                mismatched_columns = []
                
                for col_name, col_spec in required_columns.items():
                    if col_name not in existing_columns:
                        missing_columns.append(col_name)
                    else:
                        existing_col = existing_columns[col_name]
                        if (existing_col['type'] != col_spec['type'] or 
                            existing_col['nullable'] != col_spec['nullable']):
                            mismatched_columns.append({
                                'column': col_name,
                                'existing': existing_col,
                                'required': col_spec
                            })
                
                if missing_columns or mismatched_columns:
                    comparison['mismatched_tables'].append({
                        'table': table_name,
                        'missing_columns': missing_columns,
                        'mismatched_columns': mismatched_columns
                    })
        
        # Generate recommendations
        if comparison['missing_tables']:
            comparison['recommendations'].append(
                f"Create {len(comparison['missing_tables'])} missing tables: " +
                ", ".join([t['table'] for t in comparison['missing_tables']])
            )
        
        if comparison['mismatched_tables']:
            comparison['recommendations'].append(
                f"Fix {len(comparison['mismatched_tables'])} tables with mismatched columns"
            )
        
        return comparison
    
    def check_table_compatibility(self, table_name: str) -> bool:
        """Check if existing table is compatible with required structure"""
        if table_name not in self.inspector.get_table_names():
            return False
        
        # Check if table has essential columns
        columns = [col['name'] for col in self.inspector.get_columns(table_name)]
        
        if table_name == 'signals':
            required_columns = ['symbol', 'timeframe', 'direction', 'confidence', 'timestamp']
        elif table_name == 'logs':
            required_columns = ['pattern_type', 'confidence_score', 'timestamp']
        elif table_name == 'feedback':
            required_columns = ['signal_id', 'market_outcome']
        else:
            return False
        
        return all(col in columns for col in required_columns)
    
    def generate_migration_plan(self, comparison: Dict) -> str:
        """Generate a migration plan based on comparison results"""
        plan = []
        
        if comparison['missing_tables']:
            plan.append("## Missing Tables to Create:")
            for table in comparison['missing_tables']:
                plan.append(f"- {table['table']}: {table['description']}")
        
        if comparison['mismatched_tables']:
            plan.append("\n## Tables Requiring Updates:")
            for table in comparison['mismatched_tables']:
                plan.append(f"- {table['table']}:")
                if table['missing_columns']:
                    plan.append(f"  - Add columns: {', '.join(table['missing_columns'])}")
                if table['mismatched_columns']:
                    plan.append(f"  - Fix column types: {len(table['mismatched_columns'])} columns")
        
        return "\n".join(plan)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive database analysis report"""
        print("🔍 Starting database structure analysis...")
        
        # Scan existing structure
        existing_structure = self.scan_existing_structure()
        
        # Get required structure
        required_structure = self.get_required_structure()
        
        # Compare structures
        comparison = self.compare_structures(existing_structure, required_structure)
        
        # Generate migration plan
        migration_plan = self.generate_migration_plan(comparison)
        
        # Create report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_url': self.database_url,
            'existing_tables_count': len(existing_structure),
            'required_tables_count': len(required_structure),
            'existing_structure': existing_structure,
            'required_structure': required_structure,
            'comparison': comparison,
            'migration_plan': migration_plan,
            'summary': {
                'total_tables': len(existing_structure),
                'missing_tables': len(comparison['missing_tables']),
                'mismatched_tables': len(comparison['mismatched_tables']),
                'compatible_tables': len(comparison['existing_tables']) - len(comparison['mismatched_tables'])
            }
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = None) -> str:
        """Save analysis report to file"""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"database_analysis_report_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {filepath}")
        return filepath
    
    def print_summary(self, report: Dict):
        """Print a summary of the analysis"""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("📊 DATABASE STRUCTURE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Database URL: {report['database_url']}")
        print(f"Analysis Time: {report['timestamp']}")
        print()
        print(f"📋 Total Tables Found: {summary['total_tables']}")
        print(f"✅ Compatible Tables: {summary['compatible_tables']}")
        print(f"❌ Missing Tables: {summary['missing_tables']}")
        print(f"⚠️  Mismatched Tables: {summary['mismatched_tables']}")
        print()
        
        if summary['missing_tables'] > 0:
            print("🔧 RECOMMENDATIONS:")
            print("   - Create missing tables using Alembic migrations")
            print("   - Consider redesigning existing tables if needed")
        
        if summary['mismatched_tables'] > 0:
            print("   - Update existing tables to match required schema")
            print("   - Add missing columns and indexes")
        
        print("="*60)


def main():
    """Main function to run database analysis"""
    try:
        # Initialize scanner
        scanner = DatabaseScanner()
        
        # Generate report
        report = scanner.generate_report()
        
        # Print summary
        scanner.print_summary(report)
        
        # Save report
        filename = scanner.save_report(report)
        
        print(f"\n✅ Database analysis completed successfully!")
        print(f"📄 Detailed report saved to: {filename}")
        
        return report
        
    except Exception as e:
        print(f"❌ Error during database analysis: {e}")
        return None


if __name__ == "__main__":
    main()
