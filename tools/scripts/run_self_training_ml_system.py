#!/usr/bin/env python3
"""
Self-Training ML System Runner
Runs the complete self-training ML pipeline for news impact prediction
"""

import asyncio
import logging
import asyncpg
import sys
import os
from datetime import datetime, timedelta
from typing import List, Optional

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.self_training_ml_orchestrator import SelfTrainingMLOrchestrator
from services.auto_labeling_service import LabelingConfig
from services.feature_engineering_service_simple import FeatureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('self_training_ml.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SelfTrainingMLRunner:
    """Runner for the self-training ML system"""
    
    def __init__(self):
        self.db_pool = None
        self.orchestrator = None
        
    async def initialize(self):
        """Initialize the system"""
        try:
            # Create database connection pool
            self.db_pool = await asyncpg.create_pool(
                "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse",
                min_size=5,
                max_size=20
            )
            logger.info("[SUCCESS] Database connection pool created")
            
            # Initialize orchestrator
            self.orchestrator = SelfTrainingMLOrchestrator(
                db_pool=self.db_pool,
                labeling_config=LabelingConfig(),
                feature_config=FeatureConfig(),
                model_config=None  # Use defaults
            )
            logger.info("[SUCCESS] Self-training ML orchestrator initialized")
            
        except Exception as e:
            logger.error(f"[ERROR] Error initializing system: {e}")
            raise
    
    async def run_database_migration(self):
        """Run the database migration for self-training ML tables"""
        try:
            logger.info("[MIGRATION] Running database migration...")
            
            # Run migration directly
            import subprocess
            result = subprocess.run([
                'python', 'backend/database/migrations/020_self_training_ml_system.py'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Migration failed: {result.stderr}")
            
            logger.info("[SUCCESS] Database migration completed")
            
        except Exception as e:
            logger.error(f"[ERROR] Error running database migration: {e}")
            raise
    
    async def run_full_pipeline(self, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              symbols: Optional[List[str]] = None) -> dict:
        """Run the complete self-training ML pipeline"""
        
        try:
            if not self.orchestrator:
                await self.initialize()
            
            logger.info("[PIPELINE] Starting full self-training ML pipeline...")
            
            # Run the pipeline
            result = await self.orchestrator.run_full_pipeline(
                start_time=start_time,
                end_time=end_time,
                symbols=symbols
            )
            
            logger.info(f"[SUCCESS] Pipeline completed with result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Error running pipeline: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_system_status(self) -> dict:
        """Get system status"""
        try:
            if not self.orchestrator:
                await self.initialize()
            
            status = await self.orchestrator.get_system_status()
            return status
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_auto_labeling_only(self, 
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None,
                                   symbols: Optional[List[str]] = None) -> dict:
        """Run only the auto-labeling step"""
        
        try:
            if not self.orchestrator:
                await self.initialize()
            
            logger.info("[LABELING] Running auto-labeling only...")
            
            # Generate labels
            labeled_data = await self.orchestrator.auto_labeling_service.generate_labels_for_news_articles(
                start_time, end_time, symbols
            )
            
            if not labeled_data:
                return {'status': 'no_data', 'message': 'No labeled data generated'}
            
            # Store labels
            stored_count = await self.orchestrator.auto_labeling_service.store_labeled_data(labeled_data)
            
            # Get statistics
            stats = await self.orchestrator.auto_labeling_service.get_labeling_statistics()
            
            return {
                'status': 'success',
                'labeled_data_count': len(labeled_data),
                'stored_count': stored_count,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error in auto-labeling: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_feature_engineering_only(self) -> dict:
        """Run only the feature engineering step"""
        
        try:
            if not self.orchestrator:
                await self.initialize()
            
            logger.info("[FEATURES] Running feature engineering only...")
            
            # Get labeled data that needs features
            query = """
                SELECT DISTINCT l.news_id, l.symbol
                FROM labels_news_market l
                LEFT JOIN feature_engineering_pipeline f ON l.news_id = f.news_id AND l.symbol = f.symbol
                WHERE f.news_id IS NULL
                AND l.confidence_score >= 0.5
                ORDER BY l.publish_time DESC
                LIMIT 100
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query)
            
            if not rows:
                return {'status': 'no_data', 'message': 'No data needs feature engineering'}
            
            processed_count = 0
            for row in rows:
                try:
                    # Get article data
                    article_data = await self.orchestrator._get_article_data(row['news_id'])
                    
                    if not article_data:
                        continue
                    
                    # Extract features
                    feature_set = await self.orchestrator.feature_engineering_service.extract_features_for_news_article(
                        row['news_id'],
                        row['symbol'],
                        article_data
                    )
                    
                    if feature_set:
                        # Store features
                        success = await self.orchestrator.feature_engineering_service.store_feature_set(feature_set)
                        if success:
                            processed_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error processing features for news {row['news_id']}: {e}")
                    continue
            
            return {
                'status': 'success',
                'processed_count': processed_count,
                'total_available': len(rows)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error in feature engineering: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_model_training_only(self) -> dict:
        """Run only the model training step"""
        
        try:
            if not self.orchestrator:
                await self.initialize()
            
            logger.info("[TRAINING] Running model training only...")
            
            # Train models for all targets
            training_results = await self.orchestrator._train_models_for_all_targets()
            
            if not training_results:
                return {'status': 'no_data', 'message': 'No models trained'}
            
            # Store results
            stored_count = await self.orchestrator._store_training_results(training_results)
            
            return {
                'status': 'success',
                'training_results_count': len(training_results),
                'stored_results_count': stored_count,
                'results': [
                    {
                        'model_name': result.model_name,
                        'target_variable': result.target_variable,
                        'model_type': result.model_type,
                        'f1_score': result.f1_score,
                        'auc_score': result.auc_score
                    }
                    for result in training_results
                ]
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error in model training: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("[SUCCESS] Database connection pool closed")

async def main():
    """Main function"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Self-Training ML System Runner')
    parser.add_argument('--mode', choices=['full', 'migration', 'labeling', 'features', 'training', 'status'],
                       default='full', help='Mode to run')
    parser.add_argument('--start-time', type=str, help='Start time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-time', type=str, help='End time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to process')
    
    args = parser.parse_args()
    
    # Parse times
    start_time = None
    end_time = None
    
    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
    if args.end_time:
        end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
    
    # Create runner
    runner = SelfTrainingMLRunner()
    
    try:
        # Initialize
        await runner.initialize()
        
        # Run based on mode
        if args.mode == 'migration':
            await runner.run_database_migration()
            
        elif args.mode == 'labeling':
            result = await runner.run_auto_labeling_only(start_time, end_time, args.symbols)
            print(f"Labeling result: {result}")
            
        elif args.mode == 'features':
            result = await runner.run_feature_engineering_only()
            print(f"Feature engineering result: {result}")
            
        elif args.mode == 'training':
            result = await runner.run_model_training_only()
            print(f"Model training result: {result}")
            
        elif args.mode == 'status':
            status = await runner.get_system_status()
            print(f"System status: {status}")
            
        elif args.mode == 'full':
            result = await runner.run_full_pipeline(start_time, end_time, args.symbols)
            print(f"Full pipeline result: {result}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error in main: {e}")
        sys.exit(1)
        
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
