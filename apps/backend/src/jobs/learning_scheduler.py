#!/usr/bin/env python3
"""
Learning Scheduler
Manages all scheduled learning jobs (daily, weekly, hourly)
"""

import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timezone
import asyncpg

from .daily_learning_job import run_daily_learning
from .weekly_retraining_job import run_weekly_retraining

logger = logging.getLogger(__name__)

class LearningScheduler:
    """
    Manages all scheduled learning jobs
    Automatically runs daily learning and weekly retraining
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.scheduler = AsyncIOScheduler(timezone='UTC')
        
        # Track job status
        self.job_history = {
            'daily_learning': [],
            'weekly_retraining': []
        }
        
        # Configuration
        self.daily_time = "00:00"  # Midnight UTC
        self.weekly_day = "sun"     # Sunday
        self.weekly_time = "02:00"  # 2 AM UTC
        
        logger.info("📅 Learning Scheduler initialized")
    
    def start(self):
        """
        Start all scheduled learning jobs
        """
        try:
            # Daily Learning Job (runs at 00:00 UTC)
            self.scheduler.add_job(
                self._run_daily_learning_job,
                CronTrigger(hour=0, minute=0, timezone='UTC'),
                id='daily_learning',
                name='Daily Learning Job',
                replace_existing=True
            )
            logger.info(f"✓ Daily Learning Job scheduled: {self.daily_time} UTC")
            
            # Weekly Retraining Job (runs Sunday at 02:00 UTC)
            self.scheduler.add_job(
                self._run_weekly_retraining_job,
                CronTrigger(day_of_week='sun', hour=2, minute=0, timezone='UTC'),
                id='weekly_retraining',
                name='Weekly Retraining Job',
                replace_existing=True
            )
            logger.info(f"✓ Weekly Retraining Job scheduled: Sunday {self.weekly_time} UTC")
            
            # Start scheduler
            self.scheduler.start()
            logger.info("✅ Learning Scheduler started - jobs will run automatically")
            
            # Log next run times
            self._log_next_run_times()
            
        except Exception as e:
            logger.error(f"❌ Error starting learning scheduler: {e}")
            raise
    
    async def _run_daily_learning_job(self):
        """
        Wrapper for daily learning job with error handling and logging
        """
        job_start = datetime.now(timezone.utc)
        logger.info("🌙 Daily Learning Job triggered by scheduler")
        
        try:
            result = await run_daily_learning(self.db_pool)
            
            # Record job execution
            execution_record = {
                'timestamp': job_start.isoformat(),
                'duration_seconds': (datetime.now(timezone.utc) - job_start).total_seconds(),
                'status': result.get('status', 'completed'),
                'outcomes_processed': result.get('outcomes_processed', 0)
            }
            
            self.job_history['daily_learning'].append(execution_record)
            
            # Keep only last 30 days of history
            if len(self.job_history['daily_learning']) > 30:
                self.job_history['daily_learning'] = self.job_history['daily_learning'][-30:]
            
            logger.info(f"✅ Daily Learning Job completed in {execution_record['duration_seconds']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Daily Learning Job failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _run_weekly_retraining_job(self):
        """
        Wrapper for weekly retraining job with error handling and logging
        """
        job_start = datetime.now(timezone.utc)
        logger.info("📅 Weekly Retraining Job triggered by scheduler")
        
        try:
            result = await run_weekly_retraining(self.db_pool)
            
            # Record job execution
            execution_record = {
                'timestamp': job_start.isoformat(),
                'duration_seconds': (datetime.now(timezone.utc) - job_start).total_seconds(),
                'status': result.get('status', 'completed'),
                'outcomes_analyzed': result.get('outcomes_analyzed', 0),
                'weights_deployed': result.get('weights_deployed', False)
            }
            
            self.job_history['weekly_retraining'].append(execution_record)
            
            # Keep only last 12 weeks of history
            if len(self.job_history['weekly_retraining']) > 12:
                self.job_history['weekly_retraining'] = self.job_history['weekly_retraining'][-12:]
            
            logger.info(f"✅ Weekly Retraining Job completed in {execution_record['duration_seconds']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Weekly Retraining Job failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _log_next_run_times(self):
        """
        Log next scheduled run times for all jobs
        """
        try:
            jobs = self.scheduler.get_jobs()
            
            for job in jobs:
                next_run = job.next_run_time
                if next_run:
                    logger.info(f"   Next {job.name}: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not log next run times: {e}")
    
    def stop(self):
        """
        Stop the scheduler and all jobs
        """
        try:
            self.scheduler.shutdown(wait=False)
            logger.info("🛑 Learning Scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {e}")
    
    def get_status(self) -> dict:
        """
        Get scheduler status and job history
        """
        try:
            jobs = self.scheduler.get_jobs()
            
            return {
                'scheduler_running': self.scheduler.running,
                'total_jobs': len(jobs),
                'jobs': [
                    {
                        'id': job.id,
                        'name': job.name,
                        'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
                    }
                    for job in jobs
                ],
                'job_history': self.job_history,
                'configuration': {
                    'daily_time': self.daily_time,
                    'weekly_day': self.weekly_day,
                    'weekly_time': self.weekly_time
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting scheduler status: {e}")
            return {'error': str(e)}
    
    async def run_daily_now(self):
        """
        Manually trigger daily learning job (for testing)
        """
        logger.info("🔧 Manually triggering daily learning job...")
        await self._run_daily_learning_job()
    
    async def run_weekly_now(self):
        """
        Manually trigger weekly retraining job (for testing)
        """
        logger.info("🔧 Manually triggering weekly retraining job...")
        await self._run_weekly_retraining_job()

