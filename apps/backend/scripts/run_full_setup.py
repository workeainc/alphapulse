#!/usr/bin/env python3
"""
Master execution script for historical data download and verification
Runs all setup steps in sequence with proper error handling
"""

import asyncio
import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Script paths
SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent

class SetupRunner:
    """Run full setup process"""
    
    def __init__(self):
        self.results = {
            'start_time': datetime.now(),
            'steps': {},
            'errors': [],
            'warnings': []
        }
    
    async def run_script(self, script_name: str, description: str) -> bool:
        """Run a Python script and capture results"""
        logger.info("=" * 80)
        logger.info(f"STEP: {description}")
        logger.info("=" * 80)
        
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            logger.error(f"❌ {error_msg}")
            self.results['errors'].append(error_msg)
            self.results['steps'][description] = {'status': 'failed', 'error': error_msg}
            return False
        
        try:
            # Run script as subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=SCRIPTS_DIR
            )
            
            stdout, stderr = await process.communicate()
            
            # Log output
            if stdout:
                logger.info(stdout.decode('utf-8', errors='replace'))
            if stderr:
                logger.warning(stderr.decode('utf-8', errors='replace'))
            
            if process.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                self.results['steps'][description] = {'status': 'success'}
                return True
            else:
                error_msg = f"Script exited with code {process.returncode}"
                logger.error(f"❌ {description} failed: {error_msg}")
                if stderr:
                    error_msg += f"\n{stderr.decode('utf-8', errors='replace')}"
                self.results['errors'].append(error_msg)
                self.results['steps'][description] = {'status': 'failed', 'error': error_msg}
                return False
                
        except Exception as e:
            error_msg = f"Error running {script_name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.results['errors'].append(error_msg)
            self.results['steps'][description] = {'status': 'failed', 'error': error_msg}
            return False
    
    async def run_test_suite(self) -> bool:
        """Run the automated test suite"""
        logger.info("=" * 80)
        logger.info("STEP: Running Automated Test Suite")
        logger.info("=" * 80)
        
        test_path = ROOT_DIR / 'tests' / 'test_historical_integration.py'
        
        if not test_path.exists():
            logger.warning(f"⚠️ Test file not found: {test_path}")
            self.results['warnings'].append(f"Test file not found: {test_path}")
            return True  # Don't fail setup if tests are missing
        
        try:
            # Run pytest or direct execution
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(test_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ROOT_DIR
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                logger.info(stdout.decode('utf-8', errors='replace'))
            if stderr:
                logger.warning(stderr.decode('utf-8', errors='replace'))
            
            if process.returncode == 0:
                logger.info("✅ Test suite completed successfully")
                self.results['steps']['Test Suite'] = {'status': 'success'}
                return True
            else:
                error_msg = f"Tests failed with code {process.returncode}"
                logger.warning(f"⚠️ {error_msg}")
                self.results['warnings'].append(error_msg)
                self.results['steps']['Test Suite'] = {'status': 'partial', 'error': error_msg}
                return True  # Don't fail setup on test failures
                
        except Exception as e:
            error_msg = f"Error running tests: {str(e)}"
            logger.warning(f"⚠️ {error_msg}")
            self.results['warnings'].append(error_msg)
            self.results['steps']['Test Suite'] = {'status': 'partial', 'error': error_msg}
            return True  # Don't fail setup on test errors
    
    def print_summary(self):
        """Print execution summary"""
        self.results['end_time'] = datetime.now()
        duration = (self.results['end_time'] - self.results['start_time']).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("SETUP EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        logger.info("\nStep Results:")
        for step_name, step_result in self.results['steps'].items():
            status = step_result['status']
            status_icon = '✅' if status == 'success' else '⚠️' if status == 'partial' else '❌'
            logger.info(f"  {status_icon} {step_name}: {status}")
            if 'error' in step_result:
                logger.info(f"     Error: {step_result['error']}")
        
        if self.results['errors']:
            logger.info(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                logger.error(f"  - {error}")
        
        if self.results['warnings']:
            logger.info(f"\n⚠️ Warnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                logger.warning(f"  - {warning}")
        
        # Determine overall success
        failed_steps = [s for s in self.results['steps'].values() if s['status'] == 'failed']
        
        if failed_steps:
            logger.error("\n❌ SETUP FAILED")
            logger.error("Some steps failed. Please review the errors above.")
            return False
        else:
            logger.info("\n✅ SETUP COMPLETED SUCCESSFULLY")
            logger.info("Historical data is ready. You can now start the backend.")
            return True
    
    async def run_all(self):
        """Run all setup steps in sequence"""
        logger.info("=" * 80)
        logger.info("ALPHAPULSE HISTORICAL DATA SETUP")
        logger.info("=" * 80)
        logger.info("This script will:")
        logger.info("  1. Verify database schema")
        logger.info("  2. Mark existing signals as test data")
        logger.info("  3. Download 1 year of historical data (~25 minutes)")
        logger.info("  4. Verify data integrity")
        logger.info("  5. Run automated tests")
        logger.info("=" * 80)
        logger.info("")
        
        # Step 0: Fix unique constraint if needed
        step0_success = await self.run_script(
            'fix_unique_constraint.py',
            'Fix Unique Constraint'
        )
        
        if not step0_success:
            logger.warning("⚠️ Could not fix unique constraint. Continuing anyway...")
        
        # Step 1: Database schema verification
        step1_success = await self.run_script(
            'verify_database_schema.py',
            'Database Schema Verification'
        )
        
        if not step1_success:
            logger.error("❌ Schema verification failed. Please fix database issues first.")
            self.print_summary()
            return False
        
        # Step 2: Mark test signals
        step2_success = await self.run_script(
            'mark_test_signals.py',
            'Mark Test Signals'
        )
        
        if not step2_success:
            logger.warning("⚠️ Could not mark test signals. Continuing anyway...")
        
        # Step 3: Download historical data (this takes ~25 minutes)
        logger.info("\n" + "=" * 80)
        logger.info("⚠️ IMPORTANT: Historical data download will take ~25 minutes")
        logger.info("This is normal. The script will download ~3.37 million candles.")
        logger.info("=" * 80)
        
        step3_success = await self.run_script(
            'download_1year_historical.py',
            'Download 1 Year Historical Data'
        )
        
        if not step3_success:
            logger.error("❌ Historical data download failed.")
            logger.error("You can retry by running: python scripts/download_1year_historical.py")
            self.print_summary()
            return False
        
        # Step 4: Verify data again
        step4_success = await self.run_script(
            'verify_database_schema.py',
            'Post-Download Verification'
        )
        
        # Step 5: Run tests
        await self.run_test_suite()
        
        # Print summary
        return self.print_summary()


async def main():
    """Main execution"""
    runner = SetupRunner()
    
    try:
        success = await runner.run_all()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Setup interrupted by user")
        runner.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        runner.print_summary()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

