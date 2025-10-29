#!/usr/bin/env python3
"""
CLI Interface for Active Learning Manual Labeling
Phase 3 - Priority 7: Active Learning Loop Implementation
"""

import asyncio
import logging
from typing import Optional
from active_learning_service import ActiveLearningService, PredictionLabel

logger = logging.getLogger(__name__)


class ActiveLearningCLI:
    """Command-line interface for manual labeling"""
    
    def __init__(self, service: ActiveLearningService):
        self.service = service
    
    async def start_cli(self):
        """Start the CLI interface"""
        print("🎯 AlphaPulse Active Learning CLI")
        print("=" * 50)
        
        while True:
            try:
                print("\nOptions:")
                print("1. View pending items")
                print("2. Label an item")
                print("3. Skip an item")
                print("4. View statistics")
                print("5. Exit")
                
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    await self._view_pending_items()
                elif choice == "2":
                    await self._label_item()
                elif choice == "3":
                    await self._skip_item()
                elif choice == "4":
                    await self._view_statistics()
                elif choice == "5":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _view_pending_items(self):
        """View pending items for labeling"""
        items = await self.service.get_pending_items(limit=10)
        
        if not items:
            print("📭 No pending items to label")
            return
        
        print(f"\n📋 Pending Items ({len(items)}):")
        print("-" * 80)
        
        for item in items:
            print(f"ID: {item.id} | Symbol: {item.symbol} | Timeframe: {item.timeframe}")
            print(f"Confidence: {item.prediction_confidence:.3f} | Predicted: {item.predicted_label}")
            print(f"Model: {item.model_id} | Priority: {item.priority}")
            print(f"Created: {item.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)
    
    async def _label_item(self):
        """Label an item"""
        queue_id = input("Enter queue item ID: ").strip()
        
        if not queue_id.isdigit():
            print("❌ Invalid ID")
            return
        
        print("\nAvailable labels:")
        for label in PredictionLabel:
            print(f"- {label.value}")
        
        manual_label = input("Enter manual label: ").strip().upper()
        if manual_label not in [label.value for label in PredictionLabel]:
            print("❌ Invalid label")
            return
        
        labeled_by = input("Enter your name/ID: ").strip()
        if not labeled_by:
            print("❌ Name/ID required")
            return
        
        labeling_notes = input("Enter notes (optional): ").strip()
        if not labeling_notes:
            labeling_notes = None
        
        success = await self.service.label_item(
            int(queue_id), manual_label, labeled_by, labeling_notes
        )
        
        if success:
            print("✅ Item labeled successfully!")
        else:
            print("❌ Failed to label item")
    
    async def _skip_item(self):
        """Skip an item"""
        queue_id = input("Enter queue item ID: ").strip()
        
        if not queue_id.isdigit():
            print("❌ Invalid ID")
            return
        
        reason = input("Enter reason for skipping (optional): ").strip()
        if not reason:
            reason = None
        
        success = await self.service.skip_item(int(queue_id), reason)
        
        if success:
            print("⏭️ Item skipped successfully!")
        else:
            print("❌ Failed to skip item")
    
    async def _view_statistics(self):
        """View statistics"""
        stats = await self.service.get_statistics()
        
        print(f"\n📊 Active Learning Statistics:")
        print("-" * 40)
        print(f"Total Items: {stats.total_items}")
        print(f"Pending: {stats.pending_items}")
        print(f"Labeled: {stats.labeled_items}")
        print(f"Processed: {stats.processed_items}")
        print(f"Skipped: {stats.skipped_items}")
        print(f"Avg Confidence: {stats.avg_confidence:.3f}")
        print(f"Confidence Range: {stats.min_confidence:.3f} - {stats.max_confidence:.3f}")
        
        if stats.label_distribution:
            print(f"\nLabel Distribution:")
            for label, count in stats.label_distribution.items():
                print(f"  {label}: {count}")
        
        if stats.model_distribution:
            print(f"\nModel Distribution:")
            for model, count in stats.model_distribution.items():
                print(f"  {model}: {count}")


async def main():
    """Main function to run the CLI"""
    try:
        # Initialize the active learning service
        service = ActiveLearningService()
        
        # Start the service
        await service.start()
        
        # Create and start CLI
        cli = ActiveLearningCLI(service)
        await cli.start_cli()
        
    except Exception as e:
        logger.error(f"❌ Error running CLI: {e}")
    finally:
        # Stop the service
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
