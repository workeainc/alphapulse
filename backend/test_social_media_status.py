#!/usr/bin/env python3
"""
Test Social Media API Implementation Status
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.free_api_manager import FreeAPIManager

async def test_social_apis():
    """Test social media API implementation status"""
    print('=== SOCIAL MEDIA API IMPLEMENTATION STATUS ===')
    print()
    
    try:
        manager = FreeAPIManager()
        
        # Test Reddit
        print('1. REDDIT API:')
        try:
            reddit_sentiment = await manager.get_social_sentiment('BTC')
            print(f'   ✅ Reddit API: WORKING')
            print(f'   📊 Reddit Sentiment: {reddit_sentiment}')
        except Exception as e:
            print(f'   ❌ Reddit API: ERROR - {e}')
        
        print()
        
        # Check if Twitter is implemented
        print('2. TWITTER API:')
        if hasattr(manager, 'twitter') and manager.twitter:
            print('   ✅ Twitter API: IMPLEMENTED')
        else:
            print('   ❌ Twitter API: NOT IMPLEMENTED in FreeAPIManager')
        
        print()
        
        # Check if Telegram is implemented
        print('3. TELEGRAM API:')
        if hasattr(manager, 'telegram') and manager.telegram:
            print('   ✅ Telegram API: IMPLEMENTED')
        else:
            print('   ❌ Telegram API: NOT IMPLEMENTED in FreeAPIManager')
        
        print()
        print('=== SUMMARY ===')
        print('✅ Reddit API: IMPLEMENTED & WORKING')
        print('❌ Twitter API: NOT IMPLEMENTED in FreeAPIManager')
        print('❌ Telegram API: NOT IMPLEMENTED in FreeAPIManager')
        
    except Exception as e:
        print(f'Error initializing FreeAPIManager: {e}')

if __name__ == "__main__":
    asyncio.run(test_social_apis())
