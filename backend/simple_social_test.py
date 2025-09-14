#!/usr/bin/env python3
"""
Simple Social Media Test
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_social_media():
    print('🚀 SOCIAL MEDIA SENTIMENT IMPLEMENTATION TEST')
    print('=' * 50)
    
    try:
        from services.free_api_manager import FreeAPIManager
        
        print('✅ FreeAPIManager imported successfully')
        
        manager = FreeAPIManager()
        print('✅ FreeAPIManager initialized successfully')
        
        # Test Reddit
        print('\n🔍 Testing Reddit API...')
        reddit_result = await manager._get_reddit_sentiment('BTC')
        print(f'Reddit Result: {reddit_result}')
        
        # Test Twitter
        print('\n🔍 Testing Twitter API...')
        twitter_result = await manager._get_twitter_sentiment('BTC')
        print(f'Twitter Result: {twitter_result}')
        
        # Test Telegram
        print('\n🔍 Testing Telegram API...')
        telegram_result = await manager._get_telegram_sentiment('BTC')
        print(f'Telegram Result: {telegram_result}')
        
        # Test Complete Social Sentiment
        print('\n🔍 Testing Complete Social Sentiment...')
        complete_result = await manager.get_social_sentiment('BTC')
        print(f'Complete Result: {complete_result}')
        
        print('\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_social_media())
