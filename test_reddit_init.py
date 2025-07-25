#!/usr/bin/env python3
"""
Test Reddit initialization specifically
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.sentiment.reddit_analyzer import RedditSentimentAnalyzer, RedditConfig

async def test_reddit_initialization():
    """Test Reddit initialization with environment variables"""
    print("🔧 Testing Reddit Initialization...")
    
    # Check environment variables
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    print(f"  📋 REDDIT_CLIENT_ID: {'✅ Set' if client_id else '❌ Missing'}")
    print(f"  📋 REDDIT_CLIENT_SECRET: {'✅ Set' if client_secret else '❌ Missing'}")
    print(f"  📋 REDDIT_USER_AGENT: {'✅ Set' if user_agent else '❌ Missing'}")
    
    if not all([client_id, client_secret, user_agent]):
        print("❌ Missing Reddit credentials!")
        return False
    
    try:
        # Create Reddit config
        reddit_config = RedditConfig(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            subreddits=['wallstreetbets', 'stocks']
        )
        
        # Initialize Reddit analyzer
        reddit_analyzer = RedditSentimentAnalyzer(reddit_config)
        print("  ✅ Reddit analyzer created")
        
        # Initialize connection
        success = reddit_analyzer.initialize()
        if success:
            print("  ✅ Reddit API initialized successfully")
            
            # Test basic functionality
            posts = await reddit_analyzer.get_recent_posts('stocks', limit=2)
            print(f"  ✅ Retrieved {len(posts)} posts from r/stocks")
            
            if posts:
                print(f"    📝 Sample post: '{posts[0].get('title', 'N/A')[:50]}...'")
            
            return True
        else:
            print("  ❌ Reddit initialization failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Reddit test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_reddit_initialization())
    if result:
        print("\n🎉 Reddit initialization working perfectly!")
    else:
        print("\n💥 Reddit initialization failed!")