#!/usr/bin/env python3
"""
Test Reddit API Connection

Quick script to verify your Reddit API credentials are working.
Run: python scripts/test_reddit_api.py
"""

import os
import sys
from pathlib import Path
import praw
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def test_reddit_connection():
    """Test Reddit API connection"""
    
    # Get credentials from environment
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET') 
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    print("🔍 Testing Reddit API Connection...")
    print(f"Client ID: {client_id[:8]}... (showing first 8 chars)")
    print(f"User Agent: {user_agent}")
    
    if not all([client_id, client_secret, user_agent]):
        print("❌ Missing Reddit API credentials in .env file")
        print("Please check your .env file has:")
        print("  REDDIT_CLIENT_ID=your_client_id")
        print("  REDDIT_CLIENT_SECRET=your_client_secret")
        print("  REDDIT_USER_AGENT=YourBot/1.0 by /u/yourusername")
        return False
    
    try:
        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Test by getting WSB info
        print("\n📡 Connecting to Reddit...")
        wsb = reddit.subreddit('wallstreetbets')
        print(f"✅ Successfully connected to r/{wsb.display_name}")
        print(f"📊 Subreddit has {wsb.subscribers:,} subscribers")
        
        # Test getting a few posts
        print("\n📋 Testing post retrieval...")
        hot_posts = list(wsb.hot(limit=3))
        print(f"✅ Retrieved {len(hot_posts)} hot posts:")
        
        for i, post in enumerate(hot_posts, 1):
            print(f"  {i}. {post.title[:60]}...")
            print(f"     👍 {post.score} upvotes, 💬 {post.num_comments} comments")
        
        print("\n🎉 Reddit API test successful!")
        print("✅ Your credentials are working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Reddit API test failed: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Double-check your Client ID and Secret from Reddit app")
        print("2. Make sure you created a 'script' type app")
        print("3. Verify your User Agent follows format: 'AppName/1.0 by /u/username'")
        print("4. Check if your Reddit app is still active")
        return False

if __name__ == "__main__":
    success = test_reddit_connection()
    sys.exit(0 if success else 1)