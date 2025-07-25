#!/usr/bin/env python3
"""
VERIFY REDDIT IS ACTUALLY WORKING - NOT JUST SUPPRESSING ERRORS
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.sentiment.reddit_analyzer import RedditSentimentAnalyzer, RedditConfig

async def test_reddit_real_functionality():
    """Test if Reddit is ACTUALLY working and returning real data"""
    print("🔍 TESTING IF REDDIT ACTUALLY WORKS (NOT JUST ERROR SUPPRESSION)")
    print("=" * 70)
    
    # Check environment variables
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    if not all([client_id, client_secret, user_agent]):
        print("❌ Missing Reddit credentials!")
        return False
    
    try:
        # Create Reddit config
        reddit_config = RedditConfig(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            subreddits=['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
        )
        
        # Initialize Reddit analyzer
        reddit_analyzer = RedditSentimentAnalyzer(reddit_config)
        success = await reddit_analyzer.initialize()
        
        if not success:
            print("❌ Reddit initialization failed")
            return False
            
        print("✅ Reddit initialized successfully")
        
        # Test each subreddit individually
        test_subreddits = ['stocks', 'investing', 'SecurityAnalysis']  # Start with less problematic ones
        total_posts = 0
        working_subreddits = 0
        
        for subreddit in test_subreddits:
            print(f"\n📡 Testing r/{subreddit}...")
            try:
                posts = await reddit_analyzer.get_recent_posts(subreddit, limit=5)
                
                if posts and len(posts) > 0:
                    print(f"  ✅ SUCCESS: Got {len(posts)} posts from r/{subreddit}")
                    working_subreddits += 1
                    total_posts += len(posts)
                    
                    # Show actual post data to prove it's real
                    for i, post in enumerate(posts[:2]):  # Show first 2 posts
                        title = post.get('title', 'No title')[:60]
                        score = post.get('score', 0)
                        author = post.get('author', 'unknown')
                        print(f"    📝 Post {i+1}: '{title}...' (Score: {score}, Author: {author})")
                        
                else:
                    print(f"  ⚠️  NO DATA: r/{subreddit} returned {len(posts) if posts else 0} posts")
                    
            except Exception as e:
                print(f"  ❌ ERROR: r/{subreddit} failed: {e}")
        
        # Test sentiment analysis on a known symbol
        print(f"\n🤖 Testing sentiment analysis...")
        try:
            sentiment_result = await reddit_analyzer.analyze_symbol('AAPL', hours_back=24)
            print(f"  ✅ Sentiment analysis returned: {len(str(sentiment_result))} characters of data")
            print(f"  📊 Total posts analyzed: {sentiment_result.get('total_posts', 0)}")
            print(f"  📈 Sentiment score: {sentiment_result.get('sentiment_score', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Sentiment analysis failed: {e}")
        
        # Test trending symbols
        print(f"\n🔥 Testing trending symbols...")
        try:
            trending = await reddit_analyzer.get_trending_symbols(limit=5)
            if trending and len(trending) > 0:
                print(f"  ✅ Found {len(trending)} trending symbols")
                for symbol in trending[:3]:
                    print(f"    📈 {symbol.get('symbol', 'Unknown')}: {symbol.get('mentions', 0)} mentions")
            else:
                print(f"  ⚠️  No trending symbols found")
        except Exception as e:
            print(f"  ❌ Trending symbols failed: {e}")
        
        # Close connection
        await reddit_analyzer.close()
        
        # Final assessment
        print(f"\n" + "=" * 70)
        print(f"📊 FINAL REDDIT TEST RESULTS:")
        print(f"  Working subreddits: {working_subreddits}/{len(test_subreddits)}")
        print(f"  Total posts retrieved: {total_posts}")
        
        if total_posts > 0:
            print(f"  🎉 REDDIT IS ACTUALLY WORKING! Got real data from {working_subreddits} subreddits")
            return True
        else:
            print(f"  💥 REDDIT IS NOT WORKING! No real data retrieved - just error suppression")
            return False
            
    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_reddit_real_functionality())
    print(f"\n🔍 CONCLUSION: Reddit is {'ACTUALLY WORKING' if result else 'NOT WORKING (just suppressing errors)'}")