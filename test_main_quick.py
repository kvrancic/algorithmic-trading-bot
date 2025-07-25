#!/usr/bin/env python3
"""
Quick test of main pipeline components to verify all fixes
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.configuration import Config
from src.data.alpaca_client import AlpacaClient
from src.sentiment.reddit_analyzer import RedditSentimentAnalyzer
from src.sentiment.news_aggregator import NewsAggregator
from src.features.feature_pipeline import FeaturePipeline
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

async def test_main_components():
    """Test main pipeline components individually"""
    print("🔧 Testing main pipeline components...")
    
    # Load configuration
    config = Config('config/config.yaml')
    print("✅ Configuration loaded")
    
    # Test Alpaca client
    try:
        alpaca = AlpacaClient(config.data_sources.alpaca)
        await alpaca.initialize()
        
        # Get some market data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        market_data = await alpaca.get_market_data('AAPL', start_time, end_time, '1Hour')
        print(f"✅ Alpaca data: {len(market_data)} records for AAPL")
        
    except Exception as e:
        print(f"❌ Alpaca test failed: {e}")
        market_data = pd.DataFrame()
    
    # Test Reddit analyzer (should handle not being initialized)
    try:
        from src.sentiment.reddit_analyzer import RedditConfig
        reddit_config = RedditConfig(
            client_id=os.getenv('REDDIT_CLIENT_ID', ''),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'QuantumSentiment/1.0'),
            subreddits=config.data_sources.reddit.subreddits
        )
        reddit_analyzer = RedditSentimentAnalyzer(reddit_config)
        await reddit_analyzer.initialize()  # Initialize Reddit API
        
        # Test the new method we added
        posts = await reddit_analyzer.get_recent_posts('stocks')
        print(f"✅ Reddit analyzer: {len(posts)} recent posts initialized")
        
        # Close the session properly
        await reddit_analyzer.close()
        
    except Exception as e:
        print(f"❌ Reddit test failed: {e}")
    
    # Test News aggregator
    try:
        news_config = getattr(config.data_sources, 'news', None)
        if not news_config:
            # Create a minimal news config
            from src.sentiment.news_aggregator import NewsConfig
            news_config = NewsConfig()
        
        news_aggregator = NewsAggregator(news_config)
        
        # Test the new method we added
        articles = await news_aggregator.get_recent_articles()
        print(f"✅ News aggregator: {len(articles)} recent articles")
        
    except Exception as e:
        print(f"❌ News test failed: {e}")
    
    # Test Feature pipeline
    try:
        feature_pipeline = FeaturePipeline()
        
        if not market_data.empty:
            # Generate features with minimal data
            features = feature_pipeline.generate_features(
                symbol='AAPL',
                market_data=market_data.tail(100),  # Just last 100 rows
                sentiment_data=None
            )
            print(f"✅ Feature pipeline: {len(features['features'])} features generated")
            
            # Test the DataFrame conversion fix
            features_df = pd.DataFrame([features['features']])
            print(f"✅ DataFrame conversion: {features_df.shape}")
        else:
            print("⚠️  No market data to test feature pipeline")
        
    except Exception as e:
        print(f"❌ Feature pipeline test failed: {e}")
    
    print("\n🎉 Main pipeline component tests completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_main_components())