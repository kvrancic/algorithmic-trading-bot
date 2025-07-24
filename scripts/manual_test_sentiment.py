#!/usr/bin/env python3
"""
Manual Testing Script for Phase 3.1 Sentiment Analysis

Simple script to manually test sentiment analysis components.
Perfect for quick verification and debugging.

Usage:
  python manual_test_sentiment.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sentiment import (
    RedditSentimentAnalyzer, RedditConfig,
    NewsAggregator, NewsConfig,
    UnusualWhalesAnalyzer, UnusualWhalesConfig,
    SentimentFusion, FusionConfig
)

def test_reddit_analyzer():
    """Test Reddit sentiment analyzer with mock data"""
    print("\n🔍 Testing Reddit Sentiment Analyzer")
    print("-" * 40)
    
    try:
        # Create configuration
        config = RedditConfig(
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            user_agent="QuantumSentiment Test Bot",
            subreddits=['wallstreetbets', 'investing'],
            max_posts_per_subreddit=10
        )
        
        # Initialize analyzer
        analyzer = RedditSentimentAnalyzer(config)
        print("✅ Reddit analyzer initialized")
        
        # Test emoji signal detection
        test_posts = [
            {'full_text': 'AAPL to the moon! 🚀🚀🚀 diamond hands 💎'},
            {'full_text': 'Bearish on TSLA 🐻📉 puts printing'},
            {'full_text': 'MSFT looking strong 📈 bought calls'}
        ]
        
        emoji_signals = analyzer._extract_emoji_signals(test_posts)
        print(f"✅ Emoji signals: {emoji_signals['bullish_signals']} bullish, {emoji_signals['bearish_signals']} bearish")
        
        # Test option flow detection
        option_signals = analyzer._extract_option_signals(test_posts)
        print(f"✅ Option signals: {option_signals['calls_mentions']} calls, {option_signals['puts_mentions']} puts")
        
        return True
        
    except Exception as e:
        print(f"❌ Reddit analyzer test failed: {e}")
        return False

def test_news_aggregator():
    """Test news aggregator with mock data"""
    print("\n📰 Testing News Aggregator")
    print("-" * 40)
    
    try:
        # Create configuration
        config = NewsConfig(
            alpha_vantage_key="mock_av_key",
            newsapi_key="mock_newsapi_key",
            max_articles_per_source=5
        )
        
        # Initialize aggregator
        aggregator = NewsAggregator(config)
        print("✅ News aggregator initialized")
        
        # Test market impact analysis
        test_articles = [
            {
                'title': 'Apple announces record earnings beat',
                'summary': 'Apple reports stronger than expected quarterly results',
                'source': 'Reuters'
            },
            {
                'title': 'Tesla faces regulatory investigation',
                'summary': 'Authorities probe Tesla autopilot system',
                'source': 'Bloomberg'
            }
        ]
        
        impact_analysis = aggregator._analyze_market_impact(test_articles)
        print(f"✅ Market impact score: {impact_analysis['impact_score']:.3f}")
        
        # Test source credibility
        reuters_score = aggregator._get_source_credibility('Reuters')
        unknown_score = aggregator._get_source_credibility('Random Blog')
        print(f"✅ Source credibility: Reuters={reuters_score:.2f}, Unknown={unknown_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ News aggregator test failed: {e}")
        return False

def test_unusual_whales_analyzer():
    """Test UnusualWhales analyzer with mock data"""
    print("\n🐋 Testing UnusualWhales Analyzer")
    print("-" * 40)
    
    try:
        # Create configuration
        config = UnusualWhalesConfig(
            use_selenium=False,  # Disable for testing
            lookback_days=7
        )
        
        # Initialize analyzer
        analyzer = UnusualWhalesAnalyzer(config)
        print("✅ UnusualWhales analyzer initialized")
        
        # Test political sentiment analysis
        mock_congress_trades = [
            {'politician': 'Nancy Pelosi', 'party': 'Democrat', 'trade_type': 'buy', 'value': 100000},
            {'politician': 'Mitch McConnell', 'party': 'Republican', 'trade_type': 'sell', 'value': 200000}
        ]
        
        mock_insider_trades = [
            {'trade_type': 'buy', 'value': 500000}
        ]
        
        sentiment_analysis = analyzer._analyze_political_sentiment(mock_congress_trades, mock_insider_trades)
        print(f"✅ Political sentiment: {sentiment_analysis['overall_sentiment']:.3f}")
        
        # Test party divergence
        party_analysis = analyzer._analyze_party_divergence(mock_congress_trades)
        print(f"✅ Party divergence: {party_analysis['divergence_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ UnusualWhales analyzer test failed: {e}")
        return False

def test_sentiment_fusion():
    """Test sentiment fusion algorithm"""
    print("\n🧠 Testing Sentiment Fusion")
    print("-" * 40)
    
    try:
        # Create configuration
        config = FusionConfig()
        
        # Initialize fusion
        fusion = SentimentFusion(config)
        print("✅ Sentiment fusion initialized")
        
        # Test with mock sentiment data
        test_sentiment_data = {
            'reddit': {
                'sentiment_score': 0.6,
                'confidence': 0.8,
                'timestamp': datetime.utcnow(),
                'total_mentions': 50
            },
            'news': {
                'sentiment_score': 0.3,
                'confidence': 0.9,
                'timestamp': datetime.utcnow(),
                'total_mentions': 20
            },
            'unusual_whales': {
                'sentiment_score': -0.2,
                'confidence': 0.7,
                'timestamp': datetime.utcnow(),
                'total_mentions': 5
            }
        }
        
        # Perform fusion
        result = fusion.fuse_sentiment(test_sentiment_data, 'AAPL')
        
        print(f"✅ Fused sentiment: {result['fused_sentiment']:.3f}")
        print(f"✅ Fusion confidence: {result['fusion_confidence']:.3f}")
        print(f"✅ Trading signal: {result['trading_signal']}")
        print(f"✅ Sources used: {len(result['sources_used'])}")
        print(f"✅ Source agreement: {result['source_agreement']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment fusion test failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline"""
    print("\n🚀 Testing End-to-End Pipeline")
    print("-" * 40)
    
    try:
        symbol = 'AAPL'
        print(f"Testing complete pipeline for {symbol}...")
        
        # Generate mock data from each source
        reddit_data = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'source': 'reddit',
            'sentiment_score': 0.4,
            'confidence': 0.75,
            'total_mentions': 45,
            'emoji_signals': {'bullish_signals': 12, 'bearish_signals': 3}
        }
        
        news_data = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'source': 'news',
            'sentiment_score': 0.2,
            'confidence': 0.85,
            'total_mentions': 15,
            'market_impact_score': 0.6
        }
        
        political_data = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'source': 'unusual_whales',
            'sentiment_score': 0.1,
            'confidence': 0.65,
            'total_mentions': 8,
            'political_sentiment': 0.1
        }
        
        # Combine all sentiment data
        all_sentiment_data = {
            'reddit': reddit_data,
            'news': news_data,
            'unusual_whales': political_data
        }
        
        # Perform fusion
        fusion_config = FusionConfig()
        fusion = SentimentFusion(fusion_config)
        
        final_result = fusion.fuse_sentiment(all_sentiment_data, symbol)
        
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 Final Results for {symbol}:")
        print(f"   • Fused Sentiment: {final_result['fused_sentiment']:.3f}")
        print(f"   • Confidence: {final_result['fusion_confidence']:.3f}")
        print(f"   • Signal Strength: {final_result['signal_strength']:.3f}")
        print(f"   • Trading Signal: {final_result['trading_signal']}")
        print(f"   • Signal Direction: {final_result['signal_direction']}")
        print(f"   • Sources Used: {', '.join(final_result['sources_used'])}")
        print(f"   • Source Agreement: {final_result['source_agreement']:.3f}")
        
        if final_result['anomaly_detected']:
            print(f"   ⚠️ Anomaly detected (score: {final_result['anomaly_score']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end pipeline test failed: {e}")
        return False

def main():
    """Run all manual tests"""
    
    print("🚀 Phase 3.1 Multi-Source Sentiment Analysis - Manual Testing")
    print("=" * 70)
    
    tests = [
        ("Reddit Analyzer", test_reddit_analyzer),
        ("News Aggregator", test_news_aggregator), 
        ("UnusualWhales Analyzer", test_unusual_whales_analyzer),
        ("Sentiment Fusion", test_sentiment_fusion),
        ("End-to-End Pipeline", test_end_to_end_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Phase 3.1 sentiment analysis is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Review the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)