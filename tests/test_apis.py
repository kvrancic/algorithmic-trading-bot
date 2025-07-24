#!/usr/bin/env python3
"""
API Test Script for QuantumSentiment Trading Bot

Tests all data APIs to prove operational status:
- Alpaca (market data + account)
- Reddit (sentiment analysis)  
- Alpha Vantage (fundamentals)
- Crypto (price data)
"""

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_interface import DataInterface
from data.data_fetcher import DataFetcher, FetcherConfig

async def test_alpaca_api():
    """Test Alpaca API functionality"""
    print("🔍 Testing Alpaca API...")
    
    data_interface = DataInterface()
    
    if 'alpaca' not in data_interface.clients:
        print("❌ Alpaca client not initialized")
        return False
    
    alpaca_client = data_interface.clients['alpaca']
    
    try:
        # Test account info
        account = alpaca_client.get_account()
        print(f"✅ Account Status: {account.status}")
        print(f"✅ Buying Power: ${float(account.buying_power):,.2f}")
        print(f"✅ Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Test market data
        print("🔍 Testing market data...")
        start_date = datetime.now() - timedelta(days=5)
        df = alpaca_client.get_bars('AAPL', '1Day', start=start_date)
        print(f"✅ Market Data: Retrieved {len(df)} AAPL daily bars")
        
        if not df.empty:
            latest = df.iloc[-1]
            print(f"✅ Latest AAPL Close: ${latest['close']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpaca API Error: {e}")
        return False

async def test_reddit_api():
    """Test Reddit API functionality"""
    print("\n🔍 Testing Reddit API...")
    
    data_interface = DataInterface()
    
    if 'reddit' not in data_interface.clients:
        print("❌ Reddit client not initialized")
        return False
    
    try:
        # Test sentiment analysis
        sentiment = data_interface.get_sentiment_analysis('AAPL', sources=['reddit'], hours_back=24)
        
        print(f"✅ Reddit API Connected")
        print(f"✅ Sentiment Score: {sentiment['overall_sentiment']:.3f}")
        print(f"✅ Confidence: {sentiment['confidence']:.3f}")
        
        if 'reddit' in sentiment['sources']:
            reddit_data = sentiment['sources']['reddit']
            print(f"✅ Reddit Mentions: {reddit_data.get('mention_count', 0)}")
        
        # Test trending tickers
        trending = data_interface.get_trending_topics('reddit', limit=5)
        print(f"✅ Trending Tickers: {len(trending)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Reddit API Error: {e}")
        return False

async def test_alpha_vantage_api():
    """Test Alpha Vantage API functionality"""
    print("\n🔍 Testing Alpha Vantage API...")
    
    data_interface = DataInterface()
    
    if 'alphavantage' not in data_interface.clients:
        print("❌ Alpha Vantage client not initialized")
        return False
    
    try:
        # Test company fundamentals
        fundamentals = data_interface.get_company_fundamentals('AAPL')
        
        print(f"✅ Alpha Vantage API Connected")
        print(f"✅ Company: {fundamentals.get('Name', 'N/A')}")
        print(f"✅ Sector: {fundamentals.get('Sector', 'N/A')}")
        print(f"✅ Market Cap: ${fundamentals.get('MarketCapitalization', 'N/A')}")
        print(f"✅ PE Ratio: {fundamentals.get('PERatio', 'N/A')}")
        print(f"✅ Data Fields: {len(fundamentals)} total")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpha Vantage API Error: {e}")
        return False

async def test_crypto_api():
    """Test Crypto API functionality"""
    print("\n🔍 Testing Crypto API...")
    
    data_interface = DataInterface()
    
    if 'crypto' not in data_interface.clients:
        print("❌ Crypto client not initialized")
        return False
    
    try:
        # Test crypto quote
        btc_quote = data_interface.get_quote('BTC', asset_type='crypto')
        
        print(f"✅ Crypto API Connected")
        print(f"✅ BTC Price: ${btc_quote.get('price', 0):,.2f}")
        print(f"✅ Source: {btc_quote.get('source', 'unknown')}")
        
        # Test historical data
        btc_history = data_interface.get_historical_prices('BTC', timeframe='1D', days_back=7)
        print(f"✅ BTC Historical: {len(btc_history)} days")
        
        return True
        
    except Exception as e:
        print(f"❌ Crypto API Error: {e}")
        return False

async def test_data_fetcher():
    """Test DataFetcher high-level interface"""
    print("\n🔍 Testing DataFetcher...")
    
    try:
        data_interface = DataInterface()
        config = FetcherConfig(max_workers=2)
        fetcher = DataFetcher(config, data_interface)
        
        # Test market data fetching
        market_data = await fetcher.fetch_market_data(['AAPL'], timeframe='1Day', days_back=2)
        print(f"✅ Market Fetch: {len(market_data)} symbols")
        
        # Test sentiment data fetching
        sentiment_data = await fetcher.fetch_sentiment_data(['AAPL'], hours_back=24)
        print(f"✅ Sentiment Fetch: {len(sentiment_data)} symbols")
        
        # Test fundamental data fetching
        fundamental_data = await fetcher.fetch_fundamental_data(['AAPL'])
        print(f"✅ Fundamental Fetch: {len(fundamental_data)} symbols")
        
        # Test statistics
        stats = fetcher.get_statistics()
        print(f"✅ Success Rate: {stats['success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ DataFetcher Error: {e}")
        return False

async def test_download_with_storage():
    """Test full data download and storage pipeline"""
    print("\n🔍 Testing Download Pipeline...")
    
    try:
        # Import and run download script logic
        from scripts.download_historical_data import HistoricalDataDownloader
        
        # Create minimal config for testing
        config = {
            'symbols': ['AAPL'],
            'days': 1,
            'timeframes': ['1day'],  # Only test daily data
            'download_market_data': True,
            'download_sentiment_data': True,
            'download_fundamental_data': True,
            'generate_features': False,  # Skip features for speed
            'validate_data': False,      # Skip validation for speed
            'clean_data': False          # Skip cleaning for speed
        }
        
        downloader = HistoricalDataDownloader()
        downloader.config.update(config)
        
        # Test single symbol download
        await downloader._download_symbol_data('AAPL', 1)
        
        print("✅ Download Pipeline: Working")
        return True
        
    except Exception as e:
        print(f"❌ Download Pipeline Error: {e}")
        return False

async def main():
    """Run all API tests"""
    print("🚀 QUANTUMSENTIMENT API TESTING")
    print("=" * 50)
    
    tests = [
        ("Alpaca API", test_alpaca_api),
        ("Reddit API", test_reddit_api), 
        ("Alpha Vantage API", test_alpha_vantage_api),
        ("Crypto API", test_crypto_api),
        ("DataFetcher", test_data_fetcher),
        ("Download Pipeline", test_download_with_storage)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 API TEST RESULTS")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 50)
    print(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL APIs OPERATIONAL!")
    else:
        print("⚠️  Some APIs need attention")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    asyncio.run(main())