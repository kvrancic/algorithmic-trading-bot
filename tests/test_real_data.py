#!/usr/bin/env python3
"""
Test Real UnusualWhales Data - Demonstrate working congressional trade data retrieval
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sentiment.unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesConfig
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def test_real_data():
    print("🎯 Testing Real UnusualWhales Congressional Trade Data")
    print("=" * 60)
    
    config = UnusualWhalesConfig(
        use_playwright=True,
        use_selenium=False,
        lookback_days=30,  # Use longer period to trigger real scraping
        headless=True
    )
    
    analyzer = UnusualWhalesAnalyzer(config)
    
    if not analyzer.initialize():
        print("❌ Failed to initialize")
        return
    
    try:
        # Test for a popular stock symbol
        symbol = 'AAPL'
        print(f"🔍 Searching for {symbol} congressional trades...")
        
        result = analyzer.analyze_symbol(symbol, days_back=30)
        
        print("\n📊 RESULTS:")
        print("=" * 40)
        print(f"Symbol: {result['symbol']}")
        print(f"Total Congressional Trades: {result['total_congress_trades']}")
        print(f"Total Insider Trades: {result['total_insider_trades']}")
        print(f"Political Sentiment: {result['political_sentiment']:.3f}")
        print(f"Party Divergence: {result['party_divergence']:.3f}")
        print(f"Sector: {result['sector_classification']}")
        
        if result['total_congress_trades'] > 0:
            print(f"\n✅ SUCCESS! Found {result['total_congress_trades']} real congressional trades")
            print("\n🎯 Sample Trade Data:")
            for i, trade in enumerate(result['congress_trades_sample'][:3]):
                print(f"  {i+1}. {trade['politician']} ({trade['party']})")
                print(f"     • {trade['trade_type'].title()}: ${trade['value']:,.0f}")
                print(f"     • Date: {trade['trade_date'].strftime('%Y-%m-%d')}")
                print(f"     • Source: {trade['raw_data'].get('source', 'unknown')}")
        else:
            print(f"\nℹ️ No {symbol} trades found in recent data")
            print("   This is normal - not all stocks have recent congressional activity")
        
        # Test with a broader search by trying another symbol
        if result['total_congress_trades'] == 0:
            print(f"\n🔄 Trying broader search...")
            # Just test the JSON API directly to show it's working
            cutoff_date = analyzer._parse_date("2025-01-01")
            if cutoff_date:
                json_trades = analyzer._get_trades_from_json_api("", cutoff_date)  # Empty symbol = all trades
                print(f"📈 Found {len(json_trades)} total congressional trades in system")
                
                if json_trades:
                    print("\n🎯 Recent Congressional Activity (any symbol):")
                    for i, trade in enumerate(json_trades[:5]):
                        print(f"  {i+1}. {trade['politician']} ({trade['party']})")
                        print(f"     • Symbol: {trade['symbol']}")
                        print(f"     • {trade['trade_type'].title()}: ${trade['value']:,.0f}")
                        print(f"     • Date: {trade['trade_date'].strftime('%Y-%m-%d')}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    test_real_data()