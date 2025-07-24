#!/usr/bin/env python3
"""
Quick script to show real UnusualWhales congressional trade data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sentiment.unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesConfig
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def show_real_data():
    print("📊 Real UnusualWhales Congressional Trade Data")
    print("=" * 50)
    
    config = UnusualWhalesConfig(use_playwright=True, headless=True)
    analyzer = UnusualWhalesAnalyzer(config)
    
    if not analyzer.initialize():
        print("❌ Failed to initialize")
        return
    
    try:
        # Get all recent trades (no symbol filter)
        cutoff_date = analyzer._parse_date("2025-06-01")
        trades = analyzer._get_trades_from_json_api("", cutoff_date)
        
        print(f"✅ Found {len(trades)} real congressional trades")
        print("\n🎯 Recent Congressional Trades:")
        print("-" * 50)
        
        for i, trade in enumerate(trades[:10]):  # Show first 10
            print(f"{i+1:2d}. {trade['politician']}")
            print(f"    Symbol: {trade['symbol']}")
            print(f"    Action: {trade['trade_type'].title()}")
            print(f"    Value:  ${trade['value']:,.0f}")
            print(f"    Date:   {trade['trade_date'].strftime('%Y-%m-%d')}")
            print(f"    Party:  {trade['party']}")
            print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    show_real_data()