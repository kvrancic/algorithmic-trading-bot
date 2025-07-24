#!/usr/bin/env python3
"""
Test Real UnusualWhales Scraping
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sentiment.unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesConfig
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def test_real_scraping():
    print("🔍 Testing Real UnusualWhales Scraping")
    print("=" * 50)
    
    # Test with real scraping enabled
    config = UnusualWhalesConfig(
        use_playwright=True,
        use_selenium=False,  # Only test Playwright 
        lookback_days=30,    # Use longer period to avoid mock data fallback
        headless=True
    )
    
    analyzer = UnusualWhalesAnalyzer(config)
    print(f'Analyzer: {analyzer}')
    
    # Try to initialize
    success = analyzer.initialize() 
    print(f'Initialization success: {success}')
    
    if success:
        print('🎉 Real scraping infrastructure is available!')
        
        # Test a quick analysis (will attempt real scraping but likely fail due to website structure)
        try:
            print(f"Attempting to scrape AAPL data from UnusualWhales...")
            result = analyzer.analyze_symbol('AAPL', days_back=30)
            
            print(f'Analysis completed:')
            print(f'  • Congress trades found: {result["total_congress_trades"]}')
            print(f'  • Insider trades found: {result["total_insider_trades"]}')
            print(f'  • Political sentiment: {result["political_sentiment"]:.3f}')
            
            if result['total_congress_trades'] > 0:
                print('✅ Real congressional data retrieved!')
                print(f'  • Sample trades: {result.get("congress_trades_sample", [])}')
            else:
                print('ℹ️ No real congressional data found')
                print('   This is expected - the current implementation uses placeholder')
                print('   HTML selectors that would need to match UnusualWhales actual structure')
                
            if result['total_insider_trades'] > 0:
                print('✅ Real insider data retrieved!')
            else:
                print('ℹ️ No real insider data found (expected)')
                
        except Exception as e:
            print(f'Analysis failed: {e}')
            import traceback
            traceback.print_exc()
        finally:
            analyzer.cleanup()
            print("🧹 Cleanup completed")
    else:
        print('❌ Real scraping not available')
    
    return success

if __name__ == "__main__":
    test_real_scraping()