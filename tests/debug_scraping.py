#!/usr/bin/env python3
"""
Debug UnusualWhales Scraping - Show detailed information about what's on the page
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sentiment.unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesConfig
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def debug_scraping():
    print("🕵️ Debugging UnusualWhales Scraping")
    print("=" * 50)
    
    config = UnusualWhalesConfig(
        use_playwright=True,
        use_selenium=False,
        lookback_days=30,
        headless=False  # Make it visible for debugging
    )
    
    analyzer = UnusualWhalesAnalyzer(config)
    
    if not analyzer.initialize():
        print("❌ Failed to initialize")
        return
    
    try:
        # Navigate to politics page
        politics_url = "https://unusualwhales.com/politics"
        print(f"🔗 Navigating to: {politics_url}")
        
        analyzer.page.goto(politics_url, timeout=30000)
        analyzer.page.wait_for_timeout(3000)  # Wait for JS hydration
        
        print("📄 Page loaded, analyzing content...")
        
        # Check for __NEXT_DATA__ script
        next_data_script = analyzer.page.query_selector('#__NEXT_DATA__')
        if next_data_script:
            print("✅ Found __NEXT_DATA__ script for JSON API approach")
            try:
                import json
                next_data_content = next_data_script.inner_text()
                next_data = json.loads(next_data_content)
                build_id = next_data.get('buildId')
                print(f"📊 Build ID: {build_id}")
                
                if build_id:
                    api_url = f"https://unusualwhales.com/_next/data/{build_id}/politics.json"
                    print(f"🔗 API URL: {api_url}")
                    
                    # Try to fetch from API
                    try:
                        response = analyzer.page.request.get(api_url)
                        print(f"📡 API Response Status: {response.status}")
                        if response.status == 200:
                            api_data = response.json()
                            trade_data = api_data.get('pageProps', {}).get('trade_data', [])
                            print(f"📈 Found {len(trade_data)} trades in JSON API")
                            if trade_data:
                                print("🎯 Sample trade data:")
                                sample = trade_data[0]
                                for key, value in sample.items():
                                    print(f"   • {key}: {value}")
                        else:
                            print(f"❌ API request failed with status {response.status}")
                    except Exception as e:
                        print(f"❌ API request failed: {e}")
            except Exception as e:
                print(f"❌ Failed to parse __NEXT_DATA__: {e}")
        else:
            print("❌ No __NEXT_DATA__ script found")
        
        # Check for "see all" link
        see_all_elements = analyzer.page.query_selector_all("text=see all")
        print(f"🔍 Found {len(see_all_elements)} 'see all' elements")
        
        if see_all_elements:
            print("🖱️ Clicking 'see all' link...")
            analyzer.page.click("text=see all")
            analyzer.page.wait_for_timeout(2000)
            print("✅ Clicked 'see all' link")
        
        # Check for tables
        tables = analyzer.page.query_selector_all("table")
        print(f"📊 Found {len(tables)} table elements")
        
        tbody_elements = analyzer.page.query_selector_all("tbody")
        print(f"📊 Found {len(tbody_elements)} tbody elements")
        
        table_rows = analyzer.page.query_selector_all("tbody tr")
        print(f"📊 Found {len(table_rows)} table rows")
        
        if table_rows:
            print("🎯 Analyzing first few rows:")
            for i, row in enumerate(table_rows[:3]):  # Check first 3 rows
                print(f"   Row {i + 1}:")
                try:
                    # Try to extract data from each column
                    cells = row.query_selector_all("td")
                    print(f"     • Cells found: {len(cells)}")
                    
                    for j, cell in enumerate(cells[:4]):  # First 4 columns
                        text = cell.inner_text().strip()
                        print(f"     • Column {j + 1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    
                    # Try specific selectors
                    name_link = row.query_selector("td:nth-child(1) a")
                    if name_link:
                        print(f"     ✅ Politician name link found: '{name_link.inner_text().strip()}'")
                    else:
                        print("     ❌ No politician name link found")
                        
                except Exception as e:
                    print(f"     ❌ Error analyzing row: {e}")
        else:
            print("❌ No table rows found")
            
            # Check what elements are actually on the page
            print("\n🔍 Checking for other possible containers:")
            divs_with_text = analyzer.page.query_selector_all("div")
            text_content = analyzer.page.inner_text()
            if "congress" in text_content.lower() or "trade" in text_content.lower():
                print("✅ Page contains congress/trade related text")
            else:
                print("❌ Page doesn't seem to contain congress/trade text")
        
        print(f"\n📝 Page title: {analyzer.page.title()}")
        print(f"🔗 Current URL: {analyzer.page.url}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    debug_scraping()