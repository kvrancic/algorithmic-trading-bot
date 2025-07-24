#!/usr/bin/env python3
"""
Investigate UnusualWhales API Access - Verify what we're actually accessing
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sentiment.unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesConfig
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def investigate_api_access():
    print("🕵️ INVESTIGATING UnusualWhales API Access")
    print("=" * 60)
    print("⚠️  CHECKING: Are we accessing paid content or public data?")
    print("=" * 60)
    
    config = UnusualWhalesConfig(
        use_playwright=True,
        headless=True
    )
    
    analyzer = UnusualWhalesAnalyzer(config)
    
    if not analyzer.initialize():
        print("❌ Failed to initialize")
        return
    
    try:
        # Navigate to politics page like a normal user would
        politics_url = "https://unusualwhales.com/politics"
        print(f"🌐 Navigating to PUBLIC page: {politics_url}")
        
        analyzer.page.goto(politics_url, timeout=30000)
        analyzer.page.wait_for_timeout(3000)
        
        # Check what a normal user sees
        page_title = analyzer.page.title()
        current_url = analyzer.page.url
        print(f"📄 Page Title: {page_title}")
        print(f"🔗 Current URL: {current_url}")
        
        # Check for paywall indicators
        paywall_indicators = [
            "subscribe", "premium", "paid", "login required", 
            "upgrade", "billing", "payment", "trial"
        ]
        
        page_content = analyzer.page.content()
        paywall_detected = any(indicator in page_content.lower() for indicator in paywall_indicators)
        print(f"🚫 Paywall detected: {paywall_detected}")
        
        if paywall_detected:
            print("⚠️  WARNING: This page may contain paid content!")
        
        # Check the __NEXT_DATA__ script (this is what we're accessing)
        next_data_script = analyzer.page.query_selector('#__NEXT_DATA__')
        if next_data_script:
            print("\n📊 ANALYZING __NEXT_DATA__ SCRIPT:")
            print("=" * 40)
            next_data_content = next_data_script.inner_text()
            next_data = json.loads(next_data_content)
            
            # Show what kind of data is embedded
            build_id = next_data.get('buildId')
            print(f"🔧 Build ID: {build_id}")
            
            # Check if there's already trade data in the initial page load
            page_props = next_data.get('pageProps', {})
            embedded_trade_data = page_props.get('trade_data', [])
            
            if embedded_trade_data:
                print(f"📈 Trade data embedded in page: {len(embedded_trade_data)} trades")
                print("✅ This suggests the data is publicly available on page load!")
                
                # Show sample of what's embedded
                if embedded_trade_data:
                    sample = embedded_trade_data[0]
                    print(f"\n🎯 Sample embedded trade:")
                    for key, value in list(sample.items())[:8]:  # First 8 fields
                        print(f"   • {key}: {value}")
                    
            else:
                print("📈 No trade data embedded in initial page")
                print("⚠️  Data might come from separate API calls")
            
            # Now test the API endpoint we discovered
            api_url = f"https://unusualwhales.com/_next/data/{build_id}/politics.json"
            print(f"\n🔗 Testing API endpoint: {api_url}")
            
            try:
                # Make the API request
                response = analyzer.page.request.get(api_url)
                print(f"📡 API Response Status: {response.status}")
                
                if response.status == 200:
                    print("✅ API request successful - this appears to be public data!")
                    
                    api_data = response.json()
                    api_trade_data = api_data.get('pageProps', {}).get('trade_data', [])
                    print(f"📊 API returned {len(api_trade_data)} trades")
                    
                    # Check if API data matches embedded data
                    if len(api_trade_data) == len(embedded_trade_data):
                        print("✅ API data matches embedded data - confirms it's public!")
                    
                    # Check response headers for authentication requirements
                    headers = response.headers
                    auth_headers = ['authorization', 'x-api-key', 'authenticate']
                    auth_required = any(header in str(headers).lower() for header in auth_headers)
                    print(f"🔐 Authentication headers detected: {auth_required}")
                    
                    if not auth_required:
                        print("✅ No authentication headers - appears to be public data!")
                    
                elif response.status == 401:
                    print("❌ 401 Unauthorized - This is paid/protected content!")
                elif response.status == 403:
                    print("❌ 403 Forbidden - This is paid/protected content!")
                else:
                    print(f"⚠️  Unexpected status code: {response.status}")
                    
            except Exception as e:
                print(f"❌ API request failed: {e}")
        
        else:
            print("❌ No __NEXT_DATA__ script found")
        
        # Final assessment
        print(f"\n{'='*60}")
        print("📋 FINAL ASSESSMENT:")
        print("=" * 60)
        
        if not paywall_detected and embedded_trade_data and response.status == 200:
            print("✅ LIKELY PUBLIC DATA:")
            print("   • No paywall indicators found")
            print("   • Data embedded in public page")
            print("   • API responds without authentication")
            print("   • This appears to be the same data a user sees")
        else:
            print("⚠️  POTENTIALLY PAID DATA:")
            print("   • Consider checking UnusualWhales terms of service")
            print("   • May need to verify access permissions")
    
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    investigate_api_access()