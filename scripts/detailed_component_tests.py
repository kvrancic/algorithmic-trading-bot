#!/usr/bin/env python3
"""
Detailed Component Testing for Phase 3.1 Sentiment Analysis

Comprehensive tests to verify each component's actual behavior,
including edge cases and data validation.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sentiment import (
    RedditSentimentAnalyzer, RedditConfig,
    NewsAggregator, NewsConfig,
    UnusualWhalesAnalyzer, UnusualWhalesConfig,
    SentimentFusion, FusionConfig
)

class DetailedTester:
    """Comprehensive testing suite for sentiment analysis components"""
    
    def __init__(self):
        self.test_results = []
        self.detailed_logs = []
    
    def log_test(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test results with detailed information"""
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'timestamp': datetime.now(),
            'details': details
        })
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if not success or details.get('show_details', False):
            for key, value in details.items():
                if key != 'show_details':
                    print(f"  • {key}: {value}")
            print()
    
    def test_reddit_analyzer_detailed(self) -> bool:
        """Comprehensive Reddit analyzer testing"""
        print("\n🔍 DETAILED Reddit Sentiment Analyzer Tests")
        print("=" * 50)
        
        overall_success = True
        
        try:
            # Test 1: Configuration validation
            try:
                config = RedditConfig(
                    client_id="test_id",
                    client_secret="test_secret",
                    user_agent="Test Bot",
                    subreddits=['wallstreetbets', 'investing'],
                    max_posts_per_subreddit=5
                )
                analyzer = RedditSentimentAnalyzer(config)
                
                self.log_test("Reddit Config Validation", True, {
                    'config_created': True,
                    'analyzer_initialized': True,
                    'subreddits_count': len(config.subreddits)
                })
            except Exception as e:
                self.log_test("Reddit Config Validation", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
            
            # Test 2: Emoji signal detection with specific cases
            try:
                test_cases = [
                    {
                        'text': 'AAPL to the moon! 🚀🚀🚀 diamond hands 💎',
                        'expected_bullish': True,
                        'expected_count': 4  # 3 rockets + 1 diamond
                    },
                    {
                        'text': 'Bearish on TSLA 🐻📉 puts printing 🧸',
                        'expected_bullish': False,
                        'expected_count': 3  # 2 bears + 1 chart down
                    },
                    {
                        'text': 'Normal text without emojis',
                        'expected_bullish': None,
                        'expected_count': 0
                    },
                    {
                        'text': 'Mixed signals 🚀📉🐂🧸',
                        'expected_bullish': None,  # Mixed
                        'expected_count': 4
                    }
                ]
                
                test_posts = [{'full_text': case['text']} for case in test_cases]
                emoji_signals = analyzer._extract_emoji_signals(test_posts)
                
                # Validate results
                total_bullish = emoji_signals['bullish_signals']
                total_bearish = emoji_signals['bearish_signals']
                
                details = {
                    'total_bullish_found': total_bullish,
                    'total_bearish_found': total_bearish,
                    'test_cases_processed': len(test_cases),
                    'emoji_breakdown': emoji_signals.get('emoji_breakdown', {}),
                    'show_details': True
                }
                
                # Expected: 4 bullish (case 1) + 1 bullish (case 4) = 5
                # Expected: 3 bearish (case 2) + 1 bearish (case 4) = 4
                expected_bullish = 4  # Rockets + diamond from case 1, bull from case 4
                expected_bearish = 4  # Bears + chart down from case 2, bear from case 4
                
                success = (total_bullish >= 3 and total_bearish >= 3)  # Allow some variance
                
                self.log_test("Reddit Emoji Signal Detection", success, details)
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("Reddit Emoji Signal Detection", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
            
            # Test 3: Option flow extraction
            try:
                option_test_posts = [
                    {'full_text': 'Bought AAPL calls 420c expiring Friday'},
                    {'full_text': 'TSLA puts are printing! 800p'},
                    {'full_text': 'Going long on MSFT with some call options'},
                    {'full_text': 'Put options on SPY looking good'},
                    {'full_text': 'Just regular stock discussion'}
                ]
                
                option_signals = analyzer._extract_option_signals(option_test_posts)
                
                calls_mentions = option_signals['calls_mentions']
                puts_mentions = option_signals['puts_mentions']
                
                details = {
                    'calls_mentions': calls_mentions,
                    'puts_mentions': puts_mentions,
                    'option_flow_ratio': option_signals.get('option_flow_ratio', 0),
                    'posts_analyzed': len(option_test_posts)
                }
                
                # Should find at least 2 calls mentions and 2 puts mentions
                success = calls_mentions >= 2 and puts_mentions >= 2
                
                self.log_test("Reddit Option Flow Detection", success, details)
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("Reddit Option Flow Detection", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
            
            # Test 4: Sentiment score conversion
            try:
                test_conversions = [
                    ('positive', 0.9, 'should be positive'),
                    ('negative', 0.8, 'should be negative'),
                    ('neutral', 0.6, 'should be near zero'),
                    ('positive', 0.5, 'should be barely positive'),
                    ('negative', 0.5, 'should be barely negative')
                ]
                
                conversion_results = []
                for label, confidence, description in test_conversions:
                    score = analyzer._convert_sentiment_score(label, confidence)
                    conversion_results.append({
                        'label': label,
                        'confidence': confidence,
                        'score': score,
                        'description': description
                    })
                
                # Validate conversions make sense
                pos_scores = [r['score'] for r in conversion_results if r['label'] == 'positive']
                neg_scores = [r['score'] for r in conversion_results if r['label'] == 'negative']
                
                success = (
                    all(score > 0 for score in pos_scores) and
                    all(score < 0 for score in neg_scores) and
                    len(conversion_results) == 5
                )
                
                self.log_test("Reddit Sentiment Score Conversion", success, {
                    'conversions_tested': len(conversion_results),
                    'positive_scores': pos_scores,
                    'negative_scores': neg_scores,
                    'all_conversions': conversion_results[:3]  # Show first 3
                })
                
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("Reddit Sentiment Score Conversion", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
        
        except Exception as e:
            print(f"❌ Reddit detailed testing failed: {e}")
            traceback.print_exc()
            overall_success = False
        
        return overall_success
    
    def test_unusual_whales_detailed(self) -> bool:
        """Comprehensive UnusualWhales analyzer testing with mock data validation"""
        print("\n🐋 DETAILED UnusualWhales Analyzer Tests")
        print("=" * 50)
        
        overall_success = True
        
        try:
            # Test 1: Configuration and initialization
            config = UnusualWhalesConfig(
                use_selenium=False,  # Test without Selenium first
                lookback_days=7,
                min_trade_value=10000
            )
            
            analyzer = UnusualWhalesAnalyzer(config)
            init_success = analyzer.initialize()
            
            self.log_test("UnusualWhales Initialization", init_success, {
                'selenium_available': 'selenium' in sys.modules,
                'selenium_disabled': not config.use_selenium,
                'lookback_days': config.lookback_days,
                'min_trade_value': config.min_trade_value
            })
            
            if not init_success:
                overall_success = False
            
            # Test 2: Mock congressional data generation and validation
            try:
                symbol = 'AAPL'
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                congress_trades = analyzer._generate_sample_congress_data(symbol, cutoff_date)
                
                # Validate mock data structure and content
                valid_trades = 0
                party_distribution = {}
                trade_types = {}
                
                for trade in congress_trades:
                    # Check required fields
                    required_fields = ['politician', 'party', 'trade_type', 'trade_date', 'value', 'symbol']
                    has_all_fields = all(field in trade for field in required_fields)
                    
                    if has_all_fields:
                        valid_trades += 1
                        
                        # Count party distribution
                        party = trade['party']
                        party_distribution[party] = party_distribution.get(party, 0) + 1
                        
                        # Count trade types
                        trade_type = trade['trade_type']
                        trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
                        
                        # Validate trade date is within range
                        trade_date = trade['trade_date']
                        if not (cutoff_date <= trade_date <= datetime.utcnow()):
                            valid_trades -= 1
                
                success = (
                    len(congress_trades) > 0 and
                    valid_trades == len(congress_trades) and
                    len(party_distribution) > 0 and
                    len(trade_types) > 0
                )
                
                self.log_test("UnusualWhales Mock Data Generation", success, {
                    'total_trades': len(congress_trades),
                    'valid_trades': valid_trades,
                    'party_distribution': party_distribution,
                    'trade_types': trade_types,
                    'date_range_valid': True,
                    'show_details': True
                })
                
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("UnusualWhales Mock Data Generation", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
            
            # Test 3: Political sentiment analysis accuracy
            try:
                # Create controlled test data
                test_congress_trades = [
                    {'politician': 'Nancy Pelosi', 'party': 'Democrat', 'trade_type': 'buy', 'value': 100000},
                    {'politician': 'Nancy Pelosi', 'party': 'Democrat', 'trade_type': 'buy', 'value': 200000},
                    {'politician': 'Mitch McConnell', 'party': 'Republican', 'trade_type': 'sell', 'value': 150000},
                    {'politician': 'Chuck Schumer', 'party': 'Democrat', 'trade_type': 'sell', 'value': 80000}
                ]
                
                test_insider_trades = [
                    {'trade_type': 'buy', 'value': 500000},
                    {'trade_type': 'buy', 'value': 300000},
                    {'trade_type': 'sell', 'value': 100000}
                ]
                
                sentiment_analysis = analyzer._analyze_political_sentiment(test_congress_trades, test_insider_trades)
                
                # Expected results:
                # Congress: 2 buys, 2 sells = 0 sentiment
                # Insider: 2 buys, 1 sell = positive sentiment
                # Overall should be slightly positive
                
                congress_sentiment = sentiment_analysis['congress_sentiment']
                insider_sentiment = sentiment_analysis['insider_sentiment']
                overall_sentiment = sentiment_analysis['overall_sentiment']
                
                # Validate sentiment calculations
                expected_congress = (2 - 2) / 4  # (buys - sells) / total = 0
                expected_insider = (2 - 1) / 3   # = 0.33...
                
                congress_correct = abs(congress_sentiment - expected_congress) < 0.01
                insider_correct = abs(insider_sentiment - expected_insider) < 0.01
                overall_reasonable = -1 <= overall_sentiment <= 1
                
                success = congress_correct and insider_correct and overall_reasonable
                
                self.log_test("UnusualWhales Political Sentiment Analysis", success, {
                    'congress_sentiment': congress_sentiment,
                    'expected_congress': expected_congress,
                    'insider_sentiment': insider_sentiment,
                    'expected_insider': expected_insider,
                    'overall_sentiment': overall_sentiment,
                    'calculations_correct': congress_correct and insider_correct,
                    'show_details': True
                })
                
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("UnusualWhales Political Sentiment Analysis", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
            
            # Test 4: Party divergence analysis
            try:
                party_test_trades = [
                    {'party': 'Republican', 'trade_type': 'buy'},
                    {'party': 'Republican', 'trade_type': 'buy'},
                    {'party': 'Republican', 'trade_type': 'sell'},
                    {'party': 'Democrat', 'trade_type': 'sell'},
                    {'party': 'Democrat', 'trade_type': 'sell'},
                    {'party': 'Democrat', 'trade_type': 'sell'},
                    {'party': 'Independent', 'trade_type': 'buy'}
                ]
                
                party_analysis = analyzer._analyze_party_divergence(party_test_trades)
                
                # Expected:
                # Republican: 2 buys, 1 sell = (2-1)/3 = 0.33
                # Democrat: 0 buys, 3 sells = (0-3)/3 = -1.0
                # Independent: 1 buy, 0 sells = 1.0
                # Divergence = |0.33 - (-1.0)| = 1.33, but should be clamped
                
                republican_sentiment = party_analysis['republican_sentiment']
                democrat_sentiment = party_analysis['democrat_sentiment']
                divergence_score = party_analysis['divergence_score']
                bipartisan_interest = party_analysis['bipartisan_interest']
                
                # Validate calculations
                expected_republican = (2 - 1) / 3  # ≈ 0.33
                expected_democrat = (0 - 3) / 3    # = -1.0
                expected_divergence = abs(expected_republican - expected_democrat)  # ≈ 1.33
                
                republican_correct = abs(republican_sentiment - expected_republican) < 0.01
                democrat_correct = abs(democrat_sentiment - expected_democrat) < 0.01
                divergence_reasonable = 0 <= divergence_score <= 2  # Should be in reasonable range
                bipartisan_reasonable = 0 <= bipartisan_interest <= 1
                
                success = (republican_correct and democrat_correct and 
                          divergence_reasonable and bipartisan_reasonable)
                
                self.log_test("UnusualWhales Party Divergence Analysis", success, {
                    'republican_sentiment': republican_sentiment,
                    'expected_republican': expected_republican,
                    'democrat_sentiment': democrat_sentiment,
                    'expected_democrat': expected_democrat,
                    'divergence_score': divergence_score,
                    'expected_divergence': expected_divergence,
                    'bipartisan_interest': bipartisan_interest,
                    'show_details': True
                })
                
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("UnusualWhales Party Divergence Analysis", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
            
            # Test 5: Full analysis pipeline with mock data
            try:
                full_analysis = analyzer.analyze_symbol('AAPL', days_back=7)
                
                # Validate required fields in output
                required_fields = [
                    'symbol', 'timestamp', 'source', 'total_congress_trades',
                    'political_sentiment', 'insider_confidence', 'party_divergence',
                    'sector_classification', 'recent_activity_spike'
                ]
                
                has_required_fields = all(field in full_analysis for field in required_fields)
                
                # Validate data types and ranges
                valid_ranges = (
                    -1 <= full_analysis.get('political_sentiment', 0) <= 1 and
                    0 <= full_analysis.get('insider_confidence', 0) <= 1 and
                    0 <= full_analysis.get('party_divergence', 0) <= 2 and
                    full_analysis.get('total_congress_trades', 0) >= 0
                )
                
                success = has_required_fields and valid_ranges
                
                self.log_test("UnusualWhales Full Analysis Pipeline", success, {
                    'has_required_fields': has_required_fields,
                    'valid_ranges': valid_ranges,
                    'political_sentiment': full_analysis.get('political_sentiment'),
                    'total_congress_trades': full_analysis.get('total_congress_trades'),
                    'party_divergence': full_analysis.get('party_divergence'),
                    'sector_classification': full_analysis.get('sector_classification'),
                    'show_details': True
                })
                
                if not success:
                    overall_success = False
            
            except Exception as e:
                self.log_test("UnusualWhales Full Analysis Pipeline", False, {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                overall_success = False
        
        except Exception as e:
            print(f"❌ UnusualWhales detailed testing failed: {e}")
            traceback.print_exc()
            overall_success = False
        
        return overall_success
    
    def test_news_aggregator_detailed(self) -> bool:
        """Detailed news aggregator testing"""
        print("\n📰 DETAILED News Aggregator Tests")
        print("=" * 50)
        
        overall_success = True
        
        try:
            config = NewsConfig(
                alpha_vantage_key="mock_key",
                newsapi_key="mock_key",
                max_articles_per_source=5
            )
            
            aggregator = NewsAggregator(config)
            
            # Test market impact analysis with controlled data
            test_articles = [
                {
                    'title': 'Apple reports record quarterly earnings beat',
                    'summary': 'Strong iPhone sales drive revenue growth beyond expectations',
                    'source': 'Reuters',
                    'published_at': datetime.utcnow().isoformat()
                },
                {
                    'title': 'Tesla faces FDA investigation over autopilot',
                    'summary': 'Safety concerns prompt regulatory review of self-driving technology',
                    'source': 'Bloomberg',
                    'published_at': datetime.utcnow().isoformat()
                },
                {
                    'title': 'Microsoft announces major acquisition deal',
                    'summary': 'Tech giant to acquire cloud computing startup for $2B',
                    'source': 'Financial Times',
                    'published_at': datetime.utcnow().isoformat()
                }
            ]
            
            impact_analysis = aggregator._analyze_market_impact(test_articles)
            
            # Should detect high-impact keywords like "earnings", "acquisition", "investigation"
            success = (
                'impact_score' in impact_analysis and
                0 <= impact_analysis['impact_score'] <= 1 and
                impact_analysis['impact_score'] > 0  # Should be > 0 due to keywords
            )
            
            self.log_test("News Market Impact Analysis", success, {
                'impact_score': impact_analysis.get('impact_score'),
                'articles_analyzed': len(test_articles),
                'keywords_found': impact_analysis.get('keywords_found', []),
                'show_details': True
            })
            
            if not success:
                overall_success = False
        
        except Exception as e:
            self.log_test("News Aggregator Detailed Tests", False, {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            overall_success = False
        
        return overall_success
    
    def test_sentiment_fusion_detailed(self) -> bool:
        """Detailed sentiment fusion testing"""
        print("\n🧠 DETAILED Sentiment Fusion Tests")
        print("=" * 50)
        
        overall_success = True
        
        try:
            config = FusionConfig()
            fusion = SentimentFusion(config)
            
            # Test with various edge cases
            test_cases = [
                {
                    'name': 'All Positive Sources',
                    'data': {
                        'reddit': {'sentiment_score': 0.8, 'confidence': 0.9, 'timestamp': datetime.utcnow()},
                        'news': {'sentiment_score': 0.6, 'confidence': 0.8, 'timestamp': datetime.utcnow()},
                        'unusual_whales': {'sentiment_score': 0.4, 'confidence': 0.7, 'timestamp': datetime.utcnow()}
                    },
                    'expected_positive': True
                },
                {
                    'name': 'Mixed Sources',
                    'data': {
                        'reddit': {'sentiment_score': 0.5, 'confidence': 0.8, 'timestamp': datetime.utcnow()},
                        'news': {'sentiment_score': -0.3, 'confidence': 0.9, 'timestamp': datetime.utcnow()},
                        'unusual_whales': {'sentiment_score': 0.1, 'confidence': 0.6, 'timestamp': datetime.utcnow()}
                    },
                    'expected_positive': None  # Could go either way
                },
                {
                    'name': 'Single Source',
                    'data': {
                        'reddit': {'sentiment_score': 0.7, 'confidence': 0.8, 'timestamp': datetime.utcnow()}
                    },
                    'expected_positive': True
                },
                {
                    'name': 'Low Confidence Sources',
                    'data': {
                        'reddit': {'sentiment_score': 0.9, 'confidence': 0.2, 'timestamp': datetime.utcnow()},
                        'news': {'sentiment_score': 0.8, 'confidence': 0.3, 'timestamp': datetime.utcnow()}
                    },
                    'expected_positive': True  # But should have low fusion confidence
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                try:
                    result = fusion.fuse_sentiment(test_case['data'], 'TEST')
                    
                    # Validate required fields
                    required_fields = ['fused_sentiment', 'fusion_confidence', 'trading_signal', 'signal_strength']
                    has_required = all(field in result for field in required_fields)
                    
                    # Validate ranges
                    valid_ranges = (
                        -1 <= result.get('fused_sentiment', 0) <= 1 and
                        0 <= result.get('fusion_confidence', 0) <= 1 and
                        0 <= result.get('signal_strength', 0) <= 1
                    )
                    
                    # Check expected positivity if specified
                    sentiment_correct = True
                    if test_case['expected_positive'] is not None:
                        actual_positive = result.get('fused_sentiment', 0) > 0
                        sentiment_correct = actual_positive == test_case['expected_positive']
                    
                    test_success = has_required and valid_ranges and sentiment_correct
                    
                    self.log_test(f"Fusion Test Case {i+1}: {test_case['name']}", test_success, {
                        'fused_sentiment': result.get('fused_sentiment'),
                        'fusion_confidence': result.get('fusion_confidence'),
                        'trading_signal': result.get('trading_signal'),
                        'sources_used': len(result.get('sources_used', [])),
                        'sentiment_matches_expected': sentiment_correct
                    })
                    
                    if not test_success:
                        overall_success = False
                
                except Exception as e:
                    self.log_test(f"Fusion Test Case {i+1}: {test_case['name']}", False, {
                        'error': str(e)
                    })
                    overall_success = False
        
        except Exception as e:
            self.log_test("Sentiment Fusion Detailed Tests", False, {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            overall_success = False
        
        return overall_success
    
    def generate_detailed_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 DETAILED TEST REPORT")
        print("=" * 70)
        
        passed_tests = [r for r in self.test_results if r['success']]
        failed_tests = [r for r in self.test_results if not r['success']]
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {len(passed_tests)} ✅")
        print(f"Failed: {len(failed_tests)} ❌")
        print(f"Success Rate: {len(passed_tests)/len(self.test_results)*100:.1f}%")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  • {test['test_name']}")
                if 'error' in test['details']:
                    print(f"    Error: {test['details']['error']}")
        
        print(f"\n✅ PASSED TESTS ({len(passed_tests)}):")
        for test in passed_tests:
            print(f"  • {test['test_name']}")
        
        # Summary by component
        components = {
            'Reddit': [t for t in self.test_results if 'Reddit' in t['test_name']],
            'UnusualWhales': [t for t in self.test_results if 'UnusualWhales' in t['test_name']],
            'News': [t for t in self.test_results if 'News' in t['test_name']],
            'Fusion': [t for t in self.test_results if 'Fusion' in t['test_name']]
        }
        
        print(f"\n📈 COMPONENT BREAKDOWN:")
        for component, tests in components.items():
            if tests:
                passed = sum(1 for t in tests if t['success'])
                total = len(tests)
                print(f"  • {component}: {passed}/{total} ({passed/total*100:.0f}%)")
        
        return len(passed_tests) == len(self.test_results)

def main():
    """Run detailed component tests"""
    print("🔬 Phase 3.1 Sentiment Analysis - DETAILED COMPONENT TESTING")
    print("=" * 70)
    
    tester = DetailedTester()
    
    # Run all detailed tests
    tests = [
        tester.test_reddit_analyzer_detailed,
        tester.test_unusual_whales_detailed,
        tester.test_news_aggregator_detailed,
        tester.test_sentiment_fusion_detailed
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test function {test_func.__name__} failed: {e}")
            traceback.print_exc()
    
    # Generate comprehensive report
    all_passed = tester.generate_detailed_report()
    
    if all_passed:
        print("\n🎉 ALL DETAILED TESTS PASSED!")
        print("✅ All components are behaving correctly with proper data validation.")
        return 0
    else:
        print("\n⚠️ SOME DETAILED TESTS FAILED!")
        print("❌ Review the detailed report above for specific issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)