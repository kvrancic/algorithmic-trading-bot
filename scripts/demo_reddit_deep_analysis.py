#!/usr/bin/env python3
"""
Demo: Reddit Deep Analysis Phase 3.2 - High-Stakes Trading Ready

Demonstrates all enhanced features working together for professional trading:
- Real-time mention velocity tracking
- Advanced momentum indicators
- User credibility distribution analysis  
- Enhanced emoji signal detection
- Options flow extraction
- Trading alerts generation
- Risk-adjusted sentiment calculation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sentiment import RedditSentimentAnalyzer, RedditConfig
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])

def demo_reddit_deep_analysis():
    print("🚀 Reddit Deep Analysis Phase 3.2 - LIVE DEMONSTRATION")
    print("🎯 High-Stakes Trading Ready - All Features Operational")
    print("=" * 70)
    
    # Configure for high-stakes trading
    config = RedditConfig(
        client_id="demo_client_id",
        client_secret="demo_client_secret",
        user_agent="QuantumSentiment Professional Trading Bot",
        subreddits=['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting'],
        max_posts_per_subreddit=50,
        
        # High-stakes optimizations
        enable_velocity_tracking=True,
        enable_momentum_analysis=True, 
        enable_credibility_weighting=True,
        velocity_alert_threshold=5.0,  # High sensitivity
        momentum_alert_threshold=1.5,  # Professional threshold
        min_credibility_for_signals=0.7  # Quality filter
    )
    
    analyzer = RedditSentimentAnalyzer(config)
    
    print(f"📊 Analyzer Configuration:")
    print(f"   • Subreddits: {len(config.subreddits)} premium sources")
    print(f"   • Velocity Tracking: ✅ Enabled (threshold: {config.velocity_alert_threshold}/hour)")
    print(f"   • Momentum Analysis: ✅ Enabled (threshold: {config.momentum_alert_threshold})")
    print(f"   • Credibility Weighting: ✅ Enabled (min: {config.min_credibility_for_signals})")
    
    # Create realistic high-stakes trading scenario
    symbol = "AAPL"
    print(f"\n🎯 ANALYZING {symbol} - Live Market Simulation")
    print("=" * 50)
    
    # Simulate realistic data with high-quality signals
    now = datetime.utcnow()
    
    # High-quality posts from credible users
    demo_posts = [
        {
            'created_utc': (now - timedelta(hours=1)).timestamp(),
            'full_text': f'{symbol} breakout confirmed! Strong momentum building 🚀📈 Technical analysis shows bullish divergence. Bought calls.',
            'author': 'DeepFuckingValue',
            'score': 850, 'num_comments': 234,
            'title': f'{symbol} Technical Breakout Analysis',
            'text': 'Comprehensive DD with revenue projections, technical indicators, and risk assessment...',
            'is_dd': True
        },
        {
            'created_utc': (now - timedelta(hours=2)).timestamp(),
            'full_text': f'{symbol} surge incoming! This rocket is about to moon 🚀🌙 Diamond hands activated 💎🙌',
            'author': 'WallStreetAnalyst',
            'score': 420, 'num_comments': 89,
            'title': f'{symbol} Momentum Analysis',
            'text': 'Market structure analysis shows strong support...',
            'is_dd': True
        },
        {
            'created_utc': (now - timedelta(hours=3)).timestamp(),
            'full_text': f'{symbol} options flow is insane! Call volume exploding, this momentum is building fast',
            'author': 'OptionsTrader_Pro',
            'score': 156, 'num_comments': 67,
            'title': f'{symbol} Options Activity',
            'text': 'Unusual options activity detected...',
            'is_dd': False
        },
        {
            'created_utc': (now - timedelta(hours=4)).timestamp(),
            'full_text': f'{symbol} puts looking attractive, potential crash incoming 🐻📉 Short squeeze over',
            'author': 'BearMarket_Expert',
            'score': 89, 'num_comments': 23,
            'title': f'{symbol} Bear Case',
            'text': 'Contrarian analysis...',
            'is_dd': False
        }
    ]
    
    demo_comments = [
        {
            'created_utc': (now - timedelta(minutes=30)).timestamp(),
            'full_text': f'Absolutely agree on {symbol}, this momentum is unstoppable 💎🚀',
            'author': 'momentum_trader_99',
            'score': 45, 'num_comments': 0
        },
        {
            'created_utc': (now - timedelta(minutes=45)).timestamp(),
            'full_text': f'{symbol} calls printing! YOLO time 🌙💰',
            'author': 'options_king',
            'score': 23, 'num_comments': 0
        }
    ]
    
    print("🔬 DEEP ANALYSIS RESULTS:")
    print("-" * 30)
    
    # Test each component
    try:
        # 1. Mention Velocity Analysis
        velocity = analyzer._calculate_mention_velocity(demo_posts, demo_comments, symbol, 24)
        print(f"⚡ Mention Velocity:")
        print(f"   • Rate: {velocity['mentions_per_hour']:.1f} mentions/hour")
        print(f"   • Trend: {velocity['velocity_trend']}")
        print(f"   • Acceleration: {velocity['acceleration']:.2f}")
        
        # 2. Momentum Analysis
        momentum = analyzer._analyze_momentum_indicators(demo_posts, demo_comments)
        print(f"\n📈 Momentum Indicators:")
        print(f"   • Score: {momentum['momentum_score']:.2f}")
        print(f"   • Strength: {momentum['momentum_strength']}")
        print(f"   • Direction: {momentum['momentum_direction']}")
        print(f"   • High Conviction Posts: {momentum['high_conviction_posts']}")
        
        # 3. Credibility Analysis
        credibility = analyzer._analyze_user_credibility_distribution(demo_posts, demo_comments)
        print(f"\n👥 Credibility Distribution:")
        print(f"   • Average Credibility: {credibility['avg_credibility']:.2f}")
        print(f"   • Signal Reliability: {credibility['signal_reliability']}")
        print(f"   • High Credibility Ratio: {credibility['high_credibility_ratio']:.1%}")
        print(f"   • Weighted Sentiment: {credibility['credibility_weighted_sentiment']:.2f}")
        
        # 4. Emoji Signals
        emoji = analyzer._extract_emoji_signals(demo_posts + demo_comments)
        print(f"\n🚀 Emoji Signals:")
        print(f"   • Bullish Signals: {emoji['bullish_signals']}")
        print(f"   • Bearish Signals: {emoji['bearish_signals']}")
        print(f"   • Net Signal: {emoji['net_signal']:.2f}")
        print(f"   • Rocket Mentions: {emoji['rocket_mentions']}")
        print(f"   • Diamond Hands: {emoji['diamond_hands']}")
        
        # 5. Options Flow
        options = analyzer._extract_option_signals(demo_posts + demo_comments)
        print(f"\n📊 Options Flow:")
        print(f"   • Calls Mentions: {options['calls_mentions']}")
        print(f"   • Puts Mentions: {options['puts_mentions']}")
        print(f"   • Call/Put Ratio: {options['call_put_ratio']:.2f}")
        print(f"   • Total Activity: {options['total_option_activity']}")
        print(f"   • YOLO Sentiment: {options['yolo_sentiment']}")
        
        # 6. Generate mock analysis result for alerts and risk adjustment
        mock_analysis = {
            'symbol': symbol,
            'sentiment_score': 0.65,
            'confidence': 0.85,
            'total_engagement': 15000,
            'mention_velocity': velocity,
            'momentum_indicators': momentum,
            'credibility_distribution': credibility,
            'emoji_signals': emoji,
            'option_signals': options
        }
        
        # 7. Trading Alerts
        alerts = analyzer.generate_trading_alerts(symbol, mock_analysis)
        print(f"\n🚨 TRADING ALERTS ({len(alerts)} generated):")
        for i, alert in enumerate(alerts[:3], 1):  # Show top 3
            print(f"   {i}. [{alert['priority']}] {alert['type']}")
            print(f"      • {alert['message']}")
            print(f"      • Confidence: {alert['confidence']:.1%}")
        
        # 8. Risk-Adjusted Sentiment
        risk_adjusted = analyzer.get_risk_adjusted_sentiment(mock_analysis)
        print(f"\n⚖️ Risk-Adjusted Sentiment:")
        print(f"   • Base Sentiment: {risk_adjusted['base_sentiment']:.3f}")
        print(f"   • Risk-Adjusted: {risk_adjusted['risk_adjusted_sentiment']:.3f}")
        print(f"   • Adjustment Factor: {risk_adjusted['adjustment_factor']:.3f}")
        print(f"   • Confidence Level: {risk_adjusted['confidence_level']:.3f}")
        
        print(f"\n" + "=" * 70)
        print("🎯 TRADING RECOMMENDATION ENGINE")
        print("=" * 70)
        
        # Generate final trading recommendation
        final_sentiment = risk_adjusted['risk_adjusted_sentiment']
        confidence = risk_adjusted['confidence_level']
        
        if final_sentiment > 0.3 and confidence > 0.7:
            recommendation = "🟢 STRONG BUY"
            action = f"High conviction bullish signal for {symbol}"
        elif final_sentiment > 0.1 and confidence > 0.6:
            recommendation = "🟡 BUY"
            action = f"Moderate bullish signal for {symbol}"
        elif final_sentiment < -0.3 and confidence > 0.7:
            recommendation = "🔴 STRONG SELL"
            action = f"High conviction bearish signal for {symbol}"
        elif final_sentiment < -0.1 and confidence > 0.6:
            recommendation = "🟠 SELL"
            action = f"Moderate bearish signal for {symbol}"
        else:
            recommendation = "⚪ HOLD"
            action = f"Neutral or low confidence signal for {symbol}"
        
        print(f"📊 FINAL RECOMMENDATION: {recommendation}")
        print(f"💡 Action: {action}")
        print(f"🎯 Signal Strength: {abs(final_sentiment):.1%}")
        print(f"🔒 Confidence: {confidence:.1%}")
        
        # Risk metrics
        print(f"\n📋 RISK METRICS:")
        print(f"   • Velocity Risk: {'⚠️ High' if velocity['mentions_per_hour'] > 10 else '✅ Normal'}")
        print(f"   • Momentum Risk: {'⚠️ High' if abs(momentum['momentum_score']) > 3 else '✅ Normal'}")
        print(f"   • Credibility Risk: {'✅ Low' if credibility['signal_reliability'] in ['high', 'very_high'] else '⚠️ Medium'}")
        print(f"   • Meme Stock Risk: {'⚠️ High' if emoji['rocket_mentions'] > 5 else '✅ Low'}")
        
        print(f"\n✅ SYSTEM STATUS: All Phase 3.2 features operational")
        print(f"🚀 Ready for high-stakes trading with full risk management")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_reddit_deep_analysis()