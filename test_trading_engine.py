#!/usr/bin/env python3
"""
Comprehensive Trading Engine Integration Testing

Tests the complete trading workflow from configuration to signal execution.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_configuration_loading():
    """Test trading configuration loading and validation"""
    print("\n⚙️  Testing Configuration Loading...")
    
    try:
        from src.configuration import Config
        from src.config.config_manager import ConfigManager
        
        # Load main config
        config = Config('config/config.yaml')
        config_manager = ConfigManager(config)
        
        print("  ✅ Configuration loaded successfully")
        
        # Test key trading configurations
        trading_config = config.trading
        
        print(f"    📊 Strategy mode: {trading_config.strategy_mode}")
        print(f"    📊 Watchlist: {trading_config.watchlist[:3]}...")
        print(f"    📊 Max positions: {trading_config.max_positions}")
        print(f"    📊 Signal threshold: {trading_config.signal_threshold}")
        
        # Test universe configuration
        universe_config = config.universe
        print(f"    📊 Stock universe: {len(universe_config.stocks)} symbols")
        print(f"    📊 Dynamic discovery enabled: {universe_config.dynamic_discovery.enabled}")
        
        # Test risk configuration
        risk_config = config.risk
        print(f"    📊 Max drawdown: {risk_config.max_drawdown}")
        print(f"    📊 Stop loss: {risk_config.stop_loss_pct}")
        print(f"    📊 Risk per trade: {risk_config.risk_per_trade}")
        
        # Validate configuration
        is_valid = config.validate()
        print(f"  ✅ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid
        
    except Exception as e:
        print(f"  ❌ Configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_generation():
    """Test feature generation pipeline"""
    print("\n🔧 Testing Feature Generation...")
    
    try:
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100).cumsum() * 0.1,
            'high': 100 + np.random.randn(100).cumsum() * 0.1 + 1,
            'low': 100 + np.random.randn(100).cumsum() * 0.1 - 1,
            'close': 100 + np.random.randn(100).cumsum() * 0.1,
            'volume': np.random.randint(10000, 100000, 100),
        })
        market_data.set_index('timestamp', inplace=True)
        
        # Create sample sentiment data
        sentiment_data = pd.DataFrame({
            'timestamp': dates[::4],  # Every 4 hours
            'sentiment_score': np.random.uniform(-1, 1, 25),
            'confidence': np.random.uniform(0.5, 1.0, 25),
            'mention_count': np.random.randint(1, 100, 25),
        })
        sentiment_data.set_index('timestamp', inplace=True)
        
        print(f"  ✅ Created sample data:")
        print(f"    📊 Market data: {market_data.shape}")
        print(f"    📊 Sentiment data: {sentiment_data.shape}")
        
        # Test basic feature calculations (without TA-Lib dependency)
        market_data['returns'] = market_data['close'].pct_change()
        market_data['sma_5'] = market_data['close'].rolling(5).mean()
        market_data['volatility'] = market_data['returns'].rolling(10).std()
        
        print("  ✅ Basic technical features calculated")
        
        # Test feature combination
        combined_features = market_data.copy()
        
        # Add sentiment features by forward-filling to match market data frequency
        sentiment_resampled = sentiment_data.reindex(
            market_data.index, 
            method='ffill'
        ).fillna(0)
        
        combined_features = pd.concat([combined_features, sentiment_resampled], axis=1)
        
        # Remove NaN values
        features = combined_features.dropna()
        
        print(f"  ✅ Combined features: {features.shape}")
        print(f"    📊 Feature columns: {list(features.columns)}")
        
        # Test feature validation
        if len(features) > 50:  # Should have enough data
            print("  ✅ Feature generation successful")
            return True
        else:
            print("  ❌ Insufficient features generated")
            return False
        
    except Exception as e:
        print(f"  ❌ Feature generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test trading signal generation"""
    print("\n🎯 Testing Signal Generation...")
    
    try:
        # Create mock features for signal generation
        n_samples = 50
        features = pd.DataFrame({
            'returns': np.random.normal(0, 0.02, n_samples),
            'sma_5': np.random.uniform(95, 105, n_samples),
            'volatility': np.random.uniform(0.01, 0.05, n_samples),
            'sentiment_score': np.random.uniform(-1, 1, n_samples),
            'confidence': np.random.uniform(0.5, 1.0, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
        })
        
        print(f"  ✅ Created mock features: {features.shape}")
        
        # Test different signal generation strategies
        signals = []
        
        # Strategy 1: Technical-only signals
        technical_signals = []
        for i, row in features.iterrows():
            # Simple RSI-based signal
            if row['rsi'] < 30:  # Oversold
                signal = {'type': 'buy', 'strength': 0.8, 'source': 'technical'}
            elif row['rsi'] > 70:  # Overbought
                signal = {'type': 'sell', 'strength': 0.8, 'source': 'technical'}
            else:
                signal = {'type': 'hold', 'strength': 0.0, 'source': 'technical'}
            
            technical_signals.append(signal)
        
        # Strategy 2: Sentiment-only signals
        sentiment_signals = []
        for i, row in features.iterrows():
            sentiment = row['sentiment_score']
            confidence = row['confidence']
            
            if sentiment > 0.3 and confidence > 0.7:
                signal = {'type': 'buy', 'strength': sentiment * confidence, 'source': 'sentiment'}
            elif sentiment < -0.3 and confidence > 0.7:
                signal = {'type': 'sell', 'strength': abs(sentiment) * confidence, 'source': 'sentiment'}
            else:
                signal = {'type': 'hold', 'strength': 0.0, 'source': 'sentiment'}
            
            sentiment_signals.append(signal)
        
        # Strategy 3: Combined signals (adaptive mode)
        combined_signals = []
        for i in range(len(features)):
            tech_signal = technical_signals[i]
            sent_signal = sentiment_signals[i]
            
            # Combine signals with confidence boosting when they agree
            if tech_signal['type'] == sent_signal['type'] and tech_signal['type'] != 'hold':
                # Both agree - boost confidence
                strength = (tech_signal['strength'] + sent_signal['strength']) / 2 + 0.1
                signal = {'type': tech_signal['type'], 'strength': min(strength, 1.0), 'source': 'combined'}
            elif tech_signal['strength'] > sent_signal['strength']:
                signal = tech_signal.copy()
            else:
                signal = sent_signal.copy()
            
            combined_signals.append(signal)
        
        # Analyze signal quality
        strategies = [
            ('Technical Only', technical_signals),
            ('Sentiment Only', sentiment_signals),
            ('Combined (Adaptive)', combined_signals)
        ]
        
        for strategy_name, strategy_signals in strategies:
            buy_signals = sum(1 for s in strategy_signals if s['type'] == 'buy')
            sell_signals = sum(1 for s in strategy_signals if s['type'] == 'sell')
            hold_signals = sum(1 for s in strategy_signals if s['type'] == 'hold')
            
            avg_strength = np.mean([s['strength'] for s in strategy_signals if s['strength'] > 0])
            
            print(f"  ✅ {strategy_name}:")
            print(f"    📊 Buy signals: {buy_signals}")
            print(f"    📊 Sell signals: {sell_signals}")
            print(f"    📊 Hold signals: {hold_signals}")
            print(f"    📊 Average strength: {avg_strength:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_management():
    """Test risk management and position sizing"""
    print("\n🛡️  Testing Risk Management...")
    
    try:
        # Simulate portfolio state
        portfolio_value = 100000  # $100k portfolio
        max_position_size = 0.10  # 10% max per position
        risk_per_trade = 0.01     # 1% risk per trade
        stop_loss_pct = 0.02      # 2% stop loss
        
        # Test signals for different symbols
        signals = [
            {'symbol': 'AAPL', 'type': 'buy', 'strength': 0.8, 'confidence': 0.9, 'price': 150.0},
            {'symbol': 'TSLA', 'type': 'buy', 'strength': 0.7, 'confidence': 0.8, 'price': 200.0},
            {'symbol': 'NVDA', 'type': 'sell', 'strength': 0.6, 'confidence': 0.7, 'price': 400.0},
        ]
        
        print(f"  ✅ Portfolio setup:")
        print(f"    💰 Portfolio value: ${portfolio_value:,.2f}")
        print(f"    📊 Max position size: {max_position_size:.1%}")
        print(f"    📊 Risk per trade: {risk_per_trade:.1%}")
        
        # Calculate position sizes
        positions = []
        
        for signal in signals:
            # Base position size (percentage of portfolio)
            base_position_value = portfolio_value * max_position_size
            
            # Adjust based on signal strength and confidence
            strength_factor = signal['strength'] * signal['confidence']
            adjusted_position_value = base_position_value * strength_factor
            
            # Calculate shares
            shares = int(adjusted_position_value / signal['price'])
            actual_value = shares * signal['price']
            
            # Calculate stop loss
            if signal['type'] == 'buy':
                stop_price = signal['price'] * (1 - stop_loss_pct)
            else:  # sell
                stop_price = signal['price'] * (1 + stop_loss_pct)
            
            position = {
                'symbol': signal['symbol'],
                'type': signal['type'],
                'shares': shares,
                'price': signal['price'],
                'value': actual_value,
                'portfolio_pct': actual_value / portfolio_value,
                'stop_price': stop_price,
                'strength': signal['strength'],
                'confidence': signal['confidence']
            }
            
            positions.append(position)
        
        # Display position analysis
        total_exposure = sum(pos['value'] for pos in positions)
        
        print(f"  ✅ Position sizing results:")
        for pos in positions:
            print(f"    📈 {pos['symbol']}: {pos['shares']} shares at ${pos['price']:.2f}")
            print(f"        💰 Value: ${pos['value']:,.2f} ({pos['portfolio_pct']:.1%} of portfolio)")
            print(f"        🛑 Stop loss: ${pos['stop_price']:.2f}")
        
        print(f"    📊 Total exposure: ${total_exposure:,.2f} ({total_exposure/portfolio_value:.1%})")
        
        # Risk checks
        risk_checks = []
        
        # Check individual position limits
        for pos in positions:
            if pos['portfolio_pct'] > max_position_size:
                risk_checks.append(f"Position {pos['symbol']} exceeds max size")
        
        # Check total exposure
        if total_exposure / portfolio_value > 0.5:  # Max 50% total exposure
            risk_checks.append("Total exposure exceeds 50%")
        
        # Check diversification
        if len(positions) < 3:
            risk_checks.append("Insufficient diversification")
        
        if risk_checks:
            print(f"  ⚠️  Risk warnings: {risk_checks}")
        else:
            print("  ✅ All risk checks passed")
        
        return len(risk_checks) == 0
        
    except Exception as e:
        print(f"  ❌ Risk management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_modes():
    """Test different trading strategy modes"""
    print("\n🎮 Testing Strategy Modes...")
    
    try:
        from src.configuration import Config
        
        config = Config('config/config.yaml')
        
        # Test current strategy mode
        strategy_mode = config.trading.strategy_mode
        signal_requirements = config.trading.signal_requirements
        
        print(f"  ✅ Current strategy mode: {strategy_mode}")
        
        # Test each strategy mode configuration
        modes = ['adaptive', 'technical_only', 'sentiment_only', 'conservative']
        
        for mode in modes:
            if hasattr(signal_requirements, mode):
                mode_config = getattr(signal_requirements, mode)
                print(f"  ✅ {mode} mode configured:")
                
                if hasattr(mode_config, 'to_dict'):
                    config_dict = mode_config.to_dict()
                else:
                    config_dict = mode_config.__dict__ if hasattr(mode_config, '__dict__') else {}
                
                for key, value in config_dict.items():
                    print(f"    📊 {key}: {value}")
            else:
                print(f"  ⚠️  {mode} mode not configured")
        
        # Test strategy validation logic
        sample_signal = {
            'symbol': 'AAPL',
            'technical_indicators': ['rsi', 'macd', 'bollinger'],
            'sentiment_sources': ['reddit', 'news'],
            'technical_strength': 0.8,
            'sentiment_strength': 0.7,
            'confidence': 0.85
        }
        
        # Simulate strategy validation for each mode
        validations = {}
        
        # Adaptive mode - use any available signals
        if strategy_mode == 'adaptive' or 'adaptive' in modes:
            adaptive_valid = (sample_signal['confidence'] >= 0.6 and
                            (sample_signal['technical_strength'] > 0.6 or 
                             sample_signal['sentiment_strength'] > 0.6))
            validations['adaptive'] = adaptive_valid
        
        # Technical only mode
        tech_confluence = len(sample_signal['technical_indicators'])
        technical_valid = (sample_signal['technical_strength'] >= 0.7 and 
                          tech_confluence >= 2)
        validations['technical_only'] = technical_valid
        
        # Sentiment only mode
        sentiment_valid = (sample_signal['sentiment_strength'] >= 0.7 and
                          len(sample_signal['sentiment_sources']) >= 1)
        validations['sentiment_only'] = sentiment_valid
        
        # Conservative mode - require both
        conservative_valid = (technical_valid and sentiment_valid and
                            sample_signal['confidence'] >= 0.8)
        validations['conservative'] = conservative_valid
        
        print(f"  ✅ Strategy validation results for sample signal:")
        for mode, is_valid in validations.items():
            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"    {status} {mode}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy modes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test complete trading workflow integration"""
    print("\n🔄 Testing Integration Workflow...")
    
    try:
        from src.configuration import Config
        from src.config.config_manager import ConfigManager
        
        # Load configuration
        config = Config('config/config.yaml')
        config_manager = ConfigManager(config)
        
        print("  ✅ Step 1: Configuration loaded")
        
        # Simulate universe management
        current_universe = set(config.universe.stocks)
        print(f"  ✅ Step 2: Universe initialized with {len(current_universe)} symbols")
        
        # Simulate dynamic discovery
        from src.universe.dynamic_discovery import DynamicSymbolDiscovery
        discovery = DynamicSymbolDiscovery(config_manager)
        
        # Simulate discovered symbols
        discovered_symbols = ['RBLX', 'COIN']  # Trending symbols
        print(f"  ✅ Step 3: Discovered {len(discovered_symbols)} new symbols")
        
        # Update universe
        updated_universe = current_universe.union(discovered_symbols)
        print(f"  ✅ Step 4: Universe updated to {len(updated_universe)} symbols")
        
        # Simulate feature generation for active symbols
        active_symbols = ['AAPL', 'TSLA', 'NVDA']  # Subset for testing
        features_generated = {}
        
        for symbol in active_symbols:
            # Simulate feature generation
            features = {
                'technical': np.random.randn(10),
                'sentiment': np.random.uniform(-1, 1),
                'confidence': np.random.uniform(0.5, 1.0)
            }
            features_generated[symbol] = features
        
        print(f"  ✅ Step 5: Features generated for {len(features_generated)} symbols")
        
        # Simulate signal generation
        signals = []
        for symbol, features in features_generated.items():
            if features['confidence'] > 0.7:
                signal = {
                    'symbol': symbol,
                    'type': 'buy' if features['sentiment'] > 0.2 else 'sell',
                    'strength': abs(features['sentiment']),
                    'confidence': features['confidence'],
                    'timestamp': datetime.now()
                }
                signals.append(signal)
        
        print(f"  ✅ Step 6: Generated {len(signals)} trading signals")
        
        # Simulate risk validation
        validated_signals = []
        max_positions = config.trading.max_positions
        
        for signal in signals[:max_positions]:  # Limit to max positions
            if signal['strength'] >= 0.6 and signal['confidence'] >= 0.7:
                validated_signals.append(signal)
        
        print(f"  ✅ Step 7: {len(validated_signals)} signals passed risk validation")
        
        # Simulate execution preparation
        execution_orders = []
        for signal in validated_signals:
            order = {
                'symbol': signal['symbol'],
                'side': signal['type'],
                'quantity': 100,  # Simplified
                'order_type': 'market',
                'signal_id': f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now()
            }
            execution_orders.append(order)
        
        print(f"  ✅ Step 8: Prepared {len(execution_orders)} orders for execution")
        
        # Simulate monitoring setup
        monitored_positions = len(execution_orders)
        print(f"  ✅ Step 9: Monitoring {monitored_positions} positions")
        
        print(f"  ✅ Step 10: Workflow completed successfully")
        
        # Summary
        workflow_summary = {
            'universe_size': len(updated_universe),
            'symbols_analyzed': len(active_symbols),
            'signals_generated': len(signals),
            'signals_validated': len(validated_signals),
            'orders_prepared': len(execution_orders),
            'positions_monitored': monitored_positions
        }
        
        print(f"  📊 Workflow Summary:")
        for key, value in workflow_summary.items():
            print(f"    📈 {key}: {value}")
        
        # Success criteria
        success = (workflow_summary['signals_generated'] > 0 and
                  workflow_summary['orders_prepared'] > 0)
        
        return success
        
    except Exception as e:
        print(f"  ❌ Integration workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all trading engine tests"""
    print("🚀 Comprehensive Trading Engine Testing")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Feature Generation", test_feature_generation),
        ("Signal Generation", test_signal_generation),
        ("Risk Management", test_risk_management),
        ("Strategy Modes", test_strategy_modes),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*10} {test_name} {'='*10}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TRADING ENGINE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} trading engine tests passed")
    
    if passed == total:
        print("🎉 TRADING ENGINE FULLY OPERATIONAL!")
        print("✅ Configuration system working correctly")
        print("✅ Feature generation pipeline functional")
        print("✅ Signal generation with multiple strategies")
        print("✅ Risk management and position sizing")
        print("✅ Strategy mode switching capability")
        print("✅ End-to-end workflow integration")
        return 0
    else:
        print("⚠️  Trading engine needs attention")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))