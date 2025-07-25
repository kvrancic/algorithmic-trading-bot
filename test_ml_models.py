#!/usr/bin/env python3
"""
Comprehensive ML Model Testing Script

Tests all ML algorithms with small synthetic datasets to verify they work correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_sample_price_data(n_samples=100, n_features=10):
    """Create sample price/market data"""
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # Create realistic price movements
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)  # 2% volatility
    prices = base_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.randint(10000, 100000, n_samples),
    })
    
    # Add technical indicators as features
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    data.set_index('timestamp', inplace=True)
    return data

def create_sample_sentiment_data(n_samples=50):
    """Create sample sentiment data"""
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='4H')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'sentiment_score': np.random.uniform(-1, 1, n_samples),
        'confidence': np.random.uniform(0.5, 1.0, n_samples),
        'mention_count': np.random.randint(1, 100, n_samples),
        'source': np.random.choice(['reddit', 'news', 'twitter'], n_samples),
        'symbol': 'AAPL'
    })
    
    data.set_index('timestamp', inplace=True)
    return data

def test_price_lstm():
    """Test LSTM price prediction model"""
    print("\n🧠 Testing Price LSTM Model...")
    
    try:
        from src.models.lstm.price_lstm import PriceLSTM, PriceLSTMConfig
        
        # Create config
        config = PriceLSTMConfig(
            sequence_length=10,
            forecast_horizon=1,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=16,
            epochs=2  # Small for testing
        )
        
        # Create model
        model = PriceLSTM(config)
        print("  ✅ LSTM model created successfully")
        
        # Create sample data
        data = create_sample_price_data(50)
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        targets = data['close'].shift(-1).dropna().values  # Next period close
        
        print(f"  📊 Training data shape: {features.shape}")
        print(f"  📊 Target data shape: {targets.shape}")
        
        # Train model
        model.train(features[:-1], targets)  # Remove last row due to shift
        print("  ✅ LSTM training completed")
        
        # Test prediction
        test_input = features[-10:]  # Last 10 samples
        prediction = model.predict(test_input)
        print(f"  ✅ LSTM prediction: {prediction[:3]}... (shape: {prediction.shape})")
        
        # Test model state
        print(f"  ✅ LSTM is_trained: {model.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chart_pattern_cnn():
    """Test CNN chart pattern recognition model"""
    print("\n👁️  Testing Chart Pattern CNN Model...")
    
    try:
        from src.models.cnn.chart_pattern_cnn import ChartPatternCNN, ChartPatternCNNConfig
        
        # Create config
        config = ChartPatternCNNConfig(
            image_size=(64, 64),
            num_classes=4,  # head_shoulders, triangle, flag, wedge
            learning_rate=0.001,
            batch_size=8,
            epochs=2
        )
        
        # Create model
        model = ChartPatternCNN(config)
        print("  ✅ CNN model created successfully")
        
        # Create sample chart images (simulate candlestick charts)
        n_samples = 20
        images = np.random.rand(n_samples, 64, 64, 3)  # RGB images
        labels = np.random.randint(0, 4, n_samples)  # Pattern classes
        
        print(f"  📊 Training images shape: {images.shape}")
        print(f"  📊 Training labels shape: {labels.shape}")
        
        # Train model
        model.train(images, labels)
        print("  ✅ CNN training completed")
        
        # Test prediction
        test_image = images[:5]
        prediction = model.predict(test_image)
        print(f"  ✅ CNN prediction: {prediction[:3]}... (shape: {prediction.shape})")
        
        print(f"  ✅ CNN is_trained: {model.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_regime_xgboost():
    """Test XGBoost market regime classification"""
    print("\n🌳 Testing Market Regime XGBoost Model...")
    
    try:
        from src.models.xgboost.market_regime_xgboost import MarketRegimeXGBoost, MarketRegimeXGBoostConfig
        
        # Create config
        config = MarketRegimeXGBoostConfig(
            n_estimators=50,  # Reduced for testing
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        # Create model
        model = MarketRegimeXGBoost(config)
        print("  ✅ XGBoost model created successfully")
        
        # Create sample data
        data = create_sample_price_data(100, 15)
        features = data.select_dtypes(include=[np.number]).values
        
        # Create regime labels (0=trending, 1=ranging, 2=volatile)
        volatility = np.std(data['close'].rolling(10).apply(lambda x: np.std(x)), axis=0)
        regimes = np.random.randint(0, 3, len(features))
        
        print(f"  📊 Training features shape: {features.shape}")
        print(f"  📊 Training regimes shape: {regimes.shape}")
        
        # Train model
        model.train(features, regimes)
        print("  ✅ XGBoost training completed")
        
        # Test prediction
        test_input = features[-10:]
        prediction = model.predict(test_input)
        print(f"  ✅ XGBoost prediction: {prediction[:5]}... (shape: {prediction.shape})")
        
        print(f"  ✅ XGBoost is_trained: {model.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ XGBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_finbert():
    """Test FinBERT sentiment analysis model"""
    print("\n🤖 Testing FinBERT Sentiment Model...")
    
    try:
        from src.models.transformers.finbert import FinBERT, FinBERTConfig
        
        # Create config
        config = FinBERTConfig(
            model_name="ProsusAI/finbert",
            max_length=128,  # Reduced for testing
            batch_size=4
        )
        
        # Create model
        model = FinBERT(config)
        print("  ✅ FinBERT model created successfully")
        
        # Create sample text data
        texts = [
            "Apple stock is going to the moon! Great earnings report.",
            "Tesla disappointing quarterly results, stock might drop.",
            "Market volatility increasing, investors are nervous.",
            "Strong bull market continues with record highs.",
            "Economic uncertainty causing market selloff today."
        ]
        
        # Create sentiment labels (0=negative, 1=neutral, 2=positive)
        labels = np.array([2, 0, 0, 2, 0])
        
        print(f"  📊 Training texts: {len(texts)} samples")
        print(f"  📊 Training labels shape: {labels.shape}")
        
        # Note: FinBERT is pre-trained, so we'll just test prediction
        print("  ✅ FinBERT using pre-trained weights (no additional training needed)")
        
        # Test prediction
        predictions = model.predict(texts[:3])
        print(f"  ✅ FinBERT predictions: {predictions}... (shape: {predictions.shape})")
        
        # Test individual text
        single_text = "Apple earnings beat expectations, stock surging!"
        single_pred = model.predict([single_text])
        print(f"  ✅ Single prediction: '{single_text}' → {single_pred[0]:.3f}")
        
        print(f"  ✅ FinBERT is_trained: {hasattr(model, 'is_trained') and model.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FinBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stacked_ensemble():
    """Test Stacked Ensemble meta-learning model"""
    print("\n🎯 Testing Stacked Ensemble Model...")
    
    try:
        from src.models.ensemble.stacked_ensemble import StackedEnsemble, StackedEnsembleConfig
        
        # Create config
        config = StackedEnsembleConfig(
            meta_learner_type="xgboost",
            use_probabilities=True,
            include_original_features=True,
            cv_folds=3  # Reduced for testing
        )
        
        # Create model
        model = StackedEnsemble(config)
        print("  ✅ Ensemble model created successfully")
        
        # Create sample data
        n_samples = 100
        n_features = 10
        
        # Simulate base model predictions
        base_predictions = {
            'lstm': np.random.uniform(-1, 1, n_samples),
            'cnn': np.random.uniform(0, 1, n_samples),
            'xgboost': np.random.randint(0, 3, n_samples),
            'finbert': np.random.uniform(-1, 1, n_samples)
        }
        
        # Original features
        features = np.random.randn(n_samples, n_features)
        
        # Combine into ensemble input
        ensemble_features = np.column_stack([
            features,
            base_predictions['lstm'],
            base_predictions['cnn'],
            base_predictions['xgboost'],
            base_predictions['finbert']
        ])
        
        # Create targets (regression)
        targets = np.random.uniform(-0.5, 0.5, n_samples)
        
        print(f"  📊 Ensemble features shape: {ensemble_features.shape}")
        print(f"  📊 Target shape: {targets.shape}")
        
        # Train ensemble
        model.train(ensemble_features, targets)
        print("  ✅ Ensemble training completed")
        
        # Test prediction
        test_input = ensemble_features[-10:]
        prediction = model.predict(test_input)
        print(f"  ✅ Ensemble prediction: {prediction[:3]}... (shape: {prediction.shape})")
        
        print(f"  ✅ Ensemble is_trained: {model.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_pipeline():
    """Test feature generation pipeline"""
    print("\n⚙️  Testing Feature Pipeline...")
    
    try:
        from src.features.feature_pipeline import FeaturePipeline, FeatureConfig
        from src.configuration import Config
        from src.database.database import DatabaseManager
        
        # Create config
        feature_config = FeatureConfig(
            enable_technical=True,
            enable_sentiment=True,
            enable_fundamental=False,  # Disable for testing
            enable_market_structure=True
        )
        
        # Create mock database manager
        config = Config('config/config.yaml')
        db_manager = DatabaseManager(":memory:")  # In-memory SQLite for testing
        
        # Create pipeline
        pipeline = FeaturePipeline(feature_config, db_manager)
        print("  ✅ Feature pipeline created successfully")
        
        # Create sample data
        market_data = create_sample_price_data(100, 5)
        sentiment_data = create_sample_sentiment_data(25)
        
        print(f"  📊 Market data shape: {market_data.shape}")
        print(f"  📊 Sentiment data shape: {sentiment_data.shape}")
        
        # Generate features
        features = pipeline.generate_features(
            symbol='AAPL',
            market_data=market_data,
            sentiment_data=sentiment_data
        )
        
        print(f"  ✅ Generated features shape: {features.shape}")
        print(f"  ✅ Feature columns: {list(features.columns)[:5]}...")
        
        # Check for NaN values
        nan_count = features.isnull().sum().sum()
        print(f"  ✅ NaN values in features: {nan_count}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all ML model tests"""
    print("🧠 Comprehensive ML Model Testing")
    print("=" * 60)
    
    tests = [
        ("Feature Pipeline", test_feature_pipeline),
        ("Price LSTM", test_price_lstm),
        ("Chart Pattern CNN", test_chart_pattern_cnn),
        ("Market Regime XGBoost", test_market_regime_xgboost),
        ("FinBERT Sentiment", test_finbert),
        ("Stacked Ensemble", test_stacked_ensemble),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
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
    print("🎯 ML MODEL TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} ML models working")
    
    if passed == total:
        print("🎉 ALL ML MODELS WORKING PERFECTLY!")
        return 0
    else:
        print("⚠️  Some ML models need attention")
        return 1

if __name__ == "__main__":
    exit(main())