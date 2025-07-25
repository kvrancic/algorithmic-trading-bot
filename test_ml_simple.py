#!/usr/bin/env python3
"""
Simple ML Model Testing Script

Tests core ML functionality without external dependencies.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_imports():
    """Test that all model classes can be imported"""
    print("\n📦 Testing Model Imports...")
    
    try:
        # Test base model imports
        from src.models.base.base_model import BaseModel, ModelConfig, ModelType
        print("  ✅ Base model classes imported")
        
        from src.models.base.time_series_model import TimeSeriesModel, TimeSeriesConfig
        print("  ✅ Time series model classes imported")
        
        # Test individual model imports
        from src.models.lstm.price_lstm import PriceLSTM, PriceLSTMConfig
        print("  ✅ LSTM model imported")
        
        from src.models.transformers.finbert import FinBERT, FinBERTConfig
        print("  ✅ FinBERT model imported")
        
        from src.models.ensemble.stacked_ensemble import StackedEnsemble, StackedEnsembleConfig
        print("  ✅ Ensemble model imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_model_configs():
    """Test model configuration creation"""
    print("\n⚙️  Testing Model Configurations...")
    
    try:
        # Test LSTM config
        from src.models.lstm.price_lstm import PriceLSTMConfig
        lstm_config = PriceLSTMConfig(
            sequence_length=10,
            forecast_horizon=1,
            hidden_size=32,
            num_layers=2
        )
        print(f"  ✅ LSTM config created: sequence_length={lstm_config.sequence_length}")
        
        # Test FinBERT config  
        from src.models.transformers.finbert import FinBERTConfig
        bert_config = FinBERTConfig(
            model_name="ProsusAI/finbert",
            max_length=128
        )
        print(f"  ✅ FinBERT config created: model_name={bert_config.model_name}")
        
        # Test Ensemble config
        from src.models.ensemble.stacked_ensemble import StackedEnsembleConfig
        ensemble_config = StackedEnsembleConfig(
            meta_learner_type="xgboost",
            cv_folds=3
        )
        print(f"  ✅ Ensemble config created: meta_learner={ensemble_config.meta_learner_type}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test model object creation"""
    print("\n🧠 Testing Model Initialization...")
    
    try:
        # Test LSTM initialization
        from src.models.lstm.price_lstm import PriceLSTM, PriceLSTMConfig
        lstm_config = PriceLSTMConfig(sequence_length=10, hidden_size=32)
        lstm_model = PriceLSTM(lstm_config)
        print(f"  ✅ LSTM model initialized: {lstm_model.__class__.__name__}")
        print(f"    📊 Is trained: {lstm_model.is_trained}")
        
        # Test FinBERT initialization
        from src.models.transformers.finbert import FinBERT, FinBERTConfig
        bert_config = FinBERTConfig(model_name="ProsusAI/finbert")
        bert_model = FinBERT(bert_config)
        print(f"  ✅ FinBERT model initialized: {bert_model.__class__.__name__}")
        print(f"    📊 Is trained: {getattr(bert_model, 'is_trained', 'Not available')}")
        
        # Test Ensemble initialization
        from src.models.ensemble.stacked_ensemble import StackedEnsemble, StackedEnsembleConfig
        ensemble_config = StackedEnsembleConfig(meta_learner_type="xgboost")
        ensemble_model = StackedEnsemble(ensemble_config)
        print(f"  ✅ Ensemble model initialized: {ensemble_model.__class__.__name__}")
        print(f"    📊 Is trained: {ensemble_model.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_ml_operations():
    """Test basic ML operations with numpy"""
    print("\n🔢 Testing Basic ML Operations...")
    
    try:
        # Test data generation
        np.random.seed(42)
        n_samples, n_features = 100, 10
        
        # Generate synthetic data
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        print(f"  ✅ Synthetic data generated: X{X.shape}, y{y.shape}")
        
        # Test basic sklearn operations
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            print(f"  ✅ sklearn RandomForest: MSE={mse:.4f}")
            
        except ImportError:
            print("  ⚠️  sklearn not available, skipping")
        
        # Test basic PyTorch operations
        try:
            import torch
            import torch.nn as nn
            
            # Create simple neural network
            class SimpleNet(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.linear = nn.Linear(input_size, output_size)
                
                def forward(self, x):
                    return self.linear(x)
            
            # Test model creation and forward pass
            net = SimpleNet(n_features, 1)
            X_tensor = torch.FloatTensor(X[:10])  # Use first 10 samples
            output = net(X_tensor)
            
            print(f"  ✅ PyTorch SimpleNet: output_shape={output.shape}")
            
        except ImportError:
            print("  ⚠️  PyTorch not available, skipping")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic ML operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing capabilities"""
    print("\n📊 Testing Data Processing...")
    
    try:
        # Test pandas operations
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        print(f"  ✅ DataFrame created: {df.shape}")
        
        # Test basic operations
        df['returns'] = df['price'].pct_change()
        df['ma_5'] = df['price'].rolling(5).mean()
        df['volatility'] = df['returns'].rolling(10).std()
        
        print(f"  ✅ Technical indicators calculated")
        print(f"    📊 Latest price: {df['price'].iloc[-1]:.2f}")
        print(f"    📊 Latest 5-day MA: {df['ma_5'].iloc[-1]:.2f}")
        
        # Test data cleaning
        clean_df = df.dropna()
        print(f"  ✅ Data cleaned: {clean_df.shape} (removed {len(df) - len(clean_df)} NaN rows)")
        
        # Test feature scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = ['feature1', 'feature2', 'returns']
        scaled_features = scaler.fit_transform(clean_df[features])
        
        print(f"  ✅ Features scaled: {scaled_features.shape}")
        print(f"    📊 Mean after scaling: {scaled_features.mean(axis=0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_interfaces():
    """Test that models follow the expected interface"""
    print("\n🔌 Testing Model Interfaces...")
    
    try:
        from src.models.base.base_model import BaseModel
        
        # Test that our models inherit from BaseModel
        from src.models.lstm.price_lstm import PriceLSTM, PriceLSTMConfig
        from src.models.transformers.finbert import FinBERT, FinBERTConfig
        from src.models.ensemble.stacked_ensemble import StackedEnsemble, StackedEnsembleConfig
        
        # Create model instances
        lstm = PriceLSTM(PriceLSTMConfig())
        finbert = FinBERT(FinBERTConfig())
        ensemble = StackedEnsemble(StackedEnsembleConfig())
        
        models = [
            ("LSTM", lstm),
            ("FinBERT", finbert),
            ("Ensemble", ensemble)
        ]
        
        for name, model in models:
            # Check interface compliance
            has_train = hasattr(model, 'train') and callable(getattr(model, 'train'))
            has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
            has_save = hasattr(model, 'save') and callable(getattr(model, 'save'))
            has_load = hasattr(model, 'load') and callable(getattr(model, 'load'))
            has_is_trained = hasattr(model, 'is_trained')
            
            print(f"  ✅ {name} interface check:")
            print(f"    📋 train(): {has_train}")
            print(f"    📋 predict(): {has_predict}")
            print(f"    📋 save(): {has_save}")
            print(f"    📋 load(): {has_load}")
            print(f"    📋 is_trained: {has_is_trained}")
            
            if not all([has_train, has_predict, has_save, has_load, has_is_trained]):
                print(f"    ❌ {name} missing required interface methods")
                return False
        
        print("  ✅ All models implement required interface")
        return True
        
    except Exception as e:
        print(f"  ❌ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified ML tests"""
    print("🧠 Simplified ML Model Testing")
    print("=" * 60)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("Model Configurations", test_model_configs),
        ("Model Initialization", test_model_initialization),
        ("Basic ML Operations", test_basic_ml_operations),
        ("Data Processing", test_data_processing),
        ("Model Interfaces", test_model_interfaces),
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
    print("🎯 ML FOUNDATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} ML foundation tests passed")
    
    if passed == total:
        print("🎉 ML FOUNDATION IS SOLID!")
        print("📝 Note: Individual model training requires external dependencies")
        print("   Install requirements.txt for full ML functionality")
        return 0
    else:
        print("⚠️  ML foundation needs attention")
        return 1

if __name__ == "__main__":
    exit(main())