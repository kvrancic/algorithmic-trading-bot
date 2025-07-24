"""
Simple test to validate Phase 2.1 model structure and imports
"""

import sys
from pathlib import Path

def test_import_structure():
    """Test that all model files can be imported and have correct structure"""
    print("Testing model import structure...")
    
    # Test base models
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        # Test individual file imports
        from src.models.base.base_model import BaseModel, ModelConfig, ModelType
        print("✓ Base model classes imported")
        
        from src.models.lstm.price_lstm import PriceLSTMConfig
        print("✓ PriceLSTM classes imported")
        
        from src.models.cnn.chart_pattern_cnn import ChartPatternConfig
        print("✓ ChartPatternCNN classes imported")
        
        from src.models.xgboost.market_regime_xgboost import MarketRegimeConfig
        print("✓ MarketRegimeXGBoost classes imported")
        
        from src.models.transformers.finbert import FinBERTConfig
        print("✓ FinBERT classes imported")
        
        from src.models.ensemble.stacked_ensemble import StackedEnsembleConfig
        print("✓ StackedEnsemble classes imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """Test that all model configurations can be created"""
    print("\nTesting configuration creation...")
    
    try:
        from src.models.lstm.price_lstm import PriceLSTMConfig
        from src.models.cnn.chart_pattern_cnn import ChartPatternConfig
        from src.models.xgboost.market_regime_xgboost import MarketRegimeConfig
        from src.models.transformers.finbert import FinBERTConfig
        from src.models.ensemble.stacked_ensemble import StackedEnsembleConfig
        
        # Test config creation
        lstm_config = PriceLSTMConfig()
        print(f"✓ PriceLSTMConfig created: {lstm_config.name}")
        
        cnn_config = ChartPatternConfig()
        print(f"✓ ChartPatternConfig created: {cnn_config.name}")
        
        xgb_config = MarketRegimeConfig()
        print(f"✓ MarketRegimeConfig created: {xgb_config.name}")
        
        bert_config = FinBERTConfig()
        print(f"✓ FinBERTConfig created: {bert_config.name}")
        
        ensemble_config = StackedEnsembleConfig()
        print(f"✓ StackedEnsembleConfig created: {ensemble_config.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_structure():
    """Test model class structure and methods"""
    print("\nTesting model class structure...")
    
    try:
        from src.models.base.base_model import BaseModel
        from src.models.lstm.price_lstm import PriceLSTMConfig
        
        # Check that config has required attributes
        config = PriceLSTMConfig()
        required_attrs = ['name', 'model_type', 'epochs', 'batch_size', 'learning_rate']
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"❌ Config missing attribute: {attr}")
                return False
            print(f"✓ Config has {attr}: {getattr(config, attr)}")
        
        # Check ModelType enum
        from src.models.base.base_model import ModelType
        print(f"✓ ModelType enum: {list(ModelType)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    base_path = Path("src/models")
    required_files = [
        "base/__init__.py",
        "base/base_model.py",
        "base/time_series_model.py",
        "base/classification_model.py",
        "base/ensemble_model.py",
        "lstm/__init__.py",
        "lstm/price_lstm.py",
        "cnn/__init__.py",
        "cnn/chart_pattern_cnn.py",
        "xgboost/__init__.py",
        "xgboost/market_regime_xgboost.py",
        "transformers/__init__.py",
        "transformers/finbert.py",
        "ensemble/__init__.py",
        "ensemble/stacked_ensemble.py",
        "__init__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Phase 2.1 Model Architecture Structure Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Import Structure", test_import_structure),
        ("Configuration Creation", test_config_creation),
        ("Model Structure", test_model_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
            print(f"{'✓ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✓ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Model architecture is correctly implemented.")
    else:
        print("⚠ Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)