"""
Model Training Pipeline

Orchestrates the training of all models in the QuantumSentiment system.
Handles data preparation, model training, validation, and ensemble creation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import structlog
import json
import joblib
from abc import ABC, abstractmethod

from ..models import (
    PriceLSTM, PriceLSTMConfig,
    ChartPatternCNN, ChartPatternConfig,
    MarketRegimeXGBoost, MarketRegimeConfig,
    FinBERT, FinBERTConfig,
    StackedEnsemble, StackedEnsembleConfig,
    BaseModel
)

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline"""
    
    # Data configuration
    train_start_date: str = "2020-01-01"
    train_end_date: str = "2023-12-31"
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training configuration
    parallel_training: bool = True
    max_workers: int = 4
    use_gpu: bool = True
    random_seed: int = 42
    
    # Model selection
    train_lstm: bool = True
    train_cnn: bool = True
    train_xgboost: bool = True
    train_finbert: bool = False  # Set to False by default due to heavy compute
    train_ensemble: bool = True
    
    # Output configuration
    model_save_dir: Path = field(default_factory=lambda: Path("models"))
    checkpoint_interval: int = 10  # Save checkpoint every N epochs
    save_best_only: bool = True
    
    # Performance thresholds
    min_accuracy: float = 0.55
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.2
    
    # Monitoring
    track_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'sharpe_ratio', 'max_drawdown'
    ])
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: str = "validation_f1"
    
    def __post_init__(self):
        self.model_save_dir = Path(self.model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True, parents=True)


class DataPreprocessor:
    """Handles data preprocessing for different model types"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def prepare_price_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare price data for LSTM training"""
        # Sort by timestamp
        data = raw_data.sort_values('timestamp').copy()
        
        # Calculate technical indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        
        # Create features
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi', 'volatility']
        features = data[feature_cols].dropna()
        
        # Create targets (next period return)
        targets = data['close'].pct_change().shift(-1).dropna()
        
        # Align features and targets
        min_len = min(len(features), len(targets))
        features = features.iloc[:min_len]
        targets = targets.iloc[:min_len]
        
        return features, targets
    
    def prepare_chart_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare chart data for CNN training"""
        # Chart data preparation will be handled by the CNN model itself
        # We just need to provide clean OHLCV data
        data = raw_data.sort_values('timestamp').copy()
        
        # Create pattern labels (simplified - in practice would come from expert labeling)
        patterns = self._generate_pattern_labels(data)
        
        return data, patterns
    
    def prepare_regime_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for market regime classification"""
        data = raw_data.sort_values('timestamp').copy()
        
        # XGBoost model will auto-generate regime labels
        return data, None
    
    def prepare_sentiment_data(self, text_data: List[str], sentiment_labels: List[str]) -> Tuple[List[str], np.ndarray]:
        """Prepare text data for sentiment analysis"""
        # Clean text data
        cleaned_texts = [self._clean_text(text) for text in text_data]
        
        # Encode sentiment labels
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        encoded_labels = np.array([label_map.get(label, 1) for label in sentiment_labels])
        
        return cleaned_texts, encoded_labels
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_pattern_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple pattern labels for training"""
        # This is a simplified approach - in practice, patterns would be labeled by experts
        # or detected using more sophisticated algorithms
        patterns = []
        
        for i in range(len(data)):
            # Simple heuristic based on price movement
            if i < 20:
                patterns.append('no_pattern')
                continue
                
            recent_data = data.iloc[i-20:i]
            returns = recent_data['close'].pct_change()
            volatility = returns.std()
            trend = returns.mean()
            
            if volatility > 0.03:
                patterns.append('high_volatility')
            elif trend > 0.02:
                patterns.append('bull_flag')
            elif trend < -0.02:
                patterns.append('bear_flag')
            else:
                patterns.append('sideways')
        
        return pd.Series(patterns, index=data.index)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Basic text cleaning
        import re
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text


class ModelTrainingPipeline:
    """Main training pipeline for all models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.trained_models: Dict[str, BaseModel] = {}
        self.training_results: Dict[str, Dict[str, Any]] = {}
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
    def train_all_models(
        self,
        price_data: pd.DataFrame,
        text_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, BaseModel]:
        """Train all models in the pipeline"""
        
        logger.info("Starting model training pipeline")
        
        # Prepare data splits
        train_data, val_data, test_data = self._create_data_splits(price_data)
        
        # Define training tasks
        training_tasks = []
        
        if self.config.train_lstm:
            training_tasks.append(('PriceLSTM', self._train_lstm, train_data, val_data))
        
        if self.config.train_cnn:
            training_tasks.append(('ChartPatternCNN', self._train_cnn, train_data, val_data))
        
        if self.config.train_xgboost:
            training_tasks.append(('MarketRegimeXGBoost', self._train_xgboost, train_data, val_data))
        
        if self.config.train_finbert and text_data:
            training_tasks.append(('FinBERT', self._train_finbert, text_data, None))
        
        # Train models
        if self.config.parallel_training and len(training_tasks) > 1:
            self._train_models_parallel(training_tasks)
        else:
            self._train_models_sequential(training_tasks)
        
        # Train ensemble if requested
        if self.config.train_ensemble and len(self.trained_models) > 1:
            ensemble = self._train_ensemble(train_data, val_data)
            self.trained_models['StackedEnsemble'] = ensemble
        
        # Validate all models
        self._validate_models(test_data)
        
        # Save models
        self._save_models()
        
        logger.info("Model training pipeline completed",
                   models_trained=list(self.trained_models.keys()),
                   total_models=len(self.trained_models))
        
        return self.trained_models
    
    def _create_data_splits(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits"""
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Calculate split indices
        total_len = len(data)
        test_size = int(total_len * self.config.test_split)
        val_size = int(total_len * self.config.validation_split)
        train_size = total_len - test_size - val_size
        
        # Create splits (maintaining temporal order)
        train_data = data.iloc[:train_size].copy()
        val_data = data.iloc[train_size:train_size + val_size].copy()
        test_data = data.iloc[train_size + val_size:].copy()
        
        logger.info("Data splits created",
                   train_samples=len(train_data),
                   val_samples=len(val_data),
                   test_samples=len(test_data))
        
        return train_data, val_data, test_data
    
    def _train_models_parallel(self, training_tasks: List[Tuple]):
        """Train models in parallel"""
        
        logger.info("Training models in parallel", max_workers=self.config.max_workers)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit training tasks
            future_to_model = {}
            for model_name, train_func, train_data, val_data in training_tasks:
                future = executor.submit(train_func, train_data, val_data)
                future_to_model[future] = model_name
            
            # Collect results
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model, results = future.result()
                    self.trained_models[model_name] = model
                    self.training_results[model_name] = results
                    logger.info(f"Model {model_name} training completed")
                except Exception as e:
                    logger.error(f"Model {model_name} training failed", error=str(e))
    
    def _train_models_sequential(self, training_tasks: List[Tuple]):
        """Train models sequentially"""
        
        logger.info("Training models sequentially")
        
        for model_name, train_func, train_data, val_data in training_tasks:
            try:
                logger.info(f"Starting training: {model_name}")
                model, results = train_func(train_data, val_data)
                self.trained_models[model_name] = model
                self.training_results[model_name] = results
                logger.info(f"Model {model_name} training completed")
            except Exception as e:
                logger.error(f"Model {model_name} training failed", error=str(e))
    
    def _train_lstm(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[PriceLSTM, Dict]:
        """Train LSTM model"""
        
        # Prepare data
        train_features, train_targets = self.preprocessor.prepare_price_data(train_data)
        val_features, val_targets = self.preprocessor.prepare_price_data(val_data)
        
        # Create config
        config = PriceLSTMConfig(
            sequence_length=60,  # 60 time steps
            lstm_hidden_size=128,
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            early_stopping_patience=self.config.early_stopping_patience,
            save_path=self.config.model_save_dir / "lstm"
        )
        
        # Create and train model
        model = PriceLSTM(config)
        
        validation_data = (val_features, val_targets) if val_data is not None else None
        history = model.train(train_features, train_targets, validation_data=validation_data)
        
        return model, history
    
    def _train_cnn(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[ChartPatternCNN, Dict]:
        """Train CNN model"""
        
        # Prepare data
        train_charts, train_patterns = self.preprocessor.prepare_chart_data(train_data)
        val_charts, val_patterns = self.preprocessor.prepare_chart_data(val_data)
        
        # Create config
        config = ChartPatternConfig(
            chart_height=64,
            chart_width=128,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=self.config.early_stopping_patience,
            save_path=self.config.model_save_dir / "cnn"
        )
        
        # Create and train model
        model = ChartPatternCNN(config)
        
        validation_data = (val_charts, val_patterns) if val_data is not None else None
        history = model.train(train_charts, train_patterns, validation_data=validation_data)
        
        return model, history
    
    def _train_xgboost(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[MarketRegimeXGBoost, Dict]:
        """Train XGBoost model"""
        
        # Prepare data
        train_features, train_labels = self.preprocessor.prepare_regime_data(train_data)
        val_features, val_labels = self.preprocessor.prepare_regime_data(val_data)
        
        # Create config
        config = MarketRegimeConfig(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=50,
            save_path=self.config.model_save_dir / "xgboost"
        )
        
        # Create and train model
        model = MarketRegimeXGBoost(config)
        
        validation_data = (val_features, val_labels) if val_data is not None else None
        history = model.train(train_features, train_labels, validation_data=validation_data)
        
        return model, history
    
    def _train_finbert(self, text_data: Dict[str, Any], val_data: Optional[Dict]) -> Tuple[FinBERT, Dict]:
        """Train FinBERT model"""
        
        # Extract text and labels
        texts = text_data.get('texts', [])
        labels = text_data.get('labels', [])
        
        # Prepare data
        train_texts, train_labels = self.preprocessor.prepare_sentiment_data(texts, labels)
        
        # Create config
        config = FinBERTConfig(
            epochs=5,  # Small number for transformers
            batch_size=16,
            learning_rate=2e-5,
            save_path=self.config.model_save_dir / "finbert"
        )
        
        # Create and train model
        model = FinBERT(config)
        
        # Split data for validation
        split_idx = int(len(train_texts) * 0.8)
        validation_data = (train_texts[split_idx:], train_labels[split_idx:]) if len(train_texts) > 10 else None
        
        history = model.train(
            train_texts[:split_idx], 
            train_labels[:split_idx], 
            validation_data=validation_data
        )
        
        return model, history
    
    def _train_ensemble(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> StackedEnsemble:
        """Train ensemble model"""
        
        logger.info("Training ensemble model")
        
        # Create config
        config = StackedEnsembleConfig(
            meta_learner_type="xgboost",
            save_path=self.config.model_save_dir / "ensemble"
        )
        
        # Create ensemble
        ensemble = StackedEnsemble(config)
        
        # Add base models
        for model_name, model in self.trained_models.items():
            ensemble.add_base_model(model_name, model)
        
        # Create ensemble training data (using regime classification as example)
        train_features, train_labels = self.preprocessor.prepare_regime_data(train_data)
        val_features, val_labels = self.preprocessor.prepare_regime_data(val_data)
        
        # Train ensemble
        validation_data = (val_features, val_labels) if val_data is not None else None
        history = ensemble.train(train_features, train_labels, validation_data=validation_data)
        
        self.training_results['StackedEnsemble'] = history
        
        return ensemble
    
    def _validate_models(self, test_data: pd.DataFrame):
        """Validate all trained models"""
        
        logger.info("Validating trained models")
        
        for model_name, model in self.trained_models.items():
            try:
                # Prepare test data based on model type
                if model_name == 'PriceLSTM':
                    test_features, test_targets = self.preprocessor.prepare_price_data(test_data)
                    metrics = model.evaluate(test_features, test_targets)
                elif model_name == 'ChartPatternCNN':
                    test_charts, test_patterns = self.preprocessor.prepare_chart_data(test_data)
                    metrics = model.evaluate(test_charts, test_patterns)
                elif model_name == 'MarketRegimeXGBoost':
                    test_features, test_labels = self.preprocessor.prepare_regime_data(test_data)
                    metrics = model.evaluate(test_features, test_labels)
                else:
                    continue  # Skip models that don't have test data
                
                # Store validation metrics
                if model_name not in self.training_results:
                    self.training_results[model_name] = {}
                self.training_results[model_name]['test_metrics'] = metrics
                
                logger.info(f"Model {model_name} validation completed", metrics=metrics)
                
            except Exception as e:
                logger.error(f"Model {model_name} validation failed", error=str(e))
    
    def _save_models(self):
        """Save all trained models"""
        
        logger.info("Saving trained models")
        
        for model_name, model in self.trained_models.items():
            try:
                model_path = self.config.model_save_dir / model_name
                model.save(model_path)
                logger.info(f"Model {model_name} saved to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model {model_name}", error=str(e))
        
        # Save training results
        results_path = self.config.model_save_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = self._make_json_serializable(self.training_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Training results saved", path=str(results_path))
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        
        summary = {
            'models_trained': list(self.trained_models.keys()),
            'training_config': {
                'parallel_training': self.config.parallel_training,
                'max_workers': self.config.max_workers,
                'random_seed': self.config.random_seed
            },
            'results': {}
        }
        
        for model_name, results in self.training_results.items():
            summary['results'][model_name] = {
                'training_completed': True,
                'test_metrics': results.get('test_metrics', {}),
                'best_score': results.get('best_score', None)
            }
        
        return summary