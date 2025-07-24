"""
Base Model Interface for QuantumSentiment Trading Bot

Defines the common interface and functionality for all ML models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Types of models in the system"""
    PRICE_PREDICTION = "price_prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    REGIME_CLASSIFICATION = "regime_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENSEMBLE = "ensemble"


@dataclass
class ModelConfig:
    """Base configuration for all models"""
    name: str
    version: str = "1.0.0"
    model_type: ModelType = ModelType.PRICE_PREDICTION
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Model parameters
    input_features: List[str] = field(default_factory=list)
    output_features: List[str] = field(default_factory=list)
    lookback_period: int = 60  # For time series models
    prediction_horizon: int = 1  # Steps ahead to predict
    
    # Hardware settings
    device: str = "cpu"  # "cpu" or "cuda"
    num_workers: int = 4
    
    # Persistence
    save_path: Path = Path("models/saved")
    checkpoint_interval: int = 10  # Save every N epochs
    
    # Feature parameters
    feature_importance_threshold: float = 0.01
    max_features: Optional[int] = None
    
    # Regularization
    dropout_rate: float = 0.2
    l1_regularization: float = 0.0
    l2_regularization: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type.value,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'input_features': self.input_features,
            'output_features': self.output_features,
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon,
            'device': self.device,
            'num_workers': self.num_workers,
            'save_path': str(self.save_path),
            'checkpoint_interval': self.checkpoint_interval,
            'feature_importance_threshold': self.feature_importance_threshold,
            'max_features': self.max_features,
            'dropout_rate': self.dropout_rate,
            'l1_regularization': self.l1_regularization,
            'l2_regularization': self.l2_regularization
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        config_dict = config_dict.copy()
        config_dict['model_type'] = ModelType(config_dict['model_type'])
        config_dict['save_path'] = Path(config_dict['save_path'])
        return cls(**config_dict)


class BaseModel(ABC):
    """Abstract base class for all trading models"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_trained': None,
            'total_training_time': 0,
            'training_samples': 0,
            'validation_metrics': {}
        }
        
        # Create save directory
        self.config.save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model initialized",
                   name=self.config.name,
                   type=self.config.model_type.value)
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[Any, Optional[Any]]:
        """
        Prepare data for model input
        
        Args:
            data: Input data
            labels: Target labels (optional for inference)
            is_training: Whether preparing for training or inference
            
        Returns:
            Tuple of (prepared_data, prepared_labels)
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_data: Training data
            train_labels: Training labels
            validation_data: Optional validation data tuple
            **kwargs: Additional training parameters
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions
        
        Args:
            data: Input data
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        test_data: Union[pd.DataFrame, np.ndarray],
        test_labels: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            test_data: Test data
            test_labels: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save model to disk
        
        Args:
            path: Save path (uses config default if None)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.name}_{self.config.version}_{timestamp}"
            path = self.config.save_path / filename
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.with_suffix('.pkl')
        joblib.dump(self.model, model_path)
        
        # Save config
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'metadata': self.metadata,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history
            }, f, indent=2)
        
        logger.info("Model saved",
                   model_path=str(model_path),
                   config_path=str(config_path))
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> 'BaseModel':
        """
        Load model from disk
        
        Args:
            path: Path to saved model (without extension)
            
        Returns:
            Loaded model instance
        """
        # Load config
        config_path = path.with_suffix('.json')
        with open(config_path, 'r') as f:
            saved_data = json.load(f)
        
        # Create model instance
        config = ModelConfig.from_dict(saved_data['config'])
        model_instance = cls(config)
        
        # Load model weights
        model_path = path.with_suffix('.pkl')
        model_instance.model = joblib.load(model_path)
        
        # Restore metadata
        model_instance.metadata = saved_data['metadata']
        model_instance.feature_importance = saved_data['feature_importance']
        model_instance.training_history = saved_data['training_history']
        model_instance.is_trained = True
        
        logger.info("Model loaded",
                   name=config.name,
                   version=config.version,
                   path=str(path))
        
        return model_instance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.config.to_dict()
    
    def set_params(self, **params) -> None:
        """Update model parameters"""
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("Unknown parameter", param=key)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage in MB"""
        import sys
        
        model_size = sys.getsizeof(self.model) / 1024 / 1024 if self.model else 0
        config_size = sys.getsizeof(self.config) / 1024 / 1024
        history_size = sys.getsizeof(self.training_history) / 1024 / 1024
        
        return {
            'model_mb': model_size,
            'config_mb': config_size,
            'history_mb': history_size,
            'total_mb': model_size + config_size + history_size
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, type={self.config.model_type.value}, trained={self.is_trained})"
    
    def __repr__(self) -> str:
        return self.__str__()