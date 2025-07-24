"""
Time Series Model Base Class

Base class for time series models like LSTM, GRU, etc.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import structlog

from .base_model import BaseModel, ModelConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class TimeSeriesConfig(ModelConfig):
    """Configuration specific to time series models"""
    
    # Time series specific parameters
    sequence_length: int = 60  # Number of timesteps to look back
    stride: int = 1  # Step size for sliding window
    forecast_horizon: int = 1  # Number of timesteps to predict
    
    # Preprocessing
    scaling_method: str = "standard"  # "standard", "minmax", or "none"
    detrend: bool = False
    handle_missing: str = "interpolate"  # "interpolate", "forward_fill", "drop"
    
    # Feature engineering
    add_time_features: bool = True  # Add hour, day, month features
    add_technical_indicators: bool = True
    add_lag_features: bool = True
    lag_periods: List[int] = None
    
    # Model architecture hints
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.lag_periods is None:
            self.lag_periods = [1, 5, 10, 20]
        self.model_type = ModelType.PRICE_PREDICTION


class TimeSeriesModel(BaseModel):
    """Base class for time series models"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__(config)
        self.config: TimeSeriesConfig = config
        self.scaler = None
        self.feature_columns = []
        self.target_columns = []
        
        # Initialize scaler based on config
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time series models
        
        Args:
            data: Input data (n_samples, n_features)
            targets: Target data (n_samples, n_targets)
            
        Returns:
            Tuple of (sequences, target_sequences)
        """
        n_samples = len(data)
        seq_length = self.config.sequence_length
        stride = self.config.stride
        
        # Calculate number of sequences
        n_sequences = (n_samples - seq_length - self.config.forecast_horizon + 1) // stride
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {seq_length + self.config.forecast_horizon} samples")
        
        # Create sequences
        sequences = []
        target_sequences = [] if targets is not None else None
        
        for i in range(0, n_sequences * stride, stride):
            # Input sequence
            seq = data[i:i + seq_length]
            sequences.append(seq)
            
            # Target sequence
            if targets is not None:
                if self.config.forecast_horizon == 1:
                    target = targets[i + seq_length]
                else:
                    target = targets[i + seq_length:i + seq_length + self.config.forecast_horizon]
                target_sequences.append(target)
        
        sequences = np.array(sequences)
        if target_sequences is not None:
            target_sequences = np.array(target_sequences)
        
        return sequences, target_sequences
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to dataframe"""
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found for time features")
            return df
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop intermediate columns
        df.drop(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter'], 
                axis=1, inplace=True)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add lagged features"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in self.config.lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
                # Add rolling statistics
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
                df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
                df[f'{col}_rolling_mean_30'] = df[col].rolling(window=30).mean()
                df[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()
        
        return df
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare time series data"""
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        df = data.copy()
        
        # Handle missing values
        if self.config.handle_missing == "interpolate":
            df = df.interpolate(method='linear', limit_direction='both')
        elif self.config.handle_missing == "forward_fill":
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif self.config.handle_missing == "drop":
            df = df.dropna()
        
        # Add features if configured
        if self.config.add_time_features and 'timestamp' in df.columns:
            df = self.add_time_features(df)
        
        # Add lag features for numeric columns
        if self.config.add_lag_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = self.add_lag_features(df, numeric_cols)
        
        # Drop any remaining NaN values created by lagging
        df = df.dropna()
        
        # Store feature columns
        if is_training:
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['timestamp', 'symbol']]
            if labels is not None:
                if isinstance(labels, pd.Series):
                    self.target_columns = [labels.name]
                elif isinstance(labels, pd.DataFrame):
                    self.target_columns = labels.columns.tolist()
        
        # Select features
        feature_data = df[self.feature_columns].values
        
        # Scale features
        if self.scaler is not None:
            if is_training:
                feature_data = self.scaler.fit_transform(feature_data)
            else:
                feature_data = self.scaler.transform(feature_data)
        
        # Prepare labels
        if labels is not None:
            if isinstance(labels, (pd.Series, pd.DataFrame)):
                labels = labels.values
            # Ensure labels are aligned with data
            labels = labels[-len(feature_data):]
        
        # Create sequences
        sequences, target_sequences = self.create_sequences(feature_data, labels)
        
        logger.debug("Data prepared",
                    n_sequences=len(sequences),
                    sequence_shape=sequences.shape,
                    n_features=len(self.feature_columns))
        
        return sequences, target_sequences
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions"""
        if self.scaler is None or self.config.scaling_method == "none":
            return predictions
        
        # For multi-feature scaling, we need to pad predictions
        n_features = len(self.feature_columns)
        n_targets = predictions.shape[-1] if len(predictions.shape) > 1 else 1
        
        if n_targets < n_features:
            # Pad predictions to match feature dimensions
            padding = np.zeros((len(predictions), n_features - n_targets))
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            padded = np.hstack([predictions, padding])
            
            # Inverse transform
            unscaled = self.scaler.inverse_transform(padded)
            
            # Extract target columns
            return unscaled[:, :n_targets]
        else:
            # Direct inverse transform
            return self.scaler.inverse_transform(predictions)
    
    def create_train_val_split(
        self, 
        sequences: np.ndarray, 
        targets: np.ndarray,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create time series aware train/validation split"""
        n_samples = len(sequences)
        split_idx = int(n_samples * (1 - val_size))
        
        train_X = sequences[:split_idx]
        train_y = targets[:split_idx]
        val_X = sequences[split_idx:]
        val_y = targets[split_idx:]
        
        return train_X, train_y, val_X, val_y
    
    def get_lookback_requirement(self) -> int:
        """Get minimum number of historical points needed"""
        base_lookback = self.config.sequence_length
        
        if self.config.add_lag_features and self.config.lag_periods:
            base_lookback += max(self.config.lag_periods)
        
        # Add buffer for rolling features
        if self.config.add_lag_features:
            base_lookback += 30  # For 30-day rolling features
        
        return base_lookback