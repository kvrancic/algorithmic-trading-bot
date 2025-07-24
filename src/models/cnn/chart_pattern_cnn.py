"""
Chart Pattern CNN Model

Convolutional Neural Network for recognizing chart patterns in price data.
Detects patterns like head and shoulders, triangles, flags, etc.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import cv2
import structlog
from pathlib import Path

from ..base import ClassificationModel, ClassificationConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class ChartPatternConfig(ClassificationConfig):
    """Configuration for Chart Pattern CNN model"""
    
    # Chart image parameters
    chart_height: int = 64
    chart_width: int = 128
    use_volume: bool = True
    use_indicators: bool = True
    
    # CNN architecture
    conv_layers: List[int] = None  # Number of filters in each conv layer
    kernel_sizes: List[int] = None  # Kernel sizes for each conv layer
    pool_sizes: List[int] = None  # Pool sizes for each layer
    
    # Pattern classes
    pattern_classes: List[str] = None
    
    # Data augmentation
    augment_data: bool = True
    augmentation_factor: float = 2.0
    noise_level: float = 0.01
    
    # Training specific
    use_pretrained: bool = False
    freeze_layers: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.conv_layers is None:
            self.conv_layers = [32, 64, 128, 256]
        if self.kernel_sizes is None:
            self.kernel_sizes = [5, 5, 3, 3]
        if self.pool_sizes is None:
            self.pool_sizes = [2, 2, 2, 2]
        if self.pattern_classes is None:
            self.pattern_classes = [
                'no_pattern',
                'head_shoulders',
                'inverse_head_shoulders',
                'ascending_triangle',
                'descending_triangle',
                'symmetric_triangle',
                'bull_flag',
                'bear_flag',
                'double_top',
                'double_bottom',
                'cup_handle',
                'wedge_rising',
                'wedge_falling'
            ]
        self.n_classes = len(self.pattern_classes)
        self.class_names = self.pattern_classes
        self.model_type = ModelType.PATTERN_RECOGNITION
        self.name = "ChartPatternCNN"


class ChartPatternCNNNetwork(nn.Module):
    """CNN network for chart pattern recognition"""
    
    def __init__(self, config: ChartPatternConfig):
        super().__init__()
        self.config = config
        
        # Calculate input channels
        input_channels = 1  # Price chart
        if config.use_volume:
            input_channels += 1  # Volume bars
        if config.use_indicators:
            input_channels += 3  # MA, RSI, MACD
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(config.conv_layers, config.kernel_sizes, config.pool_sizes)
        ):
            # Convolutional block
            conv_layers.extend([
                nn.Conv2d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size)
            ])
            in_channels = filters
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, config.chart_height, config.chart_width)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.n_classes)
        )
        
        # Global average pooling option
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Class logits
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x


class ChartPatternCNN(ClassificationModel):
    """Chart pattern recognition CNN model"""
    
    def __init__(self, config: ChartPatternConfig):
        super().__init__(config)
        self.config: ChartPatternConfig = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.CrossEntropyLoss()
        
    def build_model(self) -> ChartPatternCNNNetwork:
        """Build the CNN model"""
        self.model = ChartPatternCNNNetwork(self.config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.early_stopping_patience // 2
        )
        
        # Log model architecture
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("CNN model built",
                   total_params=total_params,
                   trainable_params=trainable_params)
        
        return self.model
    
    def create_chart_image(
        self, 
        price_data: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> np.ndarray:
        """
        Create chart image from price data
        
        Args:
            price_data: DataFrame with OHLCV data
            start_idx: Start index
            end_idx: End index
            
        Returns:
            Chart image as numpy array
        """
        # Extract window
        window_data = price_data.iloc[start_idx:end_idx].copy()
        
        # Initialize image channels
        channels = []
        
        # Price chart channel
        price_channel = np.zeros((self.config.chart_height, self.config.chart_width))
        
        # Normalize prices
        prices = window_data['close'].values
        price_min, price_max = prices.min(), prices.max()
        if price_max - price_min > 0:
            normalized_prices = (prices - price_min) / (price_max - price_min)
        else:
            normalized_prices = np.ones_like(prices) * 0.5
        
        # Draw price line
        x_coords = np.linspace(0, self.config.chart_width - 1, len(normalized_prices)).astype(int)
        y_coords = ((1 - normalized_prices) * (self.config.chart_height - 1)).astype(int)
        
        for i in range(len(x_coords) - 1):
            cv2.line(
                price_channel,
                (x_coords[i], y_coords[i]),
                (x_coords[i+1], y_coords[i+1]),
                1.0,
                thickness=2
            )
        
        # Add candlesticks if available
        if all(col in window_data.columns for col in ['open', 'high', 'low']):
            for i, (idx, row) in enumerate(window_data.iterrows()):
                x = x_coords[i]
                
                # Normalize OHLC
                o = (row['open'] - price_min) / (price_max - price_min + 1e-8)
                h = (row['high'] - price_min) / (price_max - price_min + 1e-8)
                l = (row['low'] - price_min) / (price_max - price_min + 1e-8)
                c = (row['close'] - price_min) / (price_max - price_min + 1e-8)
                
                # Convert to y coordinates
                y_o = int((1 - o) * (self.config.chart_height - 1))
                y_h = int((1 - h) * (self.config.chart_height - 1))
                y_l = int((1 - l) * (self.config.chart_height - 1))
                y_c = int((1 - c) * (self.config.chart_height - 1))
                
                # Draw high-low line
                cv2.line(price_channel, (x, y_h), (x, y_l), 0.5, thickness=1)
                
                # Draw open-close box
                color = 0.8 if c > o else 0.3
                cv2.rectangle(
                    price_channel,
                    (x-1, min(y_o, y_c)),
                    (x+1, max(y_o, y_c)),
                    color,
                    thickness=-1
                )
        
        channels.append(price_channel)
        
        # Volume channel
        if self.config.use_volume and 'volume' in window_data.columns:
            volume_channel = np.zeros((self.config.chart_height, self.config.chart_width))
            
            volumes = window_data['volume'].values
            if volumes.max() > 0:
                normalized_volumes = volumes / volumes.max()
            else:
                normalized_volumes = np.zeros_like(volumes)
            
            # Draw volume bars
            for i, vol in enumerate(normalized_volumes):
                x = x_coords[i]
                height = int(vol * self.config.chart_height * 0.3)  # Max 30% of chart height
                cv2.rectangle(
                    volume_channel,
                    (x-1, self.config.chart_height - 1),
                    (x+1, self.config.chart_height - height),
                    1.0,
                    thickness=-1
                )
            
            channels.append(volume_channel)
        
        # Technical indicator channels
        if self.config.use_indicators:
            # Moving average
            ma_channel = np.zeros((self.config.chart_height, self.config.chart_width))
            if len(prices) >= 20:
                ma = pd.Series(prices).rolling(20).mean().fillna(method='bfill').values
                ma_normalized = (ma - price_min) / (price_max - price_min + 1e-8)
                y_ma = ((1 - ma_normalized) * (self.config.chart_height - 1)).astype(int)
                
                for i in range(len(x_coords) - 1):
                    cv2.line(
                        ma_channel,
                        (x_coords[i], y_ma[i]),
                        (x_coords[i+1], y_ma[i+1]),
                        1.0,
                        thickness=1
                    )
            channels.append(ma_channel)
            
            # RSI
            rsi_channel = np.zeros((self.config.chart_height, self.config.chart_width))
            if len(prices) >= 14:
                rsi = self.calculate_rsi(prices, 14)
                rsi_normalized = rsi / 100.0
                y_rsi = ((1 - rsi_normalized) * (self.config.chart_height - 1)).astype(int)
                
                for i in range(len(x_coords) - 1):
                    cv2.line(
                        rsi_channel,
                        (x_coords[i], y_rsi[i]),
                        (x_coords[i+1], y_rsi[i+1]),
                        1.0,
                        thickness=1
                    )
            channels.append(rsi_channel)
            
            # MACD
            macd_channel = np.zeros((self.config.chart_height, self.config.chart_width))
            if len(prices) >= 26:
                macd, signal = self.calculate_macd(prices)
                # Normalize MACD to [0, 1]
                macd_all = np.concatenate([macd, signal])
                macd_min, macd_max = macd_all.min(), macd_all.max()
                if macd_max - macd_min > 0:
                    macd_normalized = (macd - macd_min) / (macd_max - macd_min)
                    signal_normalized = (signal - macd_min) / (macd_max - macd_min)
                    
                    y_macd = ((1 - macd_normalized) * (self.config.chart_height - 1)).astype(int)
                    y_signal = ((1 - signal_normalized) * (self.config.chart_height - 1)).astype(int)
                    
                    for i in range(len(x_coords) - 1):
                        cv2.line(
                            macd_channel,
                            (x_coords[i], y_macd[i]),
                            (x_coords[i+1], y_macd[i+1]),
                            0.7,
                            thickness=1
                        )
                        cv2.line(
                            macd_channel,
                            (x_coords[i], y_signal[i]),
                            (x_coords[i+1], y_signal[i+1]),
                            0.3,
                            thickness=1
                        )
            channels.append(macd_channel)
        
        # Stack channels
        chart_image = np.stack(channels, axis=0)
        
        return chart_image
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 50.0  # Neutral RSI for initial period
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def calculate_macd(
        self, 
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.values, signal_line.values
    
    def augment_chart(self, chart: np.ndarray) -> np.ndarray:
        """Apply data augmentation to chart image"""
        augmented = chart.copy()
        
        # Add noise
        if self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level, chart.shape)
            augmented = np.clip(augmented + noise, 0, 1)
        
        # Random horizontal flip (for symmetric patterns)
        if np.random.random() > 0.5:
            augmented = np.flip(augmented, axis=2)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness, 0, 1)
        
        return augmented
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data by creating chart images"""
        
        if isinstance(data, np.ndarray):
            raise ValueError("ChartPatternCNN requires DataFrame input with OHLCV data")
        
        # Create chart images
        charts = []
        pattern_labels = []
        
        window_size = self.config.chart_width
        stride = window_size // 2  # 50% overlap
        
        for i in range(0, len(data) - window_size, stride):
            # Create chart image
            chart = self.create_chart_image(data, i, i + window_size)
            
            # Apply augmentation if training
            if is_training and self.config.augment_data:
                # Add original
                charts.append(chart)
                if labels is not None:
                    pattern_labels.append(labels.iloc[i + window_size - 1])
                
                # Add augmented versions
                for _ in range(int(self.config.augmentation_factor)):
                    augmented_chart = self.augment_chart(chart)
                    charts.append(augmented_chart)
                    if labels is not None:
                        pattern_labels.append(labels.iloc[i + window_size - 1])
            else:
                charts.append(chart)
                if labels is not None:
                    pattern_labels.append(labels.iloc[i + window_size - 1])
        
        # Convert to numpy arrays
        X = np.array(charts, dtype=np.float32)
        y = None
        
        if labels is not None:
            y = np.array(pattern_labels)
            # Encode labels if string
            if y.dtype == object:
                y = self.encode_labels(y, fit=is_training)
        
        return X, y
    
    def train(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the CNN model"""
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, train_labels, is_training=True)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        if validation_data:
            X_val, y_val = self.prepare_data(
                validation_data[0], 
                validation_data[1], 
                is_training=False
            )
        else:
            # Create validation split
            split_idx = int(len(X_train) * (1 - self.config.validation_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Balance dataset if configured
        if self.config.balance_classes:
            X_train, y_train = self.balance_dataset(X_train, y_train)
        
        # Ensure labels are encoded as integers
        if y_train.dtype == object or y_train.dtype.kind in ['U', 'S']:
            y_train = self.encode_labels(y_train, fit=True)
        if y_val.dtype == object or y_val.dtype.kind in ['U', 'S']:
            y_val = self.encode_labels(y_val, fit=False)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train.astype(int)).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val.astype(int)).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Early stopping
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_preds = []
            train_true = []
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Store metrics
                train_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_true.extend(batch_y.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.loss_fn(outputs, batch_y)
                    
                    val_losses.append(loss.item())
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss = np.mean(train_losses)
            train_acc = accuracy_score(train_true, train_preds)
            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average='weighted')
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.config.epochs}",
                           train_loss=train_loss,
                           train_acc=train_acc,
                           val_loss=val_loss,
                           val_acc=val_acc,
                           val_f1=val_f1)
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Update metadata
        self.is_trained = True
        self.metadata['last_trained'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        self.metadata['best_val_f1'] = best_val_f1
        self.training_history.append(history)
        
        logger.info("CNN training completed",
                   epochs_trained=len(history['train_loss']),
                   best_val_f1=best_val_f1)
        
        return history
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with the CNN model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        X, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i:i + self.config.batch_size]
                outputs = self.model(batch)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Decode labels if needed
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.decode_labels(predictions)
        
        return predictions
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Get prediction probabilities"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        X, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get probabilities
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i:i + self.config.batch_size]
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save CNN model"""
        # Save base model data
        base_path = super().save(path)
        
        # Save PyTorch model
        if self.model is not None:
            model_path = base_path.with_suffix('.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config.to_dict(),
                'label_encoder_classes': self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else None
            }, model_path)
            
            logger.info("CNN model saved", path=str(model_path))
        
        return base_path
    
    @classmethod
    def load(cls, path: Path) -> 'ChartPatternCNN':
        """Load CNN model"""
        # Load base model data
        model_instance = super().load(path)
        
        # Load PyTorch model
        model_path = path.with_suffix('.pt')
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=model_instance.device)
            
            # Rebuild model
            model_instance.build_model()
            
            # Load weights
            model_instance.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer if available
            if checkpoint.get('optimizer_state_dict') and model_instance.optimizer:
                model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore label encoder
            if checkpoint.get('label_encoder_classes') is not None:
                model_instance.label_encoder.classes_ = checkpoint['label_encoder_classes']
            
            logger.info("CNN model loaded", path=str(model_path))
        
        return model_instance