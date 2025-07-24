"""
Market Regime XGBoost Classifier

XGBoost model for classifying market regimes (bull, bear, sideways, volatile).
Uses technical indicators and market microstructure features.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import structlog
from pathlib import Path
import json
import joblib

from ..base import ClassificationModel, ClassificationConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class MarketRegimeConfig(ClassificationConfig):
    """Configuration for Market Regime XGBoost model"""
    
    # XGBoost parameters
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    gamma: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.05
    reg_lambda: float = 1.0
    
    # Training parameters
    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"  # For multi-class
    
    # Feature engineering
    technical_indicators: List[str] = None
    market_structure_features: List[str] = None
    lookback_periods: List[int] = None
    
    # Market regime classes
    regime_classes: List[str] = None
    regime_thresholds: Dict[str, float] = None
    
    # Hyperparameter tuning
    enable_hyperopt: bool = False
    hyperopt_trials: int = 50
    hyperopt_cv_folds: int = 5
    
    # Feature importance
    track_feature_importance: bool = True
    importance_type: str = "gain"  # "gain", "weight", "cover", "total_gain", "total_cover"
    
    def __post_init__(self):
        super().__post_init__()
        if self.technical_indicators is None:
            self.technical_indicators = [
                'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
                'atr', 'adx', 'obv', 'vwap', 'pivot_points',
                'fibonacci_retracements', 'ichimoku', 'stochastic'
            ]
        if self.market_structure_features is None:
            self.market_structure_features = [
                'spread', 'depth_imbalance', 'trade_flow_toxicity',
                'kyle_lambda', 'roll_measure', 'amihud_illiquidity',
                'microstructure_noise', 'pin_score', 'vpin'
            ]
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100, 200]
        if self.regime_classes is None:
            self.regime_classes = [
                'strong_bull',
                'bull',
                'sideways',
                'bear',
                'strong_bear',
                'high_volatility'
            ]
        if self.regime_thresholds is None:
            # Default thresholds for regime classification
            self.regime_thresholds = {
                'return_threshold': 0.02,  # 2% for strong moves
                'volatility_threshold': 0.03,  # 3% for high volatility
                'trend_strength': 0.7  # ADX threshold for strong trend
            }
        self.n_classes = len(self.regime_classes)
        self.class_names = self.regime_classes
        self.model_type = ModelType.REGIME_CLASSIFICATION
        self.name = "MarketRegimeXGBoost"


class MarketRegimeXGBoost(ClassificationModel):
    """XGBoost model for market regime classification"""
    
    def __init__(self, config: MarketRegimeConfig):
        super().__init__(config)
        self.config: MarketRegimeConfig = config
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        
    def build_model(self) -> xgb.XGBClassifier:
        """Build the XGBoost model"""
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective='multi:softprob',
            eval_metric=self.config.eval_metric,
            use_label_encoder=False,
            random_state=self.config.random_seed,
            n_jobs=-1
        )
        
        logger.info("XGBoost model built",
                   n_estimators=self.config.n_estimators,
                   max_depth=self.config.max_depth)
        
        return self.model
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for regime classification"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based indicators
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Simple and Exponential Moving Averages
        for period in self.config.lookback_periods:
            if len(close) >= period:
                features[f'sma_{period}'] = close.rolling(period).mean()
                features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
                features[f'sma_ratio_{period}'] = close / features[f'sma_{period}']
        
        # RSI
        for period in [14, 21, 28]:
            if len(close) >= period + 1:
                features[f'rsi_{period}'] = self.calculate_rsi(close, period)
        
        # MACD
        if len(close) >= 26:
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            if len(close) >= period:
                sma = close.rolling(period).mean()
                std = close.rolling(period).std()
                features[f'bb_upper_{period}'] = sma + 2 * std
                features[f'bb_lower_{period}'] = sma - 2 * std
                features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
                features[f'bb_position_{period}'] = (close - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']
        
        # ATR (Average True Range)
        for period in [14, 21]:
            if len(close) >= period + 1:
                features[f'atr_{period}'] = self.calculate_atr(high, low, close, period)
        
        # ADX (Average Directional Index)
        if len(close) >= 15:
            features['adx'] = self.calculate_adx(high, low, close, 14)
        
        # Volume indicators
        if volume is not None and len(volume) > 0:
            # OBV (On Balance Volume)
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            features['obv'] = obv
            
            # VWAP
            features['vwap'] = (close * volume).cumsum() / volume.cumsum()
            
            # Volume moving averages
            for period in [5, 10, 20]:
                if len(volume) >= period:
                    features[f'volume_ma_{period}'] = volume.rolling(period).mean()
                    features[f'volume_ratio_{period}'] = volume / features[f'volume_ma_{period}']
        
        # Returns and volatility
        for period in self.config.lookback_periods:
            if len(close) >= period + 1:
                features[f'return_{period}'] = close.pct_change(period)
                features[f'volatility_{period}'] = close.pct_change().rolling(period).std()
                features[f'skew_{period}'] = close.pct_change().rolling(period).skew()
                features[f'kurtosis_{period}'] = close.pct_change().rolling(period).kurt()
        
        # High-Low spread
        features['hl_spread'] = (high - low) / close
        features['hl_spread_ma'] = features['hl_spread'].rolling(20).mean()
        
        return features
    
    def calculate_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        features = pd.DataFrame(index=data.index)
        
        # Basic microstructure
        if all(col in data.columns for col in ['bid', 'ask']):
            features['spread'] = data['ask'] - data['bid']
            features['spread_pct'] = features['spread'] / data['mid_price']
            features['spread_ma'] = features['spread'].rolling(20).mean()
        
        # Price impact and liquidity measures
        if 'volume' in data.columns and 'close' in data.columns:
            # Amihud illiquidity measure
            returns = data['close'].pct_change().abs()
            features['amihud_illiquidity'] = (returns / data['volume']).rolling(20).mean()
            
            # Kyle's lambda (simplified)
            price_changes = data['close'].diff()
            signed_volume = data['volume'] * np.sign(price_changes)
            features['kyle_lambda'] = price_changes.rolling(20).cov(signed_volume) / signed_volume.rolling(20).var()
        
        # Order flow imbalance
        if all(col in data.columns for col in ['bid_volume', 'ask_volume']):
            features['order_imbalance'] = (data['bid_volume'] - data['ask_volume']) / (data['bid_volume'] + data['ask_volume'])
            features['order_imbalance_ma'] = features['order_imbalance'].rolling(20).mean()
        
        # Volatility clustering
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            features['garch_volatility'] = returns.abs().rolling(20).mean()
            features['volatility_ratio'] = returns.rolling(5).std() / returns.rolling(20).std()
        
        return features
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Calculate directional movement
        up_move = high.diff()
        down_move = -low.diff()
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)
        
        # Calculate directional indicators
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def classify_market_regime(self, features: pd.DataFrame) -> pd.Series:
        """Classify market regime based on features (for creating labels)"""
        # This is a rule-based classification for creating training labels
        # In practice, these labels might come from domain experts or more sophisticated methods
        
        regimes = pd.Series(index=features.index, dtype=str)
        
        # Calculate key metrics
        returns = features.get('return_20', pd.Series(index=features.index))
        volatility = features.get('volatility_20', pd.Series(index=features.index))
        trend_strength = features.get('adx', pd.Series(index=features.index))
        
        # Default to sideways
        regimes[:] = 'sideways'
        
        # Strong bull market
        strong_bull_mask = (
            (returns > self.config.regime_thresholds['return_threshold']) &
            (trend_strength > self.config.regime_thresholds['trend_strength'])
        )
        regimes[strong_bull_mask] = 'strong_bull'
        
        # Bull market
        bull_mask = (
            (returns > 0) &
            (returns <= self.config.regime_thresholds['return_threshold']) &
            (trend_strength > 30)
        )
        regimes[bull_mask] = 'bull'
        
        # Bear market
        bear_mask = (
            (returns < 0) &
            (returns >= -self.config.regime_thresholds['return_threshold']) &
            (trend_strength > 30)
        )
        regimes[bear_mask] = 'bear'
        
        # Strong bear market
        strong_bear_mask = (
            (returns < -self.config.regime_thresholds['return_threshold']) &
            (trend_strength > self.config.regime_thresholds['trend_strength'])
        )
        regimes[strong_bear_mask] = 'strong_bear'
        
        # High volatility regime
        high_vol_mask = volatility > self.config.regime_thresholds['volatility_threshold']
        regimes[high_vol_mask] = 'high_volatility'
        
        return regimes
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data by calculating all features"""
        
        if isinstance(data, np.ndarray):
            raise ValueError("MarketRegimeXGBoost requires DataFrame input with OHLCV data")
        
        # Calculate technical indicators
        technical_features = self.calculate_technical_indicators(data)
        
        # Calculate market microstructure features
        microstructure_features = self.calculate_market_microstructure_features(data)
        
        # Combine all features
        all_features = pd.concat([technical_features, microstructure_features], axis=1)
        
        # Drop rows with NaN values
        all_features = all_features.dropna()
        
        # Store feature names
        if is_training:
            self.feature_names = all_features.columns.tolist()
        
        # Convert to numpy
        X = all_features.values
        
        # Process labels
        y = None
        if labels is not None:
            if isinstance(labels, pd.Series):
                # Align labels with features index
                y = labels.loc[all_features.index]
            else:
                # Assume labels are already aligned
                y = labels[-len(all_features):]
            
            # Encode labels if string
            if hasattr(y, 'dtype') and y.dtype == np.object:
                y = self.encode_labels(y, fit=is_training)
        elif is_training:
            # Auto-generate labels based on regime classification rules
            y = self.classify_market_regime(all_features)
            y = self.encode_labels(y, fit=True)
        
        return X, y
    
    def hyperparameter_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Perform hyperparameter optimization using GridSearchCV"""
        
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # Use stratified k-fold for better class balance
        cv = StratifiedKFold(n_splits=self.config.hyperopt_cv_folds, shuffle=True, random_state=self.config.random_seed)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Starting hyperparameter optimization")
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info("Hyperparameter optimization completed",
                   best_params=best_params,
                   best_score=best_score)
        
        # Update model with best parameters
        self.model.set_params(**best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    
    def train(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the XGBoost model"""
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, train_labels, is_training=True)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        eval_set = []
        if validation_data:
            X_val, y_val = self.prepare_data(
                validation_data[0], 
                validation_data[1], 
                is_training=False
            )
            eval_set = [(X_val, y_val)]
        else:
            # Create validation split
            split_idx = int(len(X_train) * (1 - self.config.validation_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            eval_set = [(X_val, y_val)]
        
        # Balance dataset if configured
        if self.config.balance_classes:
            X_train, y_train = self.balance_dataset(X_train, y_train)
        
        # Hyperparameter optimization if enabled
        if self.config.enable_hyperopt:
            hyperopt_results = self.hyperparameter_optimization(X_train, y_train)
            self.metadata['hyperopt_results'] = hyperopt_results
        
        # Train model
        logger.info("Training XGBoost model",
                   n_samples=len(X_train),
                   n_features=X_train.shape[1])
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False
        )
        
        # Get feature importance
        if self.config.track_feature_importance:
            importance = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            sorted_importance = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Log top features
            top_features = sorted_importance[:10]
            logger.info("Top 10 important features",
                       features=[f"{name}: {imp:.4f}" for name, imp in top_features])
        
        # Get training history
        history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_report = classification_report(
            y_val, val_predictions,
            target_names=self.config.regime_classes,
            output_dict=True
        )
        
        history['validation_report'] = val_report
        history['validation_f1'] = val_report['weighted avg']['f1-score']
        
        # Update metadata
        self.is_trained = True
        self.metadata['last_trained'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        self.metadata['n_features'] = X_train.shape[1]
        self.metadata['feature_names'] = self.feature_names
        self.training_history.append(history)
        
        logger.info("XGBoost training completed",
                   best_iteration=history['best_iteration'],
                   validation_f1=history['validation_f1'])
        
        return history
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with the XGBoost model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        X, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
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
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N important features"""
        if not self.feature_importance:
            raise ValueError("Feature importance not available. Train model with track_feature_importance=True")
        
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_importance[:top_n])
    
    def explain_prediction(self, data: pd.DataFrame, index: int = -1) -> Dict[str, Any]:
        """Explain a single prediction using feature contributions"""
        
        # Prepare data
        X, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Get prediction
        prediction = self.model.predict(X[index:index+1])[0]
        probabilities = self.model.predict_proba(X[index:index+1])[0]
        
        # Get feature contributions (SHAP values would be better, but keeping it simple)
        feature_values = dict(zip(self.feature_names, X[index]))
        
        # Sort features by importance for this prediction
        important_features = []
        for feature, importance in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            if feature in feature_values:
                important_features.append({
                    'feature': feature,
                    'value': feature_values[feature],
                    'importance': importance
                })
        
        explanation = {
            'prediction': self.config.regime_classes[prediction],
            'probabilities': dict(zip(self.config.regime_classes, probabilities)),
            'important_features': important_features
        }
        
        return explanation
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save XGBoost model"""
        # Save base model data
        base_path = super().save(path)
        
        # Save XGBoost model
        if self.model is not None:
            model_path = base_path.with_suffix('.xgb')
            self.model.save_model(model_path)
            
            # Save additional metadata
            metadata_path = base_path.with_suffix('.meta.json')
            metadata = {
                'config': self.config.to_dict(),
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("XGBoost model saved", path=str(model_path))
        
        return base_path
    
    @classmethod
    def load(cls, path: Path) -> 'MarketRegimeXGBoost':
        """Load XGBoost model"""
        # Load base model data
        model_instance = super().load(path)
        
        # Load XGBoost model
        model_path = path.with_suffix('.xgb')
        if model_path.exists():
            model_instance.model = xgb.XGBClassifier()
            model_instance.model.load_model(model_path)
            
            # Load metadata
            metadata_path = path.with_suffix('.meta.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_instance.feature_names = metadata.get('feature_names', [])
                model_instance.feature_importance = metadata.get('feature_importance', {})
                
                # Restore label encoder
                if metadata.get('label_encoder_classes'):
                    model_instance.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
            
            logger.info("XGBoost model loaded", path=str(model_path))
        
        return model_instance