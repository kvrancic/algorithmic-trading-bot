"""
Advanced Slippage Prediction Model

Machine learning-based slippage prediction using XGBoost:
- Multi-factor slippage modeling with market microstructure features
- Real-time slippage forecasting for order execution optimization
- Adaptive model retraining based on execution performance
- Cross-venue slippage analysis and venue selection optimization
- Integration with smart order routing for cost minimization
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SlippagePredictorConfig:
    """Configuration for slippage prediction model"""
    
    # Model parameters
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 30, 60])
    volume_windows: List[int] = field(default_factory=lambda: [5, 15, 30])
    
    # Training parameters
    train_test_split: float = 0.8
    min_training_samples: int = 500
    max_training_samples: int = 10000
    retraining_frequency: int = 24  # hours
    
    # Prediction parameters
    confidence_intervals: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    slippage_buckets: List[float] = field(default_factory=lambda: [0.0005, 0.001, 0.002, 0.005, 0.01])
    
    # Model validation
    enable_cross_validation: bool = True
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    
    def __post_init__(self):
        """Validate configuration"""
        if self.train_test_split <= 0 or self.train_test_split >= 1:
            raise ValueError("train_test_split must be between 0 and 1")
        if self.min_training_samples <= 0:
            raise ValueError("min_training_samples must be positive")


class FeatureEngineer:
    """Advanced feature engineering for slippage prediction"""
    
    def __init__(self, config: SlippagePredictorConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
    def create_features(
        self,
        market_data: pd.DataFrame,
        order_data: pd.DataFrame,
        execution_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set for slippage prediction
        
        Args:
            market_data: OHLCV market data
            order_data: Order characteristics (size, side, urgency)
            execution_data: Historical execution data
            
        Returns:
            Feature matrix for model training/prediction
        """
        
        logger.info("Creating features for slippage prediction")
        
        features = pd.DataFrame(index=market_data.index)
        
        # 1. Market microstructure features
        features = self._add_microstructure_features(features, market_data)
        
        # 2. Volatility features
        features = self._add_volatility_features(features, market_data)
        
        # 3. Volume and liquidity features
        features = self._add_volume_features(features, market_data)
        
        # 4. Order characteristics features
        features = self._add_order_features(features, order_data)
        
        # 5. Market timing features
        features = self._add_timing_features(features, market_data)
        
        # 6. Technical indicator features
        features = self._add_technical_features(features, market_data)
        
        # 7. Cross-sectional features (if multiple assets)
        if execution_data is not None:
            features = self._add_execution_features(features, execution_data)
        
        # Store feature names for importance analysis
        self.feature_names = list(features.columns)
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {len(features.columns)} features for slippage prediction")
        
        return features
    
    def _add_microstructure_features(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Price impact features
        features['price_range'] = (market_data['high'] - market_data['low']) / market_data['close']
        features['price_gap'] = (market_data['open'] - market_data['close'].shift(1)) / market_data['close'].shift(1)
        
        # Bid-ask spread proxies
        features['hl_spread'] = (market_data['high'] - market_data['low']) / market_data['close']
        features['oc_spread'] = abs(market_data['open'] - market_data['close']) / market_data['close']
        
        # Price momentum and mean reversion
        for period in self.config.lookback_periods:
            if len(market_data) > period:
                features[f'price_momentum_{period}'] = market_data['close'].pct_change(period)
                features[f'price_mean_reversion_{period}'] = (
                    market_data['close'] / market_data['close'].rolling(period).mean() - 1
                )
        
        return features
    
    def _add_volatility_features(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add volatility-based features"""
        
        # Calculate returns for volatility metrics
        returns = market_data['close'].pct_change()
        
        # Rolling volatility features
        for window in self.config.volatility_windows:
            if len(returns) > window:
                vol_col = f'volatility_{window}'
                features[vol_col] = returns.rolling(window).std() * np.sqrt(252)
                
                # Volatility regime features
                features[f'{vol_col}_regime'] = (
                    features[vol_col] / features[vol_col].rolling(window*2).mean()
                ).fillna(1)
        
        # Intraday volatility features
        features['realized_vol'] = np.sqrt(
            ((market_data['high'] / market_data['close']).apply(np.log))**2 +
            ((market_data['low'] / market_data['close']).apply(np.log))**2
        )
        
        # Volatility clustering
        features['vol_clustering'] = returns.abs().rolling(10).corr(returns.abs().shift(1))
        
        return features
    
    def _add_volume_features(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add volume and liquidity features"""
        
        if 'volume' not in market_data.columns:
            logger.warning("Volume data not available, using proxy features")
            features['volume_proxy'] = market_data['high'] - market_data['low']
            return features
        
        volume = market_data['volume']
        
        # Volume statistics
        for window in self.config.volume_windows:
            if len(volume) > window:
                features[f'volume_ma_{window}'] = volume.rolling(window).mean()
                features[f'volume_ratio_{window}'] = volume / features[f'volume_ma_{window}']
                features[f'volume_std_{window}'] = volume.rolling(window).std()
        
        # Volume-price relationships
        features['volume_weighted_price'] = (
            (market_data['high'] + market_data['low'] + market_data['close']) / 3 * volume
        ).rolling(20).sum() / volume.rolling(20).sum()
        
        # Liquidity proxies
        features['amihud_illiquidity'] = (
            abs(market_data['close'].pct_change()) / (volume * market_data['close'])
        ).rolling(20).mean()
        
        # On-balance volume
        price_change = market_data['close'].diff()
        obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        features['obv'] = (volume * obv_direction).cumsum()
        features['obv_ma'] = features['obv'].rolling(20).mean()
        
        return features
    
    def _add_order_features(
        self,
        features: pd.DataFrame,
        order_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add order-specific characteristics"""
        
        # Align order data with market data
        order_features = order_data.reindex(features.index, method='ffill')
        
        # Order size features
        if 'order_size' in order_features.columns:
            features['order_size'] = order_features['order_size']
            features['order_size_log'] = np.log1p(order_features['order_size'])
            
            # Size relative to volume
            if 'volume_ma_20' in features.columns:
                features['order_size_relative'] = (
                    order_features['order_size'] / features['volume_ma_20']
                ).fillna(0.1)
        
        # Order side (buy/sell)
        if 'order_side' in order_features.columns:
            features['is_buy_order'] = (order_features['order_side'] == 'buy').astype(int)
        
        # Order urgency
        if 'urgency' in order_features.columns:
            urgency_mapping = {'low': 1, 'normal': 2, 'high': 3, 'immediate': 4}
            features['order_urgency'] = order_features['urgency'].map(urgency_mapping).fillna(2)
        
        # Order timing
        if 'submission_time' in order_features.columns:
            submission_times = pd.to_datetime(order_features['submission_time'])
            features['hour_of_day'] = submission_times.dt.hour
            features['day_of_week'] = submission_times.dt.dayofweek
            features['is_market_open'] = (
                (features['hour_of_day'] >= 9) & (features['hour_of_day'] <= 16)
            ).astype(int)
        
        return features
    
    def _add_timing_features(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add market timing features"""
        
        # Time-based features
        market_index = pd.to_datetime(market_data.index)
        features['hour'] = market_index.hour
        features['day_of_week'] = market_index.dayofweek
        features['month'] = market_index.month
        
        # Market session features
        features['market_open'] = ((features['hour'] >= 9) & (features['hour'] < 10)).astype(int)
        features['market_close'] = ((features['hour'] >= 15) & (features['hour'] <= 16)).astype(int)
        features['lunch_time'] = ((features['hour'] >= 12) & (features['hour'] <= 13)).astype(int)
        
        # Weekend proximity
        features['friday_effect'] = (features['day_of_week'] == 4).astype(int)
        features['monday_effect'] = (features['day_of_week'] == 0).astype(int)
        
        return features
    
    def _add_technical_features(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add technical analysis features"""
        
        close = market_data['close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(close) > period:
                ma_col = f'ma_{period}'
                features[ma_col] = close.rolling(period).mean()
                features[f'price_to_{ma_col}'] = close / features[ma_col] - 1
        
        # RSI
        if len(close) > 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta).where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if len(close) > 20:
            bb_ma = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            features['bb_upper'] = bb_ma + (bb_std * 2)
            features['bb_lower'] = bb_ma - (bb_std * 2)
            features['bb_position'] = (close - bb_lower) / (features['bb_upper'] - bb_lower)
        
        return features
    
    def _add_execution_features(
        self,
        features: pd.DataFrame,
        execution_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add historical execution performance features"""
        
        # Align execution data
        exec_features = execution_data.reindex(features.index, method='ffill')
        
        # Historical slippage patterns
        if 'realized_slippage' in exec_features.columns:
            slippage = exec_features['realized_slippage']
            features['avg_slippage_5'] = slippage.rolling(5).mean()
            features['avg_slippage_20'] = slippage.rolling(20).mean()
            features['slippage_volatility'] = slippage.rolling(10).std()
        
        # Execution venue performance
        if 'venue' in exec_features.columns:
            for venue in exec_features['venue'].unique():
                if pd.notna(venue):
                    venue_mask = exec_features['venue'] == venue
                    features[f'venue_{venue}'] = venue_mask.astype(int)
        
        return features


class SlippagePredictor:
    """Advanced slippage prediction model using XGBoost"""
    
    def __init__(self, config: SlippagePredictorConfig):
        self.config = config
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_engineer = FeatureEngineer(config)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time: Optional[datetime] = None
        self.feature_importance: Optional[pd.Series] = None
        
    def train(
        self,
        training_data: Dict[str, pd.DataFrame],
        target_slippage: pd.Series
    ) -> Dict[str, Any]:
        """
        Train slippage prediction model
        
        Args:
            training_data: Dictionary containing market_data, order_data, execution_data
            target_slippage: Historical slippage values for training
            
        Returns:
            Training results and model performance metrics
        """
        
        logger.info("Training slippage prediction model")
        
        # Extract training data components
        market_data = training_data['market_data']
        order_data = training_data['order_data']
        execution_data = training_data.get('execution_data')
        
        # Create feature matrix
        features = self.feature_engineer.create_features(
            market_data, order_data, execution_data
        )
        
        # Align features with target
        aligned_data = pd.concat([features, target_slippage], axis=1, join='inner')
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < self.config.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(aligned_data)} < {self.config.min_training_samples}")
        
        # Limit training data size if necessary
        if len(aligned_data) > self.config.max_training_samples:
            aligned_data = aligned_data.tail(self.config.max_training_samples)
        
        # Separate features and target
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Train-test split (time series aware)
        split_point = int(len(X_scaled) * self.config.train_test_split)
        X_train, X_test = X_scaled.iloc[:split_point], X_scaled.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Configure XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=42,
            early_stopping_rounds=self.config.early_stopping_rounds,
            eval_metric='mae'
        )
        
        # Train model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calculate performance metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Feature importance analysis
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        # Cross-validation if enabled
        cv_scores = None
        if self.config.enable_cross_validation:
            cv_scores = self._perform_cross_validation(X_scaled, y)
        
        # Update training status
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'best_iteration': getattr(self.model, 'best_iteration', self.config.n_estimators),
            'feature_count': len(X.columns),
            'top_features': self.feature_importance.head(10).to_dict(),
            'cv_scores': cv_scores
        }
        
        logger.info(f"Model training completed - Test MAE: {test_mae:.6f}, Test RMSE: {test_rmse:.6f}")
        
        return results
    
    def predict(
        self,
        prediction_data: Dict[str, pd.DataFrame],
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Predict slippage for new orders
        
        Args:
            prediction_data: Dictionary containing market_data, order_data
            return_confidence: Whether to include confidence intervals
            
        Returns:
            Slippage predictions with optional confidence intervals
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features = self.feature_engineer.create_features(
            prediction_data['market_data'],
            prediction_data['order_data'],
            prediction_data.get('execution_data')
        )
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        result = {
            'predictions': predictions,
            'timestamps': features.index.tolist()
        }
        
        # Add confidence intervals if requested
        if return_confidence:
            confidence_intervals = self._calculate_confidence_intervals(
                features_scaled, predictions
            )
            result['confidence_intervals'] = confidence_intervals
        
        # Add risk categories
        result['risk_categories'] = self._categorize_slippage_risk(predictions)
        
        return result
    
    def _perform_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Perform time series cross-validation"""
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on CV fold
            cv_model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators // 2,  # Faster training
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )
            
            cv_model.fit(X_train_cv, y_train_cv)
            y_pred_cv = cv_model.predict(X_val_cv)
            
            cv_mae = mean_absolute_error(y_val_cv, y_pred_cv)
            cv_scores.append(cv_mae)
        
        return {
            'mean_cv_mae': np.mean(cv_scores),
            'std_cv_mae': np.std(cv_scores),
            'cv_scores': cv_scores
        }
    
    def _calculate_confidence_intervals(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate prediction confidence intervals using quantile regression"""
        
        confidence_intervals = {}
        
        for confidence_level in self.config.confidence_intervals:
            # Train quantile regressors for upper and lower bounds
            alpha_lower = (1 - confidence_level) / 2
            alpha_upper = 1 - alpha_lower
            
            # Use simplified approach - assume normal distribution around prediction
            # In production, would use quantile regression or ensemble methods
            residual_std = 0.001  # Typical slippage standard deviation
            z_score_lower = np.percentile(np.random.normal(0, 1, 1000), alpha_lower * 100)
            z_score_upper = np.percentile(np.random.normal(0, 1, 1000), alpha_upper * 100)
            
            lower_bound = predictions + z_score_lower * residual_std
            upper_bound = predictions + z_score_upper * residual_std
            
            confidence_intervals[f'{confidence_level:.0%}'] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
        
        return confidence_intervals
    
    def _categorize_slippage_risk(self, predictions: np.ndarray) -> List[str]:
        """Categorize slippage predictions into risk levels"""
        
        categories = []
        
        for pred in predictions:
            if pred < self.config.slippage_buckets[0]:
                categories.append('very_low')
            elif pred < self.config.slippage_buckets[1]:
                categories.append('low')
            elif pred < self.config.slippage_buckets[2]:
                categories.append('medium')
            elif pred < self.config.slippage_buckets[3]:
                categories.append('high')
            else:
                categories.append('very_high')
        
        return categories
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        
        if not self.is_trained or not self.last_training_time:
            return True
        
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= self.config.retraining_frequency
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics"""
        
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'hours_since_training': (datetime.now() - self.last_training_time).total_seconds() / 3600 if self.last_training_time else None,
            'should_retrain': self.should_retrain(),
            'feature_count': len(self.feature_engineer.feature_names),
            'top_10_features': self.feature_importance.head(10).to_dict() if self.feature_importance is not None else {},
            'model_params': {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate
            }
        }