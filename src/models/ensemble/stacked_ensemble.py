"""
Stacked Ensemble Meta-Learner

Combines predictions from multiple models (LSTM, CNN, XGBoost, FinBERT)
using stacking, voting, or blending strategies.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import structlog
from pathlib import Path
import json
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base import EnsembleModel, EnsembleConfig, ModelType, BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class StackedEnsembleConfig(EnsembleConfig):
    """Configuration for Stacked Ensemble model"""
    
    # Meta-learner configuration
    meta_learner_type: str = "xgboost"  # "logistic", "ridge", "random_forest", "neural_net", "xgboost"
    meta_learner_params: Dict[str, Any] = None
    
    # Stacking configuration
    use_probabilities: bool = True  # Use probability predictions instead of class predictions
    include_original_features: bool = True  # Include original features along with base predictions
    cv_folds: int = 5  # Number of cross-validation folds for stacking
    
    # Model weights (for weighted averaging)
    model_weights: Dict[str, float] = None
    adaptive_weights: bool = True  # Learn weights based on validation performance
    
    # Feature generation
    generate_disagreement_features: bool = True  # Add features based on model disagreement
    generate_confidence_features: bool = True  # Add features based on prediction confidence
    
    # Prediction strategies
    fallback_strategy: str = "majority_vote"  # Strategy when meta-learner fails
    confidence_threshold: float = 0.6  # Minimum confidence for predictions
    
    # Performance tracking
    track_individual_performance: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if self.meta_learner_params is None:
            # Default parameters for different meta-learners
            if self.meta_learner_type == "xgboost":
                self.meta_learner_params = {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                }
            elif self.meta_learner_type == "neural_net":
                self.meta_learner_params = {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'learning_rate': 'adaptive',
                    'max_iter': 1000
                }
            elif self.meta_learner_type == "random_forest":
                self.meta_learner_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                }
        
        if self.model_weights is None:
            # Default equal weights
            self.model_weights = {
                'PriceLSTM': 0.25,
                'ChartPatternCNN': 0.25,
                'MarketRegimeXGBoost': 0.25,
                'FinBERT': 0.25
            }
        
        self.model_type = ModelType.ENSEMBLE
        self.name = "StackedEnsemble"


class StackedEnsemble(EnsembleModel):
    """Stacked ensemble meta-learner combining multiple models"""
    
    def __init__(self, config: StackedEnsembleConfig):
        super().__init__(config)
        self.config: StackedEnsembleConfig = config
        self.meta_learner = None
        self.model_performance = {}
        self.feature_columns = []
        
    def build_meta_learner(self, task_type: str = "classification"):
        """Build the meta-learner model"""
        
        if task_type == "classification":
            if self.config.meta_learner_type == "logistic":
                self.meta_learner = LogisticRegression(**self.config.meta_learner_params)
            elif self.config.meta_learner_type == "random_forest":
                self.meta_learner = RandomForestClassifier(**self.config.meta_learner_params)
            elif self.config.meta_learner_type == "neural_net":
                self.meta_learner = MLPClassifier(**self.config.meta_learner_params)
            elif self.config.meta_learner_type == "xgboost":
                self.meta_learner = xgb.XGBClassifier(**self.config.meta_learner_params)
            else:
                raise ValueError(f"Unknown meta-learner type: {self.config.meta_learner_type}")
        else:  # regression
            if self.config.meta_learner_type == "ridge":
                self.meta_learner = Ridge(**self.config.meta_learner_params)
            elif self.config.meta_learner_type == "random_forest":
                self.meta_learner = RandomForestRegressor(**self.config.meta_learner_params)
            elif self.config.meta_learner_type == "neural_net":
                self.meta_learner = MLPRegressor(**self.config.meta_learner_params)
            elif self.config.meta_learner_type == "xgboost":
                self.meta_learner = xgb.XGBRegressor(**self.config.meta_learner_params)
            else:
                raise ValueError(f"Unknown meta-learner type: {self.config.meta_learner_type}")
        
        logger.info(f"Meta-learner built: {self.config.meta_learner_type}")
        return self.meta_learner
    
    def get_base_predictions(
        self,
        models: Dict[str, BaseModel],
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_proba: bool = True
    ) -> pd.DataFrame:
        """Get predictions from all base models"""
        
        predictions = {}
        probabilities = {}
        
        # Handle different data formats for different models
        if isinstance(data, dict):
            # Data is pre-separated for different models
            model_data = data
        else:
            # Same data for all models
            model_data = {model_name: data for model_name in models.keys()}
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                model_input = model_data.get(model_name, data)
                
                # Get predictions
                if hasattr(model, 'predict'):
                    preds = model.predict(model_input)
                    predictions[f'{model_name}_pred'] = preds
                
                # Get probabilities if available and requested
                if return_proba and hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(model_input)
                    # Store probabilities for each class
                    if len(proba.shape) > 1:
                        for i in range(proba.shape[1]):
                            probabilities[f'{model_name}_proba_class_{i}'] = proba[:, i]
                    else:
                        probabilities[f'{model_name}_proba'] = proba
                
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}", error=str(e))
                # Use fallback predictions
                if isinstance(data, pd.DataFrame):
                    n_samples = len(data)
                else:
                    n_samples = len(next(iter(model_data.values())))
                predictions[f'{model_name}_pred'] = np.zeros(n_samples)
                if return_proba:
                    probabilities[f'{model_name}_proba'] = np.ones(n_samples) * 0.5
        
        # Combine all predictions into a DataFrame
        all_predictions = pd.DataFrame({**predictions, **probabilities})
        
        return all_predictions
    
    def generate_meta_features(
        self,
        base_predictions: pd.DataFrame,
        original_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate meta-features for the meta-learner"""
        
        meta_features = base_predictions.copy()
        
        # Add disagreement features
        if self.config.generate_disagreement_features:
            pred_cols = [col for col in base_predictions.columns if '_pred' in col]
            if len(pred_cols) > 1:
                # Pairwise disagreement
                for i, col1 in enumerate(pred_cols):
                    for col2 in pred_cols[i+1:]:
                        disagreement = (base_predictions[col1] != base_predictions[col2]).astype(int)
                        meta_features[f'disagree_{col1}_{col2}'] = disagreement
                
                # Overall disagreement (standard deviation of predictions)
                if all(base_predictions[col].dtype in [np.float32, np.float64] for col in pred_cols):
                    meta_features['pred_std'] = base_predictions[pred_cols].std(axis=1)
                    meta_features['pred_range'] = (
                        base_predictions[pred_cols].max(axis=1) - 
                        base_predictions[pred_cols].min(axis=1)
                    )
        
        # Add confidence features
        if self.config.generate_confidence_features:
            proba_cols = [col for col in base_predictions.columns if '_proba' in col]
            if proba_cols:
                # Average confidence
                meta_features['avg_confidence'] = base_predictions[proba_cols].mean(axis=1)
                # Confidence variance
                meta_features['confidence_var'] = base_predictions[proba_cols].var(axis=1)
                # Min and max confidence
                meta_features['min_confidence'] = base_predictions[proba_cols].min(axis=1)
                meta_features['max_confidence'] = base_predictions[proba_cols].max(axis=1)
        
        # Add original features if configured
        if self.config.include_original_features and original_features is not None:
            # Select most important original features (to avoid too many features)
            # This is a simplified approach - in practice, you'd use feature selection
            meta_features = pd.concat([meta_features, original_features], axis=1)
        
        return meta_features
    
    def train(
        self,
        train_data: Union[pd.DataFrame, Dict[str, Any]],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the ensemble model"""
        
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        # Determine task type
        if isinstance(train_labels, (pd.Series, np.ndarray)):
            unique_labels = np.unique(train_labels)
            task_type = "classification" if len(unique_labels) < 20 else "regression"
        else:
            task_type = "classification"
        
        # Build meta-learner
        self.build_meta_learner(task_type)
        
        # Get base model predictions
        logger.info("Getting base model predictions for training")
        base_predictions_train = self.get_base_predictions(
            self.base_models,
            train_data,
            return_proba=self.config.use_probabilities
        )
        
        # Generate meta-features
        original_features = train_data if isinstance(train_data, pd.DataFrame) else None
        meta_features_train = self.generate_meta_features(
            base_predictions_train,
            original_features
        )
        
        # Store feature columns
        self.feature_columns = meta_features_train.columns.tolist()
        
        # Prepare validation data if provided
        if validation_data:
            val_data, val_labels = validation_data
            base_predictions_val = self.get_base_predictions(
                self.base_models,
                val_data,
                return_proba=self.config.use_probabilities
            )
            
            original_features_val = val_data if isinstance(val_data, pd.DataFrame) else None
            meta_features_val = self.generate_meta_features(
                base_predictions_val,
                original_features_val
            )
            
            # Ensure same features
            meta_features_val = meta_features_val[self.feature_columns]
        
        # Train meta-learner
        logger.info("Training meta-learner")
        
        # Handle missing values
        meta_features_train = meta_features_train.fillna(0)
        if validation_data:
            meta_features_val = meta_features_val.fillna(0)
        
        # Fit meta-learner
        if hasattr(self.meta_learner, 'fit'):
            if validation_data and hasattr(self.meta_learner, 'set_params'):
                # For XGBoost, use validation set
                if isinstance(self.meta_learner, (xgb.XGBClassifier, xgb.XGBRegressor)):
                    self.meta_learner.fit(
                        meta_features_train, train_labels,
                        eval_set=[(meta_features_val, val_labels)],
                        early_stopping_rounds=20,
                        verbose=False
                    )
                else:
                    self.meta_learner.fit(meta_features_train, train_labels)
            else:
                self.meta_learner.fit(meta_features_train, train_labels)
        
        # Calculate adaptive weights if configured
        if self.config.adaptive_weights and validation_data:
            self.calculate_adaptive_weights(val_data, val_labels)
        
        # Track individual model performance if configured
        if self.config.track_individual_performance and validation_data:
            self.evaluate_individual_models(val_data, val_labels)
        
        # Update metadata
        self.is_trained = True
        self.metadata['last_trained'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(train_labels)
        self.metadata['n_base_models'] = len(self.base_models)
        self.metadata['meta_learner_type'] = self.config.meta_learner_type
        
        history = {
            'meta_learner_type': self.config.meta_learner_type,
            'n_features': len(self.feature_columns),
            'training_samples': len(train_labels)
        }
        
        self.training_history.append(history)
        
        logger.info("Ensemble training completed",
                   meta_learner=self.config.meta_learner_type,
                   n_features=len(self.feature_columns))
        
        return history
    
    def calculate_adaptive_weights(
        self,
        val_data: Union[pd.DataFrame, Dict[str, Any]],
        val_labels: Union[pd.Series, np.ndarray]
    ):
        """Calculate adaptive weights based on validation performance"""
        
        performances = {}
        
        for model_name, model in self.base_models.items():
            try:
                # Get model predictions
                if isinstance(val_data, dict):
                    model_input = val_data.get(model_name, val_data)
                else:
                    model_input = val_data
                
                predictions = model.predict(model_input)
                
                # Calculate performance metric
                if hasattr(model, 'model_type') and 'classification' in str(model.model_type).lower():
                    # Classification metric
                    from sklearn.metrics import f1_score
                    performance = f1_score(val_labels, predictions, average='weighted')
                else:
                    # Regression metric
                    from sklearn.metrics import r2_score
                    performance = max(0, r2_score(val_labels, predictions))
                
                performances[model_name] = performance
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}", error=str(e))
                performances[model_name] = 0.0
        
        # Normalize weights
        total_performance = sum(performances.values())
        if total_performance > 0:
            self.config.model_weights = {
                name: perf / total_performance 
                for name, perf in performances.items()
            }
        
        logger.info("Adaptive weights calculated", weights=self.config.model_weights)
    
    def evaluate_individual_models(
        self,
        val_data: Union[pd.DataFrame, Dict[str, Any]],
        val_labels: Union[pd.Series, np.ndarray]
    ):
        """Evaluate performance of individual models"""
        
        for model_name, model in self.base_models.items():
            try:
                # Get model predictions
                if isinstance(val_data, dict):
                    model_input = val_data.get(model_name, val_data)
                else:
                    model_input = val_data
                
                predictions = model.predict(model_input)
                
                # Calculate metrics
                if hasattr(model, 'evaluate'):
                    metrics = model.evaluate(model_input, val_labels)
                else:
                    # Basic metrics
                    from sklearn.metrics import accuracy_score, mean_squared_error
                    if hasattr(model, 'model_type') and 'classification' in str(model.model_type).lower():
                        metrics = {'accuracy': accuracy_score(val_labels, predictions)}
                    else:
                        metrics = {'mse': mean_squared_error(val_labels, predictions)}
                
                self.model_performance[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}", error=str(e))
                self.model_performance[model_name] = {'error': str(e)}
        
        logger.info("Individual model evaluation completed", 
                   performances=self.model_performance)
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with the ensemble"""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        try:
            # Get base model predictions
            base_predictions = self.get_base_predictions(
                self.base_models,
                data,
                return_proba=self.config.use_probabilities
            )
            
            # Generate meta-features
            original_features = data if isinstance(data, pd.DataFrame) else None
            meta_features = self.generate_meta_features(
                base_predictions,
                original_features
            )
            
            # Ensure same features as training
            meta_features = meta_features[self.feature_columns].fillna(0)
            
            # Make predictions with meta-learner
            predictions = self.meta_learner.predict(meta_features)
            
            return predictions
            
        except Exception as e:
            logger.error("Error in ensemble prediction, using fallback strategy", 
                        error=str(e))
            
            # Fallback to simple voting/averaging
            return self.fallback_predict(data)
    
    def fallback_predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> np.ndarray:
        """Fallback prediction strategy"""
        
        predictions = []
        weights = []
        
        for model_name, model in self.base_models.items():
            try:
                if isinstance(data, dict):
                    model_input = data.get(model_name, data)
                else:
                    model_input = data
                
                pred = model.predict(model_input)
                predictions.append(pred)
                weights.append(self.config.model_weights.get(model_name, 1.0))
                
            except Exception as e:
                logger.error(f"Error in fallback prediction for {model_name}", 
                            error=str(e))
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        if self.config.fallback_strategy == "majority_vote":
            # For classification
            from scipy.stats import mode
            return mode(predictions, axis=0)[0].flatten()
        else:
            # Weighted average
            return np.average(predictions, axis=0, weights=weights)
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        **kwargs
    ) -> np.ndarray:
        """Get prediction probabilities"""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        if not hasattr(self.meta_learner, 'predict_proba'):
            raise ValueError(f"Meta-learner {self.config.meta_learner_type} "
                           "does not support probability predictions")
        
        # Get base model predictions
        base_predictions = self.get_base_predictions(
            self.base_models,
            data,
            return_proba=self.config.use_probabilities
        )
        
        # Generate meta-features
        original_features = data if isinstance(data, pd.DataFrame) else None
        meta_features = self.generate_meta_features(
            base_predictions,
            original_features
        )
        
        # Ensure same features as training
        meta_features = meta_features[self.feature_columns].fillna(0)
        
        # Get probabilities
        probabilities = self.meta_learner.predict_proba(meta_features)
        
        return probabilities
    
    def get_model_contributions(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get contribution of each model to the final prediction"""
        
        # Get base predictions
        base_predictions = self.get_base_predictions(
            self.base_models,
            data,
            return_proba=True
        )
        
        # Get ensemble prediction
        ensemble_pred = self.predict(data)
        
        contributions = {
            'ensemble_prediction': ensemble_pred,
            'base_predictions': {},
            'weights': self.config.model_weights,
            'agreements': {}
        }
        
        # Calculate agreements
        for model_name in self.base_models:
            pred_col = f'{model_name}_pred'
            if pred_col in base_predictions.columns:
                base_pred = base_predictions[pred_col].values
                agreement = np.mean(base_pred == ensemble_pred)
                contributions['agreements'][model_name] = agreement
                contributions['base_predictions'][model_name] = base_pred
        
        return contributions
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save ensemble model"""
        # Save base model data
        base_path = super().save(path)
        
        # Save meta-learner
        if self.meta_learner is not None:
            meta_learner_path = base_path.with_suffix('.meta_learner.pkl')
            joblib.dump(self.meta_learner, meta_learner_path)
            
            # Save additional metadata
            metadata_path = base_path.with_suffix('.ensemble_meta.json')
            metadata = {
                'config': self.config.to_dict(),
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'base_model_names': list(self.base_models.keys())
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Ensemble model saved", path=str(base_path))
        
        return base_path
    
    @classmethod
    def load(cls, path: Path) -> 'StackedEnsemble':
        """Load ensemble model"""
        # Load base model data
        model_instance = super().load(path)
        
        # Load meta-learner
        meta_learner_path = path.with_suffix('.meta_learner.pkl')
        if meta_learner_path.exists():
            model_instance.meta_learner = joblib.load(meta_learner_path)
            
            # Load metadata
            metadata_path = path.with_suffix('.ensemble_meta.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_instance.feature_columns = metadata.get('feature_columns', [])
                model_instance.model_performance = metadata.get('model_performance', {})
            
            logger.info("Ensemble model loaded", path=str(path))
        
        return model_instance