"""
Ensemble Model Base Class

Base class for ensemble models that combine multiple models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
import structlog

from .base_model import BaseModel, ModelConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class EnsembleConfig(ModelConfig):
    """Configuration specific to ensemble models"""
    
    # Ensemble specific parameters
    ensemble_method: str = "voting"  # "voting", "stacking", "blending", "averaging"
    voting_type: str = "soft"  # "hard" or "soft" for voting
    
    # Model weights
    model_weights: Optional[Dict[str, float]] = None
    optimize_weights: bool = True
    
    # Stacking parameters
    meta_model_type: str = "logistic"  # Type of meta-learner
    use_proba_features: bool = True  # Use probabilities as features
    blend_features: bool = True  # Include original features in stacking
    
    # Model selection
    model_selection_metric: str = "f1_score"
    min_model_performance: float = 0.5  # Minimum performance to include model
    
    # Diversity metrics
    measure_diversity: bool = True
    min_correlation: float = 0.3  # Minimum correlation between models
    max_correlation: float = 0.9  # Maximum correlation between models
    
    def __post_init__(self):
        super().__post_init__()
        self.model_type = ModelType.ENSEMBLE


class EnsembleModel(BaseModel):
    """Base class for ensemble models"""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.config: EnsembleConfig = config
        self.base_models = {}
        self.model_weights = {}
        self.meta_model = None
        self.model_performance = {}
        self.diversity_metrics = {}
        
    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a base model to the ensemble"""
        self.base_models[name] = model
        
        # Initialize weight if not provided
        if self.config.model_weights and name in self.config.model_weights:
            self.model_weights[name] = self.config.model_weights[name]
        else:
            # Equal weights by default
            self.model_weights[name] = 1.0 / max(len(self.base_models), 1)
        
        # Rebalance weights
        self._normalize_weights()
        
        logger.info("Model added to ensemble",
                   name=name,
                   model_type=type(model).__name__,
                   weight=self.model_weights[name])
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble"""
        if name in self.base_models:
            del self.base_models[name]
            del self.model_weights[name]
            self._normalize_weights()
            logger.info("Model removed from ensemble", name=name)
    
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1"""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight
    
    def calculate_diversity(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity metrics between model predictions"""
        diversity_metrics = {}
        
        # Convert predictions to correlation matrix
        pred_matrix = np.column_stack(list(predictions.values()))
        correlation_matrix = np.corrcoef(pred_matrix.T)
        
        # Average pairwise correlation
        n_models = len(predictions)
        if n_models > 1:
            # Extract upper triangle of correlation matrix
            upper_triangle = correlation_matrix[np.triu_indices(n_models, k=1)]
            diversity_metrics['avg_correlation'] = np.mean(upper_triangle)
            diversity_metrics['min_correlation'] = np.min(upper_triangle)
            diversity_metrics['max_correlation'] = np.max(upper_triangle)
            
            # Disagreement measure
            disagreement_matrix = []
            model_names = list(predictions.keys())
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pred_i = predictions[model_names[i]]
                    pred_j = predictions[model_names[j]]
                    disagreement = np.mean(pred_i != pred_j)
                    disagreement_matrix.append(disagreement)
            
            diversity_metrics['avg_disagreement'] = np.mean(disagreement_matrix)
            
            # Q-statistic (Yule's Q)
            # Q = (ad - bc) / (ad + bc) where a,b,c,d are agreement/disagreement counts
            q_statistics = []
            for i in range(n_models):
                for j in range(i+1, n_models):
                    pred_i = predictions[model_names[i]]
                    pred_j = predictions[model_names[j]]
                    
                    # For binary predictions
                    if len(np.unique(pred_i)) == 2 and len(np.unique(pred_j)) == 2:
                        a = np.sum((pred_i == 1) & (pred_j == 1))  # Both correct
                        b = np.sum((pred_i == 1) & (pred_j == 0))  # i correct, j wrong
                        c = np.sum((pred_i == 0) & (pred_j == 1))  # i wrong, j correct
                        d = np.sum((pred_i == 0) & (pred_j == 0))  # Both wrong
                        
                        q = (a*d - b*c) / (a*d + b*c + 1e-10)
                        q_statistics.append(q)
            
            if q_statistics:
                diversity_metrics['avg_q_statistic'] = np.mean(q_statistics)
        
        return diversity_metrics
    
    def optimize_weights(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        metric: str = "accuracy"
    ) -> Dict[str, float]:
        """Optimize ensemble weights using validation data"""
        from scipy.optimize import minimize
        
        n_models = len(self.base_models)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X_val)
            else:
                predictions[name] = model.predict(X_val)
        
        # Objective function to minimize
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Weighted average of predictions
            weighted_pred = np.zeros_like(list(predictions.values())[0])
            for i, (name, pred) in enumerate(predictions.items()):
                weighted_pred += weights[i] * pred
            
            # Convert to class predictions
            if len(weighted_pred.shape) > 1:
                y_pred = np.argmax(weighted_pred, axis=1)
            else:
                y_pred = (weighted_pred > 0.5).astype(int)
            
            # Calculate metric (negative because we minimize)
            if metric == "accuracy":
                from sklearn.metrics import accuracy_score
                return -accuracy_score(y_val, y_pred)
            elif metric == "f1_score":
                from sklearn.metrics import f1_score
                return -f1_score(y_val, y_pred, average='weighted')
            else:
                return -accuracy_score(y_val, y_pred)
        
        # Initial weights (equal)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Update weights
        optimized_weights = result.x / np.sum(result.x)
        for i, name in enumerate(self.base_models.keys()):
            self.model_weights[name] = optimized_weights[i]
        
        logger.info("Weights optimized",
                   metric=metric,
                   final_score=-result.fun,
                   weights=self.model_weights)
        
        return self.model_weights
    
    def build_model(self) -> Any:
        """Build the ensemble model"""
        if self.config.ensemble_method == "stacking":
            # Build meta-learner for stacking
            if self.config.meta_model_type == "logistic":
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(random_state=42)
            elif self.config.meta_model_type == "xgboost":
                import xgboost as xgb
                self.meta_model = xgb.XGBClassifier(random_state=42)
            elif self.config.meta_model_type == "neural":
                from sklearn.neural_network import MLPClassifier
                self.meta_model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=42
                )
            else:
                from sklearn.linear_model import LinearRegression
                self.meta_model = LinearRegression()
        
        return self
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[Any, Optional[Any]]:
        """Prepare data for ensemble - each model prepares its own data"""
        # For ensemble, we return the raw data
        # Each base model will prepare it according to its needs
        return data, labels
    
    def train(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the ensemble"""
        training_history = {
            'base_models': {},
            'ensemble_metrics': {}
        }
        
        # Train each base model
        for name, model in self.base_models.items():
            logger.info(f"Training base model: {name}")
            
            history = model.train(
                train_data, 
                train_labels,
                validation_data=validation_data,
                **kwargs
            )
            
            training_history['base_models'][name] = history
            
            # Evaluate model performance
            if validation_data:
                val_X, val_y = validation_data
                metrics = model.evaluate(val_X, val_y)
                self.model_performance[name] = metrics
        
        # Train meta-learner if stacking
        if self.config.ensemble_method == "stacking":
            logger.info("Training meta-learner")
            
            # Get out-of-fold predictions for training meta-learner
            meta_features = []
            
            for name, model in self.base_models.items():
                # Use cross-validation to get out-of-fold predictions
                if hasattr(model.model, 'predict_proba'):
                    oof_preds = cross_val_predict(
                        model.model,
                        train_data,
                        train_labels,
                        cv=3,
                        method='predict_proba'
                    )
                else:
                    oof_preds = cross_val_predict(
                        model.model,
                        train_data,
                        train_labels,
                        cv=3
                    )
                
                if len(oof_preds.shape) == 1:
                    oof_preds = oof_preds.reshape(-1, 1)
                
                meta_features.append(oof_preds)
            
            # Stack predictions
            meta_X = np.hstack(meta_features)
            
            # Optionally blend with original features
            if self.config.blend_features:
                if isinstance(train_data, pd.DataFrame):
                    original_features = train_data.values
                else:
                    original_features = train_data
                meta_X = np.hstack([meta_X, original_features])
            
            # Train meta-learner
            self.meta_model.fit(meta_X, train_labels)
        
        # Calculate ensemble diversity if requested
        if self.config.measure_diversity and validation_data:
            val_X, val_y = validation_data
            predictions = {}
            
            for name, model in self.base_models.items():
                predictions[name] = model.predict(val_X)
            
            self.diversity_metrics = self.calculate_diversity(predictions)
            training_history['ensemble_metrics']['diversity'] = self.diversity_metrics
        
        # Optimize weights if requested
        if self.config.optimize_weights and validation_data:
            val_X, val_y = validation_data
            self.optimize_weights(val_X, val_y, metric=self.config.model_selection_metric)
            training_history['ensemble_metrics']['optimized_weights'] = self.model_weights
        
        self.is_trained = True
        self.training_history.append(training_history)
        
        return training_history
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Make ensemble predictions"""
        
        if self.config.ensemble_method == "voting":
            return self._predict_voting(data, **kwargs)
        elif self.config.ensemble_method == "averaging":
            return self._predict_averaging(data, **kwargs)
        elif self.config.ensemble_method == "stacking":
            return self._predict_stacking(data, **kwargs)
        else:
            return self._predict_voting(data, **kwargs)
    
    def _predict_voting(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Voting ensemble prediction"""
        predictions = []
        
        for name, model in self.base_models.items():
            if self.config.voting_type == "soft" and hasattr(model, 'predict_proba'):
                # Soft voting - use probabilities
                pred = model.predict_proba(data)
                predictions.append(pred * self.model_weights[name])
            else:
                # Hard voting - use class predictions
                pred = model.predict(data)
                predictions.append(pred)
        
        if self.config.voting_type == "soft":
            # Sum weighted probabilities
            ensemble_proba = np.sum(predictions, axis=0)
            # Convert to class predictions
            if len(ensemble_proba.shape) > 1:
                ensemble_pred = np.argmax(ensemble_proba, axis=1)
            else:
                ensemble_pred = (ensemble_proba > 0.5).astype(int)
        else:
            # Mode of predictions (majority vote)
            predictions = np.array(predictions)
            from scipy import stats
            ensemble_pred = stats.mode(predictions, axis=0)[0].squeeze()
        
        return ensemble_pred
    
    def _predict_averaging(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Averaging ensemble prediction (for regression)"""
        predictions = []
        
        for name, model in self.base_models.items():
            pred = model.predict(data)
            predictions.append(pred * self.model_weights[name])
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def _predict_stacking(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Stacking ensemble prediction"""
        meta_features = []
        
        # Get predictions from base models
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(data)
            else:
                pred = model.predict(data)
            
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
            
            meta_features.append(pred)
        
        # Stack predictions
        meta_X = np.hstack(meta_features)
        
        # Optionally blend with original features
        if self.config.blend_features:
            if isinstance(data, pd.DataFrame):
                original_features = data.values
            else:
                original_features = data
            meta_X = np.hstack([meta_X, original_features])
        
        # Predict with meta-learner
        ensemble_pred = self.meta_model.predict(meta_X)
        
        return ensemble_pred
    
    def evaluate(
        self,
        test_data: Union[pd.DataFrame, np.ndarray],
        test_labels: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        # Get ensemble predictions
        y_pred = self.predict(test_data)
        
        # Prepare labels
        if isinstance(test_labels, (pd.Series, pd.DataFrame)):
            y_true = test_labels.values
        else:
            y_true = test_labels
        
        # Calculate metrics based on problem type
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        
        # Try classification metrics first
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
        except:
            # Fall back to regression metrics
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
        
        # Add individual model performance
        metrics['individual_performance'] = {}
        for name, model in self.base_models.items():
            individual_metrics = model.evaluate(test_data, test_labels)
            metrics['individual_performance'][name] = individual_metrics
        
        # Add diversity metrics
        if self.config.measure_diversity:
            predictions = {}
            for name, model in self.base_models.items():
                predictions[name] = model.predict(test_data)
            
            diversity = self.calculate_diversity(predictions)
            metrics['diversity'] = diversity
        
        # Store in metadata
        self.metadata['validation_metrics'] = metrics
        
        logger.info("Ensemble evaluated",
                   primary_metric=list(metrics.items())[0])
        
        return metrics