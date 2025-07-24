"""
Classification Model Base Class

Base class for classification models like XGBoost, Random Forest, etc.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import structlog

from .base_model import BaseModel, ModelConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class ClassificationConfig(ModelConfig):
    """Configuration specific to classification models"""
    
    # Classification specific parameters
    n_classes: int = 3  # For market regime: bull, bear, sideways
    class_names: List[str] = None
    
    # Class balancing
    balance_classes: bool = True
    class_weight: Optional[Dict[int, float]] = None
    
    # Evaluation
    eval_metric: str = "accuracy"  # "accuracy", "f1", "roc_auc"
    average_method: str = "weighted"  # For multi-class metrics
    
    # Cross-validation
    cv_folds: int = 5
    stratified: bool = True
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"
    
    # Feature selection
    feature_selection_method: Optional[str] = None  # "mutual_info", "chi2", "anova"
    n_features_to_select: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.n_classes)]
        self.model_type = ModelType.REGIME_CLASSIFICATION


class ClassificationModel(BaseModel):
    """Base class for classification models"""
    
    def __init__(self, config: ClassificationConfig):
        super().__init__(config)
        self.config: ClassificationConfig = config
        self.label_encoder = LabelEncoder()
        self.class_thresholds = {}
        self.feature_selector = None
        
    def encode_labels(self, labels: Union[pd.Series, np.ndarray], fit: bool = True) -> np.ndarray:
        """Encode string labels to integers"""
        if fit:
            return self.label_encoder.fit_transform(labels)
        else:
            return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def balance_dataset(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using various strategies"""
        if not self.config.balance_classes:
            return X, y
        
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
        
        # Get class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Class distribution before balancing",
                   classes=dict(zip(unique, counts)))
        
        # Choose balancing strategy based on imbalance ratio
        min_samples = counts.min()
        max_samples = counts.max()
        imbalance_ratio = max_samples / min_samples
        
        if imbalance_ratio > 10:
            # Severe imbalance - use SMOTEENN
            sampler = SMOTEENN(random_state=42)
        elif imbalance_ratio > 3:
            # Moderate imbalance - use SMOTE
            sampler = SMOTE(random_state=42)
        else:
            # Mild imbalance - use undersampling
            sampler = RandomUnderSampler(random_state=42)
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        unique, counts = np.unique(y_balanced, return_counts=True)
        logger.info("Class distribution after balancing",
                   classes=dict(zip(unique, counts)))
        
        return X_balanced, y_balanced
    
    def select_features(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """Select most important features"""
        if self.config.feature_selection_method is None:
            return X
        
        from sklearn.feature_selection import (
            SelectKBest, chi2, f_classif, mutual_info_classif
        )
        
        if self.config.feature_selection_method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=self.config.n_features_to_select)
        elif self.config.feature_selection_method == "chi2":
            selector = SelectKBest(chi2, k=self.config.n_features_to_select)
        elif self.config.feature_selection_method == "anova":
            selector = SelectKBest(f_classif, k=self.config.n_features_to_select)
        else:
            return X
        
        if fit:
            self.feature_selector = selector
            X_selected = selector.fit_transform(X, y)
            
            # Store selected feature indices
            selected_features = selector.get_support(indices=True)
            logger.info("Features selected",
                       n_selected=len(selected_features),
                       total_features=X.shape[1])
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def optimize_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[int, float]:
        """Optimize classification threshold for each class"""
        from sklearn.metrics import precision_recall_curve
        
        thresholds = {}
        
        if len(y_proba.shape) == 1:
            # Binary classification
            precision, recall, thresh = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_threshold = thresh[np.argmax(f1_scores)]
            thresholds[1] = best_threshold
        else:
            # Multi-class - optimize per class
            for i in range(self.config.n_classes):
                if i in y_true:
                    y_binary = (y_true == i).astype(int)
                    precision, recall, thresh = precision_recall_curve(y_binary, y_proba[:, i])
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                    if len(f1_scores) > 0:
                        best_threshold = thresh[np.argmax(f1_scores)]
                        thresholds[i] = best_threshold
                    else:
                        thresholds[i] = 0.5
                else:
                    thresholds[i] = 0.5
        
        return thresholds
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare classification data"""
        
        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            X = data.select_dtypes(include=[np.number]).values
        else:
            X = data
        
        # Handle labels
        y = None
        if labels is not None:
            if isinstance(labels, (pd.Series, pd.DataFrame)):
                y = labels.values
            else:
                y = labels
            
            # Encode string labels
            if y.dtype == np.object:
                y = self.encode_labels(y, fit=is_training)
        
        # Feature selection
        if is_training and self.config.feature_selection_method:
            X = self.select_features(X, y, fit=True)
        elif not is_training and self.feature_selector:
            X = self.select_features(X, y, fit=False)
        
        # Balance dataset if training
        if is_training and y is not None and self.config.balance_classes:
            X, y = self.balance_dataset(X, y)
        
        return X, y
    
    def evaluate(
        self,
        test_data: Union[pd.DataFrame, np.ndarray],
        test_labels: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate classification model"""
        
        # Get predictions
        y_pred = self.predict(test_data)
        
        # Prepare true labels
        _, y_true = self.prepare_data(test_data, test_labels, is_training=False)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.config.average_method),
            'recall': recall_score(y_true, y_pred, average=self.config.average_method),
            'f1_score': f1_score(y_true, y_pred, average=self.config.average_method)
        }
        
        # Add class-specific metrics
        if self.config.n_classes > 2:
            for i, class_name in enumerate(self.config.class_names):
                y_binary = (y_true == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)
                
                metrics[f'{class_name}_precision'] = precision_score(y_binary, y_pred_binary)
                metrics[f'{class_name}_recall'] = recall_score(y_binary, y_pred_binary)
                metrics[f'{class_name}_f1'] = f1_score(y_binary, y_pred_binary)
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Get classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.config.class_names,
            output_dict=True
        )
        metrics['classification_report'] = report
        
        # Store in metadata
        self.metadata['validation_metrics'] = metrics
        
        logger.info("Model evaluated",
                   accuracy=metrics['accuracy'],
                   f1_score=metrics['f1_score'])
        
        return metrics
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        if self.config.stratified:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train on fold
            self.train(X_train, y_train)
            
            # Evaluate on fold
            metrics = self.evaluate(X_val, y_val)
            
            cv_scores['accuracy'].append(metrics['accuracy'])
            cv_scores['precision'].append(metrics['precision'])
            cv_scores['recall'].append(metrics['recall'])
            cv_scores['f1_score'].append(metrics['f1_score'])
            
            logger.debug(f"Fold {fold+1} completed",
                        accuracy=metrics['accuracy'],
                        f1_score=metrics['f1_score'])
        
        # Calculate mean and std
        cv_summary = {}
        for metric, scores in cv_scores.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
            cv_summary[f'{metric}_scores'] = scores
        
        logger.info("Cross-validation completed",
                   accuracy_mean=cv_summary['accuracy_mean'],
                   accuracy_std=cv_summary['accuracy_std'])
        
        return cv_summary