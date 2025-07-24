"""
Walk-Forward Optimization

Implements time-series cross-validation for financial models.
Prevents look-ahead bias by using only historical data for training.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import structlog
from pathlib import Path
import json
from copy import deepcopy

from ..models.base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization"""
    
    # Time windows
    training_window_months: int = 24  # Training window size in months
    validation_window_months: int = 3  # Validation window size
    test_window_months: int = 1  # Out-of-sample test window
    step_size_months: int = 1  # Step size for rolling forward
    
    # Optimization parameters
    min_training_samples: int = 1000  # Minimum samples for training
    max_training_samples: int = 100000  # Maximum samples to prevent memory issues
    
    # Performance tracking
    track_performance: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio', 'max_drawdown', 'total_return', 'volatility',
        'hit_rate', 'profit_factor', 'calmar_ratio'
    ])
    
    # Model management
    save_intermediate_models: bool = False
    model_save_dir: Optional[Path] = None
    
    # Parallelization
    parallel_folds: bool = True
    max_workers: int = 4
    
    # Early stopping for optimization
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 5
    early_stopping_metric: str = 'sharpe_ratio'
    early_stopping_threshold: float = 0.1  # Minimum improvement threshold
    
    def __post_init__(self):
        if self.model_save_dir:
            self.model_save_dir = Path(self.model_save_dir)
            self.model_save_dir.mkdir(exist_ok=True, parents=True)


class TimeSeriesFold:
    """Represents a single fold in walk-forward optimization"""
    
    def __init__(
        self,
        fold_id: int,
        train_start: datetime,
        train_end: datetime,
        val_start: datetime,
        val_end: datetime,
        test_start: datetime,
        test_end: datetime
    ):
        self.fold_id = fold_id
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        self.results: Dict[str, Any] = {}
    
    def __str__(self) -> str:
        return (f"Fold {self.fold_id}: "
                f"Train({self.train_start.date()}-{self.train_end.date()}) "
                f"Val({self.val_start.date()}-{self.val_end.date()}) "
                f"Test({self.test_start.date()}-{self.test_end.date()})")


class WalkForwardOptimizer:
    """Walk-forward optimization for time series models"""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.folds: List[TimeSeriesFold] = []
        self.results: Dict[str, Any] = {}
        
    def create_folds(self, data: pd.DataFrame, timestamp_col: str = 'timestamp') -> List[TimeSeriesFold]:
        """Create time-series folds for walk-forward optimization"""
        
        # Ensure data is sorted by timestamp
        data = data.sort_values(timestamp_col)
        
        start_date = data[timestamp_col].min()
        end_date = data[timestamp_col].max()
        
        logger.info("Creating walk-forward folds",
                   start_date=start_date,
                   end_date=end_date,
                   total_samples=len(data))
        
        folds = []
        fold_id = 0
        
        # Calculate initial training period end
        current_date = start_date + timedelta(days=self.config.training_window_months * 30)
        
        while current_date < end_date:
            # Calculate fold dates
            train_start = start_date
            train_end = current_date
            
            val_start = train_end
            val_end = val_start + timedelta(days=self.config.validation_window_months * 30)
            
            test_start = val_end
            test_end = test_start + timedelta(days=self.config.test_window_months * 30)
            
            # Check if we have enough data for this fold
            if test_end > end_date:
                break
            
            # Create fold
            fold = TimeSeriesFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end
            )
            
            # Extract data for this fold
            fold.train_data = data[
                (data[timestamp_col] >= train_start) & 
                (data[timestamp_col] < train_end)
            ].copy()
            
            fold.val_data = data[
                (data[timestamp_col] >= val_start) & 
                (data[timestamp_col] < val_end)
            ].copy()
            
            fold.test_data = data[
                (data[timestamp_col] >= test_start) & 
                (data[timestamp_col] < test_end)
            ].copy()
            
            # Check minimum sample requirements
            if (len(fold.train_data) >= self.config.min_training_samples and
                len(fold.val_data) > 0 and len(fold.test_data) > 0):
                folds.append(fold)
                fold_id += 1
            
            # Move to next fold
            current_date += timedelta(days=self.config.step_size_months * 30)
        
        self.folds = folds
        
        logger.info("Walk-forward folds created",
                   total_folds=len(folds),
                   avg_train_samples=np.mean([len(f.train_data) for f in folds]),
                   avg_val_samples=np.mean([len(f.val_data) for f in folds]),
                   avg_test_samples=np.mean([len(f.test_data) for f in folds]))
        
        return folds
    
    def optimize_model(
        self,
        model_class: type,
        model_configs: List[Dict[str, Any]],
        data: pd.DataFrame,
        target_column: str,
        timestamp_col: str = 'timestamp'
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize model using walk-forward validation"""
        
        logger.info("Starting walk-forward optimization",
                   model_class=model_class.__name__,
                   n_configs=len(model_configs))
        
        # Create folds if not already created
        if not self.folds:
            self.create_folds(data, timestamp_col)
        
        best_config = None
        best_score = float('-inf')
        config_results = {}
        
        # Test each configuration
        for config_idx, config_dict in enumerate(model_configs):
            logger.info(f"Testing configuration {config_idx + 1}/{len(model_configs)}")
            
            # Run walk-forward validation for this configuration
            config_score, fold_results = self._evaluate_config(
                model_class, config_dict, target_column
            )
            
            config_results[f"config_{config_idx}"] = {
                'config': config_dict,
                'score': config_score,
                'fold_results': fold_results
            }
            
            # Update best configuration
            if config_score > best_score:
                best_score = config_score
                best_config = config_dict
                
                logger.info("New best configuration found",
                           config_idx=config_idx,
                           score=config_score)
            
            # Early stopping check
            if self.config.enable_early_stopping:
                if self._should_early_stop(config_results, config_idx):
                    logger.info("Early stopping triggered", config_idx=config_idx)
                    break
        
        # Store results
        self.results = {
            'best_config': best_config,
            'best_score': best_score,
            'all_configs': config_results,
            'optimization_summary': self._create_optimization_summary(config_results)
        }
        
        logger.info("Walk-forward optimization completed",
                   best_score=best_score,
                   configs_tested=len(config_results))
        
        return best_config, self.results
    
    def _evaluate_config(
        self,
        model_class: type,
        config_dict: Dict[str, Any],
        target_column: str
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Evaluate a single configuration across all folds"""
        
        fold_results = []
        fold_scores = []
        
        if self.config.parallel_folds and len(self.folds) > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_fold = {}
                
                for fold in self.folds:
                    future = executor.submit(
                        self._evaluate_fold,
                        model_class, config_dict, fold, target_column
                    )
                    future_to_fold[future] = fold
                
                for future in as_completed(future_to_fold):
                    fold = future_to_fold[future]
                    try:
                        result = future.result()
                        fold_results.append(result)
                        fold_scores.append(result.get('score', 0))
                    except Exception as e:
                        logger.error(f"Fold {fold.fold_id} evaluation failed", error=str(e))
                        fold_results.append({'fold_id': fold.fold_id, 'error': str(e)})
                        fold_scores.append(0)
        else:
            # Sequential evaluation
            for fold in self.folds:
                try:
                    result = self._evaluate_fold(model_class, config_dict, fold, target_column)
                    fold_results.append(result)
                    fold_scores.append(result.get('score', 0))
                except Exception as e:
                    logger.error(f"Fold {fold.fold_id} evaluation failed", error=str(e))
                    fold_results.append({'fold_id': fold.fold_id, 'error': str(e)})
                    fold_scores.append(0)
        
        # Calculate aggregate score
        avg_score = np.mean(fold_scores) if fold_scores else 0
        
        return avg_score, fold_results
    
    def _evaluate_fold(
        self,
        model_class: type,
        config_dict: Dict[str, Any],
        fold: TimeSeriesFold,
        target_column: str
    ) -> Dict[str, Any]:
        """Evaluate a single fold"""
        
        try:
            # Create model instance
            config_obj = model_class.__name__.replace('Model', 'Config')
            # This is simplified - in practice, you'd need proper config class instantiation
            model = model_class(config_dict)
            
            # Prepare training data
            train_features = fold.train_data.drop(columns=[target_column, 'timestamp'])
            train_targets = fold.train_data[target_column]
            
            # Prepare validation data
            val_features = fold.val_data.drop(columns=[target_column, 'timestamp'])
            val_targets = fold.val_data[target_column]
            
            # Train model
            model.train(
                train_features, train_targets,
                validation_data=(val_features, val_targets)
            )
            
            # Evaluate on test data
            test_features = fold.test_data.drop(columns=[target_column, 'timestamp'])
            test_targets = fold.test_data[target_column]
            
            predictions = model.predict(test_features)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                test_targets, predictions, fold.test_data['timestamp']
            )
            
            # Save model if configured
            if self.config.save_intermediate_models and self.config.model_save_dir:
                model_path = self.config.model_save_dir / f"fold_{fold.fold_id}_model"
                model.save(model_path)
            
            return {
                'fold_id': fold.fold_id,
                'metrics': metrics,
                'score': metrics.get(self.config.early_stopping_metric, 0),
                'n_train_samples': len(fold.train_data),
                'n_test_samples': len(fold.test_data)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating fold {fold.fold_id}", error=str(e))
            return {
                'fold_id': fold.fold_id,
                'error': str(e),
                'score': 0
            }
    
    def _calculate_performance_metrics(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
        timestamps: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Convert to returns if needed
        if len(actual) > 1:
            actual_returns = actual.pct_change().dropna()
            pred_returns = pd.Series(predicted).pct_change().dropna()
            
            # Align series
            min_len = min(len(actual_returns), len(pred_returns))
            actual_returns = actual_returns.iloc[:min_len]
            pred_returns = pred_returns.iloc[:min_len]
        else:
            actual_returns = pd.Series([0])
            pred_returns = pd.Series([0])
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['total_return'] = pred_returns.sum()
            metrics['volatility'] = pred_returns.std() * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (pred_returns.mean() * 252) / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
            
            # Maximum drawdown
            cumulative = (1 + pred_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Hit rate (percentage of correct direction predictions)
            if len(actual_returns) > 0:
                correct_direction = (
                    (actual_returns > 0) == (pred_returns > 0)
                ).sum()
                metrics['hit_rate'] = correct_direction / len(actual_returns)
            else:
                metrics['hit_rate'] = 0.5
            
            # Profit factor
            gains = pred_returns[pred_returns > 0].sum()
            losses = abs(pred_returns[pred_returns < 0].sum())
            metrics['profit_factor'] = gains / losses if losses > 0 else float('inf')
            
            # Calmar ratio
            if metrics['max_drawdown'] < 0:
                metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = metrics['total_return']
                
        except Exception as e:
            logger.error("Error calculating metrics", error=str(e))
            # Return default metrics
            metrics = {metric: 0.0 for metric in self.config.performance_metrics}
        
        return metrics
    
    def _should_early_stop(self, config_results: Dict, current_idx: int) -> bool:
        """Check if early stopping should be triggered"""
        
        if current_idx < self.config.early_stopping_rounds:
            return False
        
        # Get recent scores
        recent_scores = []
        for i in range(max(0, current_idx - self.config.early_stopping_rounds + 1), current_idx + 1):
            config_key = f"config_{i}"
            if config_key in config_results:
                recent_scores.append(config_results[config_key]['score'])
        
        if len(recent_scores) < self.config.early_stopping_rounds:
            return False
        
        # Check if improvement is below threshold
        max_recent = max(recent_scores)
        min_recent = min(recent_scores)
        improvement = (max_recent - min_recent) / (abs(min_recent) + 1e-6)
        
        return improvement < self.config.early_stopping_threshold
    
    def _create_optimization_summary(self, config_results: Dict) -> Dict[str, Any]:
        """Create summary of optimization results"""
        
        scores = [result['score'] for result in config_results.values() if 'score' in result]
        
        summary = {
            'n_configs_tested': len(config_results),
            'best_score': max(scores) if scores else 0,
            'worst_score': min(scores) if scores else 0,
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'n_folds': len(self.folds),
            'total_evaluations': len(config_results) * len(self.folds)
        }
        
        return summary
    
    def get_best_model_config(self) -> Optional[Dict[str, Any]]:
        """Get the best model configuration from optimization"""
        return self.results.get('best_config')
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get full optimization results"""
        return self.results
    
    def save_results(self, path: Path):
        """Save optimization results to file"""
        
        with open(path, 'w') as f:
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Optimization results saved", path=str(path))
    
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
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj