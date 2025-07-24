"""
Model Validation Framework

Comprehensive validation system for trading models including:
- Statistical significance testing
- Robustness analysis
- Performance attribution
- Risk analysis
- Model comparison
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import structlog
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

from ..models.base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation"""
    
    # Statistical tests
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    enable_statistical_tests: bool = True
    
    # Performance analysis
    benchmark_return: float = 0.0  # Risk-free rate or benchmark return
    confidence_intervals: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Risk analysis
    var_confidence: float = 0.05  # Value at Risk confidence level
    cvar_confidence: float = 0.05  # Conditional VaR confidence level
    max_leverage: float = 1.0
    
    # Robustness tests
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    missing_data_ratios: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    
    # Visualization
    create_plots: bool = True
    plot_save_dir: Optional[Path] = None
    
    # Model comparison
    enable_model_comparison: bool = True
    comparison_metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'hit_rate',
        'profit_factor', 'volatility', 'skewness', 'kurtosis'
    ])
    
    def __post_init__(self):
        if self.plot_save_dir:
            self.plot_save_dir = Path(self.plot_save_dir)
            self.plot_save_dir.mkdir(exist_ok=True, parents=True)


class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_returns_metrics(returns: pd.Series, benchmark: float = 0.0) -> Dict[str, float]:
        """Calculate return-based performance metrics"""
        
        if len(returns) == 0:
            return {}
        
        # Annualization factor (assuming daily returns)
        ann_factor = 252
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** ann_factor - 1
        metrics['volatility'] = returns.std() * np.sqrt(ann_factor)
        
        # Risk-adjusted metrics
        excess_returns = returns - benchmark / ann_factor
        metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        metrics['drawdown_duration'] = PerformanceMetrics._calculate_max_drawdown_duration(drawdown)
        
        # Calmar ratio
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Higher moments
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        
        # Tail risk
        metrics['var_5'] = np.percentile(returns.dropna(), 5)
        metrics['cvar_5'] = returns[returns <= metrics['var_5']].mean()
        
        # Win/loss analysis
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        metrics['hit_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Profit factor
        total_gains = positive_returns.sum()
        total_losses = abs(negative_returns.sum())
        metrics['profit_factor'] = total_gains / total_losses if total_losses > 0 else float('inf')
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods"""
        
        in_drawdown = drawdown < 0
        duration = 0
        max_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        
        return max_duration
    
    @staticmethod
    def calculate_trading_metrics(predictions: np.ndarray, actual: np.ndarray, prices: pd.Series) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        
        if len(predictions) == 0 or len(actual) == 0:
            return {}
        
        metrics = {}
        
        # Classification metrics (if applicable)
        if len(np.unique(predictions)) <= 10:  # Assume classification
            metrics['accuracy'] = np.mean(predictions == actual)
            
            # Confusion matrix metrics
            cm = confusion_matrix(actual, predictions)
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Information coefficient (correlation between predictions and returns)
        if len(prices) > 1:
            returns = prices.pct_change().dropna()
            if len(predictions) == len(returns):
                metrics['information_coefficient'] = np.corrcoef(predictions, returns)[0, 1]
        
        return metrics


class StatisticalTests:
    """Statistical significance tests for model validation"""
    
    @staticmethod
    def sharpe_ratio_test(returns: pd.Series, benchmark_sharpe: float = 0, confidence: float = 0.95) -> Dict[str, Any]:
        """Test statistical significance of Sharpe ratio"""
        
        if len(returns) < 30:  # Minimum sample size
            return {'significant': False, 'reason': 'Insufficient sample size'}
        
        # Calculate sample Sharpe ratio
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        # Jobson-Korkie test statistic
        n = len(returns)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
        
        t_stat = (sharpe - benchmark_sharpe) / se_sharpe
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        return {
            'sharpe_ratio': sharpe,
            'standard_error': se_sharpe,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < (1 - confidence),
            'confidence_interval': [
                sharpe - stats.t.ppf(1 - (1 - confidence) / 2, n - 1) * se_sharpe,
                sharpe + stats.t.ppf(1 - (1 - confidence) / 2, n - 1) * se_sharpe
            ]
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray, 
        statistic_func: Callable, 
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Calculate bootstrap confidence interval for any statistic"""
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return {
            'statistic': statistic_func(data),
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_interval': [lower, upper],
            'confidence_level': confidence
        }
    
    @staticmethod
    def model_comparison_test(returns1: pd.Series, returns2: pd.Series) -> Dict[str, Any]:
        """Statistical test for comparing two models"""
        
        # Diebold-Mariano test for predictive accuracy
        diff = returns1 - returns2
        
        if len(diff) < 30:
            return {'significant': False, 'reason': 'Insufficient sample size'}
        
        # Test statistic
        mean_diff = diff.mean()
        se_diff = diff.std() / np.sqrt(len(diff))
        
        t_stat = mean_diff / se_diff
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(diff) - 1))
        
        return {
            'mean_difference': mean_diff,
            'standard_error': se_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better_model': 1 if mean_diff > 0 else 2
        }


class RobustnessAnalyzer:
    """Analyze model robustness to various perturbations"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def test_noise_robustness(
        self, 
        model: BaseModel, 
        test_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Test model robustness to input noise"""
        
        logger.info("Testing noise robustness")
        
        original_features = test_data.drop(columns=[target_column])
        original_targets = test_data[target_column]
        
        # Get baseline performance
        baseline_predictions = model.predict(original_features)
        baseline_metrics = PerformanceMetrics.calculate_returns_metrics(
            pd.Series(baseline_predictions)
        )
        
        noise_results = {}
        
        for noise_level in self.config.noise_levels:
            # Add noise to features
            noise = np.random.normal(0, noise_level, original_features.shape)
            noisy_features = original_features + noise
            
            # Get predictions with noisy data
            noisy_predictions = model.predict(noisy_features)
            noisy_metrics = PerformanceMetrics.calculate_returns_metrics(
                pd.Series(noisy_predictions)
            )
            
            # Calculate performance degradation
            degradation = {}
            for metric, value in baseline_metrics.items():
                if metric in noisy_metrics and value != 0:
                    degradation[metric] = (value - noisy_metrics[metric]) / abs(value)
            
            noise_results[f"noise_{noise_level}"] = {
                'metrics': noisy_metrics,
                'degradation': degradation,
                'avg_degradation': np.mean(list(degradation.values()))
            }
        
        return noise_results
    
    def test_missing_data_robustness(
        self,
        model: BaseModel,
        test_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Test model robustness to missing data"""
        
        logger.info("Testing missing data robustness")
        
        original_features = test_data.drop(columns=[target_column])
        
        baseline_predictions = model.predict(original_features)
        baseline_metrics = PerformanceMetrics.calculate_returns_metrics(
            pd.Series(baseline_predictions)
        )
        
        missing_results = {}
        
        for missing_ratio in self.config.missing_data_ratios:
            # Create missing data mask
            mask = np.random.random(original_features.shape) < missing_ratio
            missing_features = original_features.copy()
            missing_features[mask] = np.nan
            
            # Fill missing values (simple forward fill)
            missing_features = missing_features.fillna(method='ffill').fillna(0)
            
            try:
                # Get predictions with missing data
                missing_predictions = model.predict(missing_features)
                missing_metrics = PerformanceMetrics.calculate_returns_metrics(
                    pd.Series(missing_predictions)
                )
                
                # Calculate performance degradation
                degradation = {}
                for metric, value in baseline_metrics.items():
                    if metric in missing_metrics and value != 0:
                        degradation[metric] = (value - missing_metrics[metric]) / abs(value)
                
                missing_results[f"missing_{missing_ratio}"] = {
                    'metrics': missing_metrics,
                    'degradation': degradation,
                    'avg_degradation': np.mean(list(degradation.values()))
                }
                
            except Exception as e:
                missing_results[f"missing_{missing_ratio}"] = {
                    'error': str(e),
                    'avg_degradation': 1.0  # Complete failure
                }
        
        return missing_results


class ModelValidator:
    """Main model validation class"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.robustness_analyzer = RobustnessAnalyzer(config)
        self.validation_results: Dict[str, Any] = {}
    
    def validate_model(
        self,
        model: BaseModel,
        test_data: pd.DataFrame,
        target_column: str,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Comprehensive model validation"""
        
        logger.info(f"Starting validation for {model_name}")
        
        results = {
            'model_name': model_name,
            'validation_timestamp': datetime.now().isoformat(),
            'test_samples': len(test_data)
        }
        
        # Prepare data
        features = test_data.drop(columns=[target_column])
        targets = test_data[target_column]
        
        # Get predictions
        try:
            predictions = model.predict(features)
            results['prediction_success'] = True
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}", error=str(e))
            results['prediction_success'] = False
            results['error'] = str(e)
            return results
        
        # Calculate performance metrics
        if 'timestamp' in test_data.columns:
            returns = pd.Series(predictions, index=test_data['timestamp'])
        else:
            returns = pd.Series(predictions)
        
        results['performance_metrics'] = PerformanceMetrics.calculate_returns_metrics(
            returns, self.config.benchmark_return
        )
        
        results['trading_metrics'] = PerformanceMetrics.calculate_trading_metrics(
            predictions, targets.values, targets
        )
        
        # Statistical significance tests
        if self.config.enable_statistical_tests:
            results['statistical_tests'] = self._run_statistical_tests(returns)
        
        # Robustness analysis
        results['robustness_analysis'] = {
            'noise_robustness': self.robustness_analyzer.test_noise_robustness(
                model, test_data, target_column
            ),
            'missing_data_robustness': self.robustness_analyzer.test_missing_data_robustness(
                model, test_data, target_column
            )
        }
        
        # Risk analysis
        results['risk_analysis'] = self._analyze_risk(returns)
        
        # Model diagnostics
        results['diagnostics'] = self._run_model_diagnostics(predictions, targets)
        
        # Generate visualizations
        if self.config.create_plots:
            results['plots'] = self._create_validation_plots(
                returns, predictions, targets, model_name
            )
        
        # Overall assessment
        results['overall_assessment'] = self._assess_model_quality(results)
        
        self.validation_results[model_name] = results
        
        logger.info(f"Validation completed for {model_name}",
                   sharpe_ratio=results['performance_metrics'].get('sharpe_ratio', 0),
                   max_drawdown=results['performance_metrics'].get('max_drawdown', 0))
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models"""
        
        if not self.config.enable_model_comparison or len(model_results) < 2:
            return {}
        
        logger.info("Comparing models", n_models=len(model_results))
        
        comparison = {
            'models_compared': list(model_results.keys()),
            'comparison_timestamp': datetime.now().isoformat(),
            'metric_comparison': {},
            'statistical_comparison': {},
            'rankings': {}
        }
        
        # Extract metrics for comparison
        metrics_data = {}
        for model_name, results in model_results.items():
            metrics_data[model_name] = results.get('performance_metrics', {})
        
        # Compare each metric
        for metric in self.config.comparison_metrics:
            if all(metric in metrics for metrics in metrics_data.values()):
                metric_values = {name: metrics[metric] for name, metrics in metrics_data.items()}
                
                comparison['metric_comparison'][metric] = {
                    'values': metric_values,
                    'best_model': max(metric_values, key=metric_values.get),
                    'worst_model': min(metric_values, key=metric_values.get),
                    'spread': max(metric_values.values()) - min(metric_values.values())
                }
        
        # Statistical comparisons (pairwise)
        model_names = list(model_results.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Get returns for comparison
                returns1 = self._extract_returns_from_results(model_results[model1])
                returns2 = self._extract_returns_from_results(model_results[model2])
                
                if returns1 is not None and returns2 is not None:
                    test_result = StatisticalTests.model_comparison_test(returns1, returns2)
                    comparison['statistical_comparison'][f"{model1}_vs_{model2}"] = test_result
        
        # Create rankings
        for metric in self.config.comparison_metrics:
            if metric in comparison['metric_comparison']:
                values = comparison['metric_comparison'][metric]['values']
                # Higher is better for most metrics (except drawdown, volatility)
                reverse = metric not in ['max_drawdown', 'volatility', 'var_5', 'cvar_5']
                ranking = sorted(values.items(), key=lambda x: x[1], reverse=reverse)
                comparison['rankings'][metric] = [name for name, value in ranking]
        
        # Overall ranking (weighted average of ranks)
        overall_scores = {name: 0 for name in model_names}
        for metric, ranking in comparison['rankings'].items():
            for i, model_name in enumerate(ranking):
                overall_scores[model_name] += (len(ranking) - i)
        
        comparison['overall_ranking'] = sorted(
            overall_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        return comparison
    
    def _run_statistical_tests(self, returns: pd.Series) -> Dict[str, Any]:
        """Run statistical significance tests"""
        
        tests = {}
        
        # Sharpe ratio significance
        tests['sharpe_test'] = StatisticalTests.sharpe_ratio_test(
            returns, confidence=0.95
        )
        
        # Bootstrap confidence intervals for key metrics
        if len(returns) > 30:
            returns_array = returns.dropna().values
            
            # Sharpe ratio bootstrap
            sharpe_func = lambda x: x.mean() / x.std() * np.sqrt(252)
            tests['sharpe_bootstrap'] = StatisticalTests.bootstrap_confidence_interval(
                returns_array, sharpe_func, self.config.bootstrap_samples
            )
            
            # Maximum drawdown bootstrap
            def max_dd_func(x):
                cumulative = np.cumprod(1 + x)
                rolling_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - rolling_max) / rolling_max
                return np.min(drawdown)
            
            tests['max_drawdown_bootstrap'] = StatisticalTests.bootstrap_confidence_interval(
                returns_array, max_dd_func, self.config.bootstrap_samples
            )
        
        return tests
    
    def _analyze_risk(self, returns: pd.Series) -> Dict[str, Any]:
        """Comprehensive risk analysis"""
        
        risk_analysis = {}
        
        if len(returns) == 0:
            return risk_analysis
        
        returns_clean = returns.dropna()
        
        # Value at Risk
        risk_analysis['var'] = {
            f'var_{int(conf*100)}': np.percentile(returns_clean, conf * 100)
            for conf in [0.01, 0.05, 0.1]
        }
        
        # Conditional Value at Risk
        for conf in [0.01, 0.05, 0.1]:
            var_threshold = risk_analysis['var'][f'var_{int(conf*100)}']
            cvar = returns_clean[returns_clean <= var_threshold].mean()
            risk_analysis[f'cvar_{int(conf*100)}'] = cvar
        
        # Tail ratio
        positive_tail = np.percentile(returns_clean, 95)
        negative_tail = np.percentile(returns_clean, 5)
        risk_analysis['tail_ratio'] = abs(positive_tail / negative_tail) if negative_tail != 0 else float('inf')
        
        # Downside deviation
        negative_returns = returns_clean[returns_clean < 0]
        risk_analysis['downside_deviation'] = negative_returns.std() if len(negative_returns) > 0 else 0
        
        # Sortino ratio
        if risk_analysis['downside_deviation'] > 0:
            risk_analysis['sortino_ratio'] = returns_clean.mean() / risk_analysis['downside_deviation'] * np.sqrt(252)
        else:
            risk_analysis['sortino_ratio'] = float('inf')
        
        return risk_analysis
    
    def _run_model_diagnostics(self, predictions: np.ndarray, targets: pd.Series) -> Dict[str, Any]:
        """Run model diagnostic tests"""
        
        diagnostics = {}
        
        # Prediction distribution analysis
        diagnostics['prediction_stats'] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'skewness': stats.skew(predictions),
            'kurtosis': stats.kurtosis(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions)
        }
        
        # Residual analysis (if applicable)
        if len(predictions) == len(targets):
            residuals = predictions - targets.values
            
            diagnostics['residual_analysis'] = {
                'mean_residual': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals)
            }
            
            # Ljung-Box test for autocorrelation in residuals
            if len(residuals) > 20:
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    result = acorr_ljungbox(residuals, lags=10, return_df=False)
                    # Handle different return formats
                    if isinstance(result, tuple):
                        lb_stat, lb_pvalue = result
                    else:
                        lb_stat = result.get('lb_stat', 0)
                        lb_pvalue = result.get('lb_pvalue', 1.0)
                    
                    # Ensure p_value is numeric
                    if isinstance(lb_pvalue, (int, float)):
                        diagnostics['ljung_box_test'] = {
                            'statistic': float(lb_stat),
                            'p_value': float(lb_pvalue),
                            'autocorrelated': lb_pvalue < 0.05
                        }
                except Exception as e:
                    logger.warning(f"Ljung-Box test failed: {e}")
                    diagnostics['ljung_box_test'] = {'error': str(e)}
        
        return diagnostics
    
    def _create_validation_plots(
        self,
        returns: pd.Series,
        predictions: np.ndarray,
        targets: pd.Series,
        model_name: str
    ) -> Dict[str, str]:
        """Create validation plots"""
        
        if not self.config.plot_save_dir:
            return {}
        
        plots = {}
        
        try:
            # Returns distribution
            plt.figure(figsize=(10, 6))
            plt.hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'{model_name} - Returns Distribution')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plot_path = self.config.plot_save_dir / f"{model_name}_returns_dist.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['returns_distribution'] = str(plot_path)
            
            # Cumulative returns
            plt.figure(figsize=(12, 6))
            cumulative = (1 + returns).cumprod()
            plt.plot(cumulative.index, cumulative.values)
            plt.title(f'{model_name} - Cumulative Returns')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.grid(True, alpha=0.3)
            plot_path = self.config.plot_save_dir / f"{model_name}_cumulative_returns.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['cumulative_returns'] = str(plot_path)
            
            # Drawdown plot
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            plt.title(f'{model_name} - Drawdown')
            plt.xlabel('Time')
            plt.ylabel('Drawdown')
            plt.grid(True, alpha=0.3)
            plot_path = self.config.plot_save_dir / f"{model_name}_drawdown.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['drawdown'] = str(plot_path)
            
            # Q-Q plot for normality check
            plt.figure(figsize=(8, 8))
            stats.probplot(returns.dropna(), dist="norm", plot=plt)
            plt.title(f'{model_name} - Q-Q Plot (Normality Test)')
            plot_path = self.config.plot_save_dir / f"{model_name}_qq_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['qq_plot'] = str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating plots for {model_name}", error=str(e))
        
        return plots
    
    def _assess_model_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall model quality"""
        
        assessment = {
            'quality_score': 0.0,
            'quality_grade': 'F',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        metrics = results.get('performance_metrics', {})
        
        # Scoring criteria
        score = 0
        
        # Sharpe ratio (0-30 points)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2.0:
            score += 30
            assessment['strengths'].append('Excellent Sharpe ratio')
        elif sharpe > 1.5:
            score += 25
            assessment['strengths'].append('Good Sharpe ratio')
        elif sharpe > 1.0:
            score += 20
        elif sharpe > 0.5:
            score += 10
        else:
            assessment['weaknesses'].append('Poor Sharpe ratio')
        
        # Maximum drawdown (0-25 points)
        max_dd = abs(metrics.get('max_drawdown', 1))
        if max_dd < 0.05:
            score += 25
            assessment['strengths'].append('Low maximum drawdown')
        elif max_dd < 0.10:
            score += 20
        elif max_dd < 0.20:
            score += 15
        elif max_dd < 0.30:
            score += 10
        else:
            assessment['weaknesses'].append('High maximum drawdown')
        
        # Hit rate (0-15 points)
        hit_rate = metrics.get('hit_rate', 0.5)
        if hit_rate > 0.6:
            score += 15
            assessment['strengths'].append('Good hit rate')
        elif hit_rate > 0.55:
            score += 10
        elif hit_rate < 0.45:
            assessment['weaknesses'].append('Poor hit rate')
        
        # Profit factor (0-15 points)
        profit_factor = metrics.get('profit_factor', 1)
        if profit_factor > 2.0:
            score += 15
            assessment['strengths'].append('High profit factor')
        elif profit_factor > 1.5:
            score += 10
        elif profit_factor < 1.0:
            assessment['weaknesses'].append('Negative profit factor')
        
        # Statistical significance (0-15 points)
        stat_tests = results.get('statistical_tests', {})
        sharpe_test = stat_tests.get('sharpe_test', {})
        if sharpe_test.get('significant', False):
            score += 15
            assessment['strengths'].append('Statistically significant performance')
        else:
            assessment['weaknesses'].append('Performance not statistically significant')
        
        # Normalize score to 0-100
        assessment['quality_score'] = min(100, score)
        
        # Assign grade
        if assessment['quality_score'] >= 90:
            assessment['quality_grade'] = 'A+'
        elif assessment['quality_score'] >= 85:
            assessment['quality_grade'] = 'A'
        elif assessment['quality_score'] >= 80:
            assessment['quality_grade'] = 'A-'
        elif assessment['quality_score'] >= 75:
            assessment['quality_grade'] = 'B+'
        elif assessment['quality_score'] >= 70:
            assessment['quality_grade'] = 'B'
        elif assessment['quality_score'] >= 65:
            assessment['quality_grade'] = 'B-'
        elif assessment['quality_score'] >= 60:
            assessment['quality_grade'] = 'C+'
        elif assessment['quality_score'] >= 55:
            assessment['quality_grade'] = 'C'
        elif assessment['quality_score'] >= 50:
            assessment['quality_grade'] = 'C-'
        else:
            assessment['quality_grade'] = 'F'
        
        # Generate recommendations
        if max_dd > 0.2:
            assessment['recommendations'].append('Implement stricter risk controls')
        if hit_rate < 0.5:
            assessment['recommendations'].append('Improve signal quality or entry timing')
        if not sharpe_test.get('significant', False):
            assessment['recommendations'].append('Collect more data or improve model')
        
        return assessment
    
    def _extract_returns_from_results(self, results: Dict[str, Any]) -> Optional[pd.Series]:
        """Extract returns series from validation results"""
        # This would need to be implemented based on how returns are stored
        # For now, return None
        return None
    
    def save_validation_results(self, path: Path):
        """Save validation results to file"""
        
        with open(path, 'w') as f:
            # Make results JSON serializable
            serializable_results = self._make_json_serializable(self.validation_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Validation results saved", path=str(path))
    
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
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj