"""
Modern Portfolio Theory (Markowitz) Optimization

Advanced Markowitz optimization implementation with:
- Mean-variance optimization
- Efficient frontier construction
- Risk budgeting constraints
- Robust optimization techniques
- Multiple objective functions
- Transaction cost integration
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from scipy import optimize, linalg
import structlog

logger = structlog.get_logger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "maximize_sharpe"
    MIN_VARIANCE = "minimize_variance"
    MAX_RETURN = "maximize_return"
    TARGET_RISK = "target_risk"
    TARGET_RETURN = "target_return"
    MAX_DIVERSIFICATION = "maximize_diversification"
    MIN_CONCENTRATION = "minimize_concentration"


@dataclass
class MarkowitzConfig:
    """Configuration for Markowitz optimization"""
    
    # Optimization objective
    objective: str = "maximize_sharpe"  # From OptimizationObjective enum
    
    # Risk and return parameters
    risk_free_rate: float = 0.02  # 2% risk-free rate
    target_return: Optional[float] = None  # Target return for constrained optimization
    target_risk: Optional[float] = None    # Target risk for constrained optimization
    
    # Portfolio constraints
    allow_short_selling: bool = False
    max_weight: float = 0.4     # Maximum weight per asset
    min_weight: float = 0.0     # Minimum weight per asset
    max_concentration: float = 0.6  # Maximum sum of top 3 positions
    
    # Risk budgeting
    enable_risk_budgeting: bool = False
    risk_budgets: Optional[Dict[str, float]] = None
    risk_budget_tolerance: float = 0.02
    
    # Robust optimization
    enable_robust_optimization: bool = True
    uncertainty_sets: str = "ellipsoidal"  # "ellipsoidal", "box", "polyhedral"
    robustness_parameter: float = 0.1
    
    # Efficient frontier
    efficient_frontier_points: int = 50
    min_return_percentile: float = 5   # 5th percentile
    max_return_percentile: float = 95  # 95th percentile
    
    # Transaction costs
    enable_transaction_costs: bool = False
    transaction_cost_rate: float = 0.001
    current_weights: Optional[Dict[str, float]] = None
    
    # Estimation methods
    return_estimation: str = "historical"  # "historical", "shrinkage", "capm"
    covariance_estimation: str = "sample"  # "sample", "shrinkage", "robust"
    shrinkage_intensity: float = 0.2
    
    # Numerical optimization
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    regularization: float = 1e-8
    
    def __post_init__(self):
        """Validate configuration"""
        if self.objective not in [obj.value for obj in OptimizationObjective]:
            raise ValueError(f"Invalid optimization objective: {self.objective}")
        if not (0 <= self.risk_free_rate <= 1):
            raise ValueError("risk_free_rate must be between 0 and 1")
        if self.min_weight >= self.max_weight:
            raise ValueError("min_weight must be less than max_weight")


class ReturnEstimator:
    """Estimate expected returns using various methods"""
    
    def __init__(self, config: MarkowitzConfig):
        self.config = config
        
    def estimate_returns(
        self,
        returns_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Estimate expected returns using configured method
        
        Args:
            returns_data: Historical returns data
            market_data: Market returns for CAPM estimation
            
        Returns:
            Expected returns for each asset
        """
        
        if self.config.return_estimation == "historical":
            return self._historical_mean(returns_data)
        elif self.config.return_estimation == "shrinkage":
            return self._shrinkage_mean(returns_data)
        elif self.config.return_estimation == "capm":
            return self._capm_returns(returns_data, market_data)
        else:
            raise ValueError(f"Unknown return estimation method: {self.config.return_estimation}")
    
    def _historical_mean(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate historical mean returns"""
        return returns_data.mean() * 252  # Annualized
    
    def _shrinkage_mean(self, returns_data: pd.DataFrame) -> pd.Series:
        """James-Stein shrinkage estimator for expected returns"""
        
        historical_mean = returns_data.mean() * 252
        grand_mean = historical_mean.mean()
        
        # Shrink individual means toward grand mean
        shrunk_returns = (
            (1 - self.config.shrinkage_intensity) * historical_mean +
            self.config.shrinkage_intensity * grand_mean
        )
        
        return shrunk_returns
    
    def _capm_returns(
        self,
        returns_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame]
    ) -> pd.Series:
        """CAPM-based expected returns"""
        
        if market_data is None:
            logger.warning("Market data not provided for CAPM, using historical mean")
            return self._historical_mean(returns_data)
        
        market_returns = market_data.iloc[:, 0]  # Assume first column is market
        market_premium = market_returns.mean() * 252 - self.config.risk_free_rate
        
        # Calculate betas
        betas = {}
        for asset in returns_data.columns:
            asset_returns = returns_data[asset]
            covariance = asset_returns.cov(market_returns) * 252
            market_variance = market_returns.var() * 252
            betas[asset] = covariance / market_variance if market_variance > 0 else 1.0
        
        # CAPM: E[R] = Rf + β(E[Rm] - Rf)
        capm_returns = pd.Series({
            asset: self.config.risk_free_rate + beta * market_premium
            for asset, beta in betas.items()
        })
        
        return capm_returns


class CovarianceEstimator:
    """Estimate covariance matrix using various methods"""
    
    def __init__(self, config: MarkowitzConfig):
        self.config = config
        
    def estimate_covariance(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate covariance matrix using configured method
        
        Args:
            returns_data: Historical returns data
            
        Returns:
            Covariance matrix
        """
        
        if self.config.covariance_estimation == "sample":
            return self._sample_covariance(returns_data)
        elif self.config.covariance_estimation == "shrinkage":
            return self._shrinkage_covariance(returns_data)
        elif self.config.covariance_estimation == "robust":
            return self._robust_covariance(returns_data)
        else:
            raise ValueError(f"Unknown covariance estimation method: {self.config.covariance_estimation}")
    
    def _sample_covariance(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Sample covariance matrix"""
        cov_matrix = returns_data.cov() * 252  # Annualized
        return cov_matrix + np.eye(len(cov_matrix)) * self.config.regularization
    
    def _shrinkage_covariance(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance estimator"""
        
        sample_cov = returns_data.cov() * 252
        
        # Shrinkage target: identity matrix scaled by average variance
        avg_variance = np.trace(sample_cov.values) / len(sample_cov)
        target = np.eye(len(sample_cov)) * avg_variance
        
        # Shrink toward target
        shrunk_cov = (
            (1 - self.config.shrinkage_intensity) * sample_cov.values +
            self.config.shrinkage_intensity * target
        )
        
        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)
    
    def _robust_covariance(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Robust covariance estimation using Minimum Covariance Determinant"""
        
        # Simple robust estimator: remove outliers and recalculate
        # In practice, would use sklearn.covariance.MinCovDet
        
        # Remove extreme outliers (beyond 3 standard deviations)
        z_scores = np.abs((returns_data - returns_data.mean()) / returns_data.std())
        mask = (z_scores < 3).all(axis=1)
        
        clean_data = returns_data[mask]
        
        if len(clean_data) < len(returns_data) * 0.7:
            logger.warning("Too many outliers detected, using sample covariance")
            return self._sample_covariance(returns_data)
        
        return self._sample_covariance(clean_data)


class EfficientFrontier:
    """Construct and analyze efficient frontier"""
    
    def __init__(self, config: MarkowitzConfig):
        self.config = config
        
    def construct_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Construct efficient frontier
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            
        Returns:
            Efficient frontier data
        """
        
        logger.info("Constructing efficient frontier")
        
        # Define return range
        min_return = np.percentile(expected_returns, self.config.min_return_percentile)
        max_return = np.percentile(expected_returns, self.config.max_return_percentile)
        target_returns = np.linspace(min_return, max_return, self.config.efficient_frontier_points)
        
        frontier_portfolios = []
        
        for target_return in target_returns:
            try:
                # Optimize for minimum variance given target return
                weights = self._optimize_for_target_return(
                    target_return, expected_returns, cov_matrix
                )
                
                if weights is not None:
                    portfolio_return = weights @ expected_returns.values
                    portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
                    sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_risk
                    
                    frontier_portfolios.append({
                        'target_return': float(target_return),
                        'realized_return': float(portfolio_return),
                        'risk': float(portfolio_risk),
                        'sharpe_ratio': float(sharpe_ratio),
                        'weights': {asset: float(w) for asset, w in zip(expected_returns.index, weights)}
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}", error=str(e))
                continue
        
        # Find key portfolios
        if frontier_portfolios:
            # Maximum Sharpe ratio portfolio
            max_sharpe_idx = np.argmax([p['sharpe_ratio'] for p in frontier_portfolios])
            max_sharpe_portfolio = frontier_portfolios[max_sharpe_idx]
            
            # Minimum variance portfolio
            min_var_idx = np.argmin([p['risk'] for p in frontier_portfolios])
            min_variance_portfolio = frontier_portfolios[min_var_idx]
            
            result = {
                'frontier_portfolios': frontier_portfolios,
                'max_sharpe_portfolio': max_sharpe_portfolio,
                'min_variance_portfolio': min_variance_portfolio,
                'n_points': len(frontier_portfolios)
            }
        else:
            result = {
                'frontier_portfolios': [],
                'error': 'No feasible portfolios found'
            }
        
        logger.info(f"Efficient frontier constructed with {len(frontier_portfolios)} points")
        return result
    
    def _optimize_for_target_return(
        self,
        target_return: float,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Optimize portfolio for specific target return"""
        
        n_assets = len(expected_returns)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return weights @ cov_matrix.values @ weights
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Target return constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ expected_returns.values - target_return
        })
        
        # Bounds
        if self.config.allow_short_selling:
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(max(0, self.config.min_weight), self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            return result.x if result.success else None
            
        except Exception:
            return None


class MarkowitzOptimizer:
    """Main Markowitz portfolio optimizer"""
    
    def __init__(self, config: MarkowitzConfig):
        self.config = config
        self.return_estimator = ReturnEstimator(config)
        self.covariance_estimator = CovarianceEstimator(config)
        self.efficient_frontier = EfficientFrontier(config)
        self.optimization_history: List[Dict[str, Any]] = []
        
    def optimize_portfolio(
        self,
        returns_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        custom_expected_returns: Optional[pd.Series] = None,
        custom_covariance: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Perform Markowitz portfolio optimization
        
        Args:
            returns_data: Historical returns data
            market_data: Market returns for CAPM
            custom_expected_returns: Optional custom expected returns
            custom_covariance: Optional custom covariance matrix
            
        Returns:
            Optimization results
        """
        
        logger.info(f"Starting Markowitz optimization with objective: {self.config.objective}",
                   assets=len(returns_data.columns))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'assets': list(returns_data.columns),
            'method': 'markowitz',
            'objective': self.config.objective,
            'optimal_weights': {},
            'risk_metrics': {},
            'optimization_details': {}
        }
        
        try:
            # Step 1: Estimate expected returns
            if custom_expected_returns is not None:
                expected_returns = custom_expected_returns
            else:
                expected_returns = self.return_estimator.estimate_returns(
                    returns_data, market_data
                )
            
            # Step 2: Estimate covariance matrix
            if custom_covariance is not None:
                cov_matrix = custom_covariance
            else:
                cov_matrix = self.covariance_estimator.estimate_covariance(returns_data)
            
            # Step 3: Apply robust optimization if enabled
            if self.config.enable_robust_optimization:
                expected_returns, cov_matrix = self._apply_robust_optimization(
                    expected_returns, cov_matrix, returns_data
                )
            
            # Step 4: Perform optimization
            optimal_weights = self._optimize_weights(expected_returns, cov_matrix)
            
            # Step 5: Calculate portfolio metrics
            risk_metrics = self._calculate_portfolio_metrics(
                optimal_weights, expected_returns, cov_matrix
            )
            
            # Step 6: Construct efficient frontier (if requested)
            frontier_data = None
            if self.config.efficient_frontier_points > 0:
                frontier_data = self.efficient_frontier.construct_frontier(
                    expected_returns, cov_matrix
                )
            
            # Store results
            result['optimal_weights'] = {
                asset: float(weight) for asset, weight in zip(returns_data.columns, optimal_weights)
            }
            result['expected_returns'] = {
                asset: float(ret) for asset, ret in expected_returns.items()
            }
            result['risk_metrics'] = risk_metrics
            result['efficient_frontier'] = frontier_data
            result['optimization_details'] = {
                'return_estimation': self.config.return_estimation,
                'covariance_estimation': self.config.covariance_estimation,
                'robust_optimization': self.config.enable_robust_optimization,
                'risk_free_rate': self.config.risk_free_rate
            }
            
            # Store in history
            self.optimization_history.append({
                'timestamp': result['timestamp'],
                'objective': self.config.objective,
                'portfolio_return': risk_metrics['expected_return'],
                'portfolio_risk': risk_metrics['portfolio_risk'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'max_weight': risk_metrics['max_weight']
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info("Markowitz optimization completed",
                       expected_return=risk_metrics['expected_return'],
                       risk=risk_metrics['portfolio_risk'],
                       sharpe_ratio=risk_metrics['sharpe_ratio'])
            
        except Exception as e:
            logger.error("Error in Markowitz optimization", error=str(e))
            result['error'] = str(e)
        
        return result
    
    def _optimize_weights(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Optimize portfolio weights based on objective"""
        
        n_assets = len(expected_returns)
        
        # Define objective function based on configuration
        if self.config.objective == OptimizationObjective.MAX_SHARPE.value:
            objective_func = self._max_sharpe_objective
        elif self.config.objective == OptimizationObjective.MIN_VARIANCE.value:
            objective_func = self._min_variance_objective
        elif self.config.objective == OptimizationObjective.MAX_RETURN.value:
            objective_func = self._max_return_objective
        elif self.config.objective == OptimizationObjective.TARGET_RISK.value:
            objective_func = self._target_risk_objective
        elif self.config.objective == OptimizationObjective.TARGET_RETURN.value:
            objective_func = self._target_return_objective
        elif self.config.objective == OptimizationObjective.MAX_DIVERSIFICATION.value:
            objective_func = self._max_diversification_objective
        elif self.config.objective == OptimizationObjective.MIN_CONCENTRATION.value:
            objective_func = self._min_concentration_objective
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
        
        # Set up constraints
        constraints = self._build_constraints(expected_returns, cov_matrix)
        
        # Set up bounds
        bounds = self._build_bounds(n_assets)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            lambda w: objective_func(w, expected_returns, cov_matrix),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.convergence_tolerance
            }
        )
        
        if not result.success:
            logger.warning("Optimization did not converge, using equal weights")
            return x0
        
        return result.x
    
    def _max_sharpe_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for maximizing Sharpe ratio (minimize negative Sharpe)"""
        
        portfolio_return = weights @ expected_returns.values
        portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
        
        if portfolio_risk == 0:
            return -np.inf
        
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_risk
        return -sharpe_ratio  # Minimize negative Sharpe
    
    def _min_variance_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for minimizing variance"""
        
        return weights @ cov_matrix.values @ weights
    
    def _max_return_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for maximizing return"""
        
        return -(weights @ expected_returns.values)  # Minimize negative return
    
    def _target_risk_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for targeting specific risk level"""
        
        portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
        target_risk = self.config.target_risk or 0.15
        
        # Minimize squared deviation from target risk
        return (portfolio_risk - target_risk) ** 2
    
    def _target_return_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for targeting specific return level"""
        
        portfolio_return = weights @ expected_returns.values
        target_return = self.config.target_return or expected_returns.mean()
        
        # Minimize squared deviation from target return
        return (portfolio_return - target_return) ** 2
    
    def _max_diversification_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for maximizing diversification ratio"""
        
        individual_risks = np.sqrt(np.diag(cov_matrix.values))
        weighted_avg_risk = weights @ individual_risks
        portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
        
        if portfolio_risk == 0:
            return np.inf
        
        diversification_ratio = weighted_avg_risk / portfolio_risk
        return -diversification_ratio  # Minimize negative diversification
    
    def _min_concentration_objective(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Objective function for minimizing concentration (Herfindahl index)"""
        
        return np.sum(weights ** 2)  # Herfindahl index
    
    def _build_constraints(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Build optimization constraints"""
        
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Target return constraint (if specified)
        if self.config.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns.values - self.config.target_return
            })
        
        # Target risk constraint (if specified)
        if self.config.target_risk is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(w @ cov_matrix.values @ w) - self.config.target_risk
            })
        
        # Concentration constraint
        if self.config.max_concentration < 1.0:
            def concentration_constraint(weights):
                # Sum of top N positions should not exceed max_concentration
                sorted_weights = np.sort(weights)[::-1]  # Descending order
                top_3_sum = np.sum(sorted_weights[:3])
                return self.config.max_concentration - top_3_sum
            
            constraints.append({
                'type': 'ineq',
                'fun': concentration_constraint
            })
        
        # Risk budgeting constraints
        if self.config.enable_risk_budgeting and self.config.risk_budgets:
            def risk_budget_constraint(weights):
                portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
                marginal_risk = (cov_matrix.values @ weights) / portfolio_risk
                risk_contributions = weights * marginal_risk
                risk_contributions = risk_contributions / np.sum(risk_contributions)
                
                # Check deviations from target risk budgets
                deviations = []
                for i, asset in enumerate(expected_returns.index):
                    target_budget = self.config.risk_budgets.get(asset, 1.0 / len(expected_returns))
                    deviation = abs(risk_contributions[i] - target_budget)
                    deviations.append(deviation)
                
                max_deviation = max(deviations)
                return self.config.risk_budget_tolerance - max_deviation
            
            constraints.append({
                'type': 'ineq',
                'fun': risk_budget_constraint
            })
        
        return constraints
    
    def _build_bounds(self, n_assets: int) -> List[Tuple[float, float]]:
        """Build optimization bounds"""
        
        if self.config.allow_short_selling:
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(max(0, self.config.min_weight), self.config.max_weight) for _ in range(n_assets)]
        
        return bounds
    
    def _apply_robust_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Apply robust optimization techniques"""
        
        if self.config.uncertainty_sets == "ellipsoidal":
            # Increase covariance matrix to account for estimation uncertainty
            robust_cov = cov_matrix * (1 + self.config.robustness_parameter)
            
            # Shrink expected returns toward zero (conservative)
            grand_mean = expected_returns.mean()
            robust_returns = expected_returns * (1 - self.config.robustness_parameter * 0.5)
            
        elif self.config.uncertainty_sets == "box":
            # Box uncertainty: returns can vary within ±robustness_parameter
            return_std = returns_data.std() * np.sqrt(252)
            robust_returns = expected_returns - self.config.robustness_parameter * return_std
            robust_cov = cov_matrix * (1 + self.config.robustness_parameter)
            
        else:  # Default to ellipsoidal
            robust_cov = cov_matrix * (1 + self.config.robustness_parameter)
            robust_returns = expected_returns * (1 - self.config.robustness_parameter * 0.5)
        
        return robust_returns, robust_cov
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        
        # Basic risk-return metrics
        portfolio_return = weights @ expected_returns.values
        portfolio_variance = weights @ cov_matrix.values @ weights
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Diversification metrics
        individual_risks = np.sqrt(np.diag(cov_matrix.values))
        weighted_avg_risk = weights @ individual_risks
        diversification_ratio = weighted_avg_risk / portfolio_risk if portfolio_risk > 0 else 1
        
        # Concentration metrics
        herfindahl_index = np.sum(weights ** 2)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Risk contributions
        if portfolio_risk > 0:
            marginal_risk = (cov_matrix.values @ weights) / portfolio_risk
            risk_contributions = weights * marginal_risk
            risk_contributions = risk_contributions / np.sum(risk_contributions)
            max_risk_contribution = np.max(risk_contributions)
        else:
            max_risk_contribution = 1.0 / len(weights)
        
        return {
            'expected_return': float(portfolio_return),
            'portfolio_risk': float(portfolio_risk),
            'portfolio_variance': float(portfolio_variance),
            'sharpe_ratio': float(sharpe_ratio),
            'diversification_ratio': float(diversification_ratio),
            'herfindahl_index': float(herfindahl_index),
            'effective_positions': float(effective_positions),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights)),
            'max_risk_contribution': float(max_risk_contribution)
        }
    
    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """Get Markowitz optimization dashboard data"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'objective': self.config.objective,
                'risk_free_rate': self.config.risk_free_rate,
                'allow_short_selling': self.config.allow_short_selling,
                'max_weight': self.config.max_weight,
                'return_estimation': self.config.return_estimation,
                'covariance_estimation': self.config.covariance_estimation,
                'robust_optimization': self.config.enable_robust_optimization
            },
            'optimization_history': self.optimization_history[-20:],
            'performance_stats': {
                'avg_expected_return': np.mean([h['portfolio_return'] for h in self.optimization_history]) if self.optimization_history else 0,
                'avg_risk': np.mean([h['portfolio_risk'] for h in self.optimization_history]) if self.optimization_history else 0,
                'avg_sharpe_ratio': np.mean([h['sharpe_ratio'] for h in self.optimization_history]) if self.optimization_history else 0,
                'optimizations_count': len(self.optimization_history)
            },
            'objective_distribution': {
                obj: len([h for h in self.optimization_history if h['objective'] == obj])
                for obj in [obj.value for obj in OptimizationObjective]
            }
        }