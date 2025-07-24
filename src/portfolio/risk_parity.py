"""
Risk Parity Portfolio Optimization

Advanced risk parity allocation strategies with:
- Equal Risk Contribution (ERC) optimization
- Hierarchical Risk Parity (HRP) using clustering
- Risk budgeting with target allocations
- Volatility targeting and scaling
- Regime-aware risk parity
- Multi-asset class risk parity
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import optimize, cluster
from scipy.spatial.distance import squareform
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RiskParityConfig:
    """Configuration for risk parity optimization"""
    
    # Risk parity method
    method: str = "equal_risk_contribution"  # "equal_risk_contribution", "hierarchical", "risk_budgeting"
    
    # Equal Risk Contribution settings
    erc_tolerance: float = 1e-6  # Convergence tolerance for ERC
    erc_max_iterations: int = 1000
    
    # Hierarchical Risk Parity settings
    hrp_linkage_method: str = "ward"  # "ward", "single", "complete", "average"
    hrp_distance_metric: str = "correlation"  # "correlation", "covariance", "euclidean"
    
    # Risk budgeting settings
    risk_budgets: Optional[Dict[str, float]] = None  # Target risk allocations
    risk_budget_tolerance: float = 0.01  # Tolerance for risk budget matching
    
    # Volatility targeting
    enable_vol_targeting: bool = True
    target_volatility: float = 0.12  # 12% annual volatility target
    vol_lookback_days: int = 60  # Days for volatility estimation
    
    # Portfolio constraints
    min_weight: float = 0.005  # Minimum 0.5% allocation
    max_weight: float = 0.5    # Maximum 50% allocation
    allow_zero_weights: bool = True
    
    # Numerical optimization
    regularization: float = 1e-8
    optimization_method: str = "SLSQP"  # "SLSQP", "trust-constr", "L-BFGS-B"
    
    # Regime awareness
    enable_regime_adjustment: bool = False
    crisis_correlation_threshold: float = 0.8
    crisis_vol_multiplier: float = 1.5
    
    def __post_init__(self):
        """Validate configuration"""
        if self.method not in ["equal_risk_contribution", "hierarchical", "risk_budgeting"]:
            raise ValueError("Invalid risk parity method")
        if not (0 < self.target_volatility < 1):
            raise ValueError("target_volatility must be between 0 and 1")
        if self.min_weight >= self.max_weight:
            raise ValueError("min_weight must be less than max_weight")


class EqualRiskContribution:
    """Equal Risk Contribution (ERC) optimizer"""
    
    def __init__(self, config: RiskParityConfig):
        self.config = config
        
    def optimize(
        self,
        cov_matrix: pd.DataFrame,
        initial_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize portfolio for equal risk contribution
        
        Args:
            cov_matrix: Covariance matrix
            initial_weights: Starting weights for optimization
            
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        
        n_assets = len(cov_matrix)
        
        if initial_weights is None:
            # Start with equal weights
            initial_weights = np.ones(n_assets) / n_assets
        
        # Objective function: minimize sum of squared differences in risk contributions
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            marginal_risk = (cov_matrix.values @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Target equal risk contribution
            target_risk = portfolio_vol / n_assets
            
            # Minimize squared deviations from target
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Optimize
        result = optimize.minimize(
            objective,
            initial_weights,
            method=self.config.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.erc_max_iterations,
                'ftol': self.config.erc_tolerance
            }
        )
        
        optimization_info = {
            'success': result.success,
            'iterations': result.nit,
            'final_objective': result.fun,
            'convergence_message': result.message
        }
        
        if not result.success:
            logger.warning("ERC optimization did not converge", message=result.message)
            # Fallback to inverse volatility weights
            inv_vol_weights = self._calculate_inverse_volatility_weights(cov_matrix)
            return inv_vol_weights, optimization_info
        
        return result.x, optimization_info
    
    def _calculate_inverse_volatility_weights(
        self,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Calculate inverse volatility weights as fallback"""
        
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        inv_vol_weights = 1 / volatilities
        
        # Normalize
        inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Apply bounds
        inv_vol_weights = np.clip(inv_vol_weights, self.config.min_weight, self.config.max_weight)
        
        # Renormalize after clipping
        return inv_vol_weights / np.sum(inv_vol_weights)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) optimizer"""
    
    def __init__(self, config: RiskParityConfig):
        self.config = config
        
    def optimize(
        self,
        returns_data: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize portfolio using Hierarchical Risk Parity
        
        Args:
            returns_data: Historical returns data
            cov_matrix: Optional covariance matrix
            
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        
        if cov_matrix is None:
            cov_matrix = returns_data.cov()
        
        # Step 1: Calculate distance matrix
        if self.config.hrp_distance_metric == "correlation":
            corr_matrix = returns_data.corr()
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        elif self.config.hrp_distance_metric == "covariance":
            distance_matrix = np.sqrt(cov_matrix)
        else:  # euclidean
            distance_matrix = np.sqrt(((returns_data.T.values[:, None, :] - 
                                     returns_data.T.values[None, :, :]) ** 2).sum(axis=2))
        
        # Step 2: Hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = cluster.hierarchy.linkage(
            condensed_distances, 
            method=self.config.hrp_linkage_method
        )
        
        # Step 3: Quasi-diagonalization
        sorted_indices = self._get_quasi_diag_order(linkage_matrix, len(returns_data.columns))
        
        # Step 4: Recursive bisection
        weights = np.ones(len(returns_data.columns))
        self._recursive_bisection(
            weights, sorted_indices, cov_matrix.values, 0, len(sorted_indices)
        )
        
        optimization_info = {
            'success': True,
            'linkage_matrix': linkage_matrix,
            'sorted_indices': sorted_indices,
            'method': 'hierarchical_risk_parity'
        }
        
        return weights, optimization_info
    
    def _get_quasi_diag_order(self, linkage_matrix: np.ndarray, n_assets: int) -> List[int]:
        """Get quasi-diagonal ordering from hierarchical clustering"""
        
        # Convert scipy linkage to ordered list
        cluster_order = cluster.hierarchy.leaves_list(linkage_matrix)
        return cluster_order.tolist()
    
    def _recursive_bisection(
        self,
        weights: np.ndarray,
        indices: List[int],
        cov_matrix: np.ndarray,
        start: int,
        end: int
    ) -> None:
        """Recursively bisect clusters and allocate weights"""
        
        if end - start <= 1:
            return
        
        # Split point
        mid = (start + end) // 2
        
        # Left and right clusters
        left_indices = indices[start:mid]
        right_indices = indices[mid:end]
        
        # Calculate cluster variances
        left_cov = cov_matrix[np.ix_(left_indices, left_indices)]
        right_cov = cov_matrix[np.ix_(right_indices, right_indices)]
        
        # Equal weight within clusters for variance calculation
        left_w = np.ones(len(left_indices)) / len(left_indices)
        right_w = np.ones(len(right_indices)) / len(right_indices)
        
        left_var = left_w @ left_cov @ left_w
        right_var = right_w @ right_cov @ right_w
        
        # Allocate weights inversely proportional to variance
        total_var = left_var + right_var
        left_weight = right_var / total_var  # Inverse weighting
        right_weight = left_var / total_var
        
        # Update weights
        for i in left_indices:
            weights[i] *= left_weight
        for i in right_indices:
            weights[i] *= right_weight
        
        # Recursive calls
        self._recursive_bisection(weights, indices, cov_matrix, start, mid)
        self._recursive_bisection(weights, indices, cov_matrix, mid, end)


class RiskBudgeting:
    """Risk budgeting optimizer with target risk allocations"""
    
    def __init__(self, config: RiskParityConfig):
        self.config = config
        
    def optimize(
        self,
        cov_matrix: pd.DataFrame,
        risk_budgets: Dict[str, float],
        initial_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize portfolio to match target risk budgets
        
        Args:
            cov_matrix: Covariance matrix
            risk_budgets: Target risk allocations {asset: budget}
            initial_weights: Starting weights
            
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        
        n_assets = len(cov_matrix)
        
        # Convert risk budgets to array
        budget_array = np.array([
            risk_budgets.get(asset, 1.0/n_assets) 
            for asset in cov_matrix.index
        ])
        
        # Normalize budgets
        budget_array = budget_array / np.sum(budget_array)
        
        if initial_weights is None:
            initial_weights = budget_array.copy()
        
        # Objective function: minimize squared deviations from target risk budgets
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            marginal_risk = (cov_matrix.values @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Normalize risk contributions
            risk_contributions = risk_contributions / np.sum(risk_contributions)
            
            # Minimize squared deviations from target budgets
            return np.sum((risk_contributions - budget_array) ** 2)
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Optional: Risk budget constraints (soft constraints via penalty)
        if self.config.risk_budget_tolerance > 0:
            def risk_budget_constraint(weights):
                portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
                marginal_risk = (cov_matrix.values @ weights) / portfolio_vol
                risk_contributions = weights * marginal_risk
                risk_contributions = risk_contributions / np.sum(risk_contributions)
                
                # Maximum deviation from target
                max_deviation = np.max(np.abs(risk_contributions - budget_array))
                return self.config.risk_budget_tolerance - max_deviation
            
            constraints.append({
                'type': 'ineq',
                'fun': risk_budget_constraint
            })
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Optimize
        result = optimize.minimize(
            objective,
            initial_weights,
            method=self.config.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimization_info = {
            'success': result.success,
            'iterations': result.nit,
            'final_objective': result.fun,
            'target_budgets': budget_array,
            'convergence_message': result.message
        }
        
        if not result.success:
            logger.warning("Risk budgeting optimization did not converge")
            # Fallback to budget-proportional weights
            return budget_array, optimization_info
        
        return result.x, optimization_info


class RiskParityOptimizer:
    """Main risk parity portfolio optimizer"""
    
    def __init__(self, config: RiskParityConfig):
        self.config = config
        self.erc_optimizer = EqualRiskContribution(config)
        self.hrp_optimizer = HierarchicalRiskParity(config)
        self.risk_budgeting_optimizer = RiskBudgeting(config)
        self.optimization_history: List[Dict[str, Any]] = []
        
    def optimize_portfolio(
        self,
        returns_data: pd.DataFrame,
        risk_budgets: Optional[Dict[str, float]] = None,
        regime_indicator: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform risk parity portfolio optimization
        
        Args:
            returns_data: Historical returns data
            risk_budgets: Optional target risk budgets for risk budgeting method
            regime_indicator: Market regime ("normal", "crisis", "recovery")
            
        Returns:
            Optimization results including weights and metrics
        """
        
        logger.info(f"Starting risk parity optimization using {self.config.method}",
                   assets=len(returns_data.columns))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'assets': list(returns_data.columns),
            'method': self.config.method,
            'optimal_weights': {},
            'risk_contributions': {},
            'risk_metrics': {},
            'optimization_details': {}
        }
        
        try:
            # Calculate covariance matrix
            cov_matrix = self._calculate_covariance_matrix(returns_data, regime_indicator)
            
            # Apply regime adjustments if enabled
            if self.config.enable_regime_adjustment and regime_indicator == "crisis":
                cov_matrix = self._apply_regime_adjustment(cov_matrix, regime_indicator)
            
            # Perform optimization based on method
            if self.config.method == "equal_risk_contribution":
                weights, opt_info = self.erc_optimizer.optimize(cov_matrix)
            elif self.config.method == "hierarchical":
                weights, opt_info = self.hrp_optimizer.optimize(returns_data, cov_matrix)
            elif self.config.method == "risk_budgeting":
                budgets = risk_budgets or self.config.risk_budgets or {}
                weights, opt_info = self.risk_budgeting_optimizer.optimize(cov_matrix, budgets)
            else:
                raise ValueError(f"Unknown method: {self.config.method}")
            
            # Apply volatility targeting if enabled
            if self.config.enable_vol_targeting:
                weights = self._apply_volatility_targeting(weights, cov_matrix)
            
            # Calculate risk contributions and metrics
            risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)
            risk_metrics = self._calculate_portfolio_metrics(weights, cov_matrix, returns_data)
            
            # Store results
            result['optimal_weights'] = {
                asset: float(weight) for asset, weight in zip(returns_data.columns, weights)
            }
            result['risk_contributions'] = {
                asset: float(contrib) for asset, contrib in zip(returns_data.columns, risk_contributions)
            }
            result['risk_metrics'] = risk_metrics
            result['optimization_details'] = opt_info
            
            # Store in history
            self.optimization_history.append({
                'timestamp': result['timestamp'],
                'method': self.config.method,
                'portfolio_volatility': risk_metrics['portfolio_volatility'],
                'risk_concentration': risk_metrics['risk_concentration'],
                'max_risk_contribution': max(risk_contributions),
                'optimization_success': opt_info.get('success', False)
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info("Risk parity optimization completed",
                       portfolio_vol=risk_metrics['portfolio_volatility'],
                       risk_concentration=risk_metrics['risk_concentration'],
                       max_weight=risk_metrics['max_weight'])
            
        except Exception as e:
            logger.error("Error in risk parity optimization", error=str(e))
            result['error'] = str(e)
        
        return result
    
    def _calculate_covariance_matrix(
        self,
        returns_data: pd.DataFrame,
        regime_indicator: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate covariance matrix with optional regime adjustments"""
        
        # Use recent data for covariance estimation
        if len(returns_data) > self.config.vol_lookback_days:
            recent_data = returns_data.tail(self.config.vol_lookback_days)
        else:
            recent_data = returns_data
        
        # Calculate sample covariance
        cov_matrix = recent_data.cov() * 252  # Annualized
        
        # Add regularization
        cov_matrix += np.eye(len(cov_matrix)) * self.config.regularization
        
        return cov_matrix
    
    def _apply_regime_adjustment(
        self,
        cov_matrix: pd.DataFrame,
        regime: str
    ) -> pd.DataFrame:
        """Apply regime-specific adjustments to covariance matrix"""
        
        if regime == "crisis":
            # Increase correlations and volatilities during crisis
            corr_matrix = cov_matrix.corr()
            vol_vector = np.sqrt(np.diag(cov_matrix))
            
            # Increase correlations toward crisis threshold
            adjusted_corr = corr_matrix * 0.5 + self.config.crisis_correlation_threshold * 0.5
            np.fill_diagonal(adjusted_corr.values, 1.0)
            
            # Increase volatilities
            adjusted_vol = vol_vector * self.config.crisis_vol_multiplier
            
            # Reconstruct covariance matrix
            adjusted_cov = np.outer(adjusted_vol, adjusted_vol) * adjusted_corr.values
            return pd.DataFrame(adjusted_cov, index=cov_matrix.index, columns=cov_matrix.columns)
        
        return cov_matrix
    
    def _apply_volatility_targeting(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Scale portfolio to target volatility"""
        
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        
        if portfolio_vol > 0:
            vol_scalar = self.config.target_volatility / portfolio_vol
            # Don't scale up too much to avoid concentration
            vol_scalar = min(vol_scalar, 2.0)
            scaled_weights = weights * vol_scalar
            
            # Renormalize if needed
            if np.sum(scaled_weights) != 1.0:
                scaled_weights = scaled_weights / np.sum(scaled_weights)
            
            return scaled_weights
        
        return weights
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Calculate risk contributions for each asset"""
        
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        
        if portfolio_vol > 0:
            marginal_risk = (cov_matrix.values @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Normalize to percentages
            return risk_contributions / np.sum(risk_contributions)
        else:
            return weights / np.sum(weights)
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        
        # Expected return (simple historical average)
        expected_returns = returns_data.mean() * 252
        portfolio_return = weights @ expected_returns.values
        
        # Risk contributions
        risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)
        
        # Risk concentration (Herfindahl index of risk contributions)
        risk_concentration = np.sum(risk_contributions ** 2)
        
        # Diversification metrics
        individual_vols = np.sqrt(np.diag(cov_matrix.values))
        weighted_avg_vol = weights @ individual_vols
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Weight concentration
        weight_concentration = np.sum(weights ** 2)
        effective_positions = 1 / weight_concentration if weight_concentration > 0 else 0
        
        return {
            'portfolio_volatility': float(portfolio_vol),
            'expected_return': float(portfolio_return),
            'sharpe_ratio': float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0,
            'risk_concentration': float(risk_concentration),
            'weight_concentration': float(weight_concentration),
            'diversification_ratio': float(diversification_ratio),
            'effective_positions': float(effective_positions),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights)),
            'max_risk_contribution': float(np.max(risk_contributions)),
            'min_risk_contribution': float(np.min(risk_contributions))
        }
    
    def get_risk_parity_dashboard(self) -> Dict[str, Any]:
        """Get risk parity optimization dashboard data"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'method': self.config.method,
                'target_volatility': self.config.target_volatility,
                'vol_targeting_enabled': self.config.enable_vol_targeting,
                'regime_adjustment_enabled': self.config.enable_regime_adjustment,
                'min_weight': self.config.min_weight,
                'max_weight': self.config.max_weight
            },
            'optimization_history': self.optimization_history[-20:],
            'performance_stats': {
                'avg_portfolio_volatility': np.mean([h['portfolio_volatility'] for h in self.optimization_history]) if self.optimization_history else 0,
                'avg_risk_concentration': np.mean([h['risk_concentration'] for h in self.optimization_history]) if self.optimization_history else 0,
                'optimization_success_rate': np.mean([h['optimization_success'] for h in self.optimization_history]) if self.optimization_history else 0,
                'optimizations_count': len(self.optimization_history)
            },
            'method_stats': {
                method: len([h for h in self.optimization_history if h['method'] == method])
                for method in ["equal_risk_contribution", "hierarchical", "risk_budgeting"]
            }
        }