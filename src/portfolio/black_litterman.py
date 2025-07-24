"""
Black-Litterman Portfolio Optimization

Advanced Black-Litterman implementation with:
- Market equilibrium model integration
- Investor views incorporation
- Confidence-weighted view adjustment
- Dynamic tau parameter estimation
- Regime-aware market views
- Multiple confidence specification methods
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import optimize, linalg
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BlackLittermanConfig:
    """Configuration for Black-Litterman optimization"""
    
    # Risk aversion and market parameters
    risk_aversion: float = 3.0  # Risk aversion coefficient
    tau: Optional[float] = None  # Uncertainty in prior (auto-calculated if None)
    tau_method: str = "auto"  # "auto", "fixed", "1/T", "0.025"
    
    # Market equilibrium
    use_market_cap_weights: bool = True
    market_cap_weights: Optional[Dict[str, float]] = None
    equilibrium_method: str = "reverse_optimization"  # "reverse_optimization", "historical"
    
    # View incorporation
    max_views: int = 10  # Maximum number of views to incorporate
    min_view_confidence: float = 0.1  # Minimum confidence level for views
    view_decay_rate: float = 0.05  # Daily decay rate for view confidence
    
    # Optimization constraints
    allow_short_selling: bool = False
    max_weight: float = 0.3  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    target_volatility: Optional[float] = None  # Target portfolio volatility
    
    # Numerical stability
    regularization: float = 1e-8  # Regularization parameter
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration"""
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if self.tau is not None and self.tau <= 0:
            raise ValueError("tau must be positive if specified")
        if not (0 <= self.min_view_confidence <= 1):
            raise ValueError("min_view_confidence must be between 0 and 1")


class MarketEquilibrium:
    """Calculate market equilibrium expected returns"""
    
    def __init__(self, config: BlackLittermanConfig):
        self.config = config
        
    def calculate_equilibrium_returns(
        self,
        returns_data: pd.DataFrame,
        market_cap_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate market equilibrium expected returns and covariance
        
        Args:
            returns_data: Historical returns data
            market_cap_weights: Market capitalization weights
            
        Returns:
            Tuple of (equilibrium_returns, covariance_matrix)
        """
        
        logger.info("Calculating market equilibrium returns")
        
        # Calculate sample covariance matrix
        cov_matrix = returns_data.cov() * 252  # Annualized
        
        # Add regularization for numerical stability
        cov_matrix += np.eye(len(cov_matrix)) * self.config.regularization
        
        if market_cap_weights is None:
            if self.config.use_market_cap_weights and self.config.market_cap_weights is not None:
                market_cap_weights = self.config.market_cap_weights
            else:
                # Use equal weights as fallback
                n_assets = len(returns_data.columns)
                market_cap_weights = {col: 1.0/n_assets for col in returns_data.columns}
        
        # Convert to weight vector
        weight_vector = pd.Series([
            market_cap_weights.get(asset, 0) for asset in returns_data.columns
        ], index=returns_data.columns)
        
        # Normalize weights
        weight_vector = weight_vector / weight_vector.sum()
        
        if self.config.equilibrium_method == "reverse_optimization":
            # Reverse optimization: μ = λ * Σ * w
            equilibrium_returns = self.config.risk_aversion * cov_matrix @ weight_vector
        else:
            # Historical method
            equilibrium_returns = returns_data.mean() * 252  # Annualized
        
        return equilibrium_returns, cov_matrix
    
    def estimate_tau(
        self,
        returns_data: pd.DataFrame,
        method: Optional[str] = None
    ) -> float:
        """
        Estimate tau parameter for uncertainty in prior
        
        Args:
            returns_data: Historical returns data
            method: Estimation method override
            
        Returns:
            Estimated tau value
        """
        
        method = method or self.config.tau_method
        
        if method == "fixed":
            return self.config.tau or 0.025
        elif method == "1/T":
            return 1.0 / len(returns_data)
        elif method == "0.025":
            return 0.025
        else:  # auto method
            # Use standard deviation of portfolio returns
            n_assets = len(returns_data.columns)
            equal_weights = np.ones(n_assets) / n_assets
            
            portfolio_returns = (returns_data * equal_weights).sum(axis=1)
            portfolio_variance = portfolio_returns.var() * 252
            
            # Tau as fraction of portfolio variance
            return min(0.1, portfolio_variance / 10)


class ViewManager:
    """Manage investor views and confidence levels"""
    
    def __init__(self, config: BlackLittermanConfig):
        self.config = config
        self.active_views: List[Dict[str, Any]] = []
        
    def add_view(
        self,
        picking_matrix: Union[Dict[str, float], List[List[float]]],
        view_returns: Union[float, List[float]],
        confidence: Union[float, List[float]],
        view_type: str = "absolute",
        expiry_date: Optional[datetime] = None,
        view_name: Optional[str] = None
    ) -> bool:
        """
        Add an investor view to the system
        
        Args:
            picking_matrix: Asset weights for the view (P matrix)
            view_returns: Expected return for the view (Q vector)
            confidence: Confidence level(s) for the view
            view_type: "absolute" or "relative"
            expiry_date: When the view expires
            view_name: Optional name for the view
            
        Returns:
            Success flag
        """
        
        if len(self.active_views) >= self.config.max_views:
            logger.warning("Maximum number of views reached, ignoring new view")
            return False
        
        # Convert single values to lists for consistency
        if isinstance(view_returns, (int, float)):
            view_returns = [view_returns]
        if isinstance(confidence, (int, float)):
            confidence = [confidence]
        
        # Validate confidence levels
        if any(c < self.config.min_view_confidence for c in confidence):
            logger.warning("View confidence below minimum threshold, ignoring")
            return False
        
        view = {
            'id': len(self.active_views),
            'name': view_name or f"View_{len(self.active_views)}",
            'picking_matrix': picking_matrix,
            'view_returns': view_returns,
            'confidence': confidence,
            'view_type': view_type,
            'creation_date': datetime.now(),
            'expiry_date': expiry_date,
            'active': True
        }
        
        self.active_views.append(view)
        logger.info(f"Added view: {view['name']}")
        return True
    
    def update_view_confidence(self, decay_days: int = 1) -> None:
        """Update view confidence based on time decay"""
        
        current_date = datetime.now()
        decay_factor = (1 - self.config.view_decay_rate) ** decay_days
        
        for view in self.active_views:
            if not view['active']:
                continue
                
            # Check expiry
            if view['expiry_date'] and current_date > view['expiry_date']:
                view['active'] = False
                continue
            
            # Apply time decay
            view['confidence'] = [
                max(c * decay_factor, self.config.min_view_confidence)
                for c in view['confidence']
            ]
            
            # Deactivate views with very low confidence
            if all(c < self.config.min_view_confidence for c in view['confidence']):
                view['active'] = False
    
    def get_active_views(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get active views in matrix form
        
        Returns:
            Tuple of (P_matrix, Q_vector, Omega_matrix)
        """
        
        active_views = [v for v in self.active_views if v['active']]
        
        if not active_views:
            return np.array([]), np.array([]), np.array([])
        
        P_matrices = []
        Q_vectors = []
        Omega_diagonals = []
        
        for view in active_views:
            # Convert picking matrix to array format
            if isinstance(view['picking_matrix'], dict):
                # Dictionary format: {asset: weight}
                P_row = list(view['picking_matrix'].values())
            else:
                # List format
                P_row = view['picking_matrix']
            
            P_matrices.append(P_row)
            Q_vectors.extend(view['view_returns'])
            
            # Omega (uncertainty matrix) diagonal elements
            # Higher confidence = lower uncertainty
            uncertainties = [1.0 / c for c in view['confidence']]
            Omega_diagonals.extend(uncertainties)
        
        P_matrix = np.array(P_matrices)
        Q_vector = np.array(Q_vectors)
        Omega_matrix = np.diag(Omega_diagonals)
        
        return P_matrix, Q_vector, Omega_matrix
    
    def clear_expired_views(self) -> int:
        """Remove expired and inactive views"""
        
        original_count = len(self.active_views)
        self.active_views = [v for v in self.active_views if v['active']]
        removed_count = original_count - len(self.active_views)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired/inactive views")
        
        return removed_count


class BlackLittermanOptimizer:
    """Main Black-Litterman portfolio optimizer"""
    
    def __init__(self, config: BlackLittermanConfig):
        self.config = config
        self.equilibrium = MarketEquilibrium(config)
        self.view_manager = ViewManager(config)
        self.optimization_history: List[Dict[str, Any]] = []
        
    def optimize_portfolio(
        self,
        returns_data: pd.DataFrame,
        market_cap_weights: Optional[Dict[str, float]] = None,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform Black-Litterman portfolio optimization
        
        Args:
            returns_data: Historical returns data
            market_cap_weights: Market capitalization weights
            current_weights: Current portfolio weights
            
        Returns:
            Optimization results including weights and metrics
        """
        
        logger.info("Starting Black-Litterman optimization", assets=len(returns_data.columns))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'assets': list(returns_data.columns),
            'method': 'black_litterman',
            'optimal_weights': {},
            'expected_returns': {},
            'risk_metrics': {},
            'views_incorporated': 0,
            'optimization_details': {}
        }
        
        try:
            # Step 1: Calculate market equilibrium
            equilibrium_returns, cov_matrix = self.equilibrium.calculate_equilibrium_returns(
                returns_data, market_cap_weights
            )
            
            # Step 2: Estimate tau parameter
            if self.config.tau is None:
                tau = self.equilibrium.estimate_tau(returns_data)
            else:
                tau = self.config.tau
            
            # Step 3: Update view confidence and get active views
            self.view_manager.update_view_confidence()
            P_matrix, Q_vector, Omega_matrix = self.view_manager.get_active_views()
            
            # Step 4: Calculate Black-Litterman expected returns
            if P_matrix.size > 0:
                bl_returns, bl_cov = self._calculate_bl_returns(
                    equilibrium_returns, cov_matrix, tau, P_matrix, Q_vector, Omega_matrix
                )
                result['views_incorporated'] = len(P_matrix)
            else:
                # No views - use equilibrium
                bl_returns = equilibrium_returns
                bl_cov = cov_matrix
                result['views_incorporated'] = 0
            
            # Step 5: Portfolio optimization
            optimal_weights = self._optimize_weights(bl_returns, bl_cov, returns_data.columns)
            
            # Step 6: Calculate portfolio metrics
            risk_metrics = self._calculate_portfolio_metrics(
                optimal_weights, bl_returns, bl_cov
            )
            
            # Store results
            result['optimal_weights'] = {
                asset: float(weight) for asset, weight in zip(returns_data.columns, optimal_weights)
            }
            result['expected_returns'] = {
                asset: float(ret) for asset, ret in bl_returns.items()
            }
            result['risk_metrics'] = risk_metrics
            result['optimization_details'] = {
                'tau': float(tau),
                'risk_aversion': self.config.risk_aversion,
                'regularization': self.config.regularization,
                'equilibrium_method': self.config.equilibrium_method
            }
            
            # Store in history
            self.optimization_history.append({
                'timestamp': result['timestamp'],
                'portfolio_return': risk_metrics['expected_return'],
                'portfolio_volatility': risk_metrics['portfolio_volatility'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'views_count': result['views_incorporated']
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info("Black-Litterman optimization completed",
                       expected_return=risk_metrics['expected_return'],
                       volatility=risk_metrics['portfolio_volatility'],
                       views_used=result['views_incorporated'])
            
        except Exception as e:
            logger.error("Error in Black-Litterman optimization", error=str(e))
            result['error'] = str(e)
        
        return result
    
    def _calculate_bl_returns(
        self,
        equilibrium_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        tau: float,
        P_matrix: np.ndarray,
        Q_vector: np.ndarray,
        Omega_matrix: np.ndarray
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate Black-Litterman adjusted returns and covariance"""
        
        # Convert to numpy arrays
        mu_prior = equilibrium_returns.values
        Sigma = cov_matrix.values
        
        # Black-Litterman formulas
        # M1 = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        tau_sigma_inv = linalg.inv(tau * Sigma)
        P_omega_inv_P = P_matrix.T @ linalg.inv(Omega_matrix) @ P_matrix
        M1 = linalg.inv(tau_sigma_inv + P_omega_inv_P)
        
        # M2 = (τΣ)^(-1)μ + P'Ω^(-1)Q
        M2 = tau_sigma_inv @ mu_prior + P_matrix.T @ linalg.inv(Omega_matrix) @ Q_vector
        
        # New expected returns: μ_new = M1 × M2
        mu_new = M1 @ M2
        
        # New covariance: Σ_new = M1
        sigma_new = M1
        
        # Convert back to pandas
        bl_returns = pd.Series(mu_new, index=equilibrium_returns.index)
        bl_cov = pd.DataFrame(sigma_new, index=cov_matrix.index, columns=cov_matrix.columns)
        
        return bl_returns, bl_cov
    
    def _optimize_weights(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        asset_names: List[str]
    ) -> np.ndarray:
        """Optimize portfolio weights given expected returns and covariance"""
        
        n_assets = len(asset_names)
        
        # Objective function: minimize -μ'w + λ/2 * w'Σw
        def objective(weights):
            portfolio_return = weights @ expected_returns.values
            portfolio_variance = weights @ cov_matrix.values @ weights
            # Negative because we want to maximize utility
            return -(portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Target volatility constraint (if specified)
        if self.config.target_volatility is not None:
            def vol_constraint(weights):
                portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
                return self.config.target_volatility - portfolio_vol
            
            constraints.append({
                'type': 'eq',
                'fun': vol_constraint
            })
        
        # Bounds
        if self.config.allow_short_selling:
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(max(0, self.config.min_weight), self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.convergence_threshold
            }
        )
        
        if not result.success:
            logger.warning("Optimization did not converge, using equal weights")
            return x0
        
        return result.x
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        portfolio_return = weights @ expected_returns.values
        portfolio_variance = weights @ cov_matrix.values @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification metrics
        individual_vols = np.sqrt(np.diag(cov_matrix.values))
        weighted_avg_vol = weights @ individual_vols
        diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1
        
        # Concentration metrics
        herfindahl_index = np.sum(weights ** 2)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'expected_return': float(portfolio_return),
            'portfolio_volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'diversification_ratio': float(diversification_ratio),
            'herfindahl_index': float(herfindahl_index),
            'effective_positions': float(effective_positions),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights))
        }
    
    def add_market_view(
        self,
        asset_view: Dict[str, float],
        expected_return: float,
        confidence: float,
        view_name: Optional[str] = None,
        expiry_days: Optional[int] = None
    ) -> bool:
        """
        Add a market view for specific assets
        
        Args:
            asset_view: Dictionary of {asset: weight} for the view
            expected_return: Expected return for this view
            confidence: Confidence level (0-1)
            view_name: Optional name for the view
            expiry_days: Days until view expires
            
        Returns:
            Success flag
        """
        
        expiry_date = None
        if expiry_days is not None:
            expiry_date = datetime.now() + timedelta(days=expiry_days)
        
        return self.view_manager.add_view(
            picking_matrix=asset_view,
            view_returns=expected_return,
            confidence=confidence,
            view_name=view_name,
            expiry_date=expiry_date
        )
    
    def add_relative_view(
        self,
        asset_long: str,
        asset_short: str,
        expected_outperformance: float,
        confidence: float,
        view_name: Optional[str] = None,
        expiry_days: Optional[int] = None
    ) -> bool:
        """
        Add a relative view (asset A outperforms asset B)
        
        Args:
            asset_long: Asset expected to outperform
            asset_short: Asset expected to underperform
            expected_outperformance: Expected outperformance (annual)
            confidence: Confidence level (0-1)
            view_name: Optional name for the view
            expiry_days: Days until view expires
            
        Returns:
            Success flag
        """
        
        # Create picking matrix for relative view
        asset_view = {asset_long: 1.0, asset_short: -1.0}
        
        return self.add_market_view(
            asset_view=asset_view,
            expected_return=expected_outperformance,
            confidence=confidence,
            view_name=view_name or f"{asset_long}_vs_{asset_short}",
            expiry_days=expiry_days
        )
    
    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """Get Black-Litterman optimization dashboard data"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'risk_aversion': self.config.risk_aversion,
                'tau_method': self.config.tau_method,
                'allow_short_selling': self.config.allow_short_selling,
                'max_weight': self.config.max_weight,
                'equilibrium_method': self.config.equilibrium_method
            },
            'active_views': len([v for v in self.view_manager.active_views if v['active']]),
            'view_details': [
                {
                    'name': v['name'],
                    'confidence': v['confidence'],
                    'creation_date': v['creation_date'].isoformat(),
                    'active': v['active']
                }
                for v in self.view_manager.active_views
            ],
            'optimization_history': self.optimization_history[-20:],
            'performance_stats': {
                'avg_expected_return': np.mean([h['portfolio_return'] for h in self.optimization_history]) if self.optimization_history else 0,
                'avg_volatility': np.mean([h['portfolio_volatility'] for h in self.optimization_history]) if self.optimization_history else 0,
                'avg_sharpe_ratio': np.mean([h['sharpe_ratio'] for h in self.optimization_history]) if self.optimization_history else 0,
                'optimizations_count': len(self.optimization_history)
            }
        }