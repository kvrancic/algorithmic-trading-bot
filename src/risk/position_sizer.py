"""
Position Sizing with Kelly Criterion

Advanced position sizing system featuring:
- Kelly Criterion optimization
- Fractional Kelly implementation
- Multi-asset Kelly optimization
- Risk parity allocation
- Dynamic sizing based on confidence
- Volatility-adjusted sizing
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats, optimize
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing system"""
    
    # Kelly Criterion settings
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25  # Use 25% of full Kelly (fractional Kelly)
    min_kelly_lookback: int = 100  # Minimum trades for Kelly calculation
    kelly_smoothing_window: int = 20  # Smooth Kelly over recent periods
    
    # Risk management constraints
    max_position_size: float = 0.1  # Maximum 10% per position
    min_position_size: float = 0.005  # Minimum 0.5% per position
    max_total_exposure: float = 1.0  # Maximum 100% total exposure
    max_leverage: float = 1.0  # Maximum leverage
    
    # Confidence-based sizing
    enable_confidence_scaling: bool = True
    base_confidence: float = 0.6  # Base confidence level
    confidence_multiplier: float = 2.0  # How much confidence affects sizing
    
    # Volatility adjustment
    enable_volatility_scaling: bool = True
    target_volatility: float = 0.15  # Target 15% annual volatility
    vol_lookback_days: int = 30  # Days for volatility calculation
    
    # Correlation constraints
    enable_correlation_penalty: bool = True
    max_correlation: float = 0.7  # Maximum correlation between positions
    correlation_lookback: int = 60  # Days for correlation calculation
    
    # Risk parity
    enable_risk_parity: bool = False  # Alternative to Kelly
    risk_parity_target_vol: float = 0.05  # Target 5% vol per position
    
    # Dynamic adjustment
    enable_momentum_scaling: bool = True
    momentum_lookback: int = 20  # Days for momentum calculation
    momentum_multiplier: float = 1.5  # Maximum momentum adjustment
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError("kelly_fraction must be between 0 and 1")
        if not (0 < self.max_position_size <= 1):
            raise ValueError("max_position_size must be between 0 and 1")
        if self.max_position_size <= self.min_position_size:
            raise ValueError("max_position_size must be greater than min_position_size")


class KellyCriterion:
    """Kelly Criterion calculation engine"""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        
    def calculate_kelly_fraction(
        self,
        returns: pd.Series,
        probabilities: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate Kelly fraction for a single asset
        
        Args:
            returns: Historical returns series
            probabilities: Win probabilities (optional, estimated if not provided)
            
        Returns:
            Kelly calculation results
        """
        
        if len(returns) < self.config.min_kelly_lookback:
            logger.warning("Insufficient data for Kelly calculation", data_points=len(returns))
            return {
                'kelly_fraction': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'error': 'insufficient_data'
            }
        
        # Clean returns
        returns_clean = returns.dropna()
        
        # Calculate win/loss statistics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        win_rate = len(positive_returns) / len(returns_clean)
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
        
        # Standard Kelly formula: f* = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win_rate, q = 1-p
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            b = avg_win / avg_loss  # Odds ratio
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly
        adjusted_kelly = kelly_fraction * self.config.kelly_fraction
        
        # Additional Kelly statistics
        sharpe_ratio = returns_clean.mean() / returns_clean.std() if returns_clean.std() > 0 else 0
        growth_rate = self._calculate_growth_rate(returns_clean, adjusted_kelly)
        
        return {
            'kelly_fraction': float(kelly_fraction),
            'adjusted_kelly': float(adjusted_kelly),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'odds_ratio': float(avg_win / avg_loss if avg_loss > 0 else 0),
            'sharpe_ratio': float(sharpe_ratio),
            'growth_rate': float(growth_rate),
            'data_points': len(returns_clean)
        }
    
    def calculate_multivariate_kelly(
        self,
        returns_matrix: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal Kelly fractions for multiple assets
        
        Args:
            returns_matrix: Returns for multiple assets
            expected_returns: Expected returns (optional, historical mean used if not provided)
            
        Returns:
            Multivariate Kelly results
        """
        
        if len(returns_matrix) < self.config.min_kelly_lookback:
            return {'error': 'insufficient_data'}
        
        # Clean data
        returns_clean = returns_matrix.dropna()
        
        if expected_returns is None:
            expected_returns = returns_clean.mean()
        
        # Calculate covariance matrix
        cov_matrix = returns_clean.cov()
        
        # Handle singular matrix
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            inv_cov = np.linalg.pinv(cov_matrix.values)
        
        # Multivariate Kelly formula: f* = Σ^(-1) * μ
        # where Σ is covariance matrix, μ is expected returns
        kelly_fractions = inv_cov @ expected_returns.values
        
        # Apply fractional Kelly
        kelly_fractions *= self.config.kelly_fraction
        
        # Convert to dictionary
        kelly_dict = {
            asset: float(fraction) 
            for asset, fraction in zip(returns_clean.columns, kelly_fractions)
        }
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(kelly_fractions * expected_returns.values)
        portfolio_variance = kelly_fractions.T @ cov_matrix.values @ kelly_fractions
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return {
            'kelly_fractions': kelly_dict,
            'portfolio_return': float(portfolio_return),
            'portfolio_volatility': float(portfolio_volatility),
            'portfolio_sharpe': float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0,
            'total_allocation': float(np.sum(np.abs(kelly_fractions))),
            'covariance_condition': float(np.linalg.cond(cov_matrix.values))
        }
    
    def _calculate_growth_rate(self, returns: pd.Series, fraction: float) -> float:
        """Calculate expected geometric growth rate"""
        
        if fraction <= 0:
            return 0.0
        
        # Expected log growth rate: E[log(1 + f*R)]
        # Approximated using second-order expansion
        mean_return = returns.mean()
        variance = returns.var()
        
        growth_rate = fraction * mean_return - 0.5 * fraction**2 * variance
        return growth_rate


class RiskParityAllocator:
    """Risk parity allocation as alternative to Kelly"""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        
    def calculate_risk_parity_weights(
        self,
        returns_matrix: pd.DataFrame,
        target_volatilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate risk parity weights
        
        Args:
            returns_matrix: Asset returns
            target_volatilities: Target volatility for each asset
            
        Returns:
            Risk parity allocation
        """
        
        if len(returns_matrix) < 20:
            return {'error': 'insufficient_data'}
        
        returns_clean = returns_matrix.dropna()
        
        # Calculate individual asset volatilities
        asset_volatilities = returns_clean.std() * np.sqrt(252)  # Annualized
        
        if target_volatilities is None:
            target_vol = self.config.risk_parity_target_vol
            target_volatilities = {asset: target_vol for asset in returns_clean.columns}
        
        # Risk parity weights: inverse of volatility
        weights = {}
        for asset in returns_clean.columns:
            if asset_volatilities[asset] > 0:
                weights[asset] = target_volatilities.get(asset, target_vol) / asset_volatilities[asset]
            else:
                weights[asset] = 0.0
        
        # Normalize weights
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {asset: w / total_weight for asset, w in weights.items()}
        
        # Calculate portfolio metrics
        weight_array = np.array([weights[asset] for asset in returns_clean.columns])
        cov_matrix = returns_clean.cov()
        
        portfolio_volatility = np.sqrt(weight_array.T @ cov_matrix.values @ weight_array) * np.sqrt(252)
        
        # Risk contributions
        marginal_risk = cov_matrix.values @ weight_array * np.sqrt(252)
        risk_contributions = {
            asset: float(weights[asset] * marginal_risk[i])
            for i, asset in enumerate(returns_clean.columns)
        }
        
        return {
            'weights': weights,
            'portfolio_volatility': float(portfolio_volatility),
            'risk_contributions': risk_contributions,
            'target_volatilities': target_volatilities,
            'individual_volatilities': asset_volatilities.to_dict()
        }


class PositionSizer:
    """Main position sizing engine"""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.kelly_calculator = KellyCriterion(config)
        self.risk_parity_allocator = RiskParityAllocator(config)
        self.sizing_history: List[Dict[str, Any]] = []
        
    def calculate_position_sizes(
        self,
        signals: Dict[str, float],
        returns_data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        portfolio_value: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Calculate optimal position sizes for given signals
        
        Args:
            signals: Trading signals {symbol: signal_strength}
            returns_data: Historical returns data
            current_positions: Current position weights
            confidence_scores: Signal confidence scores
            portfolio_value: Total portfolio value
            
        Returns:
            Position sizing recommendations
        """
        
        logger.info("Calculating position sizes", 
                   signals=len(signals), portfolio_value=portfolio_value)
        
        if current_positions is None:
            current_positions = {}
        if confidence_scores is None:
            confidence_scores = {symbol: self.config.base_confidence for symbol in signals}
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'signals': signals,
            'portfolio_value': portfolio_value,
            'sizing_method': 'kelly' if self.config.use_kelly_criterion else 'risk_parity',
            'position_sizes': {},
            'risk_metrics': {},
            'adjustments': {},
            'constraints_applied': []
        }
        
        try:
            # Get relevant returns data
            signal_symbols = list(signals.keys())
            available_symbols = [s for s in signal_symbols if s in returns_data.columns]
            
            if not available_symbols:
                logger.warning("No historical data available for any signals")
                return result
            
            returns_subset = returns_data[available_symbols].dropna()
            
            if self.config.use_kelly_criterion:
                # Kelly-based sizing
                sizes = self._calculate_kelly_sizes(
                    signals, returns_subset, confidence_scores
                )
            else:
                # Risk parity sizing
                sizes = self._calculate_risk_parity_sizes(
                    signals, returns_subset
                )
            
            result['position_sizes'] = sizes
            
            # Apply adjustments and constraints
            adjusted_sizes = self._apply_adjustments(
                sizes, returns_subset, signals, confidence_scores
            )
            
            result['position_sizes'] = adjusted_sizes['sizes']
            result['adjustments'] = adjusted_sizes['adjustments']
            result['constraints_applied'] = adjusted_sizes['constraints']
            
            # Calculate risk metrics
            result['risk_metrics'] = self._calculate_risk_metrics(
                adjusted_sizes['sizes'], returns_subset
            )
            
            # Record in history
            self.sizing_history.append({
                'timestamp': result['timestamp'],
                'total_allocation': sum(abs(s) for s in adjusted_sizes['sizes'].values()),
                'max_position': max(abs(s) for s in adjusted_sizes['sizes'].values()) if adjusted_sizes['sizes'] else 0,
                'num_positions': len(adjusted_sizes['sizes'])
            })
            
            # Keep only recent history
            if len(self.sizing_history) > 1000:
                self.sizing_history = self.sizing_history[-1000:]
            
            logger.info("Position sizing completed",
                       total_allocation=result['risk_metrics'].get('total_allocation', 0),
                       max_position=result['risk_metrics'].get('max_position_size', 0))
            
        except Exception as e:
            logger.error("Error in position sizing calculation", error=str(e))
            result['error'] = str(e)
        
        return result
    
    def _calculate_kelly_sizes(
        self,
        signals: Dict[str, float],
        returns_data: pd.DataFrame,
        confidence_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate Kelly-based position sizes"""
        
        if self.config.min_kelly_lookback * 2 <= len(returns_data):
            # Use multivariate Kelly if enough data
            expected_returns = {}
            for symbol in signals:
                if symbol in returns_data.columns:
                    # Weight expected return by signal strength
                    hist_return = returns_data[symbol].mean()
                    expected_returns[symbol] = hist_return * signals[symbol]
            
            expected_returns_series = pd.Series(expected_returns)
            kelly_result = self.kelly_calculator.calculate_multivariate_kelly(
                returns_data, expected_returns_series
            )
            
            if 'kelly_fractions' in kelly_result:
                return kelly_result['kelly_fractions']
        
        # Fallback to individual Kelly calculations
        sizes = {}
        for symbol, signal_strength in signals.items():
            if symbol in returns_data.columns:
                returns = returns_data[symbol].dropna()
                
                # Adjust returns by signal strength
                if signal_strength != 0:
                    adjusted_returns = returns * signal_strength
                    kelly_result = self.kelly_calculator.calculate_kelly_fraction(adjusted_returns)
                    sizes[symbol] = kelly_result['adjusted_kelly']
                else:
                    sizes[symbol] = 0.0
            else:
                sizes[symbol] = 0.0
        
        return sizes
    
    def _calculate_risk_parity_sizes(
        self,
        signals: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk parity based position sizes"""
        
        rp_result = self.risk_parity_allocator.calculate_risk_parity_weights(returns_data)
        
        if 'weights' in rp_result:
            # Scale by signal strength
            sizes = {}
            for symbol, signal_strength in signals.items():
                if symbol in rp_result['weights']:
                    sizes[symbol] = rp_result['weights'][symbol] * signal_strength
                else:
                    sizes[symbol] = 0.0
            return sizes
        
        # Fallback to equal weight
        equal_weight = 1.0 / len(signals) if signals else 0.0
        return {symbol: equal_weight * signal for symbol, signal in signals.items()}
    
    def _apply_adjustments(
        self,
        base_sizes: Dict[str, float], 
        returns_data: pd.DataFrame,
        signals: Dict[str, float],
        confidence_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply various adjustments to base position sizes"""
        
        adjusted_sizes = base_sizes.copy()
        adjustments = {}
        constraints = []
        
        # Confidence scaling
        if self.config.enable_confidence_scaling:
            for symbol in adjusted_sizes:
                confidence = confidence_scores.get(symbol, self.config.base_confidence)
                confidence_adj = (confidence / self.config.base_confidence) ** self.config.confidence_multiplier
                adjusted_sizes[symbol] *= confidence_adj
                adjustments[f"{symbol}_confidence"] = confidence_adj
        
        # Volatility scaling
        if self.config.enable_volatility_scaling:
            vol_adjustments = self._calculate_volatility_adjustments(
                adjusted_sizes, returns_data
            )
            for symbol, vol_adj in vol_adjustments.items():
                adjusted_sizes[symbol] *= vol_adj
                adjustments[f"{symbol}_volatility"] = vol_adj
        
        # Momentum scaling
        if self.config.enable_momentum_scaling:
            momentum_adjustments = self._calculate_momentum_adjustments(
                adjusted_sizes, returns_data
            )
            for symbol, mom_adj in momentum_adjustments.items():
                adjusted_sizes[symbol] *= mom_adj
                adjustments[f"{symbol}_momentum"] = mom_adj
        
        # Apply position size constraints
        for symbol in list(adjusted_sizes.keys()):
            original_size = adjusted_sizes[symbol]
            
            # Minimum/maximum position size
            if abs(adjusted_sizes[symbol]) < self.config.min_position_size:
                if adjusted_sizes[symbol] != 0:
                    adjusted_sizes[symbol] = 0.0
                    constraints.append(f"{symbol}: below minimum size")
            elif abs(adjusted_sizes[symbol]) > self.config.max_position_size:
                adjusted_sizes[symbol] = np.sign(adjusted_sizes[symbol]) * self.config.max_position_size
                constraints.append(f"{symbol}: capped at maximum size")
        
        # Total exposure constraint
        total_exposure = sum(abs(s) for s in adjusted_sizes.values())
        if total_exposure > self.config.max_total_exposure:
            scale_factor = self.config.max_total_exposure / total_exposure
            adjusted_sizes = {symbol: size * scale_factor for symbol, size in adjusted_sizes.items()}
            constraints.append(f"Total exposure scaled by {scale_factor:.3f}")
        
        # Correlation constraints
        if self.config.enable_correlation_penalty:
            correlation_adjustments = self._apply_correlation_penalty(
                adjusted_sizes, returns_data
            )
            for symbol, corr_adj in correlation_adjustments.items():
                adjusted_sizes[symbol] *= corr_adj
                adjustments[f"{symbol}_correlation"] = corr_adj
        
        return {
            'sizes': adjusted_sizes,
            'adjustments': adjustments,
            'constraints': constraints
        }
    
    def _calculate_volatility_adjustments(
        self,
        sizes: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate volatility-based size adjustments"""
        
        adjustments = {}
        
        for symbol in sizes:
            if symbol in returns_data.columns:
                returns = returns_data[symbol].dropna()
                if len(returns) >= self.config.vol_lookback_days:
                    recent_returns = returns.tail(self.config.vol_lookback_days)
                    volatility = recent_returns.std() * np.sqrt(252)  # Annualized
                    
                    # Inverse volatility scaling
                    vol_adjustment = self.config.target_volatility / volatility if volatility > 0 else 1.0
                    vol_adjustment = np.clip(vol_adjustment, 0.1, 3.0)  # Bound adjustment
                    adjustments[symbol] = vol_adjustment
                else:
                    adjustments[symbol] = 1.0
            else:
                adjustments[symbol] = 1.0
        
        return adjustments
    
    def _calculate_momentum_adjustments(
        self,
        sizes: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate momentum-based size adjustments"""
        
        adjustments = {}
        
        for symbol in sizes:
            if symbol in returns_data.columns:
                returns = returns_data[symbol].dropna()
                if len(returns) >= self.config.momentum_lookback:
                    recent_returns = returns.tail(self.config.momentum_lookback)
                    
                    # Simple momentum: average return
                    momentum = recent_returns.mean()
                    
                    # Convert to adjustment factor
                    # Positive momentum increases size, negative decreases
                    momentum_adj = 1.0 + (momentum * 252 * 10)  # Scale to reasonable range
                    momentum_adj = np.clip(momentum_adj, 1/self.config.momentum_multiplier, self.config.momentum_multiplier)
                    adjustments[symbol] = momentum_adj
                else:
                    adjustments[symbol] = 1.0
            else:
                adjustments[symbol] = 1.0
        
        return adjustments
    
    def _apply_correlation_penalty(
        self,
        sizes: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Apply correlation penalty to reduce concentration"""
        
        adjustments = {symbol: 1.0 for symbol in sizes}
        
        symbols = [s for s in sizes if s in returns_data.columns and sizes[s] != 0]
        if len(symbols) < 2:
            return adjustments
        
        # Calculate correlation matrix
        correlation_data = returns_data[symbols].dropna()
        if len(correlation_data) < self.config.correlation_lookback:
            return adjustments
        
        recent_data = correlation_data.tail(self.config.correlation_lookback)
        corr_matrix = recent_data.corr()
        
        # Apply penalty for high correlations
        for i, symbol1 in enumerate(symbols):
            penalty = 1.0
            
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    correlation = abs(corr_matrix.loc[symbol1, symbol2])
                    if correlation > self.config.max_correlation:
                        # Apply penalty proportional to excess correlation
                        excess_corr = correlation - self.config.max_correlation
                        penalty_factor = 1.0 - (excess_corr * 0.5)  # Up to 50% penalty
                        penalty = min(penalty, penalty_factor)
            
            adjustments[symbol1] = penalty
        
        return adjustments
    
    def _calculate_risk_metrics(
        self,
        sizes: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk metrics for the portfolio"""
        
        metrics = {}
        
        # Basic allocation metrics
        total_allocation = sum(abs(s) for s in sizes.values())
        max_position = max(abs(s) for s in sizes.values()) if sizes else 0
        num_positions = len([s for s in sizes.values() if s != 0])
        
        metrics.update({
            'total_allocation': total_allocation,
            'max_position_size': max_position,
            'num_positions': num_positions,
            'leverage': total_allocation
        })
        
        # Portfolio volatility if possible
        active_symbols = [s for s in sizes if s != 0 and s in returns_data.columns]
        if len(active_symbols) >= 2:
            weights = np.array([sizes[s] for s in active_symbols])
            returns_subset = returns_data[active_symbols].dropna()
            
            if len(returns_subset) > 30:
                cov_matrix = returns_subset.cov()
                portfolio_variance = weights.T @ cov_matrix.values @ weights
                portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
                
                metrics['portfolio_volatility'] = float(portfolio_volatility)
                metrics['risk_concentration'] = float(np.sum(weights**2))  # Herfindahl index
        
        return metrics
    
    def get_sizing_dashboard(self) -> Dict[str, Any]:
        """Get position sizing dashboard data"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'method': 'kelly' if self.config.use_kelly_criterion else 'risk_parity',
                'kelly_fraction': self.config.kelly_fraction,
                'max_position_size': self.config.max_position_size,
                'max_total_exposure': self.config.max_total_exposure,
                'confidence_scaling': self.config.enable_confidence_scaling,
                'volatility_scaling': self.config.enable_volatility_scaling,
                'momentum_scaling': self.config.enable_momentum_scaling
            },
            'recent_history': self.sizing_history[-20:] if len(self.sizing_history) > 20 else self.sizing_history,
            'statistics': {
                'avg_total_allocation': np.mean([h['total_allocation'] for h in self.sizing_history]) if self.sizing_history else 0,
                'avg_max_position': np.mean([h['max_position'] for h in self.sizing_history]) if self.sizing_history else 0,
                'avg_num_positions': np.mean([h['num_positions'] for h in self.sizing_history]) if self.sizing_history else 0
            }
        }