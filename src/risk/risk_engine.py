"""
Enterprise Risk Engine

Comprehensive risk management system for algorithmic trading with:
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Real-time risk monitoring and alerting
- Portfolio-level risk controls
- Dynamic risk limit adjustment
- Risk attribution analysis
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import structlog
from pathlib import Path
import json
import warnings

logger = structlog.get_logger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk engine"""
    
    # VaR/CVaR settings
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    var_window: int = 252  # Trading days for VaR calculation
    var_method: str = "historical"  # "historical", "parametric", "monte_carlo"
    
    # Risk limits
    max_portfolio_var: float = 0.02  # Maximum 2% daily VaR
    max_position_weight: float = 0.1  # Maximum 10% position size
    max_sector_exposure: float = 0.3  # Maximum 30% sector exposure
    max_leverage: float = 1.0  # Maximum leverage
    max_correlation: float = 0.7  # Maximum correlation between positions
    
    # Drawdown controls
    max_daily_drawdown: float = 0.02  # Maximum 2% daily drawdown
    max_total_drawdown: float = 0.2  # Maximum 20% total drawdown
    
    # Monitoring settings
    risk_check_frequency: int = 15  # Minutes between risk checks
    alert_threshold_multiplier: float = 0.8  # Alert when 80% of limit reached
    enable_circuit_breakers: bool = True
    
    # Monte Carlo settings (if using MC VaR)
    mc_simulations: int = 10000
    mc_time_horizon: int = 1  # Days
    
    def __post_init__(self):
        """Validate configuration"""
        if self.var_method not in ["historical", "parametric", "monte_carlo"]:
            raise ValueError("var_method must be 'historical', 'parametric', or 'monte_carlo'")
        
        if not (0 < self.max_portfolio_var < 1):
            raise ValueError("max_portfolio_var must be between 0 and 1")


class VaRCalculator:
    """Value at Risk calculation engine"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        
    def calculate_var(
        self, 
        returns: Union[pd.Series, pd.DataFrame],
        confidence_level: float = 0.05,
        method: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk using specified method
        
        Args:
            returns: Historical returns (Series for single asset, DataFrame for portfolio)
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            method: VaR calculation method override
            
        Returns:
            Dictionary with VaR calculations
        """
        method = method or self.config.var_method
        
        if isinstance(returns, pd.Series):
            returns = returns.dropna()
        else:
            returns = returns.dropna()
        
        if len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation", data_points=len(returns))
            return {"var": 0.0, "cvar": 0.0, "method": method, "data_points": len(returns)}
        
        # Limit data to specified window
        if len(returns) > self.config.var_window:
            returns = returns.tail(self.config.var_window)
        
        result = {
            "confidence_level": confidence_level,
            "method": method,
            "data_points": len(returns),
            "observation_period": len(returns)
        }
        
        if method == "historical":
            result.update(self._historical_var(returns, confidence_level))
        elif method == "parametric":
            result.update(self._parametric_var(returns, confidence_level))
        elif method == "monte_carlo":
            result.update(self._monte_carlo_var(returns, confidence_level))
        
        return result
    
    def _historical_var(self, returns: Union[pd.Series, pd.DataFrame], confidence_level: float) -> Dict[str, float]:
        """Historical VaR calculation"""
        
        if isinstance(returns, pd.DataFrame):
            # Portfolio returns
            portfolio_returns = returns.sum(axis=1)
        else:
            portfolio_returns = returns
        
        # Calculate VaR as percentile
        var = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Calculate CVaR (Conditional VaR / Expected Shortfall)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        cvar = cvar_returns.mean() if len(cvar_returns) > 0 else var
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "worst_case": float(portfolio_returns.min()),
            "best_case": float(portfolio_returns.max())
        }
    
    def _parametric_var(self, returns: Union[pd.Series, pd.DataFrame], confidence_level: float) -> Dict[str, float]:
        """Parametric VaR assuming normal distribution"""
        
        if isinstance(returns, pd.DataFrame):
            portfolio_returns = returns.sum(axis=1)
        else:
            portfolio_returns = returns
        
        mean = portfolio_returns.mean()
        std = portfolio_returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(confidence_level)
        
        # VaR calculation
        var = mean + z_score * std
        
        # CVaR for normal distribution
        # CVaR = μ + σ * φ(Φ^(-1)(α)) / α
        phi_z = stats.norm.pdf(z_score)
        cvar = mean + std * phi_z / confidence_level
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "mean": float(mean),
            "std": float(std),
            "z_score": float(z_score)
        }
    
    def _monte_carlo_var(self, returns: Union[pd.Series, pd.DataFrame], confidence_level: float) -> Dict[str, float]:
        """Monte Carlo VaR simulation"""
        
        if isinstance(returns, pd.DataFrame):
            portfolio_returns = returns.sum(axis=1)
        else:
            portfolio_returns = returns
        
        mean = portfolio_returns.mean()
        std = portfolio_returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean, std, self.config.mc_simulations
        )
        
        # Calculate VaR and CVaR from simulations
        var = np.percentile(simulated_returns, confidence_level * 100)
        cvar_returns = simulated_returns[simulated_returns <= var]
        cvar = cvar_returns.mean() if len(cvar_returns) > 0 else var
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "simulations": self.config.mc_simulations,
            "simulated_mean": float(np.mean(simulated_returns)),
            "simulated_std": float(np.std(simulated_returns))
        }


class RiskMetrics:
    """Calculate comprehensive risk metrics"""
    
    @staticmethod
    def calculate_portfolio_risk(
        positions: Dict[str, float],
        returns: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            positions: Dictionary of {symbol: weight}
            returns: Historical returns DataFrame
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Portfolio risk metrics
        """
        
        if correlation_matrix is None:
            correlation_matrix = returns.corr()
        
        # Portfolio weights
        weights = np.array([positions.get(symbol, 0) for symbol in returns.columns])
        weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else weights
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic metrics
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_return = portfolio_returns.mean() * 252
        
        # Risk decomposition
        asset_contributions = {}
        for i, symbol in enumerate(returns.columns):
            if weights[i] != 0:
                # Marginal risk contribution
                marginal_risk = (correlation_matrix.iloc[i] * weights).sum() * returns[symbol].std() ** 2
                asset_contributions[symbol] = {
                    'weight': float(weights[i]),
                    'marginal_risk': float(marginal_risk),
                    'risk_contribution': float(marginal_risk * weights[i])
                }
        
        # Concentration metrics
        herfindahl_index = np.sum(weights ** 2)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'portfolio_volatility': float(portfolio_vol),
            'portfolio_return': float(portfolio_return),
            'sharpe_ratio': float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0,
            'herfindahl_index': float(herfindahl_index),
            'effective_positions': float(effective_positions),
            'max_weight': float(np.max(np.abs(weights))),
            'asset_contributions': asset_contributions,
            'total_leverage': float(np.sum(np.abs(weights)))
        }
    
    @staticmethod
    def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics"""
        
        if len(returns) == 0:
            return {}
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative.expanding().max()
        
        # Drawdown series
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append((start_idx, i - 1))
                start_idx = None
        
        # Handle ongoing drawdown
        if start_idx is not None:
            drawdown_periods.append((start_idx, len(in_drawdown) - 1))
        
        # Drawdown statistics
        drawdown_durations = [end - start + 1 for start, end in drawdown_periods]
        avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'current_drawdown': float(current_drawdown),
            'avg_drawdown': float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0,
            'drawdown_frequency': len(drawdown_periods),
            'avg_drawdown_duration': float(avg_drawdown_duration),
            'max_drawdown_duration': int(max_drawdown_duration),
            'recovery_factor': float(returns.mean() * 252 / abs(max_drawdown)) if max_drawdown != 0 else 0,
            'ulcer_index': float(np.sqrt(np.mean(drawdown ** 2)))
        }


class RiskEngine:
    """Main risk management engine"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.var_calculator = VaRCalculator(config)
        self.risk_alerts: List[Dict[str, Any]] = []
        self.risk_history: List[Dict[str, Any]] = []
        
    def assess_portfolio_risk(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment
        
        Args:
            positions: Current positions {symbol: weight/quantity}
            returns: Historical returns data
            prices: Current price data (optional)
            
        Returns:
            Complete risk assessment
        """
        
        logger.info("Starting portfolio risk assessment", positions=len(positions))
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'positions': positions,
            'risk_limits': self._get_risk_limits(),
            'var_analysis': {},
            'portfolio_metrics': {},
            'drawdown_analysis': {},
            'limit_utilization': {},
            'risk_alerts': [],
            'overall_risk_score': 0.0
        }
        
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions, returns)
            
            # VaR analysis for all confidence levels
            for confidence_level in self.config.var_confidence_levels:
                var_result = self.var_calculator.calculate_var(
                    portfolio_returns, confidence_level
                )
                assessment['var_analysis'][f'var_{int((1-confidence_level)*100)}'] = var_result
            
            # Portfolio risk metrics
            assessment['portfolio_metrics'] = RiskMetrics.calculate_portfolio_risk(
                positions, returns
            )
            
            # Drawdown analysis
            assessment['drawdown_analysis'] = RiskMetrics.calculate_drawdown_metrics(
                portfolio_returns
            )
            
            # Check risk limits
            assessment['limit_utilization'] = self._check_risk_limits(assessment)
            
            # Generate risk alerts
            assessment['risk_alerts'] = self._generate_risk_alerts(assessment)
            
            # Calculate overall risk score (0-100, higher = riskier)
            assessment['overall_risk_score'] = self._calculate_risk_score(assessment)
            
            # Store in history
            self.risk_history.append({
                'timestamp': assessment['timestamp'],
                'risk_score': assessment['overall_risk_score'],
                'var_95': assessment['var_analysis'].get('var_95', {}).get('var', 0),
                'max_drawdown': assessment['drawdown_analysis'].get('max_drawdown', 0),
                'portfolio_vol': assessment['portfolio_metrics'].get('portfolio_volatility', 0)
            })
            
            # Keep only last 1000 records
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            logger.info("Portfolio risk assessment completed",
                       risk_score=assessment['overall_risk_score'],
                       alerts=len(assessment['risk_alerts']))
            
        except Exception as e:
            logger.error("Error in portfolio risk assessment", error=str(e))
            assessment['error'] = str(e)
        
        return assessment
    
    def _calculate_portfolio_returns(
        self, 
        positions: Dict[str, float], 
        returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns from positions and asset returns"""
        
        # Normalize weights
        total_weight = sum(abs(w) for w in positions.values())
        if total_weight == 0:
            return pd.Series(index=returns.index, data=0.0)
        
        normalized_positions = {k: v/total_weight for k, v in positions.items()}
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(index=returns.index, data=0.0)
        
        for symbol, weight in normalized_positions.items():
            if symbol in returns.columns:
                portfolio_returns += weight * returns[symbol].fillna(0)
        
        return portfolio_returns.dropna()
    
    def _get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits"""
        return {
            'max_portfolio_var': self.config.max_portfolio_var,
            'max_position_weight': self.config.max_position_weight,
            'max_sector_exposure': self.config.max_sector_exposure,
            'max_leverage': self.config.max_leverage,
            'max_daily_drawdown': self.config.max_daily_drawdown,
            'max_total_drawdown': self.config.max_total_drawdown,
            'max_correlation': self.config.max_correlation
        }
    
    def _check_risk_limits(self, assessment: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Check current risk against limits"""
        
        utilization = {}
        
        # VaR limit utilization
        var_95 = assessment['var_analysis'].get('var_95', {}).get('var', 0)
        utilization['var_limit'] = {
            'current': abs(var_95),
            'limit': self.config.max_portfolio_var,
            'utilization': abs(var_95) / self.config.max_portfolio_var if self.config.max_portfolio_var > 0 else 0,
            'breach': abs(var_95) > self.config.max_portfolio_var
        }
        
        # Position weight limits
        max_weight = assessment['portfolio_metrics'].get('max_weight', 0)
        utilization['position_limit'] = {
            'current': max_weight,
            'limit': self.config.max_position_weight,
            'utilization': max_weight / self.config.max_position_weight,
            'breach': max_weight > self.config.max_position_weight
        }
        
        # Leverage limits
        total_leverage = assessment['portfolio_metrics'].get('total_leverage', 0)
        utilization['leverage_limit'] = {
            'current': total_leverage,
            'limit': self.config.max_leverage,
            'utilization': total_leverage / self.config.max_leverage,
            'breach': total_leverage > self.config.max_leverage
        }
        
        # Drawdown limits
        current_drawdown = assessment['drawdown_analysis'].get('current_drawdown', 0)
        utilization['drawdown_limit'] = {
            'current': abs(current_drawdown),
            'limit': self.config.max_total_drawdown,
            'utilization': abs(current_drawdown) / self.config.max_total_drawdown,
            'breach': abs(current_drawdown) > self.config.max_total_drawdown
        }
        
        return utilization
    
    def _generate_risk_alerts(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on assessment"""
        
        alerts = []
        limit_util = assessment['limit_utilization']
        threshold = self.config.alert_threshold_multiplier
        
        # Check each limit
        for limit_name, limit_data in limit_util.items():
            utilization = limit_data['utilization']
            is_breach = limit_data['breach']
            
            if is_breach:
                alerts.append({
                    'severity': 'CRITICAL',
                    'type': 'LIMIT_BREACH',
                    'limit': limit_name,
                    'message': f"{limit_name} breached: {utilization:.1%} of limit",
                    'current': limit_data['current'],
                    'limit_value': limit_data['limit'],
                    'timestamp': datetime.now().isoformat()
                })
            elif utilization > threshold:
                alerts.append({
                    'severity': 'WARNING',
                    'type': 'LIMIT_APPROACH',
                    'limit': limit_name,
                    'message': f"{limit_name} approaching: {utilization:.1%} of limit",
                    'current': limit_data['current'],
                    'limit_value': limit_data['limit'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Risk score alerts
        risk_score = assessment.get('overall_risk_score', 0)
        if risk_score > 80:
            alerts.append({
                'severity': 'HIGH',
                'type': 'HIGH_RISK_SCORE',
                'message': f"Portfolio risk score elevated: {risk_score:.1f}/100",
                'risk_score': risk_score,
                'timestamp': datetime.now().isoformat()
            })
        
        # Store alerts
        self.risk_alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.risk_alerts = [
            alert for alert in self.risk_alerts 
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        return alerts
    
    def _calculate_risk_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100, higher = riskier)"""
        
        score = 0.0
        
        # VaR contribution (0-30 points)
        var_95 = assessment['var_analysis'].get('var_95', {}).get('var', 0)
        var_score = min(30, abs(var_95) / self.config.max_portfolio_var * 30)
        score += var_score
        
        # Volatility contribution (0-20 points)
        vol = assessment['portfolio_metrics'].get('portfolio_volatility', 0)
        vol_score = min(20, vol * 100)  # Assuming vol is in decimal form
        score += vol_score
        
        # Concentration contribution (0-20 points)
        herfindahl = assessment['portfolio_metrics'].get('herfindahl_index', 0)
        concentration_score = min(20, herfindahl * 20)
        score += concentration_score
        
        # Drawdown contribution (0-20 points)
        current_dd = assessment['drawdown_analysis'].get('current_drawdown', 0)
        dd_score = min(20, abs(current_dd) / self.config.max_total_drawdown * 20)
        score += dd_score
        
        # Leverage contribution (0-10 points)
        leverage = assessment['portfolio_metrics'].get('total_leverage', 0)
        leverage_score = min(10, leverage / self.config.max_leverage * 10)
        score += leverage_score
        
        return min(100.0, score)
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get risk monitoring dashboard data"""
        
        recent_alerts = [
            alert for alert in self.risk_alerts
            if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=1)
        ]
        
        return {
            'current_time': datetime.now().isoformat(),
            'recent_alerts': recent_alerts,
            'alert_summary': {
                'critical': len([a for a in recent_alerts if a['severity'] == 'CRITICAL']),
                'high': len([a for a in recent_alerts if a['severity'] == 'HIGH']),
                'warning': len([a for a in recent_alerts if a['severity'] == 'WARNING'])
            },
            'risk_history': self.risk_history[-24:] if len(self.risk_history) > 24 else self.risk_history,
            'system_status': 'OPERATIONAL' if len([a for a in recent_alerts if a['severity'] == 'CRITICAL']) == 0 else 'ALERT',
            'config': {
                'var_method': self.config.var_method,
                'max_portfolio_var': self.config.max_portfolio_var,
                'max_drawdown': self.config.max_total_drawdown,
                'max_leverage': self.config.max_leverage
            }
        }
    
    def export_risk_report(self, file_path: Path) -> None:
        """Export comprehensive risk report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'recent_assessments': self.risk_history[-100:],
            'recent_alerts': self.risk_alerts[-50:],
            'statistics': {
                'total_assessments': len(self.risk_history),
                'total_alerts': len(self.risk_alerts),
                'avg_risk_score': np.mean([h['risk_score'] for h in self.risk_history]) if self.risk_history else 0
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Risk report exported", file_path=str(file_path))