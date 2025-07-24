"""
Dynamic Portfolio Rebalancing System

Advanced dynamic rebalancing with:
- Threshold-based rebalancing triggers
- Time-based periodic rebalancing
- Volatility-adjusted rebalancing frequencies
- Transaction cost optimization
- Market regime-aware rebalancing
- Risk-budgeting maintenance rebalancing
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from scipy import optimize
import structlog

logger = structlog.get_logger(__name__)


class RebalancingTrigger(Enum):
    """Rebalancing trigger types"""
    THRESHOLD = "threshold"
    TIME_BASED = "time_based"
    VOLATILITY = "volatility"
    RISK_BUDGET = "risk_budget"
    REGIME_CHANGE = "regime_change"
    TRANSACTION_COST = "transaction_cost"


@dataclass
class RebalancingConfig:
    """Configuration for dynamic rebalancing"""
    
    # Threshold rebalancing
    weight_threshold: float = 0.05  # 5% deviation triggers rebalancing
    absolute_threshold: float = 0.02  # 2% absolute threshold
    
    # Time-based rebalancing
    rebalancing_frequency: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    min_days_between_rebalancing: int = 7  # Minimum days between rebalances
    
    # Volatility-based rebalancing
    enable_volatility_adjustment: bool = True
    vol_lookback_days: int = 30
    high_vol_threshold: float = 0.25  # 25% annualized vol threshold
    low_vol_threshold: float = 0.10   # 10% annualized vol threshold
    vol_adjustment_factor: float = 0.5  # Frequency adjustment factor
    
    # Risk budget rebalancing
    enable_risk_budget_monitoring: bool = True
    risk_budget_threshold: float = 0.03  # 3% risk budget deviation
    
    # Transaction cost considerations
    enable_transaction_cost_optimization: bool = True
    transaction_cost_rate: float = 0.001  # 0.1% per transaction
    min_rebalancing_size: float = 0.01    # 1% minimum position change
    
    # Market regime considerations
    enable_regime_awareness: bool = False
    crisis_rebalancing_multiplier: float = 0.5  # Reduce frequency in crisis
    
    # Portfolio constraints
    max_position_drift: float = 0.15  # 15% maximum drift from target
    emergency_rebalancing_threshold: float = 0.25  # 25% emergency threshold
    
    # Optimization parameters
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.weight_threshold < 1):
            raise ValueError("weight_threshold must be between 0 and 1")
        if self.min_days_between_rebalancing < 1:
            raise ValueError("min_days_between_rebalancing must be positive")
        if not (0 < self.transaction_cost_rate < 0.1):
            raise ValueError("transaction_cost_rate must be between 0 and 0.1")


class TransactionCostModel:
    """Model transaction costs for rebalancing optimization"""
    
    def __init__(self, config: RebalancingConfig):
        self.config = config
        
    def calculate_transaction_cost(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        market_impact_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs for rebalancing
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            market_impact_factor: Market impact adjustment
            
        Returns:
            Transaction cost breakdown
        """
        
        # Calculate position changes
        weight_changes = np.abs(target_weights - current_weights)
        total_turnover = np.sum(weight_changes) / 2  # Divide by 2 to avoid double counting
        
        # Base transaction costs
        base_cost = total_turnover * self.config.transaction_cost_rate * portfolio_value
        
        # Market impact costs (quadratic in turnover)
        market_impact_cost = (total_turnover ** 2) * portfolio_value * 0.0001 * market_impact_factor
        
        # Bid-ask spread costs
        bid_ask_cost = total_turnover * portfolio_value * 0.0002
        
        total_cost = base_cost + market_impact_cost + bid_ask_cost
        cost_percentage = total_cost / portfolio_value
        
        return {
            'total_cost': float(total_cost),
            'cost_percentage': float(cost_percentage),
            'base_cost': float(base_cost),
            'market_impact_cost': float(market_impact_cost),
            'bid_ask_cost': float(bid_ask_cost),
            'turnover': float(total_turnover)
        }
    
    def optimize_rebalancing_timing(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        weight_drifts: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Optimize rebalancing timing considering transaction costs
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights  
            weight_drifts: Historical weight drift patterns
            portfolio_value: Total portfolio value
            
        Returns:
            Timing optimization results
        """
        
        # Calculate current transaction cost
        current_cost = self.calculate_transaction_cost(
            current_weights, target_weights, portfolio_value
        )
        
        # Estimate future drift costs (opportunity cost of not rebalancing)
        drift_variance = np.var(weight_drifts, axis=0)
        expected_future_drift = np.sqrt(drift_variance) * np.sqrt(30)  # 30-day drift
        
        # Utility loss from drift (risk penalty)
        drift_penalty = np.sum(expected_future_drift ** 2) * portfolio_value * 0.5
        
        # Net benefit of rebalancing
        net_benefit = drift_penalty - current_cost['total_cost']
        
        # Rebalancing recommendation
        should_rebalance = (
            net_benefit > 0 and 
            current_cost['turnover'] > self.config.min_rebalancing_size
        )
        
        return {
            'should_rebalance': should_rebalance,
            'net_benefit': float(net_benefit),
            'transaction_cost': current_cost,
            'drift_penalty': float(drift_penalty),
            'cost_benefit_ratio': float(current_cost['total_cost'] / max(drift_penalty, 1))
        }


class RebalancingTriggerEngine:
    """Engine to detect rebalancing triggers"""
    
    def __init__(self, config: RebalancingConfig):
        self.config = config
        self.last_rebalancing_date: Optional[datetime] = None
        self.trigger_history: List[Dict[str, Any]] = []
        
    def check_rebalancing_triggers(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
        regime_indicator: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check all rebalancing triggers and recommend action
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            returns_data: Recent returns data
            portfolio_value: Total portfolio value
            regime_indicator: Market regime ("normal", "crisis", "recovery")
            
        Returns:
            Trigger analysis and recommendations
        """
        
        logger.info("Checking rebalancing triggers")
        
        current_date = datetime.now()
        
        # Convert to arrays for calculations
        assets = list(target_weights.keys())
        current_array = np.array([current_weights.get(asset, 0) for asset in assets])
        target_array = np.array([target_weights[asset] for asset in assets])
        
        triggers_fired = []
        trigger_details = {}
        
        # 1. Threshold-based triggers
        threshold_trigger = self._check_threshold_trigger(current_array, target_array)
        if threshold_trigger['triggered']:
            triggers_fired.append(RebalancingTrigger.THRESHOLD)
            trigger_details['threshold'] = threshold_trigger
        
        # 2. Time-based triggers
        time_trigger = self._check_time_trigger(current_date)
        if time_trigger['triggered']:
            triggers_fired.append(RebalancingTrigger.TIME_BASED)
            trigger_details['time_based'] = time_trigger
        
        # 3. Volatility-based triggers
        vol_trigger = self._check_volatility_trigger(returns_data)
        if vol_trigger['triggered']:
            triggers_fired.append(RebalancingTrigger.VOLATILITY)
            trigger_details['volatility'] = vol_trigger
        
        # 4. Risk budget triggers
        if self.config.enable_risk_budget_monitoring:
            risk_trigger = self._check_risk_budget_trigger(
                current_array, target_array, returns_data
            )
            if risk_trigger['triggered']:
                triggers_fired.append(RebalancingTrigger.RISK_BUDGET)
                trigger_details['risk_budget'] = risk_trigger
        
        # 5. Regime change triggers
        if self.config.enable_regime_awareness:
            regime_trigger = self._check_regime_trigger(regime_indicator)
            if regime_trigger['triggered']:
                triggers_fired.append(RebalancingTrigger.REGIME_CHANGE)
                trigger_details['regime'] = regime_trigger
        
        # 6. Emergency triggers
        emergency_trigger = self._check_emergency_trigger(current_array, target_array)
        
        # Overall recommendation
        should_rebalance = (
            len(triggers_fired) > 0 or 
            emergency_trigger['triggered']
        )
        
        # Apply regime-based frequency adjustment
        if regime_indicator == "crisis" and self.config.enable_regime_awareness:
            # Reduce rebalancing frequency in crisis
            days_since_last = self._days_since_last_rebalancing(current_date)
            min_days_adjusted = self.config.min_days_between_rebalancing / self.config.crisis_rebalancing_multiplier
            
            if days_since_last < min_days_adjusted and not emergency_trigger['triggered']:
                should_rebalance = False
                logger.info("Rebalancing suppressed due to crisis regime", 
                           days_since_last=days_since_last)
        
        result = {
            'timestamp': current_date.isoformat(),
            'should_rebalance': should_rebalance,
            'triggers_fired': [t.value for t in triggers_fired],
            'trigger_count': len(triggers_fired),
            'emergency_trigger': emergency_trigger,
            'trigger_details': trigger_details,
            'regime_indicator': regime_indicator,
            'days_since_last_rebalancing': self._days_since_last_rebalancing(current_date)
        }
        
        # Store in history
        self.trigger_history.append({
            'timestamp': current_date.isoformat(),
            'triggers_fired': len(triggers_fired),
            'should_rebalance': should_rebalance,
            'emergency': emergency_trigger['triggered']
        })
        
        # Keep only recent history
        if len(self.trigger_history) > 1000:
            self.trigger_history = self.trigger_history[-1000:]
        
        return result
    
    def _check_threshold_trigger(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> Dict[str, Any]:
        """Check threshold-based rebalancing triggers"""
        
        weight_diffs = np.abs(current_weights - target_weights)
        max_deviation = np.max(weight_diffs)
        avg_deviation = np.mean(weight_diffs)
        
        # Relative threshold (percentage of target weight)
        relative_deviations = weight_diffs / np.maximum(target_weights, 0.01)  # Avoid division by zero
        max_relative_deviation = np.max(relative_deviations)
        
        triggered = (
            max_deviation > self.config.weight_threshold or
            max_relative_deviation > self.config.weight_threshold or
            avg_deviation > self.config.absolute_threshold
        )
        
        return {
            'triggered': triggered,
            'max_deviation': float(max_deviation),
            'avg_deviation': float(avg_deviation),
            'max_relative_deviation': float(max_relative_deviation),
            'threshold': self.config.weight_threshold
        }
    
    def _check_time_trigger(self, current_date: datetime) -> Dict[str, Any]:
        """Check time-based rebalancing triggers"""
        
        if self.last_rebalancing_date is None:
            return {'triggered': True, 'reason': 'first_rebalancing'}
        
        days_since_last = (current_date - self.last_rebalancing_date).days
        
        # Frequency mapping
        frequency_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90
        }
        
        required_days = frequency_days.get(self.config.rebalancing_frequency, 30)
        min_days = self.config.min_days_between_rebalancing
        
        triggered = days_since_last >= max(required_days, min_days)
        
        return {
            'triggered': triggered,
            'days_since_last': days_since_last,
            'required_days': required_days,
            'frequency': self.config.rebalancing_frequency
        }
    
    def _check_volatility_trigger(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Check volatility-based rebalancing triggers"""
        
        if not self.config.enable_volatility_adjustment or len(returns_data) < self.config.vol_lookback_days:
            return {'triggered': False, 'reason': 'insufficient_data'}
        
        # Calculate recent portfolio volatility
        recent_data = returns_data.tail(self.config.vol_lookback_days)
        portfolio_returns = recent_data.mean(axis=1)  # Equal weight for simplicity
        current_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Trigger based on volatility regime
        high_vol_trigger = current_vol > self.config.high_vol_threshold
        low_vol_trigger = current_vol < self.config.low_vol_threshold
        
        triggered = high_vol_trigger or low_vol_trigger
        
        return {
            'triggered': triggered,
            'current_volatility': float(current_vol),
            'high_vol_threshold': self.config.high_vol_threshold,
            'low_vol_threshold': self.config.low_vol_threshold,
            'volatility_regime': 'high' if high_vol_trigger else 'low' if low_vol_trigger else 'normal'
        }
    
    def _check_risk_budget_trigger(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        returns_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check risk budget deviation triggers"""
        
        if len(returns_data) < 30:  # Need sufficient data
            return {'triggered': False, 'reason': 'insufficient_data'}
        
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Calculate risk contributions for current and target portfolios
            current_risk_contrib = self._calculate_risk_contributions(current_weights, cov_matrix.values)
            target_risk_contrib = self._calculate_risk_contributions(target_weights, cov_matrix.values)
            
            # Calculate risk budget deviations
            risk_deviations = np.abs(current_risk_contrib - target_risk_contrib)
            max_risk_deviation = np.max(risk_deviations)
            
            triggered = max_risk_deviation > self.config.risk_budget_threshold
            
            return {
                'triggered': triggered,
                'max_risk_deviation': float(max_risk_deviation),
                'threshold': self.config.risk_budget_threshold,
                'current_risk_contrib': current_risk_contrib.tolist(),
                'target_risk_contrib': target_risk_contrib.tolist()
            }
            
        except Exception as e:
            logger.warning("Error calculating risk budget trigger", error=str(e))
            return {'triggered': False, 'error': str(e)}
    
    def _check_regime_trigger(self, regime_indicator: Optional[str]) -> Dict[str, Any]:
        """Check market regime change triggers"""
        
        # This would typically compare current regime to previous regime
        # For now, we'll implement a simple version
        
        if regime_indicator == "crisis":
            # More frequent rebalancing in crisis
            return {
                'triggered': True,
                'regime': regime_indicator,
                'reason': 'crisis_regime_detected'
            }
        
        return {'triggered': False, 'regime': regime_indicator}
    
    def _check_emergency_trigger(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> Dict[str, Any]:
        """Check emergency rebalancing triggers"""
        
        max_drift = np.max(np.abs(current_weights - target_weights))
        
        triggered = max_drift > self.config.emergency_rebalancing_threshold
        
        return {
            'triggered': triggered,
            'max_drift': float(max_drift),
            'threshold': self.config.emergency_rebalancing_threshold,
            'severity': 'critical' if triggered else 'normal'
        }
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contributions for portfolio weights"""
        
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if portfolio_vol > 0:
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            return risk_contributions / np.sum(risk_contributions)  # Normalize
        else:
            return weights / np.sum(weights)
    
    def _days_since_last_rebalancing(self, current_date: datetime) -> int:
        """Calculate days since last rebalancing"""
        
        if self.last_rebalancing_date is None:
            return 999  # Large number to force rebalancing
        
        return (current_date - self.last_rebalancing_date).days
    
    def mark_rebalancing_completed(self, date: Optional[datetime] = None) -> None:
        """Mark rebalancing as completed"""
        
        self.last_rebalancing_date = date or datetime.now()
        logger.info("Rebalancing marked as completed", date=self.last_rebalancing_date.isoformat())


class DynamicRebalancer:
    """Main dynamic rebalancing system"""
    
    def __init__(self, config: RebalancingConfig):
        self.config = config
        self.trigger_engine = RebalancingTriggerEngine(config)
        self.transaction_cost_model = TransactionCostModel(config)
        self.rebalancing_history: List[Dict[str, Any]] = []
        
    def evaluate_rebalancing_need(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
        regime_indicator: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate whether portfolio needs rebalancing
        
        Args:
            current_portfolio: Current portfolio weights
            target_portfolio: Target portfolio weights
            returns_data: Historical returns data
            portfolio_value: Total portfolio value
            regime_indicator: Market regime indicator
            
        Returns:
            Comprehensive rebalancing evaluation
        """
        
        logger.info("Evaluating rebalancing need", 
                   assets=len(current_portfolio),
                   portfolio_value=portfolio_value)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'comprehensive',
            'portfolio_value': portfolio_value,
            'assets_count': len(current_portfolio)
        }
        
        try:
            # 1. Check rebalancing triggers
            trigger_analysis = self.trigger_engine.check_rebalancing_triggers(
                current_portfolio, target_portfolio, returns_data, 
                portfolio_value, regime_indicator
            )
            
            # 2. Transaction cost analysis
            assets = list(target_portfolio.keys())
            current_array = np.array([current_portfolio.get(asset, 0) for asset in assets])
            target_array = np.array([target_portfolio[asset] for asset in assets])
            
            cost_analysis = self.transaction_cost_model.calculate_transaction_cost(
                current_array, target_array, portfolio_value
            )
            
            # 3. Timing optimization
            if len(self.rebalancing_history) > 5:  # Need history for drift estimation
                recent_drifts = self._calculate_historical_drifts()
                timing_analysis = self.transaction_cost_model.optimize_rebalancing_timing(
                    current_array, target_array, recent_drifts, portfolio_value
                )
            else:
                timing_analysis = {'should_rebalance': trigger_analysis['should_rebalance']}
            
            # 4. Final recommendation
            should_rebalance = (
                trigger_analysis['should_rebalance'] and
                (not self.config.enable_transaction_cost_optimization or 
                 timing_analysis['should_rebalance'])
            )
            
            # 5. Calculate rebalancing trades
            rebalancing_trades = {}
            if should_rebalance:
                rebalancing_trades = self._calculate_rebalancing_trades(
                    current_portfolio, target_portfolio, portfolio_value
                )
            
            # Store results
            result.update({
                'should_rebalance': should_rebalance,
                'trigger_analysis': trigger_analysis,
                'cost_analysis': cost_analysis,
                'timing_analysis': timing_analysis,
                'rebalancing_trades': rebalancing_trades,
                'regime_indicator': regime_indicator
            })
            
            # Store in history if rebalancing is recommended
            if should_rebalance:
                self.rebalancing_history.append({
                    'timestamp': result['timestamp'],
                    'portfolio_value': portfolio_value,
                    'transaction_cost': cost_analysis['cost_percentage'],
                    'triggers_fired': trigger_analysis['trigger_count'],
                    'regime': regime_indicator
                })
                
                # Keep only recent history
                if len(self.rebalancing_history) > 100:
                    self.rebalancing_history = self.rebalancing_history[-100:]
            
            logger.info("Rebalancing evaluation completed",
                       should_rebalance=should_rebalance,
                       triggers_fired=trigger_analysis['trigger_count'],
                       transaction_cost=cost_analysis['cost_percentage'])
            
        except Exception as e:
            logger.error("Error in rebalancing evaluation", error=str(e))
            result['error'] = str(e)
            result['should_rebalance'] = False
        
        return result
    
    def execute_rebalancing(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float,
        execution_method: str = "gradual"
    ) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing
        
        Args:
            current_portfolio: Current portfolio weights
            target_portfolio: Target portfolio weights
            portfolio_value: Total portfolio value
            execution_method: "immediate" or "gradual"
            
        Returns:
            Execution results
        """
        
        logger.info("Executing portfolio rebalancing", method=execution_method)
        
        # Calculate trades
        trades = self._calculate_rebalancing_trades(
            current_portfolio, target_portfolio, portfolio_value
        )
        
        # Execute trades (this would interface with broker API)
        execution_results = self._simulate_trade_execution(trades, execution_method)
        
        # Mark rebalancing as completed
        self.trigger_engine.mark_rebalancing_completed()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'execution_method': execution_method,
            'trades_executed': len(trades),
            'total_trade_value': sum(abs(trade['value']) for trade in trades.values()),
            'execution_results': execution_results,
            'new_portfolio': target_portfolio
        }
        
        logger.info("Rebalancing execution completed", 
                   trades_count=len(trades),
                   total_value=result['total_trade_value'])
        
        return result
    
    def _calculate_rebalancing_trades(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate specific trades needed for rebalancing"""
        
        trades = {}
        
        for asset in target_portfolio:
            current_weight = current_portfolio.get(asset, 0)
            target_weight = target_portfolio[asset]
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > self.config.min_rebalancing_size:
                trade_value = weight_diff * portfolio_value
                
                trades[asset] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_diff,
                    'value': trade_value,
                    'direction': 'buy' if weight_diff > 0 else 'sell',
                    'size': abs(trade_value)
                }
        
        return trades
    
    def _simulate_trade_execution(
        self,
        trades: Dict[str, Dict[str, Any]],
        method: str
    ) -> Dict[str, Any]:
        """Simulate trade execution (would interface with real broker)"""
        
        execution_results = {
            'total_trades': len(trades),
            'successful_trades': len(trades),  # Assume all successful for now
            'failed_trades': 0,
            'execution_time': datetime.now().isoformat(),
            'method': method,
            'trade_details': {}
        }
        
        for asset, trade in trades.items():
            # Simulate execution latency and slippage
            execution_price = 1.0  # Would be actual market price
            slippage = 0.0001 * abs(trade['value'])  # Simple slippage model
            
            execution_results['trade_details'][asset] = {
                'executed': True,
                'execution_price': execution_price,
                'slippage': slippage,
                'executed_value': trade['value'] - slippage
            }
        
        return execution_results
    
    def _calculate_historical_drifts(self) -> np.ndarray:
        """Calculate historical weight drift patterns"""
        
        # This would use actual historical weight changes
        # For now, return dummy data
        return np.random.normal(0, 0.02, (len(self.rebalancing_history), 10))
    
    def get_rebalancing_dashboard(self) -> Dict[str, Any]:
        """Get rebalancing system dashboard data"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'weight_threshold': self.config.weight_threshold,
                'rebalancing_frequency': self.config.rebalancing_frequency,
                'transaction_cost_optimization': self.config.enable_transaction_cost_optimization,
                'volatility_adjustment': self.config.enable_volatility_adjustment,
                'regime_awareness': self.config.enable_regime_awareness
            },
            'recent_rebalancing': self.rebalancing_history[-10:],
            'trigger_history': self.trigger_engine.trigger_history[-20:],
            'performance_stats': {
                'total_rebalancings': len(self.rebalancing_history),
                'avg_transaction_cost': np.mean([r['transaction_cost'] for r in self.rebalancing_history]) if self.rebalancing_history else 0,
                'avg_triggers_per_rebalancing': np.mean([r['triggers_fired'] for r in self.rebalancing_history]) if self.rebalancing_history else 0,
                'last_rebalancing': self.trigger_engine.last_rebalancing_date.isoformat() if self.trigger_engine.last_rebalancing_date else None
            }
        }