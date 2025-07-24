"""
Risk Management Module

Enterprise-grade risk management components for algorithmic trading:
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Adaptive stop-loss systems
- Maximum drawdown controllers
- Position sizing with Kelly Criterion
- Correlation penalty systems
- Real-time risk monitoring
"""

from .risk_engine import RiskEngine, RiskConfig
from .position_sizer import PositionSizer, PositionSizingConfig, KellyCriterion
from .drawdown_controller import DrawdownController, DrawdownConfig
from .stop_loss_manager import AdaptiveStopLossManager, StopLossConfig
from .correlation_manager import CorrelationPenaltyManager, CorrelationConfig

__all__ = [
    'RiskEngine',
    'RiskConfig', 
    'PositionSizer',
    'PositionSizingConfig',
    'KellyCriterion',
    'DrawdownController',
    'DrawdownConfig',
    'AdaptiveStopLossManager',
    'StopLossConfig',
    'CorrelationPenaltyManager',
    'CorrelationConfig'
]