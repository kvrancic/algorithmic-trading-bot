"""
Portfolio Optimization Module

Advanced portfolio optimization components for algorithmic trading:
- Black-Litterman optimization with market views
- Modern Portfolio Theory (Markowitz) optimization
- Risk parity allocation strategies  
- Dynamic rebalancing systems
- Regime-specific allocation models
- Multi-objective optimization
"""

from .black_litterman import BlackLittermanOptimizer, BlackLittermanConfig
from .markowitz import MarkowitzOptimizer, MarkowitzConfig
from .risk_parity import RiskParityOptimizer, RiskParityConfig
from .rebalancer import DynamicRebalancer, RebalancingConfig
from .regime_allocator import RegimeSpecificAllocator, RegimeConfig

# Alias for backward compatibility  
RegimeAwareAllocator = RegimeSpecificAllocator

__all__ = [
    'BlackLittermanOptimizer',
    'BlackLittermanConfig',
    'MarkowitzOptimizer', 
    'MarkowitzConfig',
    'RiskParityOptimizer',
    'RiskParityConfig',
    'DynamicRebalancer',
    'RebalancingConfig',
    'RegimeSpecificAllocator',
    'RegimeConfig',
    'RegimeAwareAllocator'  # Alias
]