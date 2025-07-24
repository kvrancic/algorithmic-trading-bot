"""
Execution Engine Module

Advanced trade execution system for algorithmic trading:
- Smart order routing with TWAP/VWAP/Iceberg strategies
- Slippage prediction and market impact minimization
- Order book analysis and execution performance tracking
- Real-time execution monitoring and risk controls
- Multi-venue routing and liquidity aggregation
"""

from .order_strategies import TWAPStrategy, VWAPStrategy, IcebergStrategy, OrderStrategyFactory
from .slippage_predictor import SlippagePredictor, SlippagePredictorConfig
from .market_impact import MarketImpactModel, MarketImpactConfig
from .orderbook_analyzer import OrderBookAnalyzer, OrderBookConfig
from .execution_tracker import ExecutionTracker, ExecutionConfig
from .smart_router import SmartOrderRouter, RoutingConfig

__all__ = [
    'TWAPStrategy',
    'VWAPStrategy', 
    'IcebergStrategy',
    'OrderStrategyFactory',
    'SlippagePredictor',
    'SlippagePredictorConfig',
    'MarketImpactModel',
    'MarketImpactConfig',
    'OrderBookAnalyzer',
    'OrderBookConfig',
    'ExecutionTracker',
    'ExecutionConfig',
    'SmartOrderRouter',
    'RoutingConfig'
]