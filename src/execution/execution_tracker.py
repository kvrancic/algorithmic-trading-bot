"""
Advanced Execution Performance Tracking

Comprehensive execution performance monitoring and analysis:
- Real-time execution quality measurement and benchmarking
- Implementation shortfall analysis and attribution
- Venue performance comparison and routing optimization
- Transaction cost analysis (TCA) with market impact decomposition
- Execution algorithm performance evaluation and tuning
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)


class ExecutionStatus(Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionConfig:
    """Configuration for execution tracking"""
    
    # Performance measurement
    benchmark_method: str = "arrival_price"  # "arrival_price", "vwap", "twap", "close"
    measurement_window: int = 300  # seconds for performance measurement
    slippage_calculation_method: str = "implementation_shortfall"
    
    # TCA parameters
    enable_market_impact_decomposition: bool = True
    timing_risk_window: int = 600  # seconds for timing risk calculation
    spread_cost_method: str = "effective_spread"  # "quoted_spread", "effective_spread"
    
    # Venue analysis
    venue_performance_tracking: bool = True
    min_venue_trades_for_analysis: int = 10
    venue_comparison_window: int = 86400  # 24 hours
    
    # Algorithm evaluation
    algorithm_benchmarking: bool = True
    min_algorithm_samples: int = 20
    performance_attribution_enabled: bool = True
    
    # Risk monitoring
    execution_risk_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_slippage_bps': 50.0,
        'max_timing_risk_bps': 30.0,
        'max_market_impact_bps': 25.0
    })
    
    # Reporting
    performance_reporting_frequency: int = 3600  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'poor_execution_threshold': 20.0,  # bps
        'venue_underperformance_threshold': 15.0,  # bps
        'algorithm_degradation_threshold': 10.0  # bps
    })
    
    def __post_init__(self):
        """Validate configuration"""
        if self.measurement_window <= 0:
            raise ValueError("measurement_window must be positive")
        if self.min_venue_trades_for_analysis < 1:
            raise ValueError("min_venue_trades_for_analysis must be positive")


@dataclass
class TradeRecord:
    """Individual trade execution record"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    algorithm: str
    parent_order_id: str
    execution_duration: float  # seconds
    
    # Market data at execution
    arrival_price: float
    benchmark_price: float
    market_volume: Optional[float] = None
    spread_at_execution: Optional[float] = None
    
    # Performance metrics (calculated)
    slippage_bps: Optional[float] = None
    market_impact_bps: Optional[float] = None
    timing_cost_bps: Optional[float] = None
    spread_cost_bps: Optional[float] = None
    
    @property
    def trade_value(self) -> float:
        """Calculate trade value"""
        return self.quantity * self.price
    
    @property
    def implementation_shortfall_bps(self) -> float:
        """Calculate implementation shortfall in basis points"""
        if self.side == 'buy':
            shortfall = (self.price - self.arrival_price) / self.arrival_price
        else:
            shortfall = (self.arrival_price - self.price) / self.arrival_price
        
        return shortfall * 10000


@dataclass
class OrderExecution:
    """Complete order execution with multiple trades"""
    order_id: str
    symbol: str
    side: str
    total_quantity: float
    algorithm: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Trades belonging to this order
    trades: List[TradeRecord] = field(default_factory=list)
    
    # Benchmark data
    arrival_price: float = 0.0
    vwap_benchmark: Optional[float] = None
    twap_benchmark: Optional[float] = None
    close_benchmark: Optional[float] = None
    
    # Performance metrics
    total_slippage_bps: Optional[float] = None
    market_impact_bps: Optional[float] = None
    timing_risk_bps: Optional[float] = None
    execution_shortfall_bps: Optional[float] = None
    
    @property
    def filled_quantity(self) -> float:
        """Calculate filled quantity"""
        return sum(trade.quantity for trade in self.trades)
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate percentage"""
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0
    
    @property
    def average_execution_price(self) -> float:
        """Calculate volume-weighted average execution price"""
        if not self.trades:
            return 0.0
        
        total_value = sum(trade.quantity * trade.price for trade in self.trades)
        total_quantity = sum(trade.quantity for trade in self.trades)
        
        return total_value / total_quantity if total_quantity > 0 else 0.0
    
    @property
    def execution_duration(self) -> float:
        """Calculate total execution duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class PerformanceMeasurement:
    """Calculate execution performance metrics"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        
    def calculate_trade_metrics(
        self,
        trade: TradeRecord,
        market_data: Optional[pd.DataFrame] = None
    ) -> TradeRecord:
        """
        Calculate performance metrics for individual trade
        
        Args:
            trade: Trade record to analyze
            market_data: Market data around execution time
            
        Returns:
            Trade record with calculated metrics
        """
        
        # Implementation shortfall (basic slippage)
        trade.slippage_bps = trade.implementation_shortfall_bps
        
        # Market impact estimation
        trade.market_impact_bps = self._estimate_market_impact(trade, market_data)
        
        # Timing cost
        trade.timing_cost_bps = self._calculate_timing_cost(trade, market_data)
        
        # Spread cost
        trade.spread_cost_bps = self._calculate_spread_cost(trade)
        
        return trade
    
    def calculate_order_metrics(
        self,
        order: OrderExecution,
        market_data: Optional[pd.DataFrame] = None
    ) -> OrderExecution:
        """
        Calculate performance metrics for complete order execution
        
        Args:
            order: Order execution to analyze
            market_data: Market data during execution period
            
        Returns:
            Order execution with calculated metrics
        """
        
        if not order.trades:
            return order
        
        # Calculate VWAP and TWAP benchmarks
        if market_data is not None:
            order.vwap_benchmark = self._calculate_vwap_benchmark(order, market_data)
            order.twap_benchmark = self._calculate_twap_benchmark(order, market_data)
            order.close_benchmark = self._get_close_benchmark(order, market_data)
        
        # Aggregate trade metrics
        total_quantity = order.filled_quantity
        
        if total_quantity > 0:
            # Volume-weighted aggregation
            order.total_slippage_bps = sum(
                trade.slippage_bps * (trade.quantity / total_quantity)
                for trade in order.trades
                if trade.slippage_bps is not None
            )
            
            order.market_impact_bps = sum(
                trade.market_impact_bps * (trade.quantity / total_quantity)
                for trade in order.trades
                if trade.market_impact_bps is not None
            )
            
            order.timing_risk_bps = sum(
                trade.timing_cost_bps * (trade.quantity / total_quantity)
                for trade in order.trades
                if trade.timing_cost_bps is not None
            )
        
        # Execution shortfall vs benchmark
        order.execution_shortfall_bps = self._calculate_execution_shortfall(order)
        
        return order
    
    def _estimate_market_impact(
        self,
        trade: TradeRecord,
        market_data: Optional[pd.DataFrame]
    ) -> float:
        """Estimate market impact component of trade cost"""
        
        if market_data is None or trade.market_volume is None:
            # Use simplified model based on trade size
            size_factor = min(trade.quantity / 10000, 0.1)  # Assume 10k shares baseline
            return size_factor * 5.0  # 5 bps per 10k shares
        
        # More sophisticated model using market data
        try:
            # Find market data around execution time
            execution_time = trade.timestamp
            time_window = timedelta(minutes=2)
            
            start_time = execution_time - time_window
            end_time = execution_time + time_window
            
            relevant_data = market_data[
                (market_data.index >= start_time) & 
                (market_data.index <= end_time)
            ]
            
            if len(relevant_data) < 2:
                return 2.0  # Default impact
            
            # Calculate price movement around execution
            pre_price = relevant_data['close'].iloc[0]
            post_price = relevant_data['close'].iloc[-1]
            
            if trade.side == 'buy':
                price_impact = (post_price - pre_price) / pre_price
            else:
                price_impact = (pre_price - post_price) / pre_price
            
            # Attribute portion to market impact (vs random movement)
            volume_ratio = trade.quantity / trade.market_volume if trade.market_volume > 0 else 0.01
            impact_attribution = min(volume_ratio * 2, 0.8)  # Max 80% attribution
            
            attributed_impact = price_impact * impact_attribution * 10000
            
            return max(0, min(attributed_impact, 20.0))  # Cap at 20 bps
            
        except Exception as e:
            logger.warning(f"Error calculating market impact: {e}")
            return 2.0  # Default impact
    
    def _calculate_timing_cost(
        self,
        trade: TradeRecord,
        market_data: Optional[pd.DataFrame]
    ) -> float:
        """Calculate timing cost (delay from decision to execution)"""
        
        # Simplified timing cost model
        execution_delay = trade.execution_duration
        
        if execution_delay <= 60:  # Less than 1 minute
            return 0.5  # 0.5 bps
        elif execution_delay <= 300:  # Less than 5 minutes
            return 1.5  # 1.5 bps
        elif execution_delay <= 900:  # Less than 15 minutes
            return 3.0  # 3.0 bps
        else:
            return 5.0  # 5.0 bps for longer delays
    
    def _calculate_spread_cost(self, trade: TradeRecord) -> float:
        """Calculate spread cost component"""
        
        if trade.spread_at_execution is None:
            return 2.0  # Default spread cost
        
        # Spread cost is typically half the spread
        spread_bps = (trade.spread_at_execution / trade.price) * 10000
        return spread_bps * 0.5
    
    def _calculate_vwap_benchmark(
        self,
        order: OrderExecution,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate VWAP benchmark for the execution period"""
        
        if 'volume' not in market_data.columns:
            return self._calculate_twap_benchmark(order, market_data)
        
        # Filter data for execution period
        execution_data = market_data[
            (market_data.index >= order.start_time) &
            (market_data.index <= (order.end_time or datetime.now()))
        ]
        
        if len(execution_data) == 0:
            return order.arrival_price
        
        # Calculate VWAP
        total_value = (execution_data['close'] * execution_data['volume']).sum()
        total_volume = execution_data['volume'].sum()
        
        return total_value / total_volume if total_volume > 0 else order.arrival_price
    
    def _calculate_twap_benchmark(
        self,
        order: OrderExecution,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate TWAP benchmark for the execution period"""
        
        # Filter data for execution period
        execution_data = market_data[
            (market_data.index >= order.start_time) &
            (market_data.index <= (order.end_time or datetime.now()))
        ]
        
        if len(execution_data) == 0:
            return order.arrival_price
        
        # Calculate TWAP
        return execution_data['close'].mean()
    
    def _get_close_benchmark(
        self,
        order: OrderExecution,
        market_data: pd.DataFrame
    ) -> float:
        """Get closing price benchmark"""
        
        # Get the last available price
        if len(market_data) > 0:
            return market_data['close'].iloc[-1]
        else:
            return order.arrival_price
    
    def _calculate_execution_shortfall(self, order: OrderExecution) -> float:
        """Calculate execution shortfall vs selected benchmark"""
        
        avg_price = order.average_execution_price
        
        if self.config.benchmark_method == "arrival_price":
            benchmark = order.arrival_price
        elif self.config.benchmark_method == "vwap" and order.vwap_benchmark:
            benchmark = order.vwap_benchmark
        elif self.config.benchmark_method == "twap" and order.twap_benchmark:
            benchmark = order.twap_benchmark
        elif self.config.benchmark_method == "close" and order.close_benchmark:
            benchmark = order.close_benchmark
        else:
            benchmark = order.arrival_price
        
        if benchmark == 0:
            return 0.0
        
        if order.side == 'buy':
            shortfall = (avg_price - benchmark) / benchmark
        else:
            shortfall = (benchmark - avg_price) / benchmark
        
        return shortfall * 10000  # Convert to basis points


class VenueAnalyzer:
    """Analyze execution performance across different venues"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.venue_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def add_venue_execution(
        self,
        venue: str,
        execution_metrics: Dict[str, Any]
    ) -> None:
        """Add execution metrics for venue analysis"""
        
        self.venue_metrics[venue].append({
            'timestamp': datetime.now(),
            'metrics': execution_metrics
        })
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(seconds=self.config.venue_comparison_window)
        self.venue_metrics[venue] = [
            entry for entry in self.venue_metrics[venue]
            if entry['timestamp'] > cutoff_time
        ]
    
    def compare_venue_performance(self) -> Dict[str, Any]:
        """
        Compare performance across venues
        
        Returns:
            Venue performance comparison results
        """
        
        venue_stats = {}
        
        for venue, executions in self.venue_metrics.items():
            if len(executions) < self.config.min_venue_trades_for_analysis:
                continue
            
            # Extract metrics
            slippages = [e['metrics'].get('slippage_bps', 0) for e in executions]
            market_impacts = [e['metrics'].get('market_impact_bps', 0) for e in executions]
            fill_rates = [e['metrics'].get('fill_rate', 0) for e in executions]
            execution_times = [e['metrics'].get('execution_duration', 0) for e in executions]
            
            venue_stats[venue] = {
                'sample_count': len(executions),
                'avg_slippage_bps': np.mean(slippages),
                'median_slippage_bps': np.median(slippages),
                'slippage_volatility': np.std(slippages),
                'avg_market_impact_bps': np.mean(market_impacts),
                'avg_fill_rate': np.mean(fill_rates),
                'avg_execution_time': np.mean(execution_times),
                'percentile_75_slippage': np.percentile(slippages, 75),
                'percentile_25_slippage': np.percentile(slippages, 25),
                'success_rate': len([e for e in executions if e['metrics'].get('fill_rate', 0) > 0.95])
            }
        
        # Rank venues
        if venue_stats:
            # Composite score: lower slippage, higher fill rate, faster execution
            for venue, stats in venue_stats.items():
                slippage_score = 1 / (1 + stats['avg_slippage_bps'] / 10)  # Normalize
                fill_score = stats['avg_fill_rate']
                speed_score = 1 / (1 + stats['avg_execution_time'] / 60)  # Normalize to minutes
                
                stats['composite_score'] = (slippage_score + fill_score + speed_score) / 3
            
            # Sort by composite score
            ranked_venues = sorted(
                venue_stats.items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )
        else:
            ranked_venues = []
        
        return {
            'venue_statistics': venue_stats,
            'ranked_venues': ranked_venues,
            'best_venue': ranked_venues[0][0] if ranked_venues else None,
            'analysis_timestamp': datetime.now().isoformat(),
            'venues_analyzed': len(venue_stats)
        }
    
    def get_venue_routing_recommendation(
        self,
        order_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recommend optimal venue routing based on historical performance
        
        Args:
            order_characteristics: Order size, urgency, symbol characteristics
            
        Returns:
            Venue routing recommendations
        """
        
        venue_comparison = self.compare_venue_performance()
        
        if not venue_comparison['venue_statistics']:
            return {'recommendation': None, 'reason': 'insufficient_data'}
        
        order_size = order_characteristics.get('size', 0)
        urgency = order_characteristics.get('urgency', 'normal')
        
        recommendations = []
        
        for venue, stats in venue_comparison['venue_statistics'].items():
            # Score based on order characteristics
            if urgency == 'high':
                # Prioritize fast execution and high fill rates
                score = stats['avg_fill_rate'] * 0.5 + (1 / (1 + stats['avg_execution_time'] / 60)) * 0.3 + (1 / (1 + stats['avg_slippage_bps'] / 10)) * 0.2
            elif urgency == 'low':
                # Prioritize low slippage
                score = (1 / (1 + stats['avg_slippage_bps'] / 10)) * 0.6 + stats['avg_fill_rate'] * 0.3 + (1 / (1 + stats['avg_execution_time'] / 60)) * 0.1
            else:  # normal
                # Balanced approach
                score = stats['composite_score']
            
            recommendations.append({
                'venue': venue,
                'score': score,
                'expected_slippage_bps': stats['avg_slippage_bps'],
                'expected_fill_rate': stats['avg_fill_rate'],
                'expected_execution_time': stats['avg_execution_time']
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'alternative_venues': recommendations[1:3],  # Top 3 alternatives
            'routing_strategy': 'single_venue' if urgency == 'high' else 'multi_venue',
            'confidence': min(len(venue_comparison['venue_statistics']) / 3, 1.0)
        }


class AlgorithmEvaluator:
    """Evaluate execution algorithm performance"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.algorithm_history: Dict[str, List[OrderExecution]] = defaultdict(list)
        
    def add_execution(self, execution: OrderExecution) -> None:
        """Add completed execution for algorithm evaluation"""
        
        if execution.status == ExecutionStatus.COMPLETED:
            self.algorithm_history[execution.algorithm].append(execution)
            
            # Keep only recent history
            for algorithm in self.algorithm_history:
                if len(self.algorithm_history[algorithm]) > 1000:
                    self.algorithm_history[algorithm] = self.algorithm_history[algorithm][-1000:]
    
    def evaluate_algorithm_performance(self, algorithm: str) -> Dict[str, Any]:
        """
        Evaluate performance of specific execution algorithm
        
        Args:
            algorithm: Algorithm name to evaluate
            
        Returns:
            Algorithm performance analysis
        """
        
        executions = self.algorithm_history.get(algorithm, [])
        
        if len(executions) < self.config.min_algorithm_samples:
            return {
                'algorithm': algorithm,
                'status': 'insufficient_data',
                'sample_count': len(executions)
            }
        
        # Recent performance (last 50 executions)
        recent_executions = executions[-50:]
        
        # Calculate performance metrics
        fill_rates = [e.fill_rate for e in recent_executions]
        slippages = [e.total_slippage_bps for e in recent_executions if e.total_slippage_bps is not None]
        execution_times = [e.execution_duration for e in recent_executions]
        shortfalls = [e.execution_shortfall_bps for e in recent_executions if e.execution_shortfall_bps is not None]
        
        # Performance attribution
        attribution = self._perform_performance_attribution(recent_executions)
        
        # Trend analysis
        trend_analysis = self._analyze_performance_trend(executions[-100:] if len(executions) >= 100 else executions)
        
        return {
            'algorithm': algorithm,
            'status': 'analyzed',
            'sample_count': len(executions),
            'recent_sample_count': len(recent_executions),
            'performance_metrics': {
                'avg_fill_rate': np.mean(fill_rates),
                'fill_rate_volatility': np.std(fill_rates),
                'avg_slippage_bps': np.mean(slippages) if slippages else 0,
                'slippage_volatility': np.std(slippages) if len(slippages) > 1 else 0,
                'avg_execution_time': np.mean(execution_times),
                'avg_shortfall_bps': np.mean(shortfalls) if shortfalls else 0,
                'percentile_75_slippage': np.percentile(slippages, 75) if slippages else 0,
                'percentile_25_slippage': np.percentile(slippages, 25) if slippages else 0
            },
            'performance_attribution': attribution,
            'trend_analysis': trend_analysis,
            'quality_score': self._calculate_algorithm_quality_score(recent_executions),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def compare_algorithms(self) -> Dict[str, Any]:
        """Compare performance across all algorithms"""
        
        algorithm_comparisons = {}
        
        for algorithm in self.algorithm_history:
            if len(self.algorithm_history[algorithm]) >= self.config.min_algorithm_samples:
                algorithm_comparisons[algorithm] = self.evaluate_algorithm_performance(algorithm)
        
        if not algorithm_comparisons:
            return {'status': 'no_algorithms_to_compare'}
        
        # Rank algorithms by quality score
        ranked_algorithms = sorted(
            algorithm_comparisons.items(),
            key=lambda x: x[1].get('quality_score', 0),
            reverse=True
        )
        
        return {
            'algorithm_comparisons': algorithm_comparisons,
            'ranked_algorithms': ranked_algorithms,
            'best_algorithm': ranked_algorithms[0][0] if ranked_algorithms else None,
            'algorithms_analyzed': len(algorithm_comparisons)
        }
    
    def _perform_performance_attribution(
        self,
        executions: List[OrderExecution]
    ) -> Dict[str, Any]:
        """Perform performance attribution analysis"""
        
        if not self.config.performance_attribution_enabled:
            return {}
        
        # Attribute performance to different factors
        total_slippage = []
        market_impact = []
        timing_risk = []
        
        for execution in executions:
            if execution.total_slippage_bps is not None:
                total_slippage.append(execution.total_slippage_bps)
            if execution.market_impact_bps is not None:
                market_impact.append(execution.market_impact_bps)
            if execution.timing_risk_bps is not None:
                timing_risk.append(execution.timing_risk_bps)
        
        attribution = {}
        
        if total_slippage:
            avg_total = np.mean(total_slippage)
            attribution['total_slippage_bps'] = avg_total
            
            if market_impact and timing_risk:
                avg_market_impact = np.mean(market_impact)
                avg_timing_risk = np.mean(timing_risk)
                
                # Calculate residual (unexplained slippage)
                residual = avg_total - avg_market_impact - avg_timing_risk
                
                attribution.update({
                    'market_impact_contribution': avg_market_impact / avg_total if avg_total != 0 else 0,
                    'timing_risk_contribution': avg_timing_risk / avg_total if avg_total != 0 else 0,
                    'unexplained_contribution': residual / avg_total if avg_total != 0 else 0,
                    'avg_market_impact_bps': avg_market_impact,
                    'avg_timing_risk_bps': avg_timing_risk,
                    'avg_residual_bps': residual
                })
        
        return attribution
    
    def _analyze_performance_trend(
        self,
        executions: List[OrderExecution]
    ) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(executions) < 20:
            return {'status': 'insufficient_data_for_trend'}
        
        # Sort by execution time
        sorted_executions = sorted(executions, key=lambda x: x.start_time)
        
        # Extract time series of key metrics
        timestamps = [e.start_time for e in sorted_executions]
        slippages = [e.total_slippage_bps for e in sorted_executions if e.total_slippage_bps is not None]
        fill_rates = [e.fill_rate for e in sorted_executions]
        
        trends = {}
        
        # Calculate slippage trend
        if len(slippages) >= 10:
            x = np.arange(len(slippages))
            slippage_trend = np.polyfit(x, slippages, 1)[0]  # Linear trend
            trends['slippage_trend_bps_per_execution'] = slippage_trend
            trends['slippage_trend_direction'] = 'improving' if slippage_trend < -0.1 else 'deteriorating' if slippage_trend > 0.1 else 'stable'
        
        # Calculate fill rate trend
        if len(fill_rates) >= 10:
            x = np.arange(len(fill_rates))
            fill_rate_trend = np.polyfit(x, fill_rates, 1)[0]
            trends['fill_rate_trend_per_execution'] = fill_rate_trend
            trends['fill_rate_trend_direction'] = 'improving' if fill_rate_trend > 0.001 else 'deteriorating' if fill_rate_trend < -0.001 else 'stable'
        
        # Recent vs historical comparison
        if len(sorted_executions) >= 40:
            recent_20 = sorted_executions[-20:]
            historical_20 = sorted_executions[-40:-20]
            
            recent_avg_slippage = np.mean([e.total_slippage_bps for e in recent_20 if e.total_slippage_bps is not None])
            historical_avg_slippage = np.mean([e.total_slippage_bps for e in historical_20 if e.total_slippage_bps is not None])
            
            if recent_avg_slippage and historical_avg_slippage:
                trends['recent_vs_historical_slippage_change_bps'] = recent_avg_slippage - historical_avg_slippage
                trends['performance_direction'] = 'improving' if recent_avg_slippage < historical_avg_slippage else 'deteriorating'
        
        return trends
    
    def _calculate_algorithm_quality_score(
        self,
        executions: List[OrderExecution]
    ) -> float:
        """Calculate overall quality score for algorithm"""
        
        if not executions:
            return 0.0
        
        # Components of quality score
        fill_rates = [e.fill_rate for e in executions]
        slippages = [e.total_slippage_bps for e in executions if e.total_slippage_bps is not None]
        execution_times = [e.execution_duration for e in executions]
        
        # Fill rate score (0-1)
        fill_score = np.mean(fill_rates) if fill_rates else 0
        
        # Slippage score (0-1, lower slippage = higher score)
        if slippages:
            avg_slippage = np.mean(slippages)
            slippage_score = max(0, 1 - avg_slippage / 50)  # Normalize to 50 bps max
        else:
            slippage_score = 0.5
        
        # Execution time score (0-1, faster = higher score)
        if execution_times:
            avg_time = np.mean(execution_times)
            time_score = max(0, 1 - avg_time / 1800)  # Normalize to 30 minutes max
        else:
            time_score = 0.5
        
        # Weighted composite score
        quality_score = (fill_score * 0.4 + slippage_score * 0.4 + time_score * 0.2)
        
        return quality_score


class ExecutionTracker:
    """Main execution performance tracking system"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.performance_measurement = PerformanceMeasurement(config)
        self.venue_analyzer = VenueAnalyzer(config)
        self.algorithm_evaluator = AlgorithmEvaluator(config)
        
        # Execution tracking
        self.active_orders: Dict[str, OrderExecution] = {}
        self.completed_orders: deque = deque(maxlen=10000)
        self.trade_records: deque = deque(maxlen=50000)
        
        # Performance monitoring
        self.performance_alerts: List[Dict[str, Any]] = []
        self.last_performance_report: Optional[datetime] = None
        
    def start_order_tracking(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        algorithm: str,
        arrival_price: float
    ) -> None:
        """Start tracking a new order execution"""
        
        order_execution = OrderExecution(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=algorithm,
            start_time=datetime.now(),
            arrival_price=arrival_price,
            status=ExecutionStatus.ACTIVE
        )
        
        self.active_orders[order_id] = order_execution
        
        logger.info(f"Started tracking order {order_id}: {quantity} {symbol} {side}")
    
    def record_trade(
        self,
        trade_id: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        venue: str,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> None:
        """Record individual trade execution"""
        
        # Create trade record
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            venue=venue,
            algorithm=self.active_orders.get(order_id, OrderExecution('', '', '', 0, '', datetime.now())).algorithm,
            parent_order_id=order_id,
            execution_duration=kwargs.get('execution_duration', 0),
            arrival_price=kwargs.get('arrival_price', price),
            benchmark_price=kwargs.get('benchmark_price', price),
            market_volume=kwargs.get('market_volume'),
            spread_at_execution=kwargs.get('spread_at_execution')
        )
        
        # Calculate performance metrics
        trade = self.performance_measurement.calculate_trade_metrics(trade, market_data)
        
        # Add to records
        self.trade_records.append(trade)
        
        # Update parent order
        if order_id in self.active_orders:
            self.active_orders[order_id].trades.append(trade)
        
        # Add to venue analysis
        if self.config.venue_performance_tracking:
            self.venue_analyzer.add_venue_execution(venue, {
                'slippage_bps': trade.slippage_bps,
                'market_impact_bps': trade.market_impact_bps,
                'fill_rate': 1.0,  # Individual trade is fully filled
                'execution_duration': trade.execution_duration
            })
        
        logger.info(f"Recorded trade {trade_id}: {quantity} {symbol} @ {price} on {venue}")
    
    def complete_order(
        self,
        order_id: str,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Complete order tracking and calculate final metrics"""
        
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return {}
        
        order = self.active_orders[order_id]
        order.end_time = datetime.now()
        order.status = ExecutionStatus.COMPLETED
        
        # Calculate final order metrics
        order = self.performance_measurement.calculate_order_metrics(order, market_data)
        
        # Add to algorithm evaluation
        if self.config.algorithm_benchmarking:
            self.algorithm_evaluator.add_execution(order)
        
        # Move to completed orders
        self.completed_orders.append(order)
        del self.active_orders[order_id]
        
        # Check for performance alerts
        self._check_performance_alerts(order)
        
        logger.info(f"Completed order {order_id}: Fill rate {order.fill_rate:.1%}, "
                   f"Slippage {order.total_slippage_bps:.1f} bps")
        
        return {
            'order_id': order_id,
            'fill_rate': order.fill_rate,
            'average_price': order.average_execution_price,
            'total_slippage_bps': order.total_slippage_bps,
            'execution_shortfall_bps': order.execution_shortfall_bps,
            'execution_duration': order.execution_duration,
            'trades_count': len(order.trades)
        }
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""
        
        current_time = datetime.now()
        
        # Recent performance (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        recent_orders = [
            order for order in self.completed_orders
            if order.end_time and order.end_time > cutoff_time
        ]
        
        # Calculate aggregate metrics
        analytics = {
            'timestamp': current_time.isoformat(),
            'active_orders': len(self.active_orders),
            'completed_orders_24h': len(recent_orders),
            'total_completed_orders': len(self.completed_orders),
            'total_trades': len(self.trade_records)
        }
        
        if recent_orders:
            fill_rates = [order.fill_rate for order in recent_orders]
            slippages = [order.total_slippage_bps for order in recent_orders if order.total_slippage_bps is not None]
            shortfalls = [order.execution_shortfall_bps for order in recent_orders if order.execution_shortfall_bps is not None]
            execution_times = [order.execution_duration for order in recent_orders]
            
            analytics['performance_24h'] = {
                'avg_fill_rate': np.mean(fill_rates),
                'avg_slippage_bps': np.mean(slippages) if slippages else 0,
                'median_slippage_bps': np.median(slippages) if slippages else 0,
                'avg_shortfall_bps': np.mean(shortfalls) if shortfalls else 0,
                'avg_execution_time': np.mean(execution_times),
                'percentile_95_slippage': np.percentile(slippages, 95) if slippages else 0,
                'success_rate': len([fr for fr in fill_rates if fr > 0.95]) / len(fill_rates)
            }
        
        # Venue analysis
        if self.config.venue_performance_tracking:
            analytics['venue_analysis'] = self.venue_analyzer.compare_venue_performance()
        
        # Algorithm analysis
        if self.config.algorithm_benchmarking:
            analytics['algorithm_analysis'] = self.algorithm_evaluator.compare_algorithms()
        
        # Performance alerts
        analytics['active_alerts'] = len([
            alert for alert in self.performance_alerts
            if (current_time - datetime.fromisoformat(alert['timestamp'])).total_seconds() < 3600
        ])
        
        return analytics
    
    def _check_performance_alerts(self, order: OrderExecution) -> None:
        """Check for performance alerts based on order execution"""
        
        alerts = []
        current_time = datetime.now()
        
        # Poor execution alert
        if (order.total_slippage_bps and 
            order.total_slippage_bps > self.config.alert_thresholds['poor_execution_threshold']):
            alerts.append({
                'type': 'poor_execution',
                'order_id': order.order_id,
                'slippage_bps': order.total_slippage_bps,
                'threshold': self.config.alert_thresholds['poor_execution_threshold'],
                'timestamp': current_time.isoformat(),
                'severity': 'high' if order.total_slippage_bps > 50 else 'medium'
            })
        
        # Algorithm degradation alert
        algorithm_analysis = self.algorithm_evaluator.evaluate_algorithm_performance(order.algorithm)
        if (algorithm_analysis.get('status') == 'analyzed' and
            'trend_analysis' in algorithm_analysis and
            algorithm_analysis['trend_analysis'].get('recent_vs_historical_slippage_change_bps', 0) > 
            self.config.alert_thresholds['algorithm_degradation_threshold']):
            
            alerts.append({
                'type': 'algorithm_degradation',
                'algorithm': order.algorithm,
                'performance_change_bps': algorithm_analysis['trend_analysis']['recent_vs_historical_slippage_change_bps'],
                'threshold': self.config.alert_thresholds['algorithm_degradation_threshold'],
                'timestamp': current_time.isoformat(),
                'severity': 'medium'
            })
        
        # Add alerts to queue
        self.performance_alerts.extend(alerts)
        
        # Keep only recent alerts
        self.performance_alerts = [
            alert for alert in self.performance_alerts
            if (current_time - datetime.fromisoformat(alert['timestamp'])).total_seconds() < 86400
        ]
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Performance alert: {alert['type']}", alert_data=alert)
    
    def get_venue_routing_recommendation(
        self,
        order_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get venue routing recommendation for new order"""
        
        if self.config.venue_performance_tracking:
            return self.venue_analyzer.get_venue_routing_recommendation(order_characteristics)
        else:
            return {'recommendation': None, 'reason': 'venue_tracking_disabled'}
    
    def get_algorithm_recommendation(
        self,
        order_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get algorithm recommendation for new order"""
        
        if not self.config.algorithm_benchmarking:
            return {'recommendation': None, 'reason': 'algorithm_benchmarking_disabled'}
        
        algorithm_comparison = self.algorithm_evaluator.compare_algorithms()
        
        if algorithm_comparison.get('status') == 'no_algorithms_to_compare':
            return {'recommendation': None, 'reason': 'insufficient_algorithm_data'}
        
        # Simple recommendation based on best performing algorithm
        best_algorithm = algorithm_comparison.get('best_algorithm')
        
        if best_algorithm:
            best_performance = algorithm_comparison['algorithm_comparisons'][best_algorithm]
            
            return {
                'recommendation': best_algorithm,
                'expected_performance': {
                    'avg_slippage_bps': best_performance['performance_metrics']['avg_slippage_bps'],
                    'avg_fill_rate': best_performance['performance_metrics']['avg_fill_rate'],
                    'quality_score': best_performance['quality_score']
                },
                'confidence': min(best_performance['recent_sample_count'] / 50, 1.0)
            }
        
        return {'recommendation': None, 'reason': 'no_suitable_algorithm'}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        current_time = datetime.now()
        
        # Check if report is due
        if (self.last_performance_report and
            (current_time - self.last_performance_report).total_seconds() < 
            self.config.performance_reporting_frequency):
            return {'status': 'report_not_due'}
        
        # Generate comprehensive analytics
        analytics = self.get_execution_analytics()
        
        # Additional report sections
        report = {
            'report_timestamp': current_time.isoformat(),
            'report_period': f"{self.config.performance_reporting_frequency}s",
            'executive_summary': self._generate_executive_summary(analytics),
            'detailed_analytics': analytics,
            'recommendations': self._generate_recommendations(analytics),
            'performance_alerts': self.performance_alerts[-10:],  # Last 10 alerts
            'next_report_due': (current_time + timedelta(seconds=self.config.performance_reporting_frequency)).isoformat()
        }
        
        self.last_performance_report = current_time
        
        return report
    
    def _generate_executive_summary(self, analytics: Dict[str, Any]) -> Dict[str, str]:
        """Generate executive summary from analytics"""
        
        summary = {}
        
        if 'performance_24h' in analytics:
            perf = analytics['performance_24h']
            
            summary['execution_quality'] = (
                'Excellent' if perf['avg_slippage_bps'] < 5 else
                'Good' if perf['avg_slippage_bps'] < 15 else
                'Fair' if perf['avg_slippage_bps'] < 30 else
                'Poor'
            )
            
            summary['fill_performance'] = (
                'Excellent' if perf['success_rate'] > 0.95 else
                'Good' if perf['success_rate'] > 0.90 else
                'Fair' if perf['success_rate'] > 0.80 else
                'Poor'
            )
            
            summary['key_metrics'] = (
                f"Avg Slippage: {perf['avg_slippage_bps']:.1f} bps, "
                f"Success Rate: {perf['success_rate']:.1%}, "
                f"Orders: {analytics['completed_orders_24h']}"
            )
        
        return summary
    
    def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if 'performance_24h' in analytics:
            perf = analytics['performance_24h']
            
            if perf['avg_slippage_bps'] > 20:
                recommendations.append("Consider using more passive execution strategies to reduce slippage")
            
            if perf['success_rate'] < 0.90:
                recommendations.append("Review order sizing and timing to improve fill rates")
            
            if perf['avg_execution_time'] > 600:  # 10 minutes
                recommendations.append("Optimize execution algorithms for faster completion")
        
        if 'venue_analysis' in analytics and analytics['venue_analysis'].get('venues_analyzed', 0) > 1:
            venue_analysis = analytics['venue_analysis']
            best_venue = venue_analysis.get('best_venue')
            
            if best_venue:
                recommendations.append(f"Consider increasing allocation to {best_venue} based on superior performance")
        
        return recommendations