"""
Advanced Market Impact Minimization Model

Sophisticated market impact modeling and minimization:
- Kyle's Lambda model for permanent price impact
- Almgren-Chriss implementation shortfall optimization
- Temporary impact decay modeling with microstructure
- Cross-venue impact analysis and optimal routing
- Real-time impact monitoring and adaptive execution
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import optimize
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MarketImpactConfig:
    """Configuration for market impact modeling"""
    
    # Kyle's Lambda parameters
    kyle_lambda_estimation_window: int = 252  # Trading days for lambda estimation
    kyle_lambda_decay_factor: float = 0.05    # Daily decay of impact information
    
    # Almgren-Chriss parameters
    temporary_impact_decay: float = 0.5       # Temporary impact half-life (trades)
    permanent_impact_factor: float = 0.1      # Fraction of temporary impact that becomes permanent
    risk_aversion: float = 1e-6               # Risk aversion parameter (A in Almgren-Chriss)
    
    # Microstructure parameters
    bid_ask_spread_impact: float = 0.5        # Fraction of spread captured as impact
    market_depth_levels: int = 5              # Order book levels to analyze
    volume_participation_limit: float = 0.30  # Maximum participation in volume
    
    # Cross-venue parameters
    venue_impact_correlation: float = 0.7     # Impact correlation across venues
    venue_latency_penalty: float = 0.0001    # Cost per millisecond latency difference
    
    # Dynamic parameters
    enable_adaptive_impact: bool = True       # Adapt to real-time conditions
    impact_measurement_window: int = 50       # Trades for impact measurement
    regime_detection_window: int = 100        # Periods for regime detection
    
    # Optimization parameters
    optimization_horizon: int = 300           # Seconds for optimization
    reoptimization_frequency: int = 60        # Seconds between reoptimizations
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.volume_participation_limit <= 1):
            raise ValueError("volume_participation_limit must be between 0 and 1")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")


class MarketImpactModel(ABC):
    """Abstract base class for market impact models"""
    
    @abstractmethod
    def estimate_impact(
        self,
        order_size: float,
        market_conditions: Dict[str, Any],
        execution_schedule: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Estimate market impact for given order and conditions"""
        pass
    
    @abstractmethod
    def optimize_execution(
        self,
        total_size: float,
        time_horizon: int,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize execution schedule to minimize impact"""
        pass


class KyleLambdaModel(MarketImpactModel):
    """Kyle's Lambda model for permanent price impact estimation"""
    
    def __init__(self, config: MarketImpactConfig):
        self.config = config
        self.lambda_estimates: Dict[str, float] = {}
        self.lambda_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def estimate_lambda(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Estimate Kyle's Lambda (price impact per unit volume)
        
        Lambda = Cov(ΔP, Q) / Var(Q)
        where ΔP = price change, Q = signed order flow
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            volume_data: Trading volume data
            order_flow_data: Optional signed order flow data
            
        Returns:
            Estimated lambda value
        """
        
        logger.info(f"Estimating Kyle's Lambda for {symbol}")
        
        # Calculate price changes
        if 'close' in price_data.columns:
            price_changes = price_data['close'].pct_change().dropna()
        else:
            logger.warning("No close price available, using first column")
            price_changes = price_data.iloc[:, 0].pct_change().dropna()
        
        # Estimate signed order flow if not provided
        if order_flow_data is None:
            signed_volume = self._estimate_signed_volume(price_data, volume_data)
        else:
            signed_volume = order_flow_data.iloc[:, 0]
        
        # Align data
        aligned_data = pd.concat([price_changes, signed_volume], axis=1).dropna()
        
        if len(aligned_data) < 50:
            logger.warning(f"Insufficient data for lambda estimation: {len(aligned_data)}")
            return 1e-6  # Default small lambda
        
        price_changes_aligned = aligned_data.iloc[:, 0]
        signed_volume_aligned = aligned_data.iloc[:, 1]
        
        # Calculate Kyle's Lambda
        covariance = price_changes_aligned.cov(signed_volume_aligned)
        volume_variance = signed_volume_aligned.var()
        
        if volume_variance > 0:
            lambda_estimate = abs(covariance / volume_variance)
        else:
            lambda_estimate = 1e-6
        
        # Apply bounds to lambda estimate
        lambda_estimate = np.clip(lambda_estimate, 1e-8, 1e-3)
        
        # Store estimate with timestamp
        self.lambda_estimates[symbol] = lambda_estimate
        
        if symbol not in self.lambda_history:
            self.lambda_history[symbol] = []
        
        self.lambda_history[symbol].append((datetime.now(), lambda_estimate))
        
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=self.config.kyle_lambda_estimation_window)
        self.lambda_history[symbol] = [
            (date, val) for date, val in self.lambda_history[symbol]
            if date > cutoff_date
        ]
        
        logger.info(f"Kyle's Lambda for {symbol}: {lambda_estimate:.2e}")
        
        return lambda_estimate
    
    def _estimate_signed_volume(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> pd.Series:
        """Estimate signed volume using price tick rule"""
        
        if 'close' in price_data.columns:
            prices = price_data['close']
        else:
            prices = price_data.iloc[:, 0]
        
        if 'volume' in volume_data.columns:
            volumes = volume_data['volume']
        else:
            volumes = volume_data.iloc[:, 0]
        
        # Price tick rule: positive if price increased, negative if decreased
        price_changes = prices.diff()
        
        # Sign determination
        signs = np.where(price_changes > 0, 1, 
                np.where(price_changes < 0, -1, 0))
        
        # Handle zero price changes (use previous non-zero change)
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i-1]
        
        signed_volume = volumes * signs
        
        return pd.Series(signed_volume, index=volumes.index)
    
    def estimate_impact(
        self,
        order_size: float,
        market_conditions: Dict[str, Any],
        execution_schedule: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Estimate permanent impact using Kyle's Lambda"""
        
        symbol = market_conditions.get('symbol', 'UNKNOWN')
        lambda_val = self.lambda_estimates.get(symbol, 1e-6)
        
        # Permanent impact = Lambda * Order Size
        permanent_impact = lambda_val * order_size
        
        # Apply decay if execution is scheduled over time
        if execution_schedule:
            total_duration = sum(slice_info.get('duration', 60) for slice_info in execution_schedule)
            decay_factor = np.exp(-self.config.kyle_lambda_decay_factor * total_duration / 3600)
            permanent_impact *= decay_factor
        
        return {
            'permanent_impact': permanent_impact,
            'lambda': lambda_val,
            'impact_bps': permanent_impact * 10000,  # Convert to basis points
            'confidence': min(len(self.lambda_history.get(symbol, [])) / 100, 1.0)
        }
    
    def optimize_execution(
        self,
        total_size: float,
        time_horizon: int,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize execution to minimize permanent impact"""
        
        symbol = market_conditions.get('symbol', 'UNKNOWN')
        lambda_val = self.lambda_estimates.get(symbol, 1e-6)
        
        # For Kyle's model, optimal strategy is to spread execution evenly
        # to minimize cumulative impact
        
        n_slices = max(1, time_horizon // 60)  # One slice per minute
        slice_size = total_size / n_slices
        slice_duration = time_horizon / n_slices
        
        execution_schedule = []
        cumulative_impact = 0
        
        for i in range(n_slices):
            slice_impact = lambda_val * slice_size
            cumulative_impact += slice_impact
            
            execution_schedule.append({
                'slice_id': i + 1,
                'size': slice_size,
                'start_time': i * slice_duration,
                'duration': slice_duration,
                'expected_impact': slice_impact,
                'cumulative_impact': cumulative_impact
            })
        
        return {
            'execution_schedule': execution_schedule,
            'total_expected_impact': cumulative_impact,
            'impact_bps': cumulative_impact * 10000,
            'strategy': 'kyle_lambda_optimal'
        }


class AlmgrenChrissModel(MarketImpactModel):
    """Almgren-Chriss implementation shortfall model"""
    
    def __init__(self, config: MarketImpactConfig):
        self.config = config
        self.volatility_estimates: Dict[str, float] = {}
        self.temporary_impact_params: Dict[str, Dict[str, float]] = {}
        
    def estimate_impact_parameters(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        execution_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Estimate parameters for Almgren-Chriss model
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            volume_data: Volume data
            execution_data: Historical execution data
            
        Returns:
            Dictionary of estimated parameters
        """
        
        logger.info(f"Estimating Almgren-Chriss parameters for {symbol}")
        
        # Estimate volatility
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
        else:
            returns = price_data.iloc[:, 0].pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        self.volatility_estimates[symbol] = volatility
        
        # Estimate temporary impact parameters from execution data
        if execution_data is not None and len(execution_data) > 20:
            temp_impact_params = self._estimate_temporary_impact(execution_data, volume_data)
        else:
            # Use default parameters
            temp_impact_params = {
                'eta': 1e-6,  # Temporary impact coefficient
                'gamma': 1e-7  # Permanent impact coefficient
            }
        
        self.temporary_impact_params[symbol] = temp_impact_params
        
        params = {
            'volatility': volatility,
            'eta': temp_impact_params['eta'],
            'gamma': temp_impact_params['gamma'],
            'risk_aversion': self.config.risk_aversion
        }
        
        logger.info(f"Almgren-Chriss parameters for {symbol}: {params}")
        
        return params
    
    def _estimate_temporary_impact(
        self,
        execution_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Estimate temporary impact parameters from execution data"""
        
        try:
            # Assume execution_data has columns: 'order_size', 'execution_shortfall'
            if 'order_size' not in execution_data.columns or 'execution_shortfall' not in execution_data.columns:
                logger.warning("Execution data missing required columns, using defaults")
                return {'eta': 1e-6, 'gamma': 1e-7}
            
            order_sizes = execution_data['order_size']
            shortfalls = execution_data['execution_shortfall']
            
            # Align with volume data
            if 'volume' in volume_data.columns:
                avg_volume = volume_data['volume'].mean()
            else:
                avg_volume = volume_data.iloc[:, 0].mean()
            
            # Normalize order sizes by average volume
            normalized_sizes = order_sizes / avg_volume
            
            # Fit linear relationship: shortfall = eta * normalized_size
            if len(normalized_sizes) > 10 and normalized_sizes.var() > 0:
                eta = shortfalls.cov(normalized_sizes) / normalized_sizes.var()
                eta = max(1e-8, abs(eta))  # Ensure positive and reasonable
            else:
                eta = 1e-6
            
            # Estimate permanent impact as fraction of temporary
            gamma = eta * self.config.permanent_impact_factor
            
            return {'eta': eta, 'gamma': gamma}
            
        except Exception as e:
            logger.warning(f"Error estimating temporary impact: {e}")
            return {'eta': 1e-6, 'gamma': 1e-7}
    
    def estimate_impact(
        self,
        order_size: float,
        market_conditions: Dict[str, Any],
        execution_schedule: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Estimate impact using Almgren-Chriss model"""
        
        symbol = market_conditions.get('symbol', 'UNKNOWN')
        
        # Get model parameters
        if symbol not in self.temporary_impact_params:
            # Use default parameters
            params = {
                'volatility': 0.20,
                'eta': 1e-6,
                'gamma': 1e-7,
                'risk_aversion': self.config.risk_aversion
            }
        else:
            params = {
                'volatility': self.volatility_estimates.get(symbol, 0.20),
                'eta': self.temporary_impact_params[symbol]['eta'],
                'gamma': self.temporary_impact_params[symbol]['gamma'],
                'risk_aversion': self.config.risk_aversion
            }
        
        avg_volume = market_conditions.get('avg_volume', 100000)
        
        # Normalize order size by volume
        normalized_size = order_size / avg_volume if avg_volume > 0 else 0.1
        
        # Calculate impacts
        temporary_impact = params['eta'] * normalized_size
        permanent_impact = params['gamma'] * normalized_size
        
        # If execution schedule provided, calculate time-weighted impact
        if execution_schedule:
            total_temp_impact = 0
            total_perm_impact = 0
            
            for slice_info in execution_schedule:
                slice_size = slice_info.get('size', order_size / len(execution_schedule))
                slice_normalized = slice_size / avg_volume if avg_volume > 0 else 0.1
                
                # Temporary impact decays over time
                duration = slice_info.get('duration', 60)
                decay_factor = np.exp(-duration / (self.config.temporary_impact_decay * 60))
                
                slice_temp_impact = params['eta'] * slice_normalized * decay_factor
                slice_perm_impact = params['gamma'] * slice_normalized
                
                total_temp_impact += slice_temp_impact
                total_perm_impact += slice_perm_impact
            
            temporary_impact = total_temp_impact
            permanent_impact = total_perm_impact
        
        total_impact = temporary_impact + permanent_impact
        
        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': total_impact,
            'impact_bps': total_impact * 10000,
            'volatility_cost': params['volatility'] * np.sqrt(order_size / avg_volume) if avg_volume > 0 else 0,
            'model_params': params
        }
    
    def optimize_execution(
        self,
        total_size: float,
        time_horizon: int,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize execution using Almgren-Chriss implementation shortfall
        
        Minimizes: Expected Cost + λ * Risk
        where Expected Cost = Permanent Impact + Temporary Impact + Timing Risk
        """
        
        symbol = market_conditions.get('symbol', 'UNKNOWN')
        avg_volume = market_conditions.get('avg_volume', 100000)
        
        # Get model parameters
        if symbol in self.temporary_impact_params:
            eta = self.temporary_impact_params[symbol]['eta']
            gamma = self.temporary_impact_params[symbol]['gamma']
            sigma = self.volatility_estimates.get(symbol, 0.20)
        else:
            eta, gamma, sigma = 1e-6, 1e-7, 0.20
        
        # Almgren-Chriss optimal strategy parameters
        A = self.config.risk_aversion
        T = time_horizon / 3600  # Convert to hours
        
        # Calculate optimal trajectory
        # κ = sqrt(A * σ² / η)
        kappa = np.sqrt(A * sigma**2 / eta) if eta > 0 else 1.0
        
        # τ = κ * T
        tau = kappa * T
        
        # Optimal execution rate: n(t) = X * sinh(κ(T-t)) / sinh(κT)
        n_slices = max(5, min(20, time_horizon // 30))  # 5-20 slices, ~30 seconds each
        slice_duration = time_horizon / n_slices
        
        execution_schedule = []
        cumulative_executed = 0
        
        for i in range(n_slices):
            t = i * slice_duration / 3600  # Convert to hours
            
            # Optimal holding at time t
            if tau > 1e-6:
                holding_ratio = np.sinh(kappa * (T - t)) / np.sinh(tau)
            else:
                # Linear case when tau is small
                holding_ratio = (T - t) / T
            
            target_holding = total_size * holding_ratio
            slice_size = cumulative_executed + target_holding
            
            if i > 0:
                slice_size = slice_size - cumulative_executed
            
            # Ensure non-negative slice size
            slice_size = max(0, slice_size)
            
            if slice_size > 0:
                # Calculate expected impact for this slice
                normalized_slice = slice_size / avg_volume if avg_volume > 0 else 0.1
                slice_temp_impact = eta * normalized_slice
                slice_perm_impact = gamma * normalized_slice
                
                execution_schedule.append({
                    'slice_id': i + 1,
                    'start_time': i * slice_duration,
                    'duration': slice_duration,
                    'size': slice_size,
                    'target_holding': target_holding,
                    'expected_temp_impact': slice_temp_impact,
                    'expected_perm_impact': slice_perm_impact,
                    'execution_rate': slice_size / (slice_duration / 3600) if slice_duration > 0 else 0
                })
                
                cumulative_executed += slice_size
        
        # Calculate total expected cost
        total_temp_impact = sum(s['expected_temp_impact'] for s in execution_schedule)
        total_perm_impact = sum(s['expected_perm_impact'] for s in execution_schedule)
        total_expected_cost = total_temp_impact + total_perm_impact
        
        # Calculate implementation shortfall risk
        timing_risk = sigma * np.sqrt(total_size / avg_volume * T) if avg_volume > 0 else 0
        
        return {
            'execution_schedule': execution_schedule,
            'total_expected_cost': total_expected_cost,
            'temporary_impact': total_temp_impact,
            'permanent_impact': total_perm_impact,
            'timing_risk': timing_risk,
            'impact_bps': total_expected_cost * 10000,
            'strategy': 'almgren_chriss_optimal',
            'model_params': {
                'kappa': kappa,
                'tau': tau,
                'risk_aversion': A,
                'volatility': sigma,
                'eta': eta,
                'gamma': gamma
            }
        }


class MarketImpactMinimizer:
    """Main market impact minimization system"""
    
    def __init__(self, config: MarketImpactConfig):
        self.config = config
        self.kyle_model = KyleLambdaModel(config)
        self.almgren_chriss_model = AlmgrenChrissModel(config)
        self.impact_history: List[Dict[str, Any]] = []
        self.venue_impact_map: Dict[str, Dict[str, float]] = {}
        
    def calibrate_models(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        execution_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calibrate both impact models using historical data
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data (OHLCV)
            execution_data: Historical execution data
            
        Returns:
            Calibration results for both models
        """
        
        logger.info(f"Calibrating market impact models for {symbol}")
        
        # Separate price and volume data
        price_data = market_data[['open', 'high', 'low', 'close']] if all(col in market_data.columns for col in ['open', 'high', 'low', 'close']) else market_data
        volume_data = market_data[['volume']] if 'volume' in market_data.columns else pd.DataFrame({'volume': np.ones(len(market_data))}, index=market_data.index)
        
        # Calibrate Kyle's Lambda model
        kyle_lambda = self.kyle_model.estimate_lambda(
            symbol, price_data, volume_data, execution_data
        )
        
        # Calibrate Almgren-Chriss model
        ac_params = self.almgren_chriss_model.estimate_impact_parameters(
            symbol, price_data, volume_data, execution_data
        )
        
        calibration_results = {
            'symbol': symbol,
            'calibration_timestamp': datetime.now().isoformat(),
            'kyle_lambda': kyle_lambda,
            'almgren_chriss_params': ac_params,
            'data_points': len(market_data),
            'calibration_quality': self._assess_calibration_quality(market_data, execution_data)
        }
        
        logger.info(f"Model calibration completed for {symbol}")
        
        return calibration_results
    
    def estimate_total_impact(
        self,
        order_size: float,
        market_conditions: Dict[str, Any],
        execution_strategy: str = "optimal",
        time_horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate total market impact using ensemble of models
        
        Args:
            order_size: Size of order to execute
            market_conditions: Current market conditions
            execution_strategy: Strategy for execution ("optimal", "twap", "vwap")
            time_horizon: Time horizon for execution in seconds
            
        Returns:
            Comprehensive impact estimate
        """
        
        symbol = market_conditions.get('symbol', 'UNKNOWN')
        time_horizon = time_horizon or self.config.optimization_horizon
        
        # Get estimates from both models
        kyle_estimate = self.kyle_model.estimate_impact(order_size, market_conditions)
        ac_estimate = self.almgren_chriss_model.estimate_impact(order_size, market_conditions)
        
        # Generate optimal execution schedule
        if execution_strategy == "optimal":
            # Use Almgren-Chriss for optimization
            execution_plan = self.almgren_chriss_model.optimize_execution(
                order_size, time_horizon, market_conditions
            )
        else:
            # Simple uniform execution
            n_slices = max(1, time_horizon // 60)
            execution_plan = {
                'execution_schedule': [
                    {
                        'slice_id': i + 1,
                        'start_time': i * (time_horizon / n_slices),
                        'duration': time_horizon / n_slices,
                        'size': order_size / n_slices
                    }
                    for i in range(n_slices)
                ],
                'strategy': execution_strategy
            }
        
        # Ensemble impact estimate (weighted average)
        kyle_weight = 0.4
        ac_weight = 0.6
        
        ensemble_impact = (
            kyle_weight * kyle_estimate.get('permanent_impact', 0) +
            ac_weight * ac_estimate.get('total_impact', 0)
        )
        
        # Add venue-specific adjustments
        venue_adjustment = self._calculate_venue_impact_adjustment(
            symbol, order_size, market_conditions
        )
        
        final_impact = ensemble_impact + venue_adjustment
        
        # Store in history
        impact_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'order_size': order_size,
            'execution_strategy': execution_strategy,
            'time_horizon': time_horizon,
            'kyle_impact': kyle_estimate.get('permanent_impact', 0),
            'ac_impact': ac_estimate.get('total_impact', 0),
            'ensemble_impact': ensemble_impact,
            'venue_adjustment': venue_adjustment,
            'final_impact': final_impact,
            'market_conditions': market_conditions
        }
        
        self.impact_history.append(impact_record)
        
        # Keep only recent history
        if len(self.impact_history) > 1000:
            self.impact_history = self.impact_history[-1000:]
        
        return {
            'total_estimated_impact': final_impact,
            'impact_bps': final_impact * 10000,
            'execution_plan': execution_plan,
            'model_breakdown': {
                'kyle_lambda': kyle_estimate,
                'almgren_chriss': ac_estimate,
                'ensemble_weight': {'kyle': kyle_weight, 'ac': ac_weight}
            },
            'venue_adjustment': venue_adjustment,
            'confidence_score': self._calculate_confidence_score(kyle_estimate, ac_estimate),
            'risk_assessment': self._assess_impact_risk(final_impact, market_conditions)
        }
    
    def optimize_cross_venue_execution(
        self,
        order_size: float,
        available_venues: List[Dict[str, Any]],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize execution across multiple venues to minimize total impact
        
        Args:
            order_size: Total order size
            available_venues: List of venue information with liquidity and costs
            market_conditions: Current market conditions
            
        Returns:
            Optimal venue allocation and execution plan
        """
        
        logger.info(f"Optimizing cross-venue execution for {order_size} shares")
        
        venue_allocations = {}
        total_expected_impact = 0
        
        # Sort venues by expected impact (ascending)
        venue_impacts = []
        
        for venue in available_venues:
            venue_conditions = market_conditions.copy()
            venue_conditions.update(venue)
            
            # Estimate impact for full order at this venue
            venue_impact = self.estimate_total_impact(
                order_size, venue_conditions, "optimal"
            )
            
            venue_impacts.append({
                'venue': venue['name'],
                'capacity': venue.get('liquidity', order_size),
                'impact_per_share': venue_impact['total_estimated_impact'],
                'total_cost': venue_impact['total_estimated_impact'] * order_size,
                'latency': venue.get('latency_ms', 10)
            })
        
        # Sort by impact per share
        venue_impacts.sort(key=lambda x: x['impact_per_share'])
        
        # Allocate order across venues (greedy allocation)
        remaining_size = order_size
        
        for venue_info in venue_impacts:
            if remaining_size <= 0:
                break
            
            venue_name = venue_info['venue']
            venue_capacity = min(venue_info['capacity'], remaining_size)
            allocation_size = min(venue_capacity, remaining_size * 0.5)  # Max 50% to single venue
            
            if allocation_size > order_size * 0.01:  # Minimum 1% allocation
                venue_allocations[venue_name] = {
                    'size': allocation_size,
                    'expected_impact': venue_info['impact_per_share'] * allocation_size,
                    'percentage': allocation_size / order_size,
                    'latency_ms': venue_info['latency']
                }
                
                total_expected_impact += venue_info['impact_per_share'] * allocation_size
                remaining_size -= allocation_size
        
        # If there's remaining size, allocate to venue with most capacity
        if remaining_size > 0:
            best_venue = max(venue_impacts, key=lambda x: x['capacity'])
            venue_name = best_venue['venue']
            
            if venue_name in venue_allocations:
                venue_allocations[venue_name]['size'] += remaining_size
                venue_allocations[venue_name]['expected_impact'] += best_venue['impact_per_share'] * remaining_size
                venue_allocations[venue_name]['percentage'] = venue_allocations[venue_name]['size'] / order_size
            else:
                venue_allocations[venue_name] = {
                    'size': remaining_size,
                    'expected_impact': best_venue['impact_per_share'] * remaining_size,
                    'percentage': remaining_size / order_size,
                    'latency_ms': best_venue['latency']
                }
            
            total_expected_impact += best_venue['impact_per_share'] * remaining_size
        
        return {
            'venue_allocations': venue_allocations,
            'total_expected_impact': total_expected_impact,
            'impact_bps': total_expected_impact * 10000,
            'venues_used': len(venue_allocations),
            'impact_savings': self._calculate_impact_savings(venue_impacts, venue_allocations, order_size),
            'execution_complexity': len(venue_allocations)
        }
    
    def _assess_calibration_quality(
        self,
        market_data: pd.DataFrame,
        execution_data: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Assess quality of model calibration"""
        
        quality_metrics = {
            'data_sufficiency': min(len(market_data) / 252, 1.0),  # 1 year of data = 1.0
            'execution_data_available': 1.0 if execution_data is not None and len(execution_data) > 50 else 0.0,
            'price_data_quality': 1.0 if 'close' in market_data.columns else 0.5,
            'volume_data_quality': 1.0 if 'volume' in market_data.columns else 0.0
        }
        
        overall_quality = np.mean(list(quality_metrics.values()))
        quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics
    
    def _calculate_venue_impact_adjustment(
        self,
        symbol: str,
        order_size: float,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate venue-specific impact adjustments"""
        
        venue = market_conditions.get('venue', 'default')
        
        # Base adjustment is zero
        adjustment = 0.0
        
        # Apply venue-specific factors if available
        if symbol in self.venue_impact_map and venue in self.venue_impact_map[symbol]:
            venue_factor = self.venue_impact_map[symbol][venue]
            adjustment = venue_factor * order_size * 1e-6  # Small adjustment
        
        # Add latency penalty
        latency_ms = market_conditions.get('latency_ms', 10)
        if latency_ms > 50:  # High latency penalty
            adjustment += (latency_ms - 50) * self.config.venue_latency_penalty
        
        return adjustment
    
    def _calculate_confidence_score(
        self,
        kyle_estimate: Dict[str, Any],
        ac_estimate: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for impact estimate"""
        
        # Factors affecting confidence
        kyle_confidence = kyle_estimate.get('confidence', 0.5)
        
        # Model agreement (lower difference = higher confidence)
        kyle_impact = kyle_estimate.get('permanent_impact', 0)
        ac_impact = ac_estimate.get('total_impact', 0)
        
        if kyle_impact > 0 and ac_impact > 0:
            agreement = 1 - abs(kyle_impact - ac_impact) / max(kyle_impact, ac_impact)
        else:
            agreement = 0.5
        
        # Combined confidence
        confidence = 0.6 * kyle_confidence + 0.4 * agreement
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _assess_impact_risk(
        self,
        estimated_impact: float,
        market_conditions: Dict[str, Any]
    ) -> str:
        """Assess risk level of estimated impact"""
        
        impact_bps = estimated_impact * 10000
        
        # Risk thresholds in basis points
        if impact_bps < 5:
            return "low"
        elif impact_bps < 15:
            return "medium"
        elif impact_bps < 30:
            return "high"
        else:
            return "very_high"
    
    def _calculate_impact_savings(
        self,
        venue_impacts: List[Dict[str, Any]],
        venue_allocations: Dict[str, Dict[str, Any]],
        total_order_size: float
    ) -> Dict[str, float]:
        """Calculate impact savings from cross-venue execution"""
        
        # Single venue cost (worst case)
        single_venue_cost = max(v['impact_per_share'] for v in venue_impacts) * total_order_size
        
        # Multi-venue cost
        multi_venue_cost = sum(alloc['expected_impact'] for alloc in venue_allocations.values())
        
        # Savings
        absolute_savings = single_venue_cost - multi_venue_cost
        percentage_savings = absolute_savings / single_venue_cost if single_venue_cost > 0 else 0
        
        return {
            'absolute_savings': absolute_savings,
            'percentage_savings': percentage_savings,
            'single_venue_cost': single_venue_cost,
            'multi_venue_cost': multi_venue_cost
        }
    
    def get_impact_analytics(self) -> Dict[str, Any]:
        """Get comprehensive impact analytics and model performance"""
        
        if not self.impact_history:
            return {'status': 'no_data'}
        
        recent_impacts = [record['final_impact'] for record in self.impact_history[-100:]]
        
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'total_estimates': len(self.impact_history),
            'recent_avg_impact_bps': np.mean(recent_impacts) * 10000 if recent_impacts else 0,
            'recent_impact_volatility': np.std(recent_impacts) * 10000 if len(recent_impacts) > 1 else 0,
            'model_usage': {
                'kyle_lambda_symbols': len(self.kyle_model.lambda_estimates),
                'almgren_chriss_symbols': len(self.almgren_chriss_model.temporary_impact_params)
            },
            'venue_analytics': self._analyze_venue_performance(),
            'impact_distribution': self._analyze_impact_distribution()
        }
        
        return analytics
    
    def _analyze_venue_performance(self) -> Dict[str, Any]:
        """Analyze performance across different venues"""
        
        venue_performance = {}
        
        for record in self.impact_history[-200:]:  # Recent records
            venue = record['market_conditions'].get('venue', 'unknown')
            impact = record['final_impact']
            
            if venue not in venue_performance:
                venue_performance[venue] = []
            
            venue_performance[venue].append(impact)
        
        # Calculate statistics for each venue
        venue_stats = {}
        for venue, impacts in venue_performance.items():
            if len(impacts) > 5:  # Minimum data requirement
                venue_stats[venue] = {
                    'avg_impact_bps': np.mean(impacts) * 10000,
                    'impact_volatility': np.std(impacts) * 10000,
                    'sample_count': len(impacts),
                    'percentile_75': np.percentile(impacts, 75) * 10000,
                    'percentile_25': np.percentile(impacts, 25) * 10000
                }
        
        return venue_stats
    
    def _analyze_impact_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of impact estimates"""
        
        if len(self.impact_history) < 10:
            return {}
        
        impacts = [record['final_impact'] * 10000 for record in self.impact_history[-200:]]
        
        return {
            'mean_bps': np.mean(impacts),
            'median_bps': np.median(impacts),
            'std_bps': np.std(impacts),
            'percentiles': {
                '10': np.percentile(impacts, 10),
                '25': np.percentile(impacts, 25),
                '75': np.percentile(impacts, 75),
                '90': np.percentile(impacts, 90),
                '95': np.percentile(impacts, 95)
            },
            'risk_buckets': {
                'low_risk': len([i for i in impacts if i < 5]),
                'medium_risk': len([i for i in impacts if 5 <= i < 15]),
                'high_risk': len([i for i in impacts if 15 <= i < 30]),
                'very_high_risk': len([i for i in impacts if i >= 30])
            }
        }