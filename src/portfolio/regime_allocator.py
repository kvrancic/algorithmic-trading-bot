"""
Regime-Specific Portfolio Allocation

Advanced regime-aware portfolio allocation with:
- Market regime detection and classification
- Regime-specific asset allocation strategies
- Dynamic allocation transitions between regimes
- Regime persistence modeling
- Multi-factor regime identification
- Adaptive allocation based on regime confidence
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats, cluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"


class RegimeDetectionMethod(Enum):
    """Regime detection methods"""
    HIDDEN_MARKOV = "hidden_markov"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    THRESHOLD_BASED = "threshold_based"
    VOLATILITY_REGIME = "volatility_regime"
    MULTI_FACTOR = "multi_factor"


@dataclass
class RegimeConfig:
    """Configuration for regime-specific allocation"""
    
    # Regime detection
    detection_method: str = "multi_factor"  # From RegimeDetectionMethod enum
    lookback_period: int = 252  # Days for regime detection
    regime_confidence_threshold: float = 0.7  # Minimum confidence for regime classification
    
    # Multi-factor regime detection
    factors: List[str] = field(default_factory=lambda: [
        "return", "volatility", "correlation", "momentum", "vix"
    ])
    
    # Volatility regime parameters
    vol_short_window: int = 20
    vol_long_window: int = 60
    vol_high_threshold: float = 0.25  # 25% annualized
    vol_low_threshold: float = 0.12   # 12% annualized
    
    # Momentum regime parameters
    momentum_window: int = 60
    momentum_threshold: float = 0.05  # 5% threshold
    
    # Correlation regime parameters
    correlation_window: int = 30
    correlation_high_threshold: float = 0.7  # High correlation threshold
    correlation_low_threshold: float = 0.3   # Low correlation threshold
    
    # Regime allocation strategies
    regime_allocations: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "bull_trending": {
            "equities": 0.70, "bonds": 0.20, "commodities": 0.05, "cash": 0.05
        },
        "bear_trending": {
            "equities": 0.20, "bonds": 0.50, "commodities": 0.10, "cash": 0.20
        },
        "high_volatility": {
            "equities": 0.30, "bonds": 0.40, "commodities": 0.15, "cash": 0.15
        },
        "low_volatility": {
            "equities": 0.60, "bonds": 0.30, "commodities": 0.05, "cash": 0.05
        },
        "crisis": {
            "equities": 0.10, "bonds": 0.60, "commodities": 0.05, "cash": 0.25
        },
        "recovery": {
            "equities": 0.55, "bonds": 0.25, "commodities": 0.10, "cash": 0.10
        },
        "normal": {
            "equities": 0.50, "bonds": 0.35, "commodities": 0.10, "cash": 0.05
        }
    })
    
    # Regime transition parameters
    enable_smooth_transitions: bool = True
    transition_speed: float = 0.1  # Speed of allocation transitions (0-1)
    min_regime_duration: int = 5   # Minimum days in regime before switching
    regime_persistence_factor: float = 0.8  # Persistence weighting
    
    # Risk management
    max_allocation_change: float = 0.20  # Maximum 20% allocation change per period
    enable_regime_confidence_scaling: bool = True
    
    # Asset class mapping
    asset_class_mapping: Dict[str, str] = field(default_factory=lambda: {
        # This maps individual assets to broader asset classes
        # Would be populated based on actual portfolio assets
    })
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.regime_confidence_threshold < 1):
            raise ValueError("regime_confidence_threshold must be between 0 and 1")
        if not (0 < self.transition_speed <= 1):
            raise ValueError("transition_speed must be between 0 and 1")
        if self.min_regime_duration < 1:
            raise ValueError("min_regime_duration must be positive")


class RegimeDetector:
    """Detect market regimes using various methods"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.regime_history: List[Dict[str, Any]] = []
        self.scaler = StandardScaler()
        
    def detect_regime(
        self,
        market_data: pd.DataFrame,
        vix_data: Optional[pd.Series] = None,
        additional_factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Detect current market regime
        
        Args:
            market_data: Market returns/price data
            vix_data: VIX or volatility index data
            additional_factors: Additional factor data
            
        Returns:
            Regime detection results
        """
        
        logger.info(f"Detecting market regime using {self.config.detection_method}")
        
        current_date = datetime.now()
        
        # Use recent data for regime detection
        if len(market_data) > self.config.lookback_period:
            recent_data = market_data.tail(self.config.lookback_period)
        else:
            recent_data = market_data
        
        if self.config.detection_method == RegimeDetectionMethod.MULTI_FACTOR.value:
            regime_result = self._multi_factor_detection(recent_data, vix_data, additional_factors)
        elif self.config.detection_method == RegimeDetectionMethod.VOLATILITY_REGIME.value:
            regime_result = self._volatility_regime_detection(recent_data)
        elif self.config.detection_method == RegimeDetectionMethod.THRESHOLD_BASED.value:
            regime_result = self._threshold_based_detection(recent_data)
        elif self.config.detection_method == RegimeDetectionMethod.GAUSSIAN_MIXTURE.value:
            regime_result = self._gaussian_mixture_detection(recent_data)
        else:
            regime_result = self._multi_factor_detection(recent_data, vix_data, additional_factors)
        
        # Apply regime persistence and confidence filtering
        final_regime = self._apply_regime_filters(regime_result)
        
        # Store in history
        self.regime_history.append({
            'timestamp': current_date.isoformat(),
            'regime': final_regime['regime'],
            'confidence': final_regime['confidence'],
            'raw_regime': regime_result['regime'],
            'factors': final_regime.get('factors', {})
        })
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        logger.info(f"Regime detected: {final_regime['regime']} (confidence: {final_regime['confidence']:.2f})")
        
        return final_regime
    
    def _multi_factor_detection(
        self,
        market_data: pd.DataFrame,
        vix_data: Optional[pd.Series] = None,
        additional_factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Multi-factor regime detection"""
        
        factor_signals = {}
        factor_weights = {
            'return': 0.25,
            'volatility': 0.25,
            'correlation': 0.20,
            'momentum': 0.15,
            'vix': 0.15
        }
        
        # Calculate portfolio returns (equal weight for simplicity)
        portfolio_returns = market_data.mean(axis=1)
        
        # Factor 1: Return regime
        if 'return' in self.config.factors:
            recent_return = portfolio_returns.tail(20).mean() * 252  # Annualized
            if recent_return > 0.15:
                factor_signals['return'] = ('bull_trending', 0.8)
            elif recent_return < -0.10:
                factor_signals['return'] = ('bear_trending', 0.8)
            else:
                factor_signals['return'] = ('normal', 0.6)
        
        # Factor 2: Volatility regime
        if 'volatility' in self.config.factors:
            vol_signal = self._calculate_volatility_signal(portfolio_returns)
            factor_signals['volatility'] = vol_signal
        
        # Factor 3: Correlation regime
        if 'correlation' in self.config.factors:
            corr_signal = self._calculate_correlation_signal(market_data)
            factor_signals['correlation'] = corr_signal
        
        # Factor 4: Momentum regime
        if 'momentum' in self.config.factors:
            momentum_signal = self._calculate_momentum_signal(portfolio_returns)
            factor_signals['momentum'] = momentum_signal
        
        # Factor 5: VIX regime
        if 'vix' in self.config.factors and vix_data is not None:
            vix_signal = self._calculate_vix_signal(vix_data)
            factor_signals['vix'] = vix_signal
        
        # Aggregate factor signals
        regime_scores = {}
        total_weight = 0
        
        for factor, (regime, confidence) in factor_signals.items():
            weight = factor_weights.get(factor, 0)
            total_weight += weight
            
            if regime not in regime_scores:
                regime_scores[regime] = 0
            regime_scores[regime] += weight * confidence
        
        # Normalize scores
        if total_weight > 0:
            regime_scores = {k: v / total_weight for k, v in regime_scores.items()}
        
        # Select regime with highest score
        if regime_scores:
            best_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[best_regime]
        else:
            best_regime = MarketRegime.UNCERTAIN.value
            confidence = 0.5
        
        return {
            'regime': best_regime,
            'confidence': confidence,
            'factors': factor_signals,
            'regime_scores': regime_scores
        }
    
    def _calculate_volatility_signal(self, returns: pd.Series) -> Tuple[str, float]:
        """Calculate volatility-based regime signal"""
        
        short_vol = returns.tail(self.config.vol_short_window).std() * np.sqrt(252)
        long_vol = returns.tail(self.config.vol_long_window).std() * np.sqrt(252)
        current_vol = short_vol
        
        if current_vol > self.config.vol_high_threshold:
            if current_vol > long_vol * 1.5:  # Spike in volatility
                return (MarketRegime.CRISIS.value, 0.9)
            else:
                return (MarketRegime.HIGH_VOLATILITY.value, 0.8)
        elif current_vol < self.config.vol_low_threshold:
            return (MarketRegime.LOW_VOLATILITY.value, 0.8)
        else:
            return (MarketRegime.NORMAL.value, 0.6)
    
    def _calculate_correlation_signal(self, market_data: pd.DataFrame) -> Tuple[str, float]:
        """Calculate correlation-based regime signal"""
        
        recent_data = market_data.tail(self.config.correlation_window)
        corr_matrix = recent_data.corr()
        
        # Average pairwise correlation (excluding diagonal)
        n_assets = len(corr_matrix)
        avg_correlation = (corr_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
        
        if avg_correlation > self.config.correlation_high_threshold:
            return (MarketRegime.CRISIS.value, 0.8)  # High correlation suggests crisis
        elif avg_correlation < self.config.correlation_low_threshold:
            return (MarketRegime.LOW_VOLATILITY.value, 0.7)  # Low correlation suggests calm markets
        else:
            return (MarketRegime.NORMAL.value, 0.6)
    
    def _calculate_momentum_signal(self, returns: pd.Series) -> Tuple[str, float]:
        """Calculate momentum-based regime signal"""
        
        recent_returns = returns.tail(self.config.momentum_window)
        cumulative_return = (1 + recent_returns).prod() - 1
        
        if cumulative_return > self.config.momentum_threshold:
            return (MarketRegime.BULL_TRENDING.value, 0.7)
        elif cumulative_return < -self.config.momentum_threshold:
            return (MarketRegime.BEAR_TRENDING.value, 0.7)
        else:
            return (MarketRegime.NORMAL.value, 0.6)
    
    def _calculate_vix_signal(self, vix_data: pd.Series) -> Tuple[str, float]:
        """Calculate VIX-based regime signal"""
        
        if len(vix_data) == 0:
            return (MarketRegime.NORMAL.value, 0.5)
        
        current_vix = vix_data.iloc[-1]
        vix_ma = vix_data.tail(20).mean()
        
        if current_vix > 30:  # High VIX
            if current_vix > vix_ma * 1.3:  # VIX spike
                return (MarketRegime.CRISIS.value, 0.9)
            else:
                return (MarketRegime.HIGH_VOLATILITY.value, 0.8)
        elif current_vix < 15:  # Low VIX
            return (MarketRegime.LOW_VOLATILITY.value, 0.8)
        elif 25 <= current_vix <= 30:  # Elevated VIX
            return (MarketRegime.BEAR_TRENDING.value, 0.7)
        else:
            return (MarketRegime.NORMAL.value, 0.6)
    
    def _volatility_regime_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Simple volatility-based regime detection"""
        
        portfolio_returns = market_data.mean(axis=1)
        vol_signal = self._calculate_volatility_signal(portfolio_returns)
        
        return {
            'regime': vol_signal[0],
            'confidence': vol_signal[1],
            'method': 'volatility_based'
        }
    
    def _threshold_based_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Threshold-based regime detection"""
        
        portfolio_returns = market_data.mean(axis=1)
        recent_return = portfolio_returns.tail(20).mean() * 252
        recent_vol = portfolio_returns.tail(20).std() * np.sqrt(252)
        
        # Simple threshold rules
        if recent_return > 0.15 and recent_vol < 0.20:
            regime = MarketRegime.BULL_TRENDING.value
            confidence = 0.8
        elif recent_return < -0.10 and recent_vol > 0.25:
            regime = MarketRegime.BEAR_TRENDING.value
            confidence = 0.8
        elif recent_vol > 0.30:
            regime = MarketRegime.CRISIS.value
            confidence = 0.9
        elif recent_vol < 0.12:
            regime = MarketRegime.LOW_VOLATILITY.value
            confidence = 0.7
        else:
            regime = MarketRegime.NORMAL.value
            confidence = 0.6
        
        return {
            'regime': regime,
            'confidence': confidence,
            'recent_return': recent_return,
            'recent_volatility': recent_vol
        }
    
    def _gaussian_mixture_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Gaussian Mixture Model regime detection"""
        
        try:
            # Prepare features
            portfolio_returns = market_data.mean(axis=1)
            features = []
            
            # Rolling statistics as features
            for window in [5, 20, 60]:
                if len(portfolio_returns) >= window:
                    rolling_mean = portfolio_returns.rolling(window).mean()
                    rolling_vol = portfolio_returns.rolling(window).std()
                    features.extend([rolling_mean.iloc[-1], rolling_vol.iloc[-1]])
            
            if len(features) < 4:  # Need minimum features
                return self._threshold_based_detection(market_data)
            
            # Use historical data to fit GMM
            historical_features = []
            for i in range(max(60, len(portfolio_returns) // 4), len(portfolio_returns)):
                period_features = []
                for window in [5, 20, 60]:
                    if i >= window:
                        period_return = portfolio_returns.iloc[i-window:i].mean()
                        period_vol = portfolio_returns.iloc[i-window:i].std()
                        period_features.extend([period_return, period_vol])
                
                if len(period_features) == len(features):
                    historical_features.append(period_features)
            
            if len(historical_features) < 20:  # Need sufficient data
                return self._threshold_based_detection(market_data)
            
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=3, random_state=42)
            X = self.scaler.fit_transform(historical_features)
            gmm.fit(X)
            
            # Predict current regime
            current_features = self.scaler.transform([features])
            regime_probs = gmm.predict_proba(current_features)[0]
            regime_id = np.argmax(regime_probs)
            confidence = regime_probs[regime_id]
            
            # Map regime IDs to meaningful names (simplified)
            regime_names = [
                MarketRegime.LOW_VOLATILITY.value,
                MarketRegime.NORMAL.value,
                MarketRegime.HIGH_VOLATILITY.value
            ]
            
            regime = regime_names[regime_id] if regime_id < len(regime_names) else MarketRegime.NORMAL.value
            
            return {
                'regime': regime,
                'confidence': confidence,
                'regime_probabilities': regime_probs.tolist(),
                'method': 'gaussian_mixture'
            }
            
        except Exception as e:
            logger.warning("Gaussian mixture detection failed, falling back to threshold method", error=str(e))
            return self._threshold_based_detection(market_data)
    
    def _apply_regime_filters(self, regime_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime persistence and confidence filters"""
        
        current_regime = regime_result['regime']
        current_confidence = regime_result['confidence']
        
        # Check if we have recent regime history
        if len(self.regime_history) > 0:
            last_regime = self.regime_history[-1]['regime']
            
            # Apply persistence factor if regime is changing
            if current_regime != last_regime:
                # Check minimum duration requirement
                same_regime_count = 0
                for hist in reversed(self.regime_history):
                    if hist['regime'] == last_regime:
                        same_regime_count += 1
                    else:
                        break
                
                # If we haven't been in the current regime long enough, persist
                if same_regime_count < self.config.min_regime_duration:
                    # Reduce confidence for regime change
                    adjusted_confidence = current_confidence * (1 - self.config.regime_persistence_factor)
                    
                    if adjusted_confidence < self.config.regime_confidence_threshold:
                        # Stay in previous regime
                        return {
                            'regime': last_regime,
                            'confidence': self.regime_history[-1]['confidence'] * 0.9,
                            'persistence_applied': True
                        }
        
        # Apply confidence threshold
        if current_confidence < self.config.regime_confidence_threshold:
            # Default to normal regime with moderate confidence
            return {
                'regime': MarketRegime.NORMAL.value,
                'confidence': 0.6,
                'low_confidence_fallback': True
            }
        
        return regime_result
    
    def get_regime_transition_probability(self) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities from history"""
        
        if len(self.regime_history) < 10:
            return {}
        
        # Count transitions
        transitions = {}
        for i in range(1, len(self.regime_history)):
            from_regime = self.regime_history[i-1]['regime']
            to_regime = self.regime_history[i]['regime']
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0
            
            transitions[from_regime][to_regime] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for from_regime, to_counts in transitions.items():
            total = sum(to_counts.values())
            transition_probs[from_regime] = {
                to_regime: count / total for to_regime, count in to_counts.items()
            }
        
        return transition_probs


class RegimeSpecificAllocator:
    """Main regime-specific portfolio allocator"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.regime_detector = RegimeDetector(config)
        self.current_allocation: Optional[Dict[str, float]] = None
        self.allocation_history: List[Dict[str, Any]] = []
        
    def allocate_portfolio(
        self,
        market_data: pd.DataFrame,
        asset_universe: List[str],
        vix_data: Optional[pd.Series] = None,
        additional_factors: Optional[pd.DataFrame] = None,
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform regime-specific portfolio allocation
        
        Args:
            market_data: Market data for regime detection
            asset_universe: Available assets for allocation
            vix_data: VIX or volatility index data
            additional_factors: Additional factor data
            current_portfolio: Current portfolio weights
            
        Returns:
            Allocation results with regime-specific weights
        """
        
        logger.info("Performing regime-specific portfolio allocation", assets=len(asset_universe))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'assets': asset_universe,
            'method': 'regime_specific',
            'allocation': {},
            'regime_info': {},
            'transition_info': {}
        }
        
        try:
            # Step 1: Detect current market regime
            regime_info = self.regime_detector.detect_regime(
                market_data, vix_data, additional_factors
            )
            
            # Step 2: Get regime-specific allocation
            regime_allocation = self._get_regime_allocation(
                regime_info['regime'], regime_info['confidence']
            )
            
            # Step 3: Map to specific assets
            asset_allocation = self._map_to_assets(
                regime_allocation, asset_universe
            )
            
            # Step 4: Apply smooth transitions if enabled
            if self.config.enable_smooth_transitions and self.current_allocation:
                asset_allocation = self._apply_smooth_transition(
                    self.current_allocation, asset_allocation, regime_info['confidence']
                )
            
            # Step 5: Apply risk constraints
            final_allocation = self._apply_risk_constraints(
                asset_allocation, current_portfolio
            )
            
            # Update current allocation
            self.current_allocation = final_allocation
            
            # Store results
            result['allocation'] = final_allocation
            result['regime_info'] = regime_info
            result['transition_info'] = {
                'smooth_transition_applied': self.config.enable_smooth_transitions and self.current_allocation is not None,
                'regime_allocation': regime_allocation,
                'asset_mapping_applied': True
            }
            
            # Store in history
            self.allocation_history.append({
                'timestamp': result['timestamp'],
                'regime': regime_info['regime'],
                'confidence': regime_info['confidence'],
                'allocation': final_allocation,
                'max_allocation': max(final_allocation.values()) if final_allocation else 0
            })
            
            # Keep only recent history
            if len(self.allocation_history) > 100:
                self.allocation_history = self.allocation_history[-100:]
            
            logger.info(f"Regime allocation completed for regime: {regime_info['regime']} "
                       f"(confidence: {regime_info['confidence']:.2f})")
            
        except Exception as e:
            logger.error("Error in regime-specific allocation", error=str(e))
            result['error'] = str(e)
            # Fallback to equal weights
            result['allocation'] = {asset: 1.0/len(asset_universe) for asset in asset_universe}
        
        return result
    
    def _get_regime_allocation(self, regime: str, confidence: float) -> Dict[str, float]:
        """Get allocation for specific regime"""
        
        # Get base allocation for regime
        if regime in self.config.regime_allocations:
            base_allocation = self.config.regime_allocations[regime].copy()
        else:
            # Default to normal regime allocation
            base_allocation = self.config.regime_allocations[MarketRegime.NORMAL.value].copy()
        
        # Scale allocation based on confidence if enabled
        if self.config.enable_regime_confidence_scaling:
            # Blend with normal allocation based on confidence
            normal_allocation = self.config.regime_allocations[MarketRegime.NORMAL.value]
            
            scaled_allocation = {}
            for asset_class in base_allocation:
                base_weight = base_allocation[asset_class]
                normal_weight = normal_allocation.get(asset_class, 0)
                
                # Higher confidence = more regime-specific allocation
                scaled_weight = confidence * base_weight + (1 - confidence) * normal_weight
                scaled_allocation[asset_class] = scaled_weight
            
            # Renormalize to ensure weights sum to 1
            total_weight = sum(scaled_allocation.values())
            if total_weight > 0:
                scaled_allocation = {k: v / total_weight for k, v in scaled_allocation.items()}
            
            return scaled_allocation
        
        return base_allocation
    
    def _map_to_assets(
        self,
        regime_allocation: Dict[str, float],
        asset_universe: List[str]
    ) -> Dict[str, float]:
        """Map asset class allocations to specific assets"""
        
        # If asset class mapping is not configured, use equal weights
        if not self.config.asset_class_mapping:
            return {asset: 1.0/len(asset_universe) for asset in asset_universe}
        
        asset_allocation = {}
        
        # Group assets by class
        asset_classes = {}
        for asset in asset_universe:
            asset_class = self.config.asset_class_mapping.get(asset, 'unknown')
            if asset_class not in asset_classes:
                asset_classes[asset_class] = []
            asset_classes[asset_class].append(asset)
        
        # Allocate within each asset class
        for asset_class, class_weight in regime_allocation.items():
            if asset_class in asset_classes:
                assets_in_class = asset_classes[asset_class]
                weight_per_asset = class_weight / len(assets_in_class)
                
                for asset in assets_in_class:
                    asset_allocation[asset] = weight_per_asset
        
        # Handle unallocated assets (equal weight from remaining allocation)
        allocated_assets = set(asset_allocation.keys())
        unallocated_assets = [a for a in asset_universe if a not in allocated_assets]
        
        if unallocated_assets:
            total_allocated = sum(asset_allocation.values())
            remaining_weight = max(0, 1.0 - total_allocated)
            weight_per_unallocated = remaining_weight / len(unallocated_assets)
            
            for asset in unallocated_assets:
                asset_allocation[asset] = weight_per_unallocated
        
        # Ensure all assets are included
        for asset in asset_universe:
            if asset not in asset_allocation:
                asset_allocation[asset] = 0.0
        
        # Renormalize
        total_weight = sum(asset_allocation.values())
        if total_weight > 0:
            asset_allocation = {k: v / total_weight for k, v in asset_allocation.items()}
        
        return asset_allocation
    
    def _apply_smooth_transition(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        regime_confidence: float
    ) -> Dict[str, float]:
        """Apply smooth transitions between allocations"""
        
        # Adjust transition speed based on regime confidence
        effective_speed = self.config.transition_speed * regime_confidence
        
        smooth_allocation = {}
        
        for asset in target_allocation:
            current_weight = current_allocation.get(asset, 0)
            target_weight = target_allocation[asset]
            
            # Linear interpolation between current and target
            new_weight = current_weight + effective_speed * (target_weight - current_weight)
            smooth_allocation[asset] = new_weight
        
        # Renormalize
        total_weight = sum(smooth_allocation.values())
        if total_weight > 0:
            smooth_allocation = {k: v / total_weight for k, v in smooth_allocation.items()}
        
        return smooth_allocation
    
    def _apply_risk_constraints(
        self,
        allocation: Dict[str, float],
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Apply risk management constraints to allocation"""
        
        constrained_allocation = allocation.copy()
        
        # Apply maximum allocation change constraint
        if current_portfolio and self.config.max_allocation_change < 1.0:
            for asset in constrained_allocation:
                current_weight = current_portfolio.get(asset, 0)
                target_weight = constrained_allocation[asset]
                max_change = self.config.max_allocation_change
                
                # Limit the change
                if abs(target_weight - current_weight) > max_change:
                    if target_weight > current_weight:
                        constrained_allocation[asset] = current_weight + max_change
                    else:
                        constrained_allocation[asset] = max(0, current_weight - max_change)
        
        # Renormalize after constraints
        total_weight = sum(constrained_allocation.values())
        if total_weight > 0:
            constrained_allocation = {k: v / total_weight for k, v in constrained_allocation.items()}
        
        return constrained_allocation
    
    def update_asset_class_mapping(self, mapping: Dict[str, str]) -> None:
        """Update asset class mapping"""
        
        self.config.asset_class_mapping.update(mapping)
        logger.info("Asset class mapping updated", new_mappings=len(mapping))
    
    def update_regime_allocations(self, allocations: Dict[str, Dict[str, float]]) -> None:
        """Update regime-specific allocations"""
        
        self.config.regime_allocations.update(allocations)
        logger.info("Regime allocations updated", regimes=list(allocations.keys()))
    
    def get_regime_dashboard(self) -> Dict[str, Any]:
        """Get regime allocation dashboard data"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'detection_method': self.config.detection_method,
                'lookback_period': self.config.lookback_period,
                'confidence_threshold': self.config.regime_confidence_threshold,
                'smooth_transitions': self.config.enable_smooth_transitions,
                'transition_speed': self.config.transition_speed
            },
            'current_allocation': self.current_allocation,
            'regime_history': self.regime_detector.regime_history[-20:],
            'allocation_history': self.allocation_history[-10:],
            'transition_probabilities': self.regime_detector.get_regime_transition_probability(),
            'regime_allocations': self.config.regime_allocations,
            'performance_stats': {
                'total_allocations': len(self.allocation_history),
                'regime_distribution': {
                    regime: len([h for h in self.allocation_history if h['regime'] == regime])
                    for regime in set(h['regime'] for h in self.allocation_history)
                } if self.allocation_history else {},
                'avg_confidence': np.mean([h['confidence'] for h in self.allocation_history]) if self.allocation_history else 0
            }
        }