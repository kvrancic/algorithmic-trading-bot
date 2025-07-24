"""
Correlation Penalty Management System

Advanced correlation management featuring:
- Dynamic correlation monitoring
- Sector-based correlation analysis  
- Time-varying correlation detection
- Correlation-based position limits
- Diversification optimization
- Regime-specific correlation adjustments
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation penalty system"""
    
    # Correlation limits
    max_pairwise_correlation: float = 0.7  # Maximum correlation between any two positions
    max_avg_correlation: float = 0.5  # Maximum average correlation across portfolio
    max_sector_correlation: float = 0.8  # Maximum correlation within same sector
    
    # Time windows for correlation calculation
    short_window: int = 30   # Days for short-term correlation
    medium_window: int = 90  # Days for medium-term correlation  
    long_window: int = 252   # Days for long-term correlation
    
    # Dynamic correlation detection
    enable_regime_adjustment: bool = True
    volatility_threshold: float = 0.25  # Threshold for high-vol regime
    correlation_breakpoint: float = 0.8   # Correlation spike threshold
    
    # Penalty application
    penalty_method: str = "exponential"  # "linear", "exponential", "threshold"
    penalty_strength: float = 2.0  # Strength of correlation penalty
    min_penalty: float = 0.1  # Minimum position size after penalty
    
    # Clustering settings
    enable_clustering: bool = True
    max_cluster_size: int = 5  # Maximum positions per cluster
    cluster_correlation_threshold: float = 0.6
    
    # Sector analysis
    enable_sector_analysis: bool = True
    sector_concentration_limit: float = 0.4  # Max 40% in any sector
    
    # Rolling correlation monitoring
    correlation_alert_threshold: float = 0.75  # Alert when correlation exceeds this
    correlation_window_shift: int = 5  # Days between correlation updates
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.max_pairwise_correlation <= 1):
            raise ValueError("max_pairwise_correlation must be between 0 and 1")
        if self.penalty_method not in ["linear", "exponential", "threshold"]:
            raise ValueError("penalty_method must be 'linear', 'exponential', or 'threshold'")


class CorrelationAnalyzer:
    """Analyze correlations and correlation regimes"""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        
    def calculate_correlation_metrics(
        self,
        returns_data: pd.DataFrame,
        positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive correlation metrics
        
        Args:
            returns_data: Historical returns data
            positions: Current positions (optional)
            
        Returns:
            Correlation analysis results
        """
        
        if len(returns_data) < self.config.short_window:
            return {'error': 'insufficient_data'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'correlation_matrices': {},
            'summary_stats': {},
            'regime_analysis': {},
            'position_analysis': {}
        }
        
        # Calculate correlation matrices for different time windows
        windows = {
            'short': self.config.short_window,
            'medium': self.config.medium_window,
            'long': self.config.long_window
        }
        
        for window_name, window_size in windows.items():
            if len(returns_data) >= window_size:
                recent_data = returns_data.tail(window_size)
                corr_matrix = recent_data.corr()
                
                results['correlation_matrices'][window_name] = {
                    'matrix': corr_matrix,
                    'avg_correlation': self._calculate_avg_correlation(corr_matrix),
                    'max_correlation': self._calculate_max_correlation(corr_matrix),
                    'min_correlation': self._calculate_min_correlation(corr_matrix),
                    'correlation_distribution': self._analyze_correlation_distribution(corr_matrix)
                }
        
        # Summary statistics
        if 'medium' in results['correlation_matrices']:
            base_corr = results['correlation_matrices']['medium']
            results['summary_stats'] = {
                'avg_correlation': base_corr['avg_correlation'],
                'max_correlation': base_corr['max_correlation'],
                'correlation_violations': self._count_correlation_violations(
                    base_corr['matrix']
                ),
                'diversification_ratio': self._calculate_diversification_ratio(
                    returns_data.tail(self.config.medium_window)
                )
            }
        
        # Regime analysis
        if self.config.enable_regime_adjustment:
            results['regime_analysis'] = self._analyze_correlation_regimes(returns_data)
        
        # Position-specific analysis
        if positions:
            results['position_analysis'] = self._analyze_position_correlations(
                returns_data, positions
            )
        
        return results
    
    def _calculate_avg_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average correlation excluding diagonal"""
        values = corr_matrix.values
        mask = ~np.eye(values.shape[0], dtype=bool)
        return float(np.mean(np.abs(values[mask])))
    
    def _calculate_max_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate maximum correlation excluding diagonal"""
        values = corr_matrix.values
        np.fill_diagonal(values, 0)  # Remove diagonal
        return float(np.max(np.abs(values)))
    
    def _calculate_min_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate minimum correlation excluding diagonal"""
        values = corr_matrix.values
        mask = ~np.eye(values.shape[0], dtype=bool)
        return float(np.min(np.abs(values[mask])))
    
    def _analyze_correlation_distribution(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Analyze distribution of correlation values"""
        values = corr_matrix.values
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)
        correlations = values[mask]
        
        return {
            'mean': float(np.mean(correlations)),
            'std': float(np.std(correlations)),
            'skewness': float(stats.skew(correlations)),
            'kurtosis': float(stats.kurtosis(correlations)),
            'percentile_25': float(np.percentile(correlations, 25)),
            'percentile_75': float(np.percentile(correlations, 75)),
            'high_correlation_pct': float(np.mean(np.abs(correlations) > self.config.max_pairwise_correlation))
        }
    
    def _count_correlation_violations(self, corr_matrix: pd.DataFrame) -> Dict[str, int]:
        """Count violations of correlation limits"""
        values = np.abs(corr_matrix.values)
        np.fill_diagonal(values, 0)  # Ignore diagonal
        
        return {
            'pairwise_violations': int(np.sum(values > self.config.max_pairwise_correlation)),
            'high_correlation_pairs': int(np.sum(values > 0.8)),
            'moderate_correlation_pairs': int(np.sum((values > 0.5) & (values <= 0.8)))
        }
    
    def _calculate_diversification_ratio(self, returns_data: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        if len(returns_data) < 20:
            return 1.0
        
        # Equal-weighted portfolio
        weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
        
        # Individual volatilities
        individual_vols = returns_data.std()
        weighted_avg_vol = np.sum(weights * individual_vols)
        
        # Portfolio volatility
        cov_matrix = returns_data.cov()
        portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
        
        # Diversification ratio = weighted average vol / portfolio vol
        return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
    
    def _analyze_correlation_regimes(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation regimes (normal vs crisis)"""
        
        if len(returns_data) < self.config.long_window:
            return {}
        
        # Calculate rolling correlation
        window_size = self.config.medium_window
        rolling_correlations = []
        
        for i in range(window_size, len(returns_data)):
            window_data = returns_data.iloc[i-window_size:i]
            corr_matrix = window_data.corr()
            avg_corr = self._calculate_avg_correlation(corr_matrix)
            rolling_correlations.append(avg_corr)
        
        rolling_corr_series = pd.Series(rolling_correlations)
        
        # Detect high correlation periods (crisis regimes)
        high_corr_threshold = self.config.correlation_breakpoint
        high_corr_periods = rolling_corr_series > high_corr_threshold
        
        # Calculate regime statistics
        current_regime = "high_correlation" if rolling_corr_series.iloc[-1] > high_corr_threshold else "normal"
        
        return {
            'current_regime': current_regime,
            'current_avg_correlation': float(rolling_corr_series.iloc[-1]),
            'regime_stability': float(np.std(rolling_corr_series.tail(10))),
            'high_correlation_frequency': float(np.mean(high_corr_periods)),
            'avg_normal_correlation': float(rolling_corr_series[~high_corr_periods].mean()),
            'avg_crisis_correlation': float(rolling_corr_series[high_corr_periods].mean()) if high_corr_periods.any() else 0
        }
    
    def _analyze_position_correlations(
        self,
        returns_data: pd.DataFrame,
        positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze correlations specific to current positions"""
        
        position_symbols = [s for s in positions.keys() if s in returns_data.columns and positions[s] != 0]
        
        if len(position_symbols) < 2:
            return {'num_positions': len(position_symbols)}
        
        position_returns = returns_data[position_symbols].tail(self.config.medium_window)
        position_corr = position_returns.corr()
        
        # Weight correlations by position sizes
        weights = np.array([abs(positions[s]) for s in position_symbols])
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted average correlation
        weighted_corr = 0.0
        total_weight = 0.0
        
        for i in range(len(position_symbols)):
            for j in range(i+1, len(position_symbols)):
                corr_value = position_corr.iloc[i, j]
                weight = weights[i] * weights[j]
                weighted_corr += abs(corr_value) * weight
                total_weight += weight
        
        weighted_avg_corr = weighted_corr / total_weight if total_weight > 0 else 0
        
        return {
            'num_positions': len(position_symbols),
            'position_symbols': position_symbols,
            'avg_position_correlation': self._calculate_avg_correlation(position_corr),
            'weighted_avg_correlation': float(weighted_avg_corr),
            'max_position_correlation': self._calculate_max_correlation(position_corr),
            'correlation_violations': self._count_correlation_violations(position_corr),
            'position_weights': {symbol: float(weights[i]) for i, symbol in enumerate(position_symbols)}
        }


class CorrelationClustering:
    """Cluster assets based on correlation for diversification"""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        
    def perform_correlation_clustering(
        self,
        returns_data: pd.DataFrame,
        method: str = "ward"
    ) -> Dict[str, Any]:
        """
        Cluster assets based on correlation
        
        Args:
            returns_data: Historical returns
            method: Clustering method
            
        Returns:
            Clustering results
        """
        
        if len(returns_data.columns) < 3:
            return {'error': 'insufficient_assets'}
        
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        # Convert correlation to distance (1 - |correlation|)
        distance_matrix = 1 - np.abs(corr_matrix.values)
        
        # Ensure symmetric and positive semidefinite
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed form for linkage
        condensed_distances = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=method)
        
        # Determine optimal number of clusters
        max_clusters = min(len(returns_data.columns) // 2, 10)
        
        cluster_results = {}
        for n_clusters in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Analyze cluster quality
            cluster_quality = self._analyze_cluster_quality(
                corr_matrix, clusters, n_clusters
            )
            
            cluster_results[n_clusters] = {
                'clusters': clusters,
                'quality_metrics': cluster_quality
            }
        
        # Select optimal clustering
        optimal_n_clusters = self._select_optimal_clusters(cluster_results)
        optimal_clusters = cluster_results[optimal_n_clusters]['clusters']
        
        # Create cluster mapping
        cluster_mapping = {}
        for i, symbol in enumerate(returns_data.columns):
            cluster_id = optimal_clusters[i]
            if cluster_id not in cluster_mapping:
                cluster_mapping[cluster_id] = []
            cluster_mapping[cluster_id].append(symbol)
        
        return {
            'optimal_n_clusters': optimal_n_clusters,
            'cluster_mapping': cluster_mapping,
            'cluster_quality': cluster_results[optimal_n_clusters]['quality_metrics'],
            'linkage_matrix': linkage_matrix,
            'all_cluster_results': cluster_results
        }
    
    def _analyze_cluster_quality(
        self,
        corr_matrix: pd.DataFrame,
        clusters: np.ndarray,
        n_clusters: int
    ) -> Dict[str, float]:
        """Analyze quality of clustering solution"""
        
        within_cluster_corr = []
        between_cluster_corr = []
        
        for cluster_id in range(1, n_clusters + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 1:
                # Within-cluster correlations
                cluster_corr_matrix = corr_matrix.iloc[cluster_indices, cluster_indices]
                cluster_values = cluster_corr_matrix.values
                mask = np.triu(np.ones_like(cluster_values, dtype=bool), k=1)
                within_cluster_corr.extend(np.abs(cluster_values[mask]))
        
        # Between-cluster correlations
        for i in range(1, n_clusters + 1):
            for j in range(i + 1, n_clusters + 1):
                cluster_i_indices = np.where(clusters == i)[0]
                cluster_j_indices = np.where(clusters == j)[0]
                
                between_corr_matrix = corr_matrix.iloc[cluster_i_indices, cluster_j_indices]
                between_cluster_corr.extend(np.abs(between_corr_matrix.values.flatten()))
        
        # Calculate quality metrics
        avg_within_cluster = np.mean(within_cluster_corr) if within_cluster_corr else 0
        avg_between_cluster = np.mean(between_cluster_corr) if between_cluster_corr else 0
        
        # Silhouette-like score (higher within-cluster correlation, lower between-cluster)
        silhouette_score = avg_within_cluster - avg_between_cluster
        
        return {
            'avg_within_cluster_correlation': float(avg_within_cluster),
            'avg_between_cluster_correlation': float(avg_between_cluster),
            'silhouette_score': float(silhouette_score),
            'cluster_separation': float(avg_within_cluster / avg_between_cluster) if avg_between_cluster > 0 else float('inf')
        }
    
    def _select_optimal_clusters(self, cluster_results: Dict[int, Dict]) -> int:
        """Select optimal number of clusters"""
        
        # Simple heuristic: maximize silhouette score
        best_score = -float('inf')
        best_n_clusters = 2
        
        for n_clusters, results in cluster_results.items():
            score = results['quality_metrics']['silhouette_score']
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
        
        return best_n_clusters


class CorrelationPenaltyManager:
    """Main correlation penalty management system"""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.analyzer = CorrelationAnalyzer(config)
        self.clustering = CorrelationClustering(config)
        self.correlation_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
    def calculate_correlation_penalties(
        self,
        positions: Dict[str, float],
        returns_data: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate correlation-based penalties for position sizes
        
        Args:
            positions: Current positions {symbol: weight}
            returns_data: Historical returns data
            sector_mapping: Optional sector mapping {symbol: sector}
            
        Returns:
            Correlation penalty analysis and adjustments
        """
        
        logger.info("Calculating correlation penalties", positions=len(positions))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_positions': positions.copy(),
            'penalties': {},
            'adjusted_positions': {},
            'correlation_analysis': {},
            'cluster_analysis': {},
            'sector_analysis': {},
            'alerts': []
        }
        
        try:
            # Get position symbols with data
            position_symbols = [s for s in positions.keys() if s in returns_data.columns and positions[s] != 0]
            
            if len(position_symbols) < 2:
                logger.info("Insufficient positions for correlation analysis")
                result['adjusted_positions'] = positions.copy()
                return result
            
            # Perform correlation analysis
            result['correlation_analysis'] = self.analyzer.calculate_correlation_metrics(
                returns_data, positions
            )
            
            # Calculate individual penalties
            penalties = self._calculate_pairwise_penalties(
                position_symbols, returns_data, positions
            )
            result['penalties'] = penalties
            
            # Apply clustering analysis if enabled
            if self.config.enable_clustering and len(position_symbols) >= 3:
                result['cluster_analysis'] = self._analyze_position_clusters(
                    position_symbols, returns_data, positions
                )
                
                # Apply cluster-based penalties
                cluster_penalties = self._calculate_cluster_penalties(
                    result['cluster_analysis'], positions
                )
                
                # Combine with pairwise penalties
                for symbol in penalties:
                    if symbol in cluster_penalties:
                        penalties[symbol] = min(penalties[symbol], cluster_penalties[symbol])
            
            # Apply sector analysis if enabled
            if self.config.enable_sector_analysis and sector_mapping:
                result['sector_analysis'] = self._analyze_sector_concentration(
                    positions, sector_mapping
                )
                
                sector_penalties = self._calculate_sector_penalties(
                    result['sector_analysis'], positions
                )
                
                # Combine with existing penalties
                for symbol in penalties:
                    if symbol in sector_penalties:
                        penalties[symbol] = min(penalties[symbol], sector_penalties[symbol])
            
            # Apply penalties to positions
            result['adjusted_positions'] = self._apply_penalties(positions, penalties)
            
            # Generate alerts
            result['alerts'] = self._generate_correlation_alerts(result)
            
            # Store in history
            self._update_history(result)
            
            logger.info("Correlation penalties calculated",
                       avg_penalty=np.mean(list(penalties.values())) if penalties else 1.0,
                       max_penalty=1 - min(penalties.values()) if penalties else 0)
            
        except Exception as e:
            logger.error("Error calculating correlation penalties", error=str(e))
            result['error'] = str(e)
            result['adjusted_positions'] = positions.copy()
        
        return result
    
    def _calculate_pairwise_penalties(
        self,
        symbols: List[str],
        returns_data: pd.DataFrame,
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate penalties based on pairwise correlations"""
        
        penalties = {symbol: 1.0 for symbol in symbols}
        
        if len(symbols) < 2:
            return penalties
        
        # Get correlation matrix for positions
        position_returns = returns_data[symbols].tail(self.config.medium_window)
        corr_matrix = position_returns.corr()
        
        # Calculate penalty for each position based on its correlations with others
        for i, symbol in enumerate(symbols):
            max_penalty = 1.0
            
            for j, other_symbol in enumerate(symbols):
                if i != j:
                    correlation = abs(corr_matrix.iloc[i, j])
                    
                    if correlation > self.config.max_pairwise_correlation:
                        # Calculate penalty based on excess correlation
                        excess_corr = correlation - self.config.max_pairwise_correlation
                        
                        if self.config.penalty_method == "linear":
                            penalty = 1.0 - (excess_corr * self.config.penalty_strength)
                        elif self.config.penalty_method == "exponential":
                            penalty = np.exp(-excess_corr * self.config.penalty_strength)
                        else:  # threshold
                            penalty = 0.5  # Fixed 50% penalty
                        
                        # Weight by other position size
                        other_weight = abs(positions[other_symbol])
                        weighted_penalty = 1.0 - (1.0 - penalty) * other_weight
                        
                        max_penalty = min(max_penalty, weighted_penalty)
            
            # Apply minimum penalty constraint
            penalties[symbol] = max(max_penalty, self.config.min_penalty)
        
        return penalties
    
    def _analyze_position_clusters(
        self,
        symbols: List[str],
        returns_data: pd.DataFrame,
        positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze clustering of current positions"""
        
        position_returns = returns_data[symbols]
        clustering_result = self.clustering.perform_correlation_clustering(position_returns)
        
        # Add position weights to cluster analysis
        if 'cluster_mapping' in clustering_result:
            weighted_clusters = {}
            for cluster_id, cluster_symbols in clustering_result['cluster_mapping'].items():
                cluster_weight = sum(abs(positions[s]) for s in cluster_symbols if s in positions)
                weighted_clusters[cluster_id] = {
                    'symbols': cluster_symbols,
                    'total_weight': cluster_weight,
                    'avg_weight': cluster_weight / len(cluster_symbols),
                    'num_positions': len(cluster_symbols)
                }
            
            clustering_result['weighted_clusters'] = weighted_clusters
        
        return clustering_result
    
    def _calculate_cluster_penalties(
        self,
        cluster_analysis: Dict[str, Any],
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate penalties based on cluster concentration"""
        
        penalties = {}
        
        if 'weighted_clusters' not in cluster_analysis:
            return penalties
        
        for cluster_id, cluster_data in cluster_analysis['weighted_clusters'].items():
            cluster_weight = cluster_data['total_weight']
            
            # Apply penalty if cluster is too concentrated
            if (cluster_weight > 1.0 / len(cluster_analysis['weighted_clusters']) * 1.5 and
                cluster_data['num_positions'] > self.config.max_cluster_size):
                
                # Calculate penalty based on concentration
                concentration_penalty = min(0.5, cluster_weight * 0.5)
                
                # Apply to all positions in cluster
                for symbol in cluster_data['symbols']:
                    penalties[symbol] = 1.0 - concentration_penalty
        
        return penalties
    
    def _analyze_sector_concentration(
        self,
        positions: Dict[str, float],
        sector_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Analyze sector concentration in positions"""
        
        sector_weights = {}
        position_sectors = {}
        
        for symbol, weight in positions.items():
            if symbol in sector_mapping and weight != 0:
                sector = sector_mapping[symbol]
                position_sectors[symbol] = sector
                
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += abs(weight)
        
        return {
            'sector_weights': sector_weights,
            'position_sectors': position_sectors,
            'max_sector_weight': max(sector_weights.values()) if sector_weights else 0,
            'num_sectors': len(sector_weights),
            'sector_concentration': max(sector_weights.values()) if sector_weights else 0
        }
    
    def _calculate_sector_penalties(
        self,
        sector_analysis: Dict[str, Any],
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate penalties based on sector concentration"""
        
        penalties = {}
        sector_weights = sector_analysis['sector_weights']
        position_sectors = sector_analysis['position_sectors']
        
        for symbol, sector in position_sectors.items():
            sector_weight = sector_weights[sector]
            
            if sector_weight > self.config.sector_concentration_limit:
                excess_concentration = sector_weight - self.config.sector_concentration_limit
                penalty = 1.0 - (excess_concentration * 2.0)  # Linear penalty
                penalties[symbol] = max(penalty, self.config.min_penalty)
            else:
                penalties[symbol] = 1.0
        
        return penalties
    
    def _apply_penalties(
        self,
        positions: Dict[str, float],
        penalties: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply calculated penalties to positions"""
        
        adjusted_positions = {}
        
        for symbol, original_size in positions.items():
            if symbol in penalties:
                penalty_factor = penalties[symbol]
                adjusted_positions[symbol] = original_size * penalty_factor
            else:
                adjusted_positions[symbol] = original_size
        
        return adjusted_positions
    
    def _generate_correlation_alerts(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on correlation analysis"""
        
        alerts = []
        
        # Check for high correlations
        corr_analysis = analysis_result.get('correlation_analysis', {})
        if 'summary_stats' in corr_analysis:
            max_corr = corr_analysis['summary_stats'].get('max_correlation', 0)
            if max_corr > self.config.correlation_alert_threshold:
                alerts.append({
                    'severity': 'HIGH',
                    'type': 'HIGH_CORRELATION',
                    'message': f"Maximum pairwise correlation {max_corr:.2%} exceeds threshold",
                    'correlation': max_corr,
                    'threshold': self.config.correlation_alert_threshold
                })
        
        # Check for sector concentration
        sector_analysis = analysis_result.get('sector_analysis', {})
        if 'max_sector_weight' in sector_analysis:
            max_sector = sector_analysis['max_sector_weight']
            if max_sector > self.config.sector_concentration_limit:
                alerts.append({
                    'severity': 'MEDIUM',
                    'type': 'SECTOR_CONCENTRATION',
                    'message': f"Sector concentration {max_sector:.2%} exceeds limit",
                    'concentration': max_sector,
                    'limit': self.config.sector_concentration_limit
                })
        
        # Check for regime changes
        if 'regime_analysis' in corr_analysis:
            regime = corr_analysis['regime_analysis'].get('current_regime')
            if regime == 'high_correlation':
                alerts.append({
                    'severity': 'HIGH',
                    'type': 'CORRELATION_REGIME',
                    'message': 'High correlation regime detected - increased risk',
                    'regime': regime
                })
        
        return alerts
    
    def _update_history(self, analysis_result: Dict[str, Any]):
        """Update correlation analysis history"""
        
        history_entry = {
            'timestamp': analysis_result['timestamp'],
            'max_correlation': 0,
            'avg_correlation': 0,
            'num_positions': len(analysis_result['original_positions']),
            'total_penalties': 0
        }
        
        # Extract key metrics
        corr_analysis = analysis_result.get('correlation_analysis', {})
        if 'summary_stats' in corr_analysis:
            history_entry['max_correlation'] = corr_analysis['summary_stats'].get('max_correlation', 0)
            history_entry['avg_correlation'] = corr_analysis['summary_stats'].get('avg_correlation', 0)
        
        penalties = analysis_result.get('penalties', {})
        if penalties:
            history_entry['total_penalties'] = sum(1 - p for p in penalties.values())
        
        self.correlation_history.append(history_entry)
        
        # Keep only recent history
        if len(self.correlation_history) > 1000:
            self.correlation_history = self.correlation_history[-1000:]
    
    def get_correlation_dashboard(self) -> Dict[str, Any]:
        """Get correlation monitoring dashboard"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_pairwise_correlation': self.config.max_pairwise_correlation,
                'max_avg_correlation': self.config.max_avg_correlation,
                'penalty_method': self.config.penalty_method,
                'clustering_enabled': self.config.enable_clustering,
                'sector_analysis_enabled': self.config.enable_sector_analysis
            },
            'recent_history': self.correlation_history[-20:] if len(self.correlation_history) > 20 else self.correlation_history,
            'recent_alerts': [alert for alert in self.alerts if 
                            datetime.fromisoformat(alert.get('timestamp', '2020-01-01')) > 
                            datetime.now() - timedelta(hours=24)],
            'statistics': {
                'avg_max_correlation': np.mean([h['max_correlation'] for h in self.correlation_history]) if self.correlation_history else 0,
                'avg_penalties_applied': np.mean([h['total_penalties'] for h in self.correlation_history]) if self.correlation_history else 0,
                'correlation_trend': 'increasing' if len(self.correlation_history) > 5 and 
                                   self.correlation_history[-1]['max_correlation'] > 
                                   np.mean([h['max_correlation'] for h in self.correlation_history[-5:]]) else 'stable'
            }
        }