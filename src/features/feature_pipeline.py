"""
Feature Pipeline for QuantumSentiment Trading Bot

Orchestrates the complete feature engineering process:
- Technical analysis features
- Sentiment analysis features
- Fundamental analysis features
- Market structure features
- Feature validation and cleaning
- Feature versioning and caching
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

from .technical import TechnicalFeatures
from .sentiment import SentimentFeatures
# from .fundamental import FundamentalFeatures  # Will be implemented next
# from .market_structure import MarketStructureFeatures  # Will be implemented next
# from .macro import MacroFeatures  # Will be implemented next

logger = structlog.get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature pipeline"""
    
    # Technical features
    enable_technical: bool = True
    technical_config: Dict[str, Any] = field(default_factory=dict)
    
    # Sentiment features
    enable_sentiment: bool = True
    sentiment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Fundamental features
    enable_fundamental: bool = True
    fundamental_config: Dict[str, Any] = field(default_factory=dict)
    
    # Market structure features
    enable_market_structure: bool = True
    market_structure_config: Dict[str, Any] = field(default_factory=dict)
    
    # Macro features
    enable_macro: bool = True
    macro_config: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline settings
    feature_version: str = "1.0.0"
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Validation settings
    enable_validation: bool = True
    max_missing_ratio: float = 0.3  # Max 30% missing features
    outlier_threshold: float = 5.0  # Z-score threshold for outliers
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    
    # Feature selection
    min_correlation_threshold: float = 0.01  # Remove highly correlated features
    max_features: Optional[int] = None  # Limit total features


class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        database_manager: Optional[Any] = None
    ):
        """
        Initialize feature pipeline
        
        Args:
            config: Feature pipeline configuration
            database_manager: Database manager for caching
        """
        self.config = config or FeatureConfig()
        self.db_manager = database_manager
        
        # Initialize feature generators
        self.feature_generators = {}
        self._init_feature_generators()
        
        # Feature cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Statistics
        self.stats = {
            'total_features_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_failures': 0,
            'processing_times': []
        }
        
        logger.info("Feature pipeline initialized", 
                   generators=len(self.feature_generators),
                   version=self.config.feature_version)
    
    def _init_feature_generators(self):
        """Initialize all feature generators"""
        try:
            if self.config.enable_technical:
                self.feature_generators['technical'] = TechnicalFeatures(
                    self.config.technical_config
                )
                logger.debug("Technical features generator initialized")
            
            if self.config.enable_sentiment:
                self.feature_generators['sentiment'] = SentimentFeatures(
                    self.config.sentiment_config
                )
                logger.debug("Sentiment features generator initialized")
            
            # TODO: Initialize other feature generators when implemented
            # if self.config.enable_fundamental:
            #     self.feature_generators['fundamental'] = FundamentalFeatures(
            #         self.config.fundamental_config
            #     )
            
            # if self.config.enable_market_structure:
            #     self.feature_generators['market_structure'] = MarketStructureFeatures(
            #         self.config.market_structure_config
            #     )
            
            # if self.config.enable_macro:
            #     self.feature_generators['macro'] = MacroFeatures(
            #         self.config.macro_config
            #     )
            
        except Exception as e:
            logger.error("Failed to initialize feature generators", error=str(e))
            raise
    
    def generate_features(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        current_time: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete feature set for a symbol
        
        Args:
            symbol: Asset symbol
            market_data: OHLCV market data
            sentiment_data: Sentiment data DataFrame
            fundamental_data: Fundamental data dictionary
            current_time: Current timestamp
            use_cache: Whether to use cached features
            
        Returns:
            Dictionary containing all features and metadata
        """
        start_time = datetime.now()
        current_time = current_time or start_time
        
        try:
            # Check cache first
            if use_cache and self.config.enable_caching:
                cached_features = self._get_cached_features(
                    symbol, market_data, sentiment_data, current_time
                )
                if cached_features:
                    self.stats['cache_hits'] += 1
                    return cached_features
            
            self.stats['cache_misses'] += 1
            
            # Generate features from all sources
            all_features = {}
            feature_metadata = {
                'symbol': symbol,
                'timestamp': current_time.isoformat(),
                'feature_version': self.config.feature_version,
                'data_quality': {},
                'processing_time': 0,
                'feature_counts': {},
                'validation_results': {}
            }
            
            if self.config.parallel_processing and len(self.feature_generators) > 1:
                # Parallel feature generation
                features_dict = self._generate_features_parallel(
                    symbol, market_data, sentiment_data, fundamental_data, current_time
                )
            else:
                # Sequential feature generation
                features_dict = self._generate_features_sequential(
                    symbol, market_data, sentiment_data, fundamental_data, current_time
                )
            
            # Combine all features
            for category, features in features_dict.items():
                if features:
                    all_features.update(features)
                    feature_metadata['feature_counts'][category] = len(features)
            
            # Validate features
            if self.config.enable_validation:
                validation_results = self._validate_features(all_features)
                feature_metadata['validation_results'] = validation_results
                
                if not validation_results['is_valid']:
                    self.stats['validation_failures'] += 1
                    logger.warning("Feature validation failed", 
                                 symbol=symbol,
                                 issues=validation_results['issues'])
            
            # Clean and transform features
            cleaned_features = self._clean_features(all_features)
            
            # Feature selection (if enabled)
            if self.config.max_features and len(cleaned_features) > self.config.max_features:
                cleaned_features = self._select_features(cleaned_features, market_data)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            feature_metadata['processing_time'] = processing_time
            self.stats['processing_times'].append(processing_time)
            
            # Prepare final result
            result = {
                'features': cleaned_features,
                'metadata': feature_metadata,
                'feature_categories': {
                    category: {k: v for k, v in cleaned_features.items() 
                             if k.startswith(category.replace('_', ''))}
                    for category in self.feature_generators.keys()
                }
            }
            
            # Cache result
            if self.config.enable_caching:
                self._cache_features(symbol, result, current_time)
            
            # Update statistics
            self.stats['total_features_generated'] += len(cleaned_features)
            
            logger.debug("Features generated successfully", 
                        symbol=symbol,
                        total_features=len(cleaned_features),
                        processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Failed to generate features", 
                        symbol=symbol, error=str(e))
            return {
                'features': {},
                'metadata': {
                    'symbol': symbol,
                    'timestamp': current_time.isoformat(),
                    'error': str(e),
                    'feature_version': self.config.feature_version
                },
                'feature_categories': {}
            }
    
    def _generate_features_parallel(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame],
        fundamental_data: Optional[Dict[str, Any]],
        current_time: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Generate features in parallel"""
        features_dict = {}
        
        def generate_category_features(category: str, generator: Any) -> Tuple[str, Dict[str, float]]:
            try:
                if category == 'technical' and not market_data.empty:
                    features = generator.generate_features(market_data)
                elif category == 'sentiment' and sentiment_data is not None:
                    features = generator.generate_features(sentiment_data, symbol, current_time)
                # TODO: Add other feature categories
                else:
                    features = {}
                
                return category, features
                
            except Exception as e:
                logger.error(f"Failed to generate {category} features", 
                           symbol=symbol, error=str(e))
                return category, {}
        
        # Execute feature generation in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_category = {
                executor.submit(generate_category_features, category, generator): category
                for category, generator in self.feature_generators.items()
            }
            
            for future in as_completed(future_to_category):
                category, features = future.result()
                features_dict[category] = features
        
        return features_dict
    
    def _generate_features_sequential(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame],
        fundamental_data: Optional[Dict[str, Any]],
        current_time: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Generate features sequentially"""
        features_dict = {}
        
        for category, generator in self.feature_generators.items():
            try:
                if category == 'technical' and not market_data.empty:
                    features = generator.generate_features(market_data)
                elif category == 'sentiment' and sentiment_data is not None:
                    features = generator.generate_features(sentiment_data, symbol, current_time)
                # TODO: Add other feature categories
                else:
                    features = {}
                
                features_dict[category] = features
                
            except Exception as e:
                logger.error(f"Failed to generate {category} features", 
                           symbol=symbol, error=str(e))
                features_dict[category] = {}
        
        return features_dict
    
    def _validate_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Validate generated features"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'feature_count': len(features),
            'missing_count': 0,
            'outlier_count': 0,
            'infinite_count': 0
        }
        
        try:
            if not features:
                validation_results['is_valid'] = False
                validation_results['issues'].append('No features generated')
                return validation_results
            
            # Count missing values (None, NaN)
            missing_count = sum(1 for v in features.values() 
                              if v is None or (isinstance(v, float) and np.isnan(v)))
            validation_results['missing_count'] = missing_count
            
            # Check missing ratio
            missing_ratio = missing_count / len(features)
            if missing_ratio > self.config.max_missing_ratio:
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f'Too many missing features: {missing_ratio:.2%} > {self.config.max_missing_ratio:.2%}'
                )
            
            # Count infinite values
            infinite_count = sum(1 for v in features.values() 
                               if isinstance(v, (int, float)) and np.isinf(v))
            validation_results['infinite_count'] = infinite_count
            
            if infinite_count > 0:
                validation_results['issues'].append(f'Found {infinite_count} infinite values')
            
            # Detect outliers using z-score
            numeric_values = [v for v in features.values() 
                            if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))]
            
            if len(numeric_values) > 10:  # Need sufficient data for outlier detection
                z_scores = np.abs(stats.zscore(numeric_values))
                outlier_count = np.sum(z_scores > self.config.outlier_threshold)
                validation_results['outlier_count'] = outlier_count
                
                if outlier_count > len(numeric_values) * 0.1:  # More than 10% outliers
                    validation_results['issues'].append(
                        f'High number of outliers: {outlier_count} features'
                    )
            
            # Feature name validation
            invalid_names = [name for name in features.keys() 
                           if not isinstance(name, str) or not name.replace('_', '').replace('.', '').isalnum()]
            
            if invalid_names:
                validation_results['issues'].append(f'Invalid feature names: {invalid_names[:5]}')
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Validation error: {str(e)}')
            logger.error("Feature validation error", error=str(e))
        
        return validation_results
    
    def _clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Clean and normalize features"""
        cleaned_features = {}
        
        try:
            for name, value in features.items():
                # Skip None values
                if value is None:
                    continue
                
                # Handle infinite values
                if isinstance(value, (int, float)) and np.isinf(value):
                    cleaned_features[name] = 0.0  # Replace inf with 0
                    continue
                
                # Handle NaN values
                if isinstance(value, float) and np.isnan(value):
                    cleaned_features[name] = 0.0  # Replace NaN with 0
                    continue
                
                # Ensure numeric types
                if isinstance(value, (int, float)):
                    # Clip extreme values
                    if abs(value) > 1e6:
                        cleaned_features[name] = np.sign(value) * 1e6
                    else:
                        cleaned_features[name] = float(value)
                else:
                    # Convert to float if possible
                    try:
                        cleaned_features[name] = float(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
            
            logger.debug("Features cleaned", 
                        original_count=len(features),
                        cleaned_count=len(cleaned_features))
            
        except Exception as e:
            logger.error("Feature cleaning error", error=str(e))
            return features  # Return original if cleaning fails
        
        return cleaned_features
    
    def _select_features(
        self,
        features: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Select most important features (placeholder implementation)"""
        # This is a simplified feature selection
        # In practice, you'd use more sophisticated methods like:
        # - Mutual information
        # - Feature importance from tree models
        # - Correlation-based selection
        # - PCA/dimensionality reduction
        
        try:
            if len(features) <= self.config.max_features:
                return features
            
            # Sort features by absolute value (simple heuristic)
            sorted_features = sorted(
                features.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )
            
            # Take top N features
            selected_features = dict(sorted_features[:self.config.max_features])
            
            logger.debug("Features selected", 
                        original_count=len(features),
                        selected_count=len(selected_features))
            
            return selected_features
            
        except Exception as e:
            logger.error("Feature selection error", error=str(e))
            return features
    
    def _get_cache_key(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame]
    ) -> str:
        """Generate cache key for feature set"""
        try:
            # Create hash from data characteristics
            key_components = [
                symbol,
                self.config.feature_version,
                str(len(market_data)) if not market_data.empty else "0",
                str(market_data.index[-1]) if not market_data.empty else "none",
                str(len(sentiment_data)) if sentiment_data is not None and not sentiment_data.empty else "0"
            ]
            
            key_string = "|".join(key_components)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception:
            return f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    def _get_cached_features(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame],
        current_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get features from cache if valid"""
        try:
            cache_key = self._get_cache_key(symbol, market_data, sentiment_data)
            
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                cache_time = self.cache_timestamps.get(cache_key)
                
                if cache_time:
                    age_minutes = (current_time - cache_time).total_seconds() / 60
                    
                    if age_minutes <= self.config.cache_ttl_minutes:
                        logger.debug("Cache hit", symbol=symbol, age_minutes=age_minutes)
                        return cached_data
                    else:
                        # Remove expired cache
                        del self.cache[cache_key]
                        del self.cache_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            logger.warning("Cache retrieval error", error=str(e))
            return None
    
    def _cache_features(
        self,
        symbol: str,
        features_result: Dict[str, Any],
        current_time: datetime
    ):
        """Cache generated features"""
        try:
            # Simple cache size management
            if len(self.cache) > 1000:  # Arbitrary limit
                # Remove oldest entries
                oldest_keys = sorted(
                    self.cache_timestamps.keys(),
                    key=lambda k: self.cache_timestamps[k]
                )[:100]
                
                for key in oldest_keys:
                    del self.cache[key]
                    del self.cache_timestamps[key]
            
            # Don't cache the full market_data, just derive a key
            cache_key = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M')}_{self.config.feature_version}"
            
            self.cache[cache_key] = features_result
            self.cache_timestamps[cache_key] = current_time
            
            logger.debug("Features cached", symbol=symbol, cache_key=cache_key)
            
        except Exception as e:
            logger.warning("Feature caching error", error=str(e))
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        stats = self.stats.copy()
        
        if self.stats['processing_times']:
            stats['avg_processing_time'] = np.mean(self.stats['processing_times'])
            stats['max_processing_time'] = np.max(self.stats['processing_times'])
            stats['min_processing_time'] = np.min(self.stats['processing_times'])
        
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        )
        
        stats['cached_features'] = len(self.cache)
        stats['feature_generators'] = list(self.feature_generators.keys())
        
        return stats
    
    def clear_cache(self):
        """Clear feature cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Feature cache cleared")
    
    def get_all_feature_names(self) -> List[str]:
        """Get all possible feature names from all generators"""
        all_names = []
        
        for category, generator in self.feature_generators.items():
            if hasattr(generator, 'get_feature_names'):
                category_names = generator.get_feature_names()
                all_names.extend(category_names)
        
        return all_names
    
    def __str__(self) -> str:
        return f"FeaturePipeline(generators={len(self.feature_generators)}, version={self.config.feature_version})"
    
    def __repr__(self) -> str:
        return self.__str__()