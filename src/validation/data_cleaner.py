"""
Data Cleaner for QuantumSentiment Trading Bot

Cleans and preprocesses data based on validation results:
- Remove/fix invalid data points
- Handle missing values
- Outlier treatment
- Data normalization
- Time series gap filling
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import structlog
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from .data_validator import ValidationResult, ValidationSeverity

logger = structlog.get_logger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning"""
    
    # Missing value handling
    handle_missing_values: bool = True
    missing_value_strategy: str = "interpolate"  # interpolate, forward_fill, backward_fill, drop, median
    max_consecutive_missing: int = 5  # Max consecutive missing values to interpolate
    
    # Outlier handling
    handle_outliers: bool = True
    outlier_method: str = "clip"  # clip, remove, cap, winsorize
    outlier_threshold: float = 5.0  # Z-score threshold
    winsorize_limits: Tuple[float, float] = (0.01, 0.01)  # 1% on each side
    
    # Price data specific
    fix_ohlc_relationships: bool = True
    remove_zero_volume: bool = False
    min_price_threshold: float = 0.001
    
    # Feature cleaning
    feature_scaling: bool = False
    scaling_method: str = "robust"  # standard, robust, minmax
    clip_extreme_features: bool = True
    extreme_feature_threshold: float = 1e6
    
    # Time series cleaning
    fill_time_gaps: bool = True
    max_gap_fill_hours: float = 24.0
    interpolation_method: str = "linear"  # linear, spline, polynomial
    
    # General settings
    remove_duplicates: bool = True
    sort_by_timestamp: bool = True
    validate_after_cleaning: bool = True


class DataCleaner:
    """Comprehensive data cleaner for trading data"""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize data cleaner
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Cleaning statistics
        self.stats = {
            'total_cleanings': 0,
            'records_removed': 0,
            'values_imputed': 0,
            'outliers_handled': 0,
            'gaps_filled': 0
        }
        
        logger.info("Data cleaner initialized", 
                   handle_missing=self.config.handle_missing_values,
                   handle_outliers=self.config.handle_outliers)
    
    def clean_market_data(
        self,
        data: pd.DataFrame,
        validation_result: Optional[ValidationResult] = None,
        symbol: str = "unknown"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean market data based on validation results
        
        Args:
            data: Market data DataFrame
            validation_result: Validation results to guide cleaning
            symbol: Asset symbol for logging
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        if data.empty:
            return data, {'message': 'No data to clean'}
        
        try:
            self.stats['total_cleanings'] += 1
            original_length = len(data)
            cleaned_data = data.copy()
            
            cleaning_report = {
                'symbol': symbol,
                'original_records': original_length,
                'operations_performed': [],
                'records_removed': 0,
                'values_modified': 0,
                'gaps_filled': 0
            }
            
            # Sort by timestamp if needed
            if self.config.sort_by_timestamp and hasattr(cleaned_data.index, 'sort_values'):
                try:
                    cleaned_data = cleaned_data.sort_index()
                    cleaning_report['operations_performed'].append('sorted_by_timestamp')
                except:
                    pass
            
            # Remove duplicates
            if self.config.remove_duplicates:
                before_dup = len(cleaned_data)
                cleaned_data = cleaned_data.drop_duplicates()
                duplicates_removed = before_dup - len(cleaned_data)
                
                if duplicates_removed > 0:
                    cleaning_report['records_removed'] += duplicates_removed
                    cleaning_report['operations_performed'].append(f'removed_{duplicates_removed}_duplicates')
            
            # Clean OHLCV data
            cleaned_data, ohlcv_report = self._clean_ohlcv_data(cleaned_data)
            cleaning_report['operations_performed'].extend(ohlcv_report.get('operations', []))
            cleaning_report['values_modified'] += ohlcv_report.get('values_modified', 0)
            
            # Handle missing values
            if self.config.handle_missing_values:
                cleaned_data, missing_report = self._handle_missing_values(cleaned_data, 'market')
                cleaning_report['operations_performed'].extend(missing_report.get('operations', []))
                cleaning_report['values_modified'] += missing_report.get('values_imputed', 0)
            
            # Handle outliers
            if self.config.handle_outliers:
                cleaned_data, outlier_report = self._handle_outliers(cleaned_data, 'market')
                cleaning_report['operations_performed'].extend(outlier_report.get('operations', []))
                cleaning_report['values_modified'] += outlier_report.get('outliers_handled', 0)
            
            # Fill time gaps if needed
            if self.config.fill_time_gaps and hasattr(cleaned_data.index, 'freq'):
                cleaned_data, gap_report = self._fill_time_gaps(cleaned_data)
                cleaning_report['operations_performed'].extend(gap_report.get('operations', []))
                cleaning_report['gaps_filled'] = gap_report.get('gaps_filled', 0)
            
            # Update final statistics
            cleaning_report['final_records'] = len(cleaned_data)
            cleaning_report['records_removed'] = original_length - len(cleaned_data)
            
            # Update global statistics
            self.stats['records_removed'] += cleaning_report['records_removed']
            self.stats['values_imputed'] += cleaning_report['values_modified']
            self.stats['gaps_filled'] += cleaning_report['gaps_filled']
            
            logger.debug("Market data cleaned", 
                        symbol=symbol,
                        original_records=original_length,
                        final_records=len(cleaned_data),
                        operations=len(cleaning_report['operations_performed']))
            
            return cleaned_data, cleaning_report
            
        except Exception as e:
            logger.error("Market data cleaning failed", 
                        symbol=symbol, error=str(e))
            return data, {'error': str(e)}
    
    def clean_sentiment_data(
        self,
        data: pd.DataFrame,
        validation_result: Optional[ValidationResult] = None,
        symbol: str = "unknown"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean sentiment data
        
        Args:
            data: Sentiment data DataFrame
            validation_result: Validation results
            symbol: Asset symbol
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        if data.empty:
            return data, {'message': 'No data to clean'}
        
        try:
            self.stats['total_cleanings'] += 1
            original_length = len(data)
            cleaned_data = data.copy()
            
            cleaning_report = {
                'symbol': symbol,
                'original_records': original_length,
                'operations_performed': [],
                'records_removed': 0,
                'values_modified': 0
            }
            
            # Remove duplicates
            if self.config.remove_duplicates:
                before_dup = len(cleaned_data)
                cleaned_data = cleaned_data.drop_duplicates()
                duplicates_removed = before_dup - len(cleaned_data)
                
                if duplicates_removed > 0:
                    cleaning_report['records_removed'] += duplicates_removed
                    cleaning_report['operations_performed'].append(f'removed_{duplicates_removed}_duplicates')
            
            # Clean sentiment-specific fields
            cleaned_data, sentiment_report = self._clean_sentiment_fields(cleaned_data)
            cleaning_report['operations_performed'].extend(sentiment_report.get('operations', []))
            cleaning_report['values_modified'] += sentiment_report.get('values_modified', 0)
            
            # Handle missing values
            if self.config.handle_missing_values:
                cleaned_data, missing_report = self._handle_missing_values(cleaned_data, 'sentiment')
                cleaning_report['operations_performed'].extend(missing_report.get('operations', []))
                cleaning_report['values_modified'] += missing_report.get('values_imputed', 0)
            
            # Handle outliers
            if self.config.handle_outliers:
                cleaned_data, outlier_report = self._handle_outliers(cleaned_data, 'sentiment')
                cleaning_report['operations_performed'].extend(outlier_report.get('operations', []))
                cleaning_report['values_modified'] += outlier_report.get('outliers_handled', 0)
            
            # Update final statistics
            cleaning_report['final_records'] = len(cleaned_data)
            cleaning_report['records_removed'] = original_length - len(cleaned_data)
            
            logger.debug("Sentiment data cleaned", 
                        symbol=symbol,
                        original_records=original_length,
                        final_records=len(cleaned_data))
            
            return cleaned_data, cleaning_report
            
        except Exception as e:
            logger.error("Sentiment data cleaning failed", 
                        symbol=symbol, error=str(e))
            return data, {'error': str(e)}
    
    def clean_features(
        self,
        features: Dict[str, float],
        validation_result: Optional[ValidationResult] = None,
        symbol: str = "unknown"
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Clean feature data
        
        Args:
            features: Feature dictionary
            validation_result: Validation results
            symbol: Asset symbol
            
        Returns:
            Tuple of (cleaned_features, cleaning_report)
        """
        if not features:
            return features, {'message': 'No features to clean'}
        
        try:
            self.stats['total_cleanings'] += 1
            original_count = len(features)
            cleaned_features = features.copy()
            
            cleaning_report = {
                'symbol': symbol,
                'original_features': original_count,
                'operations_performed': [],
                'features_removed': 0,
                'values_modified': 0
            }
            
            # Remove invalid features
            invalid_features = []
            for name, value in list(cleaned_features.items()):
                # Remove None and NaN values
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    invalid_features.append(name)
                    continue
                
                # Remove infinite values
                if isinstance(value, (int, float)) and np.isinf(value):
                    invalid_features.append(name)
                    continue
                
                # Remove non-numeric values
                if not isinstance(value, (int, float)):
                    invalid_features.append(name)
                    continue
            
            for name in invalid_features:
                del cleaned_features[name]
            
            if invalid_features:
                cleaning_report['features_removed'] += len(invalid_features)
                cleaning_report['operations_performed'].append(f'removed_{len(invalid_features)}_invalid_features')
            
            # Clip extreme values
            if self.config.clip_extreme_features:
                clipped_count = 0
                threshold = self.config.extreme_feature_threshold
                
                for name, value in cleaned_features.items():
                    if abs(value) > threshold:
                        cleaned_features[name] = np.sign(value) * threshold
                        clipped_count += 1
                
                if clipped_count > 0:
                    cleaning_report['values_modified'] += clipped_count
                    cleaning_report['operations_performed'].append(f'clipped_{clipped_count}_extreme_values')
            
            # Handle outliers
            if self.config.handle_outliers and len(cleaned_features) > 10:
                cleaned_features, outlier_report = self._handle_feature_outliers(cleaned_features)
                cleaning_report['operations_performed'].extend(outlier_report.get('operations', []))
                cleaning_report['values_modified'] += outlier_report.get('outliers_handled', 0)
            
            # Feature scaling (if requested)
            if self.config.feature_scaling and len(cleaned_features) > 5:
                cleaned_features, scaling_report = self._scale_features(cleaned_features)
                cleaning_report['operations_performed'].extend(scaling_report.get('operations', []))
            
            # Update final statistics
            cleaning_report['final_features'] = len(cleaned_features)
            
            logger.debug("Features cleaned", 
                        symbol=symbol,
                        original_features=original_count,
                        final_features=len(cleaned_features))
            
            return cleaned_features, cleaning_report
            
        except Exception as e:
            logger.error("Feature cleaning failed", 
                        symbol=symbol, error=str(e))
            return features, {'error': str(e)}
    
    # === HELPER METHODS ===
    
    def _clean_ohlcv_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean OHLCV specific data issues"""
        cleaned_data = data.copy()
        report = {'operations': [], 'values_modified': 0}
        
        try:
            ohlc_columns = ['open', 'high', 'low', 'close']
            available_ohlc = [col for col in ohlc_columns if col in cleaned_data.columns]
            
            if len(available_ohlc) < 4:
                return cleaned_data, report
            
            # Fix OHLC relationships
            if self.config.fix_ohlc_relationships:
                fixes_made = 0
                
                for idx in cleaned_data.index:
                    row = cleaned_data.loc[idx]
                    open_price, high_price, low_price, close_price = [
                        row[col] for col in ohlc_columns
                    ]
                    
                    # Skip if any are null
                    if any(pd.isnull([open_price, high_price, low_price, close_price])):
                        continue
                    
                    # Fix high price (should be maximum)
                    correct_high = max(open_price, high_price, low_price, close_price)
                    if high_price != correct_high:
                        cleaned_data.loc[idx, 'high'] = correct_high
                        fixes_made += 1
                    
                    # Fix low price (should be minimum)
                    correct_low = min(open_price, high_price, low_price, close_price)
                    if low_price != correct_low:
                        cleaned_data.loc[idx, 'low'] = correct_low
                        fixes_made += 1
                
                if fixes_made > 0:
                    report['operations'].append(f'fixed_{fixes_made}_ohlc_relationships')
                    report['values_modified'] += fixes_made
            
            # Remove very low prices (likely errors)
            if self.config.min_price_threshold > 0:
                for col in ohlc_columns:
                    if col in cleaned_data.columns:
                        low_price_mask = cleaned_data[col] < self.config.min_price_threshold
                        if low_price_mask.any():
                            # Replace with NaN to be handled by missing value strategy
                            cleaned_data.loc[low_price_mask, col] = np.nan
                            count = low_price_mask.sum()
                            report['operations'].append(f'removed_{count}_low_{col}_prices')
                            report['values_modified'] += count
            
            # Clean volume data
            if 'volume' in cleaned_data.columns:
                # Remove negative volume
                negative_volume = cleaned_data['volume'] < 0
                if negative_volume.any():
                    cleaned_data.loc[negative_volume, 'volume'] = 0
                    count = negative_volume.sum()
                    report['operations'].append(f'fixed_{count}_negative_volumes')
                    report['values_modified'] += count
                
                # Optionally remove zero volume
                if self.config.remove_zero_volume:
                    zero_volume = cleaned_data['volume'] == 0
                    if zero_volume.any():
                        cleaned_data = cleaned_data[~zero_volume]
                        count = zero_volume.sum()
                        report['operations'].append(f'removed_{count}_zero_volume_records')
            
        except Exception as e:
            logger.warning("OHLCV cleaning error", error=str(e))
        
        return cleaned_data, report
    
    def _clean_sentiment_fields(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean sentiment-specific fields"""
        cleaned_data = data.copy()
        report = {'operations': [], 'values_modified': 0}
        
        try:
            # Clip sentiment scores to valid range
            if 'sentiment_score' in cleaned_data.columns:
                sentiment = cleaned_data['sentiment_score']
                out_of_range = (sentiment < -1) | (sentiment > 1)
                
                if out_of_range.any():
                    cleaned_data.loc[sentiment < -1, 'sentiment_score'] = -1
                    cleaned_data.loc[sentiment > 1, 'sentiment_score'] = 1
                    count = out_of_range.sum()
                    report['operations'].append(f'clipped_{count}_sentiment_scores')
                    report['values_modified'] += count
            
            # Clip confidence scores to valid range
            if 'confidence' in cleaned_data.columns:
                confidence = cleaned_data['confidence']
                out_of_range = (confidence < 0) | (confidence > 1)
                
                if out_of_range.any():
                    cleaned_data.loc[confidence < 0, 'confidence'] = 0
                    cleaned_data.loc[confidence > 1, 'confidence'] = 1
                    count = out_of_range.sum()
                    report['operations'].append(f'clipped_{count}_confidence_scores')
                    report['values_modified'] += count
            
            # Fix negative mention counts
            if 'mention_count' in cleaned_data.columns:
                negative_mentions = cleaned_data['mention_count'] < 0
                if negative_mentions.any():
                    cleaned_data.loc[negative_mentions, 'mention_count'] = 0
                    count = negative_mentions.sum()
                    report['operations'].append(f'fixed_{count}_negative_mention_counts')
                    report['values_modified'] += count
            
            # Clean source names
            if 'source' in cleaned_data.columns:
                # Standardize source names
                source_mapping = {
                    'reddit': 'reddit',
                    'wsb': 'reddit',
                    'wallstreetbets': 'reddit',
                    'twitter': 'twitter',
                    'tweet': 'twitter',
                    'news': 'news',
                    'article': 'news',
                    'unusual_whales': 'unusual_whales',
                    'unusualwhales': 'unusual_whales'
                }
                
                cleaned_sources = cleaned_data['source'].str.lower().map(source_mapping)
                unmapped_count = cleaned_sources.isnull().sum()
                
                if unmapped_count < len(cleaned_data):  # Only update if we mapped most sources
                    cleaned_data['source'] = cleaned_sources.fillna(cleaned_data['source'])
                    if unmapped_count > 0:
                        report['operations'].append(f'standardized_source_names_{unmapped_count}_unmapped')
            
        except Exception as e:
            logger.warning("Sentiment field cleaning error", error=str(e))
        
        return cleaned_data, report
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        data_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values based on strategy"""
        cleaned_data = data.copy()
        report = {'operations': [], 'values_imputed': 0}
        
        try:
            strategy = self.config.missing_value_strategy
            
            for column in cleaned_data.columns:
                if cleaned_data[column].dtype in ['object', 'string']:
                    continue  # Skip text columns
                
                missing_mask = cleaned_data[column].isnull()
                if not missing_mask.any():
                    continue
                
                missing_count = missing_mask.sum()
                
                if strategy == "drop":
                    # Drop rows with missing values in this column
                    cleaned_data = cleaned_data.dropna(subset=[column])
                    report['operations'].append(f'dropped_{missing_count}_rows_missing_{column}')
                
                elif strategy == "forward_fill":
                    cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
                    report['operations'].append(f'forward_filled_{missing_count}_{column}')
                    report['values_imputed'] += missing_count
                
                elif strategy == "backward_fill":
                    cleaned_data[column] = cleaned_data[column].fillna(method='bfill')
                    report['operations'].append(f'backward_filled_{missing_count}_{column}')
                    report['values_imputed'] += missing_count
                
                elif strategy == "interpolate":
                    # Check for consecutive missing values
                    if self._check_consecutive_missing(cleaned_data[column]):
                        try:
                            cleaned_data[column] = cleaned_data[column].interpolate(
                                method='linear', limit=self.config.max_consecutive_missing
                            )
                            actual_filled = missing_count - cleaned_data[column].isnull().sum()
                            report['operations'].append(f'interpolated_{actual_filled}_{column}')
                            report['values_imputed'] += actual_filled
                        except:
                            # Fallback to median
                            median_value = cleaned_data[column].median()
                            cleaned_data[column] = cleaned_data[column].fillna(median_value)
                            report['operations'].append(f'median_filled_{missing_count}_{column}')
                            report['values_imputed'] += missing_count
                
                elif strategy == "median":
                    median_value = cleaned_data[column].median()
                    cleaned_data[column] = cleaned_data[column].fillna(median_value)
                    report['operations'].append(f'median_filled_{missing_count}_{column}')
                    report['values_imputed'] += missing_count
        
        except Exception as e:
            logger.warning("Missing value handling error", error=str(e))
        
        return cleaned_data, report
    
    def _handle_outliers(
        self,
        data: pd.DataFrame,
        data_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers based on method"""
        cleaned_data = data.copy()
        report = {'operations': [], 'outliers_handled': 0}
        
        try:
            method = self.config.outlier_method
            threshold = self.config.outlier_threshold
            
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column in ['timestamp', 'date']:  # Skip time columns
                    continue
                
                values = cleaned_data[column].dropna()
                if len(values) < 10:  # Need sufficient data
                    continue
                
                # Calculate z-scores
                z_scores = np.abs((values - values.mean()) / values.std())
                outlier_mask = z_scores > threshold
                outlier_count = outlier_mask.sum()
                
                if outlier_count == 0:
                    continue
                
                if method == "remove":
                    # Remove outlier rows
                    outlier_indices = values[outlier_mask].index
                    cleaned_data = cleaned_data.drop(outlier_indices)
                    report['operations'].append(f'removed_{outlier_count}_outlier_rows_{column}')
                
                elif method == "clip":
                    # Clip to threshold values
                    mean_val = values.mean()
                    std_val = values.std()
                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val
                    
                    cleaned_data[column] = cleaned_data[column].clip(lower_bound, upper_bound)
                    report['operations'].append(f'clipped_{outlier_count}_outliers_{column}')
                    report['outliers_handled'] += outlier_count
                
                elif method == "cap":
                    # Cap at percentiles
                    lower_cap = values.quantile(0.01)
                    upper_cap = values.quantile(0.99)
                    
                    cleaned_data[column] = cleaned_data[column].clip(lower_cap, upper_cap)
                    report['operations'].append(f'capped_{outlier_count}_outliers_{column}')
                    report['outliers_handled'] += outlier_count
                
                elif method == "winsorize":
                    # Winsorize based on limits
                    from scipy.stats import mstats
                    limits = self.config.winsorize_limits
                    
                    winsorized = mstats.winsorize(values, limits=limits)
                    cleaned_data.loc[values.index, column] = winsorized
                    report['operations'].append(f'winsorized_{outlier_count}_outliers_{column}')
                    report['outliers_handled'] += outlier_count
        
        except Exception as e:
            logger.warning("Outlier handling error", error=str(e))
        
        return cleaned_data, report
    
    def _handle_feature_outliers(
        self,
        features: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Handle outliers in feature dictionary"""
        cleaned_features = features.copy()
        report = {'operations': [], 'outliers_handled': 0}
        
        try:
            values = np.array(list(features.values()))
            
            if len(values) < 10:
                return cleaned_features, report
            
            # Calculate z-scores
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return cleaned_features, report
            
            z_scores = np.abs((values - mean_val) / std_val)
            outlier_mask = z_scores > self.config.outlier_threshold
            
            if not outlier_mask.any():
                return cleaned_features, report
            
            method = self.config.outlier_method
            outlier_count = outlier_mask.sum()
            
            if method == "remove":
                # Remove outlier features
                feature_names = list(features.keys())
                for i, is_outlier in enumerate(outlier_mask):
                    if is_outlier:
                        del cleaned_features[feature_names[i]]
                
                report['operations'].append(f'removed_{outlier_count}_outlier_features')
            
            elif method in ["clip", "cap"]:
                # Clip outlier values
                lower_bound = mean_val - self.config.outlier_threshold * std_val
                upper_bound = mean_val + self.config.outlier_threshold * std_val
                
                feature_names = list(features.keys())
                for i, is_outlier in enumerate(outlier_mask):
                    if is_outlier:
                        feature_name = feature_names[i]
                        value = values[i]
                        
                        if value < lower_bound:
                            cleaned_features[feature_name] = lower_bound
                        elif value > upper_bound:
                            cleaned_features[feature_name] = upper_bound
                
                report['operations'].append(f'clipped_{outlier_count}_outlier_features')
                report['outliers_handled'] += outlier_count
        
        except Exception as e:
            logger.warning("Feature outlier handling error", error=str(e))
        
        return cleaned_features, report
    
    def _scale_features(
        self,
        features: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Scale features using specified method"""
        report = {'operations': []}
        
        try:
            values = np.array(list(features.values())).reshape(-1, 1)
            
            scaler = self.scalers[self.config.scaling_method]
            scaled_values = scaler.fit_transform(values).flatten()
            
            scaled_features = dict(zip(features.keys(), scaled_values))
            report['operations'].append(f'scaled_features_{self.config.scaling_method}')
            
            return scaled_features, report
            
        except Exception as e:
            logger.warning("Feature scaling error", error=str(e))
            return features, report
    
    def _fill_time_gaps(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fill gaps in time series data"""
        report = {'operations': [], 'gaps_filled': 0}
        
        try:
            if not hasattr(data.index, 'freq') or len(data) < 2:
                return data, report
            
            # Detect gaps
            expected_freq = pd.infer_freq(data.index)
            if not expected_freq:
                return data, report
            
            # Create complete date range
            full_range = pd.date_range(
                start=data.index.min(),
                end=data.index.max(),
                freq=expected_freq
            )
            
            # Reindex to fill gaps
            filled_data = data.reindex(full_range)
            gaps_filled = filled_data.isnull().any(axis=1).sum()
            
            if gaps_filled > 0:
                # Interpolate missing values
                method = self.config.interpolation_method
                filled_data = filled_data.interpolate(method=method)
                
                report['operations'].append(f'filled_{gaps_filled}_time_gaps')
                report['gaps_filled'] = gaps_filled
            
            return filled_data, report
            
        except Exception as e:
            logger.warning("Time gap filling error", error=str(e))
            return data, report
    
    def _check_consecutive_missing(self, series: pd.Series) -> bool:
        """Check if consecutive missing values are within limits"""
        if not series.isnull().any():
            return True
        
        # Find consecutive missing value groups
        missing_groups = series.isnull().astype(int).groupby(
            series.notnull().astype(int).cumsum()
        ).sum()
        
        max_consecutive = missing_groups.max()
        return max_consecutive <= self.config.max_consecutive_missing
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return self.stats.copy()
    
    def __str__(self) -> str:
        return f"DataCleaner(cleanings={self.stats['total_cleanings']}, records_removed={self.stats['records_removed']})"
    
    def __repr__(self) -> str:
        return self.__str__()