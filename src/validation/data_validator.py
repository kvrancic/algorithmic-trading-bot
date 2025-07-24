"""
Data Validator for QuantumSentiment Trading Bot

Validates all types of data before processing:
- Market data (OHLCV) validation
- Sentiment data validation
- Feature validation
- Data consistency checks
- Time series validation
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import structlog
from enum import Enum

logger = structlog.get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Types of data to validate"""
    MARKET_DATA = "market_data"
    SENTIMENT_DATA = "sentiment_data"
    FEATURE_DATA = "feature_data"
    FUNDAMENTAL_DATA = "fundamental_data"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    data_quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        suggestion: Optional[str] = None
    ):
        """Add a validation issue"""
        issue = ValidationIssue(severity, message, field, value, suggestion)
        self.issues.append(issue)
        
        # Update validity based on severity
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False


@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    
    # Market data validation
    allow_missing_volume: bool = True
    min_price_value: float = 0.001
    max_price_value: float = 1e6
    max_price_change_pct: float = 50.0  # 50% max single-period change
    min_volume_value: float = 0.0
    max_volume_value: float = 1e12
    
    # OHLC relationship validation
    validate_ohlc_relationships: bool = True
    ohlc_tolerance: float = 0.001  # 0.1% tolerance for floating point precision
    
    # Time series validation
    validate_timestamps: bool = True
    allow_gaps: bool = True
    max_gap_hours: float = 24.0
    require_sorted_timestamps: bool = True
    
    # Sentiment data validation
    sentiment_score_range: Tuple[float, float] = (-1.0, 1.0)
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    min_mention_count: int = 0
    max_mention_count: int = 1000000
    
    # Feature validation
    max_feature_value: float = 1e6
    min_feature_value: float = -1e6
    max_missing_ratio: float = 0.3  # 30% max missing features
    detect_outliers: bool = True
    outlier_z_threshold: float = 5.0
    
    # General validation
    max_duplicate_ratio: float = 0.1  # 10% max duplicates
    validate_data_types: bool = True
    strict_mode: bool = False  # Strict mode treats warnings as errors


class DataValidator:
    """Comprehensive data validator for trading data"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize data validator
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        
        # Validation statistics
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'by_data_type': {},
            'common_issues': {}
        }
        
        logger.info("Data validator initialized", strict_mode=self.config.strict_mode)
    
    def validate_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str = "unknown"
    ) -> ValidationResult:
        """
        Validate market data (OHLCV)
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            timeframe: Data timeframe
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        result.metadata = {
            'data_type': DataType.MARKET_DATA.value,
            'symbol': symbol,
            'timeframe': timeframe,
            'record_count': len(data),
            'date_range': None
        }
        
        try:
            self.stats['total_validations'] += 1
            self._update_stats_by_type(DataType.MARKET_DATA)
            
            # Basic structure validation
            self._validate_dataframe_structure(data, result)
            
            if data.empty:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "Market data is empty",
                    suggestion="Ensure data is properly fetched and not filtered out"
                )
                return result
            
            # Required columns validation
            required_columns = ['open', 'high', 'low', 'close']
            optional_columns = ['volume', 'vwap', 'trades_count']
            
            self._validate_required_columns(data, required_columns, result)
            
            # Timestamp validation
            if self.config.validate_timestamps:
                self._validate_timestamps(data, result)
            
            # OHLC data validation
            self._validate_ohlc_data(data, result)
            
            # Volume validation
            if 'volume' in data.columns:
                self._validate_volume_data(data, result)
            elif not self.config.allow_missing_volume:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Volume data is missing",
                    field="volume",
                    suggestion="Volume data improves analysis quality"
                )
            
            # Price change validation
            self._validate_price_changes(data, result)
            
            # Duplicate validation
            self._validate_duplicates(data, result)
            
            # Data quality scoring
            result.data_quality_score = self._calculate_market_data_quality(data, result)
            
            # Update date range metadata
            if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                try:
                    result.metadata['date_range'] = {
                        'start': str(data.index.min()),
                        'end': str(data.index.max())
                    }
                except:
                    pass
            
            # Update statistics
            if result.is_valid:
                self.stats['successful_validations'] += 1
            else:
                self.stats['failed_validations'] += 1
            
            logger.debug("Market data validated", 
                        symbol=symbol,
                        valid=result.is_valid,
                        issues=len(result.issues),
                        quality_score=result.data_quality_score)
            
            return result
            
        except Exception as e:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Validation failed with exception: {str(e)}"
            )
            logger.error("Market data validation error", 
                        symbol=symbol, 
                        error=str(e))
            return result
    
    def validate_sentiment_data(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> ValidationResult:
        """
        Validate sentiment data
        
        Args:
            data: Sentiment data DataFrame
            symbol: Asset symbol
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        result.metadata = {
            'data_type': DataType.SENTIMENT_DATA.value,
            'symbol': symbol,
            'record_count': len(data),
            'sources': []
        }
        
        try:
            self.stats['total_validations'] += 1
            self._update_stats_by_type(DataType.SENTIMENT_DATA)
            
            # Basic structure validation
            self._validate_dataframe_structure(data, result)
            
            if data.empty:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Sentiment data is empty",
                    suggestion="This may be normal for less popular assets"
                )
                return result
            
            # Required columns
            required_columns = ['sentiment_score', 'confidence', 'source']
            self._validate_required_columns(data, required_columns, result)
            
            # Timestamp validation
            if 'timestamp' in data.columns and self.config.validate_timestamps:
                self._validate_timestamps(data, result)
            
            # Sentiment score validation
            if 'sentiment_score' in data.columns:
                self._validate_numeric_range(
                    data['sentiment_score'],
                    'sentiment_score',
                    self.config.sentiment_score_range,
                    result
                )
            
            # Confidence validation
            if 'confidence' in data.columns:
                self._validate_numeric_range(
                    data['confidence'],
                    'confidence',
                    self.config.confidence_range,
                    result
                )
            
            # Mention count validation
            if 'mention_count' in data.columns:
                self._validate_mention_counts(data, result)
            
            # Source validation
            if 'source' in data.columns:
                sources = data['source'].unique().tolist()
                result.metadata['sources'] = sources
                
                # Check for valid sources
                valid_sources = ['reddit', 'twitter', 'news', 'unusual_whales']
                invalid_sources = [s for s in sources if s not in valid_sources]
                
                if invalid_sources:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Unknown sentiment sources: {invalid_sources}",
                        field="source",
                        suggestion="Verify source names are correct"
                    )
            
            # Symbol validation
            if 'symbol' in data.columns:
                data_symbols = data['symbol'].str.upper().unique()
                if symbol.upper() not in data_symbols:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Requested symbol {symbol} not found in data",
                        field="symbol",
                        suggestion="Check symbol filtering logic"
                    )
            
            # Data quality scoring
            result.data_quality_score = self._calculate_sentiment_data_quality(data, result)
            
            logger.debug("Sentiment data validated", 
                        symbol=symbol,
                        valid=result.is_valid,
                        issues=len(result.issues),
                        sources=len(result.metadata['sources']))
            
            return result
            
        except Exception as e:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Sentiment validation failed: {str(e)}"
            )
            logger.error("Sentiment data validation error", 
                        symbol=symbol, 
                        error=str(e))
            return result
    
    def validate_features(
        self,
        features: Dict[str, float],
        symbol: str,
        feature_version: str = "unknown"
    ) -> ValidationResult:
        """
        Validate feature data
        
        Args:
            features: Feature dictionary
            symbol: Asset symbol
            feature_version: Version of features
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        result.metadata = {
            'data_type': DataType.FEATURE_DATA.value,
            'symbol': symbol,
            'feature_version': feature_version,
            'feature_count': len(features),
            'missing_count': 0,
            'outlier_count': 0
        }
        
        try:
            self.stats['total_validations'] += 1
            self._update_stats_by_type(DataType.FEATURE_DATA)
            
            if not features:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "No features provided",
                    suggestion="Check feature generation pipeline"
                )
                return result
            
            # Validate feature names
            self._validate_feature_names(features, result)
            
            # Validate feature values
            numeric_features = {}
            missing_count = 0
            
            for name, value in features.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    missing_count += 1
                    continue
                
                # Type validation
                if not isinstance(value, (int, float)):
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Non-numeric feature: {name}",
                        field=name,
                        value=type(value).__name__,
                        suggestion="Convert to numeric or remove"
                    )
                    continue
                
                # Infinite value check
                if np.isinf(value):
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        f"Infinite value in feature: {name}",
                        field=name,
                        value=value,
                        suggestion="Check calculation logic"
                    )
                    continue
                
                # Range validation
                if value < self.config.min_feature_value or value > self.config.max_feature_value:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Feature value out of expected range: {name}",
                        field=name,
                        value=value,
                        suggestion="Consider feature scaling or clipping"
                    )
                
                numeric_features[name] = value
            
            # Missing ratio validation
            result.metadata['missing_count'] = missing_count
            missing_ratio = missing_count / len(features)
            
            if missing_ratio > self.config.max_missing_ratio:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Too many missing features: {missing_ratio:.2%}",
                    suggestion="Check data availability and feature calculation"
                )
            
            # Outlier detection
            if self.config.detect_outliers and len(numeric_features) > 10:
                outlier_count = self._detect_feature_outliers(numeric_features, result)
                result.metadata['outlier_count'] = outlier_count
            
            # Data quality scoring
            result.data_quality_score = self._calculate_feature_quality(features, result)
            
            logger.debug("Features validated", 
                        symbol=symbol,
                        valid=result.is_valid,
                        feature_count=len(features),
                        missing_ratio=missing_ratio)
            
            return result
            
        except Exception as e:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Feature validation failed: {str(e)}"
            )
            logger.error("Feature validation error", 
                        symbol=symbol, 
                        error=str(e))
            return result
    
    # === HELPER METHODS ===
    
    def _validate_dataframe_structure(self, data: pd.DataFrame, result: ValidationResult):
        """Validate basic DataFrame structure"""
        if not isinstance(data, pd.DataFrame):
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Expected DataFrame, got {type(data).__name__}",
                suggestion="Ensure data is properly formatted as pandas DataFrame"
            )
            return
        
        if data.empty:
            result.add_issue(
                ValidationSeverity.WARNING,
                "DataFrame is empty",
                suggestion="Check data fetching and filtering logic"
            )
    
    def _validate_required_columns(
        self,
        data: pd.DataFrame,
        required_columns: List[str],
        result: ValidationResult
    ):
        """Validate required columns exist"""
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Missing required columns: {missing_columns}",
                suggestion="Check data source and column mapping"
            )
    
    def _validate_timestamps(self, data: pd.DataFrame, result: ValidationResult):
        """Validate timestamp data"""
        if data.index.name in ['timestamp', 'date'] or hasattr(data.index, 'tz'):
            timestamps = data.index
        elif 'timestamp' in data.columns:
            timestamps = data['timestamp']
        else:
            result.add_issue(
                ValidationSeverity.WARNING,
                "No timestamp column or index found",
                field="timestamp",
                suggestion="Timestamps improve data quality and analysis"
            )
            return
        
        # Check for null timestamps
        null_count = pd.isnull(timestamps).sum()
        if null_count > 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Found {null_count} null timestamps",
                field="timestamp",
                suggestion="Remove or fix null timestamp records"
            )
        
        # Check timestamp ordering
        if self.config.require_sorted_timestamps:
            if not timestamps.is_monotonic_increasing:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Timestamps are not sorted",
                    field="timestamp",
                    suggestion="Sort data by timestamp for better analysis"
                )
        
        # Check for large gaps
        if len(timestamps) > 1 and self.config.allow_gaps:
            try:
                time_diffs = timestamps.to_series().diff().dt.total_seconds() / 3600  # Hours
                max_gap = time_diffs.max()
                
                if max_gap > self.config.max_gap_hours:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Large time gap detected: {max_gap:.1f} hours",
                        field="timestamp",
                        suggestion="Check for missing data periods"
                    )
            except:
                pass  # Skip gap analysis if timestamp format is unusual
    
    def _validate_ohlc_data(self, data: pd.DataFrame, result: ValidationResult):
        """Validate OHLC relationships and values"""
        ohlc_columns = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_columns if col in data.columns]
        
        if len(available_ohlc) < 4:
            return  # Skip if not all OHLC columns available
        
        for idx in data.index:
            row = data.loc[idx]
            open_price, high_price, low_price, close_price = [
                row[col] for col in ohlc_columns
            ]
            
            # Check for null values
            if any(pd.isnull([open_price, high_price, low_price, close_price])):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Null OHLC values at {idx}",
                    suggestion="Remove or impute null price data"
                )
                continue
            
            # Check for negative prices
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Non-positive prices at {idx}",
                    suggestion="Check data source for price errors"
                )
                continue
            
            # Check price range validation
            if (high_price < self.config.min_price_value or 
                high_price > self.config.max_price_value):
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Price out of expected range at {idx}: {high_price}",
                    field="high",
                    value=high_price
                )
            
            # Check OHLC relationships
            if self.config.validate_ohlc_relationships:
                tolerance = self.config.ohlc_tolerance
                
                # High should be the highest
                if (high_price < max(open_price, close_price) - tolerance or
                    high_price < low_price - tolerance):
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        f"Invalid high price at {idx}: {high_price}",
                        field="high",
                        suggestion="High should be >= open, close, and low"
                    )
                
                # Low should be the lowest
                if (low_price > min(open_price, close_price) + tolerance or
                    low_price > high_price + tolerance):
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        f"Invalid low price at {idx}: {low_price}",
                        field="low",
                        suggestion="Low should be <= open, close, and high"
                    )
    
    def _validate_volume_data(self, data: pd.DataFrame, result: ValidationResult):
        """Validate volume data"""
        if 'volume' not in data.columns:
            return
        
        volume = data['volume']
        
        # Check for negative volume
        negative_volume = volume < 0
        if negative_volume.any():
            count = negative_volume.sum()
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Found {count} negative volume values",
                field="volume",
                suggestion="Volume should be non-negative"
            )
        
        # Check volume range
        max_volume = volume.max()
        if max_volume > self.config.max_volume_value:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Very high volume detected: {max_volume}",
                field="volume",
                suggestion="Verify volume data accuracy"
            )
        
        # Check for zero volume (suspicious)
        zero_volume_count = (volume == 0).sum()
        if zero_volume_count > len(data) * 0.1:  # More than 10%
            result.add_issue(
                ValidationSeverity.WARNING,
                f"High number of zero volume periods: {zero_volume_count}",
                field="volume",
                suggestion="Check for trading halts or data quality issues"
            )
    
    def _validate_price_changes(self, data: pd.DataFrame, result: ValidationResult):
        """Validate price change magnitudes"""
        if 'close' not in data.columns or len(data) < 2:
            return
        
        close_prices = data['close']
        price_changes = close_prices.pct_change().abs() * 100  # Percentage change
        
        large_changes = price_changes > self.config.max_price_change_pct
        if large_changes.any():
            max_change = price_changes.max()
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Large price change detected: {max_change:.1f}%",
                field="close",
                suggestion="Verify data for potential errors or stock splits"
            )
    
    def _validate_duplicates(self, data: pd.DataFrame, result: ValidationResult):
        """Validate duplicate records"""
        duplicate_count = data.duplicated().sum()
        
        if duplicate_count > 0:
            duplicate_ratio = duplicate_count / len(data)
            
            severity = (ValidationSeverity.ERROR 
                       if duplicate_ratio > self.config.max_duplicate_ratio
                       else ValidationSeverity.WARNING)
            
            result.add_issue(
                severity,
                f"Found {duplicate_count} duplicate records ({duplicate_ratio:.1%})",
                suggestion="Remove duplicate records"
            )
    
    def _validate_numeric_range(
        self,
        series: pd.Series,
        field_name: str,
        valid_range: Tuple[float, float],
        result: ValidationResult
    ):
        """Validate numeric values are within expected range"""
        min_val, max_val = valid_range
        
        out_of_range = (series < min_val) | (series > max_val)
        if out_of_range.any():
            count = out_of_range.sum()
            result.add_issue(
                ValidationSeverity.ERROR,
                f"{count} {field_name} values out of range [{min_val}, {max_val}]",
                field=field_name,
                suggestion=f"Values should be between {min_val} and {max_val}"
            )
    
    def _validate_mention_counts(self, data: pd.DataFrame, result: ValidationResult):
        """Validate mention count data"""
        if 'mention_count' not in data.columns:
            return
        
        mentions = data['mention_count']
        
        # Check for negative mentions
        if (mentions < 0).any():
            result.add_issue(
                ValidationSeverity.ERROR,
                "Negative mention counts found",
                field="mention_count",
                suggestion="Mention counts should be non-negative"
            )
        
        # Check for extremely high mentions (potential bot activity)
        if (mentions > self.config.max_mention_count).any():
            result.add_issue(
                ValidationSeverity.WARNING,
                "Extremely high mention counts detected",
                field="mention_count",
                suggestion="May indicate bot activity or viral event"
            )
    
    def _validate_feature_names(self, features: Dict[str, float], result: ValidationResult):
        """Validate feature names are valid"""
        invalid_names = []
        
        for name in features.keys():
            if not isinstance(name, str):
                invalid_names.append(name)
                continue
            
            # Check for valid characters
            if not name.replace('_', '').replace('.', '').replace('-', '').isalnum():
                invalid_names.append(name)
                continue
            
            # Check length
            if len(name) > 100:
                invalid_names.append(name)
        
        if invalid_names:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Invalid feature names: {invalid_names[:5]}",
                suggestion="Use alphanumeric characters, underscores, dots, and hyphens only"
            )
    
    def _detect_feature_outliers(
        self,
        features: Dict[str, float],
        result: ValidationResult
    ) -> int:
        """Detect outlier features using z-score"""
        values = np.array(list(features.values()))
        
        if len(values) < 10:
            return 0
        
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        outliers = z_scores > self.config.outlier_z_threshold
        outlier_count = np.sum(outliers)
        
        if outlier_count > 0:
            outlier_features = [name for name, is_outlier in 
                              zip(features.keys(), outliers) if is_outlier]
            
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Found {outlier_count} outlier features: {outlier_features[:5]}",
                suggestion="Consider feature scaling or outlier removal"
            )
        
        return outlier_count
    
    def _calculate_market_data_quality(
        self,
        data: pd.DataFrame,
        result: ValidationResult
    ) -> float:
        """Calculate market data quality score (0-1)"""
        try:
            score = 1.0
            
            # Penalize for missing required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_required = sum(1 for col in required_cols if col not in data.columns)
            score -= missing_required * 0.2
            
            # Penalize for validation errors
            error_count = sum(1 for issue in result.issues 
                            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
            score -= error_count * 0.1
            
            # Penalize for warnings
            warning_count = sum(1 for issue in result.issues 
                              if issue.severity == ValidationSeverity.WARNING)
            score -= warning_count * 0.05
            
            # Bonus for volume data
            if 'volume' in data.columns:
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _calculate_sentiment_data_quality(
        self,
        data: pd.DataFrame,
        result: ValidationResult
    ) -> float:
        """Calculate sentiment data quality score (0-1)"""
        try:
            score = 1.0
            
            # Penalize for validation errors
            error_count = sum(1 for issue in result.issues 
                            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
            score -= error_count * 0.15
            
            # Penalize for warnings
            warning_count = sum(1 for issue in result.issues 
                              if issue.severity == ValidationSeverity.WARNING)
            score -= warning_count * 0.05
            
            # Bonus for multiple sources
            source_count = len(result.metadata.get('sources', []))
            score += min(0.2, source_count * 0.05)
            
            # Bonus for confidence data
            if 'confidence' in data.columns:
                avg_confidence = data['confidence'].mean()
                score += avg_confidence * 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_feature_quality(
        self,
        features: Dict[str, float],
        result: ValidationResult
    ) -> float:
        """Calculate feature quality score (0-1)"""
        try:
            score = 1.0
            
            # Penalize for validation errors
            error_count = sum(1 for issue in result.issues 
                            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
            score -= error_count * 0.2
            
            # Penalize for missing features
            missing_ratio = result.metadata.get('missing_count', 0) / max(len(features), 1)
            score -= missing_ratio * 0.3
            
            # Penalize for outliers
            outlier_ratio = result.metadata.get('outlier_count', 0) / max(len(features), 1)
            score -= outlier_ratio * 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _update_stats_by_type(self, data_type: DataType):
        """Update validation statistics by data type"""
        type_name = data_type.value
        if type_name not in self.stats['by_data_type']:
            self.stats['by_data_type'][type_name] = 0
        self.stats['by_data_type'][type_name] += 1
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.stats.copy()
        
        if self.stats['total_validations'] > 0:
            stats['success_rate'] = (
                self.stats['successful_validations'] / self.stats['total_validations']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def __str__(self) -> str:
        return f"DataValidator(validations={self.stats['total_validations']}, success_rate={self.stats.get('success_rate', 0):.2%})"
    
    def __repr__(self) -> str:
        return self.__str__()