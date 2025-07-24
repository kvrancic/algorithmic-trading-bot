"""
Sentiment Fusion Algorithm for QuantumSentiment Trading Bot

Advanced multi-source sentiment fusion:
- Weighted sentiment aggregation
- Source reliability scoring
- Confidence-based weighting
- Temporal decay modeling
- Anomaly detection and filtering
- Signal strength calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FusionConfig:
    """Configuration for sentiment fusion"""
    
    # Source weights (should sum to 1.0)
    source_weights: Dict[str, float] = None
    
    # Fusion parameters
    confidence_threshold: float = 0.1
    recency_decay_hours: float = 12.0
    anomaly_threshold: float = 3.0  # Z-score threshold
    min_sources_required: int = 1  # Allow single source with reduced confidence
    
    # Calibration parameters
    sentiment_scale_factor: float = 1.0
    volatility_adjustment: bool = True
    market_hours_boost: float = 1.2
    
    def __post_init__(self):
        if self.source_weights is None:
            self.source_weights = {
                'reddit': 0.35,
                'news': 0.30,
                'unusual_whales': 0.20,
                'twitter': 0.15
            }
        
        # Normalize weights
        total_weight = sum(self.source_weights.values())
        if total_weight != 1.0:
            self.source_weights = {k: v/total_weight for k, v in self.source_weights.items()}


class SentimentFusion:
    """Advanced multi-source sentiment fusion algorithm"""
    
    def __init__(self, config: FusionConfig):
        """
        Initialize sentiment fusion
        
        Args:
            config: Fusion configuration
        """
        self.config = config
        self.historical_data = {}  # Cache for historical sentiment data
        
        logger.info("Sentiment fusion initialized", 
                   sources=list(config.source_weights.keys()),
                   weights=config.source_weights)
    
    def fuse_sentiment(
        self, 
        sentiment_data: Dict[str, Dict[str, Any]], 
        symbol: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Fuse sentiment from multiple sources
        
        Args:
            sentiment_data: Dict of {source: sentiment_analysis_results}
            symbol: Stock symbol
            current_time: Current timestamp
            
        Returns:
            Fused sentiment analysis with confidence metrics
        """
        current_time = current_time or datetime.utcnow()
        
        logger.debug("Starting sentiment fusion", 
                    symbol=symbol, 
                    sources=list(sentiment_data.keys()))
        
        try:
            # Validate and preprocess data
            valid_sources = self._validate_sources(sentiment_data)
            
            if len(valid_sources) < self.config.min_sources_required:
                logger.warning("Insufficient valid sources for fusion",
                             symbol=symbol,
                             valid_sources=len(valid_sources),
                             required=self.config.min_sources_required)
                return self._get_default_result(symbol, current_time)
            
            # Calculate weighted sentiment scores
            weighted_sentiment = self._calculate_weighted_sentiment(valid_sources, current_time)
            
            # Calculate fusion confidence
            fusion_confidence = self._calculate_fusion_confidence(valid_sources)
            
            # Detect anomalies
            anomaly_analysis = self._detect_anomalies(valid_sources)
            
            # Calculate source agreement
            source_agreement = self._calculate_source_agreement(valid_sources)
            
            # Apply calibration adjustments
            calibrated_sentiment = self._apply_calibration(
                weighted_sentiment, 
                fusion_confidence,
                current_time
            )
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                calibrated_sentiment,
                fusion_confidence,
                source_agreement,
                anomaly_analysis
            )
            
            # Generate trading signal
            trading_signal = self._generate_trading_signal(
                calibrated_sentiment,
                signal_strength,
                fusion_confidence
            )
            
            # Compile comprehensive result
            result = {
                'symbol': symbol.upper(),
                'timestamp': current_time,
                'fusion_method': 'weighted_confidence_fusion',
                
                # Core metrics
                'fused_sentiment': calibrated_sentiment,
                'fusion_confidence': fusion_confidence,
                'signal_strength': signal_strength,
                
                # Trading signal
                'trading_signal': trading_signal,
                'signal_direction': self._get_signal_direction(calibrated_sentiment),
                'signal_magnitude': abs(calibrated_sentiment),
                
                # Source analysis
                'sources_used': list(valid_sources.keys()),
                'source_count': len(valid_sources),
                'source_agreement': source_agreement,
                'dominant_source': self._get_dominant_source(valid_sources),
                
                # Quality indicators
                'anomaly_detected': anomaly_analysis['has_anomaly'],
                'anomaly_score': anomaly_analysis['anomaly_score'],
                'consistency_score': 1.0 - anomaly_analysis['inconsistency'],
                
                # Detailed breakdowns
                'source_contributions': self._calculate_source_contributions(valid_sources),
                'confidence_factors': self._analyze_confidence_factors(valid_sources, fusion_confidence),
                'temporal_factors': self._analyze_temporal_factors(valid_sources, current_time),
                
                # Raw source data (for debugging)
                'raw_source_sentiments': {
                    source: data.get('sentiment_score', 0) 
                    for source, data in valid_sources.items()
                },
                'raw_source_confidences': {
                    source: data.get('confidence', 0) 
                    for source, data in valid_sources.items()
                }
            }
            
            # Store for historical analysis
            self._update_historical_data(symbol, result)
            
            logger.info("Sentiment fusion completed",
                       symbol=symbol,
                       fused_sentiment=calibrated_sentiment,
                       confidence=fusion_confidence,
                       signal=trading_signal,
                       sources_used=len(valid_sources))
            
            return result
            
        except Exception as e:
            logger.error("Sentiment fusion failed", symbol=symbol, error=str(e))
            return self._get_default_result(symbol, current_time, error=str(e))
    
    def _validate_sources(self, sentiment_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate and filter source data"""
        valid_sources = {}
        
        for source, data in sentiment_data.items():
            if not isinstance(data, dict):
                logger.warning(f"Invalid data format for source {source}")
                continue
            
            # Check required fields
            if 'sentiment_score' not in data or 'confidence' not in data:
                logger.warning(f"Missing required fields for source {source}")
                continue
            
            # Check data quality
            sentiment_score = data.get('sentiment_score', 0)
            confidence = data.get('confidence', 0)
            
            if not isinstance(sentiment_score, (int, float)) or not isinstance(confidence, (int, float)):
                logger.warning(f"Invalid data types for source {source}")
                continue
            
            # Check value ranges
            if abs(sentiment_score) > 1.0 or confidence < 0 or confidence > 1.0:
                logger.warning(f"Values out of range for source {source}",
                             sentiment=sentiment_score, confidence=confidence)
                continue
            
            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                logger.debug(f"Source {source} below confidence threshold",
                           confidence=confidence, threshold=self.config.confidence_threshold)
                continue
            
            valid_sources[source] = data
        
        return valid_sources
    
    def _calculate_weighted_sentiment(
        self, 
        valid_sources: Dict[str, Dict[str, Any]], 
        current_time: datetime
    ) -> float:
        """Calculate weighted average sentiment"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source, data in valid_sources.items():
            # Base weight from configuration
            base_weight = self.config.source_weights.get(source, 0.25)
            
            # Confidence weighting
            confidence = data.get('confidence', 0.5)
            confidence_weight = confidence ** 0.5  # Square root for diminishing returns
            
            # Recency weighting
            timestamp = data.get('timestamp', current_time)
            recency_weight = self._calculate_recency_weight(timestamp, current_time)
            
            # Data quality weighting
            quality_weight = self._calculate_quality_weight(data)
            
            # Combined weight
            combined_weight = base_weight * confidence_weight * recency_weight * quality_weight
            
            sentiment_score = data.get('sentiment_score', 0)
            weighted_sum += sentiment_score * combined_weight
            total_weight += combined_weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_recency_weight(self, timestamp: datetime, current_time: datetime) -> float:
        """Calculate time-based decay weight"""
        if timestamp is None:
            return 0.5
        
        hours_ago = (current_time - timestamp).total_seconds() / 3600
        
        # Exponential decay
        decay_factor = np.exp(-hours_ago / self.config.recency_decay_hours)
        
        return max(decay_factor, 0.1)  # Minimum weight of 10%
    
    def _calculate_quality_weight(self, data: Dict[str, Any]) -> float:
        """Calculate data quality weight"""
        quality_weight = 1.0
        
        # Adjust based on data completeness
        total_mentions = data.get('total_mentions', 0)
        if total_mentions == 0:
            quality_weight *= 0.5
        elif total_mentions < 5:
            quality_weight *= 0.8
        
        # Adjust based on source diversity (for aggregated sources)
        source_diversity = data.get('source_diversity', 1.0)
        quality_weight *= (0.5 + 0.5 * source_diversity)
        
        # Adjust based on engagement
        avg_engagement = data.get('avg_engagement', 0)
        if avg_engagement > 0:
            engagement_factor = min(np.log1p(avg_engagement) / 10, 0.5)
            quality_weight *= (1.0 + engagement_factor)
        
        return quality_weight
    
    def _calculate_fusion_confidence(self, valid_sources: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall confidence in the fused sentiment"""
        
        if not valid_sources:
            return 0.0
        
        # Base confidence from individual sources
        individual_confidences = [data.get('confidence', 0) for data in valid_sources.values()]
        avg_confidence = np.mean(individual_confidences)
        
        # Source count bonus (more sources = higher confidence, but don't penalize single source too much)
        source_count_factor = min(0.7 + (len(valid_sources) - 1) * 0.1, 1.0)  # Start at 0.7 for single source
        
        # Agreement factor (similar sentiments = higher confidence)
        sentiments = [data.get('sentiment_score', 0) for data in valid_sources.values()]
        agreement_factor = 1.0 - (np.std(sentiments) / 2.0)  # Normalize by max possible std
        agreement_factor = max(agreement_factor, 0.0)
        
        # Data volume factor
        total_mentions = sum([data.get('total_mentions', 0) for data in valid_sources.values()])
        volume_factor = min(np.log1p(total_mentions) / 10, 1.0)
        
        # Combined confidence
        fusion_confidence = avg_confidence * source_count_factor * agreement_factor * (0.5 + 0.5 * volume_factor)
        
        return min(fusion_confidence, 1.0)
    
    def _detect_anomalies(self, valid_sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalous sentiment values"""
        
        if len(valid_sources) < 3:
            return {'has_anomaly': False, 'anomaly_score': 0.0, 'inconsistency': 0.0}
        
        sentiments = [data.get('sentiment_score', 0) for data in valid_sources.values()]
        
        # Calculate z-scores
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)
        
        if std_sentiment == 0:
            return {'has_anomaly': False, 'anomaly_score': 0.0, 'inconsistency': 0.0}
        
        z_scores = [(s - mean_sentiment) / std_sentiment for s in sentiments]
        max_z_score = max([abs(z) for z in z_scores])
        
        has_anomaly = max_z_score > self.config.anomaly_threshold
        inconsistency = std_sentiment  # Higher std = more inconsistency
        
        return {
            'has_anomaly': has_anomaly,
            'anomaly_score': max_z_score,
            'inconsistency': inconsistency,
            'sentiment_std': std_sentiment
        }
    
    def _calculate_source_agreement(self, valid_sources: Dict[str, Dict[str, Any]]) -> float:
        """Calculate agreement level between sources"""
        
        if len(valid_sources) < 2:
            return 1.0
        
        sentiments = [data.get('sentiment_score', 0) for data in valid_sources.values()]
        
        # Calculate pairwise agreements
        agreements = []
        for i in range(len(sentiments)):
            for j in range(i + 1, len(sentiments)):
                # Agreement based on sign and magnitude similarity
                sign_agreement = 1 if sentiments[i] * sentiments[j] >= 0 else 0
                magnitude_diff = abs(sentiments[i] - sentiments[j])
                magnitude_agreement = 1 - magnitude_diff / 2  # Normalize by max possible diff
                
                # Combined agreement (weighted toward sign agreement)
                agreement = 0.7 * sign_agreement + 0.3 * max(magnitude_agreement, 0)
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _apply_calibration(
        self, 
        raw_sentiment: float, 
        confidence: float,
        current_time: datetime
    ) -> float:
        """Apply calibration adjustments to sentiment"""
        
        calibrated = raw_sentiment * self.config.sentiment_scale_factor
        
        # Market hours adjustment
        if self._is_market_hours(current_time):
            calibrated *= self.config.market_hours_boost
        
        # Confidence-based scaling
        calibrated *= (0.5 + 0.5 * confidence)  # Scale by confidence
        
        # Volatility adjustment (placeholder - would use actual market volatility)
        if self.config.volatility_adjustment:
            # In practice, you'd get current VIX or calculate rolling volatility
            volatility_factor = 1.0  # Placeholder
            calibrated *= volatility_factor
        
        # Ensure bounded output
        return np.clip(calibrated, -1.0, 1.0)
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during market hours (simplified)"""
        # US market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        # This is a simplified version - production would handle holidays, etc.
        
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        hour = timestamp.hour
        
        # Weekend
        if weekday >= 5:
            return False
        
        # Market hours (assuming UTC, adjust for actual timezone)
        if 14 <= hour < 21:  # Roughly 9:30 AM - 4:00 PM ET in UTC
            return True
        
        return False
    
    def _calculate_signal_strength(
        self,
        sentiment: float,
        confidence: float,
        agreement: float,
        anomaly_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall signal strength"""
        
        # Base strength from sentiment magnitude
        base_strength = abs(sentiment)
        
        # Confidence factor
        confidence_factor = confidence
        
        # Agreement factor
        agreement_factor = agreement
        
        # Anomaly penalty
        anomaly_penalty = 1.0 - (anomaly_analysis['anomaly_score'] / 10.0)
        anomaly_penalty = max(anomaly_penalty, 0.1)
        
        # Combined signal strength
        signal_strength = base_strength * confidence_factor * agreement_factor * anomaly_penalty
        
        return min(signal_strength, 1.0)
    
    def _generate_trading_signal(
        self,
        sentiment: float,
        signal_strength: float,
        confidence: float
    ) -> str:
        """Generate trading signal based on fused sentiment"""
        
        # Thresholds for trading signals
        strong_threshold = 0.6
        moderate_threshold = 0.3
        weak_threshold = 0.1
        
        # Require minimum confidence and signal strength
        min_confidence = 0.3
        min_signal_strength = 0.2
        
        if confidence < min_confidence or signal_strength < min_signal_strength:
            return 'HOLD'
        
        abs_sentiment = abs(sentiment)
        
        if abs_sentiment >= strong_threshold:
            signal_type = 'STRONG'
        elif abs_sentiment >= moderate_threshold:
            signal_type = 'MODERATE'
        elif abs_sentiment >= weak_threshold:
            signal_type = 'WEAK'
        else:
            return 'HOLD'
        
        direction = 'BUY' if sentiment > 0 else 'SELL'
        
        return f'{signal_type}_{direction}'
    
    def _get_signal_direction(self, sentiment: float) -> str:
        """Get signal direction"""
        if sentiment > 0.05:
            return 'BULLISH'
        elif sentiment < -0.05:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_dominant_source(self, valid_sources: Dict[str, Dict[str, Any]]) -> str:
        """Identify the dominant source in the fusion"""
        
        if not valid_sources:
            return 'none'
        
        # Weight by contribution to final result
        contributions = {}
        
        for source, data in valid_sources.items():
            base_weight = self.config.source_weights.get(source, 0.25)
            confidence = data.get('confidence', 0.5)
            total_mentions = data.get('total_mentions', 1)
            
            contribution = base_weight * confidence * np.log1p(total_mentions)
            contributions[source] = contribution
        
        return max(contributions.items(), key=lambda x: x[1])[0]
    
    def _calculate_source_contributions(self, valid_sources: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate each source's contribution to the final result"""
        
        contributions = {}
        total_contribution = 0
        
        for source, data in valid_sources.items():
            base_weight = self.config.source_weights.get(source, 0.25)
            confidence = data.get('confidence', 0.5)
            
            contribution = base_weight * confidence
            contributions[source] = contribution
            total_contribution += contribution
        
        # Normalize to percentages
        if total_contribution > 0:
            contributions = {k: v/total_contribution for k, v in contributions.items()}
        
        return contributions
    
    def _analyze_confidence_factors(
        self, 
        valid_sources: Dict[str, Dict[str, Any]], 
        fusion_confidence: float
    ) -> Dict[str, Any]:
        """Analyze factors contributing to confidence"""
        
        factors = {
            'source_count': len(valid_sources) / 4,  # Normalize by ideal count
            'avg_source_confidence': np.mean([data.get('confidence', 0) for data in valid_sources.values()]),
            'data_volume': min(sum([data.get('total_mentions', 0) for data in valid_sources.values()]) / 100, 1.0),
            'source_diversity': min(len(valid_sources) / len(self.config.source_weights), 1.0)
        }
        
        # Calculate factor contributions
        factors['primary_factor'] = max(factors.items(), key=lambda x: x[1])[0]
        
        return factors
    
    def _analyze_temporal_factors(
        self, 
        valid_sources: Dict[str, Dict[str, Any]], 
        current_time: datetime
    ) -> Dict[str, Any]:
        """Analyze temporal factors affecting the fusion"""
        
        timestamps = []
        for data in valid_sources.values():
            if data.get('timestamp'):
                timestamps.append(data['timestamp'])
        
        if not timestamps:
            return {'data_freshness': 0.5, 'temporal_spread': 0.0}
        
        # Data freshness (average age)
        avg_age_hours = np.mean([(current_time - ts).total_seconds() / 3600 for ts in timestamps])
        data_freshness = max(0, 1 - avg_age_hours / 24)  # Normalize by 24 hours
        
        # Temporal spread
        if len(timestamps) > 1:
            oldest = min(timestamps)
            newest = max(timestamps)
            spread_hours = (newest - oldest).total_seconds() / 3600
            temporal_spread = min(spread_hours / 24, 1.0)  # Normalize by 24 hours
        else:
            temporal_spread = 0.0
        
        return {
            'data_freshness': data_freshness,
            'temporal_spread': temporal_spread,
            'avg_age_hours': avg_age_hours,
            'is_market_hours': self._is_market_hours(current_time)
        }
    
    def _update_historical_data(self, symbol: str, result: Dict[str, Any]):
        """Update historical sentiment data for trend analysis"""
        
        if symbol not in self.historical_data:
            self.historical_data[symbol] = []
        
        # Keep last 100 data points per symbol
        self.historical_data[symbol].append({
            'timestamp': result['timestamp'],
            'fused_sentiment': result['fused_sentiment'],
            'fusion_confidence': result['fusion_confidence'],
            'signal_strength': result['signal_strength'],
            'sources_used': result['sources_used']
        })
        
        # Trim to keep memory usage reasonable
        if len(self.historical_data[symbol]) > 100:
            self.historical_data[symbol] = self.historical_data[symbol][-100:]
    
    def get_sentiment_trend(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get sentiment trend for a symbol"""
        
        if symbol not in self.historical_data:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'volatility': 0.0}
        
        data = self.historical_data[symbol]
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Filter recent data
        recent_data = [d for d in data if d['timestamp'] >= cutoff_time]
        
        if len(recent_data) < 3:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'volatility': 0.0}
        
        # Calculate trend
        sentiments = [d['fused_sentiment'] for d in recent_data]
        time_points = [(d['timestamp'] - recent_data[0]['timestamp']).total_seconds() / 3600 
                      for d in recent_data]
        
        # Linear regression for trend
        if len(sentiments) >= 2:
            slope = np.polyfit(time_points, sentiments, 1)[0]
        else:
            slope = 0.0
        
        # Volatility
        volatility = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Trend classification
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0.01:
            trend = 'improving'
        else:
            trend = 'deteriorating'
        
        return {
            'trend': trend,
            'slope': slope,
            'volatility': volatility,
            'data_points': len(recent_data),
            'current_sentiment': sentiments[-1] if sentiments else 0.0
        }
    
    def _get_default_result(
        self, 
        symbol: str, 
        current_time: datetime, 
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get default result when fusion fails"""
        
        return {
            'symbol': symbol.upper(),
            'timestamp': current_time,
            'fusion_method': 'default',
            
            # Core metrics (neutral/low confidence)
            'fused_sentiment': 0.0,
            'fusion_confidence': 0.0,
            'signal_strength': 0.0,
            
            # Trading signal
            'trading_signal': 'HOLD',
            'signal_direction': 'NEUTRAL',
            'signal_magnitude': 0.0,
            
            # Source analysis
            'sources_used': [],
            'source_count': 0,
            'source_agreement': 0.0,
            'dominant_source': 'none',
            
            # Quality indicators
            'anomaly_detected': False,
            'anomaly_score': 0.0,
            'consistency_score': 0.0,
            
            # Error info
            'error': error,
            'status': 'failed' if error else 'insufficient_data'
        }
    
    def __str__(self) -> str:
        return f"SentimentFusion(sources={len(self.config.source_weights)})"
    
    def __repr__(self) -> str:
        return self.__str__()