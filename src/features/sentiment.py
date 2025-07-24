"""
Sentiment Features for QuantumSentiment Trading Bot

Processes and transforms raw sentiment data into ML-ready features:
- Reddit sentiment aggregation
- Twitter sentiment analysis
- News sentiment processing
- UnusualWhales political intelligence
- Multi-source sentiment fusion
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class SentimentFeatures:
    """Sentiment feature engineering for trading signals"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment features generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default parameters
        self.default_params = {
            'sentiment_windows': [1, 4, 12, 24, 48],  # Hours
            'decay_factor': 0.95,  # Exponential decay for older sentiment
            'confidence_threshold': 0.1,  # Minimum confidence to include
            'volume_weight': True,  # Weight by mention volume
            'source_weights': {
                'reddit': 0.4,
                'twitter': 0.2,
                'news': 0.25,
                'unusual_whales': 0.15
            },
            'emotion_categories': [
                'bullish', 'bearish', 'neutral', 'fear', 'greed', 
                'excitement', 'uncertainty', 'confidence'
            ]
        }
        
        # Merge with provided config
        self.params = {**self.default_params, **self.config}
        
        logger.info("Sentiment features initialized")
    
    def generate_features(
        self,
        sentiment_data: pd.DataFrame,
        symbol: str,
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Generate sentiment features from raw sentiment data
        
        Args:
            sentiment_data: DataFrame with sentiment data
            symbol: Asset symbol
            current_time: Current timestamp for feature generation
            
        Returns:
            Dictionary of sentiment features
        """
        if sentiment_data.empty:
            logger.warning("No sentiment data available", symbol=symbol)
            return self._get_empty_features()
        
        try:
            features = {}
            
            # Filter data for this symbol and recent time period
            symbol_data = sentiment_data[
                sentiment_data.get('symbol', pd.Series()).str.upper() == symbol.upper()
            ].copy()
            
            if symbol_data.empty:
                logger.debug("No sentiment data for symbol", symbol=symbol)
                return self._get_empty_features()
            
            # Sort by timestamp
            if 'timestamp' in symbol_data.columns:
                symbol_data = symbol_data.sort_values('timestamp')
            
            # Basic sentiment aggregation features
            features.update(self._basic_sentiment_features(symbol_data, current_time))
            
            # Time-windowed sentiment features
            features.update(self._windowed_sentiment_features(symbol_data, current_time))
            
            # Source-specific features
            features.update(self._source_specific_features(symbol_data, current_time))
            
            # Sentiment momentum features
            features.update(self._sentiment_momentum_features(symbol_data, current_time))
            
            # Emotion and engagement features
            features.update(self._emotion_engagement_features(symbol_data, current_time))
            
            # Anomaly and spike detection features
            features.update(self._anomaly_detection_features(symbol_data, current_time))
            
            # Cross-source consensus features
            features.update(self._consensus_features(symbol_data, current_time))
            
            # Political intelligence features (UnusualWhales)
            features.update(self._political_intelligence_features(symbol_data, current_time))
            
            logger.debug("Sentiment features generated", 
                        symbol=symbol,
                        feature_count=len(features),
                        data_points=len(symbol_data))
            
            return features
            
        except Exception as e:
            logger.error("Failed to generate sentiment features", 
                        symbol=symbol, error=str(e))
            return self._get_empty_features()
    
    def _basic_sentiment_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Basic sentiment aggregation features"""
        features = {}
        
        try:
            if data.empty:
                return features
            
            # Recent 24 hours data
            cutoff_24h = current_time - timedelta(hours=24)
            recent_data = data[data['timestamp'] >= cutoff_24h] if 'timestamp' in data.columns else data
            
            if recent_data.empty:
                return features
            
            # Overall sentiment metrics
            sentiment_scores = recent_data['sentiment_score'].values
            confidences = recent_data['confidence'].values
            
            # Weighted average sentiment (by confidence)
            if len(sentiment_scores) > 0:
                weighted_sentiment = np.average(sentiment_scores, weights=confidences)
                features['sentiment_avg_24h'] = weighted_sentiment
                features['sentiment_raw_avg_24h'] = np.mean(sentiment_scores)
                features['sentiment_std_24h'] = np.std(sentiment_scores)
                features['sentiment_median_24h'] = np.median(sentiment_scores)
            
            # Confidence metrics
            if len(confidences) > 0:
                features['confidence_avg_24h'] = np.mean(confidences)
                features['confidence_min_24h'] = np.min(confidences)
                features['confidence_max_24h'] = np.max(confidences)
            
            # Mention volume
            if 'mention_count' in recent_data.columns:
                mentions = recent_data['mention_count'].values
                features['mention_volume_24h'] = np.sum(mentions)
                features['mention_avg_24h'] = np.mean(mentions)
                features['mention_max_24h'] = np.max(mentions)
            
            # Sentiment distribution
            bullish_count = len(recent_data[recent_data['sentiment_score'] > 0.1])
            bearish_count = len(recent_data[recent_data['sentiment_score'] < -0.1])
            neutral_count = len(recent_data) - bullish_count - bearish_count
            
            total_count = len(recent_data)
            if total_count > 0:
                features['bullish_ratio_24h'] = bullish_count / total_count
                features['bearish_ratio_24h'] = bearish_count / total_count
                features['neutral_ratio_24h'] = neutral_count / total_count
            
            # Signal strength indicators
            if 'bullish_signals' in recent_data.columns and 'bearish_signals' in recent_data.columns:
                bullish_signals = recent_data['bullish_signals'].sum()
                bearish_signals = recent_data['bearish_signals'].sum()
                
                features['bullish_signals_24h'] = bullish_signals
                features['bearish_signals_24h'] = bearish_signals
                
                total_signals = bullish_signals + bearish_signals
                if total_signals > 0:
                    features['signal_ratio_24h'] = (bullish_signals - bearish_signals) / total_signals
            
        except Exception as e:
            logger.warning("Error in basic sentiment features", error=str(e))
        
        return features
    
    def _windowed_sentiment_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Time-windowed sentiment features"""
        features = {}
        
        try:
            if 'timestamp' not in data.columns:
                return features
            
            for window_hours in self.params['sentiment_windows']:
                cutoff_time = current_time - timedelta(hours=window_hours)
                window_data = data[data['timestamp'] >= cutoff_time]
                
                if window_data.empty:
                    features[f'sentiment_avg_{window_hours}h'] = 0.0
                    features[f'mention_count_{window_hours}h'] = 0
                    continue
                
                # Apply time decay
                time_weights = self._calculate_time_weights(window_data['timestamp'], current_time)
                sentiment_scores = window_data['sentiment_score'].values
                confidences = window_data['confidence'].values
                
                # Combined weights (time decay * confidence)
                combined_weights = time_weights * confidences
                
                if np.sum(combined_weights) > 0:
                    weighted_sentiment = np.average(sentiment_scores, weights=combined_weights)
                    features[f'sentiment_avg_{window_hours}h'] = weighted_sentiment
                else:
                    features[f'sentiment_avg_{window_hours}h'] = np.mean(sentiment_scores)
                
                # Mention count for this window
                if 'mention_count' in window_data.columns:
                    features[f'mention_count_{window_hours}h'] = window_data['mention_count'].sum()
                else:
                    features[f'mention_count_{window_hours}h'] = len(window_data)
                
                # Sentiment volatility in window
                features[f'sentiment_volatility_{window_hours}h'] = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
            
            # Cross-window comparisons
            if len(self.params['sentiment_windows']) >= 2:
                short_window = self.params['sentiment_windows'][0]
                long_window = self.params['sentiment_windows'][-1]
                
                short_sentiment = features.get(f'sentiment_avg_{short_window}h', 0)
                long_sentiment = features.get(f'sentiment_avg_{long_window}h', 0)
                
                features['sentiment_momentum'] = short_sentiment - long_sentiment
                features['sentiment_acceleration'] = features.get(f'sentiment_avg_{short_window}h', 0) - features.get(f'sentiment_avg_{self.params["sentiment_windows"][1]}h', 0) if len(self.params['sentiment_windows']) > 1 else 0
            
        except Exception as e:
            logger.warning("Error in windowed sentiment features", error=str(e))
        
        return features
    
    def _source_specific_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Source-specific sentiment features"""
        features = {}
        
        try:
            if 'source' not in data.columns:
                return features
            
            cutoff_24h = current_time - timedelta(hours=24)
            recent_data = data[data['timestamp'] >= cutoff_24h] if 'timestamp' in data.columns else data
            
            for source in self.params['source_weights'].keys():
                source_data = recent_data[recent_data['source'] == source]
                
                if source_data.empty:
                    features[f'{source}_sentiment_24h'] = 0.0
                    features[f'{source}_mentions_24h'] = 0
                    features[f'{source}_confidence_24h'] = 0.0
                    continue
                
                # Source-specific sentiment
                sentiment_scores = source_data['sentiment_score'].values
                confidences = source_data['confidence'].values
                
                weighted_sentiment = np.average(sentiment_scores, weights=confidences) if len(sentiment_scores) > 0 else 0
                features[f'{source}_sentiment_24h'] = weighted_sentiment
                features[f'{source}_confidence_24h'] = np.mean(confidences) if len(confidences) > 0 else 0
                
                # Source-specific mention count
                if 'mention_count' in source_data.columns:
                    features[f'{source}_mentions_24h'] = source_data['mention_count'].sum()
                else:
                    features[f'{source}_mentions_24h'] = len(source_data)
                
                # Source reliability (based on historical accuracy if available)
                features[f'{source}_reliability'] = self.params['source_weights'].get(source, 0.25)
            
            # Source diversity
            active_sources = len(recent_data['source'].unique()) if not recent_data.empty else 0
            features['source_diversity'] = active_sources / len(self.params['source_weights'])
            
            # Source agreement
            if len(recent_data) > 1:
                source_sentiments = []
                for source in recent_data['source'].unique():
                    source_sentiment = recent_data[recent_data['source'] == source]['sentiment_score'].mean()
                    source_sentiments.append(source_sentiment)
                
                if len(source_sentiments) > 1:
                    features['source_agreement'] = 1 - np.std(source_sentiments)  # Higher = more agreement
            
        except Exception as e:
            logger.warning("Error in source-specific features", error=str(e))
        
        return features
    
    def _sentiment_momentum_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Sentiment momentum and change features"""
        features = {}
        
        try:
            if 'timestamp' not in data.columns or len(data) < 2:
                return features
            
            # Calculate sentiment changes over different periods
            time_periods = [1, 4, 12, 24]  # hours
            
            for hours in time_periods:
                current_time_end = current_time
                current_time_start = current_time - timedelta(hours=hours/2)
                prev_time_end = current_time_start
                prev_time_start = current_time - timedelta(hours=hours)
                
                current_period = data[
                    (data['timestamp'] >= current_time_start) & 
                    (data['timestamp'] <= current_time_end)
                ]
                
                prev_period = data[
                    (data['timestamp'] >= prev_time_start) & 
                    (data['timestamp'] <= prev_time_end)
                ]
                
                if not current_period.empty and not prev_period.empty:
                    current_sentiment = np.average(
                        current_period['sentiment_score'], 
                        weights=current_period['confidence']
                    )
                    prev_sentiment = np.average(
                        prev_period['sentiment_score'], 
                        weights=prev_period['confidence']
                    )
                    
                    features[f'sentiment_change_{hours}h'] = current_sentiment - prev_sentiment
                    features[f'sentiment_momentum_{hours}h'] = (current_sentiment - prev_sentiment) / (abs(prev_sentiment) + 1e-6)
            
            # Sentiment velocity (rate of change)
            if len(data) >= 3:
                # Sort by timestamp
                sorted_data = data.sort_values('timestamp')
                
                # Calculate rolling sentiment
                window_size = min(len(sorted_data) // 3, 5)
                if window_size >= 2:
                    rolling_sentiment = sorted_data['sentiment_score'].rolling(window=window_size, center=True).mean()
                    
                    # Calculate first derivative (velocity)
                    sentiment_velocity = np.gradient(rolling_sentiment.dropna().values)
                    if len(sentiment_velocity) > 0:
                        features['sentiment_velocity'] = sentiment_velocity[-1]
                        
                        # Calculate second derivative (acceleration)
                        if len(sentiment_velocity) > 1:
                            sentiment_acceleration = np.gradient(sentiment_velocity)
                            features['sentiment_acceleration'] = sentiment_acceleration[-1]
            
        except Exception as e:
            logger.warning("Error in sentiment momentum features", error=str(e))
        
        return features
    
    def _emotion_engagement_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Emotion and engagement-based features"""
        features = {}
        
        try:
            cutoff_24h = current_time - timedelta(hours=24)
            recent_data = data[data['timestamp'] >= cutoff_24h] if 'timestamp' in data.columns else data
            
            if recent_data.empty:
                return features
            
            # Engagement metrics
            if 'total_engagement' in recent_data.columns:
                engagement = recent_data['total_engagement'].values
                features['total_engagement_24h'] = np.sum(engagement)
                features['avg_engagement_24h'] = np.mean(engagement) if len(engagement) > 0 else 0
                features['max_engagement_24h'] = np.max(engagement) if len(engagement) > 0 else 0
            
            if 'avg_engagement' in recent_data.columns:
                avg_eng = recent_data['avg_engagement'].values
                features['engagement_quality_24h'] = np.mean(avg_eng) if len(avg_eng) > 0 else 0
            
            # Emotion analysis (if available in raw_data)
            if 'emotions' in recent_data.columns:
                emotion_scores = {}
                for emotion in self.params['emotion_categories']:
                    emotion_values = []
                    for emotions_dict in recent_data['emotions'].dropna():
                        if isinstance(emotions_dict, dict) and emotion in emotions_dict:
                            emotion_values.append(emotions_dict[emotion])
                    
                    if emotion_values:
                        features[f'emotion_{emotion}_24h'] = np.mean(emotion_values)
            
            # Keyword analysis
            if 'keywords' in recent_data.columns:
                all_keywords = []
                for keywords_list in recent_data['keywords'].dropna():
                    if isinstance(keywords_list, list):
                        all_keywords.extend(keywords_list)
                
                # Count unique keywords (diversity measure)
                features['keyword_diversity_24h'] = len(set(all_keywords)) if all_keywords else 0
                features['keyword_total_24h'] = len(all_keywords)
            
            # Quality score aggregation
            if 'quality_score' in recent_data.columns:
                quality_scores = recent_data['quality_score'].values
                features['avg_quality_24h'] = np.mean(quality_scores) if len(quality_scores) > 0 else 0
                features['min_quality_24h'] = np.min(quality_scores) if len(quality_scores) > 0 else 0
            
        except Exception as e:
            logger.warning("Error in emotion/engagement features", error=str(e))
        
        return features
    
    def _anomaly_detection_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Anomaly and spike detection features"""
        features = {}
        
        try:
            if len(data) < 10:
                return features
            
            # Calculate historical baseline (past week if available)
            week_ago = current_time - timedelta(days=7)
            historical_data = data[data['timestamp'] <= week_ago] if 'timestamp' in data.columns else data[:-24]
            recent_data = data[data['timestamp'] > week_ago] if 'timestamp' in data.columns else data[-24:]
            
            if historical_data.empty or recent_data.empty:
                return features
            
            # Sentiment spike detection
            hist_sentiment_mean = historical_data['sentiment_score'].mean()
            hist_sentiment_std = historical_data['sentiment_score'].std()
            
            if hist_sentiment_std > 0:
                recent_sentiment = recent_data['sentiment_score'].mean()
                sentiment_zscore = (recent_sentiment - hist_sentiment_mean) / hist_sentiment_std
                
                features['sentiment_zscore'] = sentiment_zscore
                features['sentiment_spike'] = 1 if abs(sentiment_zscore) > 2 else 0
                features['bullish_spike'] = 1 if sentiment_zscore > 2 else 0
                features['bearish_spike'] = 1 if sentiment_zscore < -2 else 0
            
            # Mention volume spike
            if 'mention_count' in data.columns:
                hist_mentions_mean = historical_data['mention_count'].mean()
                hist_mentions_std = historical_data['mention_count'].std()
                
                if hist_mentions_std > 0:
                    recent_mentions = recent_data['mention_count'].sum()
                    mention_zscore = (recent_mentions - hist_mentions_mean * len(recent_data)) / (hist_mentions_std * np.sqrt(len(recent_data)))
                    
                    features['mention_volume_zscore'] = mention_zscore
                    features['viral_activity'] = 1 if mention_zscore > 3 else 0
            
            # Sudden sentiment shift
            if len(recent_data) >= 6:
                first_half = recent_data.iloc[:len(recent_data)//2]['sentiment_score'].mean()
                second_half = recent_data.iloc[len(recent_data)//2:]['sentiment_score'].mean()
                
                features['sentiment_shift'] = second_half - first_half
                features['major_shift'] = 1 if abs(second_half - first_half) > 0.3 else 0
            
        except Exception as e:
            logger.warning("Error in anomaly detection features", error=str(e))
        
        return features
    
    def _consensus_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """Cross-source consensus features"""
        features = {}
        
        try:
            if 'source' not in data.columns:
                return features
            
            cutoff_24h = current_time - timedelta(hours=24)
            recent_data = data[data['timestamp'] >= cutoff_24h] if 'timestamp' in data.columns else data
            
            if recent_data.empty:
                return features
            
            # Group by source and calculate average sentiment
            source_sentiments = recent_data.groupby('source')['sentiment_score'].mean()
            
            if len(source_sentiments) > 1:
                # Overall consensus (low std = high consensus)
                features['consensus_strength'] = 1 / (1 + source_sentiments.std())
                
                # Directional consensus
                positive_sources = (source_sentiments > 0.1).sum()
                negative_sources = (source_sentiments < -0.1).sum()
                total_sources = len(source_sentiments)
                
                features['bullish_consensus'] = positive_sources / total_sources
                features['bearish_consensus'] = negative_sources / total_sources
                
                # Polarization (sources strongly disagree)
                features['sentiment_polarization'] = source_sentiments.std()
                
                # Weighted consensus (by source reliability)
                weighted_sentiment = 0
                total_weight = 0
                
                for source, sentiment in source_sentiments.items():
                    weight = self.params['source_weights'].get(source, 0.25)
                    weighted_sentiment += sentiment * weight
                    total_weight += weight
                
                if total_weight > 0:
                    features['weighted_consensus'] = weighted_sentiment / total_weight
            
        except Exception as e:
            logger.warning("Error in consensus features", error=str(e))
        
        return features
    
    def _political_intelligence_features(self, data: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
        """UnusualWhales political intelligence features"""
        features = {}
        
        try:
            # Filter for UnusualWhales data
            uw_data = data[data.get('source', pd.Series()) == 'unusual_whales']
            
            if uw_data.empty:
                features['political_activity'] = 0
                features['insider_sentiment'] = 0
                features['congress_interest'] = 0
                return features
            
            cutoff_week = current_time - timedelta(days=7)
            recent_uw = uw_data[uw_data['timestamp'] >= cutoff_week] if 'timestamp' in uw_data.columns else uw_data
            
            if recent_uw.empty:
                features['political_activity'] = 0
                features['insider_sentiment'] = 0
                features['congress_interest'] = 0
                return features
            
            # Political activity level
            features['political_activity'] = len(recent_uw)
            
            # Insider sentiment (from political trades)
            if not recent_uw['sentiment_score'].empty:
                features['insider_sentiment'] = recent_uw['sentiment_score'].mean()
                features['insider_confidence'] = recent_uw['confidence'].mean()
            
            # Congress interest level (based on mention frequency)
            if 'mention_count' in recent_uw.columns:
                features['congress_interest'] = recent_uw['mention_count'].sum()
            
            # Party-specific analysis (if available in raw_data)
            if 'raw_data' in recent_uw.columns:
                republican_trades = 0
                democrat_trades = 0
                
                for raw_data_dict in recent_uw['raw_data'].dropna():
                    if isinstance(raw_data_dict, dict):
                        party = raw_data_dict.get('party', '').lower()
                        if 'republican' in party or 'gop' in party:
                            republican_trades += 1
                        elif 'democrat' in party or 'dem' in party:
                            democrat_trades += 1
                
                total_political_trades = republican_trades + democrat_trades
                if total_political_trades > 0:
                    features['republican_ratio'] = republican_trades / total_political_trades
                    features['democrat_ratio'] = democrat_trades / total_political_trades
                    features['bipartisan_interest'] = min(republican_trades, democrat_trades) / max(republican_trades, democrat_trades) if max(republican_trades, democrat_trades) > 0 else 0
            
        except Exception as e:
            logger.warning("Error in political intelligence features", error=str(e))
        
        return features
    
    def _calculate_time_weights(self, timestamps: pd.Series, current_time: datetime) -> np.ndarray:
        """Calculate exponential time decay weights"""
        try:
            time_diffs = (current_time - timestamps).dt.total_seconds() / 3600  # Hours
            weights = np.exp(-time_diffs * (1 - self.params['decay_factor']))
            return weights / np.sum(weights)  # Normalize
        except Exception:
            return np.ones(len(timestamps)) / len(timestamps)
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return dictionary of empty/zero features"""
        empty_features = {}
        
        # Basic features
        basic_features = [
            'sentiment_avg_24h', 'sentiment_raw_avg_24h', 'sentiment_std_24h',
            'confidence_avg_24h', 'mention_volume_24h', 'bullish_ratio_24h',
            'bearish_ratio_24h', 'neutral_ratio_24h'
        ]
        
        # Windowed features
        for window in self.params['sentiment_windows']:
            basic_features.extend([
                f'sentiment_avg_{window}h',
                f'mention_count_{window}h',
                f'sentiment_volatility_{window}h'
            ])
        
        # Source features
        for source in self.params['source_weights'].keys():
            basic_features.extend([
                f'{source}_sentiment_24h',
                f'{source}_mentions_24h',
                f'{source}_confidence_24h'
            ])
        
        # Set all to zero
        for feature in basic_features:
            empty_features[feature] = 0.0
        
        return empty_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible sentiment feature names"""
        feature_names = []
        
        # Basic sentiment features
        feature_names.extend([
            'sentiment_avg_24h', 'sentiment_raw_avg_24h', 'sentiment_std_24h', 'sentiment_median_24h',
            'confidence_avg_24h', 'confidence_min_24h', 'confidence_max_24h',
            'mention_volume_24h', 'mention_avg_24h', 'mention_max_24h',
            'bullish_ratio_24h', 'bearish_ratio_24h', 'neutral_ratio_24h',
            'bullish_signals_24h', 'bearish_signals_24h', 'signal_ratio_24h'
        ])
        
        # Windowed features
        for window in self.params['sentiment_windows']:
            feature_names.extend([
                f'sentiment_avg_{window}h',
                f'mention_count_{window}h',
                f'sentiment_volatility_{window}h'
            ])
        
        # Source-specific features
        for source in self.params['source_weights'].keys():
            feature_names.extend([
                f'{source}_sentiment_24h',
                f'{source}_mentions_24h', 
                f'{source}_confidence_24h',
                f'{source}_reliability'
            ])
        
        # Additional features
        feature_names.extend([
            'sentiment_momentum', 'sentiment_acceleration', 'sentiment_velocity',
            'source_diversity', 'source_agreement', 'consensus_strength',
            'bullish_consensus', 'bearish_consensus', 'sentiment_polarization',
            'weighted_consensus', 'sentiment_zscore', 'sentiment_spike',
            'bullish_spike', 'bearish_spike', 'mention_volume_zscore',
            'viral_activity', 'sentiment_shift', 'major_shift',
            'political_activity', 'insider_sentiment', 'congress_interest'
        ])
        
        return feature_names
    
    def __str__(self) -> str:
        return f"SentimentFeatures(sources={len(self.params['source_weights'])}, windows={len(self.params['sentiment_windows'])})"
    
    def __repr__(self) -> str:
        return self.__str__()