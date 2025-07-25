"""
Dynamic Symbol Discovery System

Automatically discovers trending symbols from sentiment sources and adds them
to the trading universe based on configurable criteria.
"""

import re
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import structlog
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config.config_manager import ConfigManager
from ..sentiment.reddit_analyzer import RedditSentimentAnalyzer
from ..sentiment.news_aggregator import NewsAggregator
from ..data.market_data_manager import MarketDataManager

logger = structlog.get_logger(__name__)


@dataclass
class SymbolMention:
    """Individual symbol mention from sentiment source"""
    symbol: str
    source: str
    timestamp: datetime
    context: str
    sentiment_score: float
    confidence: float
    volume_mentioned: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context[:200],  # Truncate context
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'volume_mentioned': self.volume_mentioned
        }


@dataclass
class SymbolScore:
    """Aggregated score for a discovered symbol"""
    symbol: str
    total_mentions: int
    weighted_sentiment: float
    confidence: float
    trending_score: float
    sources: List[str]
    first_seen: datetime
    last_seen: datetime
    
    def calculate_final_score(self, config: Dict[str, Any]) -> float:
        """Calculate final score based on multiple factors"""
        # Base score from mentions
        mention_score = min(self.total_mentions / config.get('mention_normalizer', 100), 1.0)
        
        # Sentiment component
        sentiment_component = max(0, self.weighted_sentiment) * config.get('sentiment_weight', 0.3)
        
        # Trending component (recency bias)
        hours_since_last = (datetime.now() - self.last_seen).total_seconds() / 3600
        trending_component = max(0, 1 - hours_since_last / 24) * config.get('trending_weight', 0.2)
        
        # Source diversity bonus
        source_diversity = len(set(self.sources)) / 4  # Max 4 sources
        diversity_bonus = source_diversity * config.get('diversity_weight', 0.1)
        
        # Confidence component
        confidence_component = self.confidence * config.get('confidence_weight', 0.4)
        
        final_score = (mention_score + sentiment_component + trending_component + 
                      diversity_bonus + confidence_component)
        
        return min(final_score, 1.0)


@dataclass
class DiscoveryConfig:
    """Configuration for dynamic symbol discovery"""
    enabled: bool = True
    sentiment_threshold: int = 50
    trending_window_hours: int = 24
    max_new_symbols_daily: int = 5
    confidence_threshold: float = 0.7
    
    # Source weights
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        'reddit': 0.4,
        'news': 0.3,
        'twitter': 0.2,
        'unusual_whales': 0.1
    })
    
    # Scoring parameters
    mention_normalizer: int = 100
    sentiment_weight: float = 0.3
    trending_weight: float = 0.2
    diversity_weight: float = 0.1
    confidence_weight: float = 0.4
    
    # Filtering criteria
    min_market_cap: float = 1e9
    min_volume: float = 1e6
    max_spread: float = 0.01
    
    # Update frequency
    discovery_interval_minutes: int = 30
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveryConfig':
        """Create config from dictionary"""
        return cls(**data)


class SymbolExtractor:
    """Extract stock symbols from text using various patterns"""
    
    def __init__(self):
        # Common stock symbol patterns
        self.symbol_patterns = [
            # $SYMBOL format
            re.compile(r'\$([A-Z]{1,5})\b', re.IGNORECASE),
            # SYMBOL: or SYMBOL - format
            re.compile(r'\b([A-Z]{2,5})(?:\s*[:|-])', re.IGNORECASE),
            # Standalone symbols (more conservative)
            re.compile(r'\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|calls|puts|options|up|down|moon|rocket))', re.IGNORECASE),
        ]
        
        # Common false positives to filter out
        self.false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'USE', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO',
            'ITS', 'DID', 'YES', 'HIS', 'HAS', 'HAD', 'LET', 'PUT', 'TOO', 'END', 'HOW', 'AIR',
            'MEN', 'OFF', 'SET', 'OWN', 'TAKE', 'COME', 'WORK', 'LIFE', 'INTO', 'TIME', 'VERY',
            'WHAT', 'WITH', 'HAVE', 'FROM', 'THEY', 'KNOW', 'WANT', 'BEEN', 'GOOD', 'MUCH',
            'SOME', 'WOULD', 'THERE', 'MAKE', 'WELL', 'YEAR', 'BACK', 'THINK', 'FIRST', 'STILL',
            'AFTER', 'OTHER', 'MANY', 'MUST', 'OVER', 'SUCH', 'USED', 'MOST', 'STATE', 'EVEN',
            'MADE', 'SCHOOL', 'HOUSE', 'WORLD', 'STILL', 'SMALL', 'FOUND', 'HERE', 'GIVE',
            'GENERAL', 'PUBLIC', 'HAND', 'PART', 'PLACE', 'CASE', 'FACT', 'GROUP', 'PROBLEM',
            'RIGHT', 'SAME', 'SEEM', 'TELL', 'POINT', 'ASKED', 'WENT', 'MONEY', 'STORY',
            'UNTIL', 'QUITE', 'BEGAN', 'MIGHT', 'NEVER', 'SYSTEM', 'ORDER', 'DIDN', 'DOESN',
            'HAVEN', 'WASN', 'WEREN', 'WOULDN', 'SHOULDN', 'COULDN', 'MUSTN', 'ISN', 'WON',
            'CAN', 'CALL', 'PUT', 'BUY', 'SELL', 'HOLD', 'MOON', 'YOLO', 'FOMO', 'HODL',
            'DD', 'TA', 'FA', 'IPO', 'CEO', 'CFO', 'USD', 'EUR', 'GBP', 'JPY', 'CAD'
        }
        
        # Load known valid symbols (you might want to update this periodically)
        self.known_symbols = self._load_known_symbols()
    
    def _load_known_symbols(self) -> Set[str]:
        """Load list of known valid stock symbols"""
        # This could be loaded from a file or API
        # For now, return common symbols
        return {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD',
            'INTC', 'CRM', 'ADBE', 'PYPL', 'SHOP', 'UBER', 'LYFT', 'ROKU', 'ZM', 'DOCU',
            'SNOW', 'PLTR', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'CRWD', 'OKTA', 'DDOG', 'NET',
            'TWLO', 'SQ', 'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK', 'SQQQ', 'TQQQ', 'UPRO',
            'GME', 'AMC', 'BB', 'NOK', 'WISH', 'CLOV', 'SPCE', 'TLRY', 'SNDL', 'MVIS'
        }
    
    def extract_symbols(self, text: str) -> List[Tuple[str, float]]:
        """Extract symbols from text with confidence scores"""
        found_symbols = []
        
        for pattern in self.symbol_patterns:
            matches = pattern.findall(text)
            for match in matches:
                symbol = match.upper().strip()
                
                # Filter out obvious false positives
                if symbol in self.false_positives:
                    continue
                
                # Must be 2-5 characters
                if not (2 <= len(symbol) <= 5):
                    continue
                
                # Calculate confidence based on context and known symbols
                confidence = self._calculate_confidence(symbol, text)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    found_symbols.append((symbol, confidence))
        
        # Remove duplicates and sort by confidence
        symbol_dict = {}
        for symbol, confidence in found_symbols:
            if symbol not in symbol_dict or confidence > symbol_dict[symbol]:
                symbol_dict[symbol] = confidence
        
        return [(symbol, conf) for symbol, conf in 
                sorted(symbol_dict.items(), key=lambda x: x[1], reverse=True)]
    
    def _calculate_confidence(self, symbol: str, text: str) -> float:
        """Calculate confidence score for extracted symbol"""
        confidence = 0.5  # Base confidence
        
        # Known symbol bonus
        if symbol in self.known_symbols:
            confidence += 0.3
        
        # Context clues
        financial_keywords = [
            'stock', 'share', 'price', 'trading', 'buy', 'sell', 'calls', 'puts',
            'earnings', 'revenue', 'profit', 'loss', 'bullish', 'bearish',
            'moon', 'rocket', 'diamond', 'hands', 'squeeze', 'short'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in text_lower)
        confidence += min(keyword_matches * 0.05, 0.2)
        
        # $ prefix bonus (strong indicator)
        if f'${symbol}' in text:
            confidence += 0.2
        
        # Length penalty for very short symbols (likely false positives)
        if len(symbol) == 2:
            confidence -= 0.1
        
        return min(confidence, 1.0)


class DynamicSymbolDiscovery:
    """Main dynamic symbol discovery system"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = DiscoveryConfig.from_dict(
            config_manager.config.universe.get('dynamic_discovery', {})
        )
        
        # Initialize components
        self.symbol_extractor = SymbolExtractor()
        self.market_data_manager = MarketDataManager(config_manager)
        
        # Storage
        self.symbol_mentions: List[SymbolMention] = []
        self.discovered_symbols: Dict[str, SymbolScore] = {}
        self.current_universe: Set[str] = set()
        
        # Initialize sentiment analyzers
        self._init_sentiment_sources()
        
        # Discovery tracking
        self.last_discovery_run = datetime.now()
        self.daily_additions = 0
        self.daily_reset_date = datetime.now().date()
        
        logger.info("Dynamic symbol discovery initialized", enabled=self.config.enabled)
    
    def _init_sentiment_sources(self):
        """Initialize sentiment data sources"""
        try:
            self.reddit_analyzer = RedditSentimentAnalyzer(self.config_manager)
        except Exception as e:
            logger.warning("Reddit analyzer not available", error=str(e))
            self.reddit_analyzer = None
        
        try:
            self.news_aggregator = NewsAggregator(self.config_manager)
        except Exception as e:
            logger.warning("News aggregator not available", error=str(e))
            self.news_aggregator = None
    
    async def run_discovery(self) -> List[str]:
        """Run symbol discovery process and return new symbols found"""
        
        if not self.config.enabled:
            return []
        
        # Reset daily counter if needed
        if datetime.now().date() > self.daily_reset_date:
            self.daily_additions = 0
            self.daily_reset_date = datetime.now().date()
        
        # Check if we've hit daily limit
        if self.daily_additions >= self.config.max_new_symbols_daily:
            logger.info("Daily symbol addition limit reached", 
                       limit=self.config.max_new_symbols_daily)
            return []
        
        logger.info("Starting symbol discovery run")
        
        # Collect mentions from all sources
        mentions = await self._collect_mentions()
        self.symbol_mentions.extend(mentions)
        
        # Clean old mentions
        self._clean_old_mentions()
        
        # Analyze and score symbols
        symbol_scores = self._analyze_symbols()
        
        # Validate and filter candidates
        candidates = await self._validate_candidates(symbol_scores)
        
        # Select final symbols to add
        new_symbols = self._select_new_symbols(candidates)
        
        if new_symbols:
            logger.info("Discovered new symbols", symbols=new_symbols)
            self.daily_additions += len(new_symbols)
        
        return new_symbols
    
    async def _collect_mentions(self) -> List[SymbolMention]:
        """Collect symbol mentions from all sentiment sources"""
        mentions = []
        
        # Collect from Reddit
        if self.reddit_analyzer:
            reddit_mentions = await self._collect_reddit_mentions()
            mentions.extend(reddit_mentions)
        
        # Collect from News
        if self.news_aggregator:
            news_mentions = await self._collect_news_mentions()
            mentions.extend(news_mentions)
        
        logger.info("Collected mentions", total=len(mentions))
        return mentions
    
    async def _collect_reddit_mentions(self) -> List[SymbolMention]:
        """Collect mentions from Reddit"""
        mentions = []
        
        try:
            # Get recent posts from trading subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
            cutoff_time = datetime.now() - timedelta(hours=self.config.trending_window_hours)
            
            for subreddit in subreddits:
                posts = await self.reddit_analyzer.get_recent_posts(
                    subreddit, limit=100, since=cutoff_time
                )
                
                for post in posts:
                    # Extract symbols from post content
                    text = f"{post.get('title', '')} {post.get('selftext', '')}"
                    extracted_symbols = self.symbol_extractor.extract_symbols(text)
                    
                    # Create mentions
                    for symbol, confidence in extracted_symbols:
                        mention = SymbolMention(
                            symbol=symbol,
                            source=f'reddit_{subreddit}',
                            timestamp=datetime.fromtimestamp(post.get('created_utc', time.time())),
                            context=text[:500],
                            sentiment_score=post.get('sentiment_score', 0.0),
                            confidence=confidence,
                            volume_mentioned=post.get('score', 1)
                        )
                        mentions.append(mention)
        
        except Exception as e:
            logger.error("Error collecting Reddit mentions", error=str(e))
        
        return mentions
    
    async def _collect_news_mentions(self) -> List[SymbolMention]:
        """Collect mentions from news sources"""
        mentions = []
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.trending_window_hours)
            news_articles = await self.news_aggregator.get_recent_articles(since=cutoff_time)
            
            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                extracted_symbols = self.symbol_extractor.extract_symbols(text)
                
                for symbol, confidence in extracted_symbols:
                    mention = SymbolMention(
                        symbol=symbol,
                        source='news',
                        timestamp=article.get('publishedAt', datetime.now()),
                        context=text[:500],
                        sentiment_score=article.get('sentiment_score', 0.0),
                        confidence=confidence,
                        volume_mentioned=1
                    )
                    mentions.append(mention)
        
        except Exception as e:
            logger.error("Error collecting news mentions", error=str(e))
        
        return mentions
    
    def _clean_old_mentions(self):
        """Remove mentions older than trending window"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.trending_window_hours * 2)
        
        initial_count = len(self.symbol_mentions)
        self.symbol_mentions = [
            mention for mention in self.symbol_mentions
            if mention.timestamp > cutoff_time
        ]
        
        removed = initial_count - len(self.symbol_mentions)
        if removed > 0:
            logger.debug("Cleaned old mentions", removed=removed)
    
    def _analyze_symbols(self) -> Dict[str, SymbolScore]:
        """Analyze collected mentions and create symbol scores"""
        symbol_data = defaultdict(list)
        
        # Group mentions by symbol
        for mention in self.symbol_mentions:
            symbol_data[mention.symbol].append(mention)
        
        symbol_scores = {}
        
        for symbol, mentions in symbol_data.items():
            if len(mentions) < self.config.sentiment_threshold / 10:  # Minimum mentions
                continue
            
            # Calculate aggregated metrics
            total_mentions = sum(m.volume_mentioned for m in mentions)
            
            # Weighted sentiment (by confidence and volume)
            sentiment_sum = sum(m.sentiment_score * m.confidence * m.volume_mentioned for m in mentions)
            weight_sum = sum(m.confidence * m.volume_mentioned for m in mentions)
            weighted_sentiment = sentiment_sum / weight_sum if weight_sum > 0 else 0
            
            # Average confidence
            confidence = np.mean([m.confidence for m in mentions])
            
            # Trending score (based on recency distribution)
            recent_mentions = [m for m in mentions 
                             if m.timestamp > datetime.now() - timedelta(hours=6)]
            trending_score = len(recent_mentions) / len(mentions) if mentions else 0
            
            # Source diversity
            sources = [m.source for m in mentions]
            
            # Time range
            timestamps = [m.timestamp for m in mentions]
            first_seen = min(timestamps)
            last_seen = max(timestamps)
            
            symbol_score = SymbolScore(
                symbol=symbol,
                total_mentions=total_mentions,
                weighted_sentiment=weighted_sentiment,
                confidence=confidence,
                trending_score=trending_score,
                sources=sources,
                first_seen=first_seen,
                last_seen=last_seen
            )
            
            symbol_scores[symbol] = symbol_score
        
        return symbol_scores
    
    async def _validate_candidates(self, symbol_scores: Dict[str, SymbolScore]) -> List[SymbolScore]:
        """Validate symbol candidates against market criteria"""
        candidates = []
        
        for symbol, score in symbol_scores.items():
            # Skip if already in universe
            if symbol in self.current_universe:
                continue
            
            # Check mention threshold
            if score.total_mentions < self.config.sentiment_threshold:
                continue
            
            # Check confidence
            if score.confidence < self.config.confidence_threshold:
                continue
            
            # Validate against market data
            try:
                is_valid = await self._validate_market_data(symbol)
                if is_valid:
                    candidates.append(score)
                else:
                    logger.debug("Symbol failed market validation", symbol=symbol)
            except Exception as e:
                logger.debug("Error validating symbol", symbol=symbol, error=str(e))
        
        # Sort by final score
        candidates.sort(key=lambda x: x.calculate_final_score(self.config.__dict__), reverse=True)
        
        return candidates
    
    async def _validate_market_data(self, symbol: str) -> bool:
        """Validate symbol against market data requirements"""
        try:
            # Get basic market data
            data = await self.market_data_manager.get_symbol_info(symbol)
            
            if not data:
                return False
            
            # Check market cap
            market_cap = data.get('market_cap', 0)
            if market_cap < self.config.min_market_cap:
                return False
            
            # Check average volume
            avg_volume = data.get('avg_volume', 0)
            if avg_volume < self.config.min_volume:
                return False
            
            # Check bid-ask spread (if available)
            spread = data.get('spread', 0)
            if spread > self.config.max_spread:
                return False
            
            return True
            
        except Exception as e:
            logger.debug("Market validation error", symbol=symbol, error=str(e))
            return False
    
    def _select_new_symbols(self, candidates: List[SymbolScore]) -> List[str]:
        """Select final symbols to add to universe"""
        remaining_slots = self.config.max_new_symbols_daily - self.daily_additions
        
        if remaining_slots <= 0:
            return []
        
        # Take top candidates up to remaining slots
        selected = candidates[:remaining_slots]
        
        return [candidate.symbol for candidate in selected]
    
    def update_universe(self, current_symbols: Set[str]):
        """Update current universe knowledge"""
        self.current_universe = current_symbols.copy()
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        return {
            'enabled': self.config.enabled,
            'total_mentions': len(self.symbol_mentions),
            'unique_symbols_mentioned': len(set(m.symbol for m in self.symbol_mentions)),
            'daily_additions_used': self.daily_additions,
            'daily_limit': self.config.max_new_symbols_daily,
            'last_discovery_run': self.last_discovery_run.isoformat(),
            'current_universe_size': len(self.current_universe)
        }
    
    def save_discovery_data(self, path: Path):
        """Save discovery data for analysis"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'mentions': [m.to_dict() for m in self.symbol_mentions[-1000:]],  # Last 1000
            'stats': self.get_discovery_stats()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Discovery data saved", path=str(path))