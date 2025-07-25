"""
News Aggregator for QuantumSentiment Trading Bot

Comprehensive financial news analysis from multiple sources:
- Alpha Vantage News API
- NewsAPI integration
- RSS feed monitoring
- Financial news sentiment analysis
- Event impact assessment
- Real-time news alerts
"""

import re
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from transformers import pipeline
import feedparser
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NewsConfig:
    """Configuration for news aggregator"""
    alpha_vantage_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    
    # News sources
    rss_feeds: List[str] = None
    news_sources: List[str] = None
    
    # Analysis parameters
    max_articles_per_source: int = 50
    lookback_hours: int = 24
    min_relevance_score: float = 0.3
    
    # Language and region
    language: str = 'en'
    country: str = 'us'
    
    def __post_init__(self):
        if self.rss_feeds is None:
            self.rss_feeds = [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://feeds.reuters.com/reuters/businessNews',
                'https://rss.cnn.com/rss/money_latest.rss',
                'https://feeds.marketwatch.com/marketwatch/marketpulse/',
                'https://seekingalpha.com/feed.xml',
                'https://feeds.benzinga.com/benzinga',
                'https://finance.yahoo.com/news/rssindex'
            ]
        
        if self.news_sources is None:
            self.news_sources = [
                'bloomberg', 'reuters', 'cnbc', 'marketwatch', 
                'seekingalpha', 'benzinga', 'yahoo-finance',
                'financial-times', 'wall-street-journal'
            ]


class NewsAggregator:
    """Advanced financial news aggregation and sentiment analysis"""
    
    def __init__(self, config: NewsConfig):
        """
        Initialize news aggregator
        
        Args:
            config: News aggregation configuration
        """
        self.config = config
        self.sentiment_analyzer = None
        
        # Market impact keywords (weighted by importance)
        self.impact_keywords = {
            'earnings': 3, 'revenue': 2, 'profit': 2, 'loss': 2,
            'acquisition': 3, 'merger': 3, 'buyout': 3,
            'ipo': 2, 'listing': 1, 'delisting': 2,
            'fda approval': 3, 'drug approval': 3, 'clinical trial': 2,
            'lawsuit': 2, 'settlement': 1, 'investigation': 2,
            'recall': 2, 'bankruptcy': 3, 'restructuring': 2,
            'dividend': 1, 'stock split': 1, 'buyback': 1,
            'guidance': 2, 'outlook': 2, 'forecast': 2,
            'ceo': 2, 'management': 1, 'resignation': 2,
            'partnership': 1, 'contract': 1, 'deal': 1
        }
        
        # Sector keywords for classification
        self.sector_keywords = {
            'technology': ['tech', 'software', 'ai', 'cloud', 'digital', 'cyber'],
            'finance': ['bank', 'financial', 'fintech', 'payment', 'credit', 'lending'],
            'healthcare': ['pharma', 'biotech', 'medical', 'health', 'drug', 'vaccine'],
            'energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'electric'],
            'retail': ['retail', 'consumer', 'e-commerce', 'shopping', 'brand'],
            'automotive': ['auto', 'car', 'vehicle', 'tesla', 'electric vehicle'],
            'real_estate': ['real estate', 'reit', 'property', 'housing', 'construction']
        }
        
        logger.info("News aggregator initialized", 
                   sources=len(config.news_sources),
                   feeds=len(config.rss_feeds))
    
    def initialize(self) -> bool:
        """Initialize news aggregator and sentiment analyzer"""
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1  # CPU
            )
            
            logger.info("FinBERT sentiment analyzer initialized for news")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize news aggregator", error=str(e))
            return False
    
    def analyze_symbol(self, symbol: str, hours_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze news sentiment for a specific symbol
        
        Args:
            symbol: Stock symbol to analyze
            hours_back: Hours to look back (default from config)
            
        Returns:
            Comprehensive news sentiment analysis
        """
        if not self.sentiment_analyzer:
            raise RuntimeError("News aggregator not initialized")
        
        hours_back = hours_back or self.config.lookback_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        logger.info("Starting news analysis", symbol=symbol, hours_back=hours_back)
        
        try:
            # Collect news from different sources
            av_articles = self._get_alpha_vantage_news(symbol, cutoff_time)
            newsapi_articles = self._get_newsapi_articles(symbol, cutoff_time)
            rss_articles = self._get_rss_articles(symbol, cutoff_time)
            
            # Combine all articles
            all_articles = av_articles + newsapi_articles + rss_articles
            
            # Remove duplicates
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Analyze sentiment
            sentiment_results = self._analyze_articles_sentiment(unique_articles)
            
            # Extract market impact signals
            impact_analysis = self._analyze_market_impact(unique_articles)
            
            # Calculate aggregated metrics
            results = {
                'symbol': symbol.upper(),
                'timestamp': datetime.utcnow(),
                'source': 'news',
                
                # Article counts
                'total_articles': len(unique_articles),
                'alpha_vantage_articles': len(av_articles),
                'newsapi_articles': len(newsapi_articles),
                'rss_articles': len(rss_articles),
                
                # Sentiment metrics
                'sentiment_score': self._calculate_weighted_sentiment(sentiment_results),
                'confidence': self._calculate_confidence(sentiment_results),
                'sentiment_distribution': self._calculate_sentiment_distribution(sentiment_results),
                
                # Impact analysis
                'market_impact_score': impact_analysis['impact_score'],
                'high_impact_events': impact_analysis['high_impact_events'],
                'sector_exposure': impact_analysis['sector_exposure'],
                
                # Timing analysis
                'recent_news_spike': self._detect_news_spike(unique_articles, cutoff_time),
                'news_velocity': self._calculate_news_velocity(unique_articles),
                
                # Source diversity
                'source_diversity': len(set([a.get('source', 'unknown') for a in unique_articles])),
                'credible_sources': len([a for a in unique_articles if self._is_credible_source(a.get('source', ''))]),
                
                # Detailed breakdowns
                'sentiment_details': sentiment_results,
                'impact_details': impact_analysis,
                'sample_articles': unique_articles[:5]  # For debugging
            }
            
            logger.info("News analysis completed",
                       symbol=symbol,
                       articles=len(unique_articles),
                       sentiment=results['sentiment_score'],
                       impact=results['market_impact_score'])
            
            return results
            
        except Exception as e:
            logger.error("News analysis failed", symbol=symbol, error=str(e))
            raise
    
    def _get_alpha_vantage_news(self, symbol: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get news from Alpha Vantage News API"""
        if not self.config.alpha_vantage_key:
            return []
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.config.alpha_vantage_key,
                'limit': self.config.max_articles_per_source
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            if 'feed' in data:
                for item in data['feed']:
                    # Parse timestamp
                    try:
                        pub_time = datetime.strptime(item['time_published'], "%Y%m%dT%H%M%S")
                    except:
                        continue
                    
                    if pub_time >= cutoff_time:
                        # Get ticker-specific sentiment if available
                        ticker_sentiment = 0.0
                        ticker_relevance = 0.0
                        
                        if 'ticker_sentiment' in item:
                            for ticker_data in item['ticker_sentiment']:
                                if ticker_data['ticker'] == symbol:
                                    ticker_sentiment = float(ticker_data.get('ticker_sentiment_score', 0))
                                    ticker_relevance = float(ticker_data.get('relevance_score', 0))
                                    break
                        
                        articles.append({
                            'title': item['title'],
                            'summary': item.get('summary', ''),
                            'url': item['url'],
                            'source': item.get('source', 'Alpha Vantage'),
                            'published_at': pub_time,
                            'authors': [item.get('authors', '')],
                            'overall_sentiment_score': float(item.get('overall_sentiment_score', 0)),
                            'ticker_sentiment_score': ticker_sentiment,
                            'relevance_score': ticker_relevance,
                            'raw_data': item
                        })
            
            logger.debug("Alpha Vantage articles collected", count=len(articles))
            return articles
            
        except Exception as e:
            logger.warning("Failed to get Alpha Vantage news", error=str(e))
            return []
    
    def _get_newsapi_articles(self, symbol: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get news from NewsAPI"""
        if not self.config.newsapi_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            
            # Search queries for the symbol
            queries = [f'"{symbol}"', f'${symbol}', symbol]
            all_articles = []
            
            for query in queries:
                params = {
                    'q': query,
                    'language': self.config.language,
                    'sortBy': 'publishedAt',
                    'apiKey': self.config.newsapi_key,
                    'pageSize': self.config.max_articles_per_source // len(queries),
                    'sources': ','.join(self.config.news_sources)
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data['status'] == 'ok':
                    for article in data['articles']:
                        # Parse timestamp
                        try:
                            pub_time = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                        except:
                            continue
                        
                        if pub_time >= cutoff_time:
                            all_articles.append({
                                'title': article['title'],
                                'summary': article.get('description', ''),
                                'url': article['url'],
                                'source': article['source']['name'],
                                'published_at': pub_time,
                                'authors': [article.get('author', '')],
                                'relevance_score': self._calculate_relevance(article, symbol),
                                'raw_data': article
                            })
                
                time.sleep(0.1)  # Rate limiting
            
            logger.debug("NewsAPI articles collected", count=len(all_articles))
            return all_articles
            
        except Exception as e:
            logger.warning("Failed to get NewsAPI articles", error=str(e))
            return []
    
    def _get_rss_articles(self, symbol: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get news from RSS feeds"""
        articles = []
        
        for feed_url in self.config.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Check if article mentions the symbol
                    content = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                    
                    if not self._mentions_symbol(content, symbol):
                        continue
                    
                    # Parse timestamp
                    try:
                        if hasattr(entry, 'published_parsed'):
                            pub_time = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed'):
                            pub_time = datetime(*entry.updated_parsed[:6])
                        else:
                            continue
                    except:
                        continue
                    
                    if pub_time >= cutoff_time:
                        articles.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'url': entry.get('link', ''),
                            'source': feed.feed.get('title', feed_url),
                            'published_at': pub_time,
                            'authors': [entry.get('author', '')],
                            'relevance_score': self._calculate_relevance(entry, symbol),
                            'raw_data': entry
                        })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to parse RSS feed {feed_url}", error=str(e))
                continue
        
        logger.debug("RSS articles collected", count=len(articles))
        return articles
    
    def _mentions_symbol(self, text: str, symbol: str) -> bool:
        """Check if text mentions the symbol"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        patterns = [
            f"\\b{symbol_lower}\\b",
            f"\\${symbol_lower}\\b",
            f"\\b{symbol_lower}\\s",
            f"\\({symbol_lower}\\)",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
    
    def _calculate_relevance(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance score for an article"""
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        
        relevance = 0.0
        
        # Direct symbol mentions
        symbol_lower = symbol.lower()
        if symbol_lower in title:
            relevance += 0.5
        if symbol_lower in summary:
            relevance += 0.3
        
        # Company name mentions (simplified - in reality you'd use a mapping)
        if len(symbol) <= 4:  # Likely a stock ticker
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            
            # Simple deduplication based on title
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _analyze_articles_sentiment(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment of articles using FinBERT"""
        sentiment_results = []
        
        for article in articles:
            try:
                # Combine title and summary for analysis
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                text = text.strip()[:512]  # FinBERT max length
                
                if not text:
                    continue
                
                # Get sentiment
                result = self.sentiment_analyzer(text)[0]
                
                # Convert to numerical score
                sentiment_score = self._convert_sentiment_score(result['label'], result['score'])
                
                # Calculate article weight based on source credibility and recency
                source_weight = self._get_source_credibility(article.get('source', ''))
                recency_weight = self._get_recency_weight(article.get('published_at'))
                relevance_weight = article.get('relevance_score', 0.5)
                
                combined_weight = source_weight * recency_weight * relevance_weight
                
                sentiment_results.append({
                    'article_title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'sentiment_label': result['label'],
                    'sentiment_score': sentiment_score,
                    'confidence': result['score'],
                    'source_weight': source_weight,
                    'recency_weight': recency_weight,
                    'relevance_weight': relevance_weight,
                    'combined_weight': combined_weight,
                    'published_at': article.get('published_at')
                })
                
            except Exception as e:
                logger.warning("Failed to analyze article sentiment", 
                             title=article.get('title', ''), error=str(e))
                continue
        
        return sentiment_results
    
    def _convert_sentiment_score(self, label: str, confidence: float) -> float:
        """Convert FinBERT label to numerical score"""
        if label.lower() == 'positive':
            return confidence
        elif label.lower() == 'negative':
            return -confidence
        else:  # neutral
            return 0.0
    
    def _analyze_market_impact(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential market impact of news"""
        impact_scores = []
        high_impact_events = []
        sector_counts = {sector: 0 for sector in self.sector_keywords.keys()}
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
            
            # Calculate impact score based on keywords
            impact_score = 0
            found_keywords = []
            
            for keyword, weight in self.impact_keywords.items():
                if keyword in text:
                    impact_score += weight
                    found_keywords.append(keyword)
            
            # Adjust by source credibility
            source_credibility = self._get_source_credibility(article.get('source', ''))
            impact_score *= source_credibility
            
            impact_scores.append(impact_score)
            
            # Track high impact events
            if impact_score >= 5:  # Threshold for high impact
                high_impact_events.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'impact_score': impact_score,
                    'keywords': found_keywords,
                    'published_at': article.get('published_at')
                })
            
            # Sector classification
            for sector, keywords in self.sector_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        sector_counts[sector] += 1
                        break
        
        # Calculate sector exposure
        total_articles = len(articles)
        sector_exposure = {}
        if total_articles > 0:
            for sector, count in sector_counts.items():
                sector_exposure[sector] = count / total_articles
        
        # Normalize impact score to 0-1 range
        mean_impact = np.mean(impact_scores) if impact_scores else 0
        max_possible_impact = 10  # Reasonable maximum (3 keywords * 3 weight * 1 credibility)
        normalized_impact = min(mean_impact / max_possible_impact, 1.0)
        
        return {
            'impact_score': normalized_impact,
            'raw_impact_score': mean_impact,  # Keep raw score for debugging
            'max_impact_score': max(impact_scores) if impact_scores else 0,
            'high_impact_events': high_impact_events,
            'sector_exposure': sector_exposure,
            'total_impact_events': len([s for s in impact_scores if s >= 3]),
            'keywords_found': list(set([kw for event in high_impact_events for kw in event.get('keywords', [])]))
        }
    
    def _detect_news_spike(self, articles: List[Dict[str, Any]], cutoff_time: datetime) -> Dict[str, Any]:
        """Detect recent news spikes"""
        if not articles:
            return {'has_spike': False, 'spike_intensity': 0}
        
        # Group articles by hour
        hourly_counts = {}
        
        for article in articles:
            pub_time = article.get('published_at')
            if pub_time:
                hour_key = pub_time.replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        
        if len(hourly_counts) < 3:
            return {'has_spike': False, 'spike_intensity': 0}
        
        # Calculate baseline and recent activity
        counts = list(hourly_counts.values())
        baseline = np.mean(counts[:-2]) if len(counts) > 2 else 0
        recent_activity = np.mean(counts[-2:])
        
        spike_ratio = recent_activity / max(baseline, 1)
        has_spike = spike_ratio > 2.0  # 2x normal activity
        
        return {
            'has_spike': has_spike,
            'spike_intensity': spike_ratio,
            'recent_articles_per_hour': recent_activity,
            'baseline_articles_per_hour': baseline
        }
    
    def _calculate_news_velocity(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate news velocity (articles per hour)"""
        if not articles:
            return 0.0
        
        # Find time span
        pub_times = [a.get('published_at') for a in articles if a.get('published_at')]
        
        if len(pub_times) < 2:
            return 0.0
        
        earliest = min(pub_times)
        latest = max(pub_times)
        
        time_span_hours = (latest - earliest).total_seconds() / 3600
        
        if time_span_hours == 0:
            return len(articles)
        
        return len(articles) / time_span_hours
    
    def _calculate_weighted_sentiment(self, sentiment_results: List[Dict[str, Any]]) -> float:
        """Calculate weighted average sentiment"""
        if not sentiment_results:
            return 0.0
        
        weighted_scores = []
        weights = []
        
        for result in sentiment_results:
            weight = result['combined_weight'] * result['confidence']
            weighted_scores.append(result['sentiment_score'] * weight)
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        return sum(weighted_scores) / total_weight
    
    def _calculate_confidence(self, sentiment_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in sentiment analysis"""
        if not sentiment_results:
            return 0.0
        
        confidences = [r['confidence'] for r in sentiment_results]
        return np.mean(confidences)
    
    def _calculate_sentiment_distribution(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of sentiment labels"""
        if not sentiment_results:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
        
        labels = [r['sentiment_label'] for r in sentiment_results]
        total = len(labels)
        
        return {
            'positive': labels.count('positive') / total,
            'negative': labels.count('negative') / total,
            'neutral': labels.count('neutral') / total
        }
    
    def _get_source_credibility(self, source: str) -> float:
        """Get credibility weight for a news source"""
        source_lower = source.lower()
        
        # Tier 1 sources (highest credibility)
        tier1_sources = ['reuters', 'bloomberg', 'wall street journal', 'financial times']
        if any(s in source_lower for s in tier1_sources):
            return 1.0
        
        # Tier 2 sources (high credibility)
        tier2_sources = ['cnbc', 'marketwatch', 'yahoo finance', 'seeking alpha']
        if any(s in source_lower for s in tier2_sources):
            return 0.8
        
        # Tier 3 sources (moderate credibility)
        tier3_sources = ['benzinga', 'the motley fool', 'investopedia']
        if any(s in source_lower for s in tier3_sources):
            return 0.6
        
        # Default credibility for unknown sources
        return 0.4
    
    def _is_credible_source(self, source: str) -> bool:
        """Check if source is considered credible"""
        return self._get_source_credibility(source) >= 0.6
    
    def _get_recency_weight(self, published_at: Optional[datetime]) -> float:
        """Calculate recency weight (more recent = higher weight)"""
        if not published_at:
            return 0.5
        
        hours_ago = (datetime.utcnow() - published_at).total_seconds() / 3600
        
        # Exponential decay: weight = exp(-hours_ago / 12)
        # Articles from 12 hours ago get ~37% weight
        return np.exp(-hours_ago / 12)
    
    async def get_recent_articles(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent articles from all configured sources
        
        Args:
            since: Only get articles after this time
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of article dictionaries
        """
        if since is None:
            since = datetime.now() - timedelta(hours=self.config.lookback_hours)
            
        all_articles = []
        
        try:
            # Get from RSS feeds
            for feed_url in self.config.rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:self.config.max_articles_per_source//len(self.config.rss_feeds)]:
                        # Parse publication date
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        
                        # Skip if too old
                        if pub_date and pub_date < since:
                            continue
                            
                        article = {
                            'title': entry.get('title', ''),
                            'description': entry.get('summary', ''),
                            'url': entry.get('link', ''),
                            'published_at': pub_date,
                            'source': feed.feed.get('title', feed_url),
                            'type': 'rss'
                        }
                        all_articles.append(article)
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch RSS feed {feed_url}", error=str(e))
                    continue
            
            # Get from NewsAPI if configured
            if self.config.newsapi_key:
                try:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'apiKey': self.config.newsapi_key,
                        'q': 'stocks OR trading OR market OR finance',
                        'language': self.config.language,
                        'sortBy': 'publishedAt',
                        'pageSize': self.config.max_articles_per_source,
                        'from': since.isoformat()
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for article_data in data.get('articles', []):
                            # Parse publication date
                            pub_date = None
                            if article_data.get('publishedAt'):
                                try:
                                    pub_date = datetime.fromisoformat(
                                        article_data['publishedAt'].replace('Z', '+00:00')
                                    )
                                except:
                                    pass
                            
                            article = {
                                'title': article_data.get('title', ''),
                                'description': article_data.get('description', ''),
                                'url': article_data.get('url', ''),
                                'published_at': pub_date,
                                'source': article_data.get('source', {}).get('name', 'NewsAPI'),
                                'type': 'newsapi'
                            }
                            all_articles.append(article)
                            
                except Exception as e:
                    logger.warning("Failed to fetch NewsAPI articles", error=str(e))
            
            # Deduplicate and sort
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Sort by publication date (newest first)
            unique_articles.sort(
                key=lambda x: x.get('published_at', datetime.min), 
                reverse=True
            )
            
            logger.debug(f"Retrieved {len(unique_articles)} recent articles")
            return unique_articles[:limit]
            
        except Exception as e:
            logger.error("Failed to get recent articles", error=str(e))
            return []

    def get_trending_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending financial news across all sources"""
        try:
            # Get general financial news
            all_articles = []
            
            # Get from RSS feeds
            for feed_url in self.config.rss_feeds[:5]:  # Limit for performance
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Limit per feed
                        try:
                            if hasattr(entry, 'published_parsed'):
                                pub_time = datetime(*entry.published_parsed[:6])
                            else:
                                continue
                            
                            # Only recent articles
                            if pub_time >= (datetime.utcnow() - timedelta(hours=6)):
                                all_articles.append({
                                    'title': entry.get('title', ''),
                                    'summary': entry.get('summary', ''),
                                    'url': entry.get('link', ''),
                                    'source': feed.feed.get('title', ''),
                                    'published_at': pub_time
                                })
                        except:
                            continue
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error getting trending from {feed_url}", error=str(e))
                    continue
            
            # Sort by recency and return top articles
            all_articles.sort(key=lambda x: x['published_at'], reverse=True)
            
            return all_articles[:limit]
            
        except Exception as e:
            logger.error("Failed to get trending news", error=str(e))
            return []
    
    def __str__(self) -> str:
        return f"NewsAggregator(sources={len(self.config.news_sources)}, feeds={len(self.config.rss_feeds)})"
    
    def __repr__(self) -> str:
        return self.__str__()