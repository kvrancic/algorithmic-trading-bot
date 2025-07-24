"""
Reddit Sentiment Analyzer for QuantumSentiment Trading Bot

Comprehensive Reddit analysis with:
- Multi-subreddit monitoring
- Emoji signal detection (🚀, 🌙, 💎, 🐻, 📈, 🐂, 📉, 🧸)
- DD post quality analysis
- User credibility scoring
- Option flow extraction
- Mention velocity tracking
"""

import re
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import praw
from transformers import pipeline
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RedditConfig:
    """Configuration for Reddit sentiment analyzer"""
    client_id: str
    client_secret: str
    user_agent: str
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Analysis parameters
    subreddits: List[str] = None
    max_posts_per_subreddit: int = 100
    max_comments_per_post: int = 50
    min_post_score: int = 5
    min_comment_score: int = 2
    lookback_hours: int = 24
    
    # Credibility scoring
    min_account_age_days: int = 30
    min_karma: int = 100
    
    def __post_init__(self):
        if self.subreddits is None:
            self.subreddits = [
                'wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis',
                'ValueInvesting', 'options', 'pennystocks', 'StockMarket',
                'financialindependence', 'trading', 'SecurityAnalysis',
                'InvestmentClub', 'cryptocurrency', 'CryptoCurrency'
            ]


class RedditSentimentAnalyzer:
    """Advanced Reddit sentiment analysis for trading signals"""
    
    def __init__(self, config: RedditConfig):
        """
        Initialize Reddit sentiment analyzer
        
        Args:
            config: Reddit API configuration
        """
        self.config = config
        self.reddit = None
        self.sentiment_analyzer = None
        
        # Emoji mappings for trading signals
        self.bullish_emojis = {
            '🚀': 3,  # Strong bullish
            '🌙': 2,  # Bullish (to the moon)
            '💎': 2,  # Diamond hands (hold)
            '🐂': 2,  # Bull market
            '📈': 1,  # Upward trend
            '💰': 1,  # Money/profits
            '🔥': 1,  # Hot stock
            '⬆️': 1,  # Up arrow
            '🟢': 1,  # Green (profits)
        }
        
        self.bearish_emojis = {
            '🐻': 3,  # Strong bearish
            '📉': 2,  # Downward trend
            '🧸': 2,  # Bear market
            '💥': 1,  # Crash
            '⬇️': 1,  # Down arrow
            '🔴': 1,  # Red (losses)
            '😱': 1,  # Fear
            '💸': 1,  # Money lost
        }
        
        # Keywords for option flow detection
        self.option_keywords = {
            'calls': 1, 'puts': -1, 'strike': 0, 'expiry': 0, 'otm': 0, 'itm': 0,
            'premium': 0, 'theta': 0, 'delta': 0, 'gamma': 0, 'vega': 0,
            'iv': 0, 'implied volatility': 0, 'yolo': 2, 'fd': 1, 'weeklies': 1
        }
        
        # DD quality indicators
        self.dd_quality_indicators = {
            'financial statements': 3, 'balance sheet': 3, 'income statement': 3,
            'cash flow': 3, 'earnings': 2, 'revenue': 2, 'profit margin': 2,
            'debt to equity': 2, 'pe ratio': 1, 'market cap': 1, 'valuation': 2,
            'competitors': 1, 'industry analysis': 2, 'management': 1,
            'insider trading': 2, 'institutional ownership': 1
        }
        
        logger.info("Reddit analyzer initialized", subreddits=len(config.subreddits))
    
    def initialize(self) -> bool:
        """Initialize Reddit API connection and sentiment analyzer"""
        try:
            # Initialize Reddit API
            self.reddit = praw.Reddit(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                user_agent=self.config.user_agent,
                username=self.config.username,
                password=self.config.password
            )
            
            # Test connection
            test_user = self.reddit.user.me()
            logger.info("Reddit API connection established", 
                       user=test_user.name if test_user else "Read-only")
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1  # CPU
            )
            
            logger.info("FinBERT sentiment analyzer initialized")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Reddit analyzer", error=str(e))
            return False
    
    def analyze_symbol(self, symbol: str, hours_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze Reddit sentiment for a specific symbol
        
        Args:
            symbol: Stock/crypto symbol to analyze
            hours_back: Hours to look back (default from config)
            
        Returns:
            Comprehensive sentiment analysis results
        """
        if not self.reddit or not self.sentiment_analyzer:
            raise RuntimeError("Reddit analyzer not initialized")
        
        hours_back = hours_back or self.config.lookback_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        logger.info("Starting Reddit analysis", symbol=symbol, hours_back=hours_back)
        
        try:
            # Collect posts and comments
            posts_data = self._collect_posts(symbol, cutoff_time)
            comments_data = self._collect_comments(symbol, cutoff_time)
            
            # Analyze sentiment
            posts_sentiment = self._analyze_posts_sentiment(posts_data)
            comments_sentiment = self._analyze_comments_sentiment(comments_data)
            
            # Extract signals
            emoji_signals = self._extract_emoji_signals(posts_data + comments_data)
            option_signals = self._extract_option_signals(posts_data + comments_data)
            dd_analysis = self._analyze_dd_quality(posts_data)
            
            # Calculate aggregated metrics
            results = {
                'symbol': symbol.upper(),
                'timestamp': datetime.utcnow(),
                'source': 'reddit',
                
                # Raw data counts
                'total_posts': len(posts_data),
                'total_comments': len(comments_data),
                'total_mentions': len(posts_data) + len(comments_data),
                
                # Sentiment metrics
                'sentiment_score': self._calculate_weighted_sentiment(posts_sentiment, comments_sentiment),
                'confidence': self._calculate_confidence(posts_sentiment, comments_sentiment),
                'sentiment_distribution': self._calculate_sentiment_distribution(posts_sentiment, comments_sentiment),
                
                # Engagement metrics
                'total_engagement': self._calculate_total_engagement(posts_data, comments_data),
                'avg_engagement': self._calculate_avg_engagement(posts_data, comments_data),
                'viral_posts': self._identify_viral_posts(posts_data),
                
                # Signal analysis
                'emoji_signals': emoji_signals,
                'option_signals': option_signals,
                'dd_analysis': dd_analysis,
                
                # User credibility
                'credibility_score': self._calculate_credibility_score(posts_data, comments_data),
                'unique_users': len(set([p['author'] for p in posts_data + comments_data if p['author']])),
                
                # Detailed breakdowns
                'posts_sentiment': posts_sentiment,
                'comments_sentiment': comments_sentiment,
                'raw_data': {
                    'posts': posts_data[:10],  # Sample for debugging
                    'comments': comments_data[:20]
                }
            }
            
            logger.info("Reddit analysis completed", 
                       symbol=symbol,
                       posts=len(posts_data),
                       comments=len(comments_data),
                       sentiment=results['sentiment_score'])
            
            return results
            
        except Exception as e:
            logger.error("Reddit analysis failed", symbol=symbol, error=str(e))
            raise
    
    def _collect_posts(self, symbol: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Collect relevant posts from monitored subreddits"""
        posts_data = []
        
        try:
            for subreddit_name in self.config.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for symbol mentions
                    search_terms = [symbol, f"${symbol}", f"#{symbol}"]
                    
                    for term in search_terms:
                        posts = list(subreddit.search(
                            term,
                            sort='new',
                            time_filter='day',
                            limit=self.config.max_posts_per_subreddit // len(search_terms)
                        ))
                        
                        for post in posts:
                            post_time = datetime.utcfromtimestamp(post.created_utc)
                            
                            if (post_time >= cutoff_time and 
                                post.score >= self.config.min_post_score and
                                not post.stickied):
                                
                                posts_data.append({
                                    'id': post.id,
                                    'subreddit': subreddit_name,
                                    'title': post.title,
                                    'text': post.selftext,
                                    'score': post.score,
                                    'upvote_ratio': post.upvote_ratio,
                                    'num_comments': post.num_comments,
                                    'created_utc': post_time,
                                    'author': str(post.author) if post.author else '[deleted]',
                                    'url': post.url,
                                    'flair': post.link_flair_text,
                                    'is_dd': self._is_dd_post(post.title, post.link_flair_text),
                                    'full_text': f"{post.title} {post.selftext}".lower()
                                })
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error collecting from {subreddit_name}", error=str(e))
                    continue
            
            logger.debug("Posts collected", count=len(posts_data))
            return posts_data
            
        except Exception as e:
            logger.error("Failed to collect posts", error=str(e))
            return []
    
    def _collect_comments(self, symbol: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Collect relevant comments mentioning the symbol"""
        comments_data = []
        
        try:
            for subreddit_name in self.config.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Get recent comments
                    comments = list(subreddit.comments(limit=self.config.max_comments_per_post))
                    
                    for comment in comments:
                        comment_time = datetime.utcfromtimestamp(comment.created_utc)
                        
                        if (comment_time >= cutoff_time and
                            comment.score >= self.config.min_comment_score and
                            self._mentions_symbol(comment.body, symbol)):
                            
                            comments_data.append({
                                'id': comment.id,
                                'subreddit': subreddit_name,
                                'text': comment.body,
                                'score': comment.score,
                                'created_utc': comment_time,
                                'author': str(comment.author) if comment.author else '[deleted]',
                                'parent_id': comment.parent_id,
                                'is_reply': comment.parent_id.startswith('t1_'),
                                'full_text': comment.body.lower()
                            })
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error collecting comments from {subreddit_name}", error=str(e))
                    continue
            
            logger.debug("Comments collected", count=len(comments_data))
            return comments_data
            
        except Exception as e:
            logger.error("Failed to collect comments", error=str(e))
            return []
    
    def _mentions_symbol(self, text: str, symbol: str) -> bool:
        """Check if text mentions the symbol"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Direct mentions
        patterns = [
            f"\\b{symbol_lower}\\b",
            f"\\${symbol_lower}\\b",
            f"#{symbol_lower}\\b",
            f"\\b{symbol_lower}\\s",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
    
    def _is_dd_post(self, title: str, flair: str) -> bool:
        """Determine if post is due diligence"""
        title_lower = title.lower()
        flair_lower = (flair or "").lower()
        
        dd_indicators = ['dd', 'due diligence', 'analysis', 'research', 'deep dive']
        
        for indicator in dd_indicators:
            if indicator in title_lower or indicator in flair_lower:
                return True
                
        return False
    
    def _analyze_posts_sentiment(self, posts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment of posts using FinBERT"""
        sentiment_results = []
        
        for post in posts_data:
            try:
                # Combine title and text for analysis
                text = f"{post['title']} {post['text']}"
                text = text.strip()[:512]  # FinBERT max length
                
                if not text:
                    continue
                
                # Get sentiment
                result = self.sentiment_analyzer(text)[0]
                
                # Convert to numerical score
                sentiment_score = self._convert_sentiment_score(result['label'], result['score'])
                
                sentiment_results.append({
                    'post_id': post['id'],
                    'text': text,
                    'sentiment_label': result['label'],
                    'sentiment_score': sentiment_score,
                    'confidence': result['score'],
                    'engagement_weight': np.log1p(post['score'] + post['num_comments']),
                    'credibility_weight': self._get_user_credibility_weight(post['author']),
                    'is_dd': post['is_dd']
                })
                
            except Exception as e:
                logger.warning("Failed to analyze post sentiment", post_id=post.get('id'), error=str(e))
                continue
        
        return sentiment_results
    
    def _analyze_comments_sentiment(self, comments_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment of comments using FinBERT"""
        sentiment_results = []
        
        for comment in comments_data:
            try:
                text = comment['text'].strip()[:512]  # FinBERT max length
                
                if not text:
                    continue
                
                # Get sentiment
                result = self.sentiment_analyzer(text)[0]
                
                # Convert to numerical score
                sentiment_score = self._convert_sentiment_score(result['label'], result['score'])
                
                sentiment_results.append({
                    'comment_id': comment['id'],
                    'text': text,
                    'sentiment_label': result['label'],
                    'sentiment_score': sentiment_score,
                    'confidence': result['score'],
                    'engagement_weight': np.log1p(comment['score']),
                    'credibility_weight': self._get_user_credibility_weight(comment['author'])
                })
                
            except Exception as e:
                logger.warning("Failed to analyze comment sentiment", comment_id=comment.get('id'), error=str(e))
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
    
    def _extract_emoji_signals(self, all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract emoji-based trading signals"""
        bullish_signals = 0
        bearish_signals = 0
        emoji_counts = {}
        
        for item in all_data:
            text = item.get('full_text', '')
            
            # Count bullish emojis
            for emoji, weight in self.bullish_emojis.items():
                count = text.count(emoji)
                if count > 0:
                    bullish_signals += count * weight
                    emoji_counts[emoji] = emoji_counts.get(emoji, 0) + count
            
            # Count bearish emojis
            for emoji, weight in self.bearish_emojis.items():
                count = text.count(emoji)
                if count > 0:
                    bearish_signals += count * weight
                    emoji_counts[emoji] = emoji_counts.get(emoji, 0) + count
        
        total_signals = bullish_signals + bearish_signals
        
        return {
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'net_signal': (bullish_signals - bearish_signals) / max(total_signals, 1),
            'emoji_counts': emoji_counts,
            'rocket_mentions': emoji_counts.get('🚀', 0),  # Special tracking
            'diamond_hands': emoji_counts.get('💎', 0)
        }
    
    def _extract_option_signals(self, all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract options flow signals from text"""
        calls_mentions = 0
        puts_mentions = 0
        option_keywords_found = {}
        yolo_count = 0
        
        for item in all_data:
            text = item.get('full_text', '')
            
            # Count option mentions
            calls_mentions += len(re.findall(r'\bcalls?\b', text, re.IGNORECASE))
            puts_mentions += len(re.findall(r'\bputs?\b', text, re.IGNORECASE))
            yolo_count += len(re.findall(r'\byolo\b', text, re.IGNORECASE))
            
            # Track other option keywords
            for keyword, sentiment in self.option_keywords.items():
                count = len(re.findall(rf'\\b{keyword}\\b', text, re.IGNORECASE))
                if count > 0:
                    option_keywords_found[keyword] = option_keywords_found.get(keyword, 0) + count
        
        total_option_mentions = calls_mentions + puts_mentions
        
        return {
            'calls_mentions': calls_mentions,
            'puts_mentions': puts_mentions,
            'call_put_ratio': calls_mentions / max(puts_mentions, 1),
            'total_option_activity': total_option_mentions,
            'yolo_sentiment': yolo_count,
            'option_keywords': option_keywords_found,
            'bullish_options_bias': (calls_mentions - puts_mentions) / max(total_option_mentions, 1)
        }
    
    def _analyze_dd_quality(self, posts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality of DD posts"""
        dd_posts = [p for p in posts_data if p.get('is_dd', False)]
        
        if not dd_posts:
            return {
                'dd_count': 0,
                'avg_dd_quality': 0,
                'quality_indicators': {},
                'high_quality_dd': 0
            }
        
        quality_scores = []
        all_indicators = {}
        
        for post in dd_posts:
            text = f"{post['title']} {post['text']}".lower()
            quality_score = 0
            
            # Check for quality indicators
            for indicator, weight in self.dd_quality_indicators.items():
                count = len(re.findall(rf'\\b{indicator}\\b', text, re.IGNORECASE))
                if count > 0:
                    quality_score += count * weight
                    all_indicators[indicator] = all_indicators.get(indicator, 0) + count
            
            # Adjust by engagement
            engagement_factor = np.log1p(post['score'] + post['num_comments']) / 5
            quality_score *= engagement_factor
            
            quality_scores.append(quality_score)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        high_quality_dd = len([q for q in quality_scores if q > 10])  # Threshold for high quality
        
        return {
            'dd_count': len(dd_posts),
            'avg_dd_quality': avg_quality,
            'quality_indicators': all_indicators,
            'high_quality_dd': high_quality_dd,
            'dd_quality_distribution': {
                'high': high_quality_dd,
                'medium': len([q for q in quality_scores if 5 < q <= 10]),
                'low': len([q for q in quality_scores if q <= 5])
            }
        }
    
    def _calculate_weighted_sentiment(self, posts_sentiment: List[Dict], comments_sentiment: List[Dict]) -> float:
        """Calculate weighted average sentiment"""
        if not posts_sentiment and not comments_sentiment:
            return 0.0
        
        weighted_scores = []
        
        # Weight posts more heavily than comments
        for post_sent in posts_sentiment:
            weight = (post_sent['engagement_weight'] * 
                     post_sent['credibility_weight'] * 
                     post_sent['confidence'] * 
                     (2 if post_sent['is_dd'] else 1))  # DD posts weighted higher
            weighted_scores.append(post_sent['sentiment_score'] * weight)
        
        for comment_sent in comments_sentiment:
            weight = (comment_sent['engagement_weight'] * 
                     comment_sent['credibility_weight'] * 
                     comment_sent['confidence'] * 0.5)  # Comments weighted less
            weighted_scores.append(comment_sent['sentiment_score'] * weight)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_confidence(self, posts_sentiment: List[Dict], comments_sentiment: List[Dict]) -> float:
        """Calculate overall confidence in sentiment analysis"""
        all_confidences = []
        
        for sent in posts_sentiment + comments_sentiment:
            all_confidences.append(sent['confidence'])
        
        return np.mean(all_confidences) if all_confidences else 0.0
    
    def _calculate_sentiment_distribution(self, posts_sentiment: List[Dict], comments_sentiment: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of sentiment labels"""
        all_labels = []
        
        for sent in posts_sentiment + comments_sentiment:
            all_labels.append(sent['sentiment_label'])
        
        if not all_labels:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
        
        total = len(all_labels)
        return {
            'positive': all_labels.count('positive') / total,
            'negative': all_labels.count('negative') / total,
            'neutral': all_labels.count('neutral') / total
        }
    
    def _calculate_total_engagement(self, posts_data: List[Dict], comments_data: List[Dict]) -> int:
        """Calculate total engagement (upvotes + comments)"""
        total = 0
        
        for post in posts_data:
            total += post['score'] + post['num_comments']
        
        for comment in comments_data:
            total += comment['score']
        
        return total
    
    def _calculate_avg_engagement(self, posts_data: List[Dict], comments_data: List[Dict]) -> float:
        """Calculate average engagement per mention"""
        total_items = len(posts_data) + len(comments_data)
        if total_items == 0:
            return 0.0
        
        total_engagement = self._calculate_total_engagement(posts_data, comments_data)
        return total_engagement / total_items
    
    def _identify_viral_posts(self, posts_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify viral posts based on engagement"""
        if not posts_data:
            return []
        
        # Calculate engagement threshold (top 10% or score > 1000)
        scores = [p['score'] for p in posts_data]
        threshold = max(np.percentile(scores, 90), 1000)
        
        viral_posts = []
        for post in posts_data:
            if post['score'] >= threshold:
                viral_posts.append({
                    'id': post['id'],
                    'title': post['title'],
                    'score': post['score'],
                    'comments': post['num_comments'],
                    'subreddit': post['subreddit'],
                    'upvote_ratio': post['upvote_ratio']
                })
        
        return viral_posts
    
    def _calculate_credibility_score(self, posts_data: List[Dict], comments_data: List[Dict]) -> float:
        """Calculate average user credibility score"""
        authors = set()
        
        for item in posts_data + comments_data:
            if item['author'] and item['author'] != '[deleted]':
                authors.add(item['author'])
        
        if not authors:
            return 0.0
        
        credibility_scores = []
        for author in authors:
            credibility_scores.append(self._get_user_credibility_weight(author))
        
        return np.mean(credibility_scores) if credibility_scores else 0.0
    
    def _get_user_credibility_weight(self, username: str) -> float:
        """Get credibility weight for a user (simplified version)"""
        if not username or username == '[deleted]':
            return 0.1
        
        try:
            # In a full implementation, you'd cache this data
            # For now, return a default moderate credibility
            return 0.5
            
        except Exception:
            return 0.3  # Default for unknown users
    
    def get_trending_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending symbols across all monitored subreddits"""
        try:
            symbol_mentions = {}
            
            for subreddit_name in self.config.subreddits[:5]:  # Limit for performance
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    posts = list(subreddit.hot(limit=50))
                    
                    for post in posts:
                        text = f"{post.title} {post.selftext}".upper()
                        
                        # Extract potential symbols
                        symbols = re.findall(r'\\$([A-Z]{1,5})\\b', text)
                        symbols += re.findall(r'\\b([A-Z]{2,5})\\b', text)
                        
                        for symbol in symbols:
                            if len(symbol) >= 2 and symbol.isalpha():
                                if symbol not in symbol_mentions:
                                    symbol_mentions[symbol] = {
                                        'count': 0,
                                        'total_score': 0,
                                        'subreddits': set()
                                    }
                                
                                symbol_mentions[symbol]['count'] += 1
                                symbol_mentions[symbol]['total_score'] += post.score
                                symbol_mentions[symbol]['subreddits'].add(subreddit_name)
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error getting trending from {subreddit_name}", error=str(e))
                    continue
            
            # Sort by mention count and engagement
            trending = []
            for symbol, data in symbol_mentions.items():
                if data['count'] >= 3:  # Minimum mentions threshold
                    trending.append({
                        'symbol': symbol,
                        'mention_count': data['count'],
                        'total_engagement': data['total_score'],
                        'avg_engagement': data['total_score'] / data['count'],
                        'subreddit_diversity': len(data['subreddits']),
                        'trend_score': data['count'] * len(data['subreddits']) * np.log1p(data['total_score'])
                    })
            
            # Sort by trend score
            trending.sort(key=lambda x: x['trend_score'], reverse=True)
            
            return trending[:limit]
            
        except Exception as e:
            logger.error("Failed to get trending symbols", error=str(e))
            return []
    
    def __str__(self) -> str:
        return f"RedditSentimentAnalyzer(subreddits={len(self.config.subreddits)})"
    
    def __repr__(self) -> str:
        return self.__str__()