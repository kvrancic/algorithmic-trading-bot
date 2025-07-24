"""
Reddit API Client for QuantumSentiment Trading Bot

Handles all interactions with Reddit API for sentiment analysis:
- Fetching posts from financial subreddits
- Analyzing sentiment and engagement metrics
- Extracting stock/crypto mentions
- Tracking user credibility and DD quality
"""

import os
import re
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import pandas as pd
import structlog
import praw
from praw.models import Submission, Comment

logger = structlog.get_logger(__name__)


class RedditClient:
    """Reddit API client optimized for financial sentiment analysis"""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        subreddits: Optional[List[str]] = None
    ):
        """
        Initialize Reddit client
        
        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
            user_agent: User agent string
            subreddits: List of subreddits to monitor
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT')
        
        if not all([self.client_id, self.client_secret, self.user_agent]):
            raise ValueError("Reddit API credentials not provided")
        
        # Initialize PRAW
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        # Set read-only mode
        self.reddit.read_only = True
        
        # Default subreddits for financial sentiment
        self.subreddits = subreddits or [
            'wallstreetbets',
            'stocks', 
            'investing',
            'SecurityAnalysis',
            'ValueInvesting',
            'cryptocurrency',
            'CryptoCurrency',
            'Bitcoin',
            'ethereum'
        ]
        
        # Compiled regex patterns for efficiency
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        self.crypto_pattern = re.compile(r'\b(BTC|ETH|BNB|ADA|SOL|DOGE|MATIC|DOT|AVAX|LINK)\b', re.IGNORECASE)
        
        # Bullish/bearish emoji patterns
        self.bullish_emojis = {'🚀', '🌙', '💎', '🦍', '📈', '💰', '🔥', '⬆️', '🟢'}
        self.bearish_emojis = {'🐻', '📉', '💀', '🩸', '⬇️', '🔴', '🌈'}
        
        logger.info("Reddit client initialized", subreddits=len(self.subreddits))
    
    # === POST FETCHING ===
    
    def get_hot_posts(
        self,
        subreddit: str,
        limit: int = 100,
        min_score: int = 10
    ) -> List[Dict[str, Any]]:
        """Get hot posts from subreddit"""
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []
            
            for submission in sub.hot(limit=limit):
                if submission.score >= min_score:
                    post_data = self._extract_post_data(submission)
                    posts.append(post_data)
            
            logger.debug("Retrieved hot posts", 
                        subreddit=subreddit, 
                        count=len(posts))
            return posts
            
        except Exception as e:
            logger.error("Failed to get hot posts", 
                        subreddit=subreddit, error=str(e))
            return []
    
    def get_new_posts(
        self,
        subreddit: str,
        limit: int = 100,
        min_score: int = 1
    ) -> List[Dict[str, Any]]:
        """Get new posts from subreddit"""
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []
            
            for submission in sub.new(limit=limit):
                if submission.score >= min_score:
                    post_data = self._extract_post_data(submission)
                    posts.append(post_data)
            
            logger.debug("Retrieved new posts", 
                        subreddit=subreddit, 
                        count=len(posts))
            return posts
            
        except Exception as e:
            logger.error("Failed to get new posts", 
                        subreddit=subreddit, error=str(e))
            return []
    
    def get_posts_by_flair(
        self,
        subreddit: str,
        flair: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get posts filtered by flair (e.g., 'DD', 'Discussion')"""
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []
            
            # Search for posts with specific flair
            for submission in sub.search(f'flair:"{flair}"', limit=limit, sort='new'):
                post_data = self._extract_post_data(submission)
                posts.append(post_data)
            
            logger.debug("Retrieved posts by flair", 
                        subreddit=subreddit, 
                        flair=flair,
                        count=len(posts))
            return posts
            
        except Exception as e:
            logger.error("Failed to get posts by flair", 
                        subreddit=subreddit, 
                        flair=flair,
                        error=str(e))
            return []
    
    def search_posts(
        self,
        query: str,
        subreddit: str = 'all',
        sort: str = 'relevance',
        time_filter: str = 'day',
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search posts by query"""
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []
            
            for submission in sub.search(
                query, 
                sort=sort, 
                time_filter=time_filter, 
                limit=limit
            ):
                post_data = self._extract_post_data(submission)
                posts.append(post_data)
            
            logger.debug("Retrieved search results", 
                        query=query,
                        subreddit=subreddit,
                        count=len(posts))
            return posts
            
        except Exception as e:
            logger.error("Failed to search posts", 
                        query=query,
                        subreddit=subreddit,
                        error=str(e))
            return []
    
    # === SENTIMENT ANALYSIS ===
    
    def analyze_ticker_sentiment(
        self,
        ticker: str,
        hours_back: int = 24,
        min_mentions: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for specific ticker across subreddits
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL', 'GME')
            hours_back: Hours to look back
            min_mentions: Minimum mentions required
            
        Returns:
            Sentiment analysis results
        """
        all_posts = []
        all_comments = []
        
        # Collect posts mentioning ticker from all subreddits
        for subreddit in self.subreddits:
            try:
                # Search for ticker mentions
                query = f'${ticker} OR {ticker}'
                posts = self.search_posts(
                    query=query,
                    subreddit=subreddit,
                    time_filter='day',
                    limit=50
                )
                
                # Filter by time
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                recent_posts = [
                    p for p in posts 
                    if datetime.fromtimestamp(p['created_utc']) > cutoff_time
                ]
                
                all_posts.extend(recent_posts)
                
                # Get comments for high-engagement posts
                for post in recent_posts[:10]:  # Limit to top 10 posts per subreddit
                    comments = self._get_post_comments(post['id'], limit=20)
                    all_comments.extend(comments)
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning("Failed to analyze subreddit", 
                             subreddit=subreddit, error=str(e))
                continue
        
        if len(all_posts) < min_mentions:
            return {
                'ticker': ticker,
                'mention_count': len(all_posts),
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'message': f'Insufficient mentions ({len(all_posts)} < {min_mentions})'
            }
        
        # Analyze sentiment
        sentiment_data = self._analyze_posts_sentiment(all_posts + all_comments, ticker)
        
        logger.info("Ticker sentiment analyzed", 
                   ticker=ticker,
                   mentions=len(all_posts),
                   sentiment=sentiment_data['sentiment_score'])
        
        return sentiment_data
    
    def get_trending_tickers(
        self,
        subreddit: str = 'wallstreetbets',
        limit: int = 100,
        min_mentions: int = 3
    ) -> List[Dict[str, Any]]:
        """Get trending stock tickers from subreddit"""
        try:
            posts = self.get_hot_posts(subreddit, limit=limit)
            ticker_counts = Counter()
            ticker_scores = defaultdict(list)
            
            for post in posts:
                tickers = self._extract_tickers(post['title'] + ' ' + post.get('selftext', ''))
                for ticker in tickers:
                    ticker_counts[ticker] += 1
                    ticker_scores[ticker].append(post['score'])
            
            # Filter and rank tickers
            trending = []
            for ticker, count in ticker_counts.most_common():
                if count >= min_mentions:
                    avg_score = sum(ticker_scores[ticker]) / len(ticker_scores[ticker])
                    trending.append({
                        'ticker': ticker,
                        'mentions': count,
                        'avg_score': round(avg_score, 2),
                        'total_score': sum(ticker_scores[ticker]),
                        'velocity': self._calculate_mention_velocity(ticker, subreddit)
                    })
            
            logger.debug("Found trending tickers", 
                        subreddit=subreddit,
                        count=len(trending))
            return trending
            
        except Exception as e:
            logger.error("Failed to get trending tickers", 
                        subreddit=subreddit, error=str(e))
            return []
    
    # === ADVANCED FEATURES ===
    
    def analyze_dd_quality(self, post_id: str) -> Dict[str, Any]:
        """Analyze quality of Due Diligence posts"""
        try:
            submission = self.reddit.submission(id=post_id)
            
            # Quality indicators
            word_count = len(submission.selftext.split())
            has_numbers = bool(re.search(r'\d+', submission.selftext))
            has_links = bool(re.search(r'http[s]?://', submission.selftext))
            award_count = sum(submission.all_awardings, key=lambda x: x['count'])
            
            # DD-specific patterns
            dd_keywords = ['revenue', 'earnings', 'pe ratio', 'debt', 'cash flow', 'valuation']
            keyword_count = sum(1 for keyword in dd_keywords if keyword in submission.selftext.lower())
            
            quality_score = (
                min(word_count / 1000, 1.0) * 0.3 +  # Length
                (1 if has_numbers else 0) * 0.2 +      # Has data
                (1 if has_links else 0) * 0.1 +        # Has sources
                min(award_count / 10, 1.0) * 0.2 +     # Community recognition
                min(keyword_count / len(dd_keywords), 1.0) * 0.2  # DD keywords
            )
            
            return {
                'post_id': post_id,
                'quality_score': round(quality_score, 3),
                'word_count': word_count,
                'has_data': has_numbers,
                'has_sources': has_links,
                'awards': award_count,
                'dd_keywords': keyword_count
            }
            
        except Exception as e:
            logger.error("Failed to analyze DD quality", post_id=post_id, error=str(e))
            return {'post_id': post_id, 'quality_score': 0.0}
    
    def get_user_credibility(self, username: str) -> Dict[str, Any]:
        """Analyze user credibility for sentiment weighting"""
        try:
            user = self.reddit.redditor(username)
            
            # Get recent submissions and comments
            recent_posts = list(user.submissions.new(limit=10))
            recent_comments = list(user.comments.new(limit=10))
            
            avg_post_score = sum(p.score for p in recent_posts) / max(len(recent_posts), 1)
            avg_comment_score = sum(c.score for c in recent_comments) / max(len(recent_comments), 1)
            
            # Account age and karma
            account_age_days = (datetime.now().timestamp() - user.created_utc) / 86400
            
            credibility_score = (
                min(user.comment_karma / 10000, 1.0) * 0.3 +
                min(user.link_karma / 5000, 1.0) * 0.2 +
                min(account_age_days / 365, 1.0) * 0.2 +
                min(avg_post_score / 100, 1.0) * 0.15 +
                min(avg_comment_score / 10, 1.0) * 0.15
            )
            
            return {
                'username': username,
                'credibility_score': round(credibility_score, 3),
                'comment_karma': user.comment_karma,
                'link_karma': user.link_karma,
                'account_age_days': int(account_age_days),
                'avg_post_score': round(avg_post_score, 2),
                'avg_comment_score': round(avg_comment_score, 2)
            }
            
        except Exception as e:
            logger.error("Failed to analyze user credibility", username=username, error=str(e))
            return {'username': username, 'credibility_score': 0.5}
    
    # === HELPER METHODS ===
    
    def _extract_post_data(self, submission: Submission) -> Dict[str, Any]:
        """Extract relevant data from Reddit submission"""
        return {
            'id': submission.id,
            'title': submission.title,
            'selftext': submission.selftext,
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'author': str(submission.author) if submission.author else '[deleted]',
            'subreddit': str(submission.subreddit),
            'flair': submission.link_flair_text,
            'awards': sum(award['count'] for award in submission.all_awardings),
            'url': submission.url,
            'tickers': self._extract_tickers(submission.title + ' ' + submission.selftext),
            'bullish_emojis': self._count_emojis(submission.title + ' ' + submission.selftext, self.bullish_emojis),
            'bearish_emojis': self._count_emojis(submission.title + ' ' + submission.selftext, self.bearish_emojis)
        }
    
    def _get_post_comments(self, post_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get comments for a specific post"""
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            comments = []
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comments.append({
                        'id': comment.id,
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'post_id': post_id,
                        'tickers': self._extract_tickers(comment.body),
                        'bullish_emojis': self._count_emojis(comment.body, self.bullish_emojis),
                        'bearish_emojis': self._count_emojis(comment.body, self.bearish_emojis)
                    })
            
            return comments
            
        except Exception as e:
            logger.error("Failed to get comments", post_id=post_id, error=str(e))
            return []
    
    def _extract_tickers(self, text: str) -> Set[str]:
        """Extract stock tickers from text"""
        tickers = set()
        
        # Find $TICKER patterns
        dollar_tickers = self.ticker_pattern.findall(text.upper())
        tickers.update(dollar_tickers)
        
        # Find crypto patterns
        crypto_matches = self.crypto_pattern.findall(text)
        tickers.update([c.upper() for c in crypto_matches])
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        tickers = tickers - false_positives
        
        return tickers
    
    def _count_emojis(self, text: str, emoji_set: Set[str]) -> int:
        """Count specific emojis in text"""
        return sum(1 for char in text if char in emoji_set)
    
    def _analyze_posts_sentiment(self, posts: List[Dict], ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from posts/comments"""
        if not posts:
            return {
                'ticker': ticker,
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'mention_count': 0,
                'bullish_signals': 0,
                'bearish_signals': 0
            }
        
        total_score = 0
        total_weight = 0
        bullish_signals = 0
        bearish_signals = 0
        
        for post in posts:
            # Weight by engagement
            weight = 1 + (post.get('score', 0) / 100)
            
            # Emoji sentiment
            bullish_count = post.get('bullish_emojis', 0)
            bearish_count = post.get('bearish_emojis', 0)
            
            bullish_signals += bullish_count
            bearish_signals += bearish_count
            
            # Simple sentiment based on emojis and score
            emoji_sentiment = (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
            score_sentiment = 1 if post.get('score', 0) > 0 else -1
            
            post_sentiment = (emoji_sentiment * 0.7 + score_sentiment * 0.3)
            
            total_score += post_sentiment * weight
            total_weight += weight
        
        # Calculate final sentiment
        sentiment_score = total_score / max(total_weight, 1)
        confidence = min(len(posts) / 20, 1.0)  # Higher confidence with more data
        
        return {
            'ticker': ticker,
            'sentiment_score': round(sentiment_score, 3),
            'confidence': round(confidence, 3),
            'mention_count': len(posts),
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'avg_score': round(sum(p.get('score', 0) for p in posts) / len(posts), 2)
        }
    
    def _calculate_mention_velocity(self, ticker: str, subreddit: str) -> float:
        """Calculate rate of mention increase (mentions per hour)"""
        try:
            # Get mentions from last 24 hours vs last 12 hours
            recent_posts = self.search_posts(
                query=f'${ticker}',
                subreddit=subreddit,
                time_filter='day',
                limit=50
            )
            
            now = datetime.now()
            last_12h = [p for p in recent_posts if (now - datetime.fromtimestamp(p['created_utc'])).total_seconds() < 12*3600]
            last_24h = recent_posts
            
            velocity = len(last_12h) * 2 - len(last_24h)  # Extrapolate 12h to 24h
            return max(velocity, 0)  # Don't return negative velocity
            
        except Exception:
            return 0.0
    
    def __str__(self) -> str:
        return f"RedditClient(subreddits={len(self.subreddits)})"
    
    def __repr__(self) -> str:
        return self.__str__()