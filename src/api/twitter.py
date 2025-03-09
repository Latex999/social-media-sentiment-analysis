"""Twitter API client for accessing Twitter data."""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

import tweepy
from tweepy import Tweet
from dotenv import load_dotenv

from src.utils.config import CACHE_DIR
from src.utils.cache import Cache

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TwitterClient:
    """Client for interacting with the Twitter API.
    
    This class provides methods to fetch tweets based on different criteria
    like hashtags, user mentions, or keyword searches.
    
    Attributes:
        api_key (str): Twitter API key
        api_secret (str): Twitter API secret
        access_token (str): Twitter access token
        access_token_secret (str): Twitter access token secret
        bearer_token (str): Twitter bearer token for v2 API
        client (tweepy.Client): Authenticated Tweepy client
        cache (Cache): Cache instance for storing API responses
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        cache_ttl: int = 3600,  # Cache time-to-live in seconds
    ):
        """Initialize the Twitter client with API credentials.
        
        Args:
            api_key: Twitter API key. If None, will try to get from environment
            api_secret: Twitter API secret. If None, will try to get from environment
            access_token: Twitter access token. If None, will try to get from environment
            access_token_secret: Twitter access token secret. If None, will try to get from environment
            bearer_token: Twitter bearer token. If None, will try to get from environment
            cache_ttl: Cache time-to-live in seconds (default: 3600)
        """
        # Use provided credentials or get from environment
        self.api_key = api_key or os.getenv("TWITTER_API_KEY")
        self.api_secret = api_secret or os.getenv("TWITTER_API_SECRET")
        self.access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = access_token_secret or os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        
        # Validate credentials
        if not self.bearer_token and not (self.api_key and self.api_secret):
            logger.error("Missing Twitter API credentials. Please provide API credentials.")
            raise ValueError("Missing Twitter API credentials. Please set environment variables or provide credentials.")
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret
        )
        
        # Initialize cache
        self.cache = Cache(cache_dir=os.path.join(CACHE_DIR, 'twitter'), ttl=cache_ttl)
        
        logger.info("Twitter client initialized successfully")
    
    def search(
        self,
        query: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for tweets that match a query.
        
        Args:
            query: The search query to use
            limit: Maximum number of tweets to return (default: 100)
            start_time: Start time for tweet search (default: 7 days ago)
            end_time: End time for tweet search (default: now)
            use_cache: Whether to use cached results if available (default: True)
            **kwargs: Additional parameters to pass to the Twitter API
        
        Returns:
            List of tweet objects as dictionaries
        """
        cache_key = f"search_{query}_{limit}_{start_time}_{end_time}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved {len(cached_result)} tweets from cache for query: {query}")
                return cached_result
        
        # Set default time range if not specified
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        logger.info(f"Searching Twitter for: {query}, limit: {limit}")
        
        # Define tweet fields to retrieve
        tweet_fields = [
            'id', 'text', 'created_at', 'public_metrics', 
            'source', 'lang', 'geo', 'entities'
        ]
        
        try:
            # Use tweepy paginator to handle pagination for us
            tweets = []
            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=tweet_fields,
                max_results=min(100, limit),  # Twitter API allows max 100 per request
                limit=max(1, limit // 100)  # Number of pages to request
            ):
                if response.data:
                    tweets.extend(self._process_tweets(response.data))
                
                # Check if we've reached the desired limit
                if len(tweets) >= limit:
                    tweets = tweets[:limit]
                    break
            
            logger.info(f"Retrieved {len(tweets)} tweets for query: {query}")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, tweets)
            
            return tweets
        
        except tweepy.TweepyException as e:
            logger.error(f"Error searching Twitter: {e}")
            raise
    
    def get_user_tweets(
        self, 
        username: str, 
        limit: int = 100,
        use_cache: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get tweets from a specific user.
        
        Args:
            username: Twitter username (without the @ symbol)
            limit: Maximum number of tweets to return (default: 100)
            use_cache: Whether to use cached results if available (default: True)
            **kwargs: Additional parameters to pass to the Twitter API
        
        Returns:
            List of tweet objects as dictionaries
        """
        cache_key = f"user_tweets_{username}_{limit}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved {len(cached_result)} tweets from cache for user: {username}")
                return cached_result
        
        logger.info(f"Retrieving tweets for user: {username}, limit: {limit}")
        
        try:
            # First get the user ID
            user_response = self.client.get_user(username=username)
            if not user_response.data:
                logger.error(f"User not found: {username}")
                return []
            
            user_id = user_response.data.id
            
            # Now get their tweets
            tweet_fields = [
                'id', 'text', 'created_at', 'public_metrics', 
                'source', 'lang', 'geo', 'entities'
            ]
            
            tweets = []
            for response in tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                exclude=['retweets', 'replies'],
                tweet_fields=tweet_fields,
                max_results=min(100, limit),
                limit=max(1, limit // 100)
            ):
                if response.data:
                    tweets.extend(self._process_tweets(response.data))
                
                # Check if we've reached the desired limit
                if len(tweets) >= limit:
                    tweets = tweets[:limit]
                    break
            
            logger.info(f"Retrieved {len(tweets)} tweets for user: {username}")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, tweets)
            
            return tweets
        
        except tweepy.TweepyException as e:
            logger.error(f"Error getting user tweets: {e}")
            raise
    
    def stream_tweets(
        self, 
        query: str, 
        callback: callable,
        max_tweets: Optional[int] = None
    ) -> None:
        """Stream tweets in real-time matching a query.
        
        Args:
            query: Filter query for the stream
            callback: Function to call for each tweet received
            max_tweets: Maximum number of tweets to receive before stopping
        """
        logger.info(f"Starting tweet stream for query: {query}")
        
        class TweetStreamListener(tweepy.StreamingClient):
            def __init__(self, bearer_token, callback_fn, max_count=None):
                super().__init__(bearer_token)
                self.callback_fn = callback_fn
                self.max_count = max_count
                self.tweet_count = 0
            
            def on_tweet(self, tweet):
                processed_tweet = self._process_single_tweet(tweet)
                self.callback_fn(processed_tweet)
                self.tweet_count += 1
                
                if self.max_count and self.tweet_count >= self.max_count:
                    self.disconnect()
            
            def on_error(self, status):
                logger.error(f"Stream error: {status}")
        
        # Create stream listener
        stream_listener = TweetStreamListener(
            self.bearer_token,
            callback,
            max_tweets
        )
        
        # Clean up existing rules
        rules = stream_listener.get_rules()
        if rules.data:
            rule_ids = [rule.id for rule in rules.data]
            stream_listener.delete_rules(rule_ids)
        
        # Add new rule
        stream_listener.add_rules(tweepy.StreamRule(query))
        
        # Start streaming
        stream_listener.filter()
    
    def get_trends(
        self, 
        location_id: int = 1,  # 1 is worldwide
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get current trending topics.
        
        Args:
            location_id: Yahoo Where On Earth ID (default: 1 for worldwide)
            use_cache: Whether to use cached results if available (default: True)
        
        Returns:
            List of trending topics
        """
        cache_key = f"trends_{location_id}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved trends from cache for location: {location_id}")
                return cached_result
        
        logger.info(f"Retrieving trends for location ID: {location_id}")
        
        # For trends, we need to use v1.1 API
        auth = tweepy.OAuth1UserHandler(
            self.api_key, self.api_secret, 
            self.access_token, self.access_token_secret
        )
        api = tweepy.API(auth)
        
        try:
            trends = api.get_place_trends(location_id)
            
            # Process trends
            processed_trends = []
            for trend in trends[0]['trends']:
                processed_trends.append({
                    'name': trend['name'],
                    'url': trend['url'],
                    'query': trend['query'],
                    'tweet_volume': trend['tweet_volume']
                })
            
            logger.info(f"Retrieved {len(processed_trends)} trends")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, processed_trends)
            
            return processed_trends
        
        except tweepy.TweepyException as e:
            logger.error(f"Error getting trends: {e}")
            raise
    
    def _process_tweets(self, tweets: List[Tweet]) -> List[Dict[str, Any]]:
        """Process a list of tweets into a standard format.
        
        Args:
            tweets: List of Tweet objects from Tweepy
        
        Returns:
            List of processed tweet dictionaries
        """
        processed_tweets = []
        
        for tweet in tweets:
            processed_tweets.append(self._process_single_tweet(tweet))
        
        return processed_tweets
    
    def _process_single_tweet(self, tweet: Tweet) -> Dict[str, Any]:
        """Process a single tweet into a standard format.
        
        Args:
            tweet: Tweet object from Tweepy
        
        Returns:
            Processed tweet as dictionary
        """
        # Extract entities if available
        hashtags = []
        mentions = []
        urls = []
        
        if hasattr(tweet, 'entities') and tweet.entities:
            if 'hashtags' in tweet.entities and tweet.entities['hashtags']:
                hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
            
            if 'mentions' in tweet.entities and tweet.entities['mentions']:
                mentions = [mention['username'] for mention in tweet.entities['mentions']]
            
            if 'urls' in tweet.entities and tweet.entities['urls']:
                urls = [url['expanded_url'] for url in tweet.entities['urls']]
        
        # Extract metrics if available
        retweet_count = 0
        like_count = 0
        reply_count = 0
        
        if hasattr(tweet, 'public_metrics') and tweet.public_metrics:
            retweet_count = tweet.public_metrics.get('retweet_count', 0)
            like_count = tweet.public_metrics.get('like_count', 0)
            reply_count = tweet.public_metrics.get('reply_count', 0)
        
        # Create standardized tweet object
        processed_tweet = {
            'id': tweet.id,
            'text': tweet.text,
            'created_at': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') and tweet.created_at else None,
            'source': tweet.source if hasattr(tweet, 'source') else None,
            'lang': tweet.lang if hasattr(tweet, 'lang') else None,
            'metrics': {
                'retweet_count': retweet_count,
                'like_count': like_count,
                'reply_count': reply_count
            },
            'entities': {
                'hashtags': hashtags,
                'mentions': mentions,
                'urls': urls
            },
            'platform': 'twitter'
        }
        
        return processed_tweet