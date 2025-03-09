"""Reddit API client for accessing Reddit data."""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

import praw
from praw.models import Submission, Comment
from dotenv import load_dotenv

from src.utils.config import CACHE_DIR
from src.utils.cache import Cache

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RedditClient:
    """Client for interacting with the Reddit API.
    
    This class provides methods to fetch posts and comments from Reddit
    based on different criteria like subreddit, search queries, or post IDs.
    
    Attributes:
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        user_agent (str): User agent string for Reddit API
        reddit (praw.Reddit): Authenticated PRAW Reddit instance
        cache (Cache): Cache instance for storing API responses
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_ttl: int = 3600,  # Cache time-to-live in seconds
    ):
        """Initialize the Reddit client with API credentials.
        
        Args:
            client_id: Reddit API client ID. If None, will try to get from environment
            client_secret: Reddit API client secret. If None, will try to get from environment
            user_agent: User agent for Reddit API. If None, will use default
            username: Reddit username for authentication (optional)
            password: Reddit password for authentication (optional)
            cache_ttl: Cache time-to-live in seconds (default: 3600)
        """
        # Use provided credentials or get from environment
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.username = username or os.getenv("REDDIT_USERNAME")
        self.password = password or os.getenv("REDDIT_PASSWORD")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", 
                                                 "social-media-sentiment-analysis:v1.0.0 (by /u/YourUsername)")
        
        # Validate credentials
        if not self.client_id or not self.client_secret:
            logger.error("Missing Reddit API credentials. Please provide API credentials.")
            raise ValueError("Missing Reddit API credentials. Please set environment variables or provide credentials.")
        
        # Initialize PRAW Reddit instance
        praw_kwargs = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "user_agent": self.user_agent,
        }
        
        # Add username and password if provided
        if self.username and self.password:
            praw_kwargs["username"] = self.username
            praw_kwargs["password"] = self.password
        
        self.reddit = praw.Reddit(**praw_kwargs)
        
        # Initialize cache
        self.cache = Cache(cache_dir=os.path.join(CACHE_DIR, 'reddit'), ttl=cache_ttl)
        
        logger.info("Reddit client initialized successfully")
    
    def search(
        self,
        query: str,
        subreddit: Optional[str] = None,
        limit: int = 100,
        time_filter: str = "month",  # Options: hour, day, week, month, year, all
        sort: str = "relevance",     # Options: relevance, hot, new, top, comments
        include_comments: bool = True,
        comments_limit: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for submissions matching a query.
        
        Args:
            query: Search query to use
            subreddit: Specific subreddit to search in (None for all subreddits)
            limit: Maximum number of submissions to return (default: 100)
            time_filter: Time filter to apply (default: month)
            sort: Sort method for results (default: relevance)
            include_comments: Whether to include comments in results (default: True)
            comments_limit: Maximum number of comments to include per submission (default: 10)
            use_cache: Whether to use cached results if available (default: True)
        
        Returns:
            List of submission objects as dictionaries
        """
        cache_key = f"search_{query}_{subreddit}_{limit}_{time_filter}_{sort}_{include_comments}_{comments_limit}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved {len(cached_result)} submissions from cache for query: {query}")
                return cached_result
        
        logger.info(f"Searching Reddit for: {query} in subreddit: {subreddit or 'all'}, limit: {limit}")
        
        try:
            # Initialize the search
            if subreddit:
                search_results = self.reddit.subreddit(subreddit).search(
                    query=query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=limit
                )
            else:
                search_results = self.reddit.subreddit("all").search(
                    query=query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=limit
                )
            
            # Process the submissions
            submissions = []
            for submission in search_results:
                processed_submission = self._process_submission(
                    submission, 
                    include_comments=include_comments,
                    comments_limit=comments_limit
                )
                submissions.append(processed_submission)
            
            logger.info(f"Retrieved {len(submissions)} submissions for query: {query}")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, submissions)
            
            return submissions
        
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            raise
    
    def get_subreddit_posts(
        self,
        subreddit: str,
        category: str = "hot",  # Options: hot, new, top, rising, controversial
        limit: int = 100,
        time_filter: str = "month",  # Only used for 'top' and 'controversial'
        include_comments: bool = True,
        comments_limit: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get posts from a specific subreddit.
        
        Args:
            subreddit: Subreddit name to fetch from
            category: Category of posts to fetch (default: hot)
            limit: Maximum number of posts to return (default: 100)
            time_filter: Time filter for top/controversial posts (default: month)
            include_comments: Whether to include comments in results (default: True)
            comments_limit: Maximum number of comments to include per post (default: 10)
            use_cache: Whether to use cached results if available (default: True)
        
        Returns:
            List of submission objects as dictionaries
        """
        cache_key = f"subreddit_{subreddit}_{category}_{limit}_{time_filter}_{include_comments}_{comments_limit}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved {len(cached_result)} posts from cache for subreddit: {subreddit}")
                return cached_result
        
        logger.info(f"Retrieving {category} posts from subreddit: {subreddit}, limit: {limit}")
        
        try:
            # Get the subreddit
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Get the posts based on category
            if category == "hot":
                posts = subreddit_obj.hot(limit=limit)
            elif category == "new":
                posts = subreddit_obj.new(limit=limit)
            elif category == "top":
                posts = subreddit_obj.top(time_filter=time_filter, limit=limit)
            elif category == "rising":
                posts = subreddit_obj.rising(limit=limit)
            elif category == "controversial":
                posts = subreddit_obj.controversial(time_filter=time_filter, limit=limit)
            else:
                logger.error(f"Invalid category: {category}")
                raise ValueError(f"Invalid category: {category}")
            
            # Process the submissions
            submissions = []
            for submission in posts:
                processed_submission = self._process_submission(
                    submission, 
                    include_comments=include_comments,
                    comments_limit=comments_limit
                )
                submissions.append(processed_submission)
            
            logger.info(f"Retrieved {len(submissions)} posts from subreddit: {subreddit}")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, submissions)
            
            return submissions
        
        except Exception as e:
            logger.error(f"Error fetching subreddit posts: {e}")
            raise
    
    def get_post_comments(
        self,
        post_id: str,
        limit: int = 100,
        sort: str = "top",  # Options: top, new, controversial, old, qa
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get comments from a specific post.
        
        Args:
            post_id: Reddit post ID to fetch comments from
            limit: Maximum number of comments to return (default: 100)
            sort: Sort method for comments (default: top)
            use_cache: Whether to use cached results if available (default: True)
        
        Returns:
            List of comment objects as dictionaries
        """
        cache_key = f"comments_{post_id}_{limit}_{sort}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved {len(cached_result)} comments from cache for post: {post_id}")
                return cached_result
        
        logger.info(f"Retrieving comments for post: {post_id}, limit: {limit}")
        
        try:
            # Get the submission
            submission = self.reddit.submission(id=post_id)
            
            # Set the comment sort
            submission.comment_sort = sort
            
            # Replace MoreComments objects with actual comments
            submission.comments.replace_more(limit=None)
            
            # Get flat list of comments
            comments = []
            for i, comment in enumerate(submission.comments.list()):
                if i >= limit:
                    break
                comments.append(self._process_comment(comment))
            
            logger.info(f"Retrieved {len(comments)} comments for post: {post_id}")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, comments)
            
            return comments
        
        except Exception as e:
            logger.error(f"Error fetching post comments: {e}")
            raise
    
    def get_trending_subreddits(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get a list of trending subreddits.
        
        Args:
            use_cache: Whether to use cached results if available (default: True)
        
        Returns:
            List of trending subreddit objects
        """
        cache_key = "trending_subreddits"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Retrieved trending subreddits from cache")
                return cached_result
        
        logger.info("Retrieving trending subreddits")
        
        try:
            # Get trending subreddits
            trending = []
            for subreddit in self.reddit.subreddits.popular(limit=25):
                trending.append({
                    'name': subreddit.display_name,
                    'title': subreddit.title,
                    'description': subreddit.public_description,
                    'subscribers': subreddit.subscribers,
                    'url': f"https://www.reddit.com/r/{subreddit.display_name}/",
                    'created_utc': datetime.fromtimestamp(subreddit.created_utc).isoformat(),
                    'nsfw': subreddit.over18
                })
            
            logger.info(f"Retrieved {len(trending)} trending subreddits")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, trending)
            
            return trending
        
        except Exception as e:
            logger.error(f"Error fetching trending subreddits: {e}")
            raise
    
    def get_user_posts(
        self,
        username: str,
        limit: int = 100,
        include_comments: bool = False,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get submissions from a specific user.
        
        Args:
            username: Reddit username to fetch posts from
            limit: Maximum number of posts to return (default: 100)
            include_comments: Whether to include the user's comments (default: False)
            use_cache: Whether to use cached results if available (default: True)
        
        Returns:
            List of user submissions as dictionaries
        """
        cache_key = f"user_posts_{username}_{limit}_{include_comments}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Retrieved {len(cached_result)} posts from cache for user: {username}")
                return cached_result
        
        logger.info(f"Retrieving posts for user: {username}, limit: {limit}")
        
        try:
            # Get the user
            user = self.reddit.redditor(username)
            
            # Get posts and possibly comments
            submissions = []
            
            # Get submissions
            post_count = 0
            for submission in user.submissions.new(limit=limit):
                processed_submission = self._process_submission(submission, include_comments=False)
                submissions.append(processed_submission)
                post_count += 1
                if post_count >= limit:
                    break
            
            # Get comments if requested
            if include_comments:
                comment_count = 0
                comment_limit = limit - post_count if post_count < limit else 0
                
                if comment_limit > 0:
                    for comment in user.comments.new(limit=comment_limit):
                        processed_comment = self._process_comment(comment)
                        # Add extra info to link it to the parent post
                        processed_comment['parent_id'] = comment.parent_id
                        processed_comment['link_id'] = comment.link_id
                        processed_comment['is_comment'] = True
                        submissions.append(processed_comment)
                        comment_count += 1
                        if comment_count >= comment_limit:
                            break
            
            logger.info(f"Retrieved {len(submissions)} items for user: {username}")
            
            # Store in cache
            if use_cache:
                self.cache.set(cache_key, submissions)
            
            return submissions
        
        except Exception as e:
            logger.error(f"Error fetching user posts: {e}")
            raise
    
    def _process_submission(
        self, 
        submission: Submission, 
        include_comments: bool = True,
        comments_limit: int = 10
    ) -> Dict[str, Any]:
        """Process a submission into a standard format.
        
        Args:
            submission: PRAW Submission object
            include_comments: Whether to include comments (default: True)
            comments_limit: Maximum number of comments to include (default: 10)
        
        Returns:
            Processed submission as dictionary
        """
        # Create standardized submission object
        processed_submission = {
            'id': submission.id,
            'title': submission.title,
            'text': submission.selftext,
            'url': submission.url,
            'permalink': f"https://www.reddit.com{submission.permalink}",
            'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
            'author': submission.author.name if submission.author else '[deleted]',
            'subreddit': submission.subreddit.display_name,
            'upvote_ratio': submission.upvote_ratio,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'is_self': submission.is_self,
            'is_video': submission.is_video,
            'is_original_content': submission.is_original_content,
            'over_18': submission.over_18,
            'spoiler': submission.spoiler,
            'distinguished': submission.distinguished,
            'flair': {
                'text': submission.link_flair_text,
                'css_class': submission.link_flair_css_class
            },
            'platform': 'reddit',
            'type': 'submission'
        }
        
        # Add comments if requested
        if include_comments and comments_limit > 0:
            # Limit the comment depth
            submission.comment_sort = 'top'
            submission.comments.replace_more(limit=0)
            
            processed_comments = []
            for i, comment in enumerate(submission.comments):
                if i >= comments_limit:
                    break
                processed_comments.append(self._process_comment(comment))
            
            processed_submission['comments'] = processed_comments
        
        return processed_submission
    
    def _process_comment(self, comment: Comment) -> Dict[str, Any]:
        """Process a comment into a standard format.
        
        Args:
            comment: PRAW Comment object
        
        Returns:
            Processed comment as dictionary
        """
        # Create standardized comment object
        processed_comment = {
            'id': comment.id,
            'text': comment.body,
            'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
            'author': comment.author.name if comment.author else '[deleted]',
            'score': comment.score,
            'permalink': f"https://www.reddit.com{comment.permalink}",
            'is_submitter': comment.is_submitter,
            'distinguished': comment.distinguished,
            'stickied': comment.stickied,
            'edited': bool(comment.edited),
            'platform': 'reddit',
            'type': 'comment'
        }
        
        return processed_comment