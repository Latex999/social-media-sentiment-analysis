"""Caching utility for storing API responses and model results."""

import os
import time
import json
import hashlib
import logging
import pickle
from typing import Any, Optional, Union, Dict
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class Cache:
    """Cache class for storing and retrieving data.
    
    This class provides methods to cache data to disk with TTL (time-to-live)
    support, which is useful for caching API responses and model results.
    
    Attributes:
        cache_dir (str): Directory to store cache files
        ttl (int): Time-to-live in seconds (default: 3600)
    """
    
    def __init__(self, cache_dir: str, ttl: int = 3600):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default: 3600)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.debug(f"Cache initialized with directory: {cache_dir}, TTL: {ttl} seconds")
    
    def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        cache_file = self._get_cache_path(key)
        
        if not os.path.exists(cache_file):
            logger.debug(f"Cache miss: {key}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if cache is expired
            if 'expires_at' in data and data['expires_at'] < time.time():
                logger.debug(f"Cache expired: {key}")
                os.remove(cache_file)
                return None
            
            logger.debug(f"Cache hit: {key}")
            return data['value']
        
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: use instance TTL)
            
        Returns:
            True if successful, False otherwise
        """
        cache_file = self._get_cache_path(key)
        ttl = ttl if ttl is not None else self.ttl
        
        try:
            # Create data object with TTL
            data = {
                'key': key,
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Write to cache file
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Cache set: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        cache_file = self._get_cache_path(key)
        
        if not os.path.exists(cache_file):
            logger.debug(f"Cache delete (not found): {key}")
            return False
        
        try:
            os.remove(cache_file)
            logger.debug(f"Cache deleted: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached values.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.cache'):
                        os.remove(os.path.join(root, file))
            
            logger.debug("Cache cleared")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            Number of cleared cache entries
        """
        cleared_count = 0
        
        try:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.cache'):
                        cache_file = os.path.join(root, file)
                        
                        try:
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                            
                            # Check if cache is expired
                            if 'expires_at' in data and data['expires_at'] < time.time():
                                os.remove(cache_file)
                                cleared_count += 1
                        
                        except Exception:
                            # If we can't read the file, consider it corrupted and remove it
                            os.remove(cache_file)
                            cleared_count += 1
            
            logger.debug(f"Cleared {cleared_count} expired cache entries")
            return cleared_count
        
        except Exception as e:
            logger.error(f"Error clearing expired cache entries: {e}")
            return cleared_count
    
    def _get_cache_path(self, key: str) -> str:
        """Get the path for a cache file.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Create a nested directory structure to avoid too many files in one directory
        # Use the first 2 characters of the hash as directory name
        cache_dir = os.path.join(self.cache_dir, key_hash[:2])
        
        # Return full path to cache file
        return os.path.join(cache_dir, f"{key_hash}.cache")