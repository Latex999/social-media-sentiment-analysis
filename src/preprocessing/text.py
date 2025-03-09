"""Text preprocessing utilities for sentiment analysis."""

import re
import logging
import unicodedata
import contractions
import emoji
from typing import List, Dict, Any, Optional, Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Set up logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load stopwords
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    logger.warning("Could not load stopwords. NLTK resources may not be available.")
    STOP_WORDS = set()


def preprocess_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    remove_emojis: bool = False,
    convert_emojis: bool = True,
    lowercase: bool = True,
    strip_punctuation: bool = False,
    replace_contractions: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    min_token_length: int = 2,
    max_token_length: int = 50,
) -> str:
    """Preprocess text for sentiment analysis.
    
    Args:
        text: The text to preprocess
        remove_urls: Whether to remove URLs
        remove_mentions: Whether to remove user mentions (@user)
        remove_hashtags: Whether to remove hashtags (#tag)
        remove_emojis: Whether to remove emojis
        convert_emojis: Whether to convert emojis to text
        lowercase: Whether to convert to lowercase
        strip_punctuation: Whether to remove punctuation
        replace_contractions: Whether to expand contractions
        remove_numbers: Whether to remove numeric tokens
        remove_stopwords: Whether to remove stopwords
        min_token_length: Minimum token length to keep
        max_token_length: Maximum token length to keep
        
    Returns:
        Preprocessed text
    """
    # Skip processing for empty text
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove user mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags or just the hash symbol
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    else:
        # Keep the hashtag text but remove the # symbol
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle emojis
    if remove_emojis:
        text = remove_emoji(text)
    elif convert_emojis:
        text = convert_emoji_to_text(text)
    
    # Replace contractions
    if replace_contractions:
        text = expand_contractions(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Tokenize and filter tokens
    tokens = word_tokenize(text)
    filtered_tokens = []
    
    for token in tokens:
        # Skip short tokens
        if len(token) < min_token_length:
            continue
        
        # Skip long tokens
        if len(token) > max_token_length:
            continue
        
        # Skip numbers if requested
        if remove_numbers and token.isdigit():
            continue
        
        # Skip stopwords if requested
        if remove_stopwords and token.lower() in STOP_WORDS:
            continue
        
        # Remove punctuation if requested
        if strip_punctuation:
            token = re.sub(r'[^\w\s]', '', token)
            # Skip if token is empty after removing punctuation
            if not token:
                continue
        
        filtered_tokens.append(token)
    
    # Reconstruct the text
    processed_text = ' '.join(filtered_tokens)
    
    # Remove extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text


def remove_emoji(text: str) -> str:
    """Remove emojis from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with emojis removed
    """
    return emoji.replace_emoji(text, replace='')


def convert_emoji_to_text(text: str) -> str:
    """Convert emojis to their text representation.
    
    Args:
        text: Input text
        
    Returns:
        Text with emojis converted to words
    """
    def replace_emoji_with_text(match):
        emoji_char = match.group(0)
        return f" {emoji.demojize(emoji_char).replace('_', ' ')} "
    
    # Find all emojis and replace them with their textual representation
    emoji_pattern = emoji.get_emoji_regexp()
    text = emoji_pattern.sub(replace_emoji_with_text, text)
    
    # Clean up extra spaces
    return re.sub(r'\s+', ' ', text).strip()


def expand_contractions(text: str) -> str:
    """Expand contractions in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with contractions expanded
    """
    return contractions.fix(text)


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.
    
    Args:
        text: Input text
        
    Returns:
        List of hashtags
    """
    return re.findall(r'#(\w+)', text)


def extract_mentions(text: str) -> List[str]:
    """Extract mentions from text.
    
    Args:
        text: Input text
        
    Returns:
        List of user mentions
    """
    return re.findall(r'@(\w+)', text)


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of URLs
    """
    return re.findall(r'https?://\S+|www\.\S+', text)


def extract_emojis(text: str) -> List[str]:
    """Extract emojis from text.
    
    Args:
        text: Input text
        
    Returns:
        List of emojis
    """
    emoji_pattern = emoji.get_emoji_regexp()
    return emoji_pattern.findall(text)