"""Configuration settings for the application."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create necessary directories
for directory in [DATA_DIR, CACHE_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    # API Settings
    "twitter": {
        "cache_ttl": 3600,  # Cache time-to-live in seconds
        "max_results_per_query": 100,
        "default_time_filter": "week",
    },
    "reddit": {
        "cache_ttl": 3600,
        "max_results_per_query": 100,
        "default_time_filter": "month",
        "default_comment_limit": 10,
    },
    
    # Model Settings
    "models": {
        "default_model": "roberta",
        "available_models": ["vader", "textblob", "distilbert", "roberta"],
        "model_batch_size": 32,
    },
    
    # Preprocessing Settings
    "preprocessing": {
        "remove_urls": True,
        "remove_mentions": True,
        "remove_hashtags": False,
        "remove_emojis": False,
        "convert_emojis": True,
        "lowercase": True,
        "strip_punctuation": False,
        "replace_contractions": True,
        "min_token_length": 2,
        "max_token_length": 50,
    },
    
    # Visualization Settings
    "visualization": {
        "default_theme": "light",
        "color_palette": "viridis",
        "default_chart_type": "bar",
        "show_grid": True,
        "chart_height": 500,
        "chart_width": 800,
    },
    
    # Application Settings
    "app": {
        "title": "Social Media Sentiment Analysis",
        "description": "Analyze sentiment in social media posts from Twitter and Reddit",
        "debug": False,
        "port": 8501,
        "host": "localhost",
        "max_file_upload_size_mb": 10,
        "max_results_limit": 1000,
        "default_results_limit": 100,
    },
}


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a file or use default.
    
    Args:
        config_file: Path to a JSON configuration file (optional)
        
    Returns:
        Loaded configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # If config file is provided, load and update default config
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Update default config with file config
                _deep_update(config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
    
    # Override with environment variables if present
    _override_with_env(config)
    
    return config


def save_config(config: Dict[str, Any], config_file: str) -> None:
    """Save configuration to a file.
    
    Args:
        config: Configuration dictionary to save
        config_file: Path to save the configuration to
    """
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_file}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_file}: {e}")


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively update nested dictionaries.
    
    Args:
        target: Target dictionary to update
        source: Source dictionary with new values
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _override_with_env(config: Dict[str, Any], prefix: str = "SMSA_") -> None:
    """Override configuration with environment variables.
    
    Environment variables should be in the format:
    SMSA_SECTION_SUBSECTION_KEY=value
    
    For example:
    SMSA_MODELS_DEFAULT_MODEL=vader
    
    Args:
        config: Configuration dictionary to update
        prefix: Prefix for environment variables
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        
        # Remove prefix and split by underscore
        key_parts = env_key[len(prefix):].lower().split('_')
        
        # Navigate to the right place in the config
        current = config
        for i, part in enumerate(key_parts):
            # If this is the last part, set the value
            if i == len(key_parts) - 1:
                # Try to convert to the appropriate type
                if env_value.lower() == 'true':
                    current[part] = True
                elif env_value.lower() == 'false':
                    current[part] = False
                elif env_value.isdigit():
                    current[part] = int(env_value)
                elif env_value.replace('.', '', 1).isdigit() and env_value.count('.') == 1:
                    current[part] = float(env_value)
                else:
                    current[part] = env_value
            else:
                # If this part doesn't exist in the config, create it
                if part not in current:
                    current[part] = {}
                current = current[part]


# Load the configuration
config = load_config()