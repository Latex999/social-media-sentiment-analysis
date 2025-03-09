"""Main sentiment analyzer implementation."""

import os
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from tqdm import tqdm
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

from src.utils.config import MODEL_DIR, config
from src.preprocessing.text import preprocess_text
from src.models.result import SentimentResult, SentimentResultSet

# Set up logging
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Main sentiment analyzer class that provides a unified interface for
    different sentiment analysis models.
    
    This class supports multiple sentiment analysis models:
    - VADER: Rule-based sentiment analyzer
    - TextBlob: Simple lexicon-based sentiment analyzer
    - DistilBERT: A lightweight transformer model fine-tuned for sentiment analysis
    - RoBERTa: A robustly optimized BERT model for more accurate sentiment detection
    
    Attributes:
        model_name (str): Name of the model to use
        model: The loaded sentiment analysis model
        tokenizer: Tokenizer for transformer models
        device (str): Device to run model on (cuda or cpu)
        batch_size (int): Batch size for processing
        use_cache (bool): Whether to use model cache
    """
    
    # Available models
    AVAILABLE_MODELS = {
        'vader': 'VADER (Rule-based)',
        'textblob': 'TextBlob (Lexicon-based)',
        'distilbert': 'DistilBERT (cardiffnlp/twitter-roberta-base-sentiment)',
        'roberta': 'RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)'
    }
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'distilbert': {
            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'labels': ['negative', 'positive']
        },
        'roberta': {
            'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'labels': ['negative', 'neutral', 'positive']
        }
    }
    
    def __init__(
        self, 
        model_name: str = 'roberta',
        device: Optional[str] = None,
        batch_size: int = 32,
        use_cache: bool = True,
        preprocessing_options: Optional[Dict[str, bool]] = None
    ):
        """Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the model to use
            device: Device to run model on (None for auto-detect)
            batch_size: Batch size for processing
            use_cache: Whether to use model cache
            preprocessing_options: Options for text preprocessing
        """
        self.model_name = model_name.lower()
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.preprocessing_options = preprocessing_options or config['preprocessing']
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing sentiment analyzer with model: {self.model_name}, device: {self.device}")
        
        # Load the model
        self._load_model()
    
    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> SentimentResult:
        """Analyze the sentiment of a single text.
        
        Args:
            text: Text to analyze
            metadata: Additional metadata about the text
            
        Returns:
            SentimentResult object
        """
        # Preprocess the text
        processed_text = preprocess_text(text, **self.preprocessing_options)
        
        # Skip empty texts
        if not processed_text:
            logger.warning(f"Empty text after preprocessing: '{text}'")
            return SentimentResult(
                text=text,
                sentiment=0.0,
                sentiment_class='neutral',
                confidence=1.0,
                model=self.model_name,
                metadata=metadata or {}
            )
        
        try:
            # Call the appropriate model method
            if self.model_name == 'vader':
                sentiment, sentiment_class, confidence = self._analyze_vader(processed_text)
            elif self.model_name == 'textblob':
                sentiment, sentiment_class, confidence = self._analyze_textblob(processed_text)
            elif self.model_name in ['distilbert', 'roberta']:
                sentiment, sentiment_class, confidence = self._analyze_transformer(processed_text)
            else:
                raise ValueError(f"Model {self.model_name} not implemented")
            
            # Create and return the result
            return SentimentResult(
                text=text,
                sentiment=sentiment,
                sentiment_class=sentiment_class,
                confidence=confidence,
                model=self.model_name,
                metadata=metadata or {}
            )
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            logger.debug(traceback.format_exc())
            
            # Return neutral sentiment on error
            return SentimentResult(
                text=text,
                sentiment=0.0,
                sentiment_class='neutral',
                confidence=0.0,
                model=self.model_name,
                metadata=metadata or {}
            )
    
    def analyze_batch(
        self, 
        texts: Union[List[str], List[Dict[str, Any]]],
        text_field: str = 'text',
        show_progress: bool = True,
        max_workers: int = 4
    ) -> SentimentResultSet:
        """Analyze the sentiment of multiple texts.
        
        Args:
            texts: List of texts or dictionaries containing texts
            text_field: Field name to extract text from if texts is a list of dictionaries
            show_progress: Whether to show a progress bar
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            SentimentResultSet object
        """
        results = []
        
        # If texts is a list of dictionaries, extract the text and keep the rest as metadata
        if texts and isinstance(texts[0], dict):
            items_to_process = []
            for item in texts:
                if text_field in item:
                    text = item[text_field]
                    metadata = {k: v for k, v in item.items() if k != text_field}
                    items_to_process.append((text, metadata))
                else:
                    logger.warning(f"Text field '{text_field}' not found in item: {item}")
        else:
            # If texts is a list of strings, create empty metadata for each
            items_to_process = [(text, {}) for text in texts]
        
        # Process in parallel for faster analysis
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a progress bar if requested
            if show_progress:
                with tqdm(total=len(items_to_process), desc=f"Analyzing with {self.model_name}") as pbar:
                    # Submit all tasks
                    future_to_item = {
                        executor.submit(self.analyze, text, metadata): (text, metadata)
                        for text, metadata in items_to_process
                    }
                    
                    # Process results as they complete
                    for future in future_to_item:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
            else:
                # Process without progress bar
                future_to_item = {
                    executor.submit(self.analyze, text, metadata): (text, metadata)
                    for text, metadata in items_to_process
                }
                
                for future in future_to_item:
                    result = future.result()
                    results.append(result)
        
        logger.info(f"Analyzed {len(results)} texts with model: {self.model_name}")
        
        # Create and return the result set
        return SentimentResultSet(
            results=results,
            model=self.model_name
        )
    
    def _load_model(self) -> None:
        """Load the sentiment analysis model."""
        try:
            if self.model_name == 'vader':
                self.model = SentimentIntensityAnalyzer()
                logger.info("Loaded VADER sentiment analyzer")
            
            elif self.model_name == 'textblob':
                # TextBlob doesn't require explicit loading
                self.model = None
                logger.info("TextBlob sentiment analyzer ready")
            
            elif self.model_name in ['distilbert', 'roberta']:
                # Get model configuration
                model_config = self.MODEL_CONFIGS[self.model_name]
                model_id = model_config['model_name']
                
                # Set model paths
                model_dir = os.path.join(MODEL_DIR, self.model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # Load model and tokenizer (download if not cached)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=model_dir)
                
                # Move model to the specified device
                self.model.to(self.device)
                
                # Create pipeline for easier inference
                self.pipeline = pipeline(
                    task="sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == 'cuda' else -1
                )
                
                logger.info(f"Loaded {self.model_name.upper()} model from {model_id}")
            
            else:
                raise ValueError(f"Model {self.model_name} not implemented")
        
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def _analyze_vader(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using VADER.
        
        Args:
            text: Preprocessed text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_class, confidence)
        """
        # Get VADER sentiment scores
        scores = self.model.polarity_scores(text)
        
        # Extract the compound score (-1 to 1)
        sentiment = scores['compound']
        
        # Determine sentiment class
        if sentiment >= 0.05:
            sentiment_class = 'positive'
        elif sentiment <= -0.05:
            sentiment_class = 'negative'
        else:
            sentiment_class = 'neutral'
        
        # Estimate confidence based on the magnitude of the compound score
        confidence = min(1.0, abs(sentiment) * 1.5)
        
        return sentiment, sentiment_class, confidence
    
    def _analyze_textblob(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using TextBlob.
        
        Args:
            text: Preprocessed text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_class, confidence)
        """
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get polarity score (-1 to 1)
        sentiment = blob.sentiment.polarity
        
        # Determine sentiment class
        if sentiment > 0.05:
            sentiment_class = 'positive'
        elif sentiment < -0.05:
            sentiment_class = 'negative'
        else:
            sentiment_class = 'neutral'
        
        # Use subjectivity as a measure of confidence
        confidence = blob.sentiment.subjectivity
        
        return sentiment, sentiment_class, confidence
    
    def _analyze_transformer(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using a transformer model.
        
        Args:
            text: Preprocessed text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_class, confidence)
        """
        # Get model configuration for label mapping
        model_config = self.MODEL_CONFIGS[self.model_name]
        labels = model_config['labels']
        
        # Use the pipeline for inference
        result = self.pipeline(text)[0]
        label = result['label']
        score = result['score']  # Confidence score
        
        # Map the model's label to our sentiment classes
        if label == 'LABEL_0':
            idx = 0
        elif label == 'LABEL_1':
            idx = 1
        elif label == 'LABEL_2':
            idx = 2
        else:
            # Direct label name
            try:
                idx = labels.index(label.lower())
            except ValueError:
                # Default to neutral if mapping fails
                idx = len(labels) // 2
        
        sentiment_class = labels[idx]
        
        # Convert to our standard sentiment score (-1 to 1)
        if len(labels) == 2:  # Binary classification (negative, positive)
            # Map the score from [0,1] to [-1,1]
            # If idx is 0 (negative), we want score to be negative
            sentiment = score if idx == 1 else -score
        else:  # 3-class classification (negative, neutral, positive)
            if sentiment_class == 'negative':
                sentiment = -score
            elif sentiment_class == 'positive':
                sentiment = score
            else:  # neutral
                sentiment = 0.0
        
        # The model's score is already a confidence value
        confidence = score
        
        return sentiment, sentiment_class, confidence