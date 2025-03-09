"""Tests for the sentiment analysis models."""

import unittest
from unittest.mock import patch, MagicMock

from src.models import SentimentAnalyzer, SentimentResult, SentimentResultSet


class TestSentimentResult(unittest.TestCase):
    """Tests for the SentimentResult class."""
    
    def test_initialization(self):
        """Test initializing a SentimentResult."""
        result = SentimentResult(
            text="This is a great product!",
            sentiment=0.8,
            sentiment_class="positive",
            confidence=0.9,
            model="vader",
            metadata={"source": "test"}
        )
        
        self.assertEqual(result.text, "This is a great product!")
        self.assertEqual(result.sentiment, 0.8)
        self.assertEqual(result.sentiment_class, "positive")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.model, "vader")
        self.assertEqual(result.metadata, {"source": "test"})
    
    def test_to_dict(self):
        """Test converting a SentimentResult to a dictionary."""
        result = SentimentResult(
            text="This is terrible!",
            sentiment=-0.7,
            sentiment_class="negative",
            confidence=0.85,
            model="vader",
            metadata={"source": "test"}
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["text"], "This is terrible!")
        self.assertEqual(result_dict["sentiment"], -0.7)
        self.assertEqual(result_dict["sentiment_class"], "negative")
        self.assertEqual(result_dict["confidence"], 0.85)
        self.assertEqual(result_dict["model"], "vader")
        self.assertEqual(result_dict["metadata"], {"source": "test"})
    
    def test_from_dict(self):
        """Test creating a SentimentResult from a dictionary."""
        data = {
            "text": "This is neutral content.",
            "sentiment": 0.0,
            "sentiment_class": "neutral",
            "confidence": 0.7,
            "model": "textblob",
            "metadata": {"source": "test"}
        }
        
        result = SentimentResult.from_dict(data)
        
        self.assertEqual(result.text, "This is neutral content.")
        self.assertEqual(result.sentiment, 0.0)
        self.assertEqual(result.sentiment_class, "neutral")
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.model, "textblob")
        self.assertEqual(result.metadata, {"source": "test"})


class TestSentimentResultSet(unittest.TestCase):
    """Tests for the SentimentResultSet class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.result1 = SentimentResult(
            text="This is positive!",
            sentiment=0.8,
            sentiment_class="positive",
            confidence=0.9,
            model="vader"
        )
        
        self.result2 = SentimentResult(
            text="This is negative!",
            sentiment=-0.7,
            sentiment_class="negative",
            confidence=0.85,
            model="vader"
        )
        
        self.result3 = SentimentResult(
            text="This is neutral.",
            sentiment=0.0,
            sentiment_class="neutral",
            confidence=0.6,
            model="vader"
        )
        
        self.results = [self.result1, self.result2, self.result3]
        self.result_set = SentimentResultSet(
            results=self.results,
            model="vader",
            query="test query"
        )
    
    def test_initialization(self):
        """Test initializing a SentimentResultSet."""
        self.assertEqual(len(self.result_set), 3)
        self.assertEqual(self.result_set.model, "vader")
        self.assertEqual(self.result_set.query, "test query")
    
    def test_get_sentiment_distribution(self):
        """Test getting sentiment distribution."""
        distribution = self.result_set.get_sentiment_distribution()
        
        self.assertEqual(distribution["positive"], 1)
        self.assertEqual(distribution["negative"], 1)
        self.assertEqual(distribution["neutral"], 1)
    
    def test_get_average_sentiment(self):
        """Test getting average sentiment."""
        avg_sentiment = self.result_set.get_average_sentiment()
        
        # (0.8 + (-0.7) + 0.0) / 3 = 0.03333...
        self.assertAlmostEqual(avg_sentiment, 0.03333, places=4)
    
    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        df = self.result_set.to_dataframe()
        
        self.assertEqual(len(df), 3)
        self.assertTrue("sentiment" in df.columns)
        self.assertTrue("sentiment_class" in df.columns)
        self.assertTrue("confidence" in df.columns)
        self.assertTrue("text" in df.columns)


class TestSentimentAnalyzer(unittest.TestCase):
    """Tests for the SentimentAnalyzer class."""
    
    @patch('src.models.analyzer.SentimentIntensityAnalyzer')
    def test_analyze_vader(self, mock_vader):
        """Test analyzing text with VADER."""
        # Mock the VADER analyzer
        mock_vader_instance = MagicMock()
        mock_vader_instance.polarity_scores.return_value = {
            'neg': 0.0,
            'neu': 0.213,
            'pos': 0.787,
            'compound': 0.8316
        }
        mock_vader.return_value = mock_vader_instance
        
        # Create analyzer and analyze text
        analyzer = SentimentAnalyzer(model_name="vader")
        result = analyzer.analyze("This is a great test!")
        
        # Check result
        self.assertEqual(result.sentiment_class, "positive")
        self.assertTrue(result.sentiment > 0)
        self.assertTrue(result.confidence > 0)
        
        # Verify VADER was called
        mock_vader_instance.polarity_scores.assert_called_once()
    
    @patch('src.models.analyzer.pipeline')
    @patch('src.models.analyzer.AutoModelForSequenceClassification')
    @patch('src.models.analyzer.AutoTokenizer')
    def test_analyze_transformer(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test analyzing text with a transformer model."""
        # Mock the pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{'label': 'LABEL_2', 'score': 0.95}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create analyzer and analyze text
        analyzer = SentimentAnalyzer(model_name="roberta")
        result = analyzer.analyze("This is a great test!")
        
        # Check result
        self.assertEqual(result.sentiment_class, "positive")
        self.assertTrue(result.sentiment > 0)
        self.assertTrue(result.confidence > 0)
        
        # Verify pipeline was called
        mock_pipeline.assert_called_once()


if __name__ == '__main__':
    unittest.main()