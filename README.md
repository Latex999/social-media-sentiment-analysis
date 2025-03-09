# Social Media Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-success)

A professional, advanced sentiment analysis tool for social media content. This application can analyze posts from Twitter (X) and Reddit to determine sentiment trends, providing valuable insights for market research, brand monitoring, and social listening.

## 🌟 Features

- **Multi-platform Analysis**: Process data from both Twitter and Reddit
- **Advanced NLP Processing**: Utilizes state-of-the-art NLP models for accurate sentiment classification
- **Customizable Sentiment Models**: Choose between various pre-trained models or fine-tune for your specific domain
- **Real-time Analysis**: Stream and analyze social media posts in real-time
- **Historical Analysis**: Process historical data to identify sentiment trends over time
- **Data Visualization**: Interactive dashboards for sentiment insights
- **Export Capabilities**: Save results in various formats (CSV, JSON, Excel)
- **Topic Detection**: Automatically identify key topics in the analyzed content
- **Entity Recognition**: Extract and analyze mentions of products, brands, people, and more
- **Multilingual Support**: Analyze content in multiple languages

## 📊 Dashboard Preview

![Dashboard Preview](assets/dashboard_preview.png)

## 🔧 Installation

```bash
# Clone this repository
git clone https://github.com/Latex999/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API credentials
cp .env.example .env
# Edit .env file with your API keys
```

## 🚀 Quick Start

```bash
# Run the web interface
python -m src.app

# Or use the CLI for batch processing
python -m src.cli --source twitter --query "#AI" --limit 100
```

Then open your browser to http://localhost:8501 to access the dashboard.

## 🔄 Data Sources

### Twitter (X)
Access tweets via the Twitter API v2. Requires developer account and API credentials.

### Reddit
Access Reddit posts and comments via the Reddit API. Requires Reddit API credentials.

## 🤖 Models

The application includes several pre-trained models:

1. **VADER** - Rule-based sentiment analyzer (fast but less accurate)
2. **DistilBERT** - A lightweight transformer model fine-tuned for sentiment analysis
3. **RoBERTa** - A robustly optimized BERT model for more accurate sentiment detection
4. **Custom Models** - Fine-tune specific models for your domain

## 📁 Project Structure

```
social-media-sentiment-analysis/
├── src/
│   ├── api/              # API integrations for Twitter and Reddit
│   ├── models/           # Sentiment analysis models
│   ├── preprocessing/    # Text preprocessing functions
│   ├── visualization/    # Data visualization components
│   ├── utils/            # Utility functions
│   ├── app.py            # Streamlit web application
│   └── cli.py            # Command-line interface
├── notebooks/            # Jupyter notebooks for analysis and model training
├── tests/                # Unit and integration tests
├── assets/               # Images and static files
├── data/                 # Sample data and saved results
├── docs/                 # Extended documentation
├── requirements.txt      # Project dependencies
├── setup.py              # Package configuration
├── .env.example          # Example environment variables
└── README.md             # Project documentation
```

## 📈 Example Usage

### Python API

```python
from src.api import TwitterClient, RedditClient
from src.models import SentimentAnalyzer

# Initialize clients
twitter_client = TwitterClient(api_key="YOUR_API_KEY")
reddit_client = RedditClient(client_id="YOUR_CLIENT_ID", client_secret="YOUR_CLIENT_SECRET")

# Fetch data
tweets = twitter_client.search(query="#AI", limit=100)
reddit_posts = reddit_client.search(subreddit="MachineLearning", query="sentiment", limit=50)

# Analyze sentiment
analyzer = SentimentAnalyzer(model="roberta")
twitter_results = analyzer.analyze_batch(tweets)
reddit_results = analyzer.analyze_batch(reddit_posts)

# Get sentiment distribution
twitter_sentiment = twitter_results.get_sentiment_distribution()
print(f"Twitter Sentiment: {twitter_sentiment}")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_models.py
```

## 📚 Documentation

For detailed documentation, visit the [/docs](/docs) directory or run:

```bash
# Generate and serve documentation locally
cd docs
make html
python -m http.server -d _build/html
```

Then open your browser to http://localhost:8000.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have any questions or feedback, please open an issue on this repository.