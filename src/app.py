"""
Streamlit web application for the Social Media Sentiment Analysis tool.

This module provides an interactive web interface for analyzing sentiment
in social media posts from Twitter and Reddit.

Run with: streamlit run src/app.py
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import traceback

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import StringIO

from src.api import TwitterClient, RedditClient
from src.models import SentimentAnalyzer, SentimentResult, SentimentResultSet
from src.preprocessing.text import preprocess_text, extract_hashtags, extract_mentions
from src.visualization.charts import (
    create_sentiment_distribution_chart,
    create_sentiment_over_time_chart,
    create_sentiment_comparison_chart,
    create_wordcloud
)
from src.utils.config import config, RESULTS_DIR, DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# App title and description
APP_TITLE = "Social Media Sentiment Analysis"
APP_DESCRIPTION = "Analyze sentiment in social media posts from Twitter and Reddit"
GITHUB_REPO = "https://github.com/Latex999/social-media-sentiment-analysis"


@st.cache_resource
def get_twitter_client():
    """Initialize and cache the Twitter client."""
    try:
        return TwitterClient()
    except Exception as e:
        logger.error(f"Error initializing Twitter client: {e}")
        return None


@st.cache_resource
def get_reddit_client():
    """Initialize and cache the Reddit client."""
    try:
        return RedditClient()
    except Exception as e:
        logger.error(f"Error initializing Reddit client: {e}")
        return None


@st.cache_resource
def get_sentiment_analyzer(model_name="roberta"):
    """Initialize and cache the sentiment analyzer for a specific model."""
    try:
        return SentimentAnalyzer(model_name=model_name)
    except Exception as e:
        logger.error(f"Error initializing sentiment analyzer with model {model_name}: {e}")
        return None


def analyze_text_input(text_input: str, model_name: str) -> None:
    """Analyze a single text input and display results.
    
    Args:
        text_input: The text to analyze
        model_name: The model to use for analysis
    """
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
        return
    
    # Get the analyzer for the selected model
    analyzer = get_sentiment_analyzer(model_name)
    if not analyzer:
        st.error(f"Failed to initialize the {model_name} model. Please try another model.")
        return
    
    # Show a spinner while analyzing
    with st.spinner(f"Analyzing sentiment with {model_name}..."):
        # Analyze the text
        result = analyzer.analyze(text_input)
    
    # Display the result
    st.subheader("Analysis Result")
    
    # Determine result color based on sentiment
    if result.sentiment_class == 'positive':
        sentiment_color = "green"
    elif result.sentiment_class == 'negative':
        sentiment_color = "red"
    else:
        sentiment_color = "gray"
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Sentiment Score", 
            f"{result.sentiment:.2f}",
            delta=None
        )
    
    with col2:
        st.markdown(f"""
        <div style="text-align:center;">
            <h3 style="color:{sentiment_color}; margin-bottom:0;">{result.sentiment_class.upper()}</h3>
            <p>Sentiment Class</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric(
            "Confidence", 
            f"{result.confidence:.2f}",
            delta=None
        )
    
    # Show preprocessing details
    with st.expander("Preprocessing Details"):
        original_text = result.text
        processed_text = preprocess_text(original_text)
        
        st.markdown("**Original Text:**")
        st.text(original_text)
        
        st.markdown("**Processed Text:**")
        st.text(processed_text)
        
        # Extract entities
        hashtags = extract_hashtags(original_text)
        mentions = extract_mentions(original_text)
        
        if hashtags:
            st.markdown(f"**Hashtags:** {', '.join(hashtags)}")
        
        if mentions:
            st.markdown(f"**Mentions:** {', '.join(mentions)}")


def analyze_twitter_data(query: str, limit: int, model_name: str) -> None:
    """Fetch and analyze Twitter data.
    
    Args:
        query: Search query for Twitter
        limit: Maximum number of tweets to analyze
        model_name: The model to use for analysis
    """
    # Get the Twitter client
    twitter_client = get_twitter_client()
    if not twitter_client:
        st.error(
            "Failed to initialize Twitter client. Please check your API credentials "
            "in the .env file."
        )
        return
    
    # Get the analyzer for the selected model
    analyzer = get_sentiment_analyzer(model_name)
    if not analyzer:
        st.error(f"Failed to initialize the {model_name} model. Please try another model.")
        return
    
    # Show a spinner while fetching and analyzing
    with st.spinner("Fetching tweets..."):
        try:
            # Fetch tweets
            tweets = twitter_client.search(query=query, limit=limit)
            if not tweets:
                st.warning(f"No tweets found for query: {query}")
                return
            
            st.success(f"Retrieved {len(tweets)} tweets.")
            
            # Display a sample of the tweets
            with st.expander("Sample Tweets", expanded=False):
                for i, tweet in enumerate(tweets[:5]):
                    st.markdown(f"**Tweet {i+1}:** {tweet['text']}")
                    st.markdown(f"*Posted on: {tweet['created_at']}*")
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error fetching tweets: {str(e)}")
            logger.error(f"Error fetching tweets: {e}")
            logger.debug(traceback.format_exc())
            return
    
    # Analyze the tweets
    with st.spinner(f"Analyzing sentiment with {model_name}..."):
        try:
            # Perform sentiment analysis
            results = analyzer.analyze_batch(tweets, text_field='text')
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(RESULTS_DIR, f"twitter_{timestamp}.json")
            results.save_to_json(results_file)
            
            st.success(f"Analyzed {len(results)} tweets. Results saved to {results_file}")
        
        except Exception as e:
            st.error(f"Error analyzing tweets: {str(e)}")
            logger.error(f"Error analyzing tweets: {e}")
            logger.debug(traceback.format_exc())
            return
    
    # Display results
    display_analysis_results(results, source="Twitter", query=query)


def analyze_reddit_data(
    subreddit: str, 
    query: str, 
    limit: int, 
    time_filter: str, 
    include_comments: bool,
    model_name: str
) -> None:
    """Fetch and analyze Reddit data.
    
    Args:
        subreddit: Subreddit to search in (or "all" for all subreddits)
        query: Search query for Reddit
        limit: Maximum number of posts to analyze
        time_filter: Time filter for posts
        include_comments: Whether to include comments in analysis
        model_name: The model to use for analysis
    """
    # Get the Reddit client
    reddit_client = get_reddit_client()
    if not reddit_client:
        st.error(
            "Failed to initialize Reddit client. Please check your API credentials "
            "in the .env file."
        )
        return
    
    # Get the analyzer for the selected model
    analyzer = get_sentiment_analyzer(model_name)
    if not analyzer:
        st.error(f"Failed to initialize the {model_name} model. Please try another model.")
        return
    
    # Show a spinner while fetching and analyzing
    with st.spinner("Fetching Reddit posts..."):
        try:
            # Determine if we should search in a specific subreddit or all
            sub = None if subreddit.lower() == "all" else subreddit
            
            # Fetch Reddit posts
            posts = reddit_client.search(
                query=query, 
                subreddit=sub, 
                limit=limit, 
                time_filter=time_filter,
                include_comments=include_comments,
                comments_limit=10 if include_comments else 0
            )
            
            if not posts:
                st.warning(f"No Reddit posts found for query: {query}")
                return
            
            st.success(f"Retrieved {len(posts)} Reddit posts.")
            
            # Display a sample of the posts
            with st.expander("Sample Posts", expanded=False):
                for i, post in enumerate(posts[:5]):
                    st.markdown(f"**Post {i+1}:** {post['title']}")
                    st.markdown(f"*Posted on: {post['created_utc']} in r/{post['subreddit']}*")
                    if post['text']:
                        st.text(post['text'][:300] + ("..." if len(post['text']) > 300 else ""))
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Error fetching Reddit posts: {str(e)}")
            logger.error(f"Error fetching Reddit posts: {e}")
            logger.debug(traceback.format_exc())
            return
    
    # Prepare items to analyze (combine posts and comments if included)
    items_to_analyze = []
    post_count = 0
    comment_count = 0
    
    # Add posts
    for post in posts:
        items_to_analyze.append({
            'text': f"{post['title']} {post['text']}".strip(),
            'id': post['id'],
            'type': 'post',
            'created_utc': post['created_utc'],
            'subreddit': post['subreddit'],
            'author': post['author'],
            'score': post['score'],
            'permalink': post['permalink']
        })
        post_count += 1
        
        # Add comments if they exist and are requested
        if include_comments and 'comments' in post and post['comments']:
            for comment in post['comments']:
                items_to_analyze.append({
                    'text': comment['text'],
                    'id': comment['id'],
                    'type': 'comment',
                    'created_utc': comment['created_utc'],
                    'subreddit': post['subreddit'],
                    'author': comment['author'],
                    'score': comment['score'],
                    'permalink': comment['permalink'],
                    'post_id': post['id']
                })
                comment_count += 1
    
    # Analyze the items
    with st.spinner(f"Analyzing sentiment with {model_name}..."):
        try:
            # Perform sentiment analysis
            results = analyzer.analyze_batch(items_to_analyze, text_field='text')
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(RESULTS_DIR, f"reddit_{timestamp}.json")
            results.save_to_json(results_file)
            
            st.success(
                f"Analyzed {len(results)} items ({post_count} posts, {comment_count} comments). "
                f"Results saved to {results_file}"
            )
        
        except Exception as e:
            st.error(f"Error analyzing Reddit content: {str(e)}")
            logger.error(f"Error analyzing Reddit content: {e}")
            logger.debug(traceback.format_exc())
            return
    
    # Display results
    display_analysis_results(results, source="Reddit", query=query)


def analyze_csv_data(data: pd.DataFrame, text_column: str, model_name: str) -> None:
    """Analyze text data from a CSV file.
    
    Args:
        data: DataFrame containing the data
        text_column: Column name containing the text to analyze
        model_name: The model to use for analysis
    """
    # Validate data
    if text_column not in data.columns:
        st.error(f"Column '{text_column}' not found in the uploaded file.")
        return
    
    # Remove rows with empty text
    data = data[data[text_column].notna()]
    data = data[data[text_column].str.strip() != ""]
    
    if len(data) == 0:
        st.error("No valid text data found in the selected column.")
        return
    
    # Get the analyzer for the selected model
    analyzer = get_sentiment_analyzer(model_name)
    if not analyzer:
        st.error(f"Failed to initialize the {model_name} model. Please try another model.")
        return
    
    # Show a spinner while analyzing
    with st.spinner(f"Analyzing {len(data)} records with {model_name}..."):
        try:
            # Convert DataFrame to list of dicts for analysis
            items_to_analyze = data.to_dict('records')
            
            # Perform sentiment analysis
            results = analyzer.analyze_batch(items_to_analyze, text_field=text_column)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(RESULTS_DIR, f"csv_analysis_{timestamp}.json")
            results.save_to_json(results_file)
            
            # Also save as CSV with original data
            csv_results = results.to_dataframe()
            
            # Join with original data (excluding the text column to avoid duplication)
            original_data = data.drop(columns=[text_column])
            result_data = pd.concat([csv_results, original_data], axis=1)
            
            csv_results_file = os.path.join(RESULTS_DIR, f"csv_analysis_{timestamp}.csv")
            result_data.to_csv(csv_results_file, index=False)
            
            st.success(
                f"Analyzed {len(results)} records. "
                f"Results saved to {results_file} and {csv_results_file}"
            )
        
        except Exception as e:
            st.error(f"Error analyzing data: {str(e)}")
            logger.error(f"Error analyzing data: {e}")
            logger.debug(traceback.format_exc())
            return
    
    # Display results
    display_analysis_results(results, source="CSV Data", query=None)


def display_analysis_results(
    results: SentimentResultSet,
    source: str,
    query: Optional[str] = None
) -> None:
    """Display sentiment analysis results with visualizations.
    
    Args:
        results: SentimentResultSet containing analysis results
        source: Source of the data (Twitter, Reddit, etc.)
        query: Search query used to generate the results
    """
    if len(results) == 0:
        st.warning("No results to display.")
        return
    
    st.header("Analysis Results")
    
    # Display summary metrics
    sentiment_dist = results.get_sentiment_distribution()
    total_items = sum(sentiment_dist.values())
    avg_sentiment = results.get_average_sentiment()
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Items", 
            str(total_items),
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Sentiment", 
            f"{avg_sentiment:.2f}",
            delta=None
        )
    
    with col3:
        positive_pct = sentiment_dist['positive'] / total_items * 100 if total_items else 0
        st.metric(
            "Positive", 
            f"{positive_pct:.1f}%",
            delta=None
        )
    
    with col4:
        negative_pct = sentiment_dist['negative'] / total_items * 100 if total_items else 0
        st.metric(
            "Negative", 
            f"{negative_pct:.1f}%",
            delta=None
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment Distribution", 
        "Sentiment Over Time", 
        "Word Cloud",
        "Raw Data"
    ])
    
    with tab1:
        # Sentiment distribution chart
        title = f"Sentiment Distribution for {source}"
        if query:
            title += f" - '{query}'"
        
        fig = create_sentiment_distribution_chart(
            results,
            title=title,
            use_plotly=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        This chart shows the distribution of sentiment in the analyzed content:
        - **Positive**: Content with a positive tone or expressing positive emotions
        - **Neutral**: Content that is objective or doesn't express strong sentiment
        - **Negative**: Content with a negative tone or expressing negative emotions
        """)
    
    with tab2:
        # Try to extract timestamp for time-based analysis
        try:
            # Different sources might have different timestamp fields
            if source == "Twitter":
                time_field = 'timestamp'
            elif source == "Reddit":
                time_field = 'created_utc'
            else:
                # Look for common timestamp fields in the data
                df = results.to_dataframe()
                time_fields = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                time_field = time_fields[0] if time_fields else 'timestamp'
            
            fig = create_sentiment_over_time_chart(
                results,
                time_field=time_field,
                interval='H',  # Group by hour
                use_plotly=True,
                title=f"Sentiment Trend Over Time - {source}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            This chart shows how sentiment changes over time:
            - **Blue line**: Average sentiment score (-1 to 1)
            - **Gray bars**: Volume of posts/comments
            - The dotted lines show the percentage of positive and negative content over time
            """)
        
        except Exception as e:
            st.warning(
                "Could not create time-based analysis. This may happen if the data "
                "doesn't contain proper timestamp information."
            )
            logger.error(f"Error creating time chart: {e}")
    
    with tab3:
        # Word cloud for positive and negative sentiment
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Words")
            try:
                df = results.to_dataframe()
                positive_texts = df[df['sentiment_class'] == 'positive']['text'].tolist()
                
                if positive_texts:
                    fig = create_wordcloud(
                        positive_texts,
                        title="Positive Content Word Cloud",
                        colormap="Greens",
                        background_color="white"
                    )
                    st.pyplot(fig)
                else:
                    st.info("No positive content to generate word cloud.")
            
            except Exception as e:
                st.warning("Could not create positive word cloud.")
                logger.error(f"Error creating positive word cloud: {e}")
        
        with col2:
            st.subheader("Negative Words")
            try:
                df = results.to_dataframe()
                negative_texts = df[df['sentiment_class'] == 'negative']['text'].tolist()
                
                if negative_texts:
                    fig = create_wordcloud(
                        negative_texts,
                        title="Negative Content Word Cloud",
                        colormap="Reds",
                        background_color="white"
                    )
                    st.pyplot(fig)
                else:
                    st.info("No negative content to generate word cloud.")
            
            except Exception as e:
                st.warning("Could not create negative word cloud.")
                logger.error(f"Error creating negative word cloud: {e}")
    
    with tab4:
        # Display raw data in a table
        df = results.to_dataframe()
        
        # Add filters
        st.subheader("Filter Data")
        sentiment_filter = st.multiselect(
            "Sentiment Class",
            options=["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
        
        # Apply filters
        if sentiment_filter:
            df = df[df['sentiment_class'].isin(sentiment_filter)]
        
        # Display the data
        st.dataframe(df)
        
        # Add download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sentiment_analysis_{source.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    """Main function for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #1E88E5 !important;
    }
    .subheader {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/Latex999/social-media-sentiment-analysis/main/assets/logo.png", width=300)
        
        st.markdown("## Analysis Options")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            options=["Text Input", "Twitter", "Reddit", "CSV Upload"],
            index=0
        )
        
        # Model selection
        model_options = {
            "vader": "VADER (Rule-based, Fast)",
            "textblob": "TextBlob (Simple)",
            "distilbert": "DistilBERT (Balanced)",
            "roberta": "RoBERTa (Accurate)"
        }
        
        selected_model = st.selectbox(
            "Select Sentiment Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=3  # Default to RoBERTa
        )
        
        # Add info about the project
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This app analyzes sentiment in social media content "
            "using state-of-the-art NLP techniques."
        )
        st.markdown(f"[GitHub Repository]({GITHUB_REPO})")
    
    # Main content
    st.markdown('<h1 class="main-header">Social Media Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Different UI based on selected data source
    if data_source == "Text Input":
        st.markdown('<h2 class="subheader">Analyze Text</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Enter a text to analyze its sentiment. This can be a tweet, Reddit post, or any text content.
        """)
        
        text_input = st.text_area(
            "Text to analyze",
            height=150,
            help="Enter the text you want to analyze for sentiment."
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            analyze_text_input(text_input, selected_model)
    
    elif data_source == "Twitter":
        st.markdown('<h2 class="subheader">Twitter Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p>Analyze sentiment in tweets matching your search query. You can search for:</p>
        <ul>
            <li>Keywords or phrases (e.g., "climate change")</li>
            <li>Hashtags (e.g., "#AI")</li>
            <li>From specific accounts (e.g., "from:username")</li>
            <li>Combine with operators: AND, OR, -term (exclude)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Twitter search form
        with st.form("twitter_form"):
            query = st.text_input(
                "Search Query",
                help="Enter a search query for Twitter"
            )
            
            limit = st.slider(
                "Number of Tweets",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Maximum number of tweets to retrieve and analyze"
            )
            
            submit_button = st.form_submit_button("Analyze Tweets", type="primary")
            
            if submit_button and query:
                analyze_twitter_data(query, limit, selected_model)
    
    elif data_source == "Reddit":
        st.markdown('<h2 class="subheader">Reddit Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p>Analyze sentiment in Reddit posts and comments. You can target specific subreddits or search across all of Reddit.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reddit search form
        with st.form("reddit_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                subreddit = st.text_input(
                    "Subreddit",
                    value="all",
                    help="Enter a subreddit name or 'all' for all subreddits"
                )
            
            with col2:
                query = st.text_input(
                    "Search Query",
                    help="Enter a search query for Reddit"
                )
            
            col3, col4 = st.columns(2)
            
            with col3:
                limit = st.slider(
                    "Number of Posts",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Maximum number of posts to retrieve and analyze"
                )
            
            with col4:
                time_filter = st.select_slider(
                    "Time Filter",
                    options=["hour", "day", "week", "month", "year", "all"],
                    value="month",
                    help="Time period for posts"
                )
            
            include_comments = st.checkbox(
                "Include Comments",
                value=True,
                help="Include comments in the analysis"
            )
            
            submit_button = st.form_submit_button("Analyze Reddit Content", type="primary")
            
            if submit_button and query:
                analyze_reddit_data(
                    subreddit, 
                    query, 
                    limit, 
                    time_filter, 
                    include_comments,
                    selected_model
                )
    
    elif data_source == "CSV Upload":
        st.markdown('<h2 class="subheader">Analyze CSV Data</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p>Upload a CSV file containing text data to analyze. The file should have at least one column with text content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded CSV with {len(data)} rows and {len(data.columns)} columns.")
                
                # Show a preview of the data
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Let the user select the text column
                text_columns = list(data.columns)
                selected_column = st.selectbox(
                    "Select Text Column",
                    options=text_columns,
                    index=0 if text_columns else None,
                    help="Select the column containing the text to analyze"
                )
                
                if st.button("Analyze Data", type="primary"):
                    analyze_csv_data(data, selected_column, selected_model)
            
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
                logger.error(f"Error loading CSV file: {e}")
                logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()