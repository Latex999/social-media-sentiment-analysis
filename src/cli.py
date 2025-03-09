"""
Command-line interface for the Social Media Sentiment Analysis tool.

This module provides a CLI for analyzing sentiment in social media posts
from Twitter and Reddit, as well as from CSV files.

Example usage:
    # Analyze Twitter posts
    python -m src.cli --source twitter --query "#AI" --limit 100 --model roberta
    
    # Analyze Reddit posts
    python -m src.cli --source reddit --subreddit "MachineLearning" --query "sentiment" --limit 50
    
    # Analyze a CSV file
    python -m src.cli --source csv --file "data/posts.csv" --text-column "content"
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, List, Union

import pandas as pd
from tqdm import tqdm

from src.api import TwitterClient, RedditClient
from src.models import SentimentAnalyzer, SentimentResult, SentimentResultSet
from src.utils.config import config, RESULTS_DIR, DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)


def analyze_twitter(
    query: str,
    limit: int,
    model_name: str,
    output_format: str,
    output_file: Optional[str] = None
) -> None:
    """Analyze Twitter data from the command line.
    
    Args:
        query: Search query for Twitter
        limit: Maximum number of tweets to analyze
        model_name: The model to use for analysis
        output_format: Output format (json, csv, or both)
        output_file: Custom output file path (optional)
    """
    logger.info(f"Analyzing Twitter data for query: {query}")
    
    try:
        # Initialize Twitter client
        twitter_client = TwitterClient()
        
        # Fetch tweets
        logger.info(f"Fetching up to {limit} tweets...")
        tweets = twitter_client.search(query=query, limit=limit)
        
        if not tweets:
            logger.warning(f"No tweets found for query: {query}")
            return
        
        logger.info(f"Retrieved {len(tweets)} tweets.")
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer(model_name=model_name)
        
        # Analyze tweets
        logger.info(f"Analyzing sentiment with {model_name}...")
        with tqdm(total=len(tweets), desc="Analyzing", unit="tweets") as pbar:
            results = analyzer.analyze_batch(
                tweets, 
                text_field='text',
                show_progress=False
            )
            pbar.update(len(tweets))
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_query = query.replace(" ", "_").replace("#", "").replace("@", "")[:30]
            output_file = os.path.join(RESULTS_DIR, f"twitter_{sanitized_query}_{timestamp}")
        
        # Save results
        save_results(results, output_file, output_format)
        
        # Display summary
        display_summary(results)
    
    except Exception as e:
        logger.error(f"Error analyzing Twitter data: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


def analyze_reddit(
    query: str,
    subreddit: Optional[str],
    limit: int,
    time_filter: str,
    include_comments: bool,
    model_name: str,
    output_format: str,
    output_file: Optional[str] = None
) -> None:
    """Analyze Reddit data from the command line.
    
    Args:
        query: Search query for Reddit
        subreddit: Subreddit to search in (or None for all subreddits)
        limit: Maximum number of posts to analyze
        time_filter: Time filter for posts
        include_comments: Whether to include comments in analysis
        model_name: The model to use for analysis
        output_format: Output format (json, csv, or both)
        output_file: Custom output file path (optional)
    """
    logger.info(f"Analyzing Reddit data for query: {query}")
    
    try:
        # Initialize Reddit client
        reddit_client = RedditClient()
        
        # Fetch Reddit posts
        logger.info(f"Fetching up to {limit} Reddit posts...")
        posts = reddit_client.search(
            query=query, 
            subreddit=subreddit, 
            limit=limit, 
            time_filter=time_filter,
            include_comments=include_comments,
            comments_limit=10 if include_comments else 0
        )
        
        if not posts:
            logger.warning(f"No Reddit posts found for query: {query}")
            return
        
        logger.info(f"Retrieved {len(posts)} Reddit posts.")
        
        # Prepare items to analyze
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
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer(model_name=model_name)
        
        # Analyze items
        logger.info(f"Analyzing sentiment with {model_name}...")
        with tqdm(total=len(items_to_analyze), desc="Analyzing", unit="items") as pbar:
            results = analyzer.analyze_batch(
                items_to_analyze, 
                text_field='text',
                show_progress=False
            )
            pbar.update(len(items_to_analyze))
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub_part = f"{subreddit}_" if subreddit else ""
            sanitized_query = query.replace(" ", "_").replace("#", "").replace("@", "")[:30]
            output_file = os.path.join(RESULTS_DIR, f"reddit_{sub_part}{sanitized_query}_{timestamp}")
        
        # Save results
        save_results(results, output_file, output_format)
        
        # Display summary
        logger.info(f"Analyzed {post_count} posts and {comment_count} comments.")
        display_summary(results)
    
    except Exception as e:
        logger.error(f"Error analyzing Reddit data: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


def analyze_csv(
    file_path: str,
    text_column: str,
    model_name: str,
    output_format: str,
    output_file: Optional[str] = None
) -> None:
    """Analyze text data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        text_column: Column name containing the text to analyze
        model_name: The model to use for analysis
        output_format: Output format (json, csv, or both)
        output_file: Custom output file path (optional)
    """
    logger.info(f"Analyzing CSV data from {file_path}")
    
    try:
        # Read CSV file
        data = pd.read_csv(file_path)
        
        # Validate data
        if text_column not in data.columns:
            logger.error(f"Column '{text_column}' not found in the CSV file.")
            sys.exit(1)
        
        # Remove rows with empty text
        data = data[data[text_column].notna()]
        data = data[data[text_column].str.strip() != ""]
        
        if len(data) == 0:
            logger.error("No valid text data found in the selected column.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(data)} records from CSV.")
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer(model_name=model_name)
        
        # Convert DataFrame to list of dicts for analysis
        items_to_analyze = data.to_dict('records')
        
        # Analyze items
        logger.info(f"Analyzing sentiment with {model_name}...")
        with tqdm(total=len(items_to_analyze), desc="Analyzing", unit="items") as pbar:
            results = analyzer.analyze_batch(
                items_to_analyze, 
                text_field=text_column,
                show_progress=False
            )
            pbar.update(len(items_to_analyze))
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(RESULTS_DIR, f"csv_{basename}_{timestamp}")
        
        # Save results
        save_results(results, output_file, output_format)
        
        # For CSV, also save with original data
        if output_format in ['csv', 'both']:
            # Get results as DataFrame
            results_df = results.to_dataframe()
            
            # Join with original data (excluding the text column to avoid duplication)
            original_data = data.drop(columns=[text_column])
            combined_data = pd.concat([results_df, original_data], axis=1)
            
            # Save combined data
            combined_file = f"{output_file}_with_original.csv"
            combined_data.to_csv(combined_file, index=False)
            logger.info(f"Combined results with original data saved to: {combined_file}")
        
        # Display summary
        display_summary(results)
    
    except Exception as e:
        logger.error(f"Error analyzing CSV data: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


def save_results(
    results: SentimentResultSet,
    output_file: str,
    output_format: str
) -> None:
    """Save analysis results to file(s).
    
    Args:
        results: SentimentResultSet containing analysis results
        output_file: Base path for output file(s)
        output_format: Output format (json, csv, or both)
    """
    try:
        if output_format in ['json', 'both']:
            json_file = f"{output_file}.json"
            results.save_to_json(json_file)
            logger.info(f"Results saved as JSON to: {json_file}")
        
        if output_format in ['csv', 'both']:
            csv_file = f"{output_file}.csv"
            results.save_to_csv(csv_file)
            logger.info(f"Results saved as CSV to: {csv_file}")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        logger.debug(traceback.format_exc())


def display_summary(results: SentimentResultSet) -> None:
    """Display a summary of the analysis results.
    
    Args:
        results: SentimentResultSet containing analysis results
    """
    # Get sentiment distribution
    sentiment_dist = results.get_sentiment_distribution()
    total_items = sum(sentiment_dist.values())
    
    # Calculate percentages
    positive_pct = sentiment_dist['positive'] / total_items * 100 if total_items else 0
    neutral_pct = sentiment_dist['neutral'] / total_items * 100 if total_items else 0
    negative_pct = sentiment_dist['negative'] / total_items * 100 if total_items else 0
    
    # Get average sentiment
    avg_sentiment = results.get_average_sentiment()
    
    # Display summary
    print("\n" + "=" * 40)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Total items analyzed: {total_items}")
    print(f"Average sentiment score: {avg_sentiment:.2f}")
    print("\nSentiment Distribution:")
    print(f"  Positive: {sentiment_dist['positive']} ({positive_pct:.1f}%)")
    print(f"  Neutral:  {sentiment_dist['neutral']} ({neutral_pct:.1f}%)")
    print(f"  Negative: {sentiment_dist['negative']} ({negative_pct:.1f}%)")
    print("=" * 40 + "\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Social Media Sentiment Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument(
        "--source",
        choices=["twitter", "reddit", "csv"],
        required=True,
        help="Source of the data to analyze"
    )
    
    parser.add_argument(
        "--model",
        choices=["vader", "textblob", "distilbert", "roberta"],
        default="roberta",
        help="Sentiment analysis model to use"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Format for saving results"
    )
    
    parser.add_argument(
        "--output-file",
        help="Custom output file path (without extension)"
    )
    
    # Twitter-specific arguments
    twitter_group = parser.add_argument_group("Twitter options")
    twitter_group.add_argument(
        "--query",
        help="Search query for Twitter or Reddit"
    )
    
    twitter_group.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of items to retrieve"
    )
    
    # Reddit-specific arguments
    reddit_group = parser.add_argument_group("Reddit options")
    reddit_group.add_argument(
        "--subreddit",
        help="Subreddit to search in (default: search all)"
    )
    
    reddit_group.add_argument(
        "--time-filter",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="month",
        help="Time filter for Reddit posts"
    )
    
    reddit_group.add_argument(
        "--include-comments",
        action="store_true",
        help="Include comments in Reddit analysis"
    )
    
    # CSV-specific arguments
    csv_group = parser.add_argument_group("CSV options")
    csv_group.add_argument(
        "--file",
        help="Path to CSV file containing text data"
    )
    
    csv_group.add_argument(
        "--text-column",
        help="Column name containing the text to analyze"
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on selected source
    if args.source == "twitter" and not args.query:
        parser.error("--query is required for Twitter analysis")
    
    elif args.source == "reddit":
        if not args.query:
            parser.error("--query is required for Reddit analysis")
    
    elif args.source == "csv":
        if not args.file:
            parser.error("--file is required for CSV analysis")
        if not args.text_column:
            parser.error("--text-column is required for CSV analysis")
    
    return args


def main():
    """Main function for the CLI."""
    args = parse_arguments()
    
    # Process based on selected source
    if args.source == "twitter":
        analyze_twitter(
            query=args.query,
            limit=args.limit,
            model_name=args.model,
            output_format=args.output_format,
            output_file=args.output_file
        )
    
    elif args.source == "reddit":
        analyze_reddit(
            query=args.query,
            subreddit=args.subreddit,
            limit=args.limit,
            time_filter=args.time_filter,
            include_comments=args.include_comments,
            model_name=args.model,
            output_format=args.output_format,
            output_file=args.output_file
        )
    
    elif args.source == "csv":
        analyze_csv(
            file_path=args.file,
            text_column=args.text_column,
            model_name=args.model,
            output_format=args.output_format,
            output_file=args.output_file
        )


if __name__ == "__main__":
    main()