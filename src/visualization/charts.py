"""Visualization charts for sentiment analysis results."""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS

from src.models.result import SentimentResultSet
from src.utils.config import config, RESULTS_DIR

# Set up logging
logger = logging.getLogger(__name__)

# Set default styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


def create_sentiment_distribution_chart(
    data: Union[SentimentResultSet, Dict[str, int], pd.DataFrame],
    title: Optional[str] = None,
    use_plotly: bool = True,
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    width: int = 800,
    height: int = 500,
) -> Union[go.Figure, plt.Figure]:
    """Create a sentiment distribution chart.
    
    Args:
        data: Sentiment results, counts dictionary, or pandas DataFrame
        title: Chart title
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        colors: Custom colors for sentiment categories
        save_path: Path to save the chart
        width: Chart width in pixels
        height: Chart height in pixels
        
    Returns:
        Plotly or Matplotlib figure
    """
    # Default colors for sentiment categories
    if colors is None:
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']  # green, gray, red
    
    # Process different input types
    if isinstance(data, SentimentResultSet):
        distribution = data.get_sentiment_distribution()
        if title is None and data.query:
            title = f"Sentiment Distribution for '{data.query}'"
    elif isinstance(data, dict):
        distribution = data
    elif isinstance(data, pd.DataFrame):
        if 'sentiment_class' in data.columns:
            distribution = data['sentiment_class'].value_counts().to_dict()
            # Ensure we have all three categories
            for category in ['positive', 'neutral', 'negative']:
                if category not in distribution:
                    distribution[category] = 0
        else:
            logger.error("DataFrame must contain 'sentiment_class' column")
            distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
    else:
        logger.error(f"Unsupported data type: {type(data)}")
        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    # Prepare data for plotting
    categories = ['Positive', 'Neutral', 'Negative']
    values = [distribution.get('positive', 0), distribution.get('neutral', 0), distribution.get('negative', 0)]
    total = sum(values)
    percentages = [value / total * 100 if total > 0 else 0 for value in values]
    
    # Default title
    if title is None:
        title = "Sentiment Distribution"
    
    # Create chart with plotly or matplotlib
    if use_plotly:
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            text=[f"{p:.1f}%" for p in percentages],
            textposition='auto',
            marker_color=colors,
            hovertemplate='%{x}: %{y} posts (%{text})<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Sentiment",
            yaxis_title="Number of Posts",
            template="plotly_white",
            width=width,
            height=height,
            xaxis={'categoryorder': 'array', 'categoryarray': categories},
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            )
        )
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # Create bar chart
        bars = ax.bar(categories, values, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha='center',
                fontweight='bold'
            )
        
        # Add percentage inside the bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    f"{percentages[i]:.1f}%",
                    ha='center',
                    color='white',
                    fontweight='bold'
                )
        
        # Set labels and title
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('Number of Posts', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2 if values else 10)
        
        # Remove spines and add grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig


def create_sentiment_over_time_chart(
    data: Union[SentimentResultSet, pd.DataFrame],
    time_field: str = 'timestamp',
    interval: str = 'D',
    use_plotly: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    width: int = 1000,
    height: int = 500,
) -> Union[go.Figure, plt.Figure]:
    """Create a chart showing sentiment trends over time.
    
    Args:
        data: Sentiment results or pandas DataFrame
        time_field: Field to use as time
        interval: Time interval for grouping ('D' for day, 'H' for hour, etc.)
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        title: Chart title
        save_path: Path to save the chart
        width: Chart width in pixels
        height: Chart height in pixels
        
    Returns:
        Plotly or Matplotlib figure
    """
    # Convert to DataFrame if needed
    if isinstance(data, SentimentResultSet):
        df = data.to_dataframe()
        if title is None and data.query:
            title = f"Sentiment Trend for '{data.query}'"
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        logger.error(f"Unsupported data type: {type(data)}")
        return None
    
    # Ensure timestamp is datetime
    if time_field in df.columns:
        df[time_field] = pd.to_datetime(df[time_field])
    else:
        logger.error(f"Time field '{time_field}' not found in data")
        return None
    
    # Group by time interval
    df['time_group'] = df[time_field].dt.floor(interval)
    
    # Calculate aggregates by time
    time_groups = df.groupby('time_group')
    sentiment_by_time = time_groups['sentiment'].mean()
    counts_by_time = time_groups.size()
    
    # Calculate percentage of each sentiment class by time
    if 'sentiment_class' in df.columns:
        sentiment_class_by_time = pd.DataFrame({
            'positive': time_groups.apply(lambda x: (x['sentiment_class'] == 'positive').mean() * 100),
            'neutral': time_groups.apply(lambda x: (x['sentiment_class'] == 'neutral').mean() * 100),
            'negative': time_groups.apply(lambda x: (x['sentiment_class'] == 'negative').mean() * 100)
        })
    else:
        sentiment_class_by_time = None
    
    # Default title
    if title is None:
        title = "Sentiment Trend Over Time"
    
    # Create chart with plotly or matplotlib
    if use_plotly:
        # Create subplots: one for sentiment, one for volume
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=sentiment_by_time.index,
                y=sentiment_by_time.values,
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color='#3498db', width=3),
                hovertemplate='%{x}<br>Sentiment: %{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=counts_by_time.index,
                y=counts_by_time.values,
                name='Post Volume',
                marker_color='rgba(149, 165, 166, 0.5)',
                hovertemplate='%{x}<br>Posts: %{y}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add horizontal line at zero
        fig.add_shape(
            type="line",
            x0=min(sentiment_by_time.index),
            y0=0,
            x1=max(sentiment_by_time.index),
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            xref="x",
            yref="y"
        )
        
        # Add sentiment class percentages if available
        if sentiment_class_by_time is not None:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_class_by_time.index,
                    y=sentiment_class_by_time['positive'].values,
                    mode='lines',
                    name='Positive %',
                    line=dict(color='rgba(46, 204, 113, 0.7)', width=1.5, dash='dot'),
                    hovertemplate='%{x}<br>Positive: %{y:.1f}%<extra></extra>'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sentiment_class_by_time.index,
                    y=sentiment_class_by_time['negative'].values,
                    mode='lines',
                    name='Negative %',
                    line=dict(color='rgba(231, 76, 60, 0.7)', width=1.5, dash='dot'),
                    hovertemplate='%{x}<br>Negative: %{y:.1f}%<extra></extra>'
                ),
                secondary_y=False
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            template="plotly_white",
            width=width,
            height=height,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Sentiment Score", range=[-1, 1], secondary_y=False)
        fig.update_yaxes(title_text="Number of Posts", secondary_y=True)
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    else:
        # Matplotlib version
        fig, ax1 = plt.subplots(figsize=(width/100, height/100))
        
        # Plot average sentiment
        line = ax1.plot(
            sentiment_by_time.index,
            sentiment_by_time.values,
            marker='o',
            linestyle='-',
            color='#3498db',
            label='Average Sentiment'
        )
        
        # Add shaded area for neutral zone
        ax1.axhspan(-0.05, 0.05, alpha=0.1, color='gray', label='Neutral Zone')
        
        # Add count as bars on secondary y-axis
        ax2 = ax1.twinx()
        bars = ax2.bar(
            counts_by_time.index,
            counts_by_time.values,
            alpha=0.3,
            color='#95a5a6',
            label='Post Count'
        )
        
        # Plot sentiment class percentages if available
        if sentiment_class_by_time is not None:
            ax1.plot(
                sentiment_class_by_time.index,
                sentiment_class_by_time['positive'].values / 100,
                linestyle='--',
                color='#2ecc71',
                alpha=0.7,
                label='Positive %'
            )
            
            ax1.plot(
                sentiment_class_by_time.index,
                sentiment_class_by_time['negative'].values / 100,
                linestyle='--',
                color='#e74c3c',
                alpha=0.7,
                label='Negative %'
            )
        
        # Set labels and title
        ax1.set_title(title, fontsize=14, pad=20)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Sentiment Score', fontsize=12)
        ax2.set_ylabel('Number of Posts', fontsize=12)
        
        # Set sentiment y-axis limits
        ax1.set_ylim(-1, 1)
        
        # Add horizontal line at 0
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        if len(sentiment_by_time) > 10:
            fig.autofmt_xdate()
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig


def create_sentiment_comparison_chart(
    datasets: Dict[str, Union[SentimentResultSet, Dict[str, int], pd.DataFrame]],
    title: str = "Sentiment Comparison",
    use_plotly: bool = True,
    normalize: bool = True,
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    width: int = 800,
    height: int = 500,
) -> Union[go.Figure, plt.Figure]:
    """Create a chart comparing sentiment distributions across different datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to sentiment data
        title: Chart title
        use_plotly: Whether to use Plotly (interactive) or Matplotlib (static)
        normalize: Whether to normalize values as percentages
        colors: Custom colors for sentiment categories
        save_path: Path to save the chart
        width: Chart width in pixels
        height: Chart height in pixels
        
    Returns:
        Plotly or Matplotlib figure
    """
    # Default colors for sentiment categories
    if colors is None:
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']  # green, gray, red
    
    # Process each dataset to get sentiment distributions
    processed_data = {}
    
    for name, data in datasets.items():
        if isinstance(data, SentimentResultSet):
            distribution = data.get_sentiment_distribution()
        elif isinstance(data, dict):
            distribution = data
        elif isinstance(data, pd.DataFrame):
            if 'sentiment_class' in data.columns:
                distribution = data['sentiment_class'].value_counts().to_dict()
            else:
                logger.error(f"DataFrame for '{name}' must contain 'sentiment_class' column")
                distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        else:
            logger.error(f"Unsupported data type for '{name}': {type(data)}")
            distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        # Ensure we have all three categories
        for category in ['positive', 'neutral', 'negative']:
            if category not in distribution:
                distribution[category] = 0
        
        processed_data[name] = distribution
    
    # Prepare data for plotting
    categories = ['positive', 'neutral', 'negative']
    dataset_names = list(processed_data.keys())
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(
        {name: [data[cat] for cat in categories] for name, data in processed_data.items()},
        index=categories
    )
    
    # Normalize if requested
    if normalize:
        df = df.div(df.sum(axis=0)) * 100
    
    # Create chart with plotly or matplotlib
    if use_plotly:
        # Transpose to get dataset names as x-axis
        df_plot = df.T
        
        # Create grouped bar chart
        fig = go.Figure()
        
        # Add each sentiment category as a bar
        for i, category in enumerate(['positive', 'neutral', 'negative']):
            fig.add_trace(go.Bar(
                x=dataset_names,
                y=df_plot[category],
                name=category.capitalize(),
                marker_color=colors[i],
                hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>' if normalize else '%{x}<br>%{y}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Dataset",
            yaxis_title="Percentage" if normalize else "Count",
            template="plotly_white",
            width=width,
            height=height,
            barmode='group',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    else:
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # Number of groups and bar width
        n_datasets = len(dataset_names)
        bar_width = 0.25
        
        # Positions for grouped bars
        x = np.arange(len(categories))
        
        # Plot each dataset as a group of bars
        for i, name in enumerate(dataset_names):
            position = x + i * bar_width - (n_datasets - 1) * bar_width / 2
            bars = ax.bar(
                position,
                df[name],
                bar_width,
                label=name,
                alpha=0.8
            )
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}%" if normalize else f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        # Set labels and title
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Percentage' if normalize else 'Count', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in categories])
        
        # Add legend
        ax.legend()
        
        # Remove spines and add grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig


def create_wordcloud(
    texts: Union[List[str], pd.DataFrame],
    text_column: str = 'text',
    sentiment_column: Optional[str] = 'sentiment_class',
    sentiment_filter: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 500,
    max_words: int = 200,
    background_color: str = 'white',
    mask: Optional[np.ndarray] = None,
    contour_width: int = 1,
    contour_color: str = 'steelblue',
    colormap: str = 'viridis',
    additional_stopwords: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a word cloud visualization from a collection of texts.
    
    Args:
        texts: List of strings or DataFrame containing text data
        text_column: Column containing text if DataFrame is provided
        sentiment_column: Column containing sentiment if filtering by sentiment
        sentiment_filter: Filter for specific sentiment (positive, neutral, negative)
        title: Word cloud title
        width: Width in pixels
        height: Height in pixels
        max_words: Maximum number of words to include
        background_color: Background color
        mask: Optional mask array for the word cloud shape
        contour_width: Width of mask contour
        contour_color: Color of mask contour
        colormap: Matplotlib colormap name
        additional_stopwords: Additional stopwords to exclude
        save_path: Path to save the word cloud image
        
    Returns:
        Matplotlib figure
    """
    # Get the texts to process
    if isinstance(texts, pd.DataFrame):
        if text_column not in texts.columns:
            logger.error(f"Text column '{text_column}' not found in DataFrame")
            return None
        
        # Filter by sentiment if requested
        if sentiment_filter and sentiment_column in texts.columns:
            filtered_df = texts[texts[sentiment_column] == sentiment_filter]
            text_list = filtered_df[text_column].tolist()
            
            # Update title if not provided
            if title is None:
                title = f"Word Cloud for {sentiment_filter.capitalize()} Sentiment"
        else:
            text_list = texts[text_column].tolist()
    else:
        text_list = texts
    
    # Join all texts
    text = ' '.join(text_list)
    
    # If text is empty, return None
    if not text:
        logger.error("No text to process for word cloud")
        return None
    
    # Setup stopwords
    stopwords = set(STOPWORDS)
    if additional_stopwords:
        stopwords.update(additional_stopwords)
    
    # Create the word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        stopwords=stopwords,
        mask=mask,
        contour_width=contour_width,
        contour_color=contour_color,
        colormap=colormap,
        regexp=r"\w[\w']+",  # Word pattern including apostrophes
        collocations=True,  # Include collocations (bigrams)
        normalize_plurals=True
    ).generate(text)
    
    # Default title
    if title is None:
        title = "Word Cloud"
    
    # Create figure and display the cloud
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        wordcloud.to_file(save_path)
        logger.info(f"Word cloud saved to {save_path}")
    
    return fig