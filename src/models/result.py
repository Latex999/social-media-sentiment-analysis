"""Classes for sentiment analysis results."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter


@dataclass
class SentimentResult:
    """Class representing a single sentiment analysis result.
    
    Attributes:
        text: Original text that was analyzed
        sentiment: The sentiment score (typically -1 to 1 or 0 to 1)
        sentiment_class: Sentiment classification (negative, neutral, positive)
        confidence: Confidence score for the sentiment classification
        model: The model used for analysis
        metadata: Any additional metadata about the text or analysis
        timestamp: When the analysis was performed
    """
    
    text: str
    sentiment: float
    sentiment_class: str  # 'negative', 'neutral', 'positive'
    confidence: float
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and process after initialization."""
        # Ensure sentiment class is one of the valid options
        if self.sentiment_class not in ['negative', 'neutral', 'positive']:
            raise ValueError(f"Invalid sentiment class: {self.sentiment_class}")
        
        # Add source platform to metadata if available
        if 'platform' not in self.metadata and 'type' not in self.metadata:
            if isinstance(self.metadata, dict) and 'platform' in self.metadata:
                self.metadata['source'] = self.metadata['platform']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the sentiment result
        """
        return {
            'text': self.text,
            'sentiment': self.sentiment,
            'sentiment_class': self.sentiment_class,
            'confidence': self.confidence,
            'model': self.model,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert the result to a JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentResult':
        """Create a SentimentResult from a dictionary.
        
        Args:
            data: Dictionary containing sentiment result data
            
        Returns:
            New SentimentResult instance
        """
        # Convert timestamp string to datetime if needed
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SentimentResult':
        """Create a SentimentResult from a JSON string.
        
        Args:
            json_str: JSON string containing sentiment result data
            
        Returns:
            New SentimentResult instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class SentimentResultSet:
    """Class representing a set of sentiment analysis results.
    
    This class provides methods to analyze and visualize a collection
    of sentiment results.
    
    Attributes:
        results: List of SentimentResult objects
        model: The model used for analysis (if all results use the same model)
        query: The query used to generate these results (if applicable)
        metadata: Any additional metadata about the result set
    """
    
    def __init__(
        self,
        results: List[SentimentResult],
        model: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a SentimentResultSet.
        
        Args:
            results: List of SentimentResult objects
            model: The model used for analysis (if all results use the same model)
            query: The query used to generate these results (if applicable)
            metadata: Any additional metadata about the result set
        """
        self.results = results
        self.model = model or (results[0].model if results else None)
        self.query = query
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        """Get the number of results in the set.
        
        Returns:
            Number of results
        """
        return len(self.results)
    
    def __getitem__(self, index) -> Union[SentimentResult, 'SentimentResultSet']:
        """Get a result or slice of results.
        
        Args:
            index: Index or slice
            
        Returns:
            Single SentimentResult or a new SentimentResultSet for a slice
        """
        if isinstance(index, slice):
            return SentimentResultSet(
                results=self.results[index],
                model=self.model,
                query=self.query,
                metadata=self.metadata
            )
        return self.results[index]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the results to a pandas DataFrame.
        
        Returns:
            DataFrame with all results
        """
        data = []
        for result in self.results:
            row = result.to_dict()
            # Add any metadata as columns
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        row[key] = value
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Get the distribution of sentiment classes.
        
        Returns:
            Dictionary with counts for each sentiment class
        """
        counter = Counter([result.sentiment_class for result in self.results])
        return {
            'positive': counter.get('positive', 0),
            'neutral': counter.get('neutral', 0),
            'negative': counter.get('negative', 0)
        }
    
    def get_average_sentiment(self) -> float:
        """Get the average sentiment score.
        
        Returns:
            Average sentiment score
        """
        if not self.results:
            return 0.0
        
        return sum(result.sentiment for result in self.results) / len(self.results)
    
    def get_sentiment_by_attribute(self, attribute: str) -> Dict[str, Any]:
        """Group sentiment by a metadata attribute.
        
        Args:
            attribute: Metadata attribute to group by
            
        Returns:
            Dictionary mapping attribute values to average sentiment
        """
        if not self.results:
            return {}
        
        # Group results by the attribute
        groups = {}
        for result in self.results:
            if attribute in result.metadata:
                value = result.metadata[attribute]
                if value not in groups:
                    groups[value] = []
                groups[value].append(result.sentiment)
        
        # Calculate average sentiment for each group
        return {
            value: sum(sentiments) / len(sentiments)
            for value, sentiments in groups.items()
        }
    
    def plot_sentiment_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot the distribution of sentiment classes.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        distribution = self.get_sentiment_distribution()
        labels = ['Positive', 'Neutral', 'Negative']
        values = [distribution['positive'], distribution['neutral'], distribution['negative']]
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors)
        
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
        
        # Add percentage to the bars
        total = sum(values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / total * 100 if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{percentage:.1f}%",
                ha='center',
                color='white',
                fontweight='bold'
            )
        
        # Add labels and title
        title = f"Sentiment Distribution"
        if self.query:
            title += f" for '{self.query}'"
        if self.model:
            title += f" (Model: {self.model})"
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sentiment_over_time(
        self, 
        time_field: str = 'timestamp',
        interval: str = 'D',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot sentiment trends over time.
        
        Args:
            time_field: Field to use as time (default: 'timestamp')
            interval: Time interval for grouping ('D' for day, 'H' for hour, etc.)
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        df = self.to_dataframe()
        
        # Ensure timestamp is datetime
        if time_field == 'timestamp' and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_col = df['timestamp']
        elif time_field in df.columns:
            time_col = pd.to_datetime(df[time_field])
        else:
            raise ValueError(f"Time field '{time_field}' not found in data")
        
        # Group by time interval and calculate average sentiment
        df['time_group'] = time_col.dt.floor(interval)
        time_groups = df.groupby('time_group')
        
        sentiment_by_time = time_groups['sentiment'].mean()
        counts_by_time = time_groups.size()
        
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
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
        
        # Set labels and title
        title = f"Sentiment Trend Over Time"
        if self.query:
            title += f" for '{self.query}'"
        if self.model:
            title += f" (Model: {self.model})"
        
        ax1.set_title(title, fontsize=14, pad=20)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Average Sentiment', fontsize=12)
        ax2.set_ylabel('Number of Posts', fontsize=12)
        
        # Set sentiment y-axis limits to -1 to 1
        ax1.set_ylim(-1, 1)
        
        # Add horizontal line at 0
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add labels for positive and negative regions
        ax1.text(
            sentiment_by_time.index[0],
            0.75,
            'Positive',
            fontsize=10,
            color='#2ecc71'
        )
        ax1.text(
            sentiment_by_time.index[0],
            -0.75,
            'Negative',
            fontsize=10,
            color='#e74c3c'
        )
        
        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax1.legend(lines + bars, labels + bar_labels, loc='upper left')
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_to_csv(self, file_path: str) -> None:
        """Save results to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
        """
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)
    
    def save_to_json(self, file_path: str) -> None:
        """Save results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        results = [result.to_dict() for result in self.results]
        with open(file_path, 'w') as f:
            json.dump({
                'results': results,
                'model': self.model,
                'query': self.query,
                'metadata': self.metadata,
                'count': len(self.results)
            }, f, indent=2)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'SentimentResultSet':
        """Load results from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            New SentimentResultSet instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = [SentimentResult.from_dict(result) for result in data['results']]
        
        return cls(
            results=results,
            model=data.get('model'),
            query=data.get('query'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_csv(cls, file_path: str, model: Optional[str] = None) -> 'SentimentResultSet':
        """Load results from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            model: Model name (if not in CSV)
            
        Returns:
            New SentimentResultSet instance
        """
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to SentimentResult objects
        results = []
        for _, row in df.iterrows():
            data = row.to_dict()
            
            # Extract required fields
            text = data.pop('text', '')
            sentiment = data.pop('sentiment', 0.0)
            sentiment_class = data.pop('sentiment_class', 'neutral')
            confidence = data.pop('confidence', 0.0)
            result_model = data.pop('model', model)
            
            # Extract timestamp if present
            timestamp = data.pop('timestamp', None)
            if timestamp:
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError):
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Remaining fields are metadata
            result = SentimentResult(
                text=text,
                sentiment=sentiment,
                sentiment_class=sentiment_class,
                confidence=confidence,
                model=result_model or model,
                metadata=data,
                timestamp=timestamp
            )
            results.append(result)
        
        return cls(results=results, model=model)