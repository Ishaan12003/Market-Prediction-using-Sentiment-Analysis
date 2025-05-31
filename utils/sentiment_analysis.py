import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils.data_collection import clean_text
import streamlit as st
from tqdm import tqdm

def analyze_sentiment(data_df):
    """
    Analyze sentiment in the collected data.

    Args:
        data_df (pd.DataFrame): DataFrame with text data

    Returns:
        pd.DataFrame: DataFrame with sentiment scores added
    """
    if data_df.empty:
        st.warning("Empty DataFrame provided for sentiment analysis.")
        return data_df

    st.info(f"Analyzing sentiment for {len(data_df)} records...")

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Custom financial keywords to improve VADER lexicon
    fin_pos_words = {
        'bullish': 4.0, 'uptrend': 3.0, 'growth': 2.5, 'profit': 2.0, 'beat': 2.5,
        'exceeded': 3.0, 'outperform': 3.0, 'buy': 2.0, 'undervalued': 3.0,
        'momentum': 2.0, 'rally': 3.0, 'recovery': 2.0, 'upgrade': 3.0
    }

    fin_neg_words = {
        'bearish': -4.0, 'downtrend': -3.0, 'decline': -2.5, 'loss': -2.0, 'miss': -2.5,
        'missed': -3.0, 'underperform': -3.0, 'sell': -2.0, 'overvalued': -3.0,
        'slump': -3.0, 'crash': -4.0, 'recession': -3.0, 'downgrade': -3.0
    }

    # Add financial words to lexicon
    for word, score in fin_pos_words.items():
        analyzer.lexicon[word] = score

    for word, score in fin_neg_words.items():
        analyzer.lexicon[word] = score

    # Make a copy of the DataFrame
    result_df = data_df.copy()

    # Initialize sentiment columns
    result_df['sentiment_score'] = 0.0
    result_df['sentiment_label'] = 'neutral'
    result_df['sentiment_positive'] = 0.0
    result_df['sentiment_negative'] = 0.0
    result_df['sentiment_neutral'] = 0.0

    # Set up progress bar
    progress_bar = st.progress(0)
    
    # Process each record
    for idx, row in enumerate(result_df.iterrows()):
        i, row_data = row
        
        # Update progress
        progress_bar.progress((idx + 1) / len(result_df))
        
        # Get text to analyze (prefer full_text, fall back to content or title)
        text = row_data.get('full_text', row_data.get('content', row_data.get('title', '')))

        # Clean text
        cleaned_text = clean_text(text)

        if cleaned_text:
            # Get sentiment scores
            sentiment = analyzer.polarity_scores(cleaned_text)

            # Update DataFrame
            result_df.at[i, 'sentiment_score'] = sentiment['compound']
            result_df.at[i, 'sentiment_positive'] = sentiment['pos']
            result_df.at[i, 'sentiment_negative'] = sentiment['neg']
            result_df.at[i, 'sentiment_neutral'] = sentiment['neu']

            # Assign sentiment label
            if sentiment['compound'] >= 0.05:
                result_df.at[i, 'sentiment_label'] = 'positive'
            elif sentiment['compound'] <= -0.05:
                result_df.at[i, 'sentiment_label'] = 'negative'
            else:
                result_df.at[i, 'sentiment_label'] = 'neutral'

    # Clear progress bar
    progress_bar.empty()
    
    st.success("Sentiment analysis completed.")
    return result_df

def aggregate_daily_sentiment(sentiment_df):
    """
    Aggregate sentiment scores by day.

    Args:
        sentiment_df (pd.DataFrame): DataFrame with sentiment scores

    Returns:
        pd.DataFrame: DataFrame with daily sentiment aggregations
    """
    if sentiment_df.empty:
        st.warning("Empty DataFrame provided for aggregation.")
        return pd.DataFrame()

    # Ensure timestamp is datetime
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])

    # Extract date
    sentiment_df['date'] = sentiment_df['timestamp'].dt.date
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    st.info("Aggregating sentiment scores by day...")

    # Group by date and aggregate
    daily_sentiment = sentiment_df.groupby('date').agg(
        # Count by type
        total_count=('id', 'count'),
        reddit_count=('type', lambda x: sum(t in ['submission', 'comment'] for t in x)),
        news_count=('type', lambda x: sum(t == 'news' for t in x)),

        # Sentiment aggregations
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_median=('sentiment_score', 'median'),
        sentiment_std=('sentiment_score', lambda x: x.std() if len(x) > 1 else 0),
        sentiment_min=('sentiment_score', 'min'),
        sentiment_max=('sentiment_score', 'max'),

        # Average of positive/negative/neutral components
        positive_mean=('sentiment_positive', 'mean'),
        negative_mean=('sentiment_negative', 'mean'),
        neutral_mean=('sentiment_neutral', 'mean'),

        # Count by sentiment label
        positive_count=('sentiment_label', lambda x: sum(s == 'positive' for s in x)),
        negative_count=('sentiment_label', lambda x: sum(s == 'negative' for s in x)),
        neutral_count=('sentiment_label', lambda x: sum(s == 'neutral' for s in x)),

        # Calculate proportions
        positive_ratio=('sentiment_label', lambda x: sum(s == 'positive' for s in x) / len(x)),
        negative_ratio=('sentiment_label', lambda x: sum(s == 'negative' for s in x) / len(x)),
        neutral_ratio=('sentiment_label', lambda x: sum(s == 'neutral' for s in x) / len(x)),

        # Store stock symbol and company name
        stock_symbol=('stock_symbol', 'first'),
        company_name=('company_name', 'first')
    ).reset_index()

    # Calculate moving averages (3-day window)
    daily_sentiment['sentiment_ma3'] = daily_sentiment['sentiment_mean'].rolling(window=3, min_periods=1).mean()

    # Calculate sentiment momentum (day-to-day change)
    daily_sentiment['sentiment_change'] = daily_sentiment['sentiment_mean'].diff()

    # Calculate sentiment volatility (3-day standard deviation)
    daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_mean'].rolling(window=3, min_periods=1).std()

    st.success(f"Aggregated sentiment to {len(daily_sentiment)} daily records.")
    return daily_sentiment

def process_sentiment(data_df):
    """
    Process all sentiment analysis steps.

    Args:
        data_df (pd.DataFrame): DataFrame with collected data

    Returns:
        tuple: (sentiment_df, daily_sentiment_df)
    """
    # Step 1: Analyze sentiment
    sentiment_df = analyze_sentiment(data_df)

    # Step 2: Aggregate by day
    daily_sentiment_df = aggregate_daily_sentiment(sentiment_df)

    return sentiment_df, daily_sentiment_df