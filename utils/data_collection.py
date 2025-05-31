import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import re
import warnings
import streamlit as st

# Import synthetic data generation functions
from utils.data_generation import (
    generate_synthetic_reddit_data,
    generate_synthetic_news_data
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Company name to stock symbol mapping (for common companies)
COMPANY_TO_SYMBOL = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'amazon': 'AMZN',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'meta': 'META',
    'facebook': 'META',
    'tesla': 'TSLA',
    'nvidia': 'NVDA',
    'netflix': 'NFLX',
    'disney': 'DIS',
    'walmart': 'WMT',
    'coca-cola': 'KO',
    'pepsi': 'PEP',
    'mcdonalds': 'MCD',
    'nike': 'NKE',
    'starbucks': 'SBUX',
    'amd': 'AMD',
    'intel': 'INTC',
    'ibm': 'IBM'
}

def get_symbol_from_name(company_name):
    """
    Convert company name to stock symbol, checking our mapping or using the name directly.

    Args:
        company_name (str): Company name to convert

    Returns:
        str: Stock symbol
    """
    # Clean input
    company_name = company_name.strip().lower()

    # Check if it's in our mapping
    if company_name in COMPANY_TO_SYMBOL:
        return COMPANY_TO_SYMBOL[company_name]

    # Check if it might already be a symbol (all caps, 1-5 chars)
    if company_name.isupper() and 1 <= len(company_name) <= 5:
        return company_name

    # Try to get the symbol from Yahoo Finance
    try:
        search_result = yf.Ticker(company_name)
        info = search_result.info
        if 'symbol' in info:
            return info['symbol']
    except:
        pass

    # Return the input as is (possibly as a symbol)
    return company_name.upper()

def collect_reddit_data(stock_symbol, company_name=None,
                       start_date=None, end_date=None, limit=100):
    """
    Collect Reddit posts and comments related to a stock symbol.

    Args:
        stock_symbol (str): Stock symbol to search for
        company_name (str): Company name for additional context
        start_date (datetime): Start date for collecting data
        end_date (datetime): End date for collecting data
        limit (int): Maximum number of posts to retrieve

    Returns:
        pd.DataFrame: DataFrame containing posts and comments
    """
    # Get API credentials from environment variables
    REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
    REDDIT_SECRET = os.environ.get("REDDIT_SECRET", "")
    
    # Check if we have valid Reddit credentials
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_ID != "YOUR_REDDIT_CLIENT_ID" and
           REDDIT_SECRET and REDDIT_SECRET != "YOUR_REDDIT_SECRET"):
        st.warning("No valid Reddit API credentials. Using synthetic data only.")
        return pd.DataFrame()

    # Ensure dates are datetime objects
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # List of financial subreddits to search
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockmarket', 'options']

    st.info(f"Collecting Reddit data for {stock_symbol} from {start_date} to {end_date}...")

    try:
        # Import PRAW
        import praw

        # Initialize Reddit API
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent="StockSentimentAnalysis"
        )

        # List to store collected data
        data = []
        search_terms = [stock_symbol.upper(), f"${stock_symbol.upper()}"]
        if company_name:
            search_terms.append(company_name)

        # Search each subreddit
        for subreddit_name in subreddits:
            try:
                st.text(f"Searching in r/{subreddit_name}...")
                subreddit = reddit.subreddit(subreddit_name)

                # Search the subreddit for posts containing the search terms
                search_query = " OR ".join(search_terms)
                submissions = subreddit.search(search_query, sort="new", limit=limit)

                for submission in submissions:
                    # Check if submission is within date range
                    post_date = datetime.fromtimestamp(submission.created_utc)
                    if post_date < start_date or post_date > end_date:
                        continue

                    # Extract submission data
                    submission_data = {
                        'source': f"reddit:r/{subreddit_name}",
                        'id': submission.id,
                        'username': submission.author.name if submission.author else "[deleted]",
                        'title': submission.title,
                        'content': submission.selftext,
                        'full_text': f"{submission.title} {submission.selftext}",
                        'score': submission.score,
                        'timestamp': post_date,
                        'type': 'submission',
                        'url': submission.url,
                        'num_comments': submission.num_comments
                    }

                    # Check if the post actually mentions the stock
                    if any(term.lower() in submission_data['full_text'].lower() for term in search_terms):
                        data.append(submission_data)

                    # Get comments
                    submission.comments.replace_more(limit=0)  # Skip "load more comments"
                    for comment in list(submission.comments)[:10]:  # Get top 10 comments
                        if not comment.author:
                            continue

                        comment_date = datetime.fromtimestamp(comment.created_utc)
                        if comment_date < start_date or comment_date > end_date:
                            continue

                        comment_data = {
                            'source': f"reddit:r/{subreddit_name}",
                            'id': comment.id,
                            'username': comment.author.name if comment.author else "[deleted]",
                            'title': f"Re: {submission.title}",
                            'content': comment.body,
                            'full_text': comment.body,
                            'score': comment.score,
                            'timestamp': comment_date,
                            'type': 'comment',
                            'url': f"https://www.reddit.com{comment.permalink}",
                            'num_comments': 0
                        }

                        if any(term.lower() in comment_data['full_text'].lower() for term in search_terms):
                            data.append(comment_data)

            except Exception as e:
                st.warning(f"Error searching subreddit {subreddit_name}: {str(e)}")
                continue

        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
            st.success(f"Successfully collected {len(df)} Reddit posts and comments.")
            return df
        else:
            st.info("No Reddit data found matching search criteria.")
            return pd.DataFrame()

    except Exception as e:
        st.warning(f"Error initializing Reddit API: {str(e)}")
        st.info("Unable to collect real Reddit data. Using synthetic data only.")
        return pd.DataFrame()

def collect_news_data(stock_symbol, company_name=None, start_date=None, end_date=None):
    """
    Collect news articles related to a stock from News API.

    Args:
        stock_symbol (str): Stock symbol to search for
        company_name (str): Company name for additional search context
        start_date (datetime): Start date for collecting data
        end_date (datetime): End date for collecting data

    Returns:
        pd.DataFrame: DataFrame containing news articles
    """
    # Get API key from environment variables
    NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
    
    # Check if we have valid News API credentials
    if not (NEWSAPI_KEY and NEWSAPI_KEY != "YOUR_NEWSAPI_KEY"):
        st.warning("No valid News API credentials. Using synthetic data only.")
        return pd.DataFrame()

    # Ensure dates are datetime objects
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # News API free tier has a limit of 1 month
    max_days_ago = 30
    if (datetime.now() - start_date).days > max_days_ago:
        st.warning(f"News API free tier only allows data from the past {max_days_ago} days.")
        st.info(f"Adjusting start date from {start_date} to {(datetime.now() - timedelta(days=max_days_ago))}")
        start_date = datetime.now() - timedelta(days=max_days_ago)

    # Format dates for the API
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        # Import News API
        from newsapi import NewsApiClient

        # Initialize News API
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

        st.info(f"Collecting news data for {stock_symbol} from {start_date} to {end_date}...")

        # Create search query
        search_query = stock_symbol
        if company_name:
            search_query += f" OR {company_name}"

        # Fetch articles
        response = newsapi.get_everything(
            q=search_query,
            language='en',
            from_param=start_str,
            to=end_str,
            sort_by='relevancy',
            page_size=100  # Maximum allowed
        )

        if response['totalResults'] == 0:
            st.info("No news articles found.")
            return pd.DataFrame()

        st.info(f"Found {response['totalResults']} articles, fetching up to 100...")

        # Process articles
        data = []
        for article in response['articles']:
            try:
                # Parse date
                published_at = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")

                # Create article data
                article_data = {
                    'source': f"news:{article['source']['name']}",
                    'id': hash(article['url']),  # Create a hash ID
                    'username': article['author'] if article['author'] else article['source']['name'],
                    'title': article['title'],
                    'content': article['description'] or "",
                    'full_text': f"{article['title']} {article['description'] or ''}",
                    'score': 0,  # No equivalent in news
                    'timestamp': published_at,
                    'type': 'news',
                    'url': article['url'],
                    'num_comments': 0  # No equivalent in news
                }

                # Check if the article actually mentions the stock
                if stock_symbol.lower() in article_data['full_text'].lower() or (
                    company_name and company_name.lower() in article_data['full_text'].lower()):
                    data.append(article_data)

            except Exception as e:
                st.warning(f"Error processing article: {str(e)}")
                continue

        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
            st.success(f"Successfully collected {len(df)} news articles.")
            return df
        else:
            st.info("No news data could be processed.")
            return pd.DataFrame()

    except Exception as e:
        st.warning(f"Error initializing News API: {str(e)}")
        st.info("Unable to collect real news data. Using synthetic data only.")
        return pd.DataFrame()

def collect_data_for_stock(stock_symbol, company_name=None, start_date=None, end_date=None, include_synthetic=True):
    """
    Collect all data for a stock symbol, combining real and synthetic data.

    Args:
        stock_symbol (str): Stock symbol to search for
        company_name (str): Company name for additional context
        start_date (datetime): Start date for collecting data
        end_date (datetime): End date for collecting data
        include_synthetic (bool): Whether to include synthetic data

    Returns:
        pd.DataFrame: Combined DataFrame with all collected data
    """
    # Clean up and validate the stock symbol
    stock_symbol = stock_symbol.strip().upper()

    # If no company name is provided, try to determine it
    if not company_name:
        # Check if symbol is in our mapping
        for name, symbol in COMPANY_TO_SYMBOL.items():
            if symbol.upper() == stock_symbol:
                company_name = name.title()
                break

        # If still no company name, try to get it from Yahoo Finance
        if not company_name:
            try:
                ticker = yf.Ticker(stock_symbol)
                info = ticker.info
                if 'longName' in info:
                    company_name = info['longName']
                elif 'shortName' in info:
                    company_name = info['shortName']
                else:
                    company_name = stock_symbol
            except Exception as e:
                st.warning(f"Error getting company info: {str(e)}")
                company_name = stock_symbol

    # Ensure dates are datetime objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    elif not start_date:
        start_date = datetime.now() - timedelta(days=30)

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    elif not end_date:
        end_date = datetime.now()

    st.info(f"Collecting data for {company_name} ({stock_symbol}) from {start_date} to {end_date}...")

    # Collect real data (attempts to use API if credentials are available)
    real_data_frames = []

    # Reddit data
    reddit_df = collect_reddit_data(
        stock_symbol=stock_symbol,
        company_name=company_name,
        start_date=start_date,
        end_date=end_date
    )
    if not reddit_df.empty:
        real_data_frames.append(reddit_df)

    # News data
    news_df = collect_news_data(
        stock_symbol=stock_symbol,
        company_name=company_name,
        start_date=start_date,
        end_date=end_date
    )
    if not news_df.empty:
        real_data_frames.append(news_df)

    # Generate synthetic data if requested or if no real data
    synthetic_data_frames = []
    if include_synthetic or len(real_data_frames) < 2 or sum(len(df) for df in real_data_frames) < 50:
        st.info("Generating synthetic data to supplement real data...")

        # Reddit data
        synthetic_reddit = generate_synthetic_reddit_data(
            stock_symbol=stock_symbol,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            count=150
        )
        synthetic_data_frames.append(synthetic_reddit)

        # News data
        synthetic_news = generate_synthetic_news_data(
            stock_symbol=stock_symbol,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            count=100
        )
        synthetic_data_frames.append(synthetic_news)

    # Combine all data
    all_data_frames = real_data_frames + synthetic_data_frames

    if not all_data_frames:
        st.error("No data collected. Please check inputs and try again.")
        return pd.DataFrame(), stock_symbol, company_name

    # Combine data frames
    combined_df = pd.concat(all_data_frames, ignore_index=True)

    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')

    # Remove duplicates (could happen with synthetic data)
    combined_df = combined_df.drop_duplicates(subset=['id', 'source'])

    # Ensure all required columns exist
    required_columns = ['source', 'id', 'username', 'title', 'content', 'full_text',
                       'score', 'timestamp', 'type', 'url']

    for col in required_columns:
        if col not in combined_df.columns:
            combined_df[col] = None

    # Add stock symbol and company name columns
    combined_df['stock_symbol'] = stock_symbol
    combined_df['company_name'] = company_name

    st.success(f"Final dataset contains {len(combined_df)} records.")
    return combined_df, stock_symbol, company_name

# Function to clean text for analysis
def clean_text(text):
    """
    Clean text for sentiment analysis.

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove Reddit formatting (e.g., [removed], [deleted])
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)

    # Convert to lowercase for better analysis
    text = text.lower()

    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()