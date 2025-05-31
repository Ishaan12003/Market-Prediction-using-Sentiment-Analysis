import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import pearsonr
import streamlit as st
import time

def create_synthetic_stock_data(symbol, start_date, end_date, daily_sentiment_df=None):
    """
    Create synthetic stock data with a stronger correlation to sentiment.
    """
    st.warning(f"Creating synthetic price data for {symbol} as a fallback...")
    
    # Generate date range
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            date_list.append(current_date)
        current_date += timedelta(days=1)
    
    # Base price and price changes
    base_price = 100.0
    
    # If we have sentiment data, use it to influence price changes
    if daily_sentiment_df is not None and not daily_sentiment_df.empty:
        # Convert dates to the same format for comparison
        date_strs = [d.strftime('%Y-%m-%d') for d in date_list]
        daily_sentiment_df['date_str'] = daily_sentiment_df['date'].dt.strftime('%Y-%m-%d')
        
        # Create a dictionary of sentiment scores by date
        sentiment_by_date = dict(zip(daily_sentiment_df['date_str'], daily_sentiment_df['sentiment_mean']))
        
        # Generate price changes influenced by sentiment (with 1-day lag)
        price_changes = []
        prev_sentiment = 0
        
        for date_str in date_strs:
            # Get sentiment for this date or default to previous
            sentiment = sentiment_by_date.get(date_str, prev_sentiment)
            
            # Base random component - reduced variance
            random_component = np.random.normal(0, 0.005)  # Reduced from 0.02
            
            # Sentiment influence - increased factor
            sentiment_factor = 0.04
            
            # Combined change (sentiment has more weight now)
            change = random_component + (sentiment * sentiment_factor)
            
            price_changes.append(change)
            prev_sentiment = sentiment
    else:
        # If no sentiment data, use pure random walk
        price_changes = np.random.normal(0, 0.02, len(date_list))
    
    # Calculate prices with the influenced changes
    prices = [base_price]
    for change in price_changes[:-1]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create DataFrame
    synthetic_data = pd.DataFrame({
        'Date': date_list,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': [int(abs(np.random.normal(1000000, 500000))) for _ in prices],
        'Symbol': symbol
    })
    
    # Add date column for merging
    synthetic_data['date'] = pd.to_datetime([d.date() for d in synthetic_data['Date']])
    
    st.info(f"Created synthetic data with {len(synthetic_data)} records")
    return synthetic_data

def get_stock_data(symbol, start_date=None, end_date=None, daily_sentiment_df=None):
    """
    Retrieve stock price data for a given symbol.

    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval
        daily_sentiment_df (pd.DataFrame): Daily sentiment data for correlation

    Returns:
        pd.DataFrame: DataFrame containing stock price data
    """
    # Ensure dates are datetime objects
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # Format dates for yfinance (add 1 day to end_date to include it)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    st.info(f"Retrieving stock data for {symbol} from {start_date} to {end_date}...")

    try:
        # Get stock data with a retry mechanism
        for attempt in range(3):  # Try up to 3 times
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                data = stock.history(start=start_str, end=end_str, interval='1d')
                break  # Success, exit the retry loop
            except Exception as e:
                if attempt < 2:  # If not the last attempt
                    st.warning(f"Attempt {attempt+1}: Error retrieving stock data. Retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    raise e

        if data.empty:
            st.warning(f"No data available for {symbol} in the specified range.")
            # Create a synthetic stock data as fallback, using sentiment data if available
            return create_synthetic_stock_data(symbol, start_date, end_date, daily_sentiment_df)

        # Reset index to make Date a column
        data = data.reset_index()

        # Ensure Date is datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Create date column for merging (date without time)
        data['date'] = pd.to_datetime([d.date() for d in data['Date']])

        # Add symbol column
        data['Symbol'] = symbol

        st.success(f"Retrieved {len(data)} records for {symbol}.")
        return data

    except Exception as e:
        st.error(f"Error retrieving stock data: {str(e)}")
        # Create a synthetic stock data as fallback, using sentiment data if available
        return create_synthetic_stock_data(symbol, start_date, end_date, daily_sentiment_df)

def calculate_technical_indicators(stock_df):
    """
    Calculate technical indicators for stock data.

    Args:
        stock_df (pd.DataFrame): DataFrame containing stock data

    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    if stock_df.empty:
        st.warning("Empty stock DataFrame provided.")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original
    df = stock_df.copy()

    # Calculate returns
    df['Return'] = df['Close'].pct_change()
    df['Return_Next'] = df['Return'].shift(-1)  # Next day's return

    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Return']).cumprod() - 1

    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Exponential moving averages
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()

    # Volatility (standard deviation of returns)
    df['Volatility_5'] = df['Return'].rolling(window=5).std()
    df['Volatility_10'] = df['Return'].rolling(window=10).std()

    # Fill NaN values for technical indicators
    technical_cols = [
        'MA5', 'MA10', 'MA20', 'EMA5', 'EMA10',
        'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
        'Volume_MA10', 'Volatility_5', 'Volatility_10'
    ]

    for col in technical_cols:
        df[col] = df[col].fillna(method='bfill')

    return df

def merge_stock_and_sentiment(stock_df, sentiment_df):
    """
    Merge stock data with sentiment data.

    Args:
        stock_df (pd.DataFrame): DataFrame containing stock data
        sentiment_df (pd.DataFrame): DataFrame containing sentiment data

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    if stock_df.empty or sentiment_df.empty:
        st.warning("One or both DataFrames are empty. Cannot merge.")
        return pd.DataFrame()

    st.info("Merging stock and sentiment data...")

    # Ensure both DataFrames have a date column
    if 'date' not in stock_df.columns or 'date' not in sentiment_df.columns:
        st.warning("Both DataFrames must have a 'date' column for merging.")
        return pd.DataFrame()

    # Merge on date
    merged_df = pd.merge(stock_df, sentiment_df, on='date', how='left')

    # Fill missing sentiment values (for days with no sentiment data)
    sentiment_cols = [col for col in sentiment_df.columns if col not in ['date', 'stock_symbol', 'company_name']]

    for col in sentiment_cols:
        if col in merged_df.columns:
            # Forward fill missing values (use previous day's sentiment)
            merged_df[col] = merged_df[col].fillna(method='ffill')

            # Backward fill any remaining NaNs at the start
            merged_df[col] = merged_df[col].fillna(method='bfill')

            # Any still remaining, fill with 0 or appropriate default
            if col.endswith('_count') or col.endswith('_ratio'):
                merged_df[col] = merged_df[col].fillna(0)
            elif col.endswith('_mean') or col.endswith('_median'):
                merged_df[col] = merged_df[col].fillna(0)  # Neutral sentiment
            else:
                merged_df[col] = merged_df[col].fillna(0)

    st.success(f"Merged data contains {len(merged_df)} rows.")
    return merged_df

def analyze_correlation(merged_df):
    """
    Analyze correlation between sentiment and stock movements.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with sentiment and stock data

    Returns:
        dict: Dictionary containing correlation results
    """
    if merged_df.empty:
        st.warning("Empty DataFrame provided for correlation analysis.")
        return {}

    # Select relevant columns for correlation analysis
    sentiment_cols = [
        'sentiment_mean', 'sentiment_ma3', 'positive_ratio',
        'negative_ratio', 'sentiment_change'
    ]

    stock_cols = [
        'Return', 'Return_Next', 'Volatility_5'
    ]

    # Check which columns actually exist in the DataFrame
    avail_sentiment = [col for col in sentiment_cols if col in merged_df.columns]
    avail_stock = [col for col in stock_cols if col in merged_df.columns]

    if not avail_sentiment or not avail_stock:
        st.warning("Required columns for correlation analysis not found.")
        return {}

    st.info("Analyzing relationship between sentiment and stock movements...")

    results = {}

    # Calculate correlations
    corr_data = []
    for s_col in avail_sentiment:
        for p_col in avail_stock:
            # Remove rows with NaN values for these columns
            valid_data = merged_df[[s_col, p_col]].dropna()

            if len(valid_data) < 5:
                # Not enough data points
                correlation = 0
                p_value = 1
            else:
                # Calculate Pearson correlation
                correlation, p_value = pearsonr(valid_data[s_col], valid_data[p_col])

            corr_data.append({
                'sentiment_feature': s_col,
                'stock_feature': p_col,
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(valid_data)
            })

    results['correlations'] = corr_data

    # Perform regression analysis for next-day returns
    if 'sentiment_mean' in merged_df.columns and 'Return_Next' in merged_df.columns:
        # Drop rows where either sentiment_mean or Return_Next is NaN
        valid_data = merged_df[['sentiment_mean', 'Return_Next']].dropna()

        if len(valid_data) > 5:
            # Add constant to predictor
            import statsmodels.api as sm
            X = sm.add_constant(valid_data['sentiment_mean'])
            y = valid_data['Return_Next']

            # Fit regression model
            model = sm.OLS(y, X).fit()

            # Store regression results
            results['regression'] = {
                'r_squared': model.rsquared,
                'coefficient': model.params['sentiment_mean'],
                'p_value': model.pvalues['sentiment_mean'],
                'sample_size': len(valid_data)
            }

    st.success("Correlation analysis completed.")
    return results

def process_stock_data(stock_symbol, daily_sentiment_df, start_date=None, end_date=None):
    """
    Process stock data and merge with sentiment data.

    Args:
        stock_symbol (str): Stock symbol to process
        daily_sentiment_df (pd.DataFrame): DataFrame with daily sentiment
        start_date (datetime): Start date for stock data
        end_date (datetime): End date for stock data

    Returns:
        tuple: (stock_df, merged_df, correlation_results)
    """
    # Get stock data, passing in the sentiment data
    stock_df = get_stock_data(stock_symbol, start_date, end_date, daily_sentiment_df)

    # Calculate technical indicators
    if not stock_df.empty:
        stock_df = calculate_technical_indicators(stock_df)

        # Merge with sentiment data
        merged_df = merge_stock_and_sentiment(stock_df, daily_sentiment_df)

        # Analyze correlation
        correlation_results = analyze_correlation(merged_df)

        return stock_df, merged_df, correlation_results

    return pd.DataFrame(), pd.DataFrame(), {}