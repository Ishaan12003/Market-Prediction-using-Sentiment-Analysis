import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_stock_and_sentiment(merged_df, stock_symbol, company_name):
    """
    Plot stock price and sentiment together.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with stock and sentiment data
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if merged_df.empty:
        st.warning("No data available for visualization.")
        return None

    # Check for required columns
    if 'date' not in merged_df.columns or 'Close' not in merged_df.columns or 'sentiment_mean' not in merged_df.columns:
        st.warning("Required columns missing from DataFrame.")
        return None

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 2, 1]})

    # First plot: Stock price
    ax1.plot(merged_df['date'], merged_df['Close'], 'b-', linewidth=2, label='Stock Price')

    # Add moving averages
    if 'MA5' in merged_df.columns:
        ax1.plot(merged_df['date'], merged_df['MA5'], 'r--', linewidth=1.5, label='5-Day MA')
    if 'MA20' in merged_df.columns:
        ax1.plot(merged_df['date'], merged_df['MA20'], 'g--', linewidth=1.5, label='20-Day MA')

    ax1.set_title(f"{company_name} ({stock_symbol}) Stock Price", fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Format y-axis as dollars
    ax1.yaxis.set_major_formatter('${x:,.2f}')

    # Second plot: Sentiment
    ax2.plot(merged_df['date'], merged_df['sentiment_mean'], 'g-', linewidth=2, label='Daily Sentiment')

    # Add sentiment moving average if available
    if 'sentiment_ma3' in merged_df.columns:
        ax2.plot(merged_df['date'], merged_df['sentiment_ma3'], 'b--', linewidth=1.5, label='3-Day MA')

    # Add horizontal line at y=0
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    ax2.set_title("Social Media Sentiment", fontsize=16)
    ax2.set_ylabel('Sentiment Score', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Set y-axis limits for sentiment
    ax2.set_ylim(-1, 1)

    # Third plot: Volume of posts
    if 'total_count' in merged_df.columns:
        bars = ax3.bar(merged_df['date'], merged_df['total_count'], color='orange', alpha=0.7, label='Post Volume')

        # Add post type breakdown if available
        if 'news_count' in merged_df.columns and 'reddit_count' in merged_df.columns:
            # Stacked bar chart for different sources
            bottom = np.zeros(len(merged_df))

            news_bars = ax3.bar(merged_df['date'], merged_df['news_count'],
                              color='blue', alpha=0.6, label='News Articles')

            bottom += merged_df['news_count'].fillna(0)

            reddit_bars = ax3.bar(merged_df['date'], merged_df['reddit_count'],
                                bottom=bottom, color='orange', alpha=0.6, label='Reddit Posts')

    ax3.set_title("Post Volume", fontsize=16)
    ax3.set_xlabel('Date', fontsize=14)
    ax3.set_ylabel('Count', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    fig.autofmt_xdate()

    # Adjust layout
    plt.tight_layout()

    return fig

def plot_sentiment_distribution(sentiment_df, stock_symbol, company_name):
    """
    Plot distribution of sentiment scores.

    Args:
        sentiment_df (pd.DataFrame): DataFrame with sentiment scores
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if sentiment_df.empty or 'sentiment_score' not in sentiment_df.columns:
        st.warning("No sentiment data available for visualization.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot histogram of sentiment scores
    n, bins, patches = ax.hist(sentiment_df['sentiment_score'], bins=30, alpha=0.7, color='royalblue')

    # Add vertical line for mean
    mean_sentiment = sentiment_df['sentiment_score'].mean()
    ax.axvline(x=mean_sentiment, color='red', linestyle='--',
              label=f'Mean: {mean_sentiment:.2f}')

    # Customize plot
    ax.set_title(f"Distribution of Sentiment Scores for {company_name} ({stock_symbol})", fontsize=16)
    ax.set_xlabel('Sentiment Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add text for sentiment statistics
    pos_pct = (sentiment_df['sentiment_label'] == 'positive').mean() * 100
    neg_pct = (sentiment_df['sentiment_label'] == 'negative').mean() * 100
    neu_pct = (sentiment_df['sentiment_label'] == 'neutral').mean() * 100

    stats_text = (
        f"Total Posts: {len(sentiment_df)}\n"
        f"Positive: {pos_pct:.1f}%\n"
        f"Negative: {neg_pct:.1f}%\n"
        f"Neutral: {neu_pct:.1f}%\n"
        f"Average Score: {mean_sentiment:.3f}"
    )

    # Place the text box in the upper right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    return fig

def plot_sentiment_by_source(sentiment_df, stock_symbol, company_name):
    """
    Plot sentiment distribution by source.

    Args:
        sentiment_df (pd.DataFrame): DataFrame with sentiment scores
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if sentiment_df.empty or 'source' not in sentiment_df.columns:
        st.warning("No source data available for visualization.")
        return None

    # Extract source type (reddit, news, etc.)
    sentiment_df['source_type'] = sentiment_df['source'].apply(
        lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
    )

    # Group by source type
    source_stats = sentiment_df.groupby('source_type').agg(
        mean_sentiment=('sentiment_score', 'mean'),
        count=('id', 'count'),
        std_sentiment=('sentiment_score', 'std')
    ).reset_index()

    # Sort by count
    source_stats = source_stats.sort_values('count', ascending=False)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    # Bar chart for sentiment
    bars = ax1.bar(source_stats['source_type'], source_stats['mean_sentiment'],
                 yerr=source_stats['std_sentiment'], alpha=0.7, color='royalblue')

    # Line chart for count
    line = ax2.plot(source_stats['source_type'], source_stats['count'], 'ro-',
                   linewidth=2, markersize=8)

    # Add labels and title
    ax1.set_title(f"Sentiment by Source for {company_name} ({stock_symbol})", fontsize=16)
    ax1.set_xlabel('Source', fontsize=14)
    ax1.set_ylabel('Average Sentiment Score', fontsize=14, color='royalblue')
    ax2.set_ylabel('Number of Posts', fontsize=14, color='red')

    # Set colors for y-axes
    ax1.tick_params(axis='y', colors='royalblue')
    ax2.tick_params(axis='y', colors='red')

    # Add zero line for sentiment
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='royalblue', lw=4, label='Sentiment Score'),
        Line2D([0], [0], color='red', marker='o', lw=2, label='Post Count')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig

def plot_sentiment_over_time(daily_sentiment_df, stock_symbol, company_name):
    """
    Plot sentiment trends over time.

    Args:
        daily_sentiment_df (pd.DataFrame): DataFrame with daily sentiment
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if daily_sentiment_df.empty or 'date' not in daily_sentiment_df.columns:
        st.warning("No daily sentiment data available for visualization.")
        return None

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # Plot sentiment
    ax1.plot(daily_sentiment_df['date'], daily_sentiment_df['sentiment_mean'],
            'b-', linewidth=2, label='Daily Sentiment')

    # Add sentiment moving average if available
    if 'sentiment_ma3' in daily_sentiment_df.columns:
        ax1.plot(daily_sentiment_df['date'], daily_sentiment_df['sentiment_ma3'],
                'g--', linewidth=2, label='3-Day Moving Average')

    # Add horizontal line at y=0
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    # Plot post count
    bars = ax2.bar(daily_sentiment_df['date'], daily_sentiment_df['total_count'],
                  alpha=0.3, color='gray', label='Post Count')

    # Add labels and title
    ax1.set_title(f"{company_name} ({stock_symbol}) Sentiment Trends", fontsize=16)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Sentiment Score', fontsize=14, color='blue')
    ax2.set_ylabel('Post Count', fontsize=14, color='gray')

    # Set colors for y-axes
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='gray')

    # Add legend
    ax1.legend(loc='upper left', fontsize=12)

    # Format x-axis
    plt.xticks(rotation=45)
    fig.autofmt_xdate()

    # Set y-axis limits for sentiment
    ax1.set_ylim(-1, 1)

    plt.tight_layout()

    return fig

def plot_correlation_heatmap(correlation_results, stock_symbol, company_name):
    """
    Plot correlation heatmap between sentiment and stock metrics.

    Args:
        correlation_results (dict): Dictionary with correlation results
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if not correlation_results or 'correlations' not in correlation_results:
        st.warning("No correlation results available for visualization.")
        return None

    # Extract correlations
    corr_data = correlation_results['correlations']

    # Convert to DataFrame
    corr_df = pd.DataFrame(corr_data)

    # Pivot to create correlation matrix
    corr_matrix = corr_df.pivot(index='sentiment_feature', columns='stock_feature', values='correlation')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    fmt='.2f', linewidths=0.5, ax=ax)

    # Add labels and title
    ax.set_title(f"Correlation: {company_name} ({stock_symbol}) Sentiment vs. Stock Metrics", fontsize=16)
    ax.set_xlabel('Stock Metrics', fontsize=14)
    ax.set_ylabel('Sentiment Metrics', fontsize=14)

    # Clean up axis labels
    x_labels = [label.replace('_', ' ').title() for label in corr_matrix.columns]
    y_labels = [label.replace('_', ' ').title() for label in corr_matrix.index]

    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0)

    plt.tight_layout()

    return fig

def plot_sentiment_vs_returns(merged_df, stock_symbol, company_name):
    """
    Plot scatter plot of sentiment vs. stock returns.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with sentiment and stock data
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if merged_df.empty:
        st.warning("No data available for scatter plot.")
        return None

    # Check for required columns
    required_cols = ['sentiment_mean', 'Return', 'Return_Next']
    missing_cols = [col for col in required_cols if col not in merged_df.columns]

    if missing_cols:
        st.warning(f"Missing columns for scatter plot: {missing_cols}")
        return None

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # First scatter: Same-day returns
    ax1.scatter(merged_df['sentiment_mean'], merged_df['Return'],
               alpha=0.7, color='blue', s=50)

    # Add trend line
    z = np.polyfit(merged_df['sentiment_mean'].dropna(),
                  merged_df.loc[merged_df['sentiment_mean'].dropna().index, 'Return'], 1)
    p = np.poly1d(z)

    # Add trend line to plot
    x_trend = np.linspace(merged_df['sentiment_mean'].min(), merged_df['sentiment_mean'].max(), 100)
    ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.7)

    # Add correlation value
    same_day_corr = merged_df[['sentiment_mean', 'Return']].corr().iloc[0, 1]
    ax1.text(0.05, 0.95, f"Correlation: {same_day_corr:.3f}", transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Add axes labels and title
    ax1.set_title("Sentiment vs. Same-Day Returns", fontsize=14)
    ax1.set_xlabel('Sentiment Score', fontsize=12)
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add reference lines
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.2)

    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1%}'))

    # Second scatter: Next-day returns
    ax2.scatter(merged_df['sentiment_mean'], merged_df['Return_Next'],
               alpha=0.7, color='green', s=50)

    # Add trend line
    z = np.polyfit(merged_df['sentiment_mean'].dropna(),
                  merged_df.loc[merged_df['sentiment_mean'].dropna().index, 'Return_Next'], 1)
    p = np.poly1d(z)

    # Add trend line to plot
    x_trend = np.linspace(merged_df['sentiment_mean'].min(), merged_df['sentiment_mean'].max(), 100)
    ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.7)

    # Add correlation value
    next_day_corr = merged_df[['sentiment_mean', 'Return_Next']].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f"Correlation: {next_day_corr:.3f}", transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Add axes labels and title
    ax2.set_title("Sentiment vs. Next-Day Returns", fontsize=14)
    ax2.set_xlabel('Sentiment Score', fontsize=12)
    ax2.set_ylabel('Next-Day Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add reference lines
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.2)

    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1%}'))

    # Add overall title
    plt.suptitle(f"{company_name} ({stock_symbol}) Sentiment vs. Stock Returns", fontsize=16)

    plt.tight_layout()

    return fig

def plot_sentiment_components(daily_sentiment_df, stock_symbol, company_name):
    """
    Plot breakdown of sentiment into positive, negative, and neutral components.

    Args:
        daily_sentiment_df (pd.DataFrame): Daily sentiment DataFrame
        stock_symbol (str): Stock symbol for title
        company_name (str): Company name for title

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if daily_sentiment_df.empty:
        st.warning("No daily sentiment data available for components plot.")
        return None

    # Check for required columns
    required_cols = ['date', 'positive_ratio', 'negative_ratio', 'neutral_ratio']
    missing_cols = [col for col in required_cols if col not in daily_sentiment_df.columns]

    if missing_cols:
        st.warning(f"Missing columns for sentiment components plot: {missing_cols}")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot stacked area chart
    ax.fill_between(daily_sentiment_df['date'], 0, daily_sentiment_df['positive_ratio'],
                   label='Positive', alpha=0.7, color='green')

    ax.fill_between(daily_sentiment_df['date'], daily_sentiment_df['positive_ratio'],
                   daily_sentiment_df['positive_ratio'] + daily_sentiment_df['neutral_ratio'],
                   label='Neutral', alpha=0.7, color='gray')

    ax.fill_between(daily_sentiment_df['date'],
                   daily_sentiment_df['positive_ratio'] + daily_sentiment_df['neutral_ratio'],
                   1.0, label='Negative', alpha=0.7, color='red')

    # Add labels and title
    ax.set_title(f"{company_name} ({stock_symbol}) Sentiment Components", fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')

    # Format x-axis
    plt.xticks(rotation=45)
    fig.autofmt_xdate()

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.0%}'))

    # Set y-axis limits
    ax.set_ylim(0, 1)

    plt.tight_layout()

    return fig

def create_visualizations(sentiment_df, daily_sentiment_df, stock_df, merged_df,
                         correlation_results, stock_symbol, company_name):
    """
    Create all visualizations and return them as a list.

    Args:
        sentiment_df (pd.DataFrame): Sentiment DataFrame
        daily_sentiment_df (pd.DataFrame): Daily sentiment DataFrame
        stock_df (pd.DataFrame): Stock DataFrame
        merged_df (pd.DataFrame): Merged DataFrame
        correlation_results (dict): Correlation results
        stock_symbol (str): Stock symbol
        company_name (str): Company name

    Returns:
        list: List of (title, figure) tuples
    """
    visualizations = []

    # 1. Stock price and sentiment
    fig1 = plot_stock_and_sentiment(merged_df, stock_symbol, company_name)
    if fig1:
        visualizations.append(("Stock Price and Sentiment", fig1))

    # 2. Sentiment distribution
    fig2 = plot_sentiment_distribution(sentiment_df, stock_symbol, company_name)
    if fig2:
        visualizations.append(("Sentiment Distribution", fig2))

    # 3. Sentiment by source
    fig3 = plot_sentiment_by_source(sentiment_df, stock_symbol, company_name)
    if fig3:
        visualizations.append(("Sentiment by Source", fig3))

    # 4. Sentiment over time
    fig4 = plot_sentiment_over_time(daily_sentiment_df, stock_symbol, company_name)
    if fig4:
        visualizations.append(("Sentiment Over Time", fig4))

    # 5. Correlation heatmap
    fig5 = plot_correlation_heatmap(correlation_results, stock_symbol, company_name)
    if fig5:
        visualizations.append(("Correlation Analysis", fig5))

    # 6. Sentiment vs. returns
    fig6 = plot_sentiment_vs_returns(merged_df, stock_symbol, company_name)
    if fig6:
        visualizations.append(("Sentiment vs. Returns", fig6))

    # 7. Sentiment components
    fig7 = plot_sentiment_components(daily_sentiment_df, stock_symbol, company_name)
    if fig7:
        visualizations.append(("Sentiment Components", fig7))

    return visualizations