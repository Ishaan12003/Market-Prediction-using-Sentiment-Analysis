import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('app.env')

# Import utility functions
from utils.data_collection import get_symbol_from_name, collect_data_for_stock
from utils.sentiment_analysis import process_sentiment
from utils.stock_processing import process_stock_data
from utils.visualization import create_visualizations
from utils.reporting import generate_html_report

# Set page config
st.set_page_config(
    page_title="Market Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize NLTK
import nltk
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon', quiet=True)

download_nltk_data()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stat-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .positive {
        color: green;
        font-weight: bold;
    }
    .negative {
        color: red;
        font-weight: bold;
    }
    .neutral {
        color: gray;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">Market Sentiment Analysis Tool</h1>', unsafe_allow_html=True)
st.markdown('''
This application analyzes social media and news sentiment for stocks and examines the relationship between sentiment and stock price movements.
''')

# Sidebar for inputs
st.sidebar.header("Analysis Parameters")

# Input for company/stock
company_input = st.sidebar.text_input("Enter Company Name or Stock Symbol:", value="AAPL")

# Date range selection
date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

# Ensure two dates are provided (start and end)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0] if date_range else (datetime.now() - timedelta(days=30))
    end_date = datetime.now()

# Convert dates to datetime if they are date objects
if isinstance(start_date, date) and not isinstance(start_date, datetime):
    start_date = datetime.combine(start_date, datetime.min.time())
if isinstance(end_date, date) and not isinstance(end_date, datetime):
    end_date = datetime.combine(end_date, datetime.min.time())

# Include synthetic data option
include_synthetic = st.sidebar.checkbox("Include Synthetic Data", value=True, 
                                      help="Generate synthetic data to supplement real data")

# API credentials inputs (hidden by default)
with st.sidebar.expander("API Credentials (Optional)"):
    reddit_client_id = st.text_input("Reddit Client ID:", value=os.environ.get("REDDIT_CLIENT_ID", ""), 
                                    type="password" if os.environ.get("REDDIT_CLIENT_ID", "") else "default")
    reddit_secret = st.text_input("Reddit Secret:", value=os.environ.get("REDDIT_SECRET", ""), 
                                 type="password" if os.environ.get("REDDIT_SECRET", "") else "default")
    newsapi_key = st.text_input("News API Key:", value=os.environ.get("NEWSAPI_KEY", ""), 
                               type="password" if os.environ.get("NEWSAPI_KEY", "") else "default")
    
    # Update environment variables if provided
    if reddit_client_id and reddit_client_id != os.environ.get("REDDIT_CLIENT_ID", ""):
        os.environ["REDDIT_CLIENT_ID"] = reddit_client_id
    if reddit_secret and reddit_secret != os.environ.get("REDDIT_SECRET", ""):
        os.environ["REDDIT_SECRET"] = reddit_secret
    if newsapi_key and newsapi_key != os.environ.get("NEWSAPI_KEY", ""):
        os.environ["NEWSAPI_KEY"] = newsapi_key

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Function to run the data collection and analysis (cached)
@st.cache_data(ttl=3600, show_spinner=False)
def collect_and_analyze_data(company_input, start_date, end_date, include_synthetic):
    """Run the data collection and analysis with caching."""
    # Ensure start_date and end_date are datetime objects
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())
        
    with st.spinner("Collecting data..."):
        data_df, stock_symbol, company_name = collect_data_for_stock(
            stock_symbol=company_input,
            start_date=start_date,
            end_date=end_date,
            include_synthetic=include_synthetic
        )
        
    if data_df.empty:
        st.error("No data collected. Please check inputs and try again.")
        return None, None, None, None, None, None, None
        
    with st.spinner("Analyzing sentiment..."):
        sentiment_df, daily_sentiment_df = process_sentiment(data_df)
        
    with st.spinner("Processing stock data..."):
        stock_df, merged_df, correlation_results = process_stock_data(
            stock_symbol=stock_symbol,
            daily_sentiment_df=daily_sentiment_df,
            start_date=start_date,
            end_date=end_date
        )
        
    return sentiment_df, daily_sentiment_df, stock_df, merged_df, correlation_results, stock_symbol, company_name

# Non-cached function to handle visualizations and complete the analysis
def run_sentiment_analysis(company_input, start_date, end_date, include_synthetic):
    """Run the complete sentiment analysis pipeline."""
    # Run the cached part
    results = collect_and_analyze_data(company_input, start_date, end_date, include_synthetic)
    
    if results[0] is None:
        return None, None, None, None, None, [], None, None
    
    sentiment_df, daily_sentiment_df, stock_df, merged_df, correlation_results, stock_symbol, company_name = results
    
    if merged_df.empty:
        st.warning("Failed to merge sentiment and stock data. Limited analysis available.")
    
    # Create visualizations (not cached)
    with st.spinner("Creating visualizations..."):
        visualizations = create_visualizations(
            sentiment_df=sentiment_df,
            daily_sentiment_df=daily_sentiment_df,
            stock_df=stock_df,
            merged_df=merged_df,
            correlation_results=correlation_results,
            stock_symbol=stock_symbol,
            company_name=company_name
        )
    
    return sentiment_df, daily_sentiment_df, stock_df, merged_df, correlation_results, visualizations, stock_symbol, company_name

# Main app
if company_input:
    # If run button is clicked
    if run_analysis:
        # Run the analysis
        with st.spinner("Running analysis... This may take a moment."):
            results = run_sentiment_analysis(
                company_input=company_input,
                start_date=start_date,
                end_date=end_date,
                include_synthetic=include_synthetic
            )
            
            if results[0] is not None:
                sentiment_df, daily_sentiment_df, stock_df, merged_df, correlation_results, visualizations, stock_symbol, company_name = results
                
                # Store results in session state for other pages to access
                st.session_state['analysis_results'] = {
                    'sentiment_df': sentiment_df,
                    'daily_sentiment_df': daily_sentiment_df,
                    'stock_df': stock_df,
                    'merged_df': merged_df,
                    'correlation_results': correlation_results,
                    'visualizations': visualizations,
                    'stock_symbol': stock_symbol,
                    'company_name': company_name,
                    'start_date': start_date,
                    'end_date': end_date
                }
                
                # Show success message
                st.success(f"Analysis completed for {company_name} ({stock_symbol})")
                
                # Display overview
                st.markdown(f'<h2 class="sub-header">Analysis Overview: {company_name} ({stock_symbol})</h2>', unsafe_allow_html=True)
                
                # Create metrics section
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Posts Analyzed", 
                        value=f"{len(sentiment_df)}"
                    )
                    
                with col2:
                    avg_sentiment = sentiment_df['sentiment_score'].mean()
                    st.metric(
                        label="Average Sentiment", 
                        value=f"{avg_sentiment:.3f}",
                        delta=f"{avg_sentiment:.3f}",
                        delta_color="normal" if abs(avg_sentiment) < 0.1 else ("inverse" if avg_sentiment < 0 else "normal")
                    )
                    
                with col3:
                    if not merged_df.empty and 'Close' in merged_df.columns:
                        price_change = ((merged_df['Close'].iloc[-1] / merged_df['Close'].iloc[0]) - 1) * 100
                        st.metric(
                            label="Price Change", 
                            value=f"{merged_df['Close'].iloc[-1]:.2f} USD",
                            delta=f"{price_change:.2f}%",
                            delta_color="normal" if price_change >= 0 else "inverse"
                        )
                    else:
                        st.metric(label="Price Data", value="Not Available")
                
                # Display sentiment distribution in a box
                st.markdown('<div class="stat-container">', unsafe_allow_html=True)
                
                pos_pct = (sentiment_df['sentiment_label'] == 'positive').mean() * 100
                neg_pct = (sentiment_df['sentiment_label'] == 'negative').mean() * 100
                neu_pct = (sentiment_df['sentiment_label'] == 'neutral').mean() * 100
                
                st.markdown(f"""
                ### Sentiment Distribution:
                - <span class="positive">Positive: {pos_pct:.1f}%</span>
                - <span class="negative">Negative: {neg_pct:.1f}%</span>
                - <span class="neutral">Neutral: {neu_pct:.1f}%</span>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display correlation info if available
                if correlation_results and 'correlations' in correlation_results:
                    st.markdown('<div class="stat-container">', unsafe_allow_html=True)
                    
                    # Find key correlations
                    key_correlations = [c for c in correlation_results['correlations']
                                     if c['sentiment_feature'] == 'sentiment_mean' and
                                     c['stock_feature'] in ['Return', 'Return_Next']]
                    
                    if key_correlations:
                        st.markdown("### Key Correlations:")
                        for corr in key_correlations:
                            significance = "Significant" if corr['p_value'] < 0.05 else "Not Significant"
                            color_class = "positive" if corr['correlation'] > 0 else "negative"
                            st.markdown(f"""
                            - {corr['sentiment_feature'].replace('_', ' ').title()} vs {corr['stock_feature'].replace('_', ' ').title()}:
                              <span class="{color_class}">Correlation: {corr['correlation']:.3f}</span> (p-value: {corr['p_value']:.4f}, {significance})
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display a key visualization
                if visualizations:
                    st.markdown(f'<h2 class="sub-header">Key Visualization</h2>', unsafe_allow_html=True)
                    
                    # Find the stock and sentiment visualization
                    stock_sentiment_viz = next((v for v in visualizations if v[0] == "Stock Price and Sentiment"), None)
                    
                    if stock_sentiment_viz:
                        st.pyplot(stock_sentiment_viz[1])
                    else:
                        # Fall back to first visualization
                        st.pyplot(visualizations[0][1])
                
                # Navigation instructions
                st.markdown("### Explore Detailed Analysis")
                st.markdown("""
                Use the navigation in the sidebar to explore different aspects of the analysis:
                - **Dashboard**: Overview of key metrics and visualizations
                - **Data Exploration**: Explore the collected data in detail
                - **Sentiment Analysis**: Detailed sentiment analysis results
                - **Correlation Analysis**: Analysis of correlation between sentiment and stock movements
                """)
                
                # Download link for full report
                st.download_button(
                    label="Download Full HTML Report",
                    data=generate_html_report(
                        sentiment_df, daily_sentiment_df, stock_df, merged_df,
                        correlation_results, visualizations,
                        stock_symbol, company_name, start_date, end_date
                    ),
                    file_name=f"{stock_symbol}_sentiment_report.html",
                    mime="text/html"
                )
                
    else:
        # Display instructions when app first loads
        st.info("Enter a company name or stock symbol and click 'Run Analysis' to start.")
        st.markdown("""
        ### How to use this tool:
        1. Enter a company name or stock symbol in the sidebar
        2. Select a date range for analysis
        3. Choose whether to include synthetic data
        4. Optionally provide API credentials for real data
        5. Click 'Run Analysis' to start the process
        
        The analysis will collect social media and news data, analyze sentiment, and correlate it with stock price movements.
        """)
else:
    st.warning("Please enter a company name or stock symbol to analyze.")