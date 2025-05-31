import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Dashboard - Market Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Market Sentiment Dashboard")
st.write("Overview of stock performance and sentiment metrics")

# Check if analysis has been run
if 'analysis_results' not in st.session_state:
    st.warning("Please run an analysis first from the home page.")
    st.stop()

# Get data from session state
results = st.session_state['analysis_results']
sentiment_df = results['sentiment_df']
daily_sentiment_df = results['daily_sentiment_df']
stock_df = results['stock_df']
merged_df = results['merged_df']
correlation_results = results['correlation_results']
stock_symbol = results['stock_symbol']
company_name = results['company_name']
start_date = results['start_date']
end_date = results['end_date']

# Format date strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Header with stock info
st.header(f"{company_name} ({stock_symbol}) Analysis")
st.write(f"Analysis period: {start_date_str} to {end_date_str}")

# Create dashboard layout
col1, col2 = st.columns([2, 1])

# Stock price chart with sentiment overlay (Plotly)
with col1:
    st.subheader("Stock Price and Sentiment")
    
    if not merged_df.empty and 'Close' in merged_df.columns and 'sentiment_mean' in merged_df.columns:
        fig = go.Figure()
        
        # Add stock price trace
        fig.add_trace(go.Scatter(
            x=merged_df['date'],
            y=merged_df['Close'],
            name='Stock Price',
            line=dict(color='royalblue', width=2),
            yaxis='y'
        ))
        
        # Add moving average traces
        if 'MA5' in merged_df.columns:
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['MA5'],
                name='5-Day MA',
                line=dict(color='firebrick', width=1.5, dash='dash'),
                yaxis='y'
            ))
        
        if 'MA20' in merged_df.columns:
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['MA20'],
                name='20-Day MA',
                line=dict(color='green', width=1.5, dash='dash'),
                yaxis='y'
            ))
        
        # Add sentiment trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=merged_df['date'],
            y=merged_df['sentiment_mean'],
            name='Sentiment',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))
        
        # Add sentiment moving average if available
        if 'sentiment_ma3' in merged_df.columns:
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['sentiment_ma3'],
                name='3-Day Sentiment MA',
                line=dict(color='purple', width=1.5, dash='dash'),
                yaxis='y2'
            ))

        # Set up layout with dual y-axes
        fig.update_layout(
            title=f'{company_name} Stock Price and Sentiment',
            xaxis=dict(title='Date'),
            yaxis=dict(
                title='Price ($)',
                titlefont=dict(color='royalblue'),
                tickfont=dict(color='royalblue'),
                side='left'
            ),
            yaxis2=dict(
                title='Sentiment Score',
                titlefont=dict(color='orange'),
                tickfont=dict(color='orange'),
                overlaying='y',
                side='right',
                range=[-1, 1]
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Insufficient data for price chart")

# Sentiment metrics cards
with col2:
    st.subheader("Key Metrics")
    
    # Calculate key metrics
    if not sentiment_df.empty:
        avg_sentiment = sentiment_df['sentiment_score'].mean()
        pos_pct = (sentiment_df['sentiment_label'] == 'positive').mean() * 100
        neg_pct = (sentiment_df['sentiment_label'] == 'negative').mean() * 100
        neu_pct = (sentiment_df['sentiment_label'] == 'neutral').mean() * 100
        
        # Create metric cards
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.metric(
                label="Average Sentiment", 
                value=f"{avg_sentiment:.3f}", 
                delta=f"{avg_sentiment:.2f}",
                delta_color="normal" if abs(avg_sentiment) < 0.1 else ("inverse" if avg_sentiment < 0 else "normal")
            )
            
        with col2b:
            if not merged_df.empty and 'Close' in merged_df.columns:
                price_change = ((merged_df['Close'].iloc[-1] / merged_df['Close'].iloc[0]) - 1) * 100
                st.metric(
                    label="Price Change", 
                    value=f"${merged_df['Close'].iloc[-1]:.2f}", 
                    delta=f"{price_change:.2f}%",
                    delta_color="normal" if price_change >= 0 else "inverse"
                )
            else:
                st.metric(label="Price Change", value="N/A")
        
        # Sentiment distribution pie chart
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            values=[pos_pct, neu_pct, neg_pct],
            names=['Positive', 'Neutral', 'Negative'],
            color=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
            hole=0.4
        )
        fig_pie.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Correlation Key
        if correlation_results and 'correlations' in correlation_results:
            st.subheader("Correlation Results")
            
            # Find next-day returns correlation
            next_day_corr = next((c for c in correlation_results['correlations'] 
                                if c['sentiment_feature'] == 'sentiment_mean' and 
                                c['stock_feature'] == 'Return_Next'), None)
            
            if next_day_corr:
                corr_value = next_day_corr['correlation']
                p_value = next_day_corr['p_value']
                
                st.metric(
                    label="Sentiment → Next-Day Returns", 
                    value=f"{corr_value:.3f}",
                    delta="Significant" if p_value < 0.05 else "Not Significant",
                    delta_color="normal" if p_value < 0.05 else "off"
                )
                
                st.caption(f"p-value: {p_value:.4f}")
                
                if 'regression' in correlation_results:
                    r_squared = correlation_results['regression']['r_squared']
                    st.metric(
                        label="Predictive Power (R²)", 
                        value=f"{r_squared:.3f}",
                        delta=f"{r_squared*100:.1f}% of variance explained"
                    )
    else:
        st.error("No sentiment data available")

# Source distribution
st.subheader("Data Source Distribution")
if not sentiment_df.empty and 'source' in sentiment_df.columns:
    # Extract source type
    sentiment_df['source_type'] = sentiment_df['source'].apply(
        lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
    )
    
    # Get source distribution
    source_counts = sentiment_df['source_type'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    
    # Create bar chart
    fig = px.bar(
        source_counts, 
        x='Source', 
        y='Count',
        color='Source',
        labels={'Count': 'Number of Posts/Articles'},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No source data available")

# Volume and Sentiment Over Time
st.subheader("Post Volume and Sentiment Over Time")
if not daily_sentiment_df.empty:
    col3a, col3b = st.columns([3, 1])
    
    with col3a:
        # Create volume and sentiment chart
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=daily_sentiment_df['date'],
            y=daily_sentiment_df['total_count'],
            name='Post Volume',
            marker_color='lightgray',
            opacity=0.7
        ))
        
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=daily_sentiment_df['date'],
            y=daily_sentiment_df['sentiment_mean'],
            name='Sentiment',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        
        # Layout with dual y-axes
        fig.update_layout(
            yaxis=dict(
                title='Post Count',
                titlefont=dict(color='gray'),
                tickfont=dict(color='gray'),
                side='left'
            ),
            yaxis2=dict(
                title='Sentiment Score',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                overlaying='y',
                side='right',
                range=[-1, 1]
            ),
            xaxis=dict(title='Date'),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3b:
        # Display top sentiment days
        st.write("Peak Sentiment Days")
        
        # Get top positive and negative days
        top_positive = daily_sentiment_df.nlargest(3, 'sentiment_mean')[['date', 'sentiment_mean']]
        top_negative = daily_sentiment_df.nsmallest(3, 'sentiment_mean')[['date', 'sentiment_mean']]
        
        # Display as tables
        st.write("Most Positive Days:")
        st.dataframe(
            top_positive.rename(columns={'date': 'Date', 'sentiment_mean': 'Sentiment'}),
            use_container_width=True,
            hide_index=True
        )
        
        st.write("Most Negative Days:")
        st.dataframe(
            top_negative.rename(columns={'date': 'Date', 'sentiment_mean': 'Sentiment'}),
            use_container_width=True,
            hide_index=True
        )
else:
    st.error("No daily sentiment data available")

# Download section
st.subheader("Download Data")
col4a, col4b, col4c = st.columns(3)

with col4a:
    if not sentiment_df.empty:
        csv_data = sentiment_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sentiment Data",
            data=csv_data,
            file_name=f"{stock_symbol}_sentiment_data.csv",
            mime="text/csv"
        )

with col4b:
    if not daily_sentiment_df.empty:
        csv_data = daily_sentiment_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Daily Sentiment",
            data=csv_data,
            file_name=f"{stock_symbol}_daily_sentiment.csv",
            mime="text/csv"
        )

with col4c:
    if not merged_df.empty:
        csv_data = merged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Merged Data",
            data=csv_data,
            file_name=f"{stock_symbol}_merged_data.csv",
            mime="text/csv"
        )