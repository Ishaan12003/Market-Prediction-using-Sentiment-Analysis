import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(
    page_title="Sentiment Details - Market Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Sentiment Analysis Details")
st.write("Deep dive into sentiment analysis results and trends")

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
stock_symbol = results['stock_symbol']
company_name = results['company_name']
start_date = results['start_date']
end_date = results['end_date']

# Ensure we have data
if sentiment_df.empty:
    st.error("No sentiment data available for analysis.")
    st.stop()

# Header with stock info
st.header(f"{company_name} ({stock_symbol}) Sentiment Analysis")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment Distribution", 
    "Sentiment Over Time", 
    "Source Analysis", 
    "Keyword Analysis"
])

with tab1:
    st.subheader("Sentiment Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create histogram with KDE
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=sentiment_df['sentiment_score'],
            name='Frequency',
            nbinsx=30,
            marker_color='royalblue',
            opacity=0.7
        ))
        
        # Try to add KDE (using numpy's histogram and scipy's gaussian_kde)
        try:
            kde_x = np.linspace(-1, 1, 100)
            kde = stats.gaussian_kde(sentiment_df['sentiment_score'].dropna())
            kde_y = kde(kde_x) * len(sentiment_df) * (2 / 30)  # Scale to match histogram
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='Density',
                line=dict(color='red', width=2)
            ))
        except:
            st.warning("Could not compute density estimation.")
        
        # Add vertical line for mean
        mean_sentiment = sentiment_df['sentiment_score'].mean()
        fig.add_vline(
            x=mean_sentiment,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: {mean_sentiment:.3f}",
            annotation_position="top right"
        )
        
        # Add vertical line at zero
        fig.add_vline(
            x=0,
            line_dash="dot",
            line_color="black",
            line_width=1
        )
        
        # Update layout
        fig.update_layout(
            title='Distribution of Sentiment Scores',
            xaxis_title='Sentiment Score',
            yaxis_title='Frequency',
            bargap=0.05,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment statistics
        st.subheader("Sentiment Statistics")
        
        # Create a stats table
        stats_df = pd.DataFrame({
            'Metric': [
                'Mean Sentiment',
                'Median Sentiment',
                'Standard Deviation',
                'Minimum',
                'Maximum',
                'Positive Posts (%)',
                'Negative Posts (%)',
                'Neutral Posts (%)',
                'Total Posts'
            ],
            'Value': [
                f"{sentiment_df['sentiment_score'].mean():.3f}",
                f"{sentiment_df['sentiment_score'].median():.3f}",
                f"{sentiment_df['sentiment_score'].std():.3f}",
                f"{sentiment_df['sentiment_score'].min():.3f}",
                f"{sentiment_df['sentiment_score'].max():.3f}",
                f"{(sentiment_df['sentiment_label'] == 'positive').mean() * 100:.1f}%",
                f"{(sentiment_df['sentiment_label'] == 'negative').mean() * 100:.1f}%",
                f"{(sentiment_df['sentiment_label'] == 'neutral').mean() * 100:.1f}%",
                f"{len(sentiment_df)}"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Add sentiment box plots by type
        if 'type' in sentiment_df.columns:
            st.subheader("Sentiment by Post Type")
            
            fig = px.box(
                sentiment_df,
                x='type',
                y='sentiment_score',
                color='type',
                title='Sentiment Distribution by Post Type',
                points='all',
                notched=True
            )
            
            fig.update_layout(
                xaxis_title='Post Type',
                yaxis_title='Sentiment Score',
                yaxis=dict(range=[-1, 1]),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment label distribution
        st.subheader("Sentiment Labels")
        
        sentiment_counts = sentiment_df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Label', 'Count']
        
        fig = px.pie(
            sentiment_counts,
            values='Count',
            names='Label',
            color='Label',
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'},
            hole=0.4
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Sentiment Trends Over Time")
    
    # Time aggregation selector
    time_agg = st.radio(
        "Time Aggregation:",
        ["Daily", "Weekly"],
        horizontal=True
    )
    
    # Create time series based on selected aggregation
    if time_agg == "Daily":
        time_series_df = daily_sentiment_df.copy()
    else:  # Weekly
        time_series_df = daily_sentiment_df.copy()
        time_series_df['week'] = time_series_df['date'].dt.to_period('W').dt.start_time
        time_series_df = time_series_df.groupby('week').agg({
            'sentiment_mean': 'mean',
            'sentiment_median': 'mean',
            'positive_ratio': 'mean',
            'negative_ratio': 'mean',
            'neutral_ratio': 'mean',
            'total_count': 'sum',
            'reddit_count': 'sum',
            'news_count': 'sum'
        }).reset_index()
        time_series_df = time_series_df.rename(columns={'week': 'date'})
    
    # Create time series plot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=time_series_df['date'],
        y=time_series_df['sentiment_mean'],
        mode='lines+markers',
        name='Mean Sentiment',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_df['date'],
        y=time_series_df['sentiment_median'],
        mode='lines+markers',
        name='Median Sentiment',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="black",
        line_width=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{time_agg} Sentiment Trends',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1, 1]),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment components stacked area chart
    st.subheader("Sentiment Components Over Time")
    
    fig = go.Figure()
    
    # Add traces for positive, neutral, and negative components
    fig.add_trace(go.Scatter(
        x=time_series_df['date'],
        y=time_series_df['positive_ratio'],
        mode='lines',
        name='Positive',
        line=dict(width=0.5, color='green'),
        stackgroup='one',
        groupnorm='fraction'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_df['date'],
        y=time_series_df['neutral_ratio'],
        mode='lines',
        name='Neutral',
        line=dict(width=0.5, color='gray'),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_df['date'],
        y=time_series_df['negative_ratio'],
        mode='lines',
        name='Negative',
        line=dict(width=0.5, color='red'),
        stackgroup='one'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Sentiment Component Ratios Over Time',
        xaxis_title='Date',
        yaxis_title='Proportion',
        yaxis=dict(tickformat='.0%'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Post volume with sentiment overlay
    st.subheader("Post Volume and Sentiment")
    
    fig = go.Figure()
    
    # Add post volume bar chart
    fig.add_trace(go.Bar(
        x=time_series_df['date'],
        y=time_series_df['total_count'],
        name='Total Posts',
        marker_color='lightgray',
        opacity=0.7
    ))
    
    # Add sentiment line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=time_series_df['date'],
        y=time_series_df['sentiment_mean'],
        mode='lines+markers',
        name='Mean Sentiment',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title='Post Volume and Sentiment Over Time',
        xaxis_title='Date',
        yaxis=dict(
            title='Number of Posts',
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
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode="x unified",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Source Sentiment Analysis")
    
    # Extract source type
    if 'source' in sentiment_df.columns:
        sentiment_df['source_type'] = sentiment_df['source'].apply(
            lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
        )
        
        # Group by source and sentiment
        source_sentiment = sentiment_df.groupby(['source_type', 'sentiment_label']).size().reset_index(name='count')
        
        # Create grouped bar chart
        fig = px.bar(
            source_sentiment,
            x='source_type',
            y='count',
            color='sentiment_label',
            barmode='group',
            color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'},
            title='Sentiment by Source Type'
        )
        
        fig.update_layout(
            xaxis_title='Source',
            yaxis_title='Number of Posts',
            legend_title='Sentiment',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Source sentiment statistics
        source_stats = sentiment_df.groupby('source_type').agg(
            count=('id', 'count'),
            avg_sentiment=('sentiment_score', 'mean'),
            std_sentiment=('sentiment_score', 'std'),
            pos_pct=('sentiment_label', lambda x: (x == 'positive').mean() * 100),
            neg_pct=('sentiment_label', lambda x: (x == 'negative').mean() * 100),
            neu_pct=('sentiment_label', lambda x: (x == 'neutral').mean() * 100)
        ).reset_index()
        
        # Rename columns for display
        source_stats.columns = ['Source Type', 'Post Count', 'Avg Sentiment', 'Std Dev', 
                               '% Positive', '% Negative', '% Neutral']
        
        # Format numeric columns
        source_stats['Avg Sentiment'] = source_stats['Avg Sentiment'].round(3)
        source_stats['Std Dev'] = source_stats['Std Dev'].round(3)
        source_stats['% Positive'] = source_stats['% Positive'].round(1)
        source_stats['% Negative'] = source_stats['% Negative'].round(1)
        source_stats['% Neutral'] = source_stats['% Neutral'].round(1)
        
        # Display stats table
        st.dataframe(source_stats, use_container_width=True, hide_index=True)
        
        # Source comparison box plot
        fig = px.box(
            sentiment_df,
            x='source_type',
            y='sentiment_score',
            color='source_type',
            title='Sentiment Distribution by Source Type',
            notched=True
        )
        
        fig.update_layout(
            xaxis_title='Source Type',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No source data available")

with tab4:
    st.subheader("Keyword Analysis")
    
    # Function to extract top keywords
    def extract_keywords(text_series, min_count=5):
        # Combine all text
        if text_series.empty:
            return pd.DataFrame()
            
        all_text = ' '.join(text_series.dropna().astype(str))
        
        # Split into words and convert to lowercase
        words = all_text.lower().split()
        
        # Remove short words and common stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 
                         'by', 'about', 'as', 'of', 'that', 'this', 'these', 'those', 'is', 'are', 
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                         'did', 'not', 'it', 'its', 'i', 'we', 'you', 'they', 'he', 'she', 'him', 
                         'her', 'them', 'my', 'our', 'your', 'their', 'from', 'will', 'would', 'should', 
                         'could', 'can', 'all', 'any', 'some', 'no', 'nor', 'if', 'then', 'so', 'just',
                         'than', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'now', 'here',
                         'there', 'when', 'where', 'why', 'how', 'what', 'who', 'which', 'more'])
        
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Convert to DataFrame and filter by minimum count
        word_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
        word_df = word_df[word_df['count'] >= min_count].sort_values('count', ascending=False)
        
        return word_df
    
    # Extract keywords by sentiment
    positive_df = sentiment_df[sentiment_df['sentiment_label'] == 'positive']
    negative_df = sentiment_df[sentiment_df['sentiment_label'] == 'negative']
    
    # Get text column to analyze
    text_col = 'full_text'
    if text_col not in sentiment_df.columns:
        text_col = 'content'
        if text_col not in sentiment_df.columns:
            text_col = 'title'
    
    if text_col in sentiment_df.columns:
        # Extract keywords
        min_word_count = st.slider("Minimum word frequency:", 3, 20, 5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Post Keywords")
            positive_keywords = extract_keywords(positive_df[text_col], min_count=min_word_count)
            
            if not positive_keywords.empty and len(positive_keywords) > 0:
                # Create word cloud-like visualization
                fig = px.bar(
                    positive_keywords.head(20),
                    x='count',
                    y='word',
                    orientation='h',
                    color='count',
                    color_continuous_scale='Greens',
                    title='Top Keywords in Positive Posts'
                )
                
                fig.update_layout(
                    xaxis_title='Frequency',
                    yaxis_title='',
                    yaxis=dict(categoryorder='total ascending'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to extract positive keywords")
        
        with col2:
            st.subheader("Negative Post Keywords")
            negative_keywords = extract_keywords(negative_df[text_col], min_count=min_word_count)
            
            if not negative_keywords.empty and len(negative_keywords) > 0:
                # Create word cloud-like visualization
                fig = px.bar(
                    negative_keywords.head(20),
                    x='count',
                    y='word',
                    orientation='h',
                    color='count',
                    color_continuous_scale='Reds',
                    title='Top Keywords in Negative Posts'
                )
                
                fig.update_layout(
                    xaxis_title='Frequency',
                    yaxis_title='',
                    yaxis=dict(categoryorder='total ascending'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to extract negative keywords")
        
        # Keywords in common
        if not positive_keywords.empty and not negative_keywords.empty:
            st.subheader("Keyword Comparison")
            
            # Get common words
            common_words = set(positive_keywords['word']).intersection(set(negative_keywords['word']))
            
            if common_words:
                # Create a comparison dataframe
                comparison_df = pd.DataFrame()
                
                for word in common_words:
                    pos_count = positive_keywords[positive_keywords['word'] == word]['count'].values[0]
                    neg_count = negative_keywords[negative_keywords['word'] == word]['count'].values[0]
                    
                    comparison_df = pd.concat([comparison_df, pd.DataFrame({
                        'word': [word],
                        'positive_count': [pos_count],
                        'negative_count': [neg_count],
                        'total_count': [pos_count + neg_count],
                        'positive_ratio': [pos_count / (pos_count + neg_count)]
                    })])
                
                # Sort by total count
                comparison_df = comparison_df.sort_values('total_count', ascending=False)
                
                # Create comparison chart
                fig = go.Figure()
                
                # Add bars for positive and negative counts
                fig.add_trace(go.Bar(
                    y=comparison_df['word'].head(15),
                    x=comparison_df['positive_count'].head(15),
                    name='Positive Posts',
                    orientation='h',
                    marker_color='green'
                ))
                
                fig.add_trace(go.Bar(
                    y=comparison_df['word'].head(15),
                    x=-comparison_df['negative_count'].head(15),
                    name='Negative Posts',
                    orientation='h',
                    marker_color='red'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Common Keywords in Positive vs Negative Posts',
                    xaxis_title='Frequency',
                    yaxis=dict(categoryorder='total ascending'),
                    barmode='relative',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No common keywords found between positive and negative posts")
    else:
        st.warning("No text data available for keyword analysis")