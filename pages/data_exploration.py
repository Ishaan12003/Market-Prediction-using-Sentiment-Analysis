import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

st.set_page_config(
    page_title="Data Exploration - Market Sentiment Analysis",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Data Exploration")
st.write("Explore the collected social media and news data")

# Check if analysis has been run
if 'analysis_results' not in st.session_state:
    st.warning("Please run an analysis first from the home page.")
    st.stop()

# Get data from session state
results = st.session_state['analysis_results']
sentiment_df = results['sentiment_df']
daily_sentiment_df = results['daily_sentiment_df']
stock_symbol = results['stock_symbol']
company_name = results['company_name']

# Ensure we have data
if sentiment_df.empty:
    st.error("No sentiment data available for exploration.")
    st.stop()

# Header with stock info
st.header(f"{company_name} ({stock_symbol}) Data Exploration")

# Sidebar filters
st.sidebar.header("Data Filters")

# Filter by source
source_types = sentiment_df['source'].apply(
    lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
).unique()

selected_sources = st.sidebar.multiselect(
    "Select Data Sources",
    options=source_types,
    default=source_types
)

# Filter by sentiment
sentiment_options = ['positive', 'neutral', 'negative']
selected_sentiments = st.sidebar.multiselect(
    "Select Sentiment Types",
    options=sentiment_options,
    default=sentiment_options
)

# Filter by date range
date_range = st.sidebar.date_input(
    "Date Range",
    value=[sentiment_df['timestamp'].min().date(), sentiment_df['timestamp'].max().date()],
    min_value=sentiment_df['timestamp'].min().date(),
    max_value=sentiment_df['timestamp'].max().date()
)

# Apply filters
filtered_df = sentiment_df.copy()

# Source filter
if selected_sources:
    filtered_df = filtered_df[filtered_df['source'].apply(
        lambda x: x.split(':')[0] if isinstance(x, str) and ':' in x else x
    ).isin(selected_sources)]

# Sentiment filter
if selected_sentiments:
    filtered_df = filtered_df[filtered_df['sentiment_label'].isin(selected_sentiments)]

# Date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.date >= start_date) & 
        (filtered_df['timestamp'].dt.date <= end_date)
    ]

# Display filter summary
st.write(f"Displaying {len(filtered_df)} records matching selected filters")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Posts Timeline", "Source Analysis", "Content Analysis"])

with tab1:
    st.subheader("Data Overview")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Count by type
        type_counts = filtered_df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        fig = px.pie(
            type_counts,
            values='Count',
            names='Type',
            title='Data by Type',
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        sentiment_counts = filtered_df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        color_map = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        
        fig = px.bar(
            sentiment_counts,
            x='Sentiment',
            y='Count',
            color='Sentiment',
            color_discrete_map=color_map,
            title='Sentiment Distribution'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Score distribution (for Reddit)
        if 'score' in filtered_df.columns:
            reddit_df = filtered_df[filtered_df['type'].isin(['submission', 'comment'])]
            
            if not reddit_df.empty:
                fig = px.histogram(
                    reddit_df,
                    x='score',
                    nbins=20,
                    title='Post Score Distribution',
                    color_discrete_sequence=['royalblue']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Data table with search
    st.subheader("Data Table")
    search_term = st.text_input("Search in posts:", "")
    
    # Columns to display
    display_cols = ['timestamp', 'type', 'source', 'title', 'sentiment_score', 'sentiment_label']
    
    # Apply search filter if provided
    if search_term:
        search_pattern = re.compile(search_term, re.IGNORECASE)
        display_df = filtered_df[
            filtered_df['full_text'].apply(lambda x: bool(search_pattern.search(str(x))) if pd.notna(x) else False)
        ][display_cols]
        st.write(f"Found {len(display_df)} posts containing '{search_term}'")
    else:
        display_df = filtered_df[display_cols]
    
    # Display paginated table
    page_size = st.selectbox("Rows per page:", [10, 25, 50, 100])
    total_pages = (len(display_df) - 1) // page_size + 1
    
    if total_pages > 0:
        page_num = st.slider("Page:", 1, total_pages, 1)
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(display_df))
        
        st.dataframe(
            display_df.iloc[start_idx:end_idx].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.write("No data to display")

with tab2:
    st.subheader("Posts Timeline")
    
    # Select timeline resolution
    time_resolution = st.radio(
        "Time Resolution:",
        ["Hourly", "Daily", "Weekly"],
        horizontal=True
    )
    
    # Group data by selected time resolution
    if time_resolution == "Hourly":
        timeline_df = filtered_df.copy()
        timeline_df['time_bucket'] = timeline_df['timestamp'].dt.floor('H')
        group_col = 'time_bucket'
    elif time_resolution == "Daily":
        timeline_df = filtered_df.copy()
        timeline_df['time_bucket'] = timeline_df['timestamp'].dt.floor('d')
        group_col = 'time_bucket'
    else:  # Weekly
        timeline_df = filtered_df.copy()
        timeline_df['time_bucket'] = timeline_df['timestamp'].dt.to_period('W').dt.start_time
        group_col = 'time_bucket'
    
    # Group by time and type
    timeline_counts = timeline_df.groupby([group_col, 'type']).size().reset_index(name='count')
    
    # Create a pivot table
    pivot_timeline = timeline_counts.pivot(index=group_col, columns='type', values='count').fillna(0)
    
    # Ensure all types are in the pivot
    for type_name in timeline_df['type'].unique():
        if type_name not in pivot_timeline.columns:
            pivot_timeline[type_name] = 0
    
    # Reset index to make the time column available
    pivot_timeline = pivot_timeline.reset_index()
    
    # Create timeline chart
    fig = go.Figure()
    
    # Add traces for each type
    for type_name in timeline_df['type'].unique():
        if type_name in pivot_timeline.columns:
            fig.add_trace(go.Scatter(
                x=pivot_timeline[group_col],
                y=pivot_timeline[type_name],
                mode='lines+markers',
                name=type_name.title(),
                hovertemplate='%{y} posts'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{time_resolution} Post Frequency",
        xaxis_title="Date/Time",
        yaxis_title="Number of Posts",
        legend_title="Post Type",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add sentiment overlay option
    if st.checkbox("Show sentiment overlay"):
        # Calculate average sentiment by time bucket
        sentiment_timeline = timeline_df.groupby(group_col)['sentiment_score'].mean().reset_index()
        
        # Create sentiment overlay chart
        fig = go.Figure()
        
        # Post count bars (sum all types)
        post_counts = timeline_df.groupby(group_col).size().reset_index(name='count')
        
        fig.add_trace(go.Bar(
            x=post_counts[group_col],
            y=post_counts['count'],
            name='Post Count',
            marker_color='lightgray',
            opacity=0.7
        ))
        
        # Sentiment line
        fig.add_trace(go.Scatter(
            x=sentiment_timeline[group_col],
            y=sentiment_timeline['sentiment_score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            title=f"{time_resolution} Post Count and Sentiment",
            xaxis_title="Date/Time",
            yaxis=dict(
                title="Post Count",
                titlefont=dict(color='gray'),
                tickfont=dict(color='gray'),
                side='left'
            ),
            yaxis2=dict(
                title="Sentiment Score",
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                overlaying='y',
                side='right',
                range=[-1, 1]
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Source Analysis")
    
    # Extract complete source (not just type)
    source_stats = filtered_df.groupby('source').agg(
        count=('id', 'count'),
        avg_sentiment=('sentiment_score', 'mean'),
        pos_ratio=('sentiment_label', lambda x: (x == 'positive').mean()),
        neg_ratio=('sentiment_label', lambda x: (x == 'negative').mean()),
        neu_ratio=('sentiment_label', lambda x: (x == 'neutral').mean())
    ).reset_index()
    
    # Sort by count
    source_stats = source_stats.sort_values('count', ascending=False)
    
    # Display top sources table
    st.write("Top Sources:")
    
    # Format the table
    display_source_stats = source_stats.copy()
    display_source_stats.columns = ['Source', 'Post Count', 'Avg Sentiment', 
                                   '% Positive', '% Negative', '% Neutral']
    
    # Convert ratios to percentages
    for col in ['% Positive', '% Negative', '% Neutral']:
        display_source_stats[col] = (display_source_stats[col] * 100).round(1)
    
    # Display as a dataframe
    st.dataframe(
        display_source_stats.head(10),
        use_container_width=True
    )
    
    # Source comparison chart
    st.write("Source Comparison:")
    
    # Get top 10 sources by count
    top_sources = source_stats.head(10)
    
    # Create a sentiment by source bar chart
    fig = go.Figure()
    
    # Add bars for positive, neutral, and negative sentiment
    fig.add_trace(go.Bar(
        x=top_sources['source'],
        y=top_sources['pos_ratio'],
        name='Positive',
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        x=top_sources['source'],
        y=top_sources['neu_ratio'],
        name='Neutral',
        marker_color='gray'
    ))
    
    fig.add_trace(go.Bar(
        x=top_sources['source'],
        y=top_sources['neg_ratio'],
        name='Negative',
        marker_color='red'
    ))
    
    # Update layout
    fig.update_layout(
        title="Sentiment Distribution by Source",
        xaxis_title="Source",
        yaxis_title="Proportion",
        barmode='stack',
        height=500,
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Source bubble chart
    fig = px.scatter(
        top_sources,
        x='avg_sentiment',
        y='count',
        size='count',
        color='avg_sentiment',
        hover_name='source',
        color_continuous_scale='RdBu',
        size_max=50,
        range_color=[-1, 1],
        title="Sources by Volume and Sentiment"
    )
    
    fig.update_layout(
        xaxis_title="Average Sentiment",
        yaxis_title="Number of Posts",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Content Analysis")
    
    # Post view
    st.write("View Full Post Content:")
    
    # Sort options
    sort_options = {
        "Most Recent": ("timestamp", False),
        "Oldest First": ("timestamp", True),
        "Most Positive": ("sentiment_score", False),
        "Most Negative": ("sentiment_score", True),
        "Highest Score": ("score", False)
    }
    
    sort_by = st.selectbox("Sort by:", list(sort_options.keys()))
    sort_col, ascending = sort_options[sort_by]
    
    sorted_df = filtered_df.sort_values(sort_col, ascending=ascending)
    
    # Content viewer
    if not sorted_df.empty:
        # Select a post to view
        post_selector = st.selectbox(
            "Select a post to view:",
            options=range(len(sorted_df)),
            format_func=lambda x: f"{sorted_df.iloc[x]['type'].title()} from {sorted_df.iloc[x]['source']} - {sorted_df.iloc[x]['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Display selected post
        post = sorted_df.iloc[post_selector]
        
        # Create a card-like display
        st.markdown("---")
        
        # Header with metadata
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {post['title']}")
            st.markdown(f"**Source:** {post['source']} | **Posted on:** {post['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            
            if 'username' in post and pd.notna(post['username']):
                st.markdown(f"**Author:** {post['username']}")
        
        with col2:
            # Display sentiment with color
            sentiment_color = {
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            }.get(post['sentiment_label'], 'black')
            
            st.markdown(f"**Sentiment Score:** <span style='color:{sentiment_color}'>{post['sentiment_score']:.3f}</span>", unsafe_allow_html=True)
            
            if 'score' in post and pd.notna(post['score']):
                st.markdown(f"**Score:** {post['score']}")
            
            if 'url' in post and pd.notna(post['url']):
                st.markdown(f"[Original Link]({post['url']})")
        
        # Content
        if 'content' in post and pd.notna(post['content']) and post['content'].strip():
            st.markdown("### Content:")
            st.markdown(f"<div style='background-color:#f5f5f5; padding:10px; border-radius:5px;'>{post['content']}</div>", unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if post_selector > 0:
                if st.button("⬅️ Previous Post"):
                    st.experimental_rerun()
        
        with col2:
            if post_selector < len(sorted_df) - 1:
                if st.button("Next Post ➡️"):
                    st.experimental_rerun()
    else:
        st.warning("No posts to display with the current filters")