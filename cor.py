import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

st.set_page_config(
    page_title="Correlation Analysis - Market Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Correlation Analysis")
st.write("Analyze the relationship between market sentiment and stock price movements")

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

# Ensure we have data
if merged_df.empty:
    st.error("No merged data available for correlation analysis.")
    st.stop()

# Header with stock info
st.header(f"{company_name} ({stock_symbol}) Correlation Analysis")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs([
    "Correlation Overview", 
    "Regression Analysis", 
    "Time-Lagged Analysis"
])

with tab1:
    st.subheader("Correlation Overview")
    
    # Display correlation summary
    if correlation_results and 'correlations' in correlation_results:
        # Create a dataframe from the correlations
        corr_df = pd.DataFrame(correlation_results['correlations'])
        
        # Clean up column names for display
        display_corr = corr_df.copy()
        display_corr['sentiment_feature'] = display_corr['sentiment_feature'].apply(
            lambda x: ' '.join(word.capitalize() for word in x.split('_'))
        )
        display_corr['stock_feature'] = display_corr['stock_feature'].apply(
            lambda x: ' '.join(word.capitalize() for word in x.split('_'))
        )
        
        # Format p-values
        display_corr['p_value'] = display_corr['p_value'].apply(
            lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}"
        )
        
        # Round correlation values
        display_corr['correlation'] = display_corr['correlation'].round(3)
        
        # Add significance indicator
        display_corr['significance'] = display_corr['p_value'].apply(
            lambda x: "Significant (p<0.05)" if float(x.replace("e-", "0")) < 0.05 else "Not Significant"
        )
        
        # Select columns to display
        display_corr = display_corr[['sentiment_feature', 'stock_feature', 'correlation', 'p_value', 'significance']]
        display_corr.columns = ['Sentiment Feature', 'Stock Feature', 'Correlation', 'P-Value', 'Significance']
        
        # Display the dataframe
        st.dataframe(display_corr, use_container_width=True, hide_index=True)
        
        # Create correlation heatmap
        st.subheader("Correlation Heatmap")
        
        # Pivot the data for the heatmap
        heatmap_data = corr_df.pivot(
            index='sentiment_feature', 
            columns='stock_feature', 
            values='correlation'
        )
        
        # Create the heatmap using Plotly
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdBu_r',
            zmin=-1, 
            zmax=1,
            labels=dict(x="Stock Feature", y="Sentiment Feature", color="Correlation"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text_auto='.2f'
        )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                title="Stock Feature",
                tickmode='array',
                tickvals=list(range(len(heatmap_data.columns))),
                ticktext=[col.replace('_', ' ').title() for col in heatmap_data.columns]
            ),
            yaxis=dict(
                title="Sentiment Feature",
                tickmode='array',
                tickvals=list(range(len(heatmap_data.index))),
                ticktext=[idx.replace('_', ' ').title() for idx in heatmap_data.index]
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No correlation data available")
    
    # Scatter plots
    st.subheader("Scatter Plots")
    
    if 'sentiment_mean' in merged_df.columns:
        # Select which return to analyze
        return_type = st.radio(
            "Select Return Type:",
            ["Same-Day Returns", "Next-Day Returns"],
            horizontal=True
        )
        
        return_col = 'Return' if return_type == "Same-Day Returns" else 'Return_Next'
        
        if return_col in merged_df.columns:
            # Create scatter plot
            fig = px.scatter(
                merged_df,
                x='sentiment_mean',
                y=return_col,
                trendline='ols',
                labels={
                    'sentiment_mean': 'Sentiment Score',
                    return_col: 'Return (%)'
                },
                title=f'Sentiment vs. {return_type}',
                color_discrete_sequence=['royalblue']
            )
            
            # Get correlation
            corr = merged_df[['sentiment_mean', return_col]].corr().iloc[0, 1]
            
            # Add annotation with correlation
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Correlation: {corr:.3f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Sentiment Score",
                yaxis_title="Return (%)",
                yaxis=dict(tickformat='.1%'),
                height=500
            )
            
            # Add horizontal and vertical reference lines
            fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
            fig.add_vline(x=0, line_dash="dot", line_color="black", line_width=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"{return_col} not available in the data")

with tab2:
    st.subheader("Regression Analysis")
    
    if 'sentiment_mean' in merged_df.columns and 'Return_Next' in merged_df.columns:
        # Display regression results if available
        if correlation_results and 'regression' in correlation_results:
            reg_results = correlation_results['regression']
            
            # Create regression summary
            reg_summary = pd.DataFrame({
                'Metric': [
                    'R-squared',
                    'Coefficient (Sentiment)',
                    'P-value',
                    'Sample Size'
                ],
                'Value': [
                    f"{reg_results['r_squared']:.4f}",
                    f"{reg_results['coefficient']:.6f}",
                    f"{reg_results['p_value']:.4f}" if reg_results['p_value'] >= 0.0001 else f"{reg_results['p_value']:.2e}",
                    f"{reg_results['sample_size']}"
                ]
            })
            
            st.dataframe(reg_summary, use_container_width=True, hide_index=True)
            
            # Interpretation
            st.markdown("### Interpretation")
            
            r_squared = reg_results['r_squared']
            coefficient = reg_results['coefficient']
            p_value = reg_results['p_value']
            
            # R-squared interpretation
            if r_squared < 0.1:
                r_squared_interp = "very weak"
            elif r_squared < 0.3:
                r_squared_interp = "weak"
            elif r_squared < 0.5:
                r_squared_interp = "moderate"
            else:
                r_squared_interp = "strong"
            
            # Significance interpretation
            if p_value < 0.05:
                significance = "statistically significant"
            else:
                significance = "not statistically significant"
            
            st.markdown(f"""
            - The R-squared value of {r_squared:.4f} indicates that sentiment explains about {r_squared*100:.1f}% of the variance in next-day returns, which is considered a {r_squared_interp} relationship.
            
            - The coefficient of {coefficient:.6f} means that a 1-point increase in sentiment score (e.g., from 0 to 1, which is a large change) is associated with a {abs(coefficient)*100:.2f}% {'increase' if coefficient > 0 else 'decrease'} in next-day returns.
            
            - This relationship is {significance} (p-value: {p_value:.4f if p_value >= 0.0001 else p_value:.2e}).
            
            - {'This suggests that sentiment could potentially be used as a predictor of stock price movements.' if p_value < 0.05 else 'This suggests that sentiment may not be a reliable predictor of stock price movements.'}
            """)
        
        # Run custom regression
        st.markdown("### Custom Regression Model")
        st.markdown("Create a custom regression model using available features:")
        
        # Select features
        available_features = [col for col in merged_df.columns if col.startswith(('sentiment_', 'positive_', 'negative_'))]
        selected_features = st.multiselect(
            "Select sentiment features to include in the model:",
            options=available_features,
            default=['sentiment_mean'] if 'sentiment_mean' in available_features else []
        )
        
        # Select target
        target_options = [col for col in merged_df.columns if col.startswith('Return')]
        selected_target = st.selectbox(
            "Select target variable:",
            options=target_options,
            index=target_options.index('Return_Next') if 'Return_Next' in target_options else 0
        )
        
        # Run regression button
        if st.button("Run Regression Model") and selected_features and selected_target:
            # Filter out rows with NaN values
            model_df = merged_df[selected_features + [selected_target]].dropna()
            
            if len(model_df) > 5:
                try:
                    # Add constant
                    X = sm.add_constant(model_df[selected_features])
                    y = model_df[selected_target]
                    
                    # Fit model
                    model = sm.OLS(y, X).fit()
                    
                    # Display results
                    st.text("Regression Results Summary:")
                    st.text(model.summary().as_text())
                    
                    # Create a cleaner summary
                    coef_df = pd.DataFrame({
                        'Feature': model.params.index,
                        'Coefficient': model.params.values,
                        'Std Error': model.bse.values,
                        'P-value': model.pvalues.values,
                        'Significance': ['\u2605' * (4 - int(min(4, p * 10))) if p < 0.05 else '' for p in model.pvalues]
                    })
                    
                    st.markdown("#### Coefficients")
                    st.markdown("*Note: More stars (\\*) indicate higher significance (p < 0.05).*")
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)
                    
                    # Model performance
                    st.markdown("#### Model Performance")
                    st.markdown(f"**R-squared:** {model.rsquared:.4f}")
                    st.markdown(f"**Adjusted R-squared:** {model.rsquared_adj:.4f}")
                    st.markdown(f"**F-statistic:** {model.fvalue:.4f}")
                    st.markdown(f"**Prob (F-statistic):** {model.f_pvalue:.4f}")
                    
                    # Predicted vs actual plot
                    st.markdown("#### Predicted vs Actual Values")
                    
                    # Get predicted values
                    y_pred = model.predict(X)
                    
                    # Create scatter plot
                    fig = px.scatter(
                        x=y_pred, 
                        y=y,
                        labels={'x': 'Predicted Returns', 'y': 'Actual Returns'},
                        title='Predicted vs Actual Returns',
                        trendline='ols'
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis=dict(tickformat='.1%'),
                        yaxis=dict(tickformat='.1%'),
                        height=400
                    )
                    
                    # Add diagonal line (perfect prediction)
                    min_val = min(min(y_pred), min(y))
                    max_val = max(max(y_pred), max(y))
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error running regression: {str(e)}")
            else:
                st.warning("Not enough data points for regression after removing missing values")
    else:
        st.warning("Required data for regression analysis not available")

with tab3:
    st.subheader("Time-Lagged Analysis")
    
    if 'sentiment_mean' in daily_sentiment_df.columns:
        # Create a copy of daily sentiment data
        lag_df = daily_sentiment_df.copy()
        
        # Get stock returns if available
        if not stock_df.empty and 'date' in stock_df.columns and 'Return' in stock_df.columns:
            stock_returns = stock_df[['date', 'Return']].copy()
            
            # Merge with sentiment data
            lag_df = pd.merge(lag_df, stock_returns, on='date', how='left')
            
            # Create lagged stock returns (future returns)
            for i in range(1, 6):
                lag_df[f'Return_lag_{i}'] = lag_df['Return'].shift(-i)
            
            # Select lag periods
            st.markdown("### Sentiment vs Future Returns")
            st.markdown("Analyze how sentiment correlates with returns on subsequent days:")
            
            # Compute correlations
            corr_data = []
            max_lag = 5
            
            for i in range(0, max_lag + 1):
                col_name = 'Return' if i == 0 else f'Return_lag_{i}'
                if col_name in lag_df.columns:
                    # Compute correlation
                    corr = lag_df[['sentiment_mean', col_name]].corr().iloc[0, 1]
                    
                    # Compute p-value
                    from scipy.stats import pearsonr
                    valid_data = lag_df[['sentiment_mean', col_name]].dropna()
                    _, p_value = pearsonr(valid_data['sentiment_mean'], valid_data[col_name])
                    
                    # Add to data
                    corr_data.append({
                        'Lag': i,
                        'Day': "Same day" if i == 0 else f"{i} day{'s' if i > 1 else ''} later",
                        'Correlation': corr,
                        'P-value': p_value,
                        'Significant': p_value < 0.05
                    })
            
            # Create DataFrame
            corr_df = pd.DataFrame(corr_data)
            
            # Display correlation table
            st.dataframe(
                corr_df.rename(columns={'Lag': 'Lag (Days)', 'Correlation': 'Correlation Coefficient'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Create bar chart of correlations
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=corr_df['Day'],
                y=corr_df['Correlation'],
                marker_color=['royalblue' if sig else 'lightgray' for sig in corr_df['Significant']],
                text=[f"{c:.3f}<br>{'p < 0.05' if s else 'Not sig.'}" for c, s in zip(corr_df['Correlation'], corr_df['Significant'])],
                textposition='auto'
            ))
            
            # Add zero reference line
            fig.add_hline(y=0, line_dash="dot", line_color="black")
            
            # Update layout
            fig.update_layout(
                title='Correlation between Today\'s Sentiment and Future Returns',
                xaxis_title='',
                yaxis_title='Correlation Coefficient',
                yaxis=dict(range=[-1, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot for best lag
            if len(corr_df) > 0:
                # Find lag with highest absolute correlation
                best_lag_idx = corr_df['Correlation'].abs().idxmax()
                best_lag = corr_df.iloc[best_lag_idx]['Lag']
                best_lag_day = corr_df.iloc[best_lag_idx]['Day']
                best_corr = corr_df.iloc[best_lag_idx]['Correlation']
                
                best_col = 'Return' if best_lag == 0 else f'Return_lag_{best_lag}'
                
                st.markdown(f"### Sentiment vs Returns ({best_lag_day})")
                st.markdown(f"Correlation: {best_corr:.3f}")
                
                # Create scatter plot
                fig = px.scatter(
                    lag_df.dropna(subset=['sentiment_mean', best_col]),
                    x='sentiment_mean',
                    y=best_col,
                    trendline='ols',
                    labels={
                        'sentiment_mean': 'Sentiment Score',
                        best_col: 'Return (%)'
                    },
                    title=f'Sentiment vs. Returns ({best_lag_day})'
                )
                
                # Update layout
                fig.update_layout(
                    yaxis=dict(tickformat='.1%'),
                    height=400
                )
                
                # Add reference lines
                fig.add_hline(y=0, line_dash="dot", line_color="black")
                fig.add_vline(x=0, line_dash="dot", line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Stock return data not available for lag analysis")
    else:
        st.warning("Required sentiment data not available for lag analysis")

# Additional analysis section
st.header("Additional Insights")

# Sentiment components correlation
if 'positive_ratio' in merged_df.columns and 'negative_ratio' in merged_df.columns and 'Return_Next' in merged_df.columns:
    st.subheader("Sentiment Component Analysis")
    
    # Create correlation heatmap for sentiment components
    component_cols = ['positive_ratio', 'negative_ratio', 'neutral_ratio', 'sentiment_mean']
    stock_cols = ['Return', 'Return_Next', 'Volatility_5']
    
    # Create new DataFrame with correlations
    component_corrs = []
    
    for comp_col in component_cols:
        if comp_col in merged_df.columns:
            for stock_col in stock_cols:
                if stock_col in merged_df.columns:
                    corr_data = merged_df[[comp_col, stock_col]].dropna()
                    if len(corr_data) > 5:
                        corr = corr_data.corr().iloc[0, 1]
                        
                        # Compute p-value
                        from scipy.stats import pearsonr
                        _, p_value = pearsonr(corr_data[comp_col], corr_data[stock_col])
                        
                        component_corrs.append({
                            'Component': comp_col.replace('_ratio', '').replace('_', ' ').title(),
                            'Stock Metric': stock_col.replace('_', ' ').title(),
                            'Correlation': corr,
                            'P-value': p_value,
                            'Significant': p_value < 0.05
                        })
    
    if component_corrs:
        # Convert to DataFrame
        component_corr_df = pd.DataFrame(component_corrs)
        
        # Create heatmap
        fig = px.imshow(
            component_corr_df.pivot(index='Component', columns='Stock Metric', values='Correlation'),
            color_continuous_scale='RdBu_r',
            zmin=-1, 
            zmax=1,
            text_auto='.2f'
        )
        
        # Update layout
        fig.update_layout(
            title='Correlation between Sentiment Components and Stock Metrics',
            xaxis_title="Stock Metric",
            yaxis_title="Sentiment Component",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("### Component Analysis Interpretation")
        
        # Find the component with the strongest correlation to future returns
        future_return_comps = component_corr_df[component_corr_df['Stock Metric'] == 'Return Next']
        if not future_return_comps.empty:
            best_comp_idx = future_return_comps['Correlation'].abs().idxmax()
            best_comp = future_return_comps.iloc[best_comp_idx]
            
            st.markdown(f"""
            - The sentiment component with the strongest correlation to next-day returns is **{best_comp['Component']}** with a correlation of {best_comp['Correlation']:.3f} ({"significant" if best_comp['Significant'] else "not significant"}).
            
            - {"This suggests that focusing on this specific sentiment component might provide better predictive power than the overall sentiment score." if best_comp['Significant'] else "However, the correlation is not statistically significant, so this component may not be reliable for prediction."}
            """)
            
            # Compare positive vs negative
            pos_corr = future_return_comps[future_return_comps['Component'] == 'Positive']['Correlation'].values
            neg_corr = future_return_comps[future_return_comps['Component'] == 'Negative']['Correlation'].values
            
            if len(pos_corr) > 0 and len(neg_corr) > 0:
                st.markdown(f"""
                - Comparing positive and negative sentiment components: 
                  - Positive sentiment correlation with next-day returns: {pos_corr[0]:.3f}
                  - Negative sentiment correlation with next-day returns: {neg_corr[0]:.3f}
                
                - {"Positive sentiment appears to be a stronger signal than negative sentiment." if abs(pos_corr[0]) > abs(neg_corr[0]) else "Negative sentiment appears to be a stronger signal than positive sentiment."}
                """)