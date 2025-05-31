import io
import base64
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def generate_html_report(sentiment_df, daily_sentiment_df, stock_df, merged_df,
                       correlation_results, visualizations,
                       stock_symbol, company_name, start_date, end_date):
    """
    Generate an HTML report with all analysis results.

    Args:
        sentiment_df (pd.DataFrame): Sentiment DataFrame
        daily_sentiment_df (pd.DataFrame): Daily sentiment DataFrame
        stock_df (pd.DataFrame): Stock DataFrame
        merged_df (pd.DataFrame): Merged DataFrame
        correlation_results (dict): Correlation results
        visualizations (list): List of (title, figure) tuples
        stock_symbol (str): Stock symbol
        company_name (str): Company name
        start_date (datetime): Start date of analysis
        end_date (datetime): End date of analysis

    Returns:
        str: HTML report content
    """
    # Format dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Calculate overall sentiment statistics
    if not sentiment_df.empty:
        total_posts = len(sentiment_df)
        positive_pct = (sentiment_df['sentiment_label'] == 'positive').mean() * 100
        negative_pct = (sentiment_df['sentiment_label'] == 'negative').mean() * 100
        neutral_pct = (sentiment_df['sentiment_label'] == 'neutral').mean() * 100
        avg_sentiment = sentiment_df['sentiment_score'].mean()
    else:
        total_posts = 0
        positive_pct = 0
        negative_pct = 0
        neutral_pct = 0
        avg_sentiment = 0

    # Generate correlation summary
    corr_summary = ""
    if correlation_results and 'correlations' in correlation_results:
        for corr in correlation_results['correlations']:
            if corr['stock_feature'] == 'Return_Next' and corr['sentiment_feature'] == 'sentiment_mean':
                # Format p-value with scientific notation if small
                p_value = corr['p_value']
                if p_value < 0.001:
                    p_value_str = f"{p_value:.2e}"
                else:
                    p_value_str = f"{p_value:.3f}"

                corr_summary += f"""
                <tr>
                    <td>Sentiment vs. Next-Day Returns</td>
                    <td>{corr['correlation']:.3f}</td>
                    <td>{p_value_str}</td>
                    <td>{"Significant" if p_value < 0.05 else "Not Significant"}</td>
                </tr>
                """

    # Generate regression summary
    reg_summary = ""
    if correlation_results and 'regression' in correlation_results:
        reg = correlation_results['regression']
        reg_summary += f"""
        <tr>
            <td>R-Squared</td>
            <td>{reg['r_squared']:.3f}</td>
        </tr>
        <tr>
            <td>Coefficient</td>
            <td>{reg['coefficient']:.5f}</td>
        </tr>
        <tr>
            <td>P-Value</td>
            <td>{reg['p_value']:.4f}</td>
        </tr>
        <tr>
            <td>Sample Size</td>
            <td>{reg['sample_size']}</td>
        </tr>
        """

    # Convert figures to base64 images
    images = []
    for title, fig in visualizations:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        images.append((title, img_str))

    # HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{company_name} ({stock_symbol}) Sentiment Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
                text-align: center;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .summary-box {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #2c3e50;
                margin-bottom: 20px;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                font-size: 0.9em;
                color: #777;
                border-top: 1px solid #ddd;
            }}
            .positive {{
                color: green;
            }}
            .negative {{
                color: red;
            }}
            .neutral {{
                color: gray;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{company_name} ({stock_symbol}) Sentiment Analysis</h1>
                <p>Analysis period: {start_date_str} to {end_date_str}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>This report analyzes social media and news sentiment for {company_name} ({stock_symbol})
                    over the period from {start_date_str} to {end_date_str}, and examines the relationship
                    between sentiment and stock price movements.</p>

                    <h3>Key Findings:</h3>
                    <ul>
                        <li>Analyzed <strong>{total_posts}</strong> social media posts and news articles</li>
                        <li>Overall sentiment distribution:
                            <span class="positive">{positive_pct:.1f}% Positive</span>,
                            <span class="negative">{negative_pct:.1f}% Negative</span>,
                            <span class="neutral">{neutral_pct:.1f}% Neutral</span>
                        </li>
                        <li>Average sentiment score: <strong>{avg_sentiment:.3f}</strong> (on a scale from -1 to 1)</li>

                        {f"<li>Correlation between sentiment and next-day returns: <strong>{correlation_results['correlations'][0]['correlation']:.3f}</strong></li>"
                          if correlation_results and 'correlations' in correlation_results and correlation_results['correlations'] else ""}

                        {f"<li>Predictive power (R-squared): <strong>{correlation_results['regression']['r_squared']:.3f}</strong></li>"
                          if correlation_results and 'regression' in correlation_results else ""}
                    </ul>
                </div>
            </div>

            <div class="section">
                <h2>Sentiment Analysis Results</h2>
                <p>Overview of sentiment for {company_name} across social media and news sources:</p>

                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Posts Analyzed</td>
                        <td>{total_posts}</td>
                    </tr>
                    <tr>
                        <td>Positive Sentiment</td>
                        <td>{positive_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Negative Sentiment</td>
                        <td>{negative_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Neutral Sentiment</td>
                        <td>{neutral_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Average Sentiment Score</td>
                        <td>{avg_sentiment:.3f}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2>Correlation Analysis</h2>
                <p>Relationship between sentiment and stock price movements:</p>

                <table>
                    <tr>
                        <th>Relationship</th>
                        <th>Correlation</th>
                        <th>P-Value</th>
                        <th>Significance</th>
                    </tr>
                    {corr_summary if corr_summary else "<tr><td colspan='4'>No correlation data available</td></tr>"}
                </table>

                <h3>Regression Analysis</h3>
                <p>Predicting next-day returns based on sentiment:</p>

                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {reg_summary if reg_summary else "<tr><td colspan='2'>No regression data available</td></tr>"}
                </table>

                <div class="summary-box">
                    <h3>Interpretation:</h3>
                    <p>
                    {
                        f"The correlation between sentiment and next-day returns is {correlation_results['correlations'][0]['correlation']:.3f}, "
                        f"which indicates a {'strong' if abs(correlation_results['correlations'][0]['correlation']) > 0.5 else 'moderate' if abs(correlation_results['correlations'][0]['correlation']) > 0.3 else 'weak'} "
                        f"{'positive' if correlation_results['correlations'][0]['correlation'] > 0 else 'negative'} relationship. "
                        f"This correlation is {'statistically significant' if correlation_results['correlations'][0]['p_value'] < 0.05 else 'not statistically significant'} "
                        f"(p-value: {correlation_results['correlations'][0]['p_value']:.4f})."
                        if correlation_results and 'correlations' in correlation_results and correlation_results['correlations'] else
                        "No correlation data available for interpretation."
                    }
                    </p>

                    <p>
                    {
                        f"The R-squared value of {correlation_results['regression']['r_squared']:.3f} indicates that sentiment explains "
                        f"about {correlation_results['regression']['r_squared']*100:.1f}% of the variance in next-day returns. "
                        f"The coefficient of {correlation_results['regression']['coefficient']:.5f} means that a 1-point increase in sentiment "
                        f"is associated with a {abs(correlation_results['regression']['coefficient'])*100:.2f}% {'increase' if correlation_results['regression']['coefficient'] > 0 else 'decrease'} in next-day returns."
                        if correlation_results and 'regression' in correlation_results else
                        "No regression data available for interpretation."
                    }
                    </p>
                </div>
            </div>

            <div class="section">
                <h2>Visualizations</h2>

                {
                    ''.join([
                        f'''
                        <div class="chart-container">
                            <h3>{title}</h3>
                            <img src="data:image/png;base64,{img_str}" alt="{title}" style="max-width:100%;">
                        </div>
                        '''
                        for title, img_str in images
                    ])
                }
            </div>

            <div class="section">
                <h2>Conclusion</h2>
                <p>
                {
                    f"Based on the analysis of {total_posts} posts over the period from {start_date_str} to {end_date_str}, "
                    f"the overall sentiment for {company_name} ({stock_symbol}) was {'predominantly positive' if positive_pct > negative_pct + 10 else 'predominantly negative' if negative_pct > positive_pct + 10 else 'relatively balanced'}. "
                    f"The correlation between sentiment and next-day stock returns was {correlation_results['correlations'][0]['correlation']:.3f}, "
                    f"suggesting a {'strong' if abs(correlation_results['correlations'][0]['correlation']) > 0.5 else 'moderate' if abs(correlation_results['correlations'][0]['correlation']) > 0.3 else 'weak'} "
                    f"{'positive' if correlation_results['correlations'][0]['correlation'] > 0 else 'negative'} relationship. "
                    if correlation_results and 'correlations' in correlation_results and correlation_results['correlations'] else
                    f"Based on the analysis of {total_posts} posts over the period from {start_date_str} to {end_date_str}, "
                    f"the overall sentiment for {company_name} ({stock_symbol}) was {'predominantly positive' if positive_pct > negative_pct + 10 else 'predominantly negative' if negative_pct > positive_pct + 10 else 'relatively balanced'}. "
                }
                </p>

                <p>
                {
                    f"The predictive power of sentiment for stock returns (R-squared: {correlation_results['regression']['r_squared']:.3f}) "
                    f"suggests that {'sentiment has some predictive value for future stock movements' if correlation_results['regression']['r_squared'] > 0.1 else 'sentiment alone has limited predictive value for future stock movements'}. "
                    f"{'This analysis indicates that monitoring social media and news sentiment could potentially provide valuable insights for investment decisions.' if correlation_results['regression']['r_squared'] > 0.1 else 'While sentiment alone may not be sufficient for investment decisions, it could be a useful component of a broader analysis strategy.'}"
                    if correlation_results and 'regression' in correlation_results else
                    "Without sufficient correlation data, it's difficult to determine the predictive power of sentiment for stock returns. "
                    "Further analysis with more data might yield more conclusive results."
                }
                </p>
            </div>

            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Stock Sentiment Analysis Tool</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html