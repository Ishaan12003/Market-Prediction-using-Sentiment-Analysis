import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import streamlit as st

# Set up Faker for generating synthetic data
fake = Faker()

def generate_synthetic_reddit_data(stock_symbol, company_name=None,
                                 start_date=None, end_date=None, count=150):
    """
    Generate synthetic Reddit data that looks realistic.

    Args:
        stock_symbol (str): Stock symbol for synthetic data
        company_name (str): Company name for context
        start_date (datetime): Start date for synthetic data
        end_date (datetime): End date for synthetic data
        count (int): Number of synthetic posts to generate

    Returns:
        pd.DataFrame: DataFrame containing synthetic Reddit posts
    """
    if not company_name:
        company_name = stock_symbol

    # Ensure dates are datetime objects and handle defaults
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # List of realistic financial subreddits
    subreddits = [
        'wallstreetbets', 'stocks', 'investing', 'cryptocurrency', 'stockmarket',
        'finance', 'personalfinance', 'options', 'dividends', 'economy',
        'securityanalysis', 'pennystocks', 'algotrading', 'financialindependence',
        'stockpicks', 'daytrading', 'stockaday', 'robinhood'
    ]

    # Templates for Reddit posts with placeholders for company/stock
    post_templates = [
        "What do you think about {company}? Worth buying at current price?",
        "Is {ticker} a good buy right now?",
        "{ticker} earnings are coming up. What are your expectations?",
        "Just bought more {ticker}. Here's why I'm bullish long term.",
        "Should I sell my {ticker} shares after today's drop?",
        "{company} looks ready for a breakout. Technical analysis inside.",
        "What's going on with {ticker}? Why is it dropping today?",
        "{company} just announced new products. Thoughts on how this affects stock price?",
        "Why is no one talking about {ticker}? It's up 5% today!",
        "DD on {company}: Why I think it's undervalued right now",
        "Lost money on {ticker} calls. Where did I go wrong?",
        "{ticker} put options strategy discussion",
        "Breaking: {company} just announced a stock split!",
        "Is {company} overvalued at current levels?",
        "{ticker} vs competitors - which stock would you buy now?",
        "Loaded up on {ticker} today. Am I making a mistake?",
        "My price target for {ticker} is {price}. Here's why.",
        "Anyone else concerned about {company}'s debt levels?",
        "{company} insider buying/selling - what does it mean?",
        "{ticker} short interest is at {percent}%. Potential squeeze?"
    ]

    # Templates for comments on posts
    comment_templates = [
        "I agree. {ticker} is definitely {sentiment} right now.",
        "Not sure about that. {company} has some issues with {issue}.",
        "I bought {ticker} at {price} and I'm {sentiment} about it.",
        "The CEO of {company} is making smart moves lately.",
        "Have you looked at {company}'s P/E ratio? It's {sentiment}.",
        "I think {ticker} will hit {price} by end of year.",
        "{company} is facing competition from {competitor}.",
        "The market is overreacting to {company}'s news.",
        "I've been holding {ticker} for years. Great long-term investment.",
        "You should look at {ticker}'s debt to equity ratio before investing.",
        "Their new product line could boost {ticker} significantly.",
        "{company}'s revenue growth is {sentiment} compared to sector.",
        "The macro environment isn't favorable for companies like {company} right now.",
        "Don't forget about the upcoming {ticker} dividend.",
        "I'd wait for a pullback before buying more {ticker}."
    ]

    # Bullish and bearish phrases for sentiment variation
    bullish_phrases = [
        "to the moon", "bullish", "undervalued", "strong buy", "growth potential",
        "buying opportunity", "long-term hold", "diamond hands", "price target up",
        "outperform", "beat expectations", "record profits", "market leader"
    ]

    bearish_phrases = [
        "overvalued", "sell now", "bearish", "declining revenues", "weak guidance",
        "missed estimates", "losing market share", "debt concerns", "avoid",
        "poor management", "competitive pressures", "margin compression"
    ]

    # Generate random dates within the range
    date_range = (end_date - start_date).days

    data = []

    # Track the generated submission IDs to link comments
    submission_ids = []
    submission_titles = []

    # Generate submissions first
    for i in range(count // 3):  # 1/3 will be submissions, 2/3 comments
        # Random date within range
        random_days = random.randint(0, date_range)
        post_date = start_date + timedelta(days=random_days)

        # Random time
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        post_date = post_date.replace(hour=hours, minute=minutes)

        # Choose a subreddit
        subreddit = random.choice(subreddits)

        # Choose post template and fill placeholders
        template = random.choice(post_templates)

        # Random price points for templates
        base_price = random.randint(10, 1000)
        price_points = [
            f"${base_price}",
            f"${base_price}.{random.randint(10, 99)}",
            f"${round(base_price * 0.9, 2)}",
            f"${round(base_price * 1.1, 2)}"
        ]
        price = random.choice(price_points)

        # Random percentages
        percent = f"{random.randint(5, 40)}%"

        # Sentiment leaning (slightly weighted toward being bullish)
        sentiment_leaning = random.choices(
            ["bullish", "bearish", "neutral"],
            weights=[0.4, 0.3, 0.3],
            k=1
        )[0]

        # Sentiment phrases based on leaning
        if sentiment_leaning == "bullish":
            sentiment_phrases = bullish_phrases
        elif sentiment_leaning == "bearish":
            sentiment_phrases = bearish_phrases
        else:
            sentiment_phrases = bullish_phrases + bearish_phrases

        sentiment_phrase = random.choice(sentiment_phrases)

        # Fill template
        title = template.format(
            company=company_name,
            ticker=stock_symbol,
            price=price,
            percent=percent,
            sentiment=sentiment_phrase
        )

        # Generate content with higher likelihood of containing sentiment phrases
        content_parts = []
        paragraphs = random.randint(1, 5)

        for _ in range(paragraphs):
            if random.random() < 0.7:  # 70% chance to include company name
                content_parts.append(fake.paragraph() + f" {company_name} " + fake.paragraph())
            else:
                content_parts.append(fake.paragraph())

        # Add sentiment phrases to content
        sentiment_count = random.randint(1, 3)
        for _ in range(sentiment_count):
            sent_phrase = random.choice(sentiment_phrases)
            content_parts.append(f"I think {stock_symbol} is {sent_phrase}.")

        random.shuffle(content_parts)
        content = " ".join(content_parts)

        # Random user
        username = fake.user_name()

        # Random score (upvotes) - posts have higher variance
        score = random.choices(
            [
                random.randint(1, 10),      # Low score
                random.randint(10, 100),    # Medium score
                random.randint(100, 2000)   # High score
            ],
            weights=[0.6, 0.3, 0.1],        # Most posts get low scores
            k=1
        )[0]

        # Generate a random submission ID
        submission_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        submission_ids.append(submission_id)
        submission_titles.append(title)

        # Create submission data
        submission_data = {
            'source': f"reddit:r/{subreddit}",
            'id': submission_id,
            'username': username,
            'title': title,
            'content': content,
            'full_text': f"{title} {content}",
            'score': score,
            'timestamp': post_date,
            'type': 'submission',
            'url': f"https://www.reddit.com/r/{subreddit}/comments/{submission_id}/",
            'num_comments': random.randint(0, 50)
        }

        data.append(submission_data)

    # Generate comments for the submissions
    for i in range(count - len(data)):
        # Pick a random submission to comment on
        if not submission_ids:
            break

        submission_idx = random.randint(0, len(submission_ids) - 1)
        submission_id = submission_ids[submission_idx]
        submission_title = submission_titles[submission_idx]

        # Random date slightly after the submission
        submission_data = next((item for item in data if item['id'] == submission_id), None)
        if submission_data:
            # Comment is 0-24 hours after the submission
            hours_after = random.randint(0, 24)
            comment_date = submission_data['timestamp'] + timedelta(hours=hours_after)

            # Make sure it's still within the end date
            if comment_date > end_date:
                comment_date = end_date

            # Random subreddit (same as the submission)
            subreddit = submission_data['source'].split(':r/')[1]

            # Choose comment template and fill placeholders
            template = random.choice(comment_templates)

            # Random price
            base_price = random.randint(10, 1000)
            price = f"${base_price}"

            # Sentiment for comments
            sentiment_options = ["bullish", "bearish", "neutral", "concerning", "promising"]
            sentiment = random.choice(sentiment_options)

            # Random issues and events
            issues = ["debt", "management", "competition", "regulations", "supply chain"]
            issue = random.choice(issues)

            # Random competitors
            competitors = ["competitors", "other players in the space", "the competition"]
            competitor = random.choice(competitors)

            # Fill template
            content = template.format(
                company=company_name,
                ticker=stock_symbol,
                price=price,
                sentiment=sentiment,
                issue=issue,
                competitor=competitor
            )

            # Random username (different from submission)
            username = fake.user_name()
            while username == submission_data['username']:
                username = fake.user_name()

            # Random score (upvotes) - comments typically have lower scores
            score = random.choices(
                [
                    random.randint(1, 5),      # Very low score
                    random.randint(5, 25),     # Low score
                    random.randint(25, 100)    # Medium score
                ],
                weights=[0.7, 0.25, 0.05],     # Most comments get very low scores
                k=1
            )[0]

            # Generate a random comment ID
            comment_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=7))

            # Create comment data
            comment_data = {
                'source': f"reddit:r/{subreddit}",
                'id': comment_id,
                'username': username,
                'title': f"Re: {submission_title}",
                'content': content,
                'full_text': content,
                'score': score,
                'timestamp': comment_date,
                'type': 'comment',
                'url': f"https://www.reddit.com/r/{subreddit}/comments/{submission_id}/comment/{comment_id}/",
                'num_comments': 0
            }

            data.append(comment_data)

    # Create DataFrame
    df = pd.DataFrame(data)
    st.success(f"Generated {len(df)} synthetic Reddit posts and comments.")
    return df

def generate_synthetic_news_data(stock_symbol, company_name=None,
                               start_date=None, end_date=None, count=100):
    """
    Generate synthetic news data that looks realistic.

    Args:
        stock_symbol (str): Stock symbol for synthetic data
        company_name (str): Company name for context
        start_date (datetime): Start date for synthetic data
        end_date (datetime): End date for synthetic data
        count (int): Number of synthetic news articles to generate

    Returns:
        pd.DataFrame: DataFrame containing synthetic news articles
    """
    if not company_name:
        company_name = stock_symbol

    # Ensure dates are datetime objects and handle defaults
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # News sources
    news_sources = [
        'Bloomberg', 'CNBC', 'Reuters', 'Financial Times', 'Wall Street Journal',
        'Business Insider', 'Yahoo Finance', 'MarketWatch', 'Seeking Alpha', 'The Motley Fool',
        'Forbes', 'Barron\'s', 'TheStreet', 'Investing.com', 'Benzinga'
    ]

    # News headline templates
    headline_templates = [
        "{company} Reports Q{quarter} Earnings {result} Expectations",
        "{company} Stock {movement} After {reason}",
        "Analysts {action} {company} Stock to {price_target}",
        "{company} Announces New {product} to Boost Revenue",
        "{company} CEO Discusses Future Growth Plans",
        "Is {company} Stock a Buy Right Now?",
        "{company} Shares {movement} as Market Reacts to {event}",
        "{company} to {action} Dividend by {percent}",
        "{bank} Initiates Coverage of {company} with {rating} Rating",
        "{company} Reports {percent} {direction} in {metric}",
        "{company} Faces {challenge} but Remains {outlook}",
        "{company} Partners with {partner} to Expand Market Share",
        "{company} Announces Plans to {action} {number} Jobs",
        "Investors {reaction} to {company}'s Latest Quarterly Results",
        "{company} Reveals New Strategy to Combat {challenge}"
    ]

    # News description templates
    description_templates = [
        "{company} reported quarterly earnings of ${eps} per share, {compared} analyst estimates of ${est_eps}.",
        "Shares of {ticker} {movement} {percent}% after the company announced {announcement}.",
        "Analysts at {bank} have {action} their rating on {company} stock to {rating}, citing {reason}.",
        "The tech giant {company} unveiled a new {product} that aims to {goal} in the coming {timeframe}.",
        "{company} CEO {ceo_name} addressed investors during the earnings call, highlighting {highlight}.",
        "Investors are weighing whether {ticker} presents a buying opportunity following recent {event}.",
        "{company}'s stock price has {movement} {percent}% over the past {timeframe}, {compared} the S&P 500.",
        "The board of {company} has approved a {percent}% {action} in its quarterly dividend to ${dividend} per share.",
        "{bank} analysts have initiated coverage of {company} with a {rating} rating and a price target of ${price_target}.",
        "{company} reported a {percent}% {direction} in {metric}, which {impact} investor sentiment on the stock."
    ]

    # Random values for templates
    quarters = [1, 2, 3, 4]
    results = ["Beats", "Meets", "Misses", "Exceeds", "Falls Short of"]
    movements = ["Rises", "Falls", "Jumps", "Plunges", "Soars", "Dips", "Climbs", "Drops"]
    reasons = [
        "Earnings Report", "Analyst Upgrade", "Analyst Downgrade", "Product Announcement",
        "Regulatory News", "Market Selloff", "Sector Rotation", "Economic Data"
    ]
    actions = ["Upgrades", "Downgrades", "Raises", "Lowers", "Maintains", "Increases", "Decreases"]
    price_targets = [f"${random.randint(10, 500)}" for _ in range(5)]
    products = ["Product Line", "Service", "Platform", "Technology", "Solution"]
    events = ["Earnings", "Investor Day", "Product Launch", "Industry Conference"]
    percent_changes = [f"{random.randint(1, 25)}", f"{random.randint(1, 25)}.{random.randint(1, 9)}"]
    directions = ["Increase", "Decrease", "Rise", "Decline", "Growth", "Contraction"]
    metrics = ["Revenue", "Profit", "Sales", "User Growth", "Margin", "Market Share"]
    banks = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Bank of America", "Citigroup"]
    ratings = ["Buy", "Sell", "Hold", "Overweight", "Underweight", "Neutral", "Outperform"]
    challenges = ["Supply Chain Issues", "Inflation", "Competition", "Regulatory Scrutiny"]
    outlooks = ["Positive", "Negative", "Cautious", "Optimistic", "Conservative"]
    partners = ["Microsoft", "Amazon", "Google", "Apple", "Meta", "Samsung"]
    timeframes = ["Quarter", "Year", "6 Months", "12 Months", "Next Fiscal Year"]
    reactions = ["Positive", "Negative", "Mixed", "Enthusiastic", "Cautious"]

    # Generate random dates within the range
    date_range = (end_date - start_date).days

    data = []

    for i in range(count):
        # Random date within range
        random_days = random.randint(0, date_range)
        article_date = start_date + timedelta(days=random_days)

        # Random time
        hours = random.randint(0, 23)
        minutes = random.randint(0, 59)
        article_date = article_date.replace(hour=hours, minute=minutes)

        # Choose source
        source = random.choice(news_sources)

        # Choose headline template and fill placeholders
        headline_template = random.choice(headline_templates)
        description_template = random.choice(description_templates)

        # Random values for placeholders
        quarter = random.choice(quarters)
        result = random.choice(results)
        movement = random.choice(movements)
        reason = random.choice(reasons)
        action = random.choice(actions)
        price_target = random.choice(price_targets)
        product = random.choice(products)
        event = random.choice(events)
        percent = random.choice(percent_changes)
        direction = random.choice(directions)
        metric = random.choice(metrics)
        bank = random.choice(banks)
        rating = random.choice(ratings)
        challenge = random.choice(challenges)
        outlook = random.choice(outlooks)
        partner = random.choice(partners)
        timeframe = random.choice(timeframes)
        reaction = random.choice(reactions)
        eps = f"{random.randint(0, 10)}.{random.randint(10, 99)}"
        est_eps = f"{random.randint(0, 10)}.{random.randint(10, 99)}"
        compared = random.choice(["beating", "missing", "meeting", "exceeding"])
        announcement = f"new {random.choice(products)}"
        ceo_name = fake.name()
        highlight = random.choice(["growth opportunities", "cost-cutting measures", "strategic partnerships"])
        dividend = f"{random.randint(0, 2)}.{random.randint(10, 99)}"
        number = f"{random.randint(100, 5000)}"
        goal = random.choice(["increase efficiency", "reduce costs", "expand market share"])
        impact = random.choice(["positively affected", "negatively affected", "significantly influenced"])

        # Fill templates
        headline = headline_template.format(
            company=company_name,
            ticker=stock_symbol,
            quarter=quarter,
            result=result,
            movement=movement,
            reason=reason,
            action=action,
            price_target=price_target,
            product=product,
            event=event,
            percent=percent,
            direction=direction,
            metric=metric,
            bank=bank,
            rating=rating,
            challenge=challenge,
            outlook=outlook,
            partner=partner,
            timeframe=timeframe,
            reaction=reaction,
            number=number
        )

        description = description_template.format(
            company=company_name,
            ticker=stock_symbol,
            quarter=quarter,
            result=result,
            movement=movement,
            reason=reason,
            action=action,
            price_target=price_target,
            product=product,
            event=event,
            percent=percent,
            direction=direction,
            metric=metric,
            bank=bank,
            rating=rating,
            challenge=challenge,
            outlook=outlook,
            partner=partner,
            timeframe=timeframe,
            reaction=reaction,
            eps=eps,
            est_eps=est_eps,
            compared=compared,
            announcement=announcement,
            ceo_name=ceo_name,
            highlight=highlight,
            dividend=dividend,
            goal=goal,
            impact=impact
        )

        # Generate a random article ID
        article_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))

        # Random author
        author = fake.name()

        # Create article data
        article_data = {
            'source': f"news:{source}",
            'id': article_id,
            'username': author,
            'title': headline,
            'content': description,
            'full_text': f"{headline} {description}",
            'score': 0,  # No equivalent in news
            'timestamp': article_date,
            'type': 'news',
            'url': f"https://{source.lower().replace(' ', '')}.com/articles/{article_id}",
            'num_comments': 0  # No equivalent in news
        }

        data.append(article_data)

    # Create DataFrame
    df = pd.DataFrame(data)
    st.success(f"Generated {len(df)} synthetic news articles.")
    return df