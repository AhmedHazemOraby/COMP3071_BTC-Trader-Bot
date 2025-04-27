# -*- coding: utf-8 -*-
import joblib
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import re
import os
import json
import math
from collections import Counter

analyzer = SentimentIntensityAnalyzer()

reddit = praw.Reddit(
    client_id='3GFOpAyDT_DKaXwBZNqEoQ',
    client_secret='5PTAqA4bCTY8EUzVfOJTpMhtn790Rg',
    user_agent='BTCBot by u/ImprovementNearby393'
)

BITCOIN_SUBREDDITS = [
    'Bitcoin', 'btc', 'BitcoinMarkets', 'CryptoCurrency',
    'CryptoMarkets', 'CryptoCurrencyTrading'
]

KEYWORD_CATEGORIES = {
    'short_term_price': [
        "pump", "crash", "drop", "soar", "dump", "skyrocket", "moon", "plummet",
        "bounce", "spike", "bullish", "bearish", "breakout", "support", "resistance"
    ],
    'macro_events': [
        "inflation", "interest rate", "fed", "etf", "regulation", "halving",
        "ban", "approval", "recession", "cpi", "fomc", "election", "qe", "qt"
    ],
    'crypto_news': [
        "satoshi", "wallet", "exchange", "miners", "stablecoin", "SEC", "funding",
        "adoption", "whale", "institution", "lightning", "coinbase", "binance", "ledger"
    ]
}

SHORT_TERM_HINTS = [
    "today", "tomorrow", "this week", "next few days", "next week", "short-term", "soon"
]

SENTIMENT_HISTORY_FILE = "sentiment_history.json"
SENTIMENT_CACHE_FILE = "latest_sentiment.save"

def classify_post_category(text):
    text = text.lower()
    score_dict = {cat: 0 for cat in KEYWORD_CATEGORIES}

    for category, keywords in KEYWORD_CATEGORIES.items():
        for kw in keywords:
            if re.search(rf'\b{kw}\b', text):
                score_dict[category] += 1

    best_category = max(score_dict, key=score_dict.get)
    if score_dict[best_category] > 0:
        return best_category
    return None

def extract_relevance_score(text):
    text_lower = text.lower()
    score = 0
    keyword_hits = Counter()
    for category, keywords in KEYWORD_CATEGORIES.items():
        for kw in keywords:
            if re.search(rf'\b{kw}\b', text_lower):
                keyword_hits[category] += 1
                if category == 'short_term_price':
                    score += 5
                elif category == 'macro_events':
                    score += 2
                elif category == 'crypto_news':
                    score += 1
    return score, keyword_hits

def mentions_short_term(text):
    text = text.lower()
    return any(hint in text for hint in SHORT_TERM_HINTS)

def load_sentiment_history():
    if os.path.exists(SENTIMENT_HISTORY_FILE):
        with open(SENTIMENT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sentiment_history(score):
    with open(SENTIMENT_HISTORY_FILE, "w") as f:
        json.dump({"last_score": score, "timestamp": datetime.utcnow().isoformat()}, f)

def get_reddit_sentiment(days_back=1, return_posts=False):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    sentiment_scores = []
    keyword_stats = Counter()
    collected_posts = []
    total_posts = 0
    accepted_posts = 0

    for subreddit in BITCOIN_SUBREDDITS:
        try:
            posts = reddit.subreddit(subreddit).new(limit=150)
            for post in posts:
                total_posts += 1
                if not (start_time.timestamp() <= post.created_utc <= end_time.timestamp()):
                    continue
                if post.stickied or post.score < 5:
                    continue

                body = post.selftext or ""
                try:
                    post.comments.replace_more(limit=0)
                    comments = " ".join([c.body for c in post.comments[:10] if hasattr(c, 'score') and c.score > 1])
                except:
                    comments = ""

                combined_text = f"{post.title}. {body} {comments}".strip()

                relevance_score, keyword_hits = extract_relevance_score(combined_text)
                if relevance_score < 2:
                    continue

                if relevance_score < 5 and not mentions_short_term(combined_text):
                    continue

                sentiment = analyzer.polarity_scores(combined_text)['compound']
                if abs(sentiment) < 0.01:
                    continue

                upvote_weight = math.log1p(post.score)
                final_weight = int(max(1, upvote_weight * relevance_score))
                sentiment_scores.extend([sentiment] * final_weight)
                keyword_stats.update(keyword_hits)
                accepted_posts += 1

                if return_posts:
                    collected_posts.append({
                        "title": post.title,
                        "url": f"https://reddit.com{post.permalink}",
                        "score": post.score,
                        "sentiment": sentiment,
                        "subreddit": subreddit
                    })

        except Exception as e:
            print(f"Error fetching r/{subreddit}: {e}")

    print(f"Analyzed {total_posts} posts, Accepted {accepted_posts}")

    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        print(f"Reddit Sentiment Score: {avg_score:.4f}")
        print(f"Top Keywords: {keyword_stats.most_common(5)}")
        return (avg_score, collected_posts) if return_posts else avg_score
    else:
        print("No valid sentiment posts found.")
        return (0.0, []) if return_posts else 0.0

def get_daily_sentiment(days=3):
    try:
        reddit_sentiment = get_reddit_sentiment(days)

        if reddit_sentiment > 0.1:
            signal = "BUY"
        elif reddit_sentiment < -0.1:
            signal = "SELL"
        else:
            signal = "HOLD"

        print(f"\nFinal Combined Sentiment Score (Price-Relevant): {reddit_sentiment:.4f}")
        print(f"Change in Sentiment vs Previous: {reddit_sentiment - load_sentiment_history().get('last_score', 0.0):+.4f}")
        print(f"Suggested Action: {signal}")

        save_sentiment_history(reddit_sentiment)
        joblib.dump(reddit_sentiment, SENTIMENT_CACHE_FILE)

        return reddit_sentiment, signal

    except Exception as e:
        print(f"Error in get_daily_sentiment: {e}")
        return 0.0, "ERROR"