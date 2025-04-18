# news_feed/rss_parser.py

import feedparser
import pandas as pd
from datetime import datetime


def parse_rss(feed_url: str, max_items: int = 20) -> pd.DataFrame:
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:max_items]

    news_data = []
    for entry in entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")

        # Преобразуем дату, если возможно
        try:
            published = datetime(*entry.published_parsed[:6])
        except:
            published = None

        news_data.append({
            "title": title,
            "link": link,
            "published": published
        })

    return pd.DataFrame(news_data)

finam_url = "https://www.finam.ru/analysis/conews/rsspoint"
investing_url = "https://ru.investing.com/rss/news.rss"
vedomosti_markets_url = "https://www.vedomosti.ru/rss/rubric/finance/markets.xml"
vedomosti_world_economy_url = "https://www.vedomosti.ru/rss/rubric/economics/global.xml"


def get_all_news(items = 10):

    finam_df = parse_rss(finam_url, max_items=10)


    investing_df = parse_rss(investing_url, max_items=10)


    vedomosti_markets_df = parse_rss(vedomosti_markets_url, max_items=10)


    vedomosti_world_df = parse_rss(vedomosti_world_economy_url, max_items=10)



    all_news = pd.concat([finam_df, investing_df, vedomosti_markets_df, vedomosti_world_df], ignore_index=True)
    all_news['published'] = pd.to_datetime(all_news['published'])
    all_news = all_news.sort_values(by="published", ascending=False)
    return all_news

