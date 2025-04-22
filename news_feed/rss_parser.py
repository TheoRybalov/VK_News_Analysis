# news_feed/rss_parser.py

import feedparser
import pandas as pd
from datetime import datetime, timezone, timedelta

MOSCOW_TZ = timezone(timedelta(hours=3))  # GMT+3

def parse_rss(feed_url: str, source_name: str, max_items: int = 20) -> pd.DataFrame:
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:max_items]

    news_data = []
    for entry in entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")

        try:
            dt_utc = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            published = dt_utc.astimezone(MOSCOW_TZ)
        except:
            published = None

        news_data.append({
            "title": title,
            "link": link,
            "published": published,
            "source": source_name
        })

    return pd.DataFrame(news_data)

finam_url = "https://www.finam.ru/analysis/conews/rsspoint"
investing_url = "https://ru.investing.com/rss/news.rss"
vedomosti_markets_url = "https://www.vedomosti.ru/rss/rubric/finance.xml"
vedomosti_world_economy_url = "https://www.vedomosti.ru/rss/rubric/economics.xml"


def get_all_news(items=20):



    finam_df = parse_rss(finam_url, "Финам", max_items=items)
    investing_df = parse_rss(investing_url, "Investing", max_items=items)
    vedomosti_markets_df = parse_rss(vedomosti_markets_url, "Ведомости: рынки", max_items=items)
    vedomosti_world_df = parse_rss(vedomosti_world_economy_url, "Ведомости: экономика", max_items=items)

    all_news = pd.concat([finam_df, investing_df, vedomosti_markets_df, vedomosti_world_df], ignore_index=True)
    all_news['published'] = pd.to_datetime(all_news['published'])
    all_news = all_news.sort_values(by="published", ascending=False)
    return all_news


