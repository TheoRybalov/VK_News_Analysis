import pandas as pd
from data_preprocessing.feature_utils import (
    load_news_collection,
    load_sentiment_collection,
    clean_text,
    transform_sentiment_score,
    load_model_and_tokenizer,
    get_embeddings,
    apply_pca
)

def preprocess():
    news_df = load_news_collection()
    sentiment_df = load_sentiment_collection()

    df = pd.concat([news_df, sentiment_df], axis=1)
    df = df[df["title"] != "no title"]
    df["title"] = df["title"].astype(str).apply(clean_text)
    df = df.reset_index(drop=True)
    df = transform_sentiment_score(df)

    df = df.head(10)

    model, tokenizer = load_model_and_tokenizer()
    embeddings = get_embeddings(df["title"].tolist(), model, tokenizer)

    pca_df = apply_pca(embeddings, n_components=2)
    df = pd.concat([df, pca_df], axis=1)

    df = df.drop(columns=[
        "article_id", "body", "thinking", "date", "title", "source",
        "article_type", "sectors", "tickers", "country"
    ], errors="ignore")

    return df
