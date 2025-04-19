import pandas as pd
from data_preprocessing.feature_utils import (
    load_news_collection,
    load_sentiment_collection,
    clean_text,
    transform_sentiment_score,
    load_model_and_tokenizer,
    get_embeddings,
    fit_and_save_pca,
    load_pca,
    transform_with_pca
)

def preprocess(df, is_train=True):

    if is_train:

        news_df = load_news_collection()
        sentiment_df = load_sentiment_collection()
        df = pd.concat([news_df, sentiment_df], axis=1)
        

    df = pd.concat([news_df, sentiment_df], axis=1)

    df = df[df["title"] != "no title"]
    df["title"] = df["title"].astype(str).apply(clean_text)
    df = df.reset_index(drop=True)

    if is_train:
        df = transform_sentiment_score(df)

    model, tokenizer = load_model_and_tokenizer()
    embeddings = get_embeddings(df["title"].tolist(), model, tokenizer)

    if is_train:
        pca_df = fit_and_save_pca(embeddings, n_components=100)
        df = df.drop(columns=[
        "article_id", "body", "thinking", "date", "title", "source",
        "article_type", "sectors", "tickers", "country"
    ], errors="ignore")
    else:
        pca = load_pca()
        pca_df = transform_with_pca(embeddings, pca)

    
    df = pd.concat([df, pca_df], axis=1)



    return df
