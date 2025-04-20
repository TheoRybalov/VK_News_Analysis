
import pickle
from news_feed.rss_parser import get_all_news
from data_preprocessing.prepare_features import preprocess
from data_preprocessing.feature_utils import (
    load_news_collection,
    load_sentiment_collection,
    clean_text,
    transform_sentiment_score,
    load_model_and_tokenizer,
    get_embeddings,
    apply_pca,
    transform_with_pca,
    load_pca,

    get_path_in_project
)

import pandas as pd

def load_model(path="models/svc_model.pkl"):
    full_path = get_path_in_project(path)
    with open(full_path, "rb") as f:
        return pickle.load(f)

def analyze_news_sentiment():

    current_news = get_all_news()
    print(current_news.head())
    print(current_news.shape)

    current_news = current_news.reset_index(drop=True)



    model, tokenizer = load_model_and_tokenizer()
    embeddings = get_embeddings(current_news["title"].tolist(), model, tokenizer)

    print(embeddings.shape)

    pca_model = load_pca("models/pca_model.pkl")
    pca_df = transform_with_pca(embeddings, pca_model)
    current_news = pd.concat([current_news, pca_df], axis=1)

    print(current_news.head())
    print(current_news.shape)
    

    clf = load_model("models/svc_model.pkl")  # Используем правильный путь и метод

    # 5. Применение только PCA-признаков
    X = pca_df  # если модель обучалась только на PCA

    preds = clf.predict(X)
    print("Предсказания:")
    print(preds)

    label_map = {
    0: "негативная",
    1: "нейтральная",
    2: "позитивная"
    }
    preds_text = [label_map[p] for p in preds]

    current_news["sentiment_prediction"] = preds_text
    print(current_news[["title", "sentiment_prediction"]].head(20))





if __name__ == "__main__":
    analyze_news_sentiment()
    