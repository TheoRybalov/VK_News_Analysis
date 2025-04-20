
import pandas as pd
from news_feed.rss_parser import get_all_news
from data_preprocessing.feature_utils import (
    load_model_and_tokenizer,
    get_embeddings,
    transform_with_pca,
    load_pca,
    get_path_in_project
)

from model.model_utils import load_model


def analyze_news_sentiment():

    current_news = get_all_news()


    current_news = current_news.reset_index(drop=True)



    model, tokenizer = load_model_and_tokenizer()
    embeddings = get_embeddings(current_news["title"].tolist(), model, tokenizer)

    pca_model = load_pca("models/pca_model.pkl")
    pca_df = transform_with_pca(embeddings, pca_model)
    current_news = pd.concat([current_news, pca_df], axis=1)

    

    clf = load_model("models/svc_model.pkl")
    X = pca_df 

    preds = clf.predict(X)
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
    