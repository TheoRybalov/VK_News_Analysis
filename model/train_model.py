
import pandas as pd
from feature_utils import *

import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import re

from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def get_path_in_project(path: str) -> str:

    current_dir = os.path.dirname(__file__)
    needed_path = os.path.join(current_dir, path)
    return needed_path


def load_news_collection():
    news_data_path = get_path_in_project("../data/news-collection.csv")
    news_data = pd.read_csv(news_data_path, sep=',').head(15000)
    news_data = news_data.rename(columns={'Unnamed: 0': 'article_id'})
    return news_data


def load_sentiment_collection():
    sentinent_data_path = get_path_in_project("../data/news_sentiment.json")
    sentiment_df = pd.read_json(sentinent_data_path, orient='index')
    sentiment_df = sentiment_df.reset_index().rename(columns={'index': 'article_id'})
    return sentiment_df




def model_for_embeddings():
    tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")
    model = AutoModel.from_pretrained("deepvk/USER-bge-m3")
    return model, tokenizer


def get_embeddings(data, model, tokenizer):

    titles = data['title'].dropna().astype(str).tolist()

    batch_size = 64
    all_embeddings = []


    for i in tqdm(range(0, len(titles), batch_size)):
        batch_texts = titles[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embedding = model_output[0][:, 0]
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

        all_embeddings.append(sentence_embedding)

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

    return all_embeddings_tensor


def pca_from_embeddings(embeddings, num_components=100):
    pca = PCA(n_components=num_components, random_state=42)
    pca_100d = pca.fit_transform(embeddings)
    pca_columns = [f"PCA_{i+1}" for i in range(num_components)]
    pca_df = pd.DataFrame(pca_100d, columns=pca_columns)
    pca_df = pca_df.reset_index(drop=True)
    return pca_df



def transform_sentiment_score(data):
    def classify_sentiment(score):
        if score < -0.1:
            return 0
        elif score > 0.1:
            return 2
        else:
            return 1

    data["sentiment_score"] = data["sentiment_score"].apply(classify_sentiment)
    return data

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)                   # убрать лишние пробелы и переносы
    text = re.sub(r'<.*?>', '', text)                  # удалить html
    text = re.sub(r'http\S+|www\.\S+', '', text)       # удалить ссылки
    text = re.sub(r'[^\w\s.,!?]', '', text)            # оставить только буквы/цифры и базовую пунктуацию
    text = re.sub(r"[\"']", " ", text)
    return text.strip()

def main():

    news_data = load_news_collection()
    sentiment_data = load_sentiment_collection()

    print(news_data.head())
    print(sentiment_data.head())

    sentiment_data = transform_sentiment_score(sentiment_data)

    total_df = pd.concat([news_data, sentiment_data], axis=1)

    total_df = total_df[total_df["title"] != "no title"]

    total_df["title"] = total_df["title"].apply(clean_text)
    total_df = total_df.reset_index(drop=True)

    # ТЕСТИРОВАНИЕ
    total_df = total_df.head(10)
    # ТЕСТИРОВАНИЕ

    print(total_df.head())

    embedding_model, embedding_tokenizer = model_for_embeddings()
    all_embeddings = get_embeddings(total_df, embedding_model, embedding_tokenizer)

    pca = pca_from_embeddings(all_embeddings, num_components=2)

    total_df = pd.concat([total_df, pca], axis=1)



    total_df = total_df.drop(["article_id", "body", "thinking", "date", "title", "source", "article_type", "sectors", "tickers", "country"], axis = 1)

    print(total_df.head())

    X = total_df.drop(["sentiment_score"], axis = 1)
    y = total_df["sentiment_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

    model  = SVC(kernel='rbf', C=1.0, probability=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(X_test)

    print("F1 (per class):", f1_score(y_test, y_pred, average=None))
    print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))
    print("F1 (micro):", f1_score(y_test, y_pred, average='micro'))
    print("F1 (weighted):", f1_score(y_test, y_pred, average='weighted'))











if __name__ == "__main__":
    main()