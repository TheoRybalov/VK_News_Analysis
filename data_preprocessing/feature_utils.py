import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import pickle


def get_path_in_project(path: str) -> str:
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "..", path)

def load_news_collection():
    path = get_path_in_project("data/news-collection.csv")
    news_df = pd.read_csv(path).rename(columns={"Unnamed: 0": "article_id"})
    return news_df.head(15000)

def load_sentiment_collection():
    path = get_path_in_project("data/news_sentiment.json")
    sentiment_df = pd.read_json(path, orient="index").reset_index().rename(columns={"index": "article_id"})
    return sentiment_df

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    text = re.sub(r"[\"']", " ", text)
    return text.strip()

def transform_sentiment_score(df):
    def classify(score):
        if score < -0.1:
            return 0
        elif score > 0.1:
            return 2
        else:
            return 1
    df["sentiment_score"] = df["sentiment_score"].apply(classify)
    return df

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")
    model = AutoModel.from_pretrained("deepvk/USER-bge-m3")
    return model, tokenizer

def get_embeddings(titles, model, tokenizer, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(titles), batch_size)):
        batch = titles[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
            cls_embeddings = output[0][:, 0]
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        embeddings.append(cls_embeddings)
    return torch.cat(embeddings, dim=0).numpy()

def train_pca(embeddings, n_components=100):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pd.DataFrame(reduced, columns=[f"PCA_{i+1}" for i in range(n_components)])

def apply_pca(embeddings, n_components=100):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pd.DataFrame(reduced, columns=[f"PCA_{i+1}" for i in range(n_components)])

def fit_and_save_pca(embeddings, n_components=100, path="models/pca.pkl"):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    with open(path, "wb") as f:
        pickle.dump(pca, f)
    return pd.DataFrame(reduced, columns=[f"PCA_{i+1}" for i in range(n_components)])

def load_pca(path="models/pca.pkl"):
    full_path = get_path_in_project(path)
    with open(full_path, "rb") as f:
        return pickle.load(f)

def transform_with_pca(embeddings, pca):
    reduced = pca.transform(embeddings)
    n_components = reduced.shape[1]
    return pd.DataFrame(reduced, columns=[f"PCA_{i+1}" for i in range(n_components)])