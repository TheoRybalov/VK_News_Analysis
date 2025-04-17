import pickle
import os
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from data_preprocessing.prepare_features import preprocess


def train_and_save_model():
    # 1. Получаем подготовленные данные
    df = preprocess()

    print(df.head())
    print(df.shape)

    # 2. Разделение фичей и целевой переменной
    X = df.drop(columns=["sentiment_score"])
    y = df["sentiment_score"]

    # 3. Трейн/тест сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Обучение модели
    model  = SVC(kernel='rbf', C=1.0, probability=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(X_test)

    print("F1 (per class):", f1_score(y_test, y_pred, average=None))
    print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))
    print("F1 (micro):", f1_score(y_test, y_pred, average='micro'))
    print("F1 (weighted):", f1_score(y_test, y_pred, average='weighted'))
    # 6. Сохранение модели через pickle
    os.makedirs("models", exist_ok=True)

# Сохраняем модель
    with open("models/svc_sentiment.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_and_save_model()
