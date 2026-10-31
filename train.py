"""
Train and compare classical NLP models for fake news detection.

Dataset: "Fake and Real News Dataset" (Kaggle, ISOT) -> put True.csv and Fake.csv in ./data
Columns used: title, text.   Run:  python train.py
Falls back to data/sample.csv (tiny demo set) if the Kaggle files are missing.
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from preprocess import clean_text

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_data():
    if os.path.exists("data/True.csv") and os.path.exists("data/Fake.csv"):
        real = pd.read_csv("data/True.csv"); real["label"] = 1
        fake = pd.read_csv("data/Fake.csv"); fake["label"] = 0
        df = pd.concat([real, fake], ignore_index=True)
    else:
        print("Kaggle files not found -> using data/sample.csv (demo only)")
        df = pd.read_csv("data/sample.csv")
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    df = load_data()
    print("Dataset size:", df.shape, "\n", df["label"].value_counts())
    df["clean"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1 if len(df) < 500 else 3)
    Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }

    rows, best_name, best_f1, best_model = [], None, -1, None
    for name, m in models.items():
        m.fit(Xtr, y_train)
        pred = m.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        acc = accuracy_score(y_test, pred)
        rows.append({"Model": name, "Accuracy": acc, "Precision": p, "Recall": r, "F1": f})
        print(f"\n=== {name} ===\n", classification_report(y_test, pred, target_names=["Fake", "Real"], zero_division=0))
        if f > best_f1:
            best_name, best_f1, best_model, best_pred = name, f, m, pred

    res = pd.DataFrame(rows).round(4)
    res.to_csv("results/model_comparison.csv", index=False)
    print("\n", res.to_string(index=False))
    print(f"\nBest model: {best_name}")

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

    res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", figsize=(8, 4.5), ylim=(0, 1.05))
    plt.title("Model Comparison"); plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150); plt.close()

    joblib.dump(tfidf, "models/tfidf.joblib")
    joblib.dump(best_model, "models/best_model.joblib")
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)
    print("Saved model + vectorizer to ./models")


if __name__ == "__main__":
    main()
