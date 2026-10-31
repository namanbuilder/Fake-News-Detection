import joblib
from preprocess import clean_text

_tfidf = joblib.load("models/tfidf.joblib")
_model = joblib.load("models/best_model.joblib")


def predict(text: str):
    """Returns (label, confidence in 0-1 or None)."""
    X = _tfidf.transform([clean_text(text)])
    label = int(_model.predict(X)[0])
    conf = None
    if hasattr(_model, "predict_proba"):
        conf = float(_model.predict_proba(X)[0][label])
    elif hasattr(_model, "decision_function"):
        import math
        conf = 1 / (1 + math.exp(-abs(float(_model.decision_function(X)[0]))))
    return ("Real" if label == 1 else "Fake"), conf


if __name__ == "__main__":
    print(predict(input("Enter news text: ")))
