import re

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    STOP = set(stopwords.words("english"))
    LEMMA = WordNetLemmatizer()
    LEMMA.lemmatize("tests")
except Exception:  # nltk or its data unavailable (offline) -> simple fallback
    STOP = set("a an the and or but if of to in on for with is are was were be been it this that as at by from".split())
    LEMMA = None


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/numbers, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP and len(t) > 2]
    if LEMMA:
        tokens = [LEMMA.lemmatize(t) for t in tokens]
    return " ".join(tokens)
