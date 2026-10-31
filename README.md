# Fake News Detection Using NLP
ABESEC | IT 4th Year | HCLTech Project

## Pipeline
1. Data: Kaggle "Fake and Real News Dataset" (ISOT) -> put `True.csv` and `Fake.csv` in `data/`
2. Preprocessing: lowercase, remove URLs/HTML/punctuation, stopwords, lemmatization (`preprocess.py`)
3. Features: TF-IDF with unigrams + bigrams
4. Models compared: Logistic Regression, Naive Bayes, Linear SVM, Random Forest (`train.py`)
5. Evaluation: Accuracy, Precision, Recall, F1, confusion matrix (saved in `results/`)
6. Deployment: Streamlit web app (`app.py`)

## Run
```
pip install -r requirements.txt
python train.py          # trains, compares, saves best model
streamlit run app.py     # web interface
```
Without the Kaggle files, `train.py` uses `data/sample.csv` (tiny demo set, only to check the code runs).

## Note for viva
The ISOT dataset gives ~98-99% accuracy partly because real articles contain "Reuters" style cues.
Mention this limitation and test on outside data. Possible extension: BERT/DistilBERT fine-tuning.
