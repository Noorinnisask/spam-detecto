# 📩 SMS Spam Detector

A machine learning web app that classifies SMS messages as spam or not spam.

## Problem
Spam messages waste time and can be harmful. This project builds a classifier
and a simple web app to detect spam, with attention to data quality and error analysis.

## Dataset
SMS Spam Collection (UCI / Kaggle): 5,572 messages, 403 duplicates removed
(5,169 unique). Only ~13% are spam, so the data is imbalanced.

## Approach
1. **EDA:** class balance, message length, common words, word clouds.
2. **Cleaning:** lowercase, links and numbers replaced with tokens, symbols removed.
3. **Features:** TF-IDF with unigrams and bigrams.
4. **Models compared:** Logistic Regression, Naive Bayes, Linear SVM, Random Forest
   (same stratified 80/20 split).
5. **Evaluation:** precision, recall and F1 for the spam class, not just accuracy.
6. **Error analysis:** reviewed false positives and false negatives.

## Results
| Model | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|
| Logistic Regression | 0.84 | 0.95 | 0.89 |

The model achieved 0.97 overall accuracy on the stratified test split.
The spam-class F1 score is included because accuracy alone can hide errors
on the minority class.

## Key findings
- Spam messages are longer (~139 vs ~71 characters).
- Spam uses words like "free", "claim", "prize"; ham is casual conversation.
- Accuracy alone is misleading because of the imbalance.
- Spam recall (0.95) is higher than precision (0.84), so some legitimate
  messages may also be flagged.

## Data quality notes
Duplicates were removed before splitting to avoid leakage between train and test sets.
I have hands-on experience with annotation and label quality checking from my
internship, which shaped how I reviewed the data and the model's mistakes.

## Run it locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Limitations and next steps
- Small, English-only dataset from one source; real-world spam changes over time.
- Could add message length as a feature, tune the threshold, or try a transformer model.

## Tech
Python, pandas, scikit-learn, Streamlit

## Author
Noorinnisa Shaik

*Dataset: SMS Spam Collection, UCI Machine Learning Repository.*
   **Live demo:** <https://noorinnisa-spam-detector.streamlit.app/>