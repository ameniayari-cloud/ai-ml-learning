"""Baseline solution for advanced text classification."""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

X = [
    "machine learning improves healthcare",
    "new football championship starts today",
    "stock market closes higher",
    "latest deep learning model released",
]
y = ["tech", "sports", "finance", "tech"]

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("lr", LogisticRegression(max_iter=1000)),
])
clf.fit(X, y)
print(clf.predict(["new ai model for image classification"]))
