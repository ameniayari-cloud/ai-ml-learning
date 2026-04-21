"""Module 4 example: TF-IDF + Logistic Regression for sentiment baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

texts = [
    "I love this course",
    "Great introduction to machine learning",
    "This is hard and confusing",
    "I am happy with my progress",
]
labels = [1, 1, 0, 1]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])

pipe.fit(texts, labels)
print(pipe.predict(["This module is great"]))
