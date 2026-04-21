"""Starter template for advanced text classification."""
from sklearn.feature_extraction.text import TfidfVectorizer

sample = ["ai is useful", "sports news", "political debate"]
vec = TfidfVectorizer()
print(vec.fit_transform(sample).shape)
