"""Starter template for Iris exploration project."""
from sklearn.datasets import load_iris

# TODO: load data, run EDA, train a baseline model and evaluate.
iris = load_iris(as_frame=True)
print(iris.frame.head())
