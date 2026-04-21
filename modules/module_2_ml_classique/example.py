"""Module 2 example: classic classification with hyperparameter tuning."""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC()),
])

params = {
    "model__C": [0.1, 1.0, 10.0],
    "model__kernel": ["linear", "rbf"],
}

search = GridSearchCV(pipe, params, cv=5, scoring="f1_weighted")
search.fit(X_train, y_train)
preds = search.predict(X_test)

print("Best params:", search.best_params_)
print(classification_report(y_test, preds))
