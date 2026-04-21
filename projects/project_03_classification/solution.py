"""Solution comparing classification models."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logreg": Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=500))]),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "svm": Pipeline([("scaler", StandardScaler()), ("m", SVC())]),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name, "F1:", f1_score(y_test, preds))
