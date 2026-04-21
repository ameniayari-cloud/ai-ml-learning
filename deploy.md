# Guide de déploiement et productionisation

## Objectif
Déployer un modèle entraîné en API et préparer son passage en production.

## Étapes clés
1. **Sauvegarder le modèle** (`joblib`, `pickle`, `model.save`, `torch.save`).
2. **Créer une API d'inférence** avec Flask ou FastAPI.
3. **Valider les entrées** (schémas, valeurs manquantes, normalisation).
4. **Surveiller** latence, erreurs, dérive des données.
5. **Versionner** modèle + données + code.

## Exemple minimal FastAPI
```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(features: list[float]):
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    return {"prediction": float(pred)}
```

## Bonnes pratiques
- Ajouter des tests d'API et de régression modèle.
- Logger les prédictions anonymisées pour audit.
- Mettre en place CI/CD et rollback simple.
- Limiter la surface d'attaque (validation stricte + dépendances à jour).
