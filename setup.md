# Setup environnement AI/ML

## 1) Pré-requis
- Python 3.10+
- pip
- (Optionnel) GPU + pilotes CUDA pour entraînements avancés

## 2) Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate     # Windows PowerShell
```

## 3) Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Lancer Jupyter
```bash
jupyter notebook
```

## 5) Vérification rapide
```bash
python -c "import numpy,pandas,sklearn,matplotlib,torch,tensorflow; print('OK')"
```

## 6) Organisation recommandée de travail
- Semaine: lire le cours (`modules/`)
- Pratique: exécuter notebook (`notebooks/`)
- Application: faire exercices (`exercises/`) puis projets (`projects/`)
- Correction: comparer avec `solutions/`
