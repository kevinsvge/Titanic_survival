# Titanic Survival Prediction

Prediction de la survie des passagers du Titanic avec Machine Learning,
deployee sous forme d'interface web interactive avec Streamlit.

**Demo en ligne** : https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app

---

## Structure du projet

```
Titanic_survival/
├── data/
│   └── Titanic-Dataset.csv          # Dataset brut (891 passagers, 12 colonnes)
├── models/
│   ├── XGBoost.joblib               # Modele final sauvegarde
│   ├── feature_engineer.joblib      # FeatureEngineer fitte
│   └── X_train.parquet              # Train set (utilise par SHAP en production)
├── src/
│   ├── feature_engineering.py       # Classe FeatureEngineer (fit/transform)
│   ├── preprocessing.py             # Chargement + split train/test
│   ├── train_model.py               # Selection + tuning + sauvegarde
│   └── evaluate_model.py            # Metriques + visualisations + SHAP global
├── app/
│   └── interface.py                 # Interface Streamlit + SHAP par prediction
├── notebooks/
│   └── exploration.ipynb            # Analyse exploratoire des donnees
├── requirements.txt
└── README.md
```

---

## Demo

L'application est deployee sur Streamlit Cloud et accessible sans installation :

**https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app**

Fonctionnalites :
- Saisie des caracteristiques d'un passager (classe, age, sexe, etc.)
- Prediction de survie avec probabilite en temps reel
- Graphique SHAP expliquant pourquoi le modele a pris cette decision

---

## Installation locale

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Utilisation

### 1. Entrainer le modele

A faire une seule fois (ou pour reentreiner depuis zero).
Genere `models/XGBoost.joblib`, `models/feature_engineer.joblib` et `models/X_train.parquet`.

```bash
python -m src.train_model
```

### 2. Evaluer le modele

Affiche les metriques, la courbe ROC, la matrice de confusion et le graphique SHAP global.

```bash
python -m src.evaluate_model
```

### 3. Lancer l'interface Streamlit

#### En local
```bash
streamlit run app/interface.py
```
Puis ouvrir `http://localhost:8501` dans le navigateur.

#### En remote (acces depuis une autre machine)
```bash
streamlit run app/interface.py --server.address 0.0.0.0 --server.port 8501
```
Puis ouvrir `http://<IP_DU_SERVEUR>:8501` dans le navigateur.

#### Via tunnel SSH (sans ouvrir de port)
```bash
# Dans un terminal LOCAL
ssh -L 8501:localhost:8501 user@<IP_DU_SERVEUR>
```
Puis ouvrir `http://localhost:8501` dans le navigateur local.

---

## Pipeline de donnees

### Feature Engineering (`src/feature_engineering.py`)

Transformations appliquees sur le dataset brut :

| Feature cree | Source | Description |
|---|---|---|
| `Title` | `Name` | Titre extrait (Mr, Mrs, Miss, Master, Rare) |
| `AgeBand` | `Age` | 5 tranches d'age (0-16, 16-32, 32-48, 48-64, 64+) |
| `FamilySize` | `SibSp + Parch` | Taille totale du groupe familial |
| `IsAlone` | `FamilySize` | 1 si le passager voyage seul |
| `FareBand` | `Fare` | 4 quartiles de tarif |
| `Deck` | `Cabin` | Lettre du pont (U = inconnu) |

Variables encodees : `Sex` (0/1), `Embarked` (0/1/2), `Title` (0-4), `Deck` (0-8)

Variables supprimees : `Name`, `Ticket`, `Cabin`, `PassengerId`, `Age`, `Fare`, `SibSp`, `Parch`

Le pattern `fit` / `transform` evite tout data leakage : les statistiques
(medianes d'age, bins de tarif, etc.) sont apprises uniquement sur le train set.

### Preprocessing (`src/preprocessing.py`)

- Chargement du CSV
- Separation `X` / `y` avant le feature engineering
- Split stratifie 80/20 (`stratify=y` pour conserver la proportion de survivants)
- Application du `FeatureEngineer` : `fit` sur train, `transform` sur train et test

### Entrainement (`src/train_model.py`)

1. **Cross-validation (5 folds)** sur les 3 modeles candidats
2. **GridSearchCV** sur le gagnant pour optimiser les hyperparametres
3. **Sauvegarde** du modele, du FeatureEngineer et de X_train avec `joblib` / `parquet`

Resultats obtenus :
```
LogisticRegression   acc=0.8020
RandomForest         acc=0.8049
XGBoost              acc=0.8147  << gagnant

Meilleurs params XGBoost : learning_rate=0.1, max_depth=3, n_estimators=300
Score CV apres tuning    : 0.8287
```

---

## Evaluation (`src/evaluate_model.py`)

Metriques calculees sur le test set (179 passagers) :

- **Accuracy** : 81%
- **ROC-AUC** : 0.84
- **Rapport de classification** : precision, recall, F1 par classe
- **Matrice de confusion** : visualisation des erreurs de prediction
- **SHAP summary** : impact global de chaque feature sur l'ensemble du dataset

---

## Interpretabilite — SHAP

L'analyse SHAP (SHapley Additive exPlanations) explique chaque prediction individuellement.

**Exemple pour une femme de 1ere classe (Cherbourg, 28 ans, billet a 80£) :**

```
Title (Mrs)     +2.01   forte indication de survie
FareBand        +0.72   billet cher
Sex (femme)     +0.70   survie tres favorisee
Pclass (1ere)   +0.66   classe privilegiee
Deck (inconnu)  -0.18   cabine non connue
                ------
Probabilite de survie : 99%
```

Le graphique waterfall est affiche directement dans l'interface apres chaque prediction.

---

## Technologies

- Python 3.12
- scikit-learn
- XGBoost
- SHAP
- pandas, numpy
- matplotlib, seaborn
- joblib
- Streamlit

---

## Auteur

Projet realise par **Kevin Sauvage** dans une demarche d'apprentissage et de valorisation
des competences en Data Science et Machine Learning.
