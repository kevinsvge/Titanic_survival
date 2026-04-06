# 🚢 Titanic Survival Prediction

> Prédiction de la survie des passagers du Titanic à l’aide du Machine Learning
> déployée via une interface web interactive avec **Streamlit**

🔗 **Demo en ligne**
👉 https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app

---

## 📂 Structure du projet

```bash
Titanic_survival/
├── data/
│   └── Titanic-Dataset.csv          # Dataset brut (891 passagers, 12 colonnes)
├── models/
│   ├── XGBoost.joblib               # Modèle final sauvegardé
│   ├── feature_engineer.joblib      # FeatureEngineer fitté
│   └── X_train.parquet              # Train set (utilisé pour SHAP)
├── src/
│   ├── feature_engineering.py       # Feature engineering (fit/transform)
│   ├── preprocessing.py             # Chargement + split train/test
│   ├── train_model.py               # Sélection + tuning + sauvegarde
│   └── evaluate_model.py            # Metrics + visualisations + SHAP
├── app/
│   └── interface.py                 # Interface Streamlit
├── notebooks/
│   └── exploration.ipynb            # Analyse exploratoire
├── requirements.txt
└── README.md
```

---

## 🚀 Fonctionnalités

* 🎯 Saisie des caractéristiques d’un passager (âge, sexe, classe…)
* 🤖 Prédiction de survie en temps réel
* 📊 Probabilité associée
* 🔍 Explication du modèle avec **SHAP**
* 📉 Visualisation de l’impact des variables

---

## ⚙️ Installation locale

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 🧪 Utilisation

### 1. Entraîner le modèle

```bash
python -m src.train_model
```

➡️ Génère :

* `models/XGBoost.joblib`
* `models/feature_engineer.joblib`
* `models/X_train.parquet`

---

### 2. Évaluer le modèle

```bash
python -m src.evaluate_model
```

➡️ Affiche :

* Accuracy
* ROC Curve
* Matrice de confusion
* SHAP global

---

### 3. Lancer l'application Streamlit

#### 🔹 En local

```bash
streamlit run app/interface.py
```

👉 http://localhost:8501

---

#### 🔹 En remote

```bash
streamlit run app/interface.py --server.address 0.0.0.0 --server.port 8501
```

👉 http://<IP_DU_SERVEUR>:8501

---

#### 🔹 Via tunnel SSH

```bash
ssh -L 8501:localhost:8501 user@<IP_DU_SERVEUR>
```

👉 http://localhost:8501

---

## 🔄 Pipeline de données

### 🧠 Feature Engineering

| Feature    | Source        | Description                    |
| ---------- | ------------- | ------------------------------ |
| Title      | Name          | Titre extrait (Mr, Mrs, Miss…) |
| AgeBand    | Age           | 5 tranches d'âge               |
| FamilySize | SibSp + Parch | Taille de la famille           |
| IsAlone    | FamilySize    | 1 si le passager voyage seul   |
| FareBand   | Fare          | 4 quartiles                    |
| Deck       | Cabin         | Pont du bateau                 |

#### ✅ Encodage

* `Sex` → 0 / 1
* `Embarked` → 0 / 1 / 2
* `Title`, `Deck` → encodés

#### ❌ Variables supprimées

* Name, Ticket, Cabin, PassengerId
* Age, Fare, SibSp, Parch

#### ⚠️ Pas de data leakage

* `fit` uniquement sur le train
* `transform` sur train + test

---

### 🔧 Preprocessing

* Chargement du CSV
* Séparation X / y
* Split **80/20 stratifié**
* Application du Feature Engineering

---

### 🤖 Entraînement du modèle

1. Cross-validation (5 folds)
2. Comparaison de modèles
3. GridSearchCV sur le meilleur

#### 📊 Résultats

```text
LogisticRegression   acc=0.8020
RandomForest         acc=0.8049
XGBoost              acc=0.8147  ✅
```

#### ⚙️ Meilleurs paramètres

* learning_rate = 0.1
* max_depth = 3
* n_estimators = 300

📈 Score final CV : **0.8287**

---

## 📊 Évaluation

Sur le test set (179 passagers) :

* ✅ Accuracy : **81%**
* 📈 ROC-AUC : **0.84**
* 📊 Matrice de confusion
* 📉 SHAP summary plot

---

## 🔍 Interprétabilité (SHAP)

Chaque prédiction est expliquée individuellement.

```text
Title (Mrs)     +2.01
FareBand        +0.72
Sex (femme)     +0.70
Pclass (1ère)   +0.66
Deck inconnu    -0.18
-----------------------
Probabilité : 99%
```

📌 Visualisation directement dans l’interface (waterfall plot)

---

## 🛠️ Technologies utilisées

* Python 3.12
* scikit-learn
* XGBoost
* SHAP
* pandas / numpy
* matplotlib / seaborn
* Streamlit
* joblib

---

## 👨‍💻 Auteur

Projet réalisé par **Kevin Sauvage**

🎯 Objectifs :

* Développer des compétences en Data Science
* Construire un pipeline ML complet
* Déployer une application interactive

---

## ⭐ Support

Si ce projet t’a aidé ou inspiré :

👉 N’hésite pas à laisser une ⭐ sur le repo !
