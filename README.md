# 🚢 Titanic Survival Prediction

Prédiction de la survie des passagers du Titanic à l’aide du Machine Learning, déployée sous forme d’une **application web interactive avec Streamlit**.

👉 **Démo en ligne** :
[https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app](https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app)

---

## 📌 Objectif du projet

Ce projet a pour but de :

* Construire un pipeline complet de Data Science
* Comparer plusieurs modèles de classification
* Déployer une application interactive
* Expliquer les prédictions avec **SHAP**

---

## 🗂️ Structure du projet

```
Titanic_survival/
├── data/
│   └── Titanic-Dataset.csv
├── models/
│   ├── XGBoost.joblib
│   ├── feature_engineer.joblib
│   └── X_train.parquet
├── src/
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── app/
│   └── interface.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Démo

👉 [https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app](https://titanicsurvival-cvdxgqvfxtea79vpjlxz99.streamlit.app)

### Fonctionnalités

* Saisie des caractéristiques d’un passager
* Prédiction en temps réel
* Explication avec SHAP

---

## ⚙️ Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### Entraînement

```bash
python -m src.train_model
```

### Évaluation

```bash
python -m src.evaluate_model
```

### Lancer l'application

```bash
streamlit run app/interface.py
```

---

## 📊 Résultats

* Accuracy : 81%
* ROC-AUC : 0.84

---

## 🧠 Interprétabilité

Utilisation de **SHAP** pour expliquer chaque prédiction.

---

## 🛠️ Stack technique

* Python
* scikit-learn
* XGBoost
* SHAP
* Streamlit

---

## 👤 Auteur

Kevin Sauvage

---

## 💡 Améliorations possibles

* API avec FastAPI
* Docker
* MLflow
