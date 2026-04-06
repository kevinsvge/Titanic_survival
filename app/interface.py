import sys
from pathlib import Path

# Permet d'importer les modules src/ depuis app/
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from src.train_model import load_model, load_feature_engineer, load_X_train
from src.evaluate_model import compute_shap_explainer, get_shap_values_single


# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic - Prediction de survie",
    page_icon="🚢",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Chargement du modele, du FeatureEngineer et de l'explainer SHAP (en cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("XGBoost")
    fe = load_feature_engineer()
    X_train = load_X_train()
    explainer = compute_shap_explainer(model, X_train)
    return model, fe, explainer


model, fe, explainer = load_artifacts()


# ---------------------------------------------------------------------------
# En-tete
# ---------------------------------------------------------------------------
st.title("🚢 Titanic — Prediction de survie")
st.markdown(
    "Renseignez les caracteristiques d'un passager pour predire "
    "s'il aurait survecu au naufrage du Titanic."
)
st.divider()


# ---------------------------------------------------------------------------
# Formulaire de saisie
# ---------------------------------------------------------------------------
st.subheader("Caracteristiques du passager")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Classe du billet",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1ere classe", 2: "2eme classe", 3: "3eme classe"}[x],
    )
    sex = st.radio("Sexe", options=["male", "female"], format_func=lambda x: "Homme" if x == "male" else "Femme")
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    embarked = st.selectbox(
        "Port d'embarquement",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
    )

with col2:
    sibsp = st.number_input("Freres / Soeurs / Conjoint a bord", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents / Enfants a bord", min_value=0, max_value=6, value=0)
    fare = st.number_input("Prix du billet (£)", min_value=0.0, max_value=520.0, value=32.0, step=1.0)
    cabin = st.selectbox(
        "Pont (Cabin)",
        options=["Inconnu", "A", "B", "C", "D", "E", "F", "G"],
        help="Laissez 'Inconnu' si le numero de cabine n'est pas connu.",
    )

st.divider()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def build_passenger_df() -> pd.DataFrame:
    """Construit un DataFrame d'une ligne a partir des inputs du formulaire."""
    cabin_value = None if cabin == "Inconnu" else f"{cabin}1"
    return pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": pclass,
        "Name": "Doe, Mr. John" if sex == "male" else "Doe, Mrs. Jane",
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Ticket": "XXXXX",
        "Fare": float(fare),
        "Cabin": cabin_value,
        "Embarked": embarked,
    }])


if st.button("Predire la survie", type="primary", use_container_width=True):

    passenger_df = build_passenger_df()
    X = fe.transform(passenger_df)
    proba_survive = model.predict_proba(X)[0][1]
    prediction = int(proba_survive >= 0.5)

    st.divider()
    st.subheader("Resultat")

    # --- Verdict principal ---
    if prediction == 1:
        st.success(f"### Survivant — probabilite : {proba_survive:.0%}")
    else:
        st.error(f"### Non-survivant — probabilite de survie : {proba_survive:.0%}")

    # --- Jauge de probabilite ---
    st.markdown("**Probabilite de survie**")
    st.progress(float(proba_survive))

    # --- Graphique en barres survie / deces ---
    fig, ax = plt.subplots(figsize=(4, 2.5))
    bars = ax.barh(
        ["Non-survivant", "Survivant"],
        [1 - proba_survive, proba_survive],
        color=["#e74c3c", "#2ecc71"],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilite")
    ax.bar_label(bars, fmt="{:.0%}", padding=4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Analyse SHAP locale ---
    st.divider()
    st.subheader("Pourquoi cette prediction ? (SHAP)")
    st.markdown(
        "Chaque barre montre l'impact d'une feature sur la prediction. "
        "**Rouge** = pousse vers la survie. **Bleu** = pousse vers le deces."
    )

    shap_values = get_shap_values_single(explainer, X)

    fig_shap, ax_shap = plt.subplots(figsize=(7, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig_shap)

    # --- Recapitulatif du passager ---
    st.divider()
    st.subheader("Recapitulatif")
    family_size = int(sibsp) + int(parch) + 1
    st.markdown(f"""
| Caracteristique | Valeur |
|---|---|
| Classe | {pclass}ere |
| Sexe | {"Homme" if sex == "male" else "Femme"} |
| Age | {age} ans |
| Taille du groupe familial | {family_size} |
| Prix du billet | £{fare:.0f} |
| Port d'embarquement | {"Southampton" if embarked == "S" else "Cherbourg" if embarked == "C" else "Queenstown"} |
    """)


# ---------------------------------------------------------------------------
# Sidebar — infos modele
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("A propos du modele")
    st.markdown("""
**Modele** : XGBoost
**Accuracy test set** : 81%
**ROC-AUC** : 0.84

---

**Features utilisees :**
- Classe, Sexe, Age
- Taille de la famille
- Prix du billet
- Port d'embarquement
- Titre extrait du nom
- Pont (Deck)

---

**Pipeline :**
1. Feature Engineering
2. Cross-validation (5 folds)
3. GridSearchCV
    """)
    st.caption("Dataset : Titanic (Kaggle) — 891 passagers")
