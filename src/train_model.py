import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

from src.preprocessing import preprocess


MODELS_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# 1. Catalogue des modèles candidats
#    Chaque entrée est un tuple (nom, modèle, grille d'hyperparamètres)
# ---------------------------------------------------------------------------
CANDIDATES = [
    (
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    (
        "RandomForest",
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, None],
            "min_samples_split": [2, 5],
        },
    ),
    (
        "XGBoost",
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "n_estimators": [100, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    ),
]


# ---------------------------------------------------------------------------
# 2. Sélection du meilleur modèle par cross-validation rapide (5 folds)
# ---------------------------------------------------------------------------
def select_best_model(X_train, y_train, cv: int = 5) -> tuple:
    """
    Évalue chaque modèle candidat en cross-validation et retourne
    le nom + modèle ayant le meilleur score moyen d'accuracy.
    """
    print("=== Comparaison des modèles (cross-validation) ===")
    best_name, best_model, best_score = None, None, 0.0

    for name, model, _ in CANDIDATES:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        print(f"  {name:<22} acc={mean_score:.4f}  (±{scores.std():.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = model

    print(f"\n>> Meilleur modele : {best_name}  (acc={best_score:.4f})\n")
    return best_name, best_model


# ---------------------------------------------------------------------------
# 3. Optimisation des hyperparamètres avec GridSearchCV
# ---------------------------------------------------------------------------
def tune_model(name: str, model, X_train, y_train, cv: int = 5):
    """
    Lance un GridSearchCV sur la grille d'hyperparamètres du modèle sélectionné.
    Retourne le meilleur estimateur retrained sur tout X_train.
    """
    param_grid = {n: grid for n, _, grid in CANDIDATES if n == name}[name]

    print(f"=== Optimisation hyperparamètres : {name} ===")
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Meilleurs params : {grid_search.best_params_}")
    print(f"  Meilleur score CV : {grid_search.best_score_:.4f}\n")

    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# 4. Sauvegarde du modèle entraîné
# ---------------------------------------------------------------------------
def save_model(model, name: str) -> Path:
    """Serialise le modele dans models/<name>.joblib"""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Modele sauvegarde -> {path}")
    return path


def save_feature_engineer(fe) -> Path:
    """Serialise le FeatureEngineer fitte dans models/feature_engineer.joblib"""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "feature_engineer.joblib"
    joblib.dump(fe, path)
    print(f"FeatureEngineer sauvegarde -> {path}")
    return path


def save_X_train(X_train) -> Path:
    """Sauvegarde X_train en parquet dans models/ (utilise par SHAP en production)"""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "X_train.parquet"
    X_train.to_parquet(path, index=False)
    print(f"X_train sauvegarde -> {path}")
    return path


def load_X_train():
    """Charge X_train depuis models/X_train.parquet"""
    import pandas as pd
    path = MODELS_DIR / "X_train.parquet"
    return pd.read_parquet(path)


def load_model(name: str):
    """Charge un modele depuis models/<name>.joblib"""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def load_feature_engineer():
    """Charge le FeatureEngineer depuis models/feature_engineer.joblib"""
    path = MODELS_DIR / "feature_engineer.joblib"
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 5. Pipeline complet d'entraînement
# ---------------------------------------------------------------------------
def train(save: bool = True):
    """
    Lance le pipeline complet :
      chargement → sélection → tuning → sauvegarde
    Retourne le modèle final et les données de test pour évaluation.
    """
    X_train, X_test, y_train, y_test, fe = preprocess()

    best_name, best_model = select_best_model(X_train, y_train)
    final_model = tune_model(best_name, best_model, X_train, y_train)

    if save:
        save_model(final_model, best_name)
        save_feature_engineer(fe)
        save_X_train(X_train)

    return final_model, X_test, y_test


if __name__ == "__main__":
    train()
