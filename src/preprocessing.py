import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.feature_engineering import FeatureEngineer


DATA_PATH = Path(__file__).parent.parent / "data" / "Titanic-Dataset.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut."""
    return pd.read_csv(path)


def preprocess(
    path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Pipeline complet : chargement → feature engineering → split train/test.

    Retourne :
        X_train, X_test, y_train, y_test  (DataFrames / Series numpy-ready)
    """
    df = load_data(path)

    # Séparation cible / features AVANT le feature engineering
    # (pour ne pas fuiter les infos du test set dans le fit)
    y = df['Survived']
    X = df.drop(columns=['Survived'])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature engineering : fit sur le train, transform sur les deux
    fe = FeatureEngineer()
    X_train = fe.fit(X_train_raw.copy().assign(Survived=y_train)).transform(X_train_raw)
    X_test = fe.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, fe
