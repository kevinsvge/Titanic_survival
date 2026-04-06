import pandas as pd
import numpy as np


TITLE_MAP = {
    'Mr': 0,
    'Miss': 1,
    'Mrs': 2,
    'Master': 3,
    'Rare': 4,
}

EMBARKED_MAP = {'S': 0, 'C': 1, 'Q': 2}

AGE_BINS = [0, 16, 32, 48, 64, 200]
AGE_LABELS = [0, 1, 2, 3, 4]


class FeatureEngineer:
    """
    Transformations feature engineering pour le dataset Titanic.

    Utilise le pattern fit/transform (compatible sklearn) afin d'apprendre
    les paramètres uniquement sur le train set et de les appliquer
    identiquement sur le test set.
    """

    def __init__(self):
        self.age_medians: dict = {}
        self.fare_median: float = None
        self.fare_bins: list = None   # bins calcules sur le train pour pd.cut
        self.deck_categories: list = None

    # ------------------------------------------------------------------
    # Méthodes publiques
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Apprend les paramètres de transformation sur le train set."""
        df = df.copy()
        df['Title'] = self._extract_title(df)

        # Médianes d'âge par (Title, Pclass)
        self.age_medians = (
            df.groupby(['Title', 'Pclass'])['Age']
            .median()
            .to_dict()
        )

        # Mediane du tarif + bins des quartiles (stockes pour pd.cut en transform)
        self.fare_median = df['Fare'].median()
        _, self.fare_bins = pd.qcut(df['Fare'].fillna(self.fare_median), 4, retbins=True, duplicates='drop')
        self.fare_bins[0] = -np.inf   # pour couvrir les valeurs hors-range
        self.fare_bins[-1] = np.inf

        # Catégories de Deck présentes dans le train set + 'U' (Unknown)
        decks = df['Cabin'].dropna().str[0].unique().tolist()
        if 'U' not in decks:
            decks.append('U')
        self.deck_categories = sorted(decks)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les transformations (fit doit avoir été appelé avant)."""
        if self.fare_median is None:
            raise RuntimeError("Appelle fit() avant transform().")

        df = df.copy()

        df['Title'] = self._extract_title(df)
        df['Age'] = self._impute_age(df)
        df['AgeBand'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS).astype(int)

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        df['Fare'] = df['Fare'].fillna(self.fare_median)
        n_bins = len(self.fare_bins) - 1
        df['FareBand'] = pd.cut(df['Fare'], bins=self.fare_bins, labels=list(range(n_bins))).astype(int)

        df['Deck'] = self._extract_deck(df)

        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].fillna('S').map(EMBARKED_MAP)
        df['Title'] = df['Title'].map(TITLE_MAP)

        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age', 'Fare', 'SibSp', 'Parch']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raccourci fit() + transform() en une seule étape (train set uniquement)."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Méthodes privées
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(df: pd.DataFrame) -> pd.Series:
        """Extrait le titre depuis la colonne Name et regroupe les rares."""
        titles = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
        common = {'Mr', 'Miss', 'Mrs', 'Master'}
        return titles.apply(lambda t: t if t in common else 'Rare')

    def _impute_age(self, df: pd.DataFrame) -> pd.Series:
        """Impute l'âge manquant via la médiane par (Title, Pclass)."""
        age = df['Age'].copy()
        mask = age.isnull()
        for idx in df[mask].index:
            key = (df.at[idx, 'Title'], df.at[idx, 'Pclass'])
            median = self.age_medians.get(key)
            if median is None:
                # Fallback : médiane globale stockée lors du fit
                median = next(iter(self.age_medians.values()), 28.0)
            age.at[idx] = median
        return age

    def _extract_deck(self, df: pd.DataFrame) -> pd.Series:
        """Extrait la lettre du pont depuis Cabin, 'U' si inconnu, encodé numériquement."""
        deck_letter = df['Cabin'].apply(
            lambda c: c[0] if pd.notna(c) and len(c) > 0 else 'U'
        )
        # Remplace les lettres inconnues du test set par 'U'
        deck_letter = deck_letter.apply(
            lambda d: d if d in self.deck_categories else 'U'
        )
        deck_index = {d: i for i, d in enumerate(self.deck_categories)}
        return deck_letter.map(deck_index)
