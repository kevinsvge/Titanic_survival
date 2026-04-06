import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.train_model import train, load_model


# ---------------------------------------------------------------------------
# 1. Metriques de classification
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test, verbose: bool = True) -> dict:
    """
    Calcule les metriques cles sur le test set et retourne un dict.

    Metriques retournees :
      - accuracy  : taux de bonne classification global
      - roc_auc   : aire sous la courbe ROC (capacite a distinguer 0/1)
      - report    : precision, recall, f1 par classe (format texte)
      - cm        : matrice de confusion (numpy array)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # proba d'etre survivant

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, target_names=["Mort", "Survivant"]),
        "cm": confusion_matrix(y_test, y_pred),
    }

    if verbose:
        print("=== Resultats sur le test set ===")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print()
        print(metrics["report"])

    return metrics


# ---------------------------------------------------------------------------
# 2. Visualisations
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Affiche la matrice de confusion en heatmap.

    La diagonale = bonnes predictions.
    Hors diagonale = erreurs (faux positifs / faux negatifs).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predit Mort", "Predit Survivant"],
        yticklabels=["Reel Mort", "Reel Survivant"],
        ax=ax,
    )
    ax.set_title("Matrice de confusion")
    ax.set_ylabel("Valeur reelle")
    ax.set_xlabel("Valeur predite")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Matrice sauvegardee -> {save_path}")
    plt.show()


def plot_roc_curve(model, X_test, y_test, save_path: str = None):
    """
    Trace la courbe ROC.

    La courbe montre le compromis entre le taux de vrais positifs (sensibilite)
    et le taux de faux positifs pour differents seuils de decision.
    Un AUC proche de 1.0 = excellent modele.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Aleatoire")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Courbe ROC sauvegardee -> {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 9, save_path: str = None):
    """
    Affiche l'importance des features selon le modele.

    Permet de comprendre quelles variables ont le plus influence
    les predictions (interpretabilite du modele).
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color="steelblue",
    )
    ax.set_title(f"Importance des {top_n} features principales")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Importance des features sauvegardee -> {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Analyse SHAP globale
# ---------------------------------------------------------------------------
def compute_shap_explainer(model, X_train):
    """
    Cree un explainer SHAP adapte a XGBoost/RandomForest.

    TreeExplainer est optimise pour les modeles a base d'arbres :
    il calcule exactement les valeurs SHAP sans approximation.
    """
    return shap.TreeExplainer(model, X_train)


def plot_shap_summary(explainer, X, save_path: str = None):
    """
    Graphique SHAP summary (beeswarm) sur l'ensemble du dataset.

    Chaque point = un passager.
    Position horizontale = impact sur la prediction (positif = pousse vers survie).
    Couleur = valeur de la feature (rouge = haute, bleu = basse).
    """
    shap_values = explainer(X)
    # Pour la classification binaire XGBoost, on prend les SHAP de la classe 1 (survivant)
    if len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.beeswarm(sv, show=False)
    plt.title("Impact global des features (SHAP)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP summary sauvegarde -> {save_path}")
    plt.show()


def get_shap_values_single(explainer, X_single):
    """
    Retourne les valeurs SHAP pour un seul passager.
    Utilise dans l'interface Streamlit pour la prediction individuelle.
    """
    shap_values = explainer(X_single)
    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values


# ---------------------------------------------------------------------------
# 4. Pipeline complet d'evaluation
# ---------------------------------------------------------------------------
def evaluate_pipeline(from_disk: bool = False, model_name: str = "XGBoost"):
    """
    Lance l'evaluation complete :
      - Entraine (ou charge depuis disk) le modele
      - Calcule les metriques
      - Affiche les visualisations + analyse SHAP globale
    """
    if from_disk:
        from src.preprocessing import preprocess
        model = load_model(model_name)
        X_train, X_test, _, y_test, _ = preprocess()
    else:
        model, X_test, y_test = train(save=True)
        X_train = X_test  # fallback minimal

    metrics = evaluate(model, X_test, y_test)
    plot_confusion_matrix(metrics["cm"])
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X_test.columns.tolist())

    explainer = compute_shap_explainer(model, X_train)
    plot_shap_summary(explainer, X_test)

    return metrics


if __name__ == "__main__":
    evaluate_pipeline()
