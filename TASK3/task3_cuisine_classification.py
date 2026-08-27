"""
Task 3 - Cuisine Classification (Multi-class Classification)
-------------------------------------------------------------
Cognifyz ML Internship

Goal:
  Predict a restaurant's primary cuisine from structured restaurant features.

Why this task needs care:
  The raw `Cuisines` column often contains multiple comma-separated cuisines,
  while the classification target must be a single label per row. We extract
  the first cuisine token as the "primary cuisine", then collapse rare classes
  into an `Other` bucket so the model can learn stable patterns.

Run:
  python task3_cuisine_classification.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_primary_cuisine(value: object) -> str:
    """Take the first cuisine token as the primary cuisine label."""
    if pd.isna(value):
        return "Unknown"
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts[0] if parts else "Unknown"


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    """Map Yes/No columns to 1/0 for modeling."""
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    return series.map(mapping).fillna(0).astype(int)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data(dataset_path: Path, rare_threshold: int = 30) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Prepare the dataset for multi-class cuisine classification.

    Why collapsing rare classes is standard practice, not cheating:
      If a cuisine appears only a handful of times, the model has almost no
      chance to learn a generalizable pattern for it. Grouping rare cuisines
      into `Other` prevents unstable metrics and makes the task more realistic.
    """
    df = pd.read_csv(dataset_path)

    rename_map = {
        "Has Table booking": "Table booking",
        "Has Online delivery": "Online delivery",
        "Restaurant Name": "Name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    stats = {"rows_raw": int(len(df))}

    # Fill missing cuisine text before label extraction.
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # Primary cuisine = first listed cuisine.
    df["Primary Cuisine"] = df["Cuisines"].apply(extract_primary_cuisine)

    cuisine_counts = df["Primary Cuisine"].value_counts()
    frequent_cuisines = cuisine_counts[cuisine_counts >= rare_threshold].index

    df["Cuisine Target"] = df["Primary Cuisine"].where(
        df["Primary Cuisine"].isin(frequent_cuisines), "Other"
    )

    stats["unique_primary_cuisines_raw"] = int(cuisine_counts.shape[0])
    stats["rare_threshold"] = int(rare_threshold)
    stats["classes_kept"] = int(len(frequent_cuisines))
    stats["rows_mapped_to_other"] = int((df["Cuisine Target"] == "Other").sum())
    stats["final_target_classes"] = int(df["Cuisine Target"].nunique())

    # Numeric cleanup for predictor columns used below.
    for col in ["Price range", "Average Cost for two", "Votes", "Aggregate rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill small missing numeric gaps with medians to keep rows usable.
    for col in ["Price range", "Average Cost for two", "Votes", "Aggregate rating"]:
        df[col] = df[col].fillna(df[col].median())

    return df, stats


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[int, str]]:
    """
    Build predictor features.

    Features chosen for this task:
    - City / Country Code: location can be strongly tied to cuisine patterns
    - Price range / cost: some cuisines cluster in certain price bands
    - Table booking / online delivery: service patterns can differ by cuisine
    - log_votes / aggregate rating: popularity and perceived quality add signal
    """
    features = df.copy()

    features["Table booking"] = yes_no_to_binary(features["Table booking"])
    features["Online delivery"] = yes_no_to_binary(features["Online delivery"])
    features["log_votes"] = np.log1p(features["Votes"].clip(lower=0))

    def cost_zscore(group: pd.Series) -> pd.Series:
        sd = group.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / sd

    # Helps because raw cost values are not directly comparable across countries.
    features["cost_z"] = features.groupby("Country Code")["Average Cost for two"].transform(cost_zscore)

    # Use compact category codes for high-cardinality columns.
    for col in ["City", "Country Code"]:
        features[col] = features[col].astype("category").cat.codes

    # Encode target labels as integers for easier confusion-matrix handling.
    target_cat = features["Cuisine Target"].astype("category")
    target_mapping = dict(enumerate(target_cat.cat.categories))
    y = target_cat.cat.codes

    feature_cols = [
        "City",
        "Country Code",
        "Price range",
        "Average Cost for two",
        "cost_z",
        "Table booking",
        "Online delivery",
        "log_votes",
        "Aggregate rating",
    ]

    X = features[feature_cols].copy()
    return X, y, feature_cols, target_mapping


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Train a baseline and a stronger model.

    Why class balancing matters:
      Without `class_weight='balanced'`, the Random Forest will heavily favor
      majority cuisines such as `North Indian` and neglect smaller classes.
      Balancing tells the model to pay more attention to underrepresented labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    numeric_cols = ["Price range", "Average Cost for two", "cost_z", "log_votes", "Aggregate rating"]
    passthrough_cols = [c for c in feature_cols if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("pass", "passthrough", passthrough_cols),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                ),
            ),
        ]
    )

    rf_model = RandomForestClassifier(
        n_estimators=350,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    lr_metrics = {
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "macro_f1": float(f1_score(y_test, lr_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, lr_pred, average="weighted")),
    }
    rf_metrics = {
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "macro_f1": float(f1_score(y_test, rf_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, rf_pred, average="weighted")),
    }

    rf_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "lr_pipeline": lr_pipeline,
        "rf_model": rf_model,
        "lr_pred": lr_pred,
        "rf_pred": rf_pred,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "rf_importance": rf_importance,
    }


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_mapping: Dict[int, str],
    top_n_classes: int = 12,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Build a readable confusion matrix for the most common target classes.

    A full matrix across many classes is hard to read, so we focus on the most
    frequent cuisines plus `Other` when it appears.
    """
    y_true_series = pd.Series(y_true)
    common_labels = y_true_series.value_counts().head(top_n_classes).index.tolist()

    other_label = next((k for k, v in target_mapping.items() if v == "Other"), None)
    if other_label is not None and other_label not in common_labels:
        common_labels.append(other_label)

    cm = confusion_matrix(y_true, y_pred, labels=common_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[target_mapping[i] for i in common_labels],
        columns=[target_mapping[i] for i in common_labels],
    )
    return cm_df, common_labels


def save_confusion_matrix_plot(cm_df: pd.DataFrame, out_path: Path) -> None:
    """Save the confusion matrix as an image for the report/README."""
    plt.figure(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_df.values, display_labels=cm_df.columns)
    disp.plot(cmap="Blues", xticks_rotation=60, colorbar=False)
    plt.title("Confusion Matrix (Most Common Cuisine Classes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n{name}:")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1   : {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    repo_dir = task_dir.parent
    dataset_path = repo_dir / "Dataset .csv"
    output_dir = task_dir / "output"
    ensure_output_dir(output_dir)

    print("=" * 72)
    print("Task 3 - Cuisine Classification")
    print("=" * 72)

    # Step 1-2: target extraction and class-imbalance handling
    df, prep_stats = load_and_prepare_data(dataset_path, rare_threshold=30)
    print("\n[Step 1-2] Target preparation summary")
    for key, value in prep_stats.items():
        print(f"  {key}: {value}")
    print(
        "\n  Why use the first cuisine as target?\n"
        "  The model needs one label per row. The first cuisine is a simple,\n"
        "  consistent proxy for the restaurant's primary cuisine identity."
    )
    print(
        "\n  Why collapse rare classes into 'Other'?\n"
        "  Classes with very few samples produce unstable metrics and poor\n"
        "  generalization. Grouping them is standard class-imbalance handling."
    )

    # Step 3: features
    X, y, feature_cols, target_mapping = build_features(df)
    print("\n[Step 3] Predictors used")
    print(f"  {feature_cols}")

    # Step 4-6: split, models, evaluation
    results = train_models(X, y, feature_cols, random_state=42)
    print("\n[Step 4] Train/test split with stratification")
    print(f"  Train rows: {len(results['X_train'])} | Test rows: {len(results['X_test'])}")

    print("\n[Step 5-6] Model evaluation")
    print_metrics("Logistic Regression (baseline)", results["lr_metrics"])
    print_metrics("Random Forest (balanced)", results["rf_metrics"])

    print(
        "\n  Why macro-F1 matters more than accuracy here:\n"
        "  Accuracy is dominated by the biggest cuisine classes. Macro-F1 gives\n"
        "  equal weight to every class, so it reflects how well the model treats\n"
        "  minority cuisines instead of only the majority ones."
    )

    # Detailed reports
    target_names = [target_mapping[i] for i in sorted(target_mapping)]
    lr_report = classification_report(
        results["y_test"],
        results["lr_pred"],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    rf_report = classification_report(
        results["y_test"],
        results["rf_pred"],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    print("\n[Step 6] Random Forest feature importance")
    print(results["rf_importance"].to_string(index=False))

    # Step 7: confusion matrix
    cm_df, labels_used = build_confusion_matrix(
        results["y_test"].to_numpy(),
        results["rf_pred"],
        target_mapping,
        top_n_classes=12,
    )
    print("\n[Step 7] Confusion matrix saved for the most common cuisine classes.")
    print("  Use it to inspect which similar cuisines are mixed up by the model.")

    # Also surface the most common confusion pairs excluding correct predictions.
    confusion_pairs = []
    for true_label in cm_df.index:
        row = cm_df.loc[true_label].copy()
        if true_label in row.index:
            row.loc[true_label] = 0
        pred_label = row.idxmax()
        pred_count = int(row.max())
        if pred_count > 0:
            confusion_pairs.append(
                {"true_class": true_label, "predicted_as": pred_label, "count": pred_count}
            )
    confusion_pairs_df = pd.DataFrame(confusion_pairs).sort_values("count", ascending=False)
    if not confusion_pairs_df.empty:
        print("\n  Top confusion pairs:")
        print(confusion_pairs_df.head(10).to_string(index=False))

    # Save outputs
    metrics_df = pd.DataFrame(
        [
            {"model": "Logistic Regression", **results["lr_metrics"]},
            {"model": "Random Forest (balanced)", **results["rf_metrics"]},
        ]
    )
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    pd.DataFrame(lr_report).transpose().to_csv(output_dir / "classification_report_logistic_regression.csv")
    pd.DataFrame(rf_report).transpose().to_csv(output_dir / "classification_report_random_forest.csv")
    results["rf_importance"].to_csv(output_dir / "feature_importance_random_forest.csv", index=False)
    cm_df.to_csv(output_dir / "confusion_matrix_top_classes.csv")
    confusion_pairs_df.to_csv(output_dir / "top_confusion_pairs.csv", index=False)

    save_confusion_matrix_plot(cm_df, output_dir / "confusion_matrix_top_classes.png")

    with open(output_dir / "preparation_stats.json", "w", encoding="utf-8") as f:
        json.dump(prep_stats, f, indent=2)

    predictions = pd.DataFrame(
        {
            "actual_label_code": results["y_test"].to_numpy(),
            "actual_label": [target_mapping[i] for i in results["y_test"].to_numpy()],
            "lr_pred_code": results["lr_pred"],
            "lr_pred_label": [target_mapping[i] for i in results["lr_pred"]],
            "rf_pred_code": results["rf_pred"],
            "rf_pred_label": [target_mapping[i] for i in results["rf_pred"]],
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    label_map_df = pd.DataFrame(
        {"label_code": list(target_mapping.keys()), "label_name": list(target_mapping.values())}
    )
    label_map_df.to_csv(output_dir / "target_label_mapping.csv", index=False)

    print("\n[Step 8] Practical improvements beyond baseline")
    print("  1) Add more features: locality, rating color/text, and richer cuisine text features")
    print("  2) Tune Random Forest hyperparameters or try XGBoost/LightGBM")
    print("  3) Try SMOTE only after careful validation, because synthetic minority samples can be noisy")

    print("\nCommon mistake to avoid:")
    print("  Reporting only accuracy on this imbalanced multi-class problem can be misleading.")
    print("  Always lead with macro-F1 and support it with the per-class report.")

    print(f"\nSaved outputs to: {output_dir}")
    print("All done.")


if __name__ == "__main__":
    main()
