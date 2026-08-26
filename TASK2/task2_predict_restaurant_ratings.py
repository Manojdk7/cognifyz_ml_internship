"""
Task 2 (Internship Task 1) - Predict Restaurant Ratings (Regression)
--------------------------------------------------------------------
Cognifyz ML Internship

Goal:
  Predict Aggregate rating (continuous, roughly 0-5) from restaurant features.

Why this script is structured in steps:
  Each section maps to a decision you must justify in an internship report.
  The #1 dataset-specific trap is treating "Not rated" rows (rating=0) as real
  low ratings. We remove those before any modeling.

Run:
  python task2_predict_restaurant_ratings.py
  python task2_predict_restaurant_ratings.py --skip-tuning   # faster, baseline only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_cuisine_count(value: object) -> int:
    """Count cuisines from a comma-separated string."""
    if pd.isna(value):
        return 0
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return len(parts)


def extract_primary_cuisine(value: object) -> str:
    """Use the first cuisine token as a simple primary-cuisine label."""
    if pd.isna(value):
        return "Unknown"
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts[0] if parts else "Unknown"


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    """Map Yes/No columns to 1/0. Unknown values become 0."""
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    return series.map(mapping).fillna(0).astype(int)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for regression models."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load + clean for regression
# ---------------------------------------------------------------------------

def load_and_clean_for_regression(dataset_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean data specifically for rating prediction.

    Critical rule for this dataset:
      Rows with Rating text == "Not rated" have Aggregate rating = 0 by design.
      They are NOT genuinely bad restaurants; they simply have no ratings yet.
      Keeping them teaches the model that many restaurants have rating 0, which
      destroys regression performance and misleads feature importance.
    """
    df = pd.read_csv(dataset_path)

    stats = {"rows_raw": int(len(df))}

    # Standardize column references (guide names vs actual CSV names).
    rename_map = {
        "Has Table booking": "Table booking",
        "Has Online delivery": "Online delivery",
        "Restaurant Name": "Name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Coerce numeric columns used in modeling.
    numeric_cols = ["Aggregate rating", "Average Cost for two", "Price range", "Votes"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove unrated placeholder rows (the "0-rating trap").
    not_rated_mask = df["Rating text"].astype(str).str.strip().eq("Not rated")
    stats["rows_not_rated_removed"] = int(not_rated_mask.sum())

    # Optional extra guard: rating exactly 0 with zero votes often means "no signal".
    # We keep rated rows even if votes are low; only remove explicit Not rated labels.
    df = df.loc[~not_rated_mask].copy()
    stats["rows_after_not_rated_filter"] = int(len(df))

    # Drop rows missing the target.
    df = df.dropna(subset=["Aggregate rating"])
    stats["rows_after_target_dropna"] = int(len(df))

    # Fill missing cuisines (small count in this dataset).
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    stats["rows_final"] = int(len(df))

    return df, stats


# ---------------------------------------------------------------------------
# Step 2: Feature engineering + encoding
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Create model-ready features.

    Encoding choices (and why):
    - Table booking / Online delivery: binary 0/1 (natural Yes/No fields)
    - Primary Cuisine: first cuisine token -> category code
      (one-hot would explode dimensionality with 100+ cuisines)
    - City / Country Code: category codes (high cardinality; RF handles this well)
    - Votes: log1p transform because vote counts are heavily right-skewed
    - cuisine_count: simple signal for multi-cuisine restaurants
    """
    features = df.copy()

    features["Table booking"] = yes_no_to_binary(features["Table booking"])
    features["Online delivery"] = yes_no_to_binary(features["Online delivery"])

    features["Primary Cuisine"] = features["Cuisines"].apply(extract_primary_cuisine)
    features["cuisine_count"] = features["Cuisines"].apply(parse_cuisine_count)

    # log1p(Votes): compresses extreme values so linear models aren't dominated
    # by a few restaurants with thousands of votes.
    features["log_votes"] = np.log1p(features["Votes"].clip(lower=0))

    # Country-normalized cost: raw "Average Cost for two" mixes currencies across
    # countries (INR vs USD vs BRL). Z-score within each country makes price
    # comparable for modeling without manual currency conversion.
    def cost_zscore(group: pd.Series) -> pd.Series:
        sd = group.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / sd

    features["cost_z"] = features.groupby("Country Code")["Average Cost for two"].transform(cost_zscore)

    # Category codes: compact integer IDs for tree models and linear baseline.
    for col in ["Primary Cuisine", "City", "Country Code"]:
        features[col] = features[col].astype("category").cat.codes

    feature_cols = [
        "Price range",
        "Average Cost for two",
        "cost_z",
        "log_votes",
        "Table booking",
        "Online delivery",
        "Primary Cuisine",
        "City",
        "Country Code",
        "cuisine_count",
    ]

    X = features[feature_cols].copy()
    y = features["Aggregate rating"].copy()

    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Step 3-6: Train models, evaluate, interpret
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Train/test split + two models:
      1) Linear Regression (interpretable baseline; benefits from scaling)
      2) Random Forest Regressor (captures non-linear interactions)

    Split strategy:
      80/20 hold-out with fixed random_state for reproducibility.
      Rating prediction on this dataset is i.i.d. enough for a simple split;
      stratification is not used because the target is continuous.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    # Scale only continuous numeric columns for linear regression.
    numeric_cols = ["Price range", "Average Cost for two", "cost_z", "log_votes", "cuisine_count"]
    binary_and_cat_cols = [c for c in feature_cols if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("pass", "passthrough", binary_and_cat_cols),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    # --- Linear Regression baseline ---
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_metrics = regression_metrics(y_test.to_numpy(), lr_pred)

    # --- Random Forest ---

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = regression_metrics(y_test.to_numpy(), rf_pred)

    # Feature importance / coefficients
    lr_model = lr_pipeline.named_steps["model"]
    lr_coefs = lr_model.coef_

    # Column names after ColumnTransformer for linear model
    lr_feature_names = numeric_cols + binary_and_cat_cols
    lr_importance = pd.DataFrame(
        {
            "feature": lr_feature_names,
            "abs_coefficient": np.abs(lr_coefs),
            "coefficient": lr_coefs,
        }
    ).sort_values("abs_coefficient", ascending=False)

    rf_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    predictions = pd.DataFrame(
        {
            "actual_rating": y_test.to_numpy(),
            "lr_predicted": lr_pred,
            "rf_predicted": rf_pred,
            "lr_error": lr_pred - y_test.to_numpy(),
            "rf_error": rf_pred - y_test.to_numpy(),
        }
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "lr_pipeline": lr_pipeline,
        "rf_model": rf_model,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "lr_importance": lr_importance,
        "rf_importance": rf_importance,
        "predictions": predictions,
        "feature_cols": feature_cols,
    }


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Step 8: Hyperparameter tuning with 5-fold cross-validation.

    Why CV here?
      A single train/test split can be lucky or unlucky. Cross-validation averages
      performance across 5 different splits so hyperparameter choices generalize
      better than tuning on one hold-out set.

    Why GridSearchCV?
      Systematically tries combinations of max_depth, n_estimators, etc. and picks
      the set with the best mean CV score (lowest MAE = most interpretable for stars).
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 15, 25],
        "min_samples_leaf": [1, 3, 5],
        "min_samples_split": [2, 5],
    }

    base_rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",  # higher is better (less negative MAE)
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    print("\n[Step 8] Running GridSearchCV (5-fold CV, scoring=MAE)...")
    print("  This may take 1-3 minutes depending on your machine.")
    grid_search.fit(X_train, y_train)

    best_model: RandomForestRegressor = grid_search.best_estimator_
    tuned_pred = best_model.predict(X_test)
    tuned_metrics = regression_metrics(y_test.to_numpy(), tuned_pred)

    # CV score of the best model (convert neg MAE back to positive MAE)
    cv_mae = -grid_search.best_score_

    # Extra: per-fold MAE scores for the best estimator (for reporting stability)
    cv_fold_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_fold_mae = -cv_fold_scores

    tuned_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["mean_MAE"] = -cv_results["mean_test_score"]
    cv_results["std_MAE"] = cv_results["std_test_score"]
    cv_results = cv_results.sort_values("mean_MAE", ascending=True)

    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_mae": float(cv_mae),
        "cv_fold_mae_mean": float(cv_fold_mae.mean()),
        "cv_fold_mae_std": float(cv_fold_mae.std()),
        "cv_fold_mae_scores": cv_fold_mae.tolist(),
        "tuned_metrics": tuned_metrics,
        "tuned_pred": tuned_pred,
        "tuned_importance": tuned_importance,
        "cv_results": cv_results,
    }


def plot_feature_importance(importance_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Save a horizontal bar chart for feature importance."""
    top = importance_df.head(10).sort_values(importance_df.columns[1], ascending=True)
    value_col = top.columns[1]

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top[value_col], color="#2563eb")
    plt.xlabel(value_col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_metrics_block(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n{name} metrics:")
    print(f"  MAE : {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R2  : {metrics['R2']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict restaurant ratings (Task 2)")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip GridSearchCV (faster run, baseline models only)",
    )
    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent
    repo_dir = task_dir.parent
    dataset_path = repo_dir / "Dataset .csv"
    output_dir = task_dir / "output"
    ensure_output_dir(output_dir)

    print("=" * 72)
    print("Task 2 - Predict Restaurant Ratings (Regression)")
    print("=" * 72)

    # Step 1
    df, clean_stats = load_and_clean_for_regression(dataset_path)
    print("\n[Step 1] Cleaning summary")
    for key, value in clean_stats.items():
        print(f"  {key}: {value}")
    print(
        "\n  Why remove 'Not rated' rows?\n"
        "  They encode missing ratings as 0, not true quality. Keeping them\n"
        "  would bias the model toward predicting low ratings incorrectly."
    )

    # Step 2
    X, y, feature_cols = build_feature_matrix(df)
    print("\n[Step 2] Feature matrix ready")
    print(f"  Features used ({len(feature_cols)}): {feature_cols}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"  log_votes skew check - raw Votes max: {df['Votes'].max():.0f}")

    # Steps 3-6
    results = train_and_evaluate(X, y, feature_cols, random_state=42)

    print("\n[Step 3] Train/test split: 80% train / 20% test (random_state=42)")
    print(f"  Train rows: {len(results['X_train'])} | Test rows: {len(results['X_test'])}")

    print("\n[Step 4-5] Model evaluation")
    print_metrics_block("Linear Regression (baseline)", results["lr_metrics"])
    print_metrics_block("Random Forest Regressor", results["rf_metrics"])

    print(
        "\n  Why Random Forest usually wins here:\n"
        "  - Ratings interact non-linearly with votes, price, and city effects.\n"
        "  - Tree ensembles capture thresholds and interactions without manual feature crosses.\n"
        "  - Linear Regression assumes additive, linear effects and struggles with mixed signals."
    )

    print(
        "\n  What is a 'good' R² on this dataset?\n"
        "  - After removing unrated rows, R² around 0.35-0.55 is solid for restaurant rating prediction.\n"
        "  - R² > 0.60 is strong (votes/popularity often drive much of the signal).\n"
        "  - R² near 0 or negative means the model is not beating a naive mean predictor."
    )

    # Step 6: Feature importance
    print("\n[Step 6] Top features (Random Forest importance)")
    rf_imp = results["rf_importance"]
    print(rf_imp.to_string(index=False))

    print("\n[Step 6] Top features (Linear Regression |coefficient|)")
    lr_imp = results["lr_importance"]
    print(lr_imp.head(10).to_string(index=False))

    metrics_rows = [
        {"model": "Linear Regression", **results["lr_metrics"]},
        {"model": "Random Forest (baseline)", **results["rf_metrics"]},
    ]

    predictions = results["predictions"].copy()
    tuning_results = None

    if not args.skip_tuning:
        tuning_results = tune_random_forest(
            X_train=results["X_train"],
            y_train=results["y_train"],
            X_test=results["X_test"],
            y_test=results["y_test"],
            feature_cols=results["feature_cols"],
            random_state=42,
        )

        print("\n[Step 8] GridSearchCV results")
        print(f"  Best params: {tuning_results['best_params']}")
        print(f"  Best 5-fold CV MAE: {tuning_results['best_cv_mae']:.4f}")
        print(
            f"  CV fold MAEs: {tuning_results['cv_fold_mae_scores']} "
            f"(mean={tuning_results['cv_fold_mae_mean']:.4f}, "
            f"std={tuning_results['cv_fold_mae_std']:.4f})"
        )
        print_metrics_block("Random Forest (tuned)", tuning_results["tuned_metrics"])

        # Compare baseline vs tuned on the same test set
        delta_r2 = tuning_results["tuned_metrics"]["R2"] - results["rf_metrics"]["R2"]
        delta_mae = results["rf_metrics"]["MAE"] - tuning_results["tuned_metrics"]["MAE"]
        print(f"\n  Tuning gain vs baseline RF on test set:")
        print(f"    R2 improvement : {delta_r2:+.4f}")
        print(f"    MAE improvement: {delta_mae:+.4f} (positive = tuned is better)")

        print("\n[Step 8] Top features (Tuned Random Forest)")
        print(tuning_results["tuned_importance"].to_string(index=False))

        metrics_rows.append({"model": "Random Forest (tuned)", **tuning_results["tuned_metrics"]})
        predictions["rf_tuned_predicted"] = tuning_results["tuned_pred"]
        predictions["rf_tuned_error"] = tuning_results["tuned_pred"] - results["y_test"].to_numpy()

        # Save tuning artifacts
        with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_params": tuning_results["best_params"],
                    "best_cv_mae": tuning_results["best_cv_mae"],
                    "cv_fold_mae_mean": tuning_results["cv_fold_mae_mean"],
                    "cv_fold_mae_std": tuning_results["cv_fold_mae_std"],
                    "cv_fold_mae_scores": tuning_results["cv_fold_mae_scores"],
                },
                f,
                indent=2,
            )

        tuning_results["cv_results"][
            ["params", "mean_MAE", "std_MAE", "rank_test_score"]
        ].head(10).to_csv(output_dir / "gridsearch_top10.csv", index=False)
        tuning_results["tuned_importance"].to_csv(
            output_dir / "feature_importance_random_forest_tuned.csv", index=False
        )
        plot_feature_importance(
            tuning_results["tuned_importance"],
            title="Tuned Random Forest Feature Importance (Top 10)",
            out_path=output_dir / "feature_importance_random_forest_tuned.png",
        )
    else:
        print("\n[Step 8] Skipped GridSearchCV (--skip-tuning flag set)")

    # Save outputs
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    results["rf_importance"].to_csv(output_dir / "feature_importance_random_forest.csv", index=False)
    results["lr_importance"].to_csv(output_dir / "feature_importance_linear_regression.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    with open(output_dir / "cleaning_stats.json", "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=2)

    plot_feature_importance(
        results["rf_importance"],
        title="Random Forest Feature Importance (Top 10)",
        out_path=output_dir / "feature_importance_random_forest.png",
    )
    plot_feature_importance(
        results["lr_importance"].rename(columns={"abs_coefficient": "importance"}),
        title="Linear Regression |Coefficient| (Top 10)",
        out_path=output_dir / "feature_importance_linear_regression.png",
    )

    print("\n[Step 7] What we added in this iteration")
    print("  1) 5-fold cross-validation + GridSearchCV on Random Forest hyperparameters")
    print("  2) Country-normalized cost feature (cost_z) for fairer cross-country price signal")
    print("  3) Saved best params, CV scores, and tuned-model feature importance")

    print("\nSaved outputs to:", output_dir)
    print("All done.")


if __name__ == "__main__":
    main()
