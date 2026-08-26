# Task 2: Predict Restaurant Ratings (Regression)

## Overview
Build a supervised regression model to predict **Aggregate rating** (1.8–4.9 on rated restaurants) using restaurant metadata such as price, votes, cuisine, city, and service flags.

This corresponds to **Internship Task 1** in the Cognifyz guide.

## Problem type
- **Supervised learning**
- **Regression**
- Target: `Aggregate rating`

## How to run
From project root:
```bash
python TASK2/task2_predict_restaurant_ratings.py
```

Or from inside `TASK2/`:
```bash
python task2_predict_restaurant_ratings.py
```

Install dependencies (if needed):
```bash
pip install -r TASK2/requirements.txt
```

## Key dataset trap (must mention in writeup)
Rows with `Rating text == "Not rated"` have `Aggregate rating = 0` **by definition** — not because the restaurant is bad.

If you train on those rows, the model learns fake low ratings and performance collapses.

**Our fix:** remove all `"Not rated"` rows before modeling.

## Feature engineering
| Feature | Handling | Why |
|--------|----------|-----|
| `Votes` | `log1p(Votes)` | Votes are heavily skewed; log reduces outlier dominance |
| `Average Cost for two` | Raw + `cost_z` (z-score within country) | Currency scales differ by country |
| `Table booking`, `Online delivery` | Yes/No → 1/0 | Natural binary encoding |
| `Cuisines` | Primary cuisine + cuisine count | Avoids exploding one-hot dimensionality |
| `City`, `Country Code` | Category codes | Compact encoding for high-cardinality fields |
| `Price range`, `Average Cost for two` | Numeric | Direct price signals |

## Models trained
1. **Linear Regression** (baseline, with scaling on numeric features)
2. **Random Forest Regressor** (baseline: 300 trees)
3. **Random Forest (tuned)** — 5-fold CV + GridSearchCV on `max_depth`, `n_estimators`, `min_samples_leaf`, `min_samples_split`

Train/test split: **80/20**, `random_state=42`

Fast run (skip tuning):
```bash
python task2_predict_restaurant_ratings.py --skip-tuning
```

## Results (example run)
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 0.283 | 0.380 | 0.533 |
| Random Forest (baseline) | 0.257 | 0.346 | 0.612 |
| **Random Forest (tuned, 5-fold CV)** | **0.252** | **0.338** | **0.631** |

**Best hyperparameters (GridSearchCV):**
```json
{
  "n_estimators": 300,
  "max_depth": 15,
  "min_samples_leaf": 5,
  "min_samples_split": 5
}
```

**Cross-validation stability:**
- 5-fold CV MAE: **0.260** (std ≈ 0.004)
- Low std across folds = stable model, not lucky on one split

Interpretation:
- **R² ≈ 0.63** (tuned RF) = strong for restaurant rating prediction
- Tuning improved test R² by ~+0.019 vs baseline RF
- On a 0–5 scale, MAE ≈ 0.25 means predictions are typically within ~¼ star

## Top drivers of rating (Random Forest)
1. `log_votes` — popularity/review volume dominates
2. `Average Cost for two`
3. `Primary Cuisine`
4. `Country Code` / `City`
5. Service flags (`Online delivery`, `Table booking`) matter less

This matches intuition: **social proof (votes) matters more than cuisine label alone**.

## Outputs
Saved in `TASK2/output/`:
- `model_metrics.csv`
- `feature_importance_random_forest.csv`
- `feature_importance_linear_regression.csv`
- `feature_importance_random_forest.png`
- `feature_importance_linear_regression.png`
- `test_predictions.csv`
- `feature_importance_random_forest_tuned.csv` + `.png`
- `best_params.json`
- `gridsearch_top10.csv`

## Suggested improvements (next iteration)
1. ~~Cross-validation + GridSearchCV~~ ✅ Done
2. ~~Country-normalized cost (`cost_z`)~~ ✅ Done
3. **Target/frequency encoding** for `City` instead of raw category codes
4. Optional: add delivery flags (`Is delivering now`, `Switch to order menu`) as features
5. Optional: try **XGBoost** for a further boost

## Portfolio writeup checklist
Include these sections so it reads like a real internship project:
1. **Business question** — Why predict ratings? (quality benchmarking, ranking support)
2. **Data understanding** — 9,551 rows, 6 countries, unrated-row trap explained
3. **Data cleaning decisions** — what you removed and why
4. **Feature design** — encoding choices + log transform justification
5. **Modeling approach** — baseline vs advanced model
6. **Evaluation** — MAE/RMSE/R² with interpretation (not just numbers)
7. **Feature importance** — what drives ratings and does it make sense?
8. **Limitations** — no text reviews, currency not normalized, city encoding is basic
9. **Next steps** — CV, tuning, better encodings

## Tech stack
- Python
- pandas, numpy
- scikit-learn (LinearRegression, RandomForestRegressor, Pipeline, ColumnTransformer)
- matplotlib
