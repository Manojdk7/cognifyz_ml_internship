# Task 3: Cuisine Classification

## Overview
This project builds a **multi-class classification model** to predict a restaurant's primary cuisine using structured restaurant features such as city, country, price range, cost, votes, table booking, online delivery, and aggregate rating.

This corresponds to **Task 3** in the Cognifyz ML Internship.

## Problem Type
- Supervised learning
- Multi-class classification
- Target: `Cuisine Target`

## Dataset-specific challenge
The raw `Cuisines` column often contains multiple cuisines in one row, such as:

`North Indian, Chinese`

Since classification needs **one label per row**, this project uses the **first cuisine token** as the target label:
- `North Indian, Chinese` -> `North Indian`
- `Cafe, Bakery` -> `Cafe`

This is a simple and standard approximation of the restaurant's primary cuisine identity.

## Class imbalance handling
The dataset contains **120 unique primary cuisines**, but many are extremely rare.

With a threshold of **30 samples**:
- kept classes: **27**
- rows collapsed into `Other`: **604**
- final target classes: **28**

Why this is not cheating:
- Classes with only a few samples do not provide enough information for the model to learn meaningful patterns.
- Collapsing rare labels into `Other` is standard practice in imbalanced multi-class problems.
- It improves generalization and makes reported metrics more trustworthy.

## Features used
- `City`
- `Country Code`
- `Price range`
- `Average Cost for two`
- `cost_z` (cost normalized within each country)
- `Table booking`
- `Online delivery`
- `log_votes`
- `Aggregate rating`

## Encoding choices
- `Table booking`, `Online delivery`: Yes/No -> 1/0
- `City`, `Country Code`: category codes
- `Votes`: `log1p(Votes)` because vote counts are highly skewed
- `Average Cost for two`: kept raw and also normalized within country as `cost_z`

## Train/test split
Used **80/20 split with stratification**:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Stratification matters because it preserves the class distribution in both train and test sets, reducing the chance that small cuisines disappear from one split.

## Models trained
1. **Logistic Regression** (baseline)
2. **Random Forest Classifier** with `class_weight=\"balanced\"`

Why balancing matters:
- Without class balancing, the model would over-focus on dominant cuisines like `North Indian`
- `class_weight=\"balanced\"` gives more learning importance to minority classes

## Results
| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Logistic Regression | **0.348** | 0.069 | **0.206** |
| Random Forest (balanced) | 0.157 | **0.151** | 0.160 |

## How to interpret these results
At first glance, Logistic Regression looks better because its **accuracy is higher**.

But for this dataset, **accuracy is misleading** because the classes are highly imbalanced. A model can get decent accuracy by mostly predicting dominant cuisines.

The better headline metric is **Macro-F1**:
- Logistic Regression: **0.069**
- Random Forest: **0.151**

That means the balanced Random Forest does a better job across **all cuisine classes**, including minority ones.

## Confusion matrix insight
The confusion matrix focuses on the most common cuisine classes to stay readable.

Some confusions are intuitive:
- `Street Food` <-> `Mithai`
- `Chinese` -> `North Indian`
- `Bakery` -> `Mithai`
- `Fast Food` -> `Bakery`

These errors are partly understandable because cuisines can overlap in:
- customer base
- pricing
- service format
- location patterns

So not every confusion is a pure model failure; some reflect real similarity between restaurant types.

## Outputs
Saved in `TASK3/output/`:
- `model_metrics.csv`
- `classification_report_logistic_regression.csv`
- `classification_report_random_forest.csv`
- `feature_importance_random_forest.csv`
- `confusion_matrix_top_classes.csv`
- `confusion_matrix_top_classes.png`
- `top_confusion_pairs.csv`
- `test_predictions.csv`
- `target_label_mapping.csv`
- `preparation_stats.json`

## Common mistakes to avoid
1. Reporting only **accuracy** on an imbalanced multi-class dataset
2. Not collapsing rare cuisines, leading to noisy and unstable class metrics
3. Treating the full comma-separated `Cuisines` field as a single raw label
4. Ignoring stratification during train/test split
5. Claiming low accuracy means the model is useless without checking Macro-F1

## Suggested improvements
1. Add more predictive features such as `Locality`, `Rating text`, or textual cuisine embeddings
2. Tune Random Forest hyperparameters or try XGBoost / LightGBM
3. Test SMOTE carefully, only after validating that synthetic samples do not distort minority cuisines

## What to include in your internship writeup
To make this read like a genuine ML project, include:
1. Business problem: why cuisine prediction can help restaurant categorization/search
2. Target engineering: why first cuisine was chosen
3. Imbalance handling: why rare classes were grouped into `Other`
4. Modeling: Logistic Regression baseline vs balanced Random Forest
5. Evaluation: explain why **Macro-F1** is the main metric, not accuracy
6. Confusion analysis: which cuisines are confused and whether that makes sense
7. Limitations: structured features only, no menu/review text
8. Future work: richer features, better imbalance handling, stronger models

## Run
From the project root:

```bash
python TASK3/task3_cuisine_classification.py
```

