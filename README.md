# cognifyz_ml_internship
Geospatial analysis, rating regression, and cuisine classification on a multi-country restaurant dataset

Cognifyz ML Internship — Restaurant Analytics

This repository contains my Cognifyz Machine Learning internship work on a multi-country restaurant dataset (India, US, Brazil, Australia, Philippines, Singapore). The goal was to turn restaurant metadata into useful insights and predictive models.

The project covers three tasks:

Location-based analysis
Cleaned invalid coordinates, mapped restaurant distribution, identified density hotspots, and compared rating/price patterns across countries.

Predict restaurant ratings (regression)
Built models to predict aggregate rating after removing unrated (rating = 0) rows. Used Linear Regression as a baseline and a tuned Random Forest (R² = 0.631, MAE = 0.252).

Cuisine classification
Predicted a restaurant’s primary cuisine from structured features. Handled class imbalance by grouping rare cuisines into Other, then evaluated with macro-F1 instead of accuracy.

Tech stack: Python, pandas, NumPy, scikit-learn, Plotly, matplotlib

Skills shown: data cleaning, feature engineering, geospatial EDA, regression, multi-class classification, imbalance handling, model evaluation, and hyperparameter tuning.


