# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

> Predict house prices using property features with advanced regression techniques and gradient boosting.

## 🎯 Objective

Predict house prices based on property features (size, bedrooms, location, etc.) using multiple regression models including Gradient Boosting. Compare models and build an interactive price estimator.

## 📚 What You'll Learn

- **Feature engineering** (creating new features from existing ones)
- **Handling categorical data** (one-hot encoding, label encoding)
- **Advanced regression** (XGBoost, LightGBM, Gradient Boosting)
- **Regularization** (Ridge, Lasso — when and why)
- **Residual analysis** (understanding where your model fails)
- **Model evaluation** (MAE, RMSE, R², adjusted R²)
- **Feature scaling** decisions (when it matters vs when it doesn't)

## 🧠 Concepts to Revise Before Starting

| Concept | Resource |
|---------|----------|
| Gradient Boosting | [StatQuest Video](https://www.youtube.com/watch?v=3CC4N4z3GJc) |
| XGBoost | [StatQuest XGBoost](https://www.youtube.com/watch?v=OtD8wVaFm6E) |
| Ridge & Lasso | [StatQuest Regularization](https://www.youtube.com/watch?v=Q81RR3yKn30) |
| One-Hot Encoding | [Encoding Guide](https://www.youtube.com/watch?v=9yl6-HEY7_s) |
| MAE vs RMSE | [Evaluation Metrics](https://www.youtube.com/watch?v=LbX4X71-TFI) |
| Feature Engineering | [Feature Engineering Guide](https://www.youtube.com/watch?v=68ABAU_V8qI) |

## 📁 Project Structure

```
house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── notebooks/
│   └── house_price_analysis.ipynb     ← Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 ← Load & clean dataset
│   ├── feature_engineering.py         ← Create & transform features
│   ├── model.py                       ← Train & evaluate models
│   └── visualize.py                   ← Plotting functions
├── data/                               ← Kaggle house price dataset
├── models/                             ← Saved trained models
├── results/                            ← Plots and metrics
└── app/
    └── streamlit_app.py                ← House price estimator UI
```

## 🚀 Step-by-Step Implementation Guide

### Step 1: Setup Environment
```bash
conda create -n house-price python=3.11 -y
conda activate house-price
pip install -r requirements.txt
```

### Step 2: Get the Dataset
- Download from [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Place `train.csv` and `test.csv` in `data/` folder

### Step 3: Data Loading & Cleaning (`src/data_loader.py`)
- Load the dataset
- Analyze missing values (which columns, how many)
- Strategy for missing data:
  - Numerical: fill with median
  - Categorical: fill with mode or "None"
- Identify and handle outliers

### Step 4: Feature Engineering (`src/feature_engineering.py`)
- Create new features:
  - Total square footage (basement + ground + upper)
  - Total bathrooms
  - House age (current year - year built)
  - Remodel age (current year - year remodeled)
  - Has pool / Has garage (binary)
- Encode categorical variables:
  - Ordinal encoding for quality features (Ex, Gd, TA, Fa, Po)
  - One-hot encoding for nominal features
- Feature selection:
  - Correlation analysis with target
  - Remove highly correlated features (multicollinearity)
- Scale numerical features (for linear models only)

### Step 5: Model Training (`src/model.py`)
- Split data (80/20)
- Train multiple models:
  1. Linear Regression (baseline)
  2. Ridge Regression
  3. Lasso Regression
  4. Random Forest Regressor
  5. XGBoost Regressor
  6. LightGBM Regressor
- Evaluate each: MAE, RMSE, R²
- Cross-validation (5-fold) for robust comparison
- Hyperparameter tuning for the best model

### Step 6: Analysis & Visualization (`src/visualize.py`)
- Actual vs Predicted scatter plot
- Residual plots (residuals vs predicted, residuals distribution)
- Feature importance (top 20 features)
- Price distribution histogram
- Correlation heatmap (top features)
- Model comparison chart
- Learning curves (train vs validation error)

### Step 7: Jupyter Notebook
- Full pipeline with explanations
- Feature engineering reasoning
- Model selection justification
- Error analysis — where does the model fail?

### Step 8: Streamlit App (`app/streamlit_app.py`)
- Property details input form (sliders, dropdowns)
- Price prediction display
- Comparable properties chart
- Feature importance for this prediction
- Clean real-estate themed UI

### Step 9: Docker
```bash
docker build -t house-price .
docker run -p 8501:8501 house-price
```

## 🎯 Extra Challenges (Bonus Learning)

- [ ] Implement stacking ensemble (combine multiple models)
- [ ] Log-transform the target variable (prices are often skewed)
- [ ] Add geospatial visualization if location data available
- [ ] Create a model interpretability report
- [ ] Compare performance with and without feature engineering

## 📊 Results

### Model Performance
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | - | - | - |
| Ridge | - | - | - |
| Lasso | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |
| LightGBM | - | - | - |

### Screenshots
<!-- Add screenshots of your Streamlit app and key plots here -->

## 🔗 Links

- **Dataset:** [House Prices - Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Internship:** DevelopersHub Corporation AI/ML Engineering

---
*Built as part of DevelopersHub Corporation AI/ML Engineering Internship*