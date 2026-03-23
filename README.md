# House Price Prediction

This project uses machine learning to predict house prices based on features such as quality, living area, garage size, and year built.

## Objective

The goal of this project is to understand the factors affecting house prices and build a regression model to predict them.

## Dataset

The dataset contains information about houses, including:

- OverallQual (overall quality)
- GrLivArea (living area)
- GarageCars (garage capacity)
- TotalBsmtSF (basement size)
- FullBath (number of bathrooms)
- YearBuilt (year of construction)

Target variable:
- SalePrice

## Workflow

1. Data Loading
2. Data Exploration
3. Log Transformation
4. Outlier Removal
5. Feature Selection
6. Model Training (Linear Regression)
7. Model Evaluation

## Model

- Linear Regression

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- R² Score

## Key Insights

- OverallQual has the strongest impact on house price
- Larger houses tend to have higher prices
- Outlier removal improves model performance

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn