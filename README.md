# Boston House Price Prediction

Overview

This project aims to predict house prices in Boston using machine learning techniques. The dataset used is the Boston Housing Dataset, which contains various features influencing house prices, such as crime rate, average number of rooms, and property tax.

Dataset

# The dataset consists of the following features:

CRIM: Per capita crime rate by town.

ZN: Proportion of residential land zoned for large lots.

INDUS: Proportion of non-retail business acres per town.

CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).

NOX: Nitrogen oxide concentration (parts per 10 million).

RM: Average number of rooms per dwelling.

AGE: Proportion of owner-occupied units built before 1940.

DIS: Weighted distances to employment centers.

RAD: Accessibility index to radial highways.

TAX: Property tax rate per $10,000.

PTRATIO: Pupil-teacher ratio by town.

B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents.

LSTAT: Percentage of lower status population.

MEDV: Median value of owner-occupied homes (in $1000s) (Target Variable).

# Project Workflow

Data Preprocessing

Handling missing values.

Feature scaling and normalization.

Splitting the dataset into training and testing sets.

Exploratory Data Analysis (EDA)

Visualizing feature distributions.

Correlation analysis.

Identifying key factors affecting house prices.

Model Training

Training various regression models such as:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

# Model Evaluation

Using metrics such as:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared Score (R²)

Model Deployment (Optional)

Deploying the best-performing model using Flask or FastAPI.

Requirements

To run this project, install the following dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn flask

Usage

# Run the script to train the model and make predictions:

python main.py

# Results

The model achieved an R² score of ~0.85, indicating a good fit.

Feature importance analysis revealed that RM, LSTAT, and PTRATIO were the most influential factors.

Future Improvements

Hyperparameter tuning for better accuracy.

Incorporate deep learning models (e.g., Neural Networks).

Deploy the model as a web service for real-time predictions.

