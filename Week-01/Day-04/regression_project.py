# =====================================================
# HOUSE PRICE PREDICTION - REGRESSION MODEL COMPARISON
# =====================================================

# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("Housing.csv")


print("\nFIRST 5 ROWS")
print(df.head())


print("\nDATASET SHAPE")
print(df.shape)


print("\nCOLUMNS")
print(df.columns)


print("\nDATA TYPES")
print(df.dtypes)


print("\nSTATISTICS")
print(df.describe())


# =====================================================
# 2. DATA CLEANING
# =====================================================


print("\nMISSING VALUES")
print(df.isnull().sum())


# Fill missing values

df.fillna(
    df.median(numeric_only=True),
    inplace=True
)


# Remove duplicates

df.drop_duplicates(inplace=True)


print("\nDuplicates Removed")


# =====================================================
# 3. FEATURES AND TARGET
# =====================================================


X = df.drop("price", axis=1)

y = df["price"]


# Encode categorical columns

X = pd.get_dummies(
    X,
    drop_first=True
)


# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# 5. EVALUATION FUNCTION
# =====================================================


results = []


def evaluate_model(model, name):

    print("\nTraining:", name)


    # Training

    model.fit(
        X_train,
        y_train
    )


    # Prediction

    prediction = model.predict(
        X_test
    )


    # Metrics

    mae = mean_absolute_error(
        y_test,
        prediction
    )


    mse = mean_squared_error(
        y_test,
        prediction
    )


    rmse = np.sqrt(mse)


    r2 = r2_score(
        y_test,
        prediction
    )


    mape = mean_absolute_percentage_error(
        y_test,
        prediction
    )


    # Adjusted R2

    n = X_test.shape[0]

    p = X_test.shape[1]


    adjusted_r2 = (
        1 -
        ((1-r2)*(n-1))
        /
        (n-p-1)
    )


    # Repeated Cross Validation

    cv = RepeatedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=42
    )


    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2"
    )


    cv_mean = cv_score.mean()


    # Residual Analysis

    residuals = y_test - prediction


    plt.figure(figsize=(7,5))

    plt.scatter(
        prediction,
        residuals
    )

    plt.axhline(
        0,
        color="red"
    )

    plt.xlabel(
        "Predicted Price"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        name + " Residual Plot"
    )

    plt.show()



    results.append(
        [
            name,
            mae,
            mse,
            rmse,
            r2,
            adjusted_r2,
            mape,
            cv_mean
        ]
    )



# =====================================================
# 6. MODELS
# =====================================================


# Baseline Model

evaluate_model(
    LinearRegression(),
    "Linear Regression (Baseline)"
)



# Ridge

evaluate_model(
    Ridge(alpha=1),
    "Ridge Regression"
)



# Lasso

evaluate_model(
    Lasso(alpha=0.1),
    "Lasso Regression"
)



# Elastic Net

evaluate_model(
    ElasticNet(
        alpha=0.1,
        l1_ratio=0.5
    ),
    "Elastic Net"
)



# Polynomial Regression

poly_model = Pipeline(
    [
        (
            "poly",
            PolynomialFeatures(
                degree=2
            )
        ),

        (
            "linear",
            LinearRegression()
        )
    ]
)


evaluate_model(
    poly_model,
    "Polynomial Regression"
)



# Decision Tree

evaluate_model(
    DecisionTreeRegressor(
        random_state=42
    ),
    "Decision Tree Regressor"
)



# Random Forest

evaluate_model(
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Random Forest Regressor"
)



# Gradient Boosting

evaluate_model(
    GradientBoostingRegressor(
        random_state=42
    ),
    "Gradient Boosting Regressor"
)



# XGBoost

evaluate_model(
    XGBRegressor(
        random_state=42,
        objective="reg:squarederror"
    ),
    "XGBoost Regressor"
)



# =====================================================
# 7. MODEL COMPARISON TABLE
# =====================================================


comparison = pd.DataFrame(

    results,

    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score",
        "Adjusted R2",
        "MAPE",
        "Cross Validation R2"
    ]

)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison)



# =====================================================
# 8. BEST MODEL
# =====================================================


best_model = comparison.sort_values(
    by="R2 Score",
    ascending=False
)


print("\n==============================")
print("BEST MODEL")
print("==============================")


print(
    best_model.iloc[0]
)


# Save Results

comparison.to_csv(
    "Regression_Model_Comparison.csv",
    index=False
)


print(
    "\nResult file saved successfully"
)