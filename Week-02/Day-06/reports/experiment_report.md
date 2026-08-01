# Experiment Report

# NYC Taxi ETA Prediction

---

## Business Problem

The objective of this project is to predict the expected duration of NYC taxi trips before the trip begins. Accurate trip duration estimation helps passengers plan their travel while improving taxi dispatch efficiency.

---

# Dataset

Dataset:
NYC Yellow Taxi Trip Records

Months Used:

- January 2023
- February 2023
- March 2023
- April 2023
- May 2023
- June 2023

Initial Records:
600,000

Valid Records:
566,642

Target Variable:
trip_duration_minutes

---

# Data Cleaning

Performed:

- Removed duplicate rows
- Removed negative trip durations
- Removed zero duration trips
- Removed invalid passenger counts
- Removed missing records
- Created trip duration target

---

# Feature Engineering

Time Features

- Pickup Hour
- Day of Week
- Month
- Weekend Indicator
- Peak Hour Indicator
- Night Indicator
- Hour Sin Encoding
- Hour Cos Encoding
- Day Sin Encoding
- Day Cos Encoding

Geographic Features

- Pickup Borough
- Dropoff Borough
- Pickup Zone
- Dropoff Zone
- Same Borough
- Airport Pickup
- Airport Dropoff

---

# Models Trained

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting

---

# Hyperparameter Tuning

Algorithm Used:

Grid Search

Best Model:

Gradient Boosting Regressor

Best Parameters

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 5

---

# Final Evaluation

Evaluation Metrics

MAE:
4.965

RMSE:
38.993

R²:
0.143

---

# Prediction

Predictions were generated successfully using the tuned model.

Prediction file:

reports/predictions.csv

---

# Visualization

Generated Visualizations

- Actual vs Predicted
- Residual Distribution
- Prediction Error Plot

---

# Conclusion

Gradient Boosting achieved the best performance among all evaluated models.

The complete workflow includes:

- Data Validation
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Evaluation
- Prediction
- Visualization

The project follows a production-ready machine learning workflow.