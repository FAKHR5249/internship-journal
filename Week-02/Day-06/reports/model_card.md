# Model Card

# Model Information

**Model Name:** NYC Taxi ETA Prediction

**Version:** 1.0

**Algorithm:** Gradient Boosting Regressor

**Author:** Advanced AI/ML Internship

---

# Intended Use

This model predicts taxi trip duration before the trip begins.

It can be used for:

- ETA Prediction
- Ride Planning
- Taxi Dispatch
- Passenger Information

---

# Prediction Target

trip_duration_minutes

---

# Training Dataset

NYC Yellow Taxi Trip Records

Training Period

January 2023 – June 2023

Valid Records

566,642

---

# Features Used

## Time Features

- Pickup Hour
- Day of Week
- Month
- Weekend Indicator
- Peak Hour Indicator
- Night Indicator
- Hour Sin
- Hour Cos
- Day Sin
- Day Cos

## Geographic Features

- Pickup Borough
- Dropoff Borough
- Pickup Zone
- Dropoff Zone
- Same Borough
- Airport Pickup
- Airport Dropoff

## Other Features

- Passenger Count
- VendorID
- RateCodeID
- Payment Type
- Trip Distance

---

# Forbidden Features (Leakage)

The following features are NOT used because they are unavailable before the trip ends.

- Dropoff Datetime
- Fare Amount
- Tip Amount
- Total Amount
- Tolls Amount
- Airport Fee
- Improvement Surcharge
- Congestion Surcharge

---

# Data Cleaning

The following preprocessing steps were applied.

- Removed duplicate rows
- Removed negative durations
- Removed zero duration trips
- Removed invalid passenger counts
- Missing value handling
- Feature engineering
- Pipeline preprocessing

---

# Evaluation

| Metric | Value |
|---------|-------|
| MAE | 4.965 |
| RMSE | 38.993 |
| R² | 0.143 |

---

# Strengths

- Production-ready pipeline
- Automated preprocessing
- Feature engineering
- Hyperparameter tuning
- Good prediction accuracy
- Reproducible workflow

---

# Limitations

- Weather information not included
- Traffic conditions unavailable
- Holidays not modeled
- Unexpected road closures not considered

---

# Failure Cases

The model may perform worse for:

- Very long trips
- Airport traffic
- Heavy congestion
- Public holidays
- Extreme weather

---

# Ethical Considerations

- Model should assist drivers and passengers.
- Predictions should not be treated as guaranteed arrival times.
- Human judgment should always be considered.

---

# Deployment Requirements

Python 3.11+

Required Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

---

# Retraining Recommendation

Retrain the model every 3–6 months using newer NYC Taxi trip records.

---

# Monitoring

Monitor:

- MAE
- RMSE
- Prediction drift
- Data drift
- Missing values
- Model performance

---

# Champion Model

Gradient Boosting Regressor

This model achieved the best performance among all tested regression models and was selected as the production model.