# Leakage Audit Report

## Project

NYC Taxi ETA Prediction using Machine Learning

---

# What is Data Leakage?

Data leakage occurs when the model uses information that would not be available at prediction time. This can lead to unrealistically high model performance during training but poor performance in real-world deployment.

---

# Deployment-Safe Features

The following features are available before the trip starts and are safe to use.

- VendorID
- Passenger Count
- RateCodeID
- Pickup Location
- Dropoff Location
- Pickup Hour
- Day of Week
- Month
- Weekend Indicator
- Peak Hour Indicator
- Night Trip Indicator
- Pickup Borough
- Dropoff Borough
- Airport Pickup
- Airport Dropoff
- Same Borough

---

# Questionable Features

These features may or may not be available before the trip begins depending on the business scenario.

- Trip Distance
- Store and Forward Flag

For this project, Trip Distance was included because it can be estimated using navigation services before the trip starts.

---

# Forbidden Features

The following features were removed because they become available only after the trip has finished.

- tpep_dropoff_datetime
- fare_amount
- tip_amount
- total_amount
- tolls_amount
- airport_fee
- Airport_fee
- congestion_surcharge
- improvement_surcharge
- trip_duration_minutes (Target)

---

# Model A - Deployment Safe Model

Uses only information available before the trip starts.

Advantages:

- Realistic
- Suitable for production
- No leakage

---

# Model B - Oracle Model

Includes post-trip information.

Advantages:

- Higher accuracy

Disadvantages:

- Cannot be used in production
- Unfair evaluation
- Uses future information

---

# Leakage Prevention Techniques

- Removed all post-trip features.
- Target created before feature selection.
- Drop-off timestamp excluded from training.
- Preprocessing performed inside Pipeline.
- Train/Test split completed before model fitting.

---

# Conclusion

The final deployed model uses only deployment-safe features to ensure fair evaluation and reliable real-world performance.