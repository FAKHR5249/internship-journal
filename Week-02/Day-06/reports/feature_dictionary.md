# Feature Dictionary

## Target Variable

| Feature | Description |
|----------|-------------|
| trip_duration_minutes | Taxi trip duration in minutes |

---

# Original Features

| Feature | Description |
|----------|-------------|
| VendorID | Taxi vendor identifier |
| passenger_count | Number of passengers |
| trip_distance | Distance of trip |
| RatecodeID | Taxi rate code |
| PULocationID | Pickup location ID |
| DOLocationID | Dropoff location ID |
| payment_type | Payment method |
| store_and_fwd_flag | Store and forward indicator |
| tpep_pickup_datetime | Pickup timestamp |

---

# Engineered Features

| Feature | Description |
|----------|-------------|
| pickup_hour | Hour of pickup |
| day_of_week | Day of week |
| month | Pickup month |
| is_weekend | Weekend indicator |
| is_peak_hour | Peak traffic hour indicator |
| is_night | Night trip indicator |
| hour_sin | Cyclical encoding of pickup hour |
| hour_cos | Cyclical encoding of pickup hour |
| day_sin | Cyclical encoding of day of week |
| day_cos | Cyclical encoding of day of week |
| pickup_borough | Pickup borough |
| dropoff_borough | Dropoff borough |
| pickup_zone | Pickup taxi zone |
| dropoff_zone | Dropoff taxi zone |
| same_borough | Pickup and dropoff in same borough |
| airport_pickup | Airport pickup indicator |
| airport_dropoff | Airport dropoff indicator |

---

# Removed Features (Leakage)

| Feature | Reason |
|----------|--------|
| tpep_dropoff_datetime | Available after trip ends |
| fare_amount | Generated after trip |
| tip_amount | Generated after payment |
| total_amount | Generated after payment |
| tolls_amount | Post-trip value |
| airport_fee | Post-trip fee |
| Airport_fee | Duplicate airport fee |
| congestion_surcharge | Post-trip surcharge |
| improvement_surcharge | Post-trip surcharge |

---

# Summary

Total Features Used:

- Original Features
- Engineered Features

Total Engineered Features:

17

Leakage Features Removed:

9