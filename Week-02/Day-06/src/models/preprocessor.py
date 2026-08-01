from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def get_preprocessor():

    numeric_features = [
        "passenger_count",
        "trip_distance",
        "pickup_hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "same_borough",
        "airport_pickup",
        "airport_dropoff"
    ]

    categorical_features = [
        "VendorID",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "pickup_borough",
        "dropoff_borough",
        "pickup_zone",
        "dropoff_zone",
        "store_and_fwd_flag"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


if __name__ == "__main__":
    preprocessor = get_preprocessor()
    print("Preprocessor Created Successfully")
    print(preprocessor)