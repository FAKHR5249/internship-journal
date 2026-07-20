# ==========================================
# HOUSE PRICE DATA PREPROCESSING
# ==========================================


import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import (
StandardScaler,
OneHotEncoder
)


from sklearn.impute import SimpleImputer


from sklearn.compose import ColumnTransformer


from sklearn.pipeline import Pipeline



# Load Dataset


df = pd.read_csv(
"Housing.csv"
)



print(df.head())


print(df.info())



# ==========================================
# DATA QUALITY
# ==========================================


print("\nMissing Values")

print(
df.isnull().sum()
)



print("\nDuplicates")

print(
df.duplicated().sum()
)



df.drop_duplicates(
inplace=True
)



# ==========================================
# OUTLIER TREATMENT
# ==========================================


numeric_cols = df.select_dtypes(
include=np.number
).columns



for col in numeric_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1


    lower = Q1-1.5*IQR

    upper = Q3+1.5*IQR


    df[col] = np.where(

        df[col] < lower,
        lower,

        np.where(
            df[col] > upper,
            upper,
            df[col]
        )
    )



# ==========================================
# FEATURE ENGINEERING
# ==========================================



if "Area" in df.columns:

    if "price" in df.columns:

        df["price_Per_Area"] = (

        df["price"] /
        df["Area"]

        )



# Log Transformation

for col in numeric_cols:

    if (df[col] >= 0).all():

        df[col+"_log"] = np.log1p(
            df[col]
        )



# ==========================================
# SPLIT DATA
# ==========================================



target = "price"


X = df.drop(
target,
axis=1
)


y = df[target]



X_train, X_test, y_train, y_test = train_test_split(

X,

y,

test_size=0.2,

random_state=42

)



# ==========================================
# PIPELINE
# ==========================================



numeric_features = X.select_dtypes(
include=np.number
).columns



categorical_features = X.select_dtypes(
exclude=np.number
).columns



numeric_pipeline = Pipeline(

[

(
"imputer",
SimpleImputer(strategy="median")
),

(
"scaler",
StandardScaler()
)

]

)



categorical_pipeline = Pipeline(

[

(
"imputer",
SimpleImputer(strategy="most_frequent")
),

(
"encoder",
OneHotEncoder(handle_unknown="ignore")
)

]

)



preprocessor = ColumnTransformer(

[

(
"num",
numeric_pipeline,
numeric_features
),

(
"cat",
categorical_pipeline,
categorical_features
)

]

)



X_train_processed = preprocessor.fit_transform(
X_train
)


X_test_processed = preprocessor.transform(
X_test
)



print("\nHouse Price Preprocessing Completed")

print(
X_train_processed.shape
)

print(
X_test_processed.shape
)