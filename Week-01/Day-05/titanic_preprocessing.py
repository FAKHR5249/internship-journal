# ==========================================
# TITANIC DATA CLEANING AND PREPROCESSING
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
    "Titanic-Dataset.csv"
)


print("Original Data")
print(df.head())

print("\nShape:")
print(df.shape)



# ==========================================
# DATA QUALITY CHECK
# ==========================================


print("\nMissing Values")

print(df.isnull().sum())



print("\nDuplicate Records")

print(
    df.duplicated().sum()
)



# Remove duplicates

df.drop_duplicates(
    inplace=True
)



# ==========================================
# MISSING VALUE TREATMENT
# ==========================================


# Age Median

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)



# Embarked Mode

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)



# Cabin Remove

df.drop(
    "Cabin",
    axis=1,
    inplace=True
)



# ==========================================
# FEATURE ENGINEERING
# ==========================================


# Family Size

df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)



# Alone Feature

df["IsAlone"] = np.where(
    df["FamilySize"] == 1,
    1,
    0
)



# Title Extraction

df["Title"] = (
    df["Name"]
    .str.extract(
        r' ([A-Za-z]+)\.',
        expand=False
    )
)



# Rare Titles

rare = [
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer"
]


df["Title"] = df["Title"].replace(
    rare,
    "Rare"
)



# Name Length

df["Name_Length"] = (
    df["Name"]
    .apply(len)
)



# Fare per Person

df["Fare_Per_Person"] = (

df["Fare"] /

df["FamilySize"]

)



# Drop unnecessary columns

df.drop(
[
"PassengerId",
"Name",
"Ticket"
],
axis=1,
inplace=True
)



# ==========================================
# DATA SPLITTING
# ==========================================


X = df.drop(
"Survived",
axis=1
)


y = df["Survived"]



X_train, X_test, y_train, y_test = train_test_split(

X,

y,

test_size=0.2,

random_state=42,

stratify=y

)



# ==========================================
# PIPELINE
# ==========================================


numeric_features = [

"Age",
"SibSp",
"Parch",
"Fare",
"FamilySize",
"Name_Length",
"Fare_Per_Person"

]


categorical_features = [

"Sex",
"Embarked",
"Title"

]



numeric_pipeline = Pipeline(

steps=[

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

steps=[

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



print("\nTitanic Preprocessing Completed")

print(
"Training Shape:",
X_train_processed.shape
)

print(
"Testing Shape:",
X_test_processed.shape
)