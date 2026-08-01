from pathlib import Path
import joblib
import pandas as pd


def test_model_exists():
    model = Path("models/champion_model.joblib")
    assert model.exists(), "Champion model not found."


def test_model_loads():
    model = joblib.load("models/champion_model.joblib")
    assert model is not None


def test_prediction_file_exists():
    prediction_file = Path("reports/predictions.csv")
    assert prediction_file.exists(), "Prediction file not found."


def test_prediction_columns():
    df = pd.read_csv("reports/predictions.csv")

    assert "Actual" in df.columns
    assert "Predicted" in df.columns


def test_prediction_not_empty():
    df = pd.read_csv("reports/predictions.csv")

    assert len(df) > 0


def test_prediction_no_missing():
    df = pd.read_csv("reports/predictions.csv")

    assert df["Predicted"].isnull().sum() == 0


def test_prediction_positive():
    df = pd.read_csv("reports/predictions.csv")

    assert (df["Predicted"] >= 0).all()