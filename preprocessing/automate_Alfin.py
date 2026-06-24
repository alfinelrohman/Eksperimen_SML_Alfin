import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "telco_raw",
                        "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "churn_preprocessing")
OUT_FILE = os.path.join(OUT_DIR, "dataset_ready.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[load] Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank strings dengan NaN, lalu imputasi."""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fix: hindari ChainedAssignment, gunakan assignment langsung
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    print(f"[missing] TotalCharges nulls remaining: {df['TotalCharges'].isna().sum()}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop kolom yang tidak berguna untuk modelling."""
    df = df.drop(columns=["customerID"])
    print("[drop] Dropped: customerID")
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode kolom Yes/No dan biner ke 0/1."""
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    for col in binary_cols:
        if col == "gender":
            df[col] = df[col].map({"Male": 1, "Female": 0})
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"[encode binary] Columns: {binary_cols}")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode kolom kategorikal yang tersisa."""
    le = LabelEncoder()
    # Fix: gunakan 'str' bukan 'object' untuk pandas 2.x+
    cat_cols = df.select_dtypes(include="str").columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[encode label] Columns: {cat_cols}")
    return df


def scale_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """StandardScaler pada kolom numerik."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[scale] Columns: {num_cols}")
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] Dataset saved to {path} | Shape: {df.shape}")


def run_pipeline() -> pd.DataFrame:
    df = load_data(RAW_PATH)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = scale_numerical_columns(df)
    save_dataset(df, OUT_FILE)
    return df


if __name__ == "__main__":
    run_pipeline()