import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

OUT_FILE = os.path.join(os.path.dirname(__file__), "churn_preprocessing", "dataset_ready.csv")


def test_output_file_exists():
    assert os.path.exists(OUT_FILE), f"File tidak ditemukan: {OUT_FILE}"


def test_output_not_empty():
    df = pd.read_csv(OUT_FILE)
    assert len(df) > 0, "Dataset kosong!"


def test_no_missing_values():
    df = pd.read_csv(OUT_FILE)
    missing = df.isnull().sum().sum()
    assert missing == 0, f"Masih ada {missing} missing values!"


def test_no_customer_id_column():
    df = pd.read_csv(OUT_FILE)
    assert "customerID" not in df.columns, "Kolom customerID belum di-drop!"


def test_churn_column_binary():
    df = pd.read_csv(OUT_FILE)
    assert set(df["Churn"].unique()).issubset({0, 1}), "Kolom Churn bukan binary 0/1!"


def test_expected_columns_exist():
    df = pd.read_csv(OUT_FILE)
    expected = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    for col in expected:
        assert col in df.columns, f"Kolom {col} tidak ditemukan!"


def test_row_count():
    df = pd.read_csv(OUT_FILE)
    assert len(df) >= 7000, f"Row count terlalu sedikit: {len(df)}"