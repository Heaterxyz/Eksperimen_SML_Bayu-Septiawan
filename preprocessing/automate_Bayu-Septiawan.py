"""
==============================================================
 automate_Bayu-Septiawan.py
 Otomatisasi Data Preprocessing - Dataset Telco Customer Churn
 Submission: Eksperimen SML - Dicoding
==============================================================

Skrip ini adalah konversi dari notebook eksperimen
(`Eksperimen_Bayu-Septiawan.ipynb`) menjadi pipeline otomatis.
Tahapan preprocessing SAMA PERSIS dengan notebook, yaitu:

 1. Memuat dataset (data loading)
 2. Menghapus kolom tidak relevan (customerID)
 3. Menangani missing values (TotalCharges -> numeric + median)
 4. Menghapus data duplikat
 5. Deteksi & penanganan outlier (capping IQR pada kolom numerik)
 6. Binning (pengelompokan kolom tenure menjadi tenure_group)
 7. Encoding data kategorikal (LabelEncoder + One-Hot Encoding)
 8. Standarisasi fitur numerik (StandardScaler)
 9. Menyimpan dataset hasil preprocessing (siap dilatih)

Cara pakai (dari root repo):
    python preprocessing/automate_Bayu-Septiawan.py \
        --raw namadataset_raw/Telco-Customer-Churn.csv \
        --out preprocessing/namadataset_preprocessing/telco_churn_preprocessing.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------
# Konstanta preprocessing
# ---------------------------------------------------------------
TARGET = "Churn"
DROP_COLS = ["customerID"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod", "tenure_group",
]

TENURE_BINS = [0, 12, 24, 48, 60, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61-72"]

# Map encoding fitur biner (dipakai agar konsisten antar-run)
BINARY_MAP = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}

RAW_PATH = "namadataset_raw/Telco-Customer-Churn.csv"
OUT_PATH = "preprocessing/namadataset_preprocessing/telco_churn_preprocessing.csv"


# ---------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan: {path}")
    df = pd.read_csv(path)
    print(f"[load] Dataset dimuat dari {path}: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ---------------------------------------------------------------
# 2-4. Pembersihan data (kolom tidak relevan, missing, duplikat)
# ---------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus kolom tidak relevan, memperbaiki missing values, dan duplikat."""
    df = df.copy()

    # 2. Buang kolom yang tidak relevan untuk modelling
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 3. Missing values: TotalCharges tersimpan sebagai string kosong " "
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    if n_missing > 0:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    print(f"[clean] Missing TotalCharges diimputasi dengan median: {n_missing} baris")

    # 4. Hapus data duplikat
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] Duplikat dihapus: {before - len(df)} baris")

    return df


# ---------------------------------------------------------------
# 5. Deteksi & penanganan outlier (capping metode IQR)
# ---------------------------------------------------------------
def handle_outliers(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Membatasi (capping) outlier kolom numerik memakai batas IQR 1.5x."""
    df = df.copy()
    cols = cols or NUMERIC_COLS
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outlier = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower, upper)
        print(f"[outlier] {col}: {n_outlier} outlier di-capping ke [{lower:.2f}, {upper:.2f}]")
    return df


# ---------------------------------------------------------------
# 6. Binning
# ---------------------------------------------------------------
def add_tenure_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Mengelompokkan lama berlangganan (tenure) menjadi kategori tenure_group."""
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    ).astype(str)
    print(f"[binning] tenure_group dibuat: {df['tenure_group'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------
# 7. Encoding data kategorikal
# ---------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label encoding target & fitur biner, lalu one-hot encoding fitur multi-kategori."""
    df = df.copy()

    # Target: No=0, Yes=1
    df[TARGET] = LabelEncoder().fit_transform(df[TARGET])

    # Fitur biner (Yes/No, Female/Male)
    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)

    # Fitur multi-kategori -> one-hot encoding
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True, dtype=int)

    print(f"[encode] Total fitur setelah encoding: {df.shape[1] - 1} fitur + 1 target")
    return df


# ---------------------------------------------------------------
# 8. Standarisasi fitur numerik
# ---------------------------------------------------------------
def scale_features(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Standardisasi kolom numerik dengan StandardScaler (mean=0, std=1)."""
    df = df.copy()
    cols = cols or NUMERIC_COLS
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    print(f"[scale] {cols} distandarisasi (mean=0, std=1)")
    return df


# ---------------------------------------------------------------
# 9. Menyimpan hasil
# ---------------------------------------------------------------
def save_data(df: pd.DataFrame, path: str) -> None:
    """Menyimpan dataset hasil preprocessing yang siap dilatih."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] Dataset preprocessing disimpan ke {path}: {df.shape[0]} baris, {df.shape[1]} kolom")


# ---------------------------------------------------------------
# Orchestrator: raw -> siap dilatih
# ---------------------------------------------------------------
def preprocess_data(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    """Menjalankan seluruh pipeline preprocessing dan mengembalikan DataFrame siap latih."""
    df = load_data(raw_path)
    df = clean_data(df)
    df = handle_outliers(df)
    df = add_tenure_bins(df)
    df = encode_features(df)
    df = scale_features(df)
    save_data(df, out_path)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset Telco Customer Churn"
    )
    parser.add_argument("--raw", default=RAW_PATH, help="Path dataset mentah (.csv)")
    parser.add_argument("--out", default=OUT_PATH, help="Path output dataset preprocessing (.csv)")
    args = parser.parse_args()

    df = preprocess_data(args.raw, args.out)

    # Validasi akhir
    if df.isna().any().any():
        print("[VALIDASI GAGAL] Masih ada missing values!", file=sys.stderr)
        return 1
    print(
        f"[VALIDASI OK] {df.shape[0]} baris x {df.shape[1]} kolom, "
        f"tanpa missing values, distribusi target: {df[TARGET].value_counts().to_dict()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
