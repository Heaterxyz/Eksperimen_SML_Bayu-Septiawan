# Eksperimen_SML_Bayu-Septiawan

Repository eksperimen & preprocessing dataset **Telco Customer Churn (IBM)** untuk submission kelas MLOps.

## Struktur Repository

```
Eksperimen_SML_Bayu-Septiawan/
├── .github/workflows/preprocessing.yml      # GitHub Actions: preprocessing otomatis (Advanced)
├── namadataset_raw/
│   └── Telco-Customer-Churn.csv             # Dataset mentah (7.043 baris x 21 kolom)
├── preprocessing/
│   ├── Eksperimen_Bayu-Septiawan.ipynb      # Notebook eksperimen (EDA + preprocessing)
│   ├── automate_Bayu-Septiawan.py           # Skrip otomatisasi preprocessing (Skilled)
│   └── namadataset_preprocessing/
│       └── telco_churn_preprocessing.csv    # Dataset siap latih (7.021 baris x 35 kolom)
└── requirements.txt
```

## Alur Eksperimen

1. **Perkenalan Dataset** — Telco Customer Churn dari repositori publik IBM.
2. **Import Library** — pandas, numpy, matplotlib, seaborn, scikit-learn.
3. **Memuat Dataset** — pembacaan CSV + pemeriksaan struktur, missing values, duplikat.
4. **EDA** — distribusi target, histogram, boxplot, heatmap korelasi, analisis fitur kategorikal.
5. **Preprocessing** — drop `customerID`, imputasi `TotalCharges`, hapus duplikat, capping outlier IQR, binning `tenure_group`, encoding, standarisasi.

## Menjalankan Preprocessing Otomatis

```bash
pip install -r requirements.txt
python preprocessing/automate_Bayu-Septiawan.py \
  --raw namadataset_raw/Telco-Customer-Churn.csv \
  --out preprocessing/namadataset_preprocessing/telco_churn_preprocessing.csv
```

## GitHub Actions

Workflow `Preprocessing Dataset Telco Churn` terpicu saat:
- push ke branch `main` yang mengubah `namadataset_raw/**` atau skrip otomatisasi, dan
- manual melalui `workflow_dispatch`.

Workflow menjalankan skrip otomatisasi, mengunggah artefak dataset, lalu **commit & push dataset preprocessing terbaru** ke repository ini (pesan commit diberi `[skip ci]` agar tidak memicu loop).
