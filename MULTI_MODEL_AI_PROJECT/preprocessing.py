"""
Preprocessing for:
  1. ECG waveforms (WFDB .dat/.hea) -> cleaned, resampled numpy arrays
  2. Tabular clinical features -> imputed, scaled feature matrix

Requires: wfdb, scipy, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET_FS = 250          # target sampling rate (Hz)
TARGET_LEN_SEC = 10       # seconds per record
LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0


# ---------------------------------------------------------------------------
# ECG preprocessing
# ---------------------------------------------------------------------------
def bandpass_filter(signal: np.ndarray, fs: int, low=LOWCUT_HZ, high=HIGHCUT_HZ, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def load_and_clean_ecg(record_path: str) -> np.ndarray:
    """Load a WFDB record, bandpass filter, resample, normalize. Returns (leads, time)."""
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal.astype(np.float32)  # shape (n_samples, n_leads)
    fs = record.fs

    sig = bandpass_filter(sig, fs)

    target_samples = TARGET_FS * TARGET_LEN_SEC
    if fs != TARGET_FS:
        sig = resample(sig, target_samples, axis=0)
    elif sig.shape[0] != target_samples:
        sig = resample(sig, target_samples, axis=0)

    # Per-lead z-score normalization
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True) + 1e-8
    sig = (sig - mean) / std

    return sig.T  # (n_leads, n_samples) — CNN-friendly


def quality_screen(sig: np.ndarray, flat_thresh=1e-3, clip_thresh=0.999) -> bool:
    """Return True if signal passes basic quality checks (not flatline/clipped)."""
    if np.std(sig) < flat_thresh:
        return False
    frac_extreme = np.mean(np.abs(sig) > 5.0)  # >5 SD after z-score = likely artifact
    if frac_extreme > (1 - clip_thresh):
        return False
    return True


# ---------------------------------------------------------------------------
# Tabular preprocessing
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age", "heart_rate", "sbp", "dbp", "bmi",
    "creatinine", "potassium", "sodium", "hemoglobin", "bnp",
]
CATEGORICAL_FEATURES = [
    "sex", "hypertension", "diabetes", "heart_failure",
    "prior_stroke", "copd", "ckd",
]


def build_tabular_features(clinical_df: pd.DataFrame):
    """Impute + scale numeric, one-hot categorical. Returns (X, feature_names, scaler, imputer)."""
    numeric = clinical_df[NUMERIC_FEATURES]
    categorical = clinical_df[CATEGORICAL_FEATURES]

    num_imputer = SimpleImputer(strategy="median")
    numeric_imputed = num_imputer.fit_transform(numeric)

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_imputed)

    categorical_filled = categorical.fillna("missing").astype(str)
    categorical_encoded = pd.get_dummies(categorical_filled, drop_first=False)

    X = np.hstack([numeric_scaled, categorical_encoded.values]).astype(np.float32)
    feature_names = NUMERIC_FEATURES + list(categorical_encoded.columns)

    return X, feature_names, scaler, num_imputer


if __name__ == "__main__":
    cohort = pd.read_csv("cohort.csv")
    print(f"Loaded cohort: {len(cohort)} rows")
    # Example: preprocess a single ECG (path column from cohort_extraction.py)
    # sig = load_and_clean_ecg(cohort.iloc[0]["path"])
    # print("ECG shape:", sig.shape, "Quality OK:", quality_screen(sig))
