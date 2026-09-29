"""
Cohort extraction for incident AFib risk prediction.

Requires credentialed local copies of:
  - MIMIC-IV-ECG (record_list.csv / machine_measurements.csv)
  - MIMIC-IV clinical tables (patients, admissions, diagnoses_icd, d_icd_diagnoses)

Defines:
  - Index ECG: earliest ECG per patient with no prior AFib diagnosis
  - Outcome: incident AFib = new AFib ICD-9/10 code within OUTCOME_WINDOW_DAYS
             after the index ECG date
  - Exclusion: patients with AFib diagnosed on/before the index ECG date

Adjust MIMIC_IV_ECG_DIR / MIMIC_IV_DIR to your local paths.
"""

import pandas as pd
from pathlib import Path

MIMIC_IV_ECG_DIR = Path("data/mimic-iv-ecg/1.0")
MIMIC_IV_DIR = Path("data/mimiciv/3.1")
OUTCOME_WINDOW_DAYS = 365 * 3  # 3-year incident AFib window; adjust per protocol

# ICD-9: 427.31 (atrial fibrillation); ICD-10: I48.0, I48.1, I48.2, I48.91 etc.
AFIB_ICD9 = {"42731"}
AFIB_ICD10_PREFIXES = ("I48",)


def is_afib_code(icd_code: str, icd_version: int) -> bool:
    if icd_version == 9:
        return icd_code in AFIB_ICD9
    return icd_code.startswith(AFIB_ICD10_PREFIXES)


def load_ecg_index() -> pd.DataFrame:
    """Load ECG record list with subject_id and ecg_time."""
    df = pd.read_csv(MIMIC_IV_ECG_DIR / "record_list.csv")
    df["ecg_time"] = pd.to_datetime(df["ecg_time"])
    return df[["subject_id", "study_id", "ecg_time", "path"]]


def load_diagnoses() -> pd.DataFrame:
    dx = pd.read_csv(MIMIC_IV_DIR / "hosp" / "diagnoses_icd.csv.gz")
    adm = pd.read_csv(
        MIMIC_IV_DIR / "hosp" / "admissions.csv.gz",
        usecols=["hadm_id", "admittime"],
    )
    dx = dx.merge(adm, on="hadm_id", how="left")
    dx["admittime"] = pd.to_datetime(dx["admittime"])
    dx["is_afib"] = dx.apply(
        lambda r: is_afib_code(str(r["icd_code"]), int(r["icd_version"])), axis=1
    )
    return dx


def build_cohort() -> pd.DataFrame:
    ecg = load_ecg_index()
    dx = load_diagnoses()
    afib_events = dx[dx["is_afib"]][["subject_id", "admittime"]].rename(
        columns={"admittime": "afib_time"}
    )

    # Earliest ECG per patient = index ECG
    index_ecg = ecg.sort_values("ecg_time").groupby("subject_id").first().reset_index()

    # Attach earliest AFib diagnosis time (if any) per patient
    first_afib = afib_events.groupby("subject_id")["afib_time"].min().reset_index()
    cohort = index_ecg.merge(first_afib, on="subject_id", how="left")

    # Exclude prevalent AFib (diagnosed on/before index ECG)
    prevalent_mask = cohort["afib_time"].notna() & (
        cohort["afib_time"] <= cohort["ecg_time"]
    )
    cohort = cohort[~prevalent_mask].copy()

    # Label incident AFib within the outcome window
    days_to_afib = (cohort["afib_time"] - cohort["ecg_time"]).dt.days
    cohort["incident_afib"] = (
        cohort["afib_time"].notna() & (days_to_afib <= OUTCOME_WINDOW_DAYS)
    ).astype(int)

    return cohort[
        ["subject_id", "study_id", "ecg_time", "path", "afib_time", "incident_afib"]
    ]


if __name__ == "__main__":
    cohort = build_cohort()
    print(f"Cohort size: {len(cohort)}")
    print(cohort["incident_afib"].value_counts())
    cohort.to_csv("cohort.csv", index=False)
    print("Saved cohort.csv")
