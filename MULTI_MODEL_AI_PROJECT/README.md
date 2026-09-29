# Incident AFib Risk Prediction — Multimodal (ECG + EHR) Pipeline

Starter codebase matching the paper draft "A Multimodal Deep Learning Framework for
Predicting Incident Atrial Fibrillation Risk by Fusing 12-Lead ECG Waveforms with
Structured Clinical Data."

## 1. Dataset — what to use and how to get it

You need **two linked PhysioNet resources**, both under the same credentialed account:

| Resource | What it gives you | Link |
|---|---|---|
| **MIMIC-IV-ECG v1.0** | ~800,000 12-lead ECG waveform records (10s, 500Hz, WFDB format) linked to `subject_id` | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| **MIMIC-IV v3.x (or v2.2)** | Structured EHR: demographics, admissions, diagnoses (ICD), labs, vitals, comorbidities | https://physionet.org/content/mimiciv/ |

### Steps to get access
1. Create a PhysioNet account: https://physionet.org/register/
2. Complete CITI "Data or Specimens Only Research" training (required for credentialing) — https://physionet.org/about/citi-course/
3. Apply for **PhysioNet Credentialed Health Data User** status (takes a few days for review).
4. Once credentialed, sign the individual **Data Use Agreement (DUA)** on each dataset's page (MIMIC-IV and MIMIC-IV-ECG separately).
5. Download via the PhysioNet page directly, or (recommended for large files) via `wget`:
   ```bash
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-iv-ecg/1.0/
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimiciv/3.1/
   ```
6. Both datasets share `subject_id` — this is the join key between the ECG waveform index (`machine_measurements.csv` / `record_list.csv` in MIMIC-IV-ECG) and the clinical tables (`patients`, `admissions`, `diagnoses_icd`, `labevents`, `chartevents`) in MIMIC-IV.

### Why this dataset (not PTB-XL, Chapman-Shaoxing, etc.)
Those ECG-only datasets (PTB-XL, Chapman-Shaoxing/Ningbo, Georgia 12-Lead) are good for
ECG-only baselines or pretraining the ECG encoder, but **only MIMIC-IV-ECG + MIMIC-IV
gives you the linked structured EHR data** needed for genuine multimodal fusion and for
defining "incident" (future) AFib — i.e., patients with no AFib ICD code at/before the
index ECG, followed forward for a diagnosis window (e.g., 1 or 3 years).

Optional **external validation** set once the primary model works: PTB-XL (has some
follow-up metadata) or a second MIMIC-IV split by hospital era, if you want to check
generalization — mention this as a limitation/future work if skipped.

## 2. Code structure

```
afib_project/
├── cohort_extraction.py   # SQL/pandas: build incident-AFib cohort from MIMIC-IV
├── preprocessing.py       # ECG signal cleaning + tabular cleaning/imputation
├── dataset.py             # PyTorch Dataset/DataLoader for paired ECG+tabular data
├── model.py               # Dual-branch (ResNet-1D + MLP) fusion architecture
├── train.py                # Training loop with class-weighted loss, early stopping
├── evaluate.py            # AUROC, AUPRC, sensitivity/specificity, confusion matrix
├── explainability.py      # SHAP (tabular branch) + saliency maps (ECG branch)
└── requirements.txt
```

Run order: `cohort_extraction.py` → `preprocessing.py` → `train.py` → `evaluate.py` → `explainability.py`

## 3. Filling in the paper's placeholders

Running this pipeline end-to-end on your credentialed data will directly give you the
values the draft marks as `[TO BE PROVIDED]`: cohort size/class balance (Section 8),
preprocessing parameters (Section 9), hyperparameters (Section 12), split details
(Section 13), and all of Sections 15–17 (Results/Ablation/XAI findings).
