# ECG-Dominant Residual Fusion for AFib Detection

## Overview

This project implements a multimodal deep-learning approach for detecting atrial fibrillation (AFib) from 12-lead ECG recordings and structured metadata.

The final model uses an ECG encoder as the primary predictor and a gated metadata branch to learn a residual correction to the ECG prediction.

## Dataset

- Dataset: PTB-XL v1.0.3
- ECG sampling frequency: 100 Hz
- Recording duration: 10 seconds
- ECG input shape: 12 × 1,000
- Metadata: Age, Sex, and Heart Axis encoded as 11 features

The dataset is not included in this repository. Download it from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

Follow the dataset's terms and instructions.

## Model Architecture

1. ECG branch: 1D CNN with ResNet-style residual blocks.
2. Metadata branch: MLP with a learned feature-wise gate.
3. Residual fusion: a base ECG logit plus a learned metadata-informed correction.
4. Output: AFib probability, converted into a binary prediction using a validation-selected threshold.

## Models Compared

- Metadata-only
- ECG-only
- Simple multimodal fusion
- Feature-wise gated multimodal fusion
- ECG-dominant residual fusion

## Final Reported Results

| Metric | ECG-dominant residual fusion |
|---|---:|
| Accuracy | 98.10% |
| Precision | 83.84% |
| Recall | 88.48% |
| F1-score | 86.10% |
| Decision threshold | 0.945 |

These values are from the reported PTB-XL test split. Independent external validation has not been performed.

## Repository Structure

- `notebooks/`: Main implementation and experiments
- `src/`: Python modules, if applicable
- `results/`: Evaluation tables and figures
- `paper/`: Research paper
- `requirements.txt`: Python dependencies

## Setup

1. Install Python and Jupyter.
2. Install dependencies with `pip install -r requirements.txt`.
3. Download PTB-XL v1.0.3 from PhysioNet.
4. Configure the dataset path in the notebook for your computer.
5. Open Jupyter and run the notebook cells in order.

## Limitations

This project detects AFib from an existing ECG recording; it does not predict future AFib risk. The results are experimental and do not establish clinical readiness. The same test set was examined during model development, so independent patient-level evaluation is needed.

## Technology

Python, PyTorch, NumPy, pandas, scikit-learn, WFDB, Matplotlib, and SHAP.
