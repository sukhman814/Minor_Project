"""
Evaluation matching Section 14 of the paper: accuracy (secondary), precision,
recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)


@torch.no_grad()
def get_predictions(model, loader, device, threshold=0.5):
    model.eval()
    all_labels, all_probs = [], []
    for ecg, tab, label in loader:
        ecg, tab = ecg.to(device), tab.to(device)
        logits = model(ecg, tab)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def get_curves(y_true, y_prob):
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresh},
        "pr": {"precision": prec, "recall": rec, "thresholds": pr_thresh},
    }


def print_report(metrics: dict, model_name: str = "Model"):
    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"  Confusion matrix: {v}")
        else:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    print("Import get_predictions/compute_metrics after training a model; "
          "see train.py for how to obtain `model` and `test_loader`.")
