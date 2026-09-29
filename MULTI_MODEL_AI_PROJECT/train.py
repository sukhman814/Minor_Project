"""
Training loop for the fused multimodal model.
Handles class imbalance via pos_weight in BCEWithLogitsLoss, patient-level
stratified splitting must be done upstream (see cohort_extraction.py — split
by subject_id, not by record, to avoid leakage).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from model import MultimodalAFibModel
from dataset import AFibMultimodalDataset

SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7  # early stopping patience (epochs without val AUROC improvement)

torch.manual_seed(SEED)
np.random.seed(SEED)


def patient_level_split(cohort_df, test_size=0.15, val_size=0.15, seed=SEED):
    """Split by subject_id so no patient appears in more than one split."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(
        gss1.split(cohort_df, groups=cohort_df["subject_id"])
    )
    trainval_df = cohort_df.iloc[trainval_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=seed
    )
    train_idx, val_idx = next(
        gss2.split(trainval_df, groups=trainval_df["subject_id"])
    )
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]
    test_df = cohort_df.iloc[test_idx]
    return train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for ecg, tab, label in loader:
        ecg, tab, label = ecg.to(device), tab.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(ecg, tab)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ecg.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_and_auc(model, loader, criterion, device):
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for ecg, tab, label in loader:
        ecg, tab, label = ecg.to(device), tab.to(device), label.to(device)
        logits = model(ecg, tab)
        loss = criterion(logits, label)
        total_loss += loss.item() * ecg.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(label.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader.dataset), auc


def main(cohort_df, tabular_X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = patient_level_split(cohort_df)
    train_ds = AFibMultimodalDataset(train_df, tabular_X[train_df.index])
    val_ds = AFibMultimodalDataset(val_df, tabular_X[val_df.index])
    test_ds = AFibMultimodalDataset(test_df, tabular_X[test_df.index])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MultimodalAFibModel(tabular_in_features=tabular_X.shape[1]).to(device)

    n_pos = train_df["incident_afib"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate_loss_and_auc(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "best_multimodal_model.pt")
    print(f"Best val AUROC: {best_auc:.4f}")
    return model, test_loader, device


if __name__ == "__main__":
    import pandas as pd
    from preprocessing import build_tabular_features

    cohort = pd.read_csv("cohort.csv")
    # clinical_features_df must be built separately by joining cohort with
    # MIMIC-IV chartevents/labevents/diagnoses per Section 9 of the paper.
    # X, feature_names, scaler, imputer = build_tabular_features(clinical_features_df)
    # main(cohort, X)
    print("Wire up clinical_features_df, then call main(cohort, X).")
