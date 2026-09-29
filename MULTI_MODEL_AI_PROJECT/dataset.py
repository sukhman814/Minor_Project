"""PyTorch Dataset for PTB-XL multimodal AFib detection."""

import os
import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset


class PTBXLAFibDataset(Dataset):
    """
    PTB-XL dataset for multimodal AFib detection.

    Inputs:
        - 12-lead ECG from records100
        - Age
        - Sex

    Output:
        ECG tensor:    (12, 1000)
        Clinical:      (2,)
        Label:         0 or 1
    """

    def __init__(
        self,
        dataframe,
        ptbxl_path,
        age_mean,
        age_std
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.ptbxl_path = ptbxl_path
        self.age_mean = age_mean
        self.age_std = age_std

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # --------------------------------------------------
        # 1. Load ECG from PTB-XL records100
        # --------------------------------------------------

        ecg_path = os.path.join(
            self.ptbxl_path,
            row["filename_lr"]
        )

        signal, fields = wfdb.rdsamp(ecg_path)

        # Original shape:
        # (1000, 12)
        #
        # Convert to:
        # (12, 1000)

        signal = signal.T.astype(np.float32)

        # --------------------------------------------------
        # 2. Per-lead normalization
        # --------------------------------------------------

        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)

        # Prevent division by zero
        std[std < 1e-8] = 1.0

        signal = (signal - mean) / std

        ecg = torch.tensor(
            signal,
            dtype=torch.float32
        )

        # --------------------------------------------------
        # 3. Clinical features
        # --------------------------------------------------

        # Age is normalized using TRAINING-set statistics.
        age = (
            row["age"] - self.age_mean
        ) / self.age_std

        sex = row["sex"]

        clinical = torch.tensor(
            [age, sex],
            dtype=torch.float32
        )

        # --------------------------------------------------
        # 4. AFib label
        # --------------------------------------------------

        label = torch.tensor(
            float(row["AFIB"]),
            dtype=torch.float32
        )

        return ecg, clinical, label