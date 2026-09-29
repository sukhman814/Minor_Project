"""
Dual-branch multimodal architecture:
  - ECG branch: 1D ResNet-style CNN over 12-lead waveform
  - Tabular branch: MLP over structured clinical features
  - Fusion: concatenation + MLP head (intermediate fusion)

Also exposes unimodal-only variants (ECG-only, tabular-only) for baseline comparison,
matching Section 12 / Section 13 of the paper.
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ECGEncoder(nn.Module):
    """ResNet-1D encoder for 12-lead ECG -> fixed-length embedding."""

    def __init__(self, in_leads=12, base_channels=32, embed_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, base_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.layer1 = ResidualBlock1D(base_channels, base_channels * 2, stride=2)
        self.layer2 = ResidualBlock1D(base_channels * 2, base_channels * 4, stride=2)
        self.layer3 = ResidualBlock1D(base_channels * 4, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(base_channels * 4, embed_dim)

    def forward(self, x):  # x: (B, 12, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.proj(x)  # (B, embed_dim)


class TabularEncoder(nn.Module):
    def __init__(self, in_features, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionClassifier(nn.Module):
    """Intermediate fusion: concat ECG + tabular embeddings -> MLP head -> logit."""

    def __init__(self, ecg_embed_dim=128, tab_embed_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(ecg_embed_dim + tab_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ecg_embed, tab_embed):
        fused = torch.cat([ecg_embed, tab_embed], dim=-1)
        return self.head(fused).squeeze(-1)  # logit


class MultimodalAFibModel(nn.Module):
    def __init__(self, tabular_in_features, ecg_leads=12,
                 ecg_embed_dim=128, tab_embed_dim=64):
        super().__init__()
        self.ecg_encoder = ECGEncoder(in_leads=ecg_leads, embed_dim=ecg_embed_dim)
        self.tab_encoder = TabularEncoder(tabular_in_features, embed_dim=tab_embed_dim)
        self.classifier = FusionClassifier(ecg_embed_dim, tab_embed_dim)

    def forward(self, ecg, tab):
        ecg_embed = self.ecg_encoder(ecg)
        tab_embed = self.tab_encoder(tab)
        return self.classifier(ecg_embed, tab_embed)


# --- Unimodal baselines (for the comparative evaluation in Section 13) ---
class ECGOnlyModel(nn.Module):
    def __init__(self, ecg_leads=12, ecg_embed_dim=128):
        super().__init__()
        self.encoder = ECGEncoder(in_leads=ecg_leads, embed_dim=ecg_embed_dim)
        self.head = nn.Linear(ecg_embed_dim, 1)

    def forward(self, ecg):
        return self.head(self.encoder(ecg)).squeeze(-1)


class TabularOnlyModel(nn.Module):
    def __init__(self, in_features, embed_dim=64):
        super().__init__()
        self.encoder = TabularEncoder(in_features, embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, tab):
        return self.head(self.encoder(tab)).squeeze(-1)
