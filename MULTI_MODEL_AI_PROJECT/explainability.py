"""
Explainable AI for the fused model, matching Section 4.4 / Section 12 of the paper:
  - SHAP (KernelExplainer or DeepExplainer) on the tabular branch
  - Gradient-based saliency on the ECG branch (per-lead, per-timestep attribution)
"""

import numpy as np
import torch
import shap


# ---------------------------------------------------------------------------
# SHAP for tabular branch
# ---------------------------------------------------------------------------
def tabular_shap_explain(model, background_tab, sample_tab, ecg_context, device,
                          feature_names=None):
    """
    background_tab: (N_bg, F) tabular samples used as SHAP background
    sample_tab:      (N_sample, F) tabular samples to explain
    ecg_context:     single ECG tensor (or batch) held fixed while tabular varies,
                      OR a representative "average" ECG embedding if you want a
                      tabular-marginal explanation independent of one specific ECG.
    """
    model.eval()

    def predict_fn(tab_np: np.ndarray) -> np.ndarray:
        tab_t = torch.tensor(tab_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            ecg_embed = model.ecg_encoder(ecg_context.to(device))
            if ecg_embed.shape[0] == 1 and tab_t.shape[0] > 1:
                ecg_embed = ecg_embed.expand(tab_t.shape[0], -1)
            tab_embed = model.tab_encoder(tab_t)
            logits = model.classifier(ecg_embed, tab_embed)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, background_tab)
    shap_values = explainer.shap_values(sample_tab, nsamples=100)

    if feature_names is not None:
        shap.summary_plot(shap_values, sample_tab, feature_names=feature_names, show=False)

    return shap_values, explainer


# ---------------------------------------------------------------------------
# Saliency for ECG branch
# ---------------------------------------------------------------------------
def ecg_saliency(model, ecg_tensor, tab_tensor, device):
    """
    Vanilla gradient saliency: d(prob) / d(input ECG), highlighting which
    leads/timesteps most influenced the prediction for one sample.
    Returns a (leads, timesteps) saliency map.
    """
    model.eval()
    ecg = ecg_tensor.clone().unsqueeze(0).to(device).requires_grad_(True)
    tab = tab_tensor.clone().unsqueeze(0).to(device)

    logits = model(ecg, tab)
    prob = torch.sigmoid(logits)
    model.zero_grad()
    prob.backward()

    saliency = ecg.grad.abs().squeeze(0).cpu().numpy()  # (leads, timesteps)
    return saliency


def summarize_lead_importance(saliency: np.ndarray, lead_names=None):
    """Aggregate per-lead saliency (mean abs gradient) for a compact global summary."""
    lead_importance = saliency.mean(axis=1)
    if lead_names is None:
        lead_names = [f"lead_{i}" for i in range(saliency.shape[0])]
    return dict(zip(lead_names, lead_importance.tolist()))


LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]


if __name__ == "__main__":
    print("Call tabular_shap_explain(...) and ecg_saliency(...) after loading a "
          "trained model checkpoint (best_multimodal_model.pt from train.py).")
