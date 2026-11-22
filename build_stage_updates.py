from pathlib import Path
import shutil
import json
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectral import open_image
from scipy.io import loadmat

ROOT = Path.cwd()

S1 = ROOT / "Stage 1 Results"
S2 = ROOT / "Stage 2 Results"
S1_OUT = ROOT / "Stage 1 Update"
S2_OUT = ROOT / "Stage 2 Update"

for folder in [
    "Graphs", "Abundance Maps", "Endmember Analysis", "Metrics"
]:
    (S1_OUT / folder).mkdir(parents=True, exist_ok=True)
    (S2_OUT / folder).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. COPY VERIFIED STAGE 1 RESULTS
# --------------------------------------------------

def copy_files(source, destination, pattern="*"):
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for file in source.glob(pattern):
        if file.is_file():
            shutil.copy2(file, destination / file.name)

copy_files(S1 / "Graphs", S1_OUT / "Graphs", "*.png")
copy_files(S1 / "Abundance Maps", S1_OUT / "Abundance Maps", "*.png")
copy_files(S1 / "Endmember Analysis",
           S1_OUT / "Endmember Analysis", "*.png")
copy_files(S1 / "Metrics", S1_OUT / "Metrics")

# Samran's updated Stage 1 metrics are the source of truth.
with open(S1 / "Metrics" / "stage1_metrics.json",
          encoding="utf-8") as f:
    m1 = json.load(f)

stage1_summary = f"""# Stage 1 Update — Samson Baseline

## Dataset
- Image: 95 x 95 pixels
- Spectral bands: 156
- Endmembers: 3
- Ground-truth abundance layout: {m1['ground_truth_abundance_layout']}

## Reconstruction metrics
- MSE: {m1['MSE_original_units']:.6f}
- RMSE: {m1['RMSE_original_units']:.6f}
- MAE: {m1['MAE_original_units']:.6f}
- Scaled MSE: {m1['final_scaled_MSE_from_training']:.8f}
- MSE reduction against mean-spectrum baseline:
  {m1['MSE_reduction_vs_baseline_percent']:.4f}%

## Ground-truth evaluation
- Mean endmember SAD: {m1['ground_truth_endmember_mean_SAD_degrees']:.2f} degrees
- Component SAD in predicted-component order:
  {m1['ground_truth_endmember_SAD_degrees_by_predicted_component']}
- Predicted-to-ground-truth matching permutation:
  {m1['ground_truth_endmember_permutation']}
- Abundance RMSE: {m1['abundance_RMSE_ground_truth']:.6f}
- Mean abundance entropy: {m1['mean_abundance_entropy']:.6f}
- Maximum abundance-sum error: {m1['max_abundance_sum_error']}

## Important provenance note
The Stage 1 images are retained from the existing generated results.
The updated numerical evaluation above uses Samran's corrected
ground-truth layout and metrics.

The original Stage 1 prediction checkpoint is not present in this
repository snapshot. Therefore, these images have not been
recomputed from raw Stage 1 predictions.

Component labels refer to the original Stage 1 model order.
Use the matching permutation above when comparing material identities.
"""

(S1_OUT / "summary.md").write_text(stage1_summary, encoding="utf-8")

# --------------------------------------------------
# 2. LOAD STAGE 2 CHECKPOINT AND ORIGINAL SAMSON DATA
# --------------------------------------------------

checkpoint_path = (
    ROOT / "Stage2" /
    "samson_unmixing_random_warmup0_seed0_lambda0.001.pt"
)

checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)

a = checkpoint["a"].detach().cpu().numpy()
e = checkpoint["e"].detach().cpu().numpy()

height, width, bands = 95, 95, 156

if a.shape != (9025, 3) or e.shape != (156, 3):
    raise ValueError("Unexpected Stage 2 abundance/endmember dimensions")

hdr = next(ROOT.rglob("samson_1.img.hdr"))
cube = np.asarray(open_image(str(hdr)).load(), dtype=np.float64)

if cube.shape != (height, width, bands):
    raise ValueError(f"Unexpected Samson cube shape: {cube.shape}")

x = cube.reshape(-1, bands)
xh = a @ e.T

if not np.isfinite(xh).all():
    raise ValueError("Reconstructed spectra contain invalid values")

# Verify ground truth file and dimensions.
gt_path = next(ROOT.rglob("samgt.mat"))
gt = loadmat(str(gt_path))

gt_e = np.asarray(gt["M"], dtype=np.float64)
gt_a_raw = np.asarray(gt["A"], dtype=np.float64)

if gt_e.shape != (156, 3) or gt_a_raw.shape != (3, 9025):
    raise ValueError("Unexpected Samson ground-truth dimensions")

# Exact verified layout used by Samran's evaluator.
gt_a = gt_a_raw.reshape(3, 95, 95, order="F").reshape(3, -1).T

print("Loaded Samson:", cube.shape)
print("Loaded Stage 2 abundances:", a.shape)
print("Loaded Stage 2 endmembers:", e.shape)
print("Ground-truth layout: fortran_to_c")
print("Stage 2 reconstruction:", xh.shape)

# --------------------------------------------------
# 3. GENERATE THE SIX MATCHING STAGE 2 GRAPHS
# --------------------------------------------------

G = S2_OUT / "Graphs"
AM = S2_OUT / "Abundance Maps"
EA = S2_OUT / "Endmember Analysis"

# Graph 1: Center-pixel reconstruction.
center = (height // 2) * width + width // 2
b = np.arange(1, bands + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(b, x[center], label="Original spectrum", linewidth=2)
ax.plot(b, xh[center], label="Reconstructed spectrum",
        linewidth=2, linestyle="--")
ax.set_title("Original vs Reconstructed Spectrum — Centre Pixel")
ax.set_xlabel("Spectral band")
ax.set_ylabel("Spectral value (original units)")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(G / "original_vs_reconstructed_center_pixel.png", dpi=300)
plt.close(fig)

# Graph 2: Scene-mean reconstruction.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(b, x.mean(axis=0), label="Mean original spectrum", linewidth=2)
ax.plot(b, xh.mean(axis=0), label="Mean reconstructed spectrum",
        linewidth=2, linestyle="--")
ax.set_title("Scene-Mean Original vs Reconstructed Spectra")
ax.set_xlabel("Spectral band")
ax.set_ylabel("Mean spectral value (original units)")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(G / "original_vs_reconstructed_mean_spectrum.png", dpi=300)
plt.close(fig)

# Graph 3: Combined predicted abundance maps.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for k, ax in enumerate(axes):
    im = ax.imshow(a[:, k].reshape(height, width),
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Component {k + 1}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.colorbar(im, ax=list(axes), shrink=0.8,
             label="Predicted abundance")
fig.suptitle("Predicted Abundance Maps — Stage 2 Samson")
fig.tight_layout()
fig.savefig(G / "predicted_abundance_maps.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# Graph 4: Per-pixel reconstruction RMSE.
err = x - xh
pixel_rmse = np.sqrt(np.mean(err ** 2, axis=1)).reshape(height, width)

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(pixel_rmse, cmap="magma")
ax.set_title("Per-Pixel Reconstruction RMSE — Stage 2 Samson")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
fig.colorbar(im, ax=ax, label="RMSE (original units)")
fig.tight_layout()
fig.savefig(G / "reconstruction_error_heatmap.png", dpi=300)
plt.close(fig)

# Graph 5: Original versus reconstructed scatter.
flat_x = x.ravel()
flat_xh = xh.ravel()
step = max(1, flat_x.size // 20000)

lo = min(flat_x.min(), flat_xh.min())
hi = max(flat_x.max(), flat_xh.max())

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(flat_x[::step], flat_xh[::step], s=5, alpha=0.25)
ax.plot([lo, hi], [lo, hi], "--", label="Ideal reconstruction")
ax.set_title("Original vs Reconstructed Pixel-Band Values")
ax.set_xlabel("Original value")
ax.set_ylabel("Reconstructed value")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(G / "reconstruction_scatter.png", dpi=300)
plt.close(fig)

# Graph 6: Reconstruction training loss.
loss = np.asarray(checkpoint["recon_mse_history"], dtype=float)

if loss.size == 0 or not np.isfinite(loss).all():
    raise ValueError("Stage 2 reconstruction loss history is unavailable")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(np.arange(1, len(loss) + 1), loss, linewidth=2)
ax.set_title("Training Loss Curve — Stage 2")
ax.set_xlabel("Epoch")
ax.set_ylabel("Scaled reconstruction MSE")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(G / "training_loss_curve.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# 4. THREE INDIVIDUAL ABUNDANCE MAPS
# --------------------------------------------------

for k in range(3):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(a[:, k].reshape(height, width),
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Predicted Abundance Map — Component {k + 1}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="Predicted abundance")
    fig.tight_layout()
    fig.savefig(AM / f"abundance_component_{k + 1}.png", dpi=300)
    plt.close(fig)

# --------------------------------------------------
# 5. FOUR ENDMEMBER ANALYSIS IMAGES
# --------------------------------------------------

for k in range(3):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b, e[:, k], linewidth=2)
    ax.set_title(f"Learned Endmember {k + 1} — Stage 2")
    ax.set_xlabel("Spectral band")
    ax.set_ylabel("Spectral value (original units)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(EA / f"endmember_{k + 1}.png", dpi=300)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
for k in range(3):
    ax.plot(b, e[:, k], linewidth=2, label=f"Endmember {k + 1}")

ax.set_title("Learned Endmember Spectra — Stage 2 Samson")
ax.set_xlabel("Spectral band")
ax.set_ylabel("Spectral value (original units)")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(EA / "learned_endmember_spectra.png", dpi=300)
plt.close(fig)

# --------------------------------------------------
# 6. COPY STAGE 2 METRICS AND SUPPLEMENTARY PLOTS
# --------------------------------------------------

copy_files(S2 / "Metrics", S2_OUT / "Metrics", "*.csv")

sweep = S2_OUT / "Parameter Sweep"
evaluation = S2_OUT / "Evaluation Comparisons"
sweep.mkdir(exist_ok=True)
evaluation.mkdir(exist_ok=True)

for filename in [
    "01_mean_SAD_vs_logdet_weight.png",
    "02_abundance_RMSE_vs_logdet_weight.png",
]:
    shutil.copy2(S2 / "Graphs" / filename, sweep / filename)

for filename in [
    "03_endmember_spectra_vs_ground_truth.png",
    "04_abundance_maps_vs_ground_truth.png",
]:
    shutil.copy2(S2 / "Graphs" / filename, evaluation / filename)

# --------------------------------------------------
# 7. STAGE 2 SUMMARY
# --------------------------------------------------

settings = checkpoint["settings"]

report = f"""# Stage 2 Update — Log-det Volume Autoencoder

## Dataset and configuration
- Dataset: Samson
- Image dimensions: 95 x 95 pixels
- Spectral bands: 156
- Endmembers: 3
- Epochs: {settings['epochs']}
- Initialization: {settings['init']}
- Random seed: {settings['seed']}
- Log-det weight: {settings['lambda_vol']}
- Warm-up epochs: {settings['warmup_epochs']}
- Input scale: {settings['input_scale']}
- Ground-truth abundance layout: fortran_to_c

## Saved checkpoint
- Final reconstruction MSE: {checkpoint['final_loss']:.6f}
- Final scaled reconstruction MSE: {checkpoint['final_scaled_mse']:.8f}
- Final log-det value: {checkpoint['final_vol_loss']:.6f}
- Abundance shape: {a.shape}
- Endmember shape: {e.shape}

## Evaluation provenance
- 36 experiment records and 12 grouped configurations are retained
  in the parameter-sweep metric CSV files.
- Two raw checkpoints are available for independent re-evaluation.
- The remaining 34 runs are represented by recorded CSV metrics and
  cannot be independently rerun from raw checkpoints in this repository.

## Component-order note
The maps and endmember plots retain the model's native component order.
Component numbers are not guaranteed to identify the same material
across different models; use the recorded Hungarian matching permutation
when comparing components.

## Graphs
Six graph types matching the Stage 1 presentation:
1. Centre-pixel original vs reconstructed spectrum.
2. Scene-mean original vs reconstructed spectrum.
3. Combined predicted abundance maps.
4. Per-pixel reconstruction RMSE heatmap.
5. Original vs reconstructed scatter.
6. Training reconstruction-loss curve.

## Additional outputs
- Three individual abundance maps.
- Three individual endmember plots and one combined plot.
- Parameter sweep graphs.
- Ground-truth comparison graphs.
- Four evaluation CSV files.
"""

(S2_OUT / "summary.md").write_text(report, encoding="utf-8")

# --------------------------------------------------
# 8. STRICT COUNT CHECK
# --------------------------------------------------

def png_count(path):
    return len(list(path.glob("*.png")))

expected = {
    "Graphs": 6,
    "Abundance Maps": 3,
    "Endmember Analysis": 4,
}

for folder, count in expected.items():
    actual1 = png_count(S1_OUT / folder)
    actual2 = png_count(S2_OUT / folder)

    print(f"{folder}: Stage 1={actual1}, Stage 2={actual2}")

    if actual1 != count or actual2 != count:
        raise RuntimeError(
            f"Image count mismatch in {folder}: "
            f"expected {count}, got {actual1}/{actual2}"
        )

print("\nBoth update folders generated successfully.")
print("Stage 1 source images preserved; corrected metrics included.")
print("Stage 2 images generated from the saved lambda=0.001 checkpoint.")
print("No model training was performed.")
