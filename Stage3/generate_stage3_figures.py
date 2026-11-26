from pathlib import Path
import csv
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from dataset_loaders import load_benchmark_dataset
from evaluate_unmixing_benchmarks import (
    compute_sad_matrix_deg,
    gt_abundance_layouts,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Stage 3 Results"
CHECKPOINTS = ROOT / "Stage3"

GRAPH_DIR = RESULTS / "Graphs"
MAP_DIR = RESULTS / "Abundance Maps"
ANALYSIS_DIR = RESULTS / "Endmember Analysis"

for folder in (GRAPH_DIR, MAP_DIR, ANALYSIS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

METHODS = [
    "VCA",
    "SPA",
    "N-FINDR",
    "MVC-NMF",
    "AE (lambda=0)",
    "LogDet-AE",
]
DATASETS = ["jasper_ridge", "urban", "cuprite"]
DISPLAY = {
    "jasper_ridge": "Jasper Ridge",
    "urban": "Urban",
    "cuprite": "Cuprite",
}
SHORT = {
    "VCA": "VCA",
    "SPA": "SPA",
    "N-FINDR": "N-FINDR",
    "MVC-NMF": "MVC-NMF",
    "AE (lambda=0)": "AE (λ=0)",
    "LogDet-AE": "LogDet-AE",
}

# Keep method colors consistent in every figure.
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
METHOD_COLORS = {m: cycle[i % len(cycle)] for i, m in enumerate(METHODS)}
GT_COLOR = "black"

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def close_enough(a, b, atol=1e-6):
    return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=atol)

print("===== LOADING VALIDATED METRICS =====")
per_run_path = RESULTS / "real_datasets_per_run_metrics.csv"
per_run_rows = read_csv(per_run_path)

if len(per_run_rows) != 18:
    raise RuntimeError(f"Expected 18 active real-dataset rows, found {len(per_run_rows)}")

rows_by_dataset = {d: {} for d in DATASETS}
for row in per_run_rows:
    ds_name = row["dataset"]
    method = row["method"]
    if ds_name not in rows_by_dataset:
        raise RuntimeError(f"Unexpected dataset in CSV: {ds_name}")
    if method in rows_by_dataset[ds_name]:
        raise RuntimeError(f"Duplicate method row: {ds_name}, {method}")
    rows_by_dataset[ds_name][method] = row

for ds_name in DATASETS:
    if set(rows_by_dataset[ds_name]) != set(METHODS):
        raise RuntimeError(f"Incomplete method coverage for {ds_name}")

print("PASS: 18 records, 6 methods per dataset.")

# Load actual datasets and match each checkpoint's components to GT endmembers.
loaded = {}

for ds_name in DATASETS:
    print(f"\n===== VERIFYING {DISPLAY[ds_name]} =====")
    ds = load_benchmark_dataset(ds_name, repo_root=ROOT)
    h, w, bands = ds.cube.shape
    p = ds.gt_M.shape[1]
    dataset_results = {}

    if ds.gt_A is not None:
        gt_A = gt_abundance_layouts(ds.gt_A, h, w)["fortran_to_c"]
    else:
        gt_A = None

    for method in METHODS:
        row = rows_by_dataset[ds_name][method]
        ckpt_path = CHECKPOINTS / row["file"]

        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        E = as_numpy(ckpt["e"]).astype(np.float64)
        A = as_numpy(ckpt["a"]).astype(np.float64)

        if E.shape != (bands, p):
            raise ValueError(f"{ckpt_path.name}: E shape {E.shape}, expected {(bands, p)}")
        if A.shape != (h * w, p):
            raise ValueError(f"{ckpt_path.name}: A shape {A.shape}, expected {(h*w, p)}")
        if not np.isfinite(E).all() or not np.isfinite(A).all():
            raise ValueError(f"Non-finite checkpoint values: {ckpt_path.name}")
        if A.min() < -1e-5:
            raise ValueError(f"Negative abundance found: {ckpt_path.name}")

        sum_error = float(np.max(np.abs(A.sum(axis=1) - 1.0)))
        if sum_error > 5e-3:
            raise ValueError(
                f"Abundance simplex check failed for {ckpt_path.name}: {sum_error}"
            )

        sad_matrix = compute_sad_matrix_deg(ds.gt_M, E)
        gt_idx, est_idx = linear_sum_assignment(sad_matrix)
        order = np.argsort(gt_idx)
        gt_idx = gt_idx[order]
        est_idx = est_idx[order]

        E_match = E[:, est_idx]
        A_match = A[:, est_idx]
        sad_components = sad_matrix[gt_idx, est_idx]
        mean_sad = float(np.mean(sad_components))

        if not close_enough(mean_sad, row["mean_sad_deg"], atol=1e-5):
            raise RuntimeError(
                f"SAD mismatch for {ds_name}/{method}: "
                f"checkpoint={mean_sad}, CSV={row['mean_sad_deg']}"
            )

        abundance_rmse = float("nan")
        if gt_A is not None:
            abundance_rmse = float(np.sqrt(np.mean((A_match - gt_A) ** 2)))
            if not close_enough(abundance_rmse, row["rmse"], atol=1e-5):
                raise RuntimeError(
                    f"Abundance RMSE mismatch for {ds_name}/{method}: "
                    f"checkpoint={abundance_rmse}, CSV={row['rmse']}"
                )

        dataset_results[method] = {
            "row": row,
            "E": E_match,
            "A": A_match,
            "sad_components": sad_components,
            "sum_error": sum_error,
        }

        print(
            f"{method:14s} | SAD={mean_sad:8.4f}° | "
            f"Abundance sum error={sum_error:.2e}"
        )

    loaded[ds_name] = {
        "dataset": ds,
        "height": h,
        "width": w,
        "bands": bands,
        "components": p,
        "gt_A": gt_A,
        "methods": dataset_results,
    }

print("\nPASS: checkpoint arrays, endmember matching, SAD and abundance RMSE validated.")

# Verify synthetic CSVs before plotting them.
trial_rows = read_csv(RESULTS / "synthetic_purity_per_trial.csv")
phase_rows = read_csv(RESULTS / "synthetic_purity_phase_transition.csv")

if len(trial_rows) != 300 or len(phase_rows) != 60:
    raise RuntimeError("Synthetic CSV row counts changed; expected 300 and 60.")

print("PASS: synthetic CSV row counts are 300 trial rows and 60 summary rows.")

# ---------------------------------------------------------------------
# GRAPH 1: Mean endmember SAD comparison across all real datasets.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)

for ax, ds_name in zip(axes, DATASETS):
    values = [
        float(rows_by_dataset[ds_name][m]["mean_sad_deg"])
        for m in METHODS
    ]
    ax.bar(
        np.arange(len(METHODS)),
        values,
        color=[METHOD_COLORS[m] for m in METHODS],
    )
    ax.set_title(DISPLAY[ds_name])
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=35, ha="right")
    ax.set_ylabel("Mean endmember SAD (degrees)")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Stage 3 — Endmember Recovery Across Real Datasets", fontsize=15)
fig.text(
    0.5, 0.01,
    "Lower is better. One saved seed per method and dataset.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(GRAPH_DIR / "01_real_dataset_mean_sad_comparison.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------------
# GRAPH 2: Abundance RMSE (Cuprite has no abundance ground truth).
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, ds_name in zip(axes, ["jasper_ridge", "urban"]):
    values = [
        float(rows_by_dataset[ds_name][m]["rmse"])
        for m in METHODS
    ]
    ax.bar(
        np.arange(len(METHODS)),
        values,
        color=[METHOD_COLORS[m] for m in METHODS],
    )
    ax.set_title(DISPLAY[ds_name])
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=35, ha="right")
    ax.set_ylabel("Abundance RMSE")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Stage 3 — Abundance Reconstruction Error", fontsize=15)
fig.text(
    0.5, 0.01,
    "Lower is better. Cuprite is excluded because no ground-truth abundance matrix is available.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.92])
fig.savefig(GRAPH_DIR / "02_real_dataset_abundance_rmse.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------------
# GRAPH 3: Scaled reconstruction MSE for all real datasets.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

for ax, ds_name in zip(axes, DATASETS):
    values = [
        float(rows_by_dataset[ds_name][m]["final_scaled_mse"])
        for m in METHODS
    ]
    if not np.isfinite(values).all():
        raise RuntimeError(f"Missing scaled reconstruction MSE for {ds_name}")

    ax.bar(
        np.arange(len(METHODS)),
        values,
        color=[METHOD_COLORS[m] for m in METHODS],
    )
    ax.set_title(DISPLAY[ds_name])
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([SHORT[m] for m in METHODS], rotation=35, ha="right")
    ax.set_ylabel("Scaled reconstruction MSE")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Stage 3 — Reconstruction Quality Across Real Datasets", fontsize=15)
fig.text(
    0.5, 0.01,
    "Lower is better. Values are taken from the saved, independently validated checkpoints.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.92])
fig.savefig(GRAPH_DIR / "03_real_dataset_scaled_reconstruction_mse.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------------
# GRAPH 4: Synthetic purity phase transition (three linked panels).
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
metrics_to_plot = [
    ("success_rate_at_threshold", None, "Successful recovery (%)"),
    ("mean_sad_mean_deg", "mean_sad_std_deg", "Mean endmember SAD (degrees)"),
    ("rmse_mean", "rmse_std", "Abundance RMSE"),
]

for ax, (field, error_field, ylabel) in zip(axes, metrics_to_plot):
    for method in METHODS:
        subset = sorted(
            [r for r in phase_rows if r["method"] == method],
            key=lambda r: float(r["purity_cap"]),
        )
        x = np.array([float(r["purity_cap"]) for r in subset])
        y = np.array([float(r[field]) for r in subset])

        if field == "success_rate_at_threshold":
            y = 100.0 * y
            ax.plot(
                x, y, marker="o", linewidth=1.8,
                label=SHORT[method], color=METHOD_COLORS[method],
            )
            ax.set_ylim(-3, 103)
        else:
            err = np.array([float(r[error_field]) for r in subset])
            ax.errorbar(
                x, y, yerr=err, marker="o", linewidth=1.6,
                capsize=2, label=SHORT[method],
                color=METHOD_COLORS[method],
            )

    ax.set_xlabel("Maximum abundance purity")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_xlim(0.53, 1.02)

axes[0].set_title("Identifiability at 5°")
axes[1].set_title("Spectral recovery")
axes[2].set_title("Abundance recovery")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
fig.suptitle("Stage 3 — Synthetic Purity Phase Transition", fontsize=15)
fig.text(
    0.5, 0.01,
    "Five trials per purity and method. Success requires every matched endmember SAD to be within 5°.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0, 0.13, 1, 0.92])
fig.savefig(GRAPH_DIR / "04_synthetic_purity_phase_transition.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------------
# ABUNDANCE MAPS: one composite image for each real dataset.
# ---------------------------------------------------------------------
map_methods = {
    "jasper_ridge": ["LogDet-AE", "N-FINDR"],
    "urban": ["LogDet-AE", "MVC-NMF"],
    "cuprite": ["AE (lambda=0)", "LogDet-AE", "MVC-NMF"],
}

for ds_name in DATASETS:
    item = loaded[ds_name]
    h, w, p = item["height"], item["width"], item["components"]
    ds = item["dataset"]
    chosen = map_methods[ds_name]

    if item["gt_A"] is not None:
        column_names = ["Ground truth"] + chosen
        ncols = len(column_names)
    else:
        column_names = chosen
        ncols = len(column_names)

    fig, axes = plt.subplots(
        p, ncols,
        figsize=(4.1 * ncols, max(8, 1.75 * p)),
        squeeze=False,
    )

    for col, name in enumerate(column_names):
        if name == "Ground truth":
            maps = item["gt_A"].reshape(h, w, p)
        else:
            maps = item["methods"][name]["A"].reshape(h, w, p)

        for component in range(p):
            ax = axes[component, col]
            im = ax.imshow(
                maps[:, :, component],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if component == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"GT component {component + 1}", fontsize=8)

    fig.colorbar(
        im, ax=axes.ravel().tolist(),
        shrink=0.65, pad=0.015, label="Abundance",
    )
    if item["gt_A"] is None:
        title = "Cuprite — Predicted Abundance Maps (No Ground-Truth Abundances)"
        note = "Components are ordered by matched endmember spectral angle; no GT abundance comparison is possible."
    else:
        title = f"{DISPLAY[ds_name]} — Ground Truth and Matched Predicted Abundances"
        note = "Predicted components are reordered to match ground-truth endmembers by minimum total spectral angle."

    fig.suptitle(title, fontsize=13)
    fig.text(0.5, 0.005, note, ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(
        MAP_DIR / f"{ds_name}_abundance_maps.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

# ---------------------------------------------------------------------
# ENDMEMBER ANALYSIS 1–3: GT spectra vs all six matched methods.
# ---------------------------------------------------------------------
for ds_name in DATASETS:
    item = loaded[ds_name]
    ds = item["dataset"]
    p = item["components"]
    bands = item["bands"]

    ncols = 2 if p <= 4 else 4
    nrows = int(np.ceil(p / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.2 * ncols, 3.7 * nrows),
        squeeze=False,
    )

    for component in range(p):
        ax = axes.flat[component]
        gt_spec = ds.gt_M[:, component]
        gt_norm = gt_spec / max(float(np.max(np.abs(gt_spec))), 1e-12)
        ax.plot(
            np.arange(1, bands + 1), gt_norm,
            color=GT_COLOR, linewidth=2.5, label="Ground truth",
        )

        for method in METHODS:
            estimate = item["methods"][method]["E"][:, component]
            estimate_norm = estimate / max(float(np.max(np.abs(estimate))), 1e-12)
            ax.plot(
                np.arange(1, bands + 1),
                estimate_norm,
                linewidth=1.25,
                alpha=0.9,
                color=METHOD_COLORS[method],
                label=SHORT[method],
            )

        ax.set_title(f"Ground-truth endmember {component + 1}")
        ax.set_xlabel("Spectral band")
        ax.set_ylabel("Max-normalized intensity")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)

    for unused in range(p, nrows * ncols):
        axes.flat[unused].axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"{DISPLAY[ds_name]} — Ground Truth vs Estimated Endmember Signatures",
        fontsize=14,
    )
    fig.text(
        0.5, 0.01,
        "Estimated spectra are matched to ground truth by minimum total spectral angle and individually max-normalized.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(
        ANALYSIS_DIR / f"{ds_name}_endmember_spectra_comparison.png",
        dpi=200,
    )
    plt.close(fig)

# ---------------------------------------------------------------------
# ENDMEMBER ANALYSIS 4: Per-component SAD heatmap for all methods.
# ---------------------------------------------------------------------
heat_rows = []
row_labels = []

for ds_name in DATASETS:
    p = loaded[ds_name]["components"]
    for method in METHODS:
        values = loaded[ds_name]["methods"][method]["sad_components"]
        padded = np.full(12, np.nan, dtype=float)
        padded[:p] = values
        heat_rows.append(padded)
        row_labels.append(f"{DISPLAY[ds_name]} | {SHORT[method]}")

heat = np.vstack(heat_rows)
masked = np.ma.masked_invalid(heat)

fig, ax = plt.subplots(figsize=(13, 10))
cmap = plt.get_cmap("magma").copy()
cmap.set_bad("#e5e5e5")
im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")

ax.set_xticks(np.arange(12))
ax.set_xticklabels([f"GT EM {i+1}" for i in range(12)])
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_xlabel("Ground-truth endmember")
ax.set_title("Stage 3 — Per-Endmember Spectral Angle Difference")

for r in range(heat.shape[0]):
    for c in range(12):
        if np.isfinite(heat[r, c]):
            ax.text(
                c, r, f"{heat[r,c]:.1f}",
                ha="center", va="center",
                fontsize=6,
                color="white" if heat[r, c] > np.nanmedian(heat) else "black",
            )

fig.colorbar(im, ax=ax, label="Spectral angle difference (degrees)")
fig.tight_layout()
fig.savefig(ANALYSIS_DIR / "per_endmember_sad_heatmap.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------------
# FINAL COUNT AND OUTPUT CHECK.
# ---------------------------------------------------------------------
expected = {
    GRAPH_DIR: 4,
    MAP_DIR: 3,
    ANALYSIS_DIR: 4,
}

print("\n===== FINAL FIGURE COUNTS =====")
for folder, wanted in expected.items():
    files = sorted(folder.glob("*.png"))
    print(f"{folder.relative_to(ROOT)}: {len(files)} PNG files")
    for file in files:
        if file.stat().st_size < 10_000:
            raise RuntimeError(f"Figure appears unexpectedly small: {file}")
        print(f"  {file.name} ({file.stat().st_size:,} bytes)")
    if len(files) != wanted:
        raise RuntimeError(f"Expected {wanted} PNGs in {folder}, found {len(files)}")

manifest = """# Stage 3 Figure Manifest

## Generated figures
- Graphs: 4
- Abundance Maps: 3
- Endmember Analysis: 4
- Total: 11 PNG figures

## Validation
- 18 active real-dataset checkpoint records were independently evaluated.
- Existing per-run metrics matched the independent evaluation with zero mismatches.
- Checkpoint arrays were checked for expected shapes, finite values, non-negative abundances, and abundance sums.
- Endmember matching was recomputed using minimum total spectral angle.
- Jasper Ridge and Urban abundance RMSE values were recomputed and checked against the validated CSV.
- Synthetic purity CSVs contain 300 trial rows and 60 summary rows; 5 trials are present in each of 60 purity/method groups. Summary means, standard deviations, and success flags were independently checked.

## Interpretation limits
- There is one saved seed per real-dataset/method combination; standard deviations across real-dataset runs are therefore not estimable.
- Cuprite has ground-truth endmember spectra but no ground-truth abundance matrix. Its predicted abundance maps are shown without ground-truth abundance comparisons or abundance RMSE.
- Jasper Ridge and Urban checkpoint clipping metadata is blank in the existing records. Blank is treated as unknown, not as zero clipped samples.
- The historical records in real_datasets_metrics_missing_checkpoints.csv are not used in these figures because their raw checkpoints are unavailable.
- The synthetic experiment has five trials per purity and method. It uses the available Cuprite ground-truth spectra as the library fallback because no separate USGS library file is present in the dataset directory. Its results should be described as preliminary synthetic evidence, not proof of identifiability.

## Component matching
Predicted endmember and abundance components are reordered using Hungarian assignment on the spectral-angle matrix, with ground-truth component order as the reference.
"""

(RESULTS / "figure_manifest.md").write_text(manifest, encoding="utf-8")

print("\nSUCCESS: Stage 3 figure generation and checks completed.")
print("Existing CSV files and original layout-check images were not overwritten.")
