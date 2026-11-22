from pathlib import Path
import csv
import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
STAGE2 = ROOT / "Stage2"
RESULTS = ROOT / "Stage 2 Results"
GRAPHS = RESULTS / "Graphs"
METRICS = RESULTS / "Metrics"

GT_PATHS = list(ROOT.rglob("samgt.mat"))

CONTROL_PATH = STAGE2 / "samson_unmixing_random_warmup0_seed0_lambda0.pt"
LOGDET_PATH = STAGE2 / "samson_unmixing_random_warmup0_seed0_lambda0.001.pt"

PER_RUN = METRICS / "samson_sad_rmse_per_run.csv"
SUMMARY = METRICS / "samson_sad_rmse_summary_by_lambda.csv"
RECOMPUTED = METRICS / "stage2_saved_checkpoint_metrics_recomputed.csv"


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def val(row, key):
    if key not in row or row[key] == "":
        raise ValueError(f"Missing column/value: {key}")
    return float(row[key])


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    if "a" not in data or "e" not in data:
        raise ValueError(f"{path.name} must contain 'a' and 'e'")

    a = data["a"]
    e = data["e"]

    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    if torch.is_tensor(e):
        e = e.detach().cpu().numpy()

    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    if a.shape != (9025, 3):
        raise ValueError(f"Unexpected abundance shape in {path.name}: {a.shape}")
    if e.shape != (156, 3):
        raise ValueError(f"Unexpected endmember shape in {path.name}: {e.shape}")

    return a, e


# --------------------------------------------------
# 2. MATCH PREDICTED ENDMEMBERS WITH GROUND TRUTH
# --------------------------------------------------

def sad_matrix(gt, estimated):
    gt_norm = np.maximum(np.linalg.norm(gt, axis=0), 1e-12)
    est_norm = np.maximum(np.linalg.norm(estimated, axis=0), 1e-12)

    cosine = (gt.T @ estimated) / np.outer(gt_norm, est_norm)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def match_components(gt, estimated, abundance):
    angles = sad_matrix(gt, estimated)

    gt_indices, est_indices = linear_sum_assignment(angles)

    matched_e = np.empty_like(gt)
    matched_a = np.empty_like(abundance)

    for gt_idx, est_idx in zip(gt_indices, est_indices):
        matched_e[:, gt_idx] = estimated[:, est_idx]
        matched_a[:, gt_idx] = abundance[:, est_idx]

    return matched_e, matched_a, angles[gt_indices, est_indices]


# --------------------------------------------------
# 3. PARAMETER SWEEP PLOTS
# --------------------------------------------------

def lambda_text(value):
    if value == 0:
        return "0"
    return f"{value:.0e}" if value < 0.001 else f"{value:g}"


def make_sweep_plot(rows, mean_column, std_column,
                    ylabel, title, filename):

    lambdas = sorted({val(row, "lambda_vol") for row in rows})
    positions = np.arange(len(lambdas))

    groups = sorted({
        (row["init"], int(row["warmup_epochs"]))
        for row in rows
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for init, warmup in groups:
        group = [
            row for row in rows
            if row["init"] == init
            and int(row["warmup_epochs"]) == warmup
        ]

        lookup = {
            val(row, "lambda_vol"): row
            for row in group
        }

        if set(lookup) != set(lambdas):
            raise ValueError(
                f"Incomplete lambda sweep for init={init}, warmup={warmup}"
            )

        means = [val(lookup[x], mean_column) for x in lambdas]
        stds = [val(lookup[x], std_column) for x in lambdas]

        ax.errorbar(
            positions,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"{init}, warm-up={warmup}",
        )

    ax.set_xticks(positions, [lambda_text(x) for x in lambdas])
    ax.set_xlabel("Log-det regularization weight (lambda)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------

def main():
    RESULTS.mkdir(exist_ok=True)
    GRAPHS.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)

    print("Reading Stage 2 evaluation CSV files...")

    per_run = read_csv(PER_RUN)
    summary = read_csv(SUMMARY)
    recomputed = read_csv(RECOMPUTED)

    if len(per_run) != 36:
        raise ValueError(
            f"Expected 36 recorded runs, found {len(per_run)}"
        )

    if len(summary) != 12:
        raise ValueError(
            f"Expected 12 grouped configurations, found {len(summary)}"
        )

    print(f"Per-run records: {len(per_run)}")
    print(f"Grouped configurations: {len(summary)}")

    # Graph 1: SAD vs lambda
    make_sweep_plot(
        summary,
        "mean_sad_mean_deg",
        "mean_sad_std_deg",
        "Mean SAD (degrees; lower is better)",
        "Samson Endmember SAD vs Log-det Weight",
        "01_mean_SAD_vs_logdet_weight.png",
    )

    # Graph 2: Abundance RMSE vs lambda
    make_sweep_plot(
        summary,
        "rmse_mean",
        "rmse_std",
        "Abundance RMSE (lower is better)",
        "Samson Abundance RMSE vs Log-det Weight",
        "02_abundance_RMSE_vs_logdet_weight.png",
    )

    # Load ground truth
    if len(GT_PATHS) != 1:
        raise RuntimeError(
            f"Expected exactly one samgt.mat, found {len(GT_PATHS)}"
        )

    print("Loading ground truth:", GT_PATHS[0])

    mat = sio.loadmat(str(GT_PATHS[0]))
    gt_e = np.asarray(mat["M"], dtype=np.float64)
    gt_a = np.asarray(mat["A"], dtype=np.float64)

    if gt_e.shape != (156, 3) or gt_a.shape != (3, 9025):
        raise ValueError(
            f"Unexpected ground-truth shapes: M={gt_e.shape}, A={gt_a.shape}"
        )

    # Use the repository evaluator's verified fortran_to_c layout.
    gt_a = gt_a.reshape(3, 95, 95, order="F")
    gt_a = gt_a.reshape(3, -1).T

    # Load the two saved checkpoints.
    control_a, control_e = load_checkpoint(CONTROL_PATH)
    logdet_a, logdet_e = load_checkpoint(LOGDET_PATH)

    # Match estimated components independently to ground truth.
    control_e, control_a, control_angles = match_components(
        gt_e, control_e, control_a
    )

    logdet_e, logdet_a, logdet_angles = match_components(
        gt_e, logdet_e, logdet_a
    )

    # Graph 3: Matched, L2-normalized endmember spectra.
    curves = [
        ("Ground truth", gt_e),
        ("Control (lambda=0)", control_e),
        ("Log-det (lambda=0.001)", logdet_e),
    ]

    bands = np.arange(1, 157)

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), sharey=True
    )

    for component, ax in enumerate(axes):
        for label, matrix in curves:
            spectrum = matrix[:, component]
            norm = max(np.linalg.norm(spectrum), 1e-12)

            ax.plot(
                bands,
                spectrum / norm,
                linewidth=1.7,
                label=label,
            )

        ax.set_title(f"Matched endmember {component + 1}")
        ax.set_xlabel("Spectral band")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("L2-normalized spectral value")
    axes[-1].legend(fontsize=8)

    fig.suptitle(
        "Stage 2 Endmember Spectra vs Samson Ground Truth"
    )
    fig.tight_layout()

    fig.savefig(
        GRAPHS / "03_endmember_spectra_vs_ground_truth.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Graph 4: Ground truth vs predicted abundance maps.
    map_data = [
        ("Ground truth", gt_a.reshape(95, 95, 3)),
        ("Control (lambda=0)", control_a.reshape(95, 95, 3)),
        ("Log-det (lambda=0.001)", logdet_a.reshape(95, 95, 3)),
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(11, 10), constrained_layout=True
    )

    image = None

    for r, (label, maps) in enumerate(map_data):
        for c in range(3):
            ax = axes[r, c]

            image = ax.imshow(
                maps[:, :, c],
                cmap="viridis",
                vmin=0,
                vmax=1,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"Endmember {c + 1}")

            if c == 0:
                ax.set_ylabel(label)

    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Abundance",
    )

    fig.suptitle(
        "Ground-Truth vs Predicted Samson Abundance Maps"
    )

    fig.savefig(
        GRAPHS / "04_abundance_maps_vs_ground_truth.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # 5. SUMMARY REPORT
    # --------------------------------------------------

    best_sad = min(
        summary,
        key=lambda row: val(row, "mean_sad_mean_deg"),
    )

    best_rmse = min(
        summary,
        key=lambda row: val(row, "rmse_mean"),
    )

    control_row = next(
        row for row in recomputed
        if row["file"].endswith("lambda0.pt")
    )

    logdet_row = next(
        row for row in recomputed
        if row["file"].endswith("lambda0.001.pt")
    )

    lines = [
        "# Stage 2 Results — Samson",
        "",
        "## Objective",
        "Evaluate the log-det volume penalty for hyperspectral endmember "
        "recovery using the Samson dataset.",
        "",
        "## Experimental setup",
        "- Dataset: Samson",
        "- Image: 95 x 95 pixels",
        "- Spectral bands: 156",
        "- Endmembers: 3",
        "- Recorded experiment evaluations: 36",
        "- Grouped configurations: 12",
        "- Ground truth: samgt.mat",
        "- Ground-truth abundance layout: fortran_to_c",
        "- Endmember matching: Hungarian assignment using SAD",
        "",
        "## Parameter sweep results",
        f"- Best mean SAD: {val(best_sad, 'mean_sad_mean_deg'):.4f} degrees",
        f"- SAD standard deviation: {val(best_sad, 'mean_sad_std_deg'):.4f}",
        f"- Best SAD configuration: lambda={best_sad['lambda_vol']}, "
        f"warm-up={best_sad['warmup_epochs']}, init={best_sad['init']}",
        f"- Best mean abundance RMSE: {val(best_rmse, 'rmse_mean'):.6f}",
        f"- RMSE standard deviation: {val(best_rmse, 'rmse_std'):.6f}",
        f"- Best RMSE configuration: lambda={best_rmse['lambda_vol']}, "
        f"warm-up={best_rmse['warmup_epochs']}, init={best_rmse['init']}",
        "",
        "## Independently re-evaluated saved checkpoints",
        f"- Control: SAD={val(control_row, 'mean_sad_deg'):.4f} degrees; "
        f"abundance RMSE={val(control_row, 'rmse'):.6f}",
        f"- Log-det: SAD={val(logdet_row, 'mean_sad_deg'):.4f} degrees; "
        f"abundance RMSE={val(logdet_row, 'rmse'):.6f}",
        "",
        "## Interpretation",
        "- The parameter sweep is summarized from the recorded 36-run evaluation CSV.",
        "- Only two raw checkpoints are available for direct independent "
        "spectral and abundance-map comparisons.",
        "- In these two saved checkpoints, the control has slightly lower "
        "SAD and abundance RMSE than the lambda=0.001 checkpoint.",
        "- Normalized endmember curves compare spectral shape, not absolute magnitude.",
        "- These Samson experiments do not prove the sufficient-scattering "
        "identifiability theorem.",
        "",
        "## Generated graphs",
        "1. Graphs/01_mean_SAD_vs_logdet_weight.png",
        "2. Graphs/02_abundance_RMSE_vs_logdet_weight.png",
        "3. Graphs/03_endmember_spectra_vs_ground_truth.png",
        "4. Graphs/04_abundance_maps_vs_ground_truth.png",
        "",
        "## Metrics files",
        "- Metrics/samson_sad_rmse_per_run.csv",
        "- Metrics/samson_sad_rmse_summary_by_lambda.csv",
        "- Metrics/stage2_saved_checkpoint_metrics_recomputed.csv",
        "- Metrics/stage2_saved_checkpoint_summary.csv",
        "",
    ]

    summary_path = RESULTS / "stage2_results_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nStage 2 result generation completed successfully.")
    print("Recorded sweep runs:", len(per_run))
    print("Grouped configurations:", len(summary))
    print("Raw checkpoints used for direct plots: 2")
    print("Graphs folder:", GRAPHS)
    print("Summary:", summary_path)

    print("\nGenerated graph files:")
    for path in sorted(GRAPHS.glob("*.png")):
        print(f"{path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
