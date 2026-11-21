"""Evaluate Samson unmixing outputs against ground truth (samgt.mat)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from scipy.optimize import linear_sum_assignment

# Lock this once after visually verifying one good lambda=0 abundance map
# using --save-layout-check. Valid values: "fortran_to_c" or "direct_T".
DEFAULT_GT_LAYOUT = "fortran_to_c"


def find_gt(gt_path: Path | None, repo_root: Path) -> Path:
    """Resolve an explicit samgt.mat path or locate it under repo_root."""
    if gt_path is not None:
        candidate = gt_path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Ground truth file not found: {candidate}")
        return candidate

    matches = sorted(repo_root.rglob("samgt.mat"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find samgt.mat under {repo_root}. Pass it with --gt."
        )
    if len(matches) > 1:
        locations = "\n".join(str(p) for p in matches)
        raise RuntimeError(
            f"Found multiple samgt.mat files; select one with --gt:\n{locations}"
        )
    return matches[0]


def load_ground_truth(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate M (156, 3) and A (3, 9025) from samgt.mat."""
    mat = sio.loadmat(str(gt_path))
    if "M" not in mat or "A" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Expected keys 'M' and 'A' in {gt_path}, found: {keys}")

    M = np.asarray(mat["M"], dtype=np.float64)
    A = np.asarray(mat["A"], dtype=np.float64)

    if M.shape != (156, 3):
        raise ValueError(f"Expected ground-truth M shape (156, 3), got {M.shape}")
    if A.shape != (3, 9025):
        raise ValueError(f"Expected ground-truth A shape (3, 9025), got {A.shape}")
    return M, A


def gt_abundance_layouts(gt_A: np.ndarray) -> dict[str, np.ndarray]:
    """Return both candidate (9025, 3) pixel-order layouts for ground-truth A."""
    return {
        "direct_T": gt_A.T,
        "fortran_to_c": gt_A.reshape(3, 95, 95, order="F").reshape(3, -1).T,
    }


def compute_sad_matrix_deg(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute (p, p) pairwise Spectral Angle Distance matrix in degrees."""
    a_norm = mat_a / np.maximum(np.linalg.norm(mat_a, axis=0, keepdims=True), 1e-12)
    b_norm = mat_b / np.maximum(np.linalg.norm(mat_b, axis=0, keepdims=True), 1e-12)
    cos_sim = np.clip(a_norm.T @ b_norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


def evaluate_run(
    unmixing_path: Path,
    gt_M: np.ndarray,
    gt_layouts: dict[str, np.ndarray],
    gt_layout: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Match estimated endmembers to M via Hungarian SAD and score under fixed gt_layout."""
    data = torch.load(unmixing_path, map_location="cpu", weights_only=False)
    est_E = np.asarray(data["e"].detach().cpu().numpy(), dtype=np.float64)
    est_A = np.asarray(data["a"].detach().cpu().numpy(), dtype=np.float64)

    if est_E.shape != (156, 3):
        raise ValueError(f"{unmixing_path.name}: expected e shape (156, 3), got {est_E.shape}")
    if est_A.shape != (9025, 3):
        raise ValueError(f"{unmixing_path.name}: expected a shape (9025, 3), got {est_A.shape}")

    sad_matrix = compute_sad_matrix_deg(gt_M, est_E)
    gt_idx, est_idx = linear_sum_assignment(sad_matrix)
    sad_per_endmember = sad_matrix[gt_idx, est_idx]
    mean_sad_deg = float(np.mean(sad_per_endmember))

    # Pairwise angles among estimated endmembers to detect volume collapse
    est_pairwise = compute_sad_matrix_deg(est_E, est_E)
    off_diag = ~np.eye(3, dtype=bool)
    min_pairwise_sad_deg = float(est_pairwise[off_diag].min())

    # Reorder estimated abundances to align with ground-truth endmember order
    est_A_matched = est_A[:, est_idx]

    rmse_direct = float(np.sqrt(np.mean((est_A_matched - gt_layouts["direct_T"]) ** 2)))
    rmse_fortran = float(np.sqrt(np.mean((est_A_matched - gt_layouts["fortran_to_c"]) ** 2)))

    fixed_gt_A = gt_layouts[gt_layout]
    fixed_rmse = rmse_fortran if gt_layout == "fortran_to_c" else rmse_direct
    rmse_per_endmember = np.sqrt(np.mean((est_A_matched - fixed_gt_A) ** 2, axis=0))

    settings = data.get("settings", {})
    row = {
        "file": unmixing_path.name,
        "init": settings.get("init", "unknown"),
        "warmup_epochs": int(settings.get("warmup_epochs", 0)),
        "seed": int(settings.get("seed", -1)),
        "lambda_vol": float(settings.get("lambda_vol", float("nan"))),
        "gt_layout": gt_layout,
        "perm_est_to_gt": est_idx.tolist(),
        "sad_e0_deg": float(sad_per_endmember[0]),
        "sad_e1_deg": float(sad_per_endmember[1]),
        "sad_e2_deg": float(sad_per_endmember[2]),
        "mean_sad_deg": mean_sad_deg,
        "min_pairwise_sad_deg": min_pairwise_sad_deg,
        "rmse_direct_T": rmse_direct,
        "rmse_fortran_to_c": rmse_fortran,
        "rmse": fixed_rmse,
        "rmse_e0": float(rmse_per_endmember[0]),
        "rmse_e1": float(rmse_per_endmember[1]),
        "rmse_e2": float(rmse_per_endmember[2]),
        "final_scaled_mse": float(data.get("final_scaled_mse", float("nan"))),
        "final_vol_loss": float(data.get("final_vol_loss", float("nan"))),
    }
    return row, est_A_matched


def save_layout_check_figure(
    out_path: Path,
    run_row: dict[str, Any],
    est_A_matched: np.ndarray,
    gt_layouts: dict[str, np.ndarray],
) -> None:
    """Plot estimated vs direct_T vs fortran_to_c abundance maps to verify layout by eye."""
    import matplotlib.pyplot as plt

    est_maps = est_A_matched.reshape(95, 95, 3)
    direct_maps = gt_layouts["direct_T"].reshape(95, 95, 3)
    fortran_maps = gt_layouts["fortran_to_c"].reshape(95, 95, 3)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    row_specs = [
        ("Estimated (matched)", est_maps),
        (f"GT direct_T (RMSE={run_row['rmse_direct_T']:.4f})", direct_maps),
        (f"GT fortran_to_c (RMSE={run_row['rmse_fortran_to_c']:.4f})", fortran_maps),
    ]
    for r_idx, (row_title, maps) in enumerate(row_specs):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            ax.imshow(maps[:, :, c_idx], cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"Endmember {c_idx}")
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=9)

    fig.suptitle(f"Abundance Layout Verification: {run_row['file']}", fontsize=11)
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved layout verification plot to: {out_path}")


def aggregate_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate mean +/- std across seeds grouped by (init, warmup_epochs, lambda_vol)."""
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (str(r["init"]), int(r["warmup_epochs"]), float(r["lambda_vol"]))
        grouped[key].append(r)

    summary_rows: list[dict[str, Any]] = []
    for (init_name, warmup, lam), items in sorted(grouped.items()):
        sads = np.array([x["mean_sad_deg"] for x in items], dtype=np.float64)
        rmses = np.array([x["rmse"] for x in items], dtype=np.float64)
        min_angles = np.array([x["min_pairwise_sad_deg"] for x in items], dtype=np.float64)
        ddof = 1 if len(items) > 1 else 0

        summary_rows.append(
            {
                "init": init_name,
                "warmup_epochs": warmup,
                "lambda_vol": lam,
                "n_seeds": len(items),
                "gt_layout": items[0]["gt_layout"],
                "mean_sad_mean_deg": float(np.mean(sads)),
                "mean_sad_std_deg": float(np.std(sads, ddof=ddof)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses, ddof=ddof)),
                "min_angle_mean_deg": float(np.mean(min_angles)),
                "min_angle_min_deg": float(np.min(min_angles)),
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        help="One or more samson_unmixing_*.pt files to evaluate.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir,
        help="Directory to scan for samson_unmixing_*.pt when --results is omitted (default: Stage2).",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        help="Path to samgt.mat (searched automatically from repo root if omitted).",
    )
    parser.add_argument(
        "--gt-layout",
        choices=("fortran_to_c", "direct_T"),
        default=DEFAULT_GT_LAYOUT,
        help=f"Fixed ground-truth abundance layout applied to all runs (default: {DEFAULT_GT_LAYOUT}).",
    )
    parser.add_argument(
        "--save-layout-check",
        type=Path,
        help="Optional PNG path comparing a lambda=0 run's abundance maps against both GT layouts.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write per-run SAD and RMSE metrics as a CSV table.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional path to write grouped (init, warmup, lambda) mean +/- std metrics as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    gt_path = find_gt(args.gt, repo_root)
    gt_M, gt_A = load_ground_truth(gt_path)
    gt_layouts = gt_abundance_layouts(gt_A)

    if args.results:
        files = [p.expanduser().resolve() for p in args.results]
    else:
        files = sorted(args.results_dir.expanduser().resolve().glob("samson_unmixing_*.pt"))

    if not files:
        raise FileNotFoundError("No samson_unmixing_*.pt files found to evaluate.")

    evaluated = [evaluate_run(path, gt_M, gt_layouts, args.gt_layout) for path in files]
    evaluated.sort(
        key=lambda item: (
            str(item[0]["init"]),
            int(item[0]["warmup_epochs"]),
            float(item[0]["lambda_vol"]),
            int(item[0]["seed"]),
        )
    )
    rows = [item[0] for item in evaluated]

    if args.save_layout_check is not None:
        lam0_items = [item for item in evaluated if item[0]["lambda_vol"] == 0.0]
        pool = lam0_items if lam0_items else evaluated
        best_item = min(pool, key=lambda item: item[0]["mean_sad_deg"])
        save_layout_check_figure(args.save_layout_check, best_item[0], best_item[1], gt_layouts)

    print(f"Ground truth: {gt_path}")
    print(f"Fixed GT layout for all RMSE reporting: {args.gt_layout}\n")

    header = (
        f"{'Init':<6} {'WU':>3} {'Seed':>4} {'Lambda':>8} | "
        f"{'SAD0':>6} {'SAD1':>6} {'SAD2':>6} {'MeanSAD':>8} {'MinAng':>7} | "
        f"{'RMSE(T)':>8} {'RMSE(F)':>8} {'RMSE':>8} | {'Perm':<9}"
    )
    print("=== Per-Run Results ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['init']):<6} {int(r['warmup_epochs']):>3} {int(r['seed']):>4} "
            f"{float(r['lambda_vol']):>8.1e} | "
            f"{r['sad_e0_deg']:>6.2f} {r['sad_e1_deg']:>6.2f} {r['sad_e2_deg']:>6.2f} "
            f"{r['mean_sad_deg']:>8.2f} {r['min_pairwise_sad_deg']:>7.2f} | "
            f"{r['rmse_direct_T']:>8.4f} {r['rmse_fortran_to_c']:>8.4f} {r['rmse']:>8.4f} | "
            f"{str(r['perm_est_to_gt']):<9}"
        )

    summary_rows = aggregate_by_group(rows)
    sum_header = (
        f"{'Init':<6} {'WU':>3} {'Lambda':>8} {'N':>2} | "
        f"{'Mean SAD (deg)':>19} | {'Abundance RMSE':>19} | {'MinAngle (mean / min)':>21}"
    )
    print(f"\n=== Grouped Report Table (fixed layout: {args.gt_layout}) ===")
    print(sum_header)
    print("-" * len(sum_header))
    for s in summary_rows:
        sad_str = f"{s['mean_sad_mean_deg']:.2f} +/- {s['mean_sad_std_deg']:.2f}"
        rmse_str = f"{s['rmse_mean']:.4f} +/- {s['rmse_std']:.4f}"
        ang_str = f"{s['min_angle_mean_deg']:.2f} / {s['min_angle_min_deg']:.2f}"
        print(
            f"{str(s['init']):<6} {int(s['warmup_epochs']):>3} {float(s['lambda_vol']):>8.1e} "
            f"{int(s['n_seeds']):>2} | {sad_str:>19} | {rmse_str:>19} | {ang_str:>21}"
        )

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved per-run CSV to:  {csv_path}")

    if args.summary_csv is not None:
        sum_csv_path = args.summary_csv.expanduser().resolve()
        sum_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with sum_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved grouped CSV to:  {sum_csv_path}")


if __name__ == "__main__":
    main()