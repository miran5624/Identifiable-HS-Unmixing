"""Evaluate Stage 3 autoencoder and classical baseline runs on real hyperspectral benchmarks."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dataset_loaders import BenchmarkDataset, load_benchmark_dataset, resolve_repo_root


def compute_sad_matrix_deg(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    a_norm = mat_a / np.maximum(np.linalg.norm(mat_a, axis=0, keepdims=True), 1e-12)
    b_norm = mat_b / np.maximum(np.linalg.norm(mat_b, axis=0, keepdims=True), 1e-12)
    cos_sim = np.clip(a_norm.T @ b_norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


def gt_abundance_layouts(gt_A: np.ndarray, h: int, w: int) -> dict[str, np.ndarray]:
    p = gt_A.shape[0]
    return {
        "direct_T": gt_A.T,
        "fortran_to_c": gt_A.reshape(p, h, w, order="F").reshape(p, -1).T,
    }


def evaluate_file(
    pt_path: Path, ds: BenchmarkDataset, gt_layout: str
) -> tuple[dict[str, Any], np.ndarray]:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    est_E = np.asarray(data["e"].detach().cpu().numpy(), dtype=np.float64)
    est_A = np.asarray(data["a"].detach().cpu().numpy(), dtype=np.float64)

    h, w, bands = ds.cube.shape
    p = ds.gt_M.shape[1]
    if est_E.shape != (bands, p):
        raise ValueError(f"{pt_path.name}: expected E {(bands, p)}, got {est_E.shape}")
    if est_A.shape != (h * w, p):
        raise ValueError(f"{pt_path.name}: expected A {(h * w, p)}, got {est_A.shape}")

    sad_mat = compute_sad_matrix_deg(ds.gt_M, est_E)
    gt_idx, est_idx = linear_sum_assignment(sad_mat)
    sad_per_em = sad_mat[gt_idx, est_idx]
    mean_sad = float(np.mean(sad_per_em))
    max_sad = float(np.max(sad_per_em))

    pairwise = compute_sad_matrix_deg(est_E, est_E)
    min_angle = float(pairwise[~np.eye(p, dtype=bool)].min())

    est_A_matched = est_A[:, est_idx]
    if ds.gt_A is not None:
        layouts = gt_abundance_layouts(ds.gt_A, h, w)
        rmse_direct = float(np.sqrt(np.mean((est_A_matched - layouts["direct_T"]) ** 2)))
        rmse_fortran = float(np.sqrt(np.mean((est_A_matched - layouts["fortran_to_c"]) ** 2)))
        rmse = rmse_fortran if gt_layout == "fortran_to_c" else rmse_direct
    else:
        rmse_direct = float("nan")
        rmse_fortran = float("nan")
        rmse = float("nan")

    settings = data.get("settings", {})
    method = str(data.get("method", settings.get("method", "unknown")))
    lam = float(settings.get("lambda_vol", 0.0))
    warmup = int(settings.get("warmup_epochs", 0))
    seed = int(settings.get("seed", -1))

    row = {
        "dataset": ds.name,
        "method": method,
        "file": pt_path.name,
        "warmup_epochs": warmup,
        "lambda_vol": lam,
        "seed": seed,
        "negative_samples_clipped": settings.get("negative_samples_clipped"),
        "gt_layout": gt_layout if ds.gt_A is not None else "no_gt_A",
        "mean_sad_deg": mean_sad,
        "max_sad_deg": max_sad,
        "min_pairwise_angle_deg": min_angle,
        "rmse_direct_T": rmse_direct,
        "rmse_fortran_to_c": rmse_fortran,
        "rmse": rmse,
        "final_scaled_mse": float(data.get("final_scaled_mse", float("nan"))),
        "final_vol_loss": float(data.get("final_vol_loss", float("nan"))),
    }
    return row, est_A_matched


def save_layout_check_figure(
    out_path: Path, row: dict[str, Any], est_A_matched: np.ndarray, ds: BenchmarkDataset
) -> None:
    if ds.gt_A is None:
        return
    import matplotlib.pyplot as plt

    h, w, _ = ds.cube.shape
    p = ds.gt_M.shape[1]
    n_cols = min(p, 6)
    layouts = gt_abundance_layouts(ds.gt_A, h, w)

    est_maps = est_A_matched.reshape(h, w, p)
    direct_maps = layouts["direct_T"].reshape(h, w, p)
    fortran_maps = layouts["fortran_to_c"].reshape(h, w, p)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.6 * n_cols, 7.8), constrained_layout=True)
    specs = [
        ("Estimated (matched)", est_maps),
        (f"GT direct_T (RMSE={row['rmse_direct_T']:.4f})", direct_maps),
        (f"GT fortran_to_c (RMSE={row['rmse_fortran_to_c']:.4f})", fortran_maps),
    ]
    for r_idx, (title, maps) in enumerate(specs):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            ax.imshow(maps[:, :, c_idx], cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"EM {c_idx}")
            if c_idx == 0:
                ax.set_ylabel(title, fontsize=9)

    fig.suptitle(f"{ds.name} Layout Check: {row['file']}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved layout check: {out_path}")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["method"], int(r["warmup_epochs"]), float(r["lambda_vol"]))].append(r)

    summary = []
    for (dataset, method, wu, lam), items in sorted(grouped.items()):
        sads = np.array([x["mean_sad_deg"] for x in items], dtype=np.float64)
        rmses = np.array([x["rmse"] for x in items], dtype=np.float64)
        angs = np.array([x["min_pairwise_angle_deg"] for x in items], dtype=np.float64)
        ddof = 1 if len(items) > 1 else 0
        valid_rmse = rmses[np.isfinite(rmses)]
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "warmup_epochs": wu,
                "lambda_vol": lam,
                "n_runs": len(items),
                "mean_sad_mean_deg": float(np.mean(sads)),
                "mean_sad_std_deg": float(np.std(sads, ddof=ddof)),
                "rmse_mean": float(np.mean(valid_rmse)) if valid_rmse.size else float("nan"),
                "rmse_std": float(np.std(valid_rmse, ddof=1 if valid_rmse.size > 1 else 0)) if valid_rmse.size else float("nan"),
                "min_angle_mean_deg": float(np.mean(angs)),
                "min_angle_min_deg": float(np.min(angs)),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = resolve_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("jasper_ridge", "urban", "cuprite", "samson"),
        default=["jasper_ridge", "urban", "cuprite"],
    )
    parser.add_argument("--runs-dir", type=Path, default=script_dir)
    parser.add_argument("--gt-layout", choices=("fortran_to_c", "direct_T"), default="fortran_to_c")
    parser.add_argument("--save-layout-checks", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "Stage 3 Results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    runs_dir = args.runs_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for ds_name in args.datasets:
        files = sorted(
            list(runs_dir.glob(f"{ds_name}_unmixing_*.pt"))
            + list(runs_dir.glob(f"{ds_name}_baseline_*.pt"))
        )
        if not files:
            print(f"Skipping {ds_name}: no matching .pt files in {runs_dir}")
            continue

        ds = load_benchmark_dataset(ds_name, repo_root=repo_root)
        evaluated = [evaluate_file(f, ds, args.gt_layout) for f in files]
        all_rows.extend(item[0] for item in evaluated)

        if args.save_layout_checks and ds.gt_A is not None:
            best_item = min(evaluated, key=lambda x: x[0]["mean_sad_deg"])
            save_layout_check_figure(
                results_dir / f"{ds_name}_abundance_layout_check.png",
                best_item[0],
                best_item[1],
                ds,
            )

    if not all_rows:
        raise FileNotFoundError(f"No Stage 3 run files found in {runs_dir}")

    summary_rows = aggregate_rows(all_rows)

    header = (
        f"{'Dataset':<13} {'Method':<13} {'Lam':>7} {'N':>2} | "
        f"{'Mean SAD (deg)':>18} | {'Abundance RMSE':>18} | {'MinAng':>7}"
    )
    print("\n=== Stage 3 Real-Dataset Method Comparison ===")
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        sad_str = f"{s['mean_sad_mean_deg']:.2f} +/- {s['mean_sad_std_deg']:.2f}"
        rmse_str = (
            f"{s['rmse_mean']:.4f} +/- {s['rmse_std']:.4f}"
            if np.isfinite(s["rmse_mean"])
            else "N/A (no GT A)"
        )
        print(
            f"{s['dataset']:<13} {s['method']:<13} {s['lambda_vol']:>7.1e} {s['n_runs']:>2} | "
            f"{sad_str:>18} | {rmse_str:>18} | {s['min_angle_mean_deg']:>7.2f}"
        )

    per_run_csv = results_dir / "real_datasets_per_run_metrics.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_csv = results_dir / "real_datasets_method_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved per-run CSV:    {per_run_csv}")
    print(f"Saved comparison CSV: {summary_csv}")


if __name__ == "__main__":
    main()
