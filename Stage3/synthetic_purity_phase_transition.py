"""Synthetic purity phase-transition experiment using USGS mineral spectra."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from classical_unmixing_baselines import run_baseline
from dataset_loaders import load_usgs_library, resolve_repo_root
from evaluate_unmixing_benchmarks import compute_sad_matrix_deg
from logdet_volume_autoencoder import train_logdet_ae


def select_well_separated_spectra(
    library: np.ndarray, p: int, seed: int
) -> tuple[np.ndarray, list[int]]:
    """Greedy max-min spectral angle selection of p distinct USGS endmembers."""
    rng = np.random.default_rng(seed)
    n_mat = library.shape[1]
    if n_mat < p:
        raise ValueError(f"USGS library has only {n_mat} spectra, need {p}.")

    sad_all = compute_sad_matrix_deg(library, library)
    first = int(rng.integers(0, n_mat))
    selected = [first]
    for _ in range(1, p):
        min_ang = np.min(sad_all[selected, :], axis=0)
        min_ang[selected] = -1.0
        selected.append(int(np.argmax(min_ang)))
    return library[:, selected].copy(), selected


def sample_purity_constrained_abundances(
    n_pixels: int, p: int, max_purity: float, rng: np.random.Generator, alpha: float = 0.8
) -> np.ndarray:
    """Sample simplex abundances A (N, p) with max_k A_{ik} <= max_purity."""
    if max_purity <= 1.0 / p + 1e-3:
        raise ValueError(f"max_purity={max_purity} must exceed 1/p={1.0 / p:.3f}")

    if max_purity >= 0.999:
        a = rng.dirichlet(np.full(p, 0.35), size=n_pixels)
        for k in range(p):
            a[k * 5 : (k + 1) * 5, :] = 0.0
            a[k * 5 : (k + 1) * 5, k] = 1.0
        rng.shuffle(a, axis=0)
        return a

    collected = []
    total = 0
    batch = max(n_pixels * 4, 8000)
    while total < n_pixels:
        cand = rng.dirichlet(np.full(p, alpha), size=batch)
        keep = cand[np.max(cand, axis=1) <= max_purity]
        if keep.shape[0] > 0:
            collected.append(keep)
            total += keep.shape[0]
    return np.vstack(collected)[:n_pixels]


def add_gaussian_snr_noise(
    clean_x: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    if not np.isfinite(snr_db):
        return clean_x.copy()
    signal_power = float(np.mean(clean_x**2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=clean_x.shape)
    return np.clip(clean_x + noise, 0.0, None)


def score_estimate(
    gt_M: np.ndarray, gt_A: np.ndarray, est_E: np.ndarray, est_A: np.ndarray
) -> dict[str, float]:
    sad_mat = compute_sad_matrix_deg(gt_M, est_E)
    gt_idx, est_idx = linear_sum_assignment(sad_mat)
    sad_em = sad_mat[gt_idx, est_idx]
    mean_sad = float(np.mean(sad_em))
    max_sad = float(np.max(sad_em))

    est_A_matched = est_A[:, est_idx]
    rmse = float(np.sqrt(np.mean((est_A_matched - gt_A) ** 2)))

    pairwise = compute_sad_matrix_deg(est_E, est_E)
    min_angle = float(pairwise[~np.eye(gt_M.shape[1], dtype=bool)].min())
    return {
        "mean_sad_deg": mean_sad,
        "max_sad_deg": max_sad,
        "rmse": rmse,
        "min_pairwise_angle_deg": min_angle,
    }


def parse_args() -> argparse.Namespace:
    repo_root = resolve_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usgs", type=Path, default=None)
    parser.add_argument(
        "--purity-levels",
        type=float,
        nargs="+",
        default=[1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--n-pixels", type=int, default=2000)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--snr-db", type=float, default=30.0)
    parser.add_argument("--ae-epochs", type=int, default=1200)
    parser.add_argument("--ae-warmup", type=int, default=300)
    parser.add_argument("--ae-lr", type=float, default=5e-3)
    parser.add_argument("--ae-decoder-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ae", type=float, default=1e-3)
    parser.add_argument("--lambda-mvc", type=float, default=1e-3)
    parser.add_argument("--mvc-iters", type=int, default=200)
    parser.add_argument("--sad-success-deg", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "Stage 3 Results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    library, names, src_path = load_usgs_library(repo_root=repo_root, usgs_path=args.usgs)
    print(f"Loaded USGS library: {src_path} | shape={library.shape}")

    per_trial_rows: list[dict[str, Any]] = []

    for purity in args.purity_levels:
        for trial in range(args.trials):
            seed = 100 * trial + int(round(purity * 100))
            rng = np.random.default_rng(seed)

            gt_M, chosen_idx = select_well_separated_spectra(library, args.components, seed=trial)
            gt_A = sample_purity_constrained_abundances(args.n_pixels, args.components, purity, rng)
            empirical_max_purity = float(np.max(gt_A))
            clean_x = gt_A @ gt_M.T
            noisy_x = add_gaussian_snr_noise(clean_x, args.snr_db, rng)

            print(
                f"Purity={purity:.2f} (emp={empirical_max_purity:.3f}) | "
                f"Trial {trial + 1}/{args.trials} | endmembers={[names[i] for i in chosen_idx]}"
            )

            # 1. Classical baselines
            vca_e_scaled = None
            for b_method in ("vca", "spa", "nfindr", "mvc_nmf"):
                res = run_baseline(
                    b_method,
                    noisy_x,
                    args.components,
                    seed=seed,
                    lambda_vol=args.lambda_mvc,
                    mvc_iters=args.mvc_iters,
                    device=args.device,
                )
                if b_method == "vca":
                    vca_e_scaled = res["e_scaled"].numpy().astype(np.float64)
                metrics = score_estimate(
                    gt_M,
                    gt_A,
                    res["e"].numpy().astype(np.float64),
                    res["a"].numpy().astype(np.float64),
                )
                per_trial_rows.append(
                    {
                        "purity_cap": purity,
                        "empirical_max_purity": empirical_max_purity,
                        "trial": trial,
                        "snr_db": args.snr_db,
                        "method": res["method"],
                        **metrics,
                        "identifiable_5deg": int(metrics["mean_sad_deg"] <= args.sad_success_deg),
                        "identifiable_3deg": int(metrics["mean_sad_deg"] <= 3.0),
                    }
                )

            # 2. Autoencoder control (lambda=0) vs Log-Det Volume AE
            for label, lam in (("AE (lambda=0)", 0.0), ("LogDet-AE", args.lambda_ae)):
                payload, _ = train_logdet_ae(
                    noisy_x,
                    args.components,
                    init="vca",
                    init_endmembers_scaled=vca_e_scaled,
                    use_encoder_bn=True,
                    epochs=args.ae_epochs,
                    warmup_epochs=args.ae_warmup,
                    freeze_decoder_warmup=True,
                    lr=args.ae_lr,
                    decoder_lr=args.ae_decoder_lr,
                    lambda_vol=lam,
                    vol_eps=1e-4,
                    input_scale=None,
                    batch_size=0,
                    seed=seed,
                    device=args.device,
                    verbose=False,
                )
                metrics = score_estimate(
                    gt_M,
                    gt_A,
                    payload["e"].numpy().astype(np.float64),
                    payload["a"].numpy().astype(np.float64),
                )
                per_trial_rows.append(
                    {
                        "purity_cap": purity,
                        "empirical_max_purity": empirical_max_purity,
                        "trial": trial,
                        "snr_db": args.snr_db,
                        "method": label,
                        **metrics,
                        "identifiable_5deg": int(metrics["mean_sad_deg"] <= args.sad_success_deg),
                        "identifiable_3deg": int(metrics["mean_sad_deg"] <= 3.0),
                    }
                )

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_trial_rows:
        grouped[(float(row["purity_cap"]), str(row["method"]))].append(row)

    phase_rows: list[dict[str, Any]] = []
    for (purity, method), items in sorted(grouped.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        sads = np.array([x["mean_sad_deg"] for x in items], dtype=np.float64)
        rmses = np.array([x["rmse"] for x in items], dtype=np.float64)
        succ5 = np.array([x["identifiable_5deg"] for x in items], dtype=np.float64)
        succ3 = np.array([x["identifiable_3deg"] for x in items], dtype=np.float64)
        ddof = 1 if len(items) > 1 else 0
        phase_rows.append(
            {
                "purity_cap": purity,
                "method": method,
                "n_trials": len(items),
                "mean_sad_mean_deg": float(np.mean(sads)),
                "mean_sad_std_deg": float(np.std(sads, ddof=ddof)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses, ddof=ddof)),
                "success_rate_5deg": float(np.mean(succ5)),
                "success_rate_3deg": float(np.mean(succ3)),
            }
        )

    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    trial_csv = results_dir / "synthetic_purity_per_trial.csv"
    with trial_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_trial_rows)

    phase_csv = results_dir / "synthetic_purity_phase_transition.csv"
    with phase_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phase_rows)

    print("\n=== Synthetic Purity Phase Transition (Mean SAD & Success Rate <= 5 deg) ===")
    print(f"{'Purity':>6} {'Method':<14} | {'Mean SAD (deg)':>17} | {'RMSE':>15} | {'Success(<=5°)':>13}")
    print("-" * 76)
    for r in phase_rows:
        print(
            f"{r['purity_cap']:>6.2f} {r['method']:<14} | "
            f"{r['mean_sad_mean_deg']:>6.2f} +/- {r['mean_sad_std_deg']:<6.2f} | "
            f"{r['rmse_mean']:>6.4f} +/- {r['rmse_std']:<6.4f} | "
            f"{100.0 * r['success_rate_5deg']:>11.1f}%"
        )

    print(f"\nSaved per-trial CSV:        {trial_csv}")
    print(f"Saved phase-transition CSV: {phase_csv}")


if __name__ == "__main__":
    main()