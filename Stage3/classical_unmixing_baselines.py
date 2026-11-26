"""Classical hyperspectral unmixing baselines: VCA, SPA, N-FINDR, and MVC-NMF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset_loaders import clip_negative_reflectance, load_benchmark_dataset, resolve_repo_root
from logdet_volume_autoencoder import log_det_volume_loss, per_pixel_sse_loss, set_deterministic


def project_simplex_rows(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of each row of v (N, p) onto the probability simplex."""
    u = np.sort(v, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, v.shape[1] + 1, dtype=v.dtype)
    cond = u - cssv / ind > 0.0
    rho = np.sum(cond, axis=1) - 1
    theta = cssv[np.arange(v.shape[0]), rho] / (rho + 1.0)
    return np.maximum(v - theta[:, None], 0.0)


def solve_fcls(
    x: np.ndarray, e: np.ndarray, max_iter: int = 250, tol: float = 1e-6, init_a: np.ndarray | None = None
) -> np.ndarray:
    """Fast vectorized Fully Constrained Least Squares (FISTA + exact simplex projection).

    Args:
        x: (N, B) reflectance matrix
        e: (B, p) endmember matrix
    Returns:
        a: (N, p) simplex-constrained abundance matrix
    """
    x64 = np.asarray(x, dtype=np.float64)
    e64 = np.asarray(e, dtype=np.float64)
    n_pixels, p = x64.shape[0], e64.shape[1]

    ete = e64.T @ e64
    xe = x64 @ e64
    lip = float(np.linalg.eigvalsh(ete).max()) + 1e-12
    step = 1.0 / lip

    a = np.full((n_pixels, p), 1.0 / p, dtype=np.float64) if init_a is None else init_a.astype(np.float64).copy()
    y = a.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = y @ ete - xe
        a_next = project_simplex_rows(y - step * grad)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = a_next + ((t - 1.0) / t_next) * (a_next - a)
        if np.max(np.abs(a_next - a)) < tol:
            a = a_next
            break
        a = a_next
        t = t_next

    return a


def run_spa(x: np.ndarray, p: int) -> np.ndarray:
    """Successive Projection Algorithm (Gillis & Vavasis 2014). Returns E of shape (B, p)."""
    y = np.asarray(x, dtype=np.float64).T  # (B, N)
    r = y.copy()
    indices: list[int] = []

    for _ in range(p):
        norms = np.einsum("ij,ij->j", r, r)
        if indices:
            norms[indices] = -1.0
        idx = int(np.argmax(norms))
        indices.append(idx)
        u = r[:, idx]
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1e-12:
            u = u / u_norm
            r = r - np.outer(u, u @ r)

    return np.maximum(y[:, indices], 0.0)


def run_vca(x: np.ndarray, p: int, seed: int = 0) -> np.ndarray:
    """Vertex Component Analysis (Nascimento & Dias 2005). Returns E of shape (B, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float64).T  # (B, N)
    bands, n_pixels = y.shape

    r_m = y.mean(axis=1, keepdims=True)
    r_o = y - r_m
    cov = (r_o @ r_o.T) / n_pixels
    eigvals, eigvecs = np.linalg.eigh(cov)
    ud = eigvecs[:, ::-1][:, :p]  # (B, p)
    x_p = ud.T @ r_o

    p_y = float(np.sum(y * y) / n_pixels)
    p_x = float(np.sum(x_p * x_p) / n_pixels + np.sum(r_m * r_m))
    denom = max(p_y - p_x, 1e-12)
    snr = 10.0 * np.log10(max((p_x - (p / bands) * p_y) / denom, 1e-12))
    snr_th = 15.0 + 10.0 * np.log10(p)

    if snr < snr_th:
        d = p - 1
        ud_sub = ud[:, :d]
        rp = ud_sub @ (ud_sub.T @ r_o) + r_m
        x_sub = ud_sub.T @ r_o
        c = float(np.sqrt(np.max(np.sum(x_sub * x_sub, axis=0))))
        y_proj = np.vstack([x_sub, np.full((1, n_pixels), c)])
    else:
        gram = (y @ y.T) / n_pixels
        _, evecs_g = np.linalg.eigh(gram)
        ud_sub = evecs_g[:, ::-1][:, :p]
        x_sub = ud_sub.T @ y
        u_mean = x_sub.mean(axis=1, keepdims=True)
        denom_proj = np.sum(x_sub * u_mean, axis=0, keepdims=True)
        denom_proj = np.where(np.abs(denom_proj) < 1e-12, 1e-12, denom_proj)
        y_proj = x_sub / denom_proj

    indices = np.zeros(p, dtype=int)
    a_mat = np.zeros((p, p), dtype=np.float64)
    a_mat[p - 1, 0] = 1.0

    for i in range(p):
        w = rng.standard_normal((p, 1))
        if i > 0:
            a_pinv = np.linalg.pinv(a_mat[:, :i])
            f = w - a_mat[:, :i] @ (a_pinv @ w)
        else:
            f = w
        f_norm = float(np.linalg.norm(f))
        f = f / max(f_norm, 1e-12)
        v = np.abs((f.T @ y_proj).ravel())
        if i > 0:
            v[indices[:i]] = -1.0
        idx = int(np.argmax(v))
        indices[i] = idx
        a_mat[:, i] = y_proj[:, idx]

    return np.maximum(y[:, indices], 0.0)


def run_nfindr(x: np.ndarray, p: int, seed: int = 0, max_passes: int = 5) -> np.ndarray:
    """Fast vectorized N-FINDR (Winter 1999) via matrix-determinant lemma. Returns E (B, p)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float64).T  # (B, N)
    n_pixels = y.shape[1]

    r_o = y - y.mean(axis=1, keepdims=True)
    cov = (r_o @ r_o.T) / n_pixels
    _, eigvecs = np.linalg.eigh(cov)
    ud = eigvecs[:, ::-1][:, : p - 1]
    y_pca = ud.T @ r_o  # (p - 1, N)
    y_aug = np.vstack([np.ones((1, n_pixels), dtype=np.float64), y_pca])  # (p, N)

    indices = rng.choice(n_pixels, size=p, replace=False)
    e_aug = y_aug[:, indices].copy()

    for _ in range(max_passes):
        changed = False
        for k in range(p):
            try:
                inv_row = np.linalg.inv(e_aug)[k, :]
            except np.linalg.LinAlgError:
                inv_row = np.linalg.pinv(e_aug)[k, :]
            scores = np.abs(inv_row @ y_aug)
            # Avoid picking an already chosen pixel in another slot
            other = [indices[j] for j in range(p) if j != k]
            scores[other] = -1.0
            best_idx = int(np.argmax(scores))
            if best_idx != indices[k] and scores[best_idx] > 1.0 + 1e-9:
                indices[k] = best_idx
                e_aug[:, k] = y_aug[:, best_idx]
                changed = True
        if not changed:
            break

    return np.maximum(y[:, indices], 0.0)


def run_mvc_nmf(
    x_scaled: np.ndarray,
    p: int,
    *,
    lambda_vol: float = 1e-3,
    vol_eps: float = 1e-4,
    outer_iters: int = 250,
    lr: float = 5e-3,
    seed: int = 0,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Minimum-Volume Constrained NMF alternating FCLS on A and log-det volume update on E >= 0."""
    set_deterministic(seed)
    dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )

    x_np = np.asarray(x_scaled, dtype=np.float64)
    n_pixels, bands = x_np.shape

    # Standard initialization via VCA (fallback to SPA if degenerate)
    e_init = run_vca(x_np, p, seed=seed)
    if np.linalg.matrix_rank(e_init) < p:
        e_init = run_spa(x_np, p)

    a_np = solve_fcls(x_np, e_init, max_iter=80)

    x_t = torch.from_numpy(x_np.astype(np.float32)).to(dev)
    e_param = torch.nn.Parameter(torch.from_numpy(e_init.astype(np.float32)).to(dev))
    optimizer = torch.optim.Adam([e_param], lr=lr)

    recon_mse_history: list[float] = []
    vol_loss_history: list[float] = []

    for it in range(outer_iters):
        # 1. Warm-started FCLS update on abundances A
        e_curr = e_param.detach().cpu().numpy().astype(np.float64)
        a_np = solve_fcls(x_np, e_curr, max_iter=15, init_a=a_np)
        a_t = torch.from_numpy(a_np.astype(np.float32)).to(dev)

        # 2. Volume-regularized projected gradient steps on endmembers E >= 0
        for _ in range(5):
            optimizer.zero_grad()
            x_hat = a_t @ e_param.T
            recon_sse = per_pixel_sse_loss(x_hat, x_t)
            vol_loss = log_det_volume_loss(e_param, eps=vol_eps)
            loss = recon_sse + lambda_vol * vol_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                e_param.clamp_(min=0.0)

        recon_mse_history.append(float(recon_sse.detach().item()) / bands)
        vol_loss_history.append(float(vol_loss.detach().item()))

    e_final = e_param.detach().cpu().numpy().astype(np.float64)
    a_final = solve_fcls(x_np, e_final, max_iter=200, init_a=a_np)
    return e_final, a_final, recon_mse_history, vol_loss_history


def run_baseline(
    method: str,
    x_original: np.ndarray,
    components: int,
    *,
    seed: int = 0,
    lambda_vol: float = 1e-3,
    mvc_iters: int = 250,
    device: str = "auto",
) -> dict[str, Any]:
    """Unified entry point for all classical baselines; returns the same dict schema as LogDet-AE."""
    x_np = np.asarray(x_original, dtype=np.float64)
    if x_np.ndim != 2 or not np.isfinite(x_np).all():
        raise ValueError("Input spectra must be a finite 2D array.")
    x_np, negative_samples_clipped = clip_negative_reflectance(x_np)
    x_np = np.asarray(x_np, dtype=np.float64)
    scale = float(np.max(x_np))
    if scale <= 0.0:
        raise ValueError("Input spectra must contain at least one positive reflectance value.")
    x_scaled = x_np / scale
    bands = x_np.shape[1]

    method_low = method.lower().strip()
    recon_hist: list[float] = []
    vol_hist: list[float] = []

    if method_low == "vca":
        e_scaled = run_vca(x_scaled, components, seed=seed)
        a = solve_fcls(x_scaled, e_scaled)
        display_name = "VCA"
    elif method_low == "spa":
        e_scaled = run_spa(x_scaled, components)
        a = solve_fcls(x_scaled, e_scaled)
        display_name = "SPA"
    elif method_low == "nfindr":
        e_scaled = run_nfindr(x_scaled, components, seed=seed)
        a = solve_fcls(x_scaled, e_scaled)
        display_name = "N-FINDR"
    elif method_low == "mvc_nmf":
        e_scaled, a, recon_hist, vol_hist = run_mvc_nmf(
            x_scaled,
            components,
            lambda_vol=lambda_vol,
            outer_iters=mvc_iters,
            seed=seed,
            device=device,
        )
        display_name = "MVC-NMF"
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    x_hat_scaled = a @ e_scaled.T
    final_scaled_sse = float(np.mean(np.sum((x_hat_scaled - x_scaled) ** 2, axis=1)))
    final_scaled_mse = final_scaled_sse / bands

    e_tensor = torch.from_numpy((e_scaled * scale).astype(np.float32))
    e_scaled_t = torch.from_numpy(e_scaled.astype(np.float32))
    a_tensor = torch.from_numpy(a.astype(np.float32))
    final_vol = float(log_det_volume_loss(e_scaled_t, eps=1e-4).item())

    return {
        "method": display_name,
        "a": a_tensor,
        "e": e_tensor,
        "e_scaled": e_scaled_t,
        "final_loss": final_scaled_mse * (scale**2),
        "final_scaled_mse": final_scaled_mse,
        "final_scaled_sse": final_scaled_sse,
        "final_vol_loss": final_vol,
        "recon_mse_history": recon_hist,
        "vol_loss_history": vol_hist,
        "settings": {
            "method": method_low,
            "seed": seed,
            "lambda_vol": lambda_vol if method_low == "mvc_nmf" else 0.0,
            "input_scale": scale,
            "negative_samples_clipped": negative_samples_clipped,
            "components": components,
            "bands": bands,
        },
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("jasper_ridge", "urban", "cuprite", "samson"),
        required=True,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("vca", "spa", "nfindr", "mvc_nmf"),
        default=["vca", "spa", "nfindr", "mvc_nmf"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lambda-vol", type=float, default=1e-3, help="Volume weight for MVC-NMF.")
    parser.add_argument("--mvc-iters", type=int, default=250)
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_benchmark_dataset(
        args.dataset,
        repo_root=resolve_repo_root(),
        data_path=args.data,
        gt_path=args.gt,
        components_hint=args.components,
    )
    h, w, bands = ds.cube.shape
    components = args.components if args.components is not None else ds.gt_M.shape[1]
    x_original = ds.cube.reshape(-1, bands)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        # SPA is completely deterministic; 1 seed is sufficient
        run_seeds = [args.seeds[0]] if method == "spa" else args.seeds
        for seed in run_seeds:
            print(f"Running {method.upper()} on {ds.name} (seed={seed})...")
            payload = run_baseline(
                method,
                x_original,
                components,
                seed=seed,
                lambda_vol=args.lambda_vol,
                mvc_iters=args.mvc_iters,
                device=args.device,
            )
            payload["dataset"] = ds.name
            payload["settings"]["data_shape"] = (h, w, bands)
            print(
                "Negative reflectance samples clipped to zero: "
                f"{payload['settings']['negative_samples_clipped']}"
            )
            out_path = out_dir / f"{ds.name}_baseline_{method}_seed{seed}.pt"
            torch.save(payload, out_path)
            print(
                f"  Saved {out_path.name} | scaled-MSE: {payload['final_scaled_mse']:.6f} | "
                f"log-det: {payload['final_vol_loss']:.4f}"
            )


if __name__ == "__main__":
    main()
