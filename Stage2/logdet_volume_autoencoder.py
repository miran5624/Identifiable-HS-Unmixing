"""Train the Stage 2 log-det volume-regularized autoencoder on the Samson scene."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import spectral
import torch
import torch.nn.functional as F
from torch import nn, optim


class LogDetVolumeAE(nn.Module):
    """Encoder predicts simplex abundances; non-negative decoder weights are endmembers."""

    def __init__(self, bands: int, components: int) -> None:
        super().__init__()
        self.e = nn.Sequential(
            nn.Linear(bands, bands // 2),
            nn.ReLU(),
            nn.Linear(bands // 2, components),
            nn.Softmax(dim=1),
        )
        self.d = nn.Linear(components, bands, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        abundances = self.e(x)
        return self.d(abundances), abundances


def set_deterministic(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and enforce deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_endmembers_random(
    bands: int, components: int, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Initialize decoder endmembers uniformly in [0, 1] on scaled reflectance."""
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    return torch.empty((bands, components), device=device, dtype=dtype).uniform_(
        0.0, 1.0, generator=gen
    )


def init_endmembers_from_data(
    x: torch.Tensor, components: int, seed: int
) -> torch.Tensor:
    """Pick well-separated, non-negative pixels as initial endmembers (B, p)."""
    gen = torch.Generator(device=x.device).manual_seed(seed + 10_000)
    n_pixels = x.shape[0]
    first_idx = int(torch.randint(0, n_pixels, (1,), generator=gen).item())
    selected = [first_idx]

    min_sq_dists = torch.full((n_pixels,), float("inf"), device=x.device)
    for _ in range(1, components):
        last_pixel = x[selected[-1]]
        sq_dists = ((x - last_pixel) ** 2).sum(dim=1)
        min_sq_dists = torch.minimum(min_sq_dists, sq_dists)
        next_idx = int(torch.argmax(min_sq_dists).item())
        selected.append(next_idx)

    # Shape: (bands, components) matching nn.Linear(components, bands).weight
    return x[selected].T.clamp(min=1e-4).contiguous()


def per_pixel_sse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Squared reconstruction error summed over bands and averaged over pixels."""
    return ((x_hat - x) ** 2).sum(dim=1).mean()


def log_det_volume_loss(endmembers: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Compute log det(E^T E + eps * I) in float64 for E of shape (bands, components)."""
    e64 = endmembers.to(dtype=torch.float64)
    components = e64.shape[1]
    eye = torch.eye(components, device=e64.device, dtype=torch.float64)
    gram = e64.T @ e64 + eps * eye
    sign, logabsdet = torch.linalg.slogdet(gram)
    if (sign <= 0).any():
        raise RuntimeError(
            "Gram matrix E^T E + eps*I is not positive definite; increase --vol-eps."
        )
    return logabsdet.to(dtype=endmembers.dtype)


def find_header(data_path: Path | None, repo_root: Path) -> Path:
    """Resolve an explicit ENVI header or locate the Samson header under repo_root."""
    if data_path is not None:
        candidate = data_path.expanduser().resolve()
        if candidate.suffix.lower() == ".img":
            hdr_double = candidate.with_suffix(".img.hdr")
            hdr_single = candidate.with_suffix(".hdr")
            candidate = hdr_double if hdr_double.is_file() else hdr_single
        if not candidate.is_file():
            raise FileNotFoundError(f"Samson ENVI header not found: {candidate}")
        return candidate

    matches = sorted(repo_root.rglob("samson_1.img.hdr"))
    if not matches:
        matches = sorted(
            p for p in repo_root.rglob("*.hdr") if "samson" in p.name.lower()
        )
    if not matches:
        raise FileNotFoundError(
            f"Could not find Samson ENVI .hdr file under {repo_root}. "
            "Pass its exact location with --data."
        )
    if len(matches) > 1:
        locations = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            "Found more than one Samson header; select one with --data:\n" + locations
        )
    return matches[0]


def resolve_image_path(header_path: Path) -> Path:
    """Find the binary .img file matching an ENVI .hdr file."""
    direct = header_path.with_suffix("")
    if direct.is_file():
        return direct
    alt = header_path.with_suffix(".img")
    if alt.is_file():
        return alt
    raise FileNotFoundError(
        f"Matching Samson ENVI image not found for header: {header_path}"
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Samson ENVI .hdr file (or matching .img); searched automatically from repo root if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for saved unmixing outputs and model weights (default: Stage2).",
    )
    parser.add_argument(
        "--init",
        choices=("random", "data"),
        default="random",
        help="Decoder initialization: random uniform [0, 1] (default) or farthest-point data ablation.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Pure-reconstruction warmup epochs before enabling volume penalty (test 0 vs 200).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-vol",
        type=float,
        default=1e-4,
        help="Weight on log det(E^T E + eps * I) relative to per-pixel SSE (0 = matched control baseline).",
    )
    parser.add_argument(
        "--vol-eps",
        type=float,
        default=1e-4,
        help="Diagonal stabilizer delta in log det(E^T E + delta * I).",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1000.0,
        help="Divide ENVI values by this scale during training; saved endmembers use original units.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs)")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.lambda_vol < 0:
        raise ValueError("--lambda-vol must be non-negative")
    if args.vol_eps <= 0:
        raise ValueError("--vol-eps must be positive")
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")

    set_deterministic(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    header_path = find_header(args.data, repo_root)
    image_path = resolve_image_path(header_path)

    image = spectral.envi.open(str(header_path), str(image_path))
    cube_np = np.asarray(image.load(), dtype=np.float32)
    if cube_np.shape != (95, 95, 156):
        raise ValueError(
            f"Expected exact Samson cube shape (95, 95, 156), got {cube_np.shape}"
        )

    cube = torch.from_numpy(np.ascontiguousarray(cube_np))
    bands = 156
    components = 3

    x_original = cube.reshape(-1, bands).contiguous()
    if not torch.isfinite(x_original).all():
        raise ValueError("The Samson cube contains NaN or infinite values")

    # Scale inputs without clamping so all lambda runs share identical preprocessing.
    x = x_original / args.input_scale
    print(f"Loaded Samson ENVI: {image_path}")
    print(
        f"Cube shape {tuple(cube.shape)} | "
        f"raw range: [{x_original.min().item():.4f}, {x_original.max().item():.4f}] | "
        f"scaled range: [{x.min().item():.4f}, {x.max().item():.4f}]"
    )
    if x.min().item() < 0.0:
        print("NOTE: Scaled input contains negative values (not clamped).")
    if x.max().item() < 0.05 or x.max().item() > 10.0:
        print(
            "WARNING: Scaled reflectance range looks unusual. "
            "Double-check --input-scale against the raw data range above."
        )

    model = LogDetVolumeAE(bands, components)
    with torch.no_grad():
        if args.init == "random":
            init_e = init_endmembers_random(bands, components, args.seed, x.device, x.dtype)
        else:
            init_e = init_endmembers_from_data(x, components, args.seed)
        model.d.weight.copy_(init_e)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_loss_history: list[float] = []
    recon_sse_history: list[float] = []
    recon_mse_history: list[float] = []
    vol_loss_history: list[float] = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        x_hat, _ = model(x)
        recon_sse = per_pixel_sse_loss(x_hat, x)
        vol_loss = log_det_volume_loss(model.d.weight, eps=args.vol_eps)

        active_lambda = 0.0 if epoch < args.warmup_epochs else args.lambda_vol
        total_loss = recon_sse + active_lambda * vol_loss

        total_loss.backward()
        optimizer.step()

        # Enforce physical non-negativity constraint: E in R_+^{B x p}
        with torch.no_grad():
            model.d.weight.clamp_(min=0.0)

        recon_sse_val = float(recon_sse.detach().item())
        total_loss_history.append(float(total_loss.detach().item()))
        recon_sse_history.append(recon_sse_val)
        recon_mse_history.append(recon_sse_val / bands)
        vol_loss_history.append(float(vol_loss.detach().item()))

        if (epoch + 1) % 250 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            if args.lambda_vol == 0.0:
                phase = "control"
            elif epoch < args.warmup_epochs:
                phase = "warmup"
            else:
                phase = "volmin"
            print(
                f"Epoch {epoch + 1}/{args.epochs} [{phase}] | "
                f"pixel-SSE: {recon_sse_val:.6f} | "
                f"scaled-MSE: {recon_sse_val / bands:.8f} | "
                f"log-det: {vol_loss.item():.6f} | "
                f"total: {total_loss.item():.6f}"
            )

    with torch.no_grad():
        x_hat_scaled, abundances = model(x)
        final_scaled_sse = float(per_pixel_sse_loss(x_hat_scaled, x).item())
        final_scaled_mse = final_scaled_sse / bands
        final_original_mse = final_scaled_mse * (args.input_scale ** 2)
        final_vol_loss = float(log_det_volume_loss(model.d.weight, eps=args.vol_eps).item())
        final_total_loss = final_scaled_sse + args.lambda_vol * final_vol_loss

        endmembers_scaled = model.d.weight.detach().clone()
        endmembers = endmembers_scaled * args.input_scale

        simplex_err = float((abundances.sum(dim=1) - 1.0).abs().max().item())
        e_norm = F.normalize(endmembers_scaled, dim=0, eps=1e-12)
        cos_sim = (e_norm.T @ e_norm).clamp(-1.0, 1.0)
        pairwise_sad_deg = torch.rad2deg(torch.acos(cos_sim))
        off_diag = ~torch.eye(components, dtype=torch.bool, device=pairwise_sad_deg.device)
        min_pairwise_sad_deg = float(pairwise_sad_deg[off_diag].min().item())

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = (
        f"{args.init}_warmup{args.warmup_epochs}"
        f"_seed{args.seed}_lambda{args.lambda_vol:g}"
    )
    weights_path = output_dir / f"samson_weights_{run_tag}.pt"
    unmixing_path = output_dir / f"samson_unmixing_{run_tag}.pt"

    torch.save(model.state_dict(), weights_path)
    torch.save(
        {
            "a": abundances.cpu(),
            "e": endmembers.cpu(),
            "e_scaled": endmembers_scaled.cpu(),
            "final_loss": final_original_mse,
            "final_scaled_mse": final_scaled_mse,
            "final_scaled_sse": final_scaled_sse,
            "final_vol_loss": final_vol_loss,
            "final_total_loss": final_total_loss,
            "loss_history": recon_mse_history,
            "recon_sse_history": recon_sse_history,
            "recon_mse_history": recon_mse_history,
            "vol_loss_history": vol_loss_history,
            "total_loss_history": total_loss_history,
            "pairwise_sad_deg": pairwise_sad_deg.cpu(),
            "min_pairwise_sad_deg": min_pairwise_sad_deg,
            "settings": {
                "init": args.init,
                "seed": args.seed,
                "lambda_vol": args.lambda_vol,
                "vol_eps": args.vol_eps,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "lr": args.lr,
                "input_scale": args.input_scale,
                "loss_type": "per_pixel_sse + lambda_vol * logdet(E^T E + vol_eps * I)",
                "is_control_baseline": args.lambda_vol == 0.0,
                "data_shape": (95, 95, 156),
            },
        },
        unmixing_path,
    )

    print("\n--- Stage 2 Sanity Checks ---")
    print(f"Run tag:                  {run_tag}")
    print(f"Final per-pixel SSE:      {final_scaled_sse:.6f}")
    print(f"Final scaled MSE:         {final_scaled_mse:.8f}")
    print(f"Final original-unit MSE:  {final_original_mse:.4f}")
    print(f"Final log-det volume:     {final_vol_loss:.6f}")
    print(f"Final total loss:         {final_total_loss:.6f}")
    print(f"Endmember range (orig):   [{endmembers.min().item():.4f}, {endmembers.max().item():.4f}]")
    print(f"Abundance simplex max err:{simplex_err:.2e}")
    print(f"Min pairwise angle (deg): {min_pairwise_sad_deg:.2f}")
    print("Pairwise endmember angles (deg):")
    print(pairwise_sad_deg.cpu().numpy().round(2))
    print(f"\nWeights saved to:  {weights_path}")
    print(f"Unmixing saved to: {unmixing_path}")


if __name__ == "__main__":
    run()