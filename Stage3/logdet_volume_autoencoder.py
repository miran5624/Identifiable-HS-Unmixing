"""Train the Log-Det Volume-Regularized Autoencoder on Jasper Ridge, Urban, Cuprite, or Samson."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from dataset_loaders import clip_negative_reflectance, load_benchmark_dataset, resolve_repo_root


class LogDetVolumeAE(nn.Module):
    """Encoder predicts simplex abundances; non-negative decoder weights are endmembers."""

    def __init__(self, bands: int, components: int, use_encoder_bn: bool = False) -> None:
        super().__init__()
        hidden = max(bands // 2, components * 4)
        if use_encoder_bn:
            self.e = nn.Sequential(
                nn.BatchNorm1d(bands),
                nn.Linear(bands, hidden),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, components),
                nn.Softmax(dim=1),
            )
        else:
            self.e = nn.Sequential(
                nn.Linear(bands, hidden),
                nn.ReLU(),
                nn.Linear(hidden, components),
                nn.Softmax(dim=1),
            )
        self.d = nn.Linear(components, bands, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        abundances = self.e(x)
        return self.d(abundances), abundances


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_endmembers_random(
    bands: int, components: int, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    return torch.empty((bands, components), dtype=dtype).uniform_(0.0, 1.0, generator=gen).to(device)


def init_endmembers_from_data(x: torch.Tensor, components: int, seed: int) -> torch.Tensor:
    x_cpu = x.detach().cpu()
    gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    n_pixels = x_cpu.shape[0]
    first_idx = int(torch.randint(0, n_pixels, (1,), generator=gen).item())
    selected = [first_idx]
    min_sq_dists = torch.full((n_pixels,), float("inf"))
    for _ in range(1, components):
        sq_dists = ((x_cpu - x_cpu[selected[-1]]) ** 2).sum(dim=1)
        min_sq_dists = torch.minimum(min_sq_dists, sq_dists)
        selected.append(int(torch.argmax(min_sq_dists).item()))
    return x_cpu[selected].T.clamp(min=1e-4).contiguous().to(x.device)


def per_pixel_sse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((x_hat - x) ** 2).sum(dim=1).mean()


def log_det_volume_loss(endmembers: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    e64 = endmembers.to(dtype=torch.float64)
    components = e64.shape[1]
    eye = torch.eye(components, device=e64.device, dtype=torch.float64)
    gram = e64.T @ e64 + eps * eye
    sign, logabsdet = torch.linalg.slogdet(gram)
    if (sign <= 0).any():
        raise RuntimeError("Gram matrix E^T E + eps*I is not positive definite; increase --vol-eps.")
    return logabsdet.to(dtype=endmembers.dtype)


def train_logdet_ae(
    x_original: np.ndarray,
    components: int,
    *,
    init: str = "random",
    init_endmembers_scaled: np.ndarray | None = None,
    use_encoder_bn: bool = False,
    epochs: int = 2000,
    warmup_epochs: int = 0,
    freeze_decoder_warmup: bool = False,
    lr: float = 1e-3,
    decoder_lr: float | None = None,
    lambda_vol: float = 1e-3,
    vol_eps: float = 1e-4,
    input_scale: float | None = None,
    batch_size: int = 0,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Reusable training routine shared by real benchmarks and synthetic experiments."""
    set_deterministic(seed)
    dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )

    x_np = np.ascontiguousarray(x_original, dtype=np.float32)
    if not np.isfinite(x_np).all():
        raise ValueError("Input hyperspectral data contains NaN or infinite values.")
    x_np, negative_samples_clipped = clip_negative_reflectance(x_np)
    x_np = np.ascontiguousarray(x_np, dtype=np.float32)

    n_pixels, bands = x_np.shape
    scale = float(np.max(x_np)) if input_scale is None else float(input_scale)
    if scale <= 0.0:
        raise ValueError(f"Invalid input scale: {scale}")

    x_scaled_np = x_np / scale
    x = torch.from_numpy(x_scaled_np).to(dev)
    eff_batch = batch_size if batch_size > 0 else (4096 if n_pixels > 20_000 else n_pixels)

    model = LogDetVolumeAE(bands, components, use_encoder_bn=use_encoder_bn).to(dev)
    with torch.no_grad():
        if init_endmembers_scaled is not None:
            init_e = torch.from_numpy(
                np.ascontiguousarray(init_endmembers_scaled, dtype=np.float32)
            ).to(dev)
        elif init == "vca":
            from classical_unmixing_baselines import run_vca

            e_vca = run_vca(x_scaled_np, components, seed=seed)
            init_e = torch.from_numpy(np.ascontiguousarray(e_vca, dtype=np.float32)).to(dev)
        elif init == "data":
            init_e = init_endmembers_from_data(x, components, seed)
        else:
            init_e = init_endmembers_random(bands, components, seed, dev, x.dtype)
        model.d.weight.copy_(init_e.clamp(min=0.0))

    dec_lr = lr if decoder_lr is None else decoder_lr
    optimizer = optim.Adam(
        [
            {"params": model.e.parameters(), "lr": lr},
            {"params": model.d.parameters(), "lr": dec_lr},
        ]
    )
    perm_gen = torch.Generator(device="cpu").manual_seed(seed + 20_000)

    total_loss_history: list[float] = []
    recon_sse_history: list[float] = []
    recon_mse_history: list[float] = []
    vol_loss_history: list[float] = []

    model.train()
    log_interval = max(1, epochs // 8)
    for epoch in range(epochs):
        in_warmup = epoch < warmup_epochs
        active_lambda = 0.0 if in_warmup else lambda_vol

        if freeze_decoder_warmup:
            model.d.weight.requires_grad_(not in_warmup)

        if eff_batch >= n_pixels:
            optimizer.zero_grad()
            x_hat, _ = model(x)
            recon_sse = per_pixel_sse_loss(x_hat, x)
            if active_lambda > 0.0:
                vol_loss = log_det_volume_loss(model.d.weight, eps=vol_eps)
                total_loss = recon_sse + active_lambda * vol_loss
            else:
                with torch.no_grad():
                    vol_loss = log_det_volume_loss(model.d.weight, eps=vol_eps)
                total_loss = recon_sse
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.d.weight.clamp_(min=0.0)
            epoch_sse = float(recon_sse.detach().item())
            epoch_vol = float(vol_loss.detach().item())
            epoch_tot = float(total_loss.detach().item())
        else:
            perm = torch.randperm(n_pixels, generator=perm_gen)
            sse_accum = 0.0
            n_seen = 0
            for start in range(0, n_pixels, eff_batch):
                idx = perm[start : start + eff_batch].to(dev)
                if idx.numel() < 2 and use_encoder_bn:
                    continue
                xb = x[idx]
                optimizer.zero_grad()
                x_hat, _ = model(xb)
                recon_sse = per_pixel_sse_loss(x_hat, xb)
                if active_lambda > 0.0:
                    vol_loss = log_det_volume_loss(model.d.weight, eps=vol_eps)
                    total_loss = recon_sse + active_lambda * vol_loss
                else:
                    total_loss = recon_sse
                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.d.weight.clamp_(min=0.0)
                bs = xb.shape[0]
                sse_accum += float(recon_sse.detach().item()) * bs
                n_seen += bs
            epoch_sse = sse_accum / n_seen
            with torch.no_grad():
                epoch_vol = float(log_det_volume_loss(model.d.weight, eps=vol_eps).item())
            epoch_tot = epoch_sse + active_lambda * epoch_vol

        recon_sse_history.append(epoch_sse)
        recon_mse_history.append(epoch_sse / bands)
        vol_loss_history.append(epoch_vol)
        total_loss_history.append(epoch_tot)

        if verbose and ((epoch + 1) % log_interval == 0 or epoch == 0 or epoch + 1 == epochs):
            phase = "warmup" if in_warmup else ("control" if lambda_vol == 0.0 else "volmin")
            print(
                f"Epoch {epoch + 1}/{epochs} [{phase}] | "
                f"pixel-SSE: {epoch_sse:.6f} | scaled-MSE: {epoch_sse / bands:.8f} | "
                f"log-det: {epoch_vol:.6f} | total: {epoch_tot:.6f}"
            )

    model.d.weight.requires_grad_(True)
    model.eval()
    with torch.no_grad():
        abund_chunks = []
        sse_accum = 0.0
        for start in range(0, n_pixels, 16384):
            xb = x[start : start + 16384]
            xh, ab = model(xb)
            sse_accum += float(((xh - xb) ** 2).sum(dim=1).sum().item())
            abund_chunks.append(ab.cpu())
        abundances = torch.cat(abund_chunks, dim=0)

        final_scaled_sse = sse_accum / n_pixels
        final_scaled_mse = final_scaled_sse / bands
        final_original_mse = final_scaled_mse * (scale**2)
        final_vol_loss = float(log_det_volume_loss(model.d.weight, eps=vol_eps).item())
        final_total_loss = final_scaled_sse + lambda_vol * final_vol_loss

        endmembers_scaled = model.d.weight.detach().cpu().clone()
        endmembers = endmembers_scaled * scale

        e_norm = F.normalize(endmembers_scaled, dim=0, eps=1e-12)
        cos_sim = (e_norm.T @ e_norm).clamp(-1.0, 1.0)
        pairwise_sad_deg = torch.rad2deg(torch.acos(cos_sim))
        off_diag = ~torch.eye(components, dtype=torch.bool)
        min_pairwise_sad_deg = float(pairwise_sad_deg[off_diag].min().item())

    payload = {
        "method": "AE (lambda=0)" if lambda_vol == 0.0 else "LogDet-AE",
        "a": abundances,
        "e": endmembers,
        "e_scaled": endmembers_scaled,
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
        "pairwise_sad_deg": pairwise_sad_deg,
        "min_pairwise_sad_deg": min_pairwise_sad_deg,
        "settings": {
            "init": init,
            "use_encoder_bn": use_encoder_bn,
            "seed": seed,
            "lambda_vol": lambda_vol,
            "vol_eps": vol_eps,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "freeze_decoder_warmup": freeze_decoder_warmup,
            "lr": lr,
            "decoder_lr": dec_lr,
            "input_scale": scale,
            "negative_samples_clipped": negative_samples_clipped,
            "batch_size": eff_batch,
            "components": components,
            "bands": bands,
            "device": str(dev),
        },
    }
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return payload, state_dict


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("jasper_ridge", "urban", "cuprite", "samson"),
        required=True,
        help="Target hyperspectral benchmark dataset.",
    )
    parser.add_argument("--data", type=Path, help="Optional explicit data .mat or .hdr path.")
    parser.add_argument("--gt", type=Path, help="Optional explicit ground-truth .mat path.")
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    parser.add_argument("--init", choices=("random", "data", "vca"), default="random")
    parser.add_argument("--encoder-bn", action="store_true", help="Enable BatchNorm1d + LeakyReLU in encoder.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--freeze-decoder-warmup", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--lambda-vol", type=float, default=1e-3)
    parser.add_argument("--vol-eps", type=float, default=1e-4)
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    ds = load_benchmark_dataset(
        args.dataset,
        repo_root=repo_root,
        data_path=args.data,
        gt_path=args.gt,
        components_hint=args.components,
    )

    h, w, bands = ds.cube.shape
    n_pixels = h * w
    components = args.components if args.components is not None else ds.gt_M.shape[1]
    epochs = args.epochs if args.epochs is not None else (400 if n_pixels > 20_000 else 2000)

    x_original = ds.cube.reshape(-1, bands)
    print(f"Dataset: {ds.name} | Cube: {ds.cube.shape} | Endmembers: {components}")
    print(f"Data: {ds.data_path}")
    print(f"GT:   {ds.gt_path} (has abundance GT: {ds.gt_A is not None})")

    payload, state_dict = train_logdet_ae(
        x_original,
        components,
        init=args.init,
        use_encoder_bn=args.encoder_bn,
        epochs=epochs,
        warmup_epochs=args.warmup_epochs,
        freeze_decoder_warmup=args.freeze_decoder_warmup,
        lr=args.lr,
        decoder_lr=args.decoder_lr,
        lambda_vol=args.lambda_vol,
        vol_eps=args.vol_eps,
        input_scale=args.input_scale,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    payload["dataset"] = ds.name
    payload["settings"]["data_shape"] = (h, w, bands)
    print(
        "Negative reflectance samples clipped to zero: "
        f"{payload['settings']['negative_samples_clipped']}"
    )

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.init}_warmup{args.warmup_epochs}_seed{args.seed}_lambda{args.lambda_vol:g}"
    unmixing_path = out_dir / f"{ds.name}_unmixing_{run_tag}.pt"
    weights_path = out_dir / f"{ds.name}_weights_{run_tag}.pt"

    torch.save(state_dict, weights_path)
    torch.save(payload, unmixing_path)
    print(f"\nSaved unmixing run: {unmixing_path}")
    print(f"Saved weights:      {weights_path}")


if __name__ == "__main__":
    main()
