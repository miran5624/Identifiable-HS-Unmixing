"""Train the Stage 1 linear-mixing autoencoder on the Samson scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import spectral
import torch
from torch import nn, optim


class AE(nn.Module):
    """Encoder predicts simplex abundances; decoder weights are endmembers."""

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


def find_header(data_path: Path | None, project_dir: Path) -> Path:
    """Resolve an explicit header or find the bundled Samson header."""
    if data_path is not None:
        candidate = data_path.expanduser().resolve()
        if candidate.suffix.lower() == ".img":
            candidate = candidate.with_suffix(".img.hdr")
        if not candidate.is_file():
            raise FileNotFoundError(f"Samson ENVI header not found: {candidate}")
        return candidate

    matches = sorted(project_dir.rglob("samson_1.img.hdr"))
    if not matches:
        raise FileNotFoundError(
            "Could not find samson_1.img.hdr under the project folder. "
            "Pass its location with --data."
        )
    if len(matches) > 1:
        locations = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            "Found more than one Samson header; select one with --data:\n" + locations
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Samson ENVI .hdr file (or matching .img); searched automatically if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir,
        help="Directory for m.pt and r.pt (default: alongside ae.py).",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1000.0,
        help="Divide ENVI values by this scale during training; saved spectra use original units.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")

    torch.manual_seed(0)

    project_dir = Path(__file__).resolve().parent
    header_path = find_header(args.data, project_dir)
    image_path = header_path.with_suffix("")
    if not image_path.is_file():
        raise FileNotFoundError(f"Samson ENVI image not found: {image_path}")

    image = spectral.envi.open(str(header_path), str(image_path))
    cube = torch.as_tensor(image.load(), dtype=torch.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D hyperspectral cube, got shape {tuple(cube.shape)}")

    bands = int(cube.shape[-1])
    if bands != 156:
        raise ValueError(f"Expected 156 spectral bands for Samson, got {bands}")

    # Samson's ENVI values are stored at roughly 1,000 times reflectance.
    # Scale them before optimization so Adam does not saturate the softmax
    # while the decoder is still near its small random initialization.
    x_original = cube.reshape(-1, bands).contiguous()
    x = x_original / args.input_scale
    if not torch.isfinite(x).all():
        raise ValueError("The Samson cube contains NaN or infinite values")

    components = 3
    model = AE(bands, components)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loss_history: list[float] = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        x_hat, _ = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().item()))

        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1}/{args.epochs}, scaled MSE: {loss.item():.8f}")

    with torch.no_grad():
        x_hat_scaled, abundances = model(x)
        x_hat_original = x_hat_scaled * args.input_scale
        final_scaled_loss = loss_fn(x_hat_scaled, x).item()
        final_original_loss = loss_fn(x_hat_original, x_original).item()

    # Linear stores decoder weights as (bands, components), the conventional
    # orientation for the endmember matrix E in x = E @ a.
    endmembers = model.d.weight.detach() * args.input_scale

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "m.pt"
    results_path = output_dir / "r.pt"
    torch.save(model.state_dict(), model_path)
    torch.save(
        {
            "x": x_original,
            "x_h": x_hat_original,
            "a": abundances,
            "e": endmembers,
            "final_loss": final_original_loss,
            "final_scaled_loss": final_scaled_loss,
            "loss_history": loss_history,
            "input_scale": args.input_scale,
            "data_shape": tuple(cube.shape),
        },
        results_path,
    )

    print(f"Final scaled MSE: {final_scaled_loss:.8f}")
    print(f"Final MSE in original units: {final_original_loss:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    run()
