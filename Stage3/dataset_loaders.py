"""Shared loaders for Jasper Ridge, Urban, Cuprite, Samson, and USGS spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio

# Standard AVIRIS 188-band indices (0-based) used in Cuprite when SlectBands is absent.
CUPRITE_188_BANDS_0BASED = np.array(
    [
        i
        for i in range(224)
        if not (
            (0 <= i <= 1)
            or (103 <= i <= 112)
            or (147 <= i <= 166)
            or (220 <= i <= 223)
        )
    ],
    dtype=int,
)


@dataclass
class BenchmarkDataset:
    name: str
    cube: np.ndarray  # (H, W, B) float32 in row-major image layout
    gt_M: np.ndarray  # (B, p) float64
    gt_A: np.ndarray | None  # (p, H*W) float64 or None (e.g., Cuprite)
    data_path: Path
    gt_path: Path
    default_gt_layout: str = "fortran_to_c"


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _extract_scalar_int(mat: dict, keys: Iterable[str]) -> int | None:
    for k in keys:
        if k in mat:
            arr = np.asarray(mat[k]).squeeze()
            if arr.size == 1:
                return int(arr.item())
    return None


def _extract_slect_bands(mat: dict) -> np.ndarray | None:
    for k in ("SlectBands", "slctBnds", "selectBands", "valid_bands"):
        if k in mat:
            idx = np.asarray(mat[k]).squeeze().astype(int)
            if idx.ndim == 1 and idx.size > 0:
                return idx - 1 if idx.min() >= 1 else idx
    return None


def _load_envi_cube(header_path: Path) -> np.ndarray:
    import spectral

    direct = header_path.with_suffix("")
    image_path = direct if direct.is_file() else header_path.with_suffix(".img")
    if not image_path.is_file():
        raise FileNotFoundError(f"Matching ENVI .img file not found for {header_path}")
    img = spectral.envi.open(str(header_path), str(image_path))
    return np.asarray(img.load(), dtype=np.float32)


def _load_mat_cube(mat_path: Path, dataset_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    mat = sio.loadmat(str(mat_path))
    slect_bands = _extract_slect_bands(mat)

    data_key = None
    for candidate in ("Y", "X", "x", "img", "cube", "data", "V"):
        if candidate in mat and isinstance(mat[candidate], np.ndarray) and mat[candidate].ndim in (2, 3):
            data_key = candidate
            break
    if data_key is None:
        arrays = [
            (k, v)
            for k, v in mat.items()
            if not k.startswith("__")
            and isinstance(v, np.ndarray)
            and np.issubdtype(v.dtype, np.number)
            and v.ndim in (2, 3)
        ]
        if not arrays:
            raise KeyError(f"No 2D/3D numeric image array found in {mat_path}")
        data_key = max(arrays, key=lambda kv: kv[1].size)[0]

    raw = np.asarray(mat[data_key], dtype=np.float32)
    if raw.ndim == 3:
        if raw.shape[0] in (156, 162, 188, 198, 224) and raw.shape[2] not in (156, 162, 188, 198, 224):
            cube = np.transpose(raw, (1, 2, 0))
        else:
            cube = raw
        return np.ascontiguousarray(cube, dtype=np.float32), slect_bands

    # 2D matrix: orient as (B, N)
    y_bn = raw.T if raw.shape[0] > raw.shape[1] else raw
    bands, n_pixels = y_bn.shape

    n_row = _extract_scalar_int(mat, ("nRow", "nrow", "nRows", "H", "height", "lines", "Lines"))
    n_col = _extract_scalar_int(mat, ("nCol", "ncol", "nCols", "W", "width", "samples", "cols", "Columns"))

    if n_row is None or n_col is None or n_row * n_col != n_pixels:
        defaults = {
            "samson": (95, 95),
            "jasper_ridge": (100, 100),
            "urban": (307, 307),
            "cuprite": (250, 190),
        }
        if dataset_name in defaults and defaults[dataset_name][0] * defaults[dataset_name][1] == n_pixels:
            n_row, n_col = defaults[dataset_name]
        else:
            side = int(round(np.sqrt(n_pixels)))
            if side * side != n_pixels:
                raise ValueError(
                    f"Cannot infer (nRow, nCol) for {mat_path} with {n_pixels} pixels."
                )
            n_row, n_col = side, side

    # MATLAB stores 2D flattened hyperspectral cubes column-major (order='F').
    cube = y_bn.reshape(bands, n_row, n_col, order="F").transpose(1, 2, 0)
    return np.ascontiguousarray(cube, dtype=np.float32), slect_bands


def _load_gt_mat(
    gt_path: Path, expected_bands: int, expected_pixels: int, slect_bands: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    mat = sio.loadmat(str(gt_path))
    if slect_bands is None:
        slect_bands = _extract_slect_bands(mat)

    m_key = next((k for k in ("M", "E", "endmembers", "gt_M") if k in mat), None)
    if m_key is None:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Could not find endmember matrix 'M' in {gt_path}; keys={keys}")

    M = np.asarray(mat[m_key], dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"Expected 2D M in {gt_path}, got shape {M.shape}")

    # Orient M as (bands, components)
    if M.shape[1] in (expected_bands, 224) and M.shape[0] not in (expected_bands, 224):
        M = M.T

    if M.shape[0] != expected_bands:
        if slect_bands is not None and slect_bands.size == expected_bands and slect_bands.max() < M.shape[0]:
            M = M[slect_bands, :]
        elif M.shape[0] == 224 and expected_bands == 188:
            print(
                f"NOTE: Slicing 224-band ground truth {gt_path.name} to standard 188 Cuprite bands."
            )
            M = M[CUPRITE_188_BANDS_0BASED, :]
        else:
            raise ValueError(
                f"Band mismatch: cube has {expected_bands} bands but {gt_path.name} M has shape {M.shape}"
            )

    components = M.shape[1]
    a_key = next((k for k in ("A", "S", "abundances", "gt_A") if k in mat), None)
    if a_key is None:
        return M, None

    A = np.asarray(mat[a_key], dtype=np.float64)
    if A.ndim == 3:
        if A.shape[2] == components:
            A = A.reshape(-1, components).T
        elif A.shape[0] == components:
            A = A.reshape(components, -1)
    if A.shape == (expected_pixels, components):
        A = A.T
    if A.shape != (components, expected_pixels):
        raise ValueError(
            f"Unexpected abundance GT shape {A.shape} in {gt_path.name}; "
            f"expected ({components}, {expected_pixels})"
        )
    return M, A


def _find_dataset_files(
    dataset_name: str, repo_root: Path, components_hint: int | None = None
) -> tuple[Path, Path]:
    datasets_root = repo_root / "DATASETS"
    search_root = datasets_root if datasets_root.is_dir() else repo_root
    all_files = sorted(p for p in search_root.rglob("*") if p.is_file())

    def match_files(keywords: tuple[str, ...], suffixes: tuple[str, ...]) -> list[Path]:
        out = []
        for p in all_files:
            low = str(p).lower()
            if p.suffix.lower() in suffixes and any(kw in low for kw in keywords):
                out.append(p)
        return out

    if dataset_name == "samson":
        hdrs = match_files(("samson",), (".hdr",))
        data_file = hdrs[0] if hdrs else match_files(("samson_1",), (".mat",))[0]
        gts = [p for p in match_files(("samgt",), (".mat",))]
        if not gts:
            gts = [p for p in match_files(("samson",), (".mat",)) if "gt" in p.name.lower() or "end" in p.name.lower()]
        return data_file, gts[0]

    if dataset_name == "jasper_ridge":
        mats = match_files(("jasper",), (".mat",))
        gt_candidates = [p for p in mats if any(t in p.name.lower() for t in ("end", "gt", "groundtruth"))]
        data_candidates = [p for p in mats if p not in gt_candidates]
        if not data_candidates or not gt_candidates:
            raise FileNotFoundError(
                f"Could not locate Jasper Ridge data & GT .mat files under {search_root}. "
                "Pass --data and --gt explicitly."
            )
        data_file = next((p for p in data_candidates if "198" in p.name.lower()), data_candidates[0])
        gt_file = next((p for p in gt_candidates if "end4" in p.name.lower()), gt_candidates[0])
        return data_file, gt_file

    if dataset_name == "urban":
        mats = match_files(("urban",), (".mat",))
        gt_candidates = [p for p in mats if any(t in p.name.lower() for t in ("end", "gt", "groundtruth"))]
        data_candidates = [p for p in mats if p not in gt_candidates]
        if not data_candidates or not gt_candidates:
            raise FileNotFoundError(
                f"Could not locate Urban data & GT .mat files under {search_root}. "
                "Pass --data and --gt explicitly."
            )
        data_file = next((p for p in data_candidates if "162" in p.name.lower()), data_candidates[0])
        if components_hint is not None:
            tagged = [p for p in gt_candidates if f"end{components_hint}" in p.name.lower()]
            gt_file = tagged[0] if tagged else gt_candidates[0]
        else:
            gt_file = next(
                (p for p in gt_candidates if "end4" in p.name.lower()),
                next((p for p in gt_candidates if "end6" in p.name.lower()), gt_candidates[0]),
            )
        return data_file, gt_file

    if dataset_name == "cuprite":
        mats = match_files(("cuprite",), (".mat",))
        gt_candidates = [p for p in mats if any(t in p.name.lower() for t in ("end", "gt", "groundtruth"))]
        data_candidates = [p for p in mats if p not in gt_candidates]
        if not data_candidates or not gt_candidates:
            raise FileNotFoundError(
                f"Could not locate Cuprite data & GT .mat files under {search_root}. "
                "Pass --data and --gt explicitly."
            )
        data_file = next((p for p in data_candidates if "188" in p.name.lower()), data_candidates[0])
        gt_file = next((p for p in gt_candidates if "12" in p.name.lower()), gt_candidates[0])
        return data_file, gt_file

    raise ValueError(f"Unknown dataset: {dataset_name}")


def load_benchmark_dataset(
    dataset_name: str,
    repo_root: Path | None = None,
    data_path: Path | None = None,
    gt_path: Path | None = None,
    components_hint: int | None = None,
) -> BenchmarkDataset:
    """Load any supported benchmark dataset into standardized (H, W, B), M, and optional A."""
    dataset_name = dataset_name.lower().strip()
    root = repo_root or resolve_repo_root()

    if data_path is None or gt_path is None:
        auto_data, auto_gt = _find_dataset_files(dataset_name, root, components_hint=components_hint)
        data_file = data_path.expanduser().resolve() if data_path else auto_data
        gt_file = gt_path.expanduser().resolve() if gt_path else auto_gt
    else:
        data_file = data_path.expanduser().resolve()
        gt_file = gt_path.expanduser().resolve()

    if data_file.suffix.lower() in (".hdr", ".img"):
        hdr = data_file if data_file.suffix.lower() == ".hdr" else data_file.with_suffix(".img.hdr")
        cube = _load_envi_cube(hdr)
        slect_bands = None
    else:
        cube, slect_bands = _load_mat_cube(data_file, dataset_name)

    # Ensure Cuprite always removes the 36 noisy water-vapor bands (224 -> 188)
    if dataset_name == "cuprite" and cube.shape[2] == 224:
        if slect_bands is None:
            slect_bands = _extract_slect_bands(sio.loadmat(str(gt_file)))
        idx = (
            slect_bands
            if (slect_bands is not None and slect_bands.size == 188 and slect_bands.max() < 224)
            else CUPRITE_188_BANDS_0BASED
        )
        cube = np.ascontiguousarray(cube[:, :, idx], dtype=np.float32)
        slect_bands = idx

    h, w, bands = cube.shape
    gt_M, gt_A = _load_gt_mat(gt_file, expected_bands=bands, expected_pixels=h * w, slect_bands=slect_bands)

    return BenchmarkDataset(
        name=dataset_name,
        cube=cube,
        gt_M=gt_M,
        gt_A=gt_A,
        data_path=data_file,
        gt_path=gt_file,
        default_gt_layout="fortran_to_c",
    )


def load_usgs_library(
    repo_root: Path | None = None, usgs_path: Path | None = None
) -> tuple[np.ndarray, list[str], Path]:
    """Load USGS mineral spectra as (bands, n_materials) float64 scaled in [0, 1]."""
    root = repo_root or resolve_repo_root()

    if usgs_path is not None:
        candidate = usgs_path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"USGS library file not found: {candidate}")
    else:
        datasets_root = root / "DATASETS"
        search_root = datasets_root if datasets_root.is_dir() else root
        usgs_mats = sorted(
            p
            for p in search_root.rglob("*.mat")
            if any(tok in p.name.lower() for tok in ("usgs", "splib", "library"))
        )
        if usgs_mats:
            candidate = usgs_mats[0]
        else:
            cuprite_gts = sorted(
                p
                for p in search_root.rglob("*.mat")
                if "cuprite" in str(p).lower()
                and any(tok in p.name.lower() for tok in ("gt", "groundtruth", "end"))
            )
            if not cuprite_gts:
                raise FileNotFoundError(
                    f"No USGS .mat library or Cuprite USGS ground truth found under {search_root}. "
                    "Pass --usgs <path_to_usgs.mat>."
                )
            candidate = cuprite_gts[0]
            print(
                f"NOTE: Dedicated USGS library not found; using USGS mineral spectra from {candidate}"
            )

    mat = sio.loadmat(str(candidate))
    slect_bands = _extract_slect_bands(mat)
    key = next((k for k in ("datalib", "M", "A", "spectra", "library", "R") if k in mat), None)
    if key is None:
        arrays = [
            (k, v)
            for k, v in mat.items()
            if not k.startswith("__")
            and isinstance(v, np.ndarray)
            and np.issubdtype(v.dtype, np.number)
            and v.ndim == 2
        ]
        if not arrays:
            raise KeyError(f"No 2D numeric spectral matrix found in {candidate}")
        key = max(arrays, key=lambda kv: kv[1].size)[0]

    spectra = np.asarray(mat[key], dtype=np.float64)
    idx_188 = (
        slect_bands
        if (slect_bands is not None and slect_bands.size == 188 and slect_bands.max() < 224)
        else CUPRITE_188_BANDS_0BASED
    )

    if spectra.shape[0] == 224:
        # Remove wavelength column if a full USGS library stores wavelengths in column 0
        if spectra.shape[1] > 50 and np.all(np.diff(spectra[:, 0]) > 0) and spectra[:, 0].max() > 1.5:
            spectra = spectra[:, 1:]
        # Keep the clean 188 water-vapor-free AVIRIS bands
        spectra = spectra[idx_188, :]
    elif spectra.shape[1] == 224 and spectra.shape[0] != 224:
        spectra = spectra.T[idx_188, :]

    # Keep only valid non-negative finite spectra
    valid = (
        np.isfinite(spectra).all(axis=0)
        & (spectra.min(axis=0) >= 0.0)
        & (np.linalg.norm(spectra, axis=0) > 1e-6)
    )
    spectra = spectra[:, valid]
    if spectra.max() > 1.5:
        spectra = spectra / spectra.max()

    names = [f"USGS_{i}" for i in range(spectra.shape[1])]
    for name_key in ("names", "cood"):
        if name_key in mat:
            try:
                raw_names = [
                    str(x.item() if hasattr(x, "item") and x.size == 1 else x).strip()
                    for x in np.asarray(mat[name_key]).squeeze()
                ]
                if len(raw_names) == spectra.shape[1]:
                    names = raw_names
                    break
            except Exception:
                pass
    return spectra, names, candidate