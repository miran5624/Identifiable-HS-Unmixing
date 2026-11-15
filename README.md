# Identifiable Hyperspectral Unmixing with Learned Decoders
## When Can Endmembers Be Recovered?

> **NIT Jamshedpur — CS1503**  
> **GL:** Miran &nbsp;|&nbsp; **GM:** Samran, Irfan, Adnan

---

## Overview

Autoencoder-based hyperspectral unmixing papers routinely report better abundance error on benchmark scenes — but rarely ask whether the **endmembers themselves are identifiable**. Classical theory shows that linear mixtures are identifiable when pure pixels exist (*separability*) or when abundances are sufficiently spread (*sufficiently scattered condition*). This project transfers those guarantees to the decoder of an unmixing autoencoder, demonstrates when plain reconstruction loss **fails** to recover endmembers, and explores a **minimum-volume regulariser** with provable recovery.

### Problem Statement

Pixels follow the **linear mixing model**:

$$x = E\,a + n$$

where $E \in \mathbb{R}^{B \times p}_+$ is the endmember matrix and $a \in \Delta^{p-1}$ (probability simplex). An autoencoder learns $a = f_\theta(x)$ and $\hat{x} = \hat{E}\,a$. The central question is:

> *Under what conditions on the distribution of* $a$ *is* $\hat{E} = E\,\Pi$ *for some permutation* $\Pi$ *(identifiable), and what must the loss include?*

---

## Architecture

```
┌───────────── Encoder ──────────────┐     ┌──── Decoder ────┐
│ Linear(156 → 78) → ReLU           │     │                 │
│ Linear(78 → 3)   → Softmax        │ ──► │ Linear(3 → 156) │
│                                    │  a  │   (bias=False)  │
│ Output: abundances on the simplex  │     │ Weights = E     │
└────────────────────────────────────┘     └─────────────────┘

         x ─► Encoder ─► a ─► Decoder ─► x̂ = E·a
```

| Component | Details |
|-----------|---------|
| **Encoder** | 2-layer MLP → Softmax (outputs lie on the probability simplex) |
| **Decoder** | Single bias-free linear layer — weights **are** the endmember matrix $E$ |
| **Loss** | MSE reconstruction loss (Stage 1); + log-det volume regulariser (Stage 2) |
| **Optimizer** | Full-batch Adam, lr = 0.001 |
| **Training** | 2,000 epochs, seed = 0 |

---

## Dataset

### Samson Scene *(Stage 1)*

| Property | Value |
|----------|-------|
| Image size | 95 × 95 pixels (9,025 total) |
| Spectral bands | 156 (401–889 nm, 3.13 nm resolution) |
| Endmembers | 3 — **Soil**, **Tree**, **Water** |
| Format | ENVI (`.img` + `.img.hdr`) |
| Input scaling | Reflectance values ÷ 1,000 during training |

### Additional Datasets *(planned)*

| Dataset | Size | Bands | Endmembers |
|---------|------|-------|------------|
| Jasper Ridge | 100 × 100 | 198 | 4 |
| Urban (HYDICE) | 307 × 307 | 162 | 4–6 |
| Cuprite (AVIRIS) | — | — | 12 |

---

## Stage 1 Results — Autoencoder Baseline

> **Status: Complete**

### Reconstruction Metrics

| Metric | Value |
|--------|------:|
| Reconstruction MSE (original units) | **2,960.52** |
| Reconstruction RMSE | **54.41** |
| Reconstruction MAE | **31.65** |
| Final Scaled Training MSE | 0.002961 |
| Mean-Spectrum Baseline MSE | 37,246.12 |
| **MSE Reduction vs. Baseline** | **92.0%** |

### Identifiability Metrics (Ground-Truth Comparison)

| Metric | Value |
|--------|------:|
| Mean Spectral Angle to GT Endmembers | **34.26°** |
| Abundance RMSE vs. Ground Truth | **0.4112** |
| Mean Abundance Entropy | 0.8111 |
| Max Abundance Sum Error from 1 | 1.79 × 10⁻⁷ |

**Best endmember permutation:** predicted (0, 1, 2) → GT (0, 2, 1), with spectral angles of 15.57°, 45.83°, and 41.38°.

### Key Takeaway

> The autoencoder reconstructs the scene well (92% MSE reduction) and keeps abundances on the simplex, **but** endmember recovery is poor (avg spectral angle 34.26°). This confirms the core hypothesis: **reconstruction loss alone does not guarantee identifiable endmembers**. A volume regulariser is needed.

### Generated Outputs

| Output | File |
|--------|------|
| Abundance maps (per component) | `Stage 1 Results/Abundance Maps/` |
| Learned endmember spectra | `Stage 1 Results/Endmember Analysis/` |
| Original vs. reconstructed spectra | `Stage 1 Results/Graphs/` |
| Reconstruction error heatmap | `Stage 1 Results/Graphs/reconstruction_error_heatmap.png` |
| Training loss curve | `Stage 1 Results/Graphs/training_loss_curve.png` |
| Scatter plot (original vs. reconstructed) | `Stage 1 Results/Graphs/reconstruction_scatter.png` |
| Metrics (CSV + JSON) | `Stage 1 Results/Metrics/` |
| Model checkpoint | `Stage1/m.pt` |
| Full tensor results | `Stage1/r.pt` |

---

## Mathematical Core

1. **Separability (pure-pixel) condition** — identifiability via vertex search (VCA, SPA); noisy-robustness bound: error $O(\kappa(E)\,\epsilon)$ (Gillis–Vavasis).
2. **Sufficiently Scattered Condition (SSC)** — minimising $\det(\hat{E}^\top \hat{E})$ subject to data fit recovers $E$ up to permutation, even **without** pure pixels (Fu, Huang, Sidiropoulos).
3. **Transfer to the autoencoder** — decoder weights = endmembers; plain reconstruction loss has **non-identifiable minima**; adding a **log-det volume term** restores identifiability under SSC.

---

## Repository Structure

```
├── README.md
├── DATASETS/
│   └── SAMSON/                          # Samson ENVI data + reference code
├── Stage1/
│   ├── ae.py                            # Autoencoder training script
│   ├── m.pt                             # Saved model state dict
│   ├── r.pt                             # Full tensor results
│   └── stage1_results.md                # Run summary
└── Stage 1 Results/
    ├── Abundance Maps/                  # Per-component abundance maps
    ├── Endmember Analysis/              # Learned endmember spectral curves
    ├── Graphs/                          # Reconstruction plots, loss curve, heatmap
    ├── Metrics/                         # stage1_metrics.csv, .json, loss history
    └── stage1_results_summary.md        # Detailed results summary
```

---

## Getting Started

### Prerequisites

```
Python ≥ 3.9
PyTorch
Spectral Python (SPy)
```

### Installation

```bash
pip install torch spectral
```

### Running Stage 1

```bash
cd Stage1
python ae.py --epochs 2000 --lr 0.001 --output-dir .
```

The script auto-discovers `samson_1.img.hdr` under the project tree. To specify manually:

```bash
python ae.py --data ../DATASETS/SAMSON/.../samson_1.img.hdr --epochs 2000
```

**Outputs:** `m.pt` (model checkpoint) and `r.pt` (reconstructions, abundances, endmembers, loss history).

---

## Roadmap

- [x] **Stage 1** — Autoencoder baseline (reconstruction loss only)
- [ ] **Stage 2** — Minimum-volume / log-det regulariser for identifiable unmixing
- [ ] **Stage 3** — Evaluation on Jasper Ridge, Urban, and Cuprite datasets
- [ ] **Stage 4** — Synthetic phase-transition study (identifiability vs. purity level)
- [ ] **Baselines** — VCA, SPA, MVC-NMF, MiSiCNet, EndNet comparisons

---

## References

1. J. M. Bioucas-Dias et al., *"Hyperspectral unmixing overview: geometrical, statistical, and sparse regression-based approaches,"* IEEE JSTARS 5(2), 2012.
2. J. M. P. Nascimento & J. M. Bioucas-Dias, *"Vertex component analysis: a fast algorithm to unmix hyperspectral data,"* IEEE TGRS 43(4), 2005.
3. N. Gillis & S. A. Vavasis, *"Fast and robust recursive algorithms for separable nonnegative matrix factorization,"* IEEE TPAMI 36(4), 2014.
4. X. Fu, K. Huang, N. D. Sidiropoulos & W.-K. Ma, *"Nonnegative matrix factorization for signal and data analytics,"* IEEE Signal Processing Magazine 36(2), 2019.
5. C.-H. Lin et al., *"Identifiability of the simplex volume minimization criterion for blind hyperspectral unmixing,"* IEEE TGRS 53(10), 2015.
6. L. Miao & H. Qi, *"Endmember extraction from highly mixed data using minimum volume constrained NMF,"* IEEE TGRS 45(3), 2007.
7. B. Palsson et al., *"Hyperspectral unmixing using a neural network autoencoder,"* IEEE Access 6, 2018.
8. B. Rasti et al., *"Image processing and machine learning for hyperspectral unmixing: an overview and the HySUPP Python package,"* IEEE TGRS 62, 2024.

---

## License

Academic use — NIT Jamshedpur CS1503 course project.
