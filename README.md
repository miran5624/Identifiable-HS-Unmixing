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
| **Loss** | MSE reconstruction loss (Stage 1); + log-det volume regulariser (Stage 2 & 3) |
| **Optimizer** | Full-batch Adam, lr = 0.001 |
| **Training** | 2,000 epochs, seed = 0 |
| **Non-negativity** | Decoder weights clamped to ≥ 0 after each step (Stage 2 & 3) |

---

## Datasets

### Samson Scene *(Stages 1 & 2)*

| Property | Value |
|----------|-------|
| Image size | 95 × 95 pixels (9,025 total) |
| Spectral bands | 156 (401–889 nm, 3.13 nm resolution) |
| Endmembers | 3 — **Soil**, **Tree**, **Water** |
| Format | ENVI (`.img` + `.img.hdr`) |
| Ground truth | `samgt.mat` (endmember matrix M + abundance matrix A) |
| GT abundance layout | `fortran_to_c` (visually verified) |
| Input scaling | Reflectance values ÷ 1,000 during training |

### Stage 3 Benchmark & Synthetic Datasets *(Uploaded)*

| Dataset | Size | Bands | Endmembers | Files (`DATASETS/`) |
|---------|------|-------|------------|---------------------|
| **Jasper Ridge** | 100 × 100 | 198 | 4 | `JASPER/jasperRidge2_R198.mat`, `end4.mat` |
| **Urban (HYDICE)** | 307 × 307 | 162 | 4 | `URBAN/Urban_R162.mat`, `end4_groundTruth.mat` |
| **Cuprite (AVIRIS)** | Mineral scene | 224 | 12 | `CUPRITE/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat`, `groundTruth_Cuprite_nEnd12.mat` |
| **Synthetic (USGS)** | Variable | — | 3–6 | Sampled directly from USGS mineral spectra in `groundTruth_Cuprite_nEnd12.mat` |

---

## Stage 1 Results — Autoencoder Baseline

> **Status: Complete**

### Reconstruction Metrics

| Metric | Value |
|--------|------:|
| Reconstruction MSE (original units) | **2,981.47** |
| Reconstruction RMSE | **54.60** |
| Reconstruction MAE | **31.93** |
| Final Scaled Training MSE | 0.002981 |
| Mean-Spectrum Baseline MSE | 37,246.12 |
| **MSE Reduction vs. Baseline** | **92.0%** |

### Identifiability Metrics (Ground-Truth Comparison)

| Metric | Value |
|--------|------:|
| Mean Spectral Angle to GT Endmembers | **34.26°** |
| Abundance RMSE vs. Ground Truth | **0.3025** |
| Mean Abundance Entropy | 0.7682 |
| Max Abundance Sum Error from 1 | 1.79 × 10⁻⁷ |

**Best endmember permutation:** predicted (0, 1, 2) → GT (0, 2, 1), with spectral angles of 15.57°, 45.83°, and 41.38° in predicted-component order. The abundance RMSE uses the visually verified `fortran_to_c` Samson ground-truth layout.

### Key Takeaway

> The autoencoder reconstructs the scene well (92% MSE reduction) and keeps abundances on the simplex, **but** endmember recovery is poor (avg spectral angle 34.26°). This confirms the core hypothesis: **reconstruction loss alone does not guarantee identifiable endmembers**. A volume regulariser is needed.

### Generated Outputs

| Output | File |
|--------|------|
| Abundance maps (per component) | `Stage 1 Update/Abundance Maps/` |
| Learned endmember spectra | `Stage 1 Update/Endmember Analysis/` |
| Original vs. reconstructed spectra | `Stage 1 Update/Graphs/` |
| Reconstruction error heatmap | `Stage 1 Update/Graphs/reconstruction_error_heatmap.png` |
| Training loss curve | `Stage 1 Update/Graphs/training_loss_curve.png` |
| Scatter plot (original vs. reconstructed) | `Stage 1 Update/Graphs/reconstruction_scatter.png` |
| Metrics (CSV + JSON) | `Stage 1 Update/Metrics/` |
| Model outputs | Generated locally (`m.pt`, `r.pt`); not tracked |

---

## Stage 2 Results — Log-det Volume Regulariser

> **Status: Complete**

Stage 2 adds the **log-det volume penalty** to the autoencoder loss to enforce minimum-volume endmember recovery:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_i \Vert{}x_i - \hat{E}\,a_i\Vert{}^2}_{\text{per-pixel SSE}} + \lambda \cdot \underbrace{\log\det(\hat{E}^\top \hat{E} + \epsilon I)}_{\text{volume term}}$$

The decoder weights are clamped non-negative after each gradient step to enforce $\hat{E} \in \mathbb{R}^{B \times p}_+$.

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| Initialization | Random uniform [0, 1] on scaled reflectance |
| Warm-up variants | 0 and 200 epochs (pure reconstruction before enabling volume penalty) |
| Lambda values swept | 0, 1e-6, 1e-5, 1e-4, 0.001, 0.01 |
| Seeds per config | 3 (seeds 0, 1, 2) |
| Total experiment runs | **36** (12 configurations × 3 seeds) |
| Endmember matching | Hungarian assignment minimising SAD |
| GT abundance layout | `fortran_to_c` (visually verified) |

### Parameter Sweep — Best Configurations

| Metric | Best Value | Configuration |
|--------|-----------|---------------|
| **Mean SAD** | **14.60°** ± 8.49° | λ = 0.001, warm-up = 0, random init |
| **Abundance RMSE** | **0.3155** ± 0.0037 | λ = 0.001, warm-up = 200, random init |

### Saved Checkpoint Comparison (seed = 0)

| Metric | Control (λ = 0) | Log-det (λ = 0.001) |
|--------|----------------:|--------------------:|
| Mean SAD (°) | **8.74** | 8.85 |
| Abundance RMSE | **0.3114** | 0.3115 |
| Reconstruction MSE (scaled) | 0.000551 | 0.000543 |
| Log-det volume | 4.069 | 3.933 |
| Min pairwise endmember angle | 31.00° | 30.10° |

### Endmember Norm Ratios (Learned / Ground Truth)

| Component | Control | Log-det |
|-----------|--------:|--------:|
| 1 | 0.839 | 0.837 |
| 2 | 1.216 | 1.216 |
| 3 | 0.087 | 0.086 |

> Component 3 has substantially lower magnitude than ground truth — an important limitation.

### Key Takeaway

> Across the 36-run sweep, **λ = 0.001 consistently gives the best mean SAD and abundance RMSE**, indicating the log-det penalty is guiding the decoder toward better endmember recovery. However, the two saved seed-0 checkpoints show marginal differences — the control already achieves a low SAD of 8.74° for this seed. The benefit of the volume term is more apparent across multiple seeds and initializations, where it reduces variance and improves worst-case performance. These Samson experiments **demonstrate the regulariser's effect but do not conclusively prove the SSC identifiability theorem**.

### Stage 2 Generated Outputs

| Output | File |
|--------|------|
| Abundance maps (per component) | `Stage 2 Update/Abundance Maps/` |
| Endmember spectra | `Stage 2 Update/Endmember Analysis/` |
| Reconstruction plots | `Stage 2 Update/Graphs/` |
| Endmember spectra vs. ground truth | `Stage 2 Results/Graphs/03_endmember_spectra_vs_ground_truth.png` |
| Abundance maps vs. ground truth | `Stage 2 Results/Graphs/04_abundance_maps_vs_ground_truth.png` |
| Mean SAD vs. λ sweep | `Stage 2 Results/Graphs/01_mean_SAD_vs_logdet_weight.png` |
| Abundance RMSE vs. λ sweep | `Stage 2 Results/Graphs/02_abundance_RMSE_vs_logdet_weight.png` |
| Per-run metrics (36 runs) | `Stage 2 Update/Metrics/samson_sad_rmse_per_run.csv` |
| Summary by λ (12 configs) | `Stage 2 Update/Metrics/samson_sad_rmse_summary_by_lambda.csv` |
| Checkpoint re-evaluation | `Stage 2 Update/Metrics/stage2_saved_checkpoint_metrics_recomputed.csv` |

---

## Stage 3 — Multi-Scene Evaluation, Synthetic Phase-Transition & Baselines

> **Status: In Progress (Datasets Uploaded)**

Stage 3 completes the experimental evaluation required for the final paper:
- **Benchmark Scenes:** Evaluate the log-det autoencoder on **Jasper Ridge**, **Urban (HYDICE)**, and **Cuprite (AVIRIS)**.
- **Synthetic Phase-Transition Study:** Generate synthetic linear mixtures from USGS mineral spectra (`groundTruth_Cuprite_nEnd12.mat`) across varying purity levels to measure identifiability success rate vs. purity.
- **Baselines:** Compare against classical and deep unmixing baselines (**VCA**, **SPA**, **MVC-NMF**, and standard autoencoders) on SAD and abundance RMSE.

---

## Mathematical Core

1. **Separability (pure-pixel) condition** — identifiability via vertex search (VCA, SPA); noisy-robustness bound: error $O(\kappa(E)\,\epsilon)$ (Gillis–Vavasis).
2. **Sufficiently Scattered Condition (SSC)** — minimising $\det(\hat{E}^\top \hat{E})$ subject to data fit recovers $E$ up to permutation, even **without** pure pixels (Fu, Huang, Sidiropoulos).
3. **Transfer to the autoencoder** — decoder weights = endmembers; plain reconstruction loss has **non-identifiable minima**; adding a **log-det volume term** restores identifiability under SSC.

---

## Repository Structure

```
├── README.md
├── build_stage_updates.py               # Generates Stage 1 & 2 update outputs
├── DATASETS/
│   ├── CUPRITE/                         # Cuprite AVIRIS scene + 12-endmember USGS GT
│   ├── JASPER/                          # Jasper Ridge (198 bands) + 4-endmember GT
│   ├── SAMSON/                          # Samson ENVI data + ground truth (samgt.mat)
│   └── URBAN/                           # Urban HYDICE (162 bands) + 4-endmember GT
├── Stage1/
│   ├── ae.py                            # Stage 1 autoencoder training script
│   └── stage1_results.md                # Run summary
├── Stage 1 Update/
│   ├── Abundance Maps/                  # Per-component abundance maps
│   ├── Endmember Analysis/              # Learned endmember spectral curves
│   ├── Graphs/                          # Reconstruction plots, loss curve, heatmap
│   ├── Metrics/                         # stage1_metrics.csv, .json, loss history
│   └── summary.md                       # Stage 1 updated results summary
├── Stage2/
│   ├── logdet_volume_autoencoder.py     # Stage 2 log-det volume training script
│   ├── evaluate_samson_unmixing.py      # Ground-truth evaluation (SAD, RMSE, Hungarian matching)
│   ├── generate_stage2_results.py       # Generates sweep plots & results summary
│   ├── samson_unmixing_*_lambda0.pt     # Control checkpoint (λ=0)
│   └── samson_unmixing_*_lambda0.001.pt # Best log-det checkpoint (λ=0.001)
├── Stage 2 Update/
│   ├── Abundance Maps/                  # Per-component abundance maps (log-det model)
│   ├── Endmember Analysis/              # Learned endmember spectral curves
│   ├── Evaluation Comparisons/          # Endmember & abundance maps vs. ground truth
│   ├── Graphs/                          # Reconstruction plots, loss curve, heatmap
│   ├── Metrics/                         # Per-run CSV, summary CSV, checkpoint metrics
│   ├── Parameter Sweep/                 # SAD vs λ and RMSE vs λ sweep plots
│   └── summary.md                       # Stage 2 results summary
└── Stage 2 Results/
    ├── Graphs/                          # Ground-truth comparison & sweep plots
    ├── Metrics/                         # Evaluation CSVs
    └── stage2_results_summary.md        # Detailed Stage 2 results summary
```

---

## Getting Started

### Prerequisites

```
Python ≥ 3.9
PyTorch
Spectral Python (SPy)
NumPy
SciPy
Matplotlib
```

### Installation

```bash
pip install torch spectral numpy scipy matplotlib
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

### Running Stage 2

```bash
cd Stage2

# Train with log-det volume regulariser (best config)
python logdet_volume_autoencoder.py --lambda-vol 0.001 --epochs 2000

# Train control baseline (no regulariser)
python logdet_volume_autoencoder.py --lambda-vol 0 --epochs 2000

# With warm-up (200 epochs reconstruction-only before enabling volume penalty)
python logdet_volume_autoencoder.py --lambda-vol 0.001 --warmup-epochs 200 --epochs 2000
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--lambda-vol` | 1e-4 | Weight on log-det volume term (0 = control baseline) |
| `--warmup-epochs` | 0 | Pure-reconstruction epochs before enabling volume penalty |
| `--init` | `random` | Decoder init: `random` (uniform [0,1]) or `data` (farthest-point) |
| `--seed` | 0 | Random seed for reproducibility |
| `--vol-eps` | 1e-4 | Diagonal stabiliser δ in log det(E⊤E + δI) |

### Evaluating Against Ground Truth

```bash
cd Stage2

# Evaluate all checkpoints in Stage2/
python evaluate_samson_unmixing.py \
    --csv ../metrics_per_run.csv \
    --summary-csv ../metrics_summary.csv

# Generate comparison plots
python generate_stage2_results.py
```

---

## Roadmap

- [x] **Stage 1** — Autoencoder baseline (reconstruction loss only)
- [x] **Stage 2** — Log-det volume regulariser for identifiable unmixing + 36-run parameter sweep
- [x] **Stage 3 Setup** — Jasper Ridge, Urban, and Cuprite datasets uploaded to `DATASETS/`
- [ ] **Stage 3 Experiments** — Multi-scene evaluation, synthetic purity phase-transition study, and baseline comparisons (VCA, SPA, MVC-NMF, etc.)
- [ ] **Final Paper** — Write-up and figure integration

---

## References

1. J. M. Bioucas-Dias et al., *"Hyperspectral unmixing overview: geometrical, statistical, and sparse regression-based approaches,"* IEEE JSTARS 5(2), 2012.
2. J. M. P. Nascimento & J. M. Bioucas-Dias, *"Vertex component analysis: a fast algorithm to unmix hyperspectral data,"* IEEE TGRS 43(4), 2005.
3. N. Gillis & S. A. Vavasis, *"Fast and robust recursive algorithms for separable nonnegative matrix factorization,"* IEEE TPAMI 36(4), 2014.
4. X. Fu, K. Huang, N. D. Sidiropoulos & W.-K. Ma, *"Nonnegative matrix factorization for signal and data analytics: identifiability, algorithms, and applications,"* IEEE Signal Processing Magazine 36(2), 2019.
5. C.-H. Lin et al., *"Identifiability of the simplex volume minimization criterion for blind hyperspectral unmixing,"* IEEE TGRS 53(10), 2015.
6. L. Miao & H. Qi, *"Endmember extraction from highly mixed data using minimum volume constrained NMF,"* IEEE TGRS 45(3), 2007.
7. B. Palsson et al., *"Hyperspectral unmixing using a neural network autoencoder,"* IEEE Access 6, 2018.
8. B. Rasti et al., *"Image processing and machine learning for hyperspectral unmixing: an overview and the HySUPP Python package,"* IEEE TGRS 62, 2024.

---

## License

Academic use — NIT Jamshedpur CS1503 course project.
