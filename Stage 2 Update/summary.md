# Stage 2 Results — Samson

## Objective and setup
- Dataset: Samson (95 x 95 pixels, 156 bands, 3 endmembers).
- Training: full-batch Adam, 2,000 epochs per run, seeds 0, 1, and 2.
- Sweep: warm-up epochs 0 and 200; log-det weights 0, 1e-6, 1e-5, 1e-4, 1e-3, and 1e-2.
- Ground-truth abundance layout: `fortran_to_c`; endmembers matched using Hungarian assignment on spectral angle.
- All 36 runs were freshly trained and evaluated with Python 3.13, PyTorch 2.11.0+cpu, and NumPy 2.4.2.

## Recomputed sweep results
- Runs: 36; grouped configurations: 12.
- Lowest mean SAD: 14.6061 degrees (lambda=0.001, warm-up=0).
- Lowest mean abundance RMSE: 0.315213 (lambda=1e-05, warm-up=0).
- Both metric tables were rebuilt from the freshly generated checkpoints.

## Seed-0 example (warm-up 0, lambda=0.001)
- Mean SAD: 8.850217 degrees.
- Abundance RMSE: 0.311409.
- Final scaled reconstruction MSE: 0.000542661.
- Final log-det value: 3.933120.

## Matched endmember magnitude ratios
Ratios compare learned endmember norms in original data units with ground-truth endmember norms scaled to those same units. Components are listed in matched ground-truth order.

| Component | Control | Log-det |
|---|---:|---:|
| 1 | 0.8394 | 0.8372 |
| 2 | 1.2169 | 1.2156 |
| 3 | 0.0866 | 0.0860 |

The saved graphs were not regenerated during this numerical refresh and may still show previous runs. No graph was used to calculate or validate these metrics.
