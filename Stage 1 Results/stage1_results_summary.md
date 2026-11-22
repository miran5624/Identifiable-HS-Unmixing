# Stage 1 Results — Samson

## Experiment setup
- Scene: Samson
- Image size: 95 × 95 pixels (9,025 pixels)
- Spectral bands: 156
- Autoencoder components: 3
- Model: softmax abundance encoder with bias-free linear decoder
- Objective: reconstruction MSE only; no volume regularizer
- Training: full-batch Adam, learning rate 0.001, 2,000 epochs, seed 0
- Inputs were divided by 1,000 for training; reported reconstruction metrics use original units.

## Reconstruction metrics
- Final scaled training MSE: 0.002981466
- MSE (original units): 2,981.4663
- RMSE (original units): 54.6028
- MAE (original units): 31.9293
- Mean-spectrum baseline MSE: 37,246.1172
- MSE reduction versus baseline: 91.9952%

## Ground-truth comparison
- Mean spectral angle to the ground-truth endmembers: 34.26°
- Best predicted-to-ground-truth permutation: (0, 1, 2) → (0, 2, 1)
- Spectral angles in predicted-component order: 15.57°, 45.83°, and 41.38°
- Abundance RMSE: 0.302463 using the visually verified fortran_to_c Samson ground-truth layout
- Mean abundance entropy: 0.768162
- Maximum abundance-sum error from 1: 1.79 × 10⁻⁷

## Interpretation
The model reconstructs the scene substantially better than the mean-spectrum baseline and its abundance outputs satisfy the simplex constraint. However, the 34.26° mean spectral angle shows that reconstruction quality alone does not ensure recovery of the true endmembers. This is the Stage 1 baseline for comparison with the Stage 2 volume-regularized models.

## Generated outputs
- Abundance maps, endmember spectra, reconstruction plots, and metrics are saved under Stage 1 Results/.
- Model checkpoints and reconstructions (m.pt, r.pt) are generated locally and are not tracked in the repository.
