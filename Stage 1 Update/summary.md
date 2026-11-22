# Stage 1 Update — Samson Baseline

## Dataset
- Image: 95 x 95 pixels
- Spectral bands: 156
- Endmembers: 3
- Ground-truth abundance layout: fortran_to_c

## Reconstruction metrics
- MSE: 2981.466309
- RMSE: 54.602806
- MAE: 31.929304
- Scaled MSE: 0.00298147
- MSE reduction against mean-spectrum baseline:
  91.9952%

## Ground-truth evaluation
- Mean endmember SAD: 34.26 degrees
- Component SAD in predicted-component order:
  [15.57, 45.83, 41.38]
- Predicted-to-ground-truth matching permutation:
  [0, 2, 1]
- Abundance RMSE: 0.302463
- Mean abundance entropy: 0.768162
- Maximum abundance-sum error: 1.7881393432617188e-07

## Important provenance note
The Stage 1 images are retained from the existing generated results.
The updated numerical evaluation above uses Samran's corrected
ground-truth layout and metrics.

The original Stage 1 prediction checkpoint is not present in this
repository snapshot. Therefore, these images have not been
recomputed from raw Stage 1 predictions.

Component labels refer to the original Stage 1 model order.
Use the matching permutation above when comparing material identities.
