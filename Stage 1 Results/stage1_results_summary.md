# Stage 1 Results — Samson

## Experiment setup
- Scene: Samson
- Image size: 95 × 95 pixels
- Spectral bands: 156
- Autoencoder components: 3
- Pixels evaluated: 9025

## Reconstruction metrics
- MSE (original units): 2960.516113
- RMSE (original units): 54.410625
- MAE (original units): 31.649477
- Final scaled training MSE: 0.00296052

## Generated outputs
- Predicted abundance maps for each component
- Learned endmember spectral curves
- Centre-pixel and scene-mean original/reconstructed spectra
- Per-pixel reconstruction RMSE heatmap
- Training loss curve (when loss history is present)
- Original/reconstructed scatter plot
- CSV and JSON metrics

## Interpretation note
These are Stage 1 autoencoder outputs and reconstruction metrics. The abundance maps are model-predicted abundances. They are not ground-truth abundance maps, and reconstruction error alone does not establish endmember identifiability or successful ground-truth endmember recovery.
