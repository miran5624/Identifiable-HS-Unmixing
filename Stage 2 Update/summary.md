# Stage 2 Update — Log-det Volume Autoencoder

## Dataset and configuration
- Dataset: Samson
- Image dimensions: 95 x 95 pixels
- Spectral bands: 156
- Endmembers: 3
- Epochs: 2000
- Initialization: random
- Random seed: 0
- Log-det weight: 0.001
- Warm-up epochs: 0
- Input scale: 1000.0
- Ground-truth abundance layout: fortran_to_c

## Saved checkpoint
- Final reconstruction MSE: 542.554718
- Final scaled reconstruction MSE: 0.00054255
- Final log-det value: 3.933033
- Abundance shape: (9025, 3)
- Endmember shape: (156, 3)

## Evaluation provenance
- 36 experiment records and 12 grouped configurations are retained
  in the parameter-sweep metric CSV files.
- Two raw checkpoints are available for independent re-evaluation.
- The remaining 34 runs are represented by recorded CSV metrics and
  cannot be independently rerun from raw checkpoints in this repository.

## Component-order note
The maps and endmember plots retain the model's native component order.
Component numbers are not guaranteed to identify the same material
across different models; use the recorded Hungarian matching permutation
when comparing components.

## Graphs
Six graph types matching the Stage 1 presentation:
1. Centre-pixel original vs reconstructed spectrum.
2. Scene-mean original vs reconstructed spectrum.
3. Combined predicted abundance maps.
4. Per-pixel reconstruction RMSE heatmap.
5. Original vs reconstructed scatter.
6. Training reconstruction-loss curve.

## Additional outputs
- Three individual abundance maps.
- Three individual endmember plots and one combined plot.
- Parameter sweep graphs.
- Ground-truth comparison graphs.
- Four evaluation CSV files.
