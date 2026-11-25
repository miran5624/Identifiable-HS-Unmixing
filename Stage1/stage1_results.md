# Stage 1 — Basic Autoencoder Results

## Run setup

- Dataset: Samson, 95 x 95 pixels, 156 bands, 3 endmembers (9,025 pixels).
- Model: encoder predicts 3 softmax abundances; bias-free linear decoder.
- Objective: reconstruction mean squared error, without volume regularization.
- Training: full-batch Adam, learning rate 0.001, 2,000 epochs, seed 0.
- Input values divided by 1,000 during training; reconstruction and endmembers are saved in original units.

## Recomputed numerical results

| Measure | Result |
| --- | ---: |
| Final scaled reconstruction MSE | 0.00296218 |
| Final reconstruction MSE, original units | 2962.177013 |
| Reconstruction RMSE, original units | 54.425886 |
| Reconstruction MAE, original units | 31.681628 |
| Mean-spectrum baseline MSE | 37246.120891 |
| MSE reduction vs mean-spectrum baseline | 92.0470% |
| Mean abundance entropy | 0.769784 |
| Maximum abundance-sum error | 1.34110451e-07 |
| Mean matched endmember SAD | 34.3543 degrees |
| Abundance RMSE, Fortran-to-C GT layout | 0.304488 |

The predicted-to-ground-truth component permutation is [0, 2, 1]; SAD by predicted component is [15.574054697897669, 46.10851458439507, 41.38046641115759] degrees.

These numbers were recomputed from the fresh `Stage1/r.pt` generated in this project copy and the bundled Samson ground truth. The original archive's prediction checkpoint was absent, so its earlier report values could not be reproduced from that checkpoint. No graph or visual artifact was used to calculate these metrics.
