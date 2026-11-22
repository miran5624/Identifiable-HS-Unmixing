# Stage 1: Basic Autoencoder Results

## Run setup

- Dataset: Samson, 95 x 95 pixels, 156 bands, 3 endmembers (9,025 pixels total)
- Model: encoder outputs 3 softmax abundances; bias-free linear decoder
- Objective: mean squared reconstruction error only; no volume regularizer
- Training: full-batch Adam, learning rate 0.001, 2,000 epochs, seed 0
- Input handling: ENVI values divided by 1,000 during training; reconstructed spectra and decoder endmembers are saved in original units

## Results

| Measure | Result |
| --- | ---: |
| Final scaled reconstruction MSE | 0.00298147 |
| Final reconstruction MSE in original units | 2,981.47 |
| Reconstruction RMSE in original units | 54.60 |
| Reconstruction MAE in original units | 31.93 |
| Mean-spectrum baseline MSE | 37,246.12 |
| MSE reduction against mean-spectrum baseline | 92.00% |
| Mean abundance entropy | 0.7682 |
| Maximum abundance sum error from 1 | 1.79e-7 |
| Mean spectral angle to ground-truth endmembers (permutation matched) | 34.26 degrees |
| Abundance RMSE against ground truth (same matching) | 0.3025 |

The best endmember matching maps predicted components 0, 1, 2 to ground-truth components 0, 2, 1. Their spectral angles are 15.57, 45.83, and 41.38 degrees, respectively. Abundance RMSE uses the visually verified fortran_to_c Samson ground-truth layout.

## Interpretation

The scaled model learns a useful reconstruction: its MSE is about 92% lower than the constant mean-spectrum baseline, and the abundances remain on the probability simplex. The ground-truth comparison is much weaker: average endmember angle is 34.26 degrees and abundance RMSE is 0.3025. This is consistent with the project motivation: reconstruction loss alone does not ensure identifiable endmembers or accurate abundances. Stage 1 is a working basic-autoencoder baseline; the later minimum-volume/identifiability stage is still needed.

The reported values are training-scene results, not held-out generalization metrics. Running the script writes m.pt and r.pt to the selected output directory; these are generated locally and are not tracked in the repository.

## Reproducing the run

From the project folder, with PyTorch and Spectral installed, run:

python ae.py --epochs 2000 --output-dir results
