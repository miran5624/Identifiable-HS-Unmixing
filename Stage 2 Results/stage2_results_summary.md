# Stage 2 Results — Samson

## Objective
Evaluate the log-det volume penalty for hyperspectral endmember recovery using the Samson dataset.

## Experimental setup
- Dataset: Samson
- Image: 95 x 95 pixels
- Spectral bands: 156
- Endmembers: 3
- Recorded experiment evaluations: 36
- Grouped configurations: 12
- Ground truth: samgt.mat
- Ground-truth abundance layout: fortran_to_c
- Endmember matching: Hungarian assignment using SAD

## Parameter sweep results
- Best mean SAD: 14.6007 degrees
- SAD standard deviation: 8.4941
- Best SAD configuration: lambda=0.001, warm-up=0, init=random
- Best mean abundance RMSE: 0.315533
- RMSE standard deviation: 0.003677
- Best RMSE configuration: lambda=0.001, warm-up=200, init=random

## Independently re-evaluated saved checkpoints
- Control: SAD=8.7379 degrees; abundance RMSE=0.311394
- Log-det: SAD=8.8527 degrees; abundance RMSE=0.311507

## Interpretation
- The parameter sweep is summarized from the recorded 36-run evaluation CSV.
- Only two raw checkpoints are available for direct independent spectral and abundance-map comparisons.
- In these two saved checkpoints, the control has slightly lower SAD and abundance RMSE than the lambda=0.001 checkpoint.
- Normalized endmember curves compare spectral shape, not absolute magnitude.
- These Samson experiments do not prove the sufficient-scattering identifiability theorem.

## Generated graphs
1. Graphs/01_mean_SAD_vs_logdet_weight.png
2. Graphs/02_abundance_RMSE_vs_logdet_weight.png
3. Graphs/03_endmember_spectra_vs_ground_truth.png
4. Graphs/04_abundance_maps_vs_ground_truth.png

## Metrics files
- Metrics/samson_sad_rmse_per_run.csv
- Metrics/samson_sad_rmse_summary_by_lambda.csv
- Metrics/stage2_saved_checkpoint_metrics_recomputed.csv
- Metrics/stage2_saved_checkpoint_summary.csv

## Endmember magnitude limitation

The spectral curves in Graph 03 are L2-normalized to compare spectral shapes.
This normalization does not demonstrate recovery of absolute endmember
magnitudes.

The learned-to-ground-truth endmember norm ratios for the matched components
were approximately:

| Component | Control | Log-det |
|---|---:|---:|
| 1 | 0.8390 | 0.8372 |
| 2 | 1.2163 | 1.2156 |
| 3 | 0.0866 | 0.0860 |

The third learned endmember has substantially lower magnitude than its
ground-truth counterpart. This is an important limitation when interpreting
endmember recovery.

Only two raw checkpoints were available for independent re-evaluation.
The remaining 34 experiment records were taken from the existing evaluation
CSV and were not independently reproduced from raw checkpoints.
