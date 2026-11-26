# Stage 3 Figure Manifest

## Generated figures
- Graphs: 4
- Abundance Maps: 3
- Endmember Analysis: 4
- Total: 11 PNG figures

## Validation
- 18 active real-dataset checkpoint records were independently evaluated.
- Existing per-run metrics matched the independent evaluation with zero mismatches.
- Checkpoint arrays were checked for expected shapes, finite values, non-negative abundances, and abundance sums.
- Endmember matching was recomputed using minimum total spectral angle.
- Jasper Ridge and Urban abundance RMSE values were recomputed and checked against the validated CSV.
- Synthetic purity CSVs contain 300 trial rows and 60 summary rows; 5 trials are present in each of 60 purity/method groups. Summary means, standard deviations, and success flags were independently checked.

## Interpretation limits
- There is one saved seed per real-dataset/method combination; standard deviations across real-dataset runs are therefore not estimable.
- Cuprite has ground-truth endmember spectra but no ground-truth abundance matrix. Its predicted abundance maps are shown without ground-truth abundance comparisons or abundance RMSE.
- Jasper Ridge and Urban checkpoint clipping metadata is blank in the existing records. Blank is treated as unknown, not as zero clipped samples.
- The historical records in real_datasets_metrics_missing_checkpoints.csv are not used in these figures because their raw checkpoints are unavailable.
- The synthetic experiment has five trials per purity and method. It uses the available Cuprite ground-truth spectra as the library fallback because no separate USGS library file is present in the dataset directory. Its results should be described as preliminary synthetic evidence, not proof of identifiability.

## Component matching
Predicted endmember and abundance components are reordered using Hungarian assignment on the spectral-angle matrix, with ground-truth component order as the reference.
