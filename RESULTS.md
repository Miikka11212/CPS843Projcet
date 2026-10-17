# Measured results

Official GTSRB test set: 12,630 images per condition. Seed: 42.

| Model | Clean accuracy | Clean macro F1 | Mean corrupted accuracy | Mean corrupted macro F1 |
|---|---:|---:|---:|---:|
| HOG + SVM | 77.55% | 71.44% | 55.55% | 49.75% |
| Clean CNN | 98.38% | 97.75% | 88.18% | 85.33% |
| Augmented CNN | 97.33% | 95.87% | 94.47% | 92.38% |

Augmentation changes mean corrupted accuracy by **+6.29 percentage points**.

The corrupted mean weights all three severities of blur, darkness, and noise equally. These are synthetic transformations from the same families used in augmentation. This is a single-seed experiment, with no statistical significance claim.

Full per-condition measurements: `artifacts/results/metrics.csv`. Raw predictions: `artifacts/results/predictions.npz`. The IEEE report includes methods, figures, paired outcomes, and limitations.
