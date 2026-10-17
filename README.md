# SignLab: Traffic Sign Recognition Under Image Degradation

A complete CPS 843 project comparing HOG + a linear SVM, a clean-trained CNN, and the same CNN trained with corruption augmentation. Reported measurements come from experiments on the provided GTSRB dataset.

## Open the demo

Double-click **`launch.bat`**, which opens **http://127.0.0.1:8430**. Double-click **`stop_demo.bat`** to stop the background server. Trained models and a 43-class gallery are included. The full dataset is not needed to use the demo.

Choose an example or upload a **cropped German traffic sign**, select blur, darkness, or noise, and compare predictions. This does classification, not sign detection in full street scenes. SVM scores are decision margins; CNN scores are uncalibrated softmax probabilities.

On this computer, `runtime_path.txt` points to the isolated environment installed for the build. A project-local `.venv` takes priority. If neither exists, the launcher runs `setup.ps1`, requiring Python 3.12 and internet access.

## Set up on another machine

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\python.exe app.py
```

For NVIDIA GPU training, use `setup.ps1 -Cuda`. The default installs CPU PyTorch, which supports the demo. CUDA packages are several GB and are not included in the archive.

## Reproduce the experiments

The dataset root must contain `Train.csv`, `Test.csv`, `signnames.csv`, `Train/`, and `Test/`. Your copy is at `D:\download\data`. Paths beginning with `/code/` are normalized automatically. The source dataset is not modified.

```powershell
.\.venv\Scripts\python.exe -u run_experiment.py --data D:\download\data --epochs 24
.\.venv\Scripts\python.exe make_figures.py
.\.venv\Scripts\python.exe build_report.py --tectonic C:\path\to\tectonic.exe
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If using the build's environment, replace `.\.venv\Scripts\python.exe` with the path in `runtime_path.txt`. Existing matching checkpoints are reused. Reuse requires matching device, hardware/platform, Python, and library versions so the recorded training provenance cannot be silently overwritten. On another machine or CPU, use `--force` or a fresh `--output` to train your own run. These restrictions apply to experiment reuse, not demo inference. An interrupted CNN run is retrained from its seeded initialization on rerun. Use `--force` after changing experiment code/settings. The report and demo expect the main `artifacts` directory.

The first pass reads and resizes 51,839 files and may take several minutes on a hard drive. The resulting cache is omitted from the portable archive; it can be regenerated from the dataset.

## Experimental design

- 43 classes; RGB images resized to 32 × 32, bilinear interpolation, no second ROI crop.
- Seed 42; per-class split by physical-sign track. About 80% of tracks for fitting, 20% for validation. Frames of one sign stay together.
- All 12,630 official test images used only for final measurement.
- HOG: 9 bins, 8 × 8 cells, 2 × 2 blocks, L2-Hys, 324 features. Training-fitted standardization + `LinearSVC(C=1, max_iter=10000)`.
- CNN: three two-convolution blocks (32/64/128 channels), batch normalization, ReLU, max pooling; 256-unit dense head, dropout 0.3, 43 logits.
- Both CNNs: AdamW, learning rate 0.001, weight decay 0.0001, cosine schedule to 0.00001, 24 epochs, batch size 256. Best clean validation macro F1 checkpoint.
- Augmentation: equal probability of clean, blur, darkness, or noise per training image. Full settings in the report and source.
- Test blur radii: 0.6/1.2/1.8. Brightness multipliers: 0.65/0.40/0.20. Noise standard deviations: 0.04/0.08/0.12 on the normalized scale.
- Accuracy, macro F1, confusion matrices, learning curves, per-class scores, and paired outcomes. Corrupted means weight all nine conditions equally.

## Deliverables

| File | Purpose |
|---|---|
| `report/report.pdf` | Compiled report using the official IEEEtran conference class |
| `report/report.tex`, `report/template.tex` | Filled LaTeX and its data-driven template |
| `traffic.py` | Shared data pipeline, transformations, HOG, CNN |
| `run_experiment.py` | Train and evaluate all models |
| `app.py`, `templates/index.html` | Local demo |
| `artifacts/models/` | Fitted SVM and trained CNN checkpoints |
| `artifacts/results/metrics.csv` | All 30 model-condition measurements |
| `artifacts/results/predictions.npz` | Truth and predictions for all conditions |
| `artifacts/results/split.csv` | Exact fitting/validation assignments |
| `artifacts/results/manifest.json` | CSV/code hashes, versions, hardware, settings |
| `artifacts/results/summary.json` | Aggregate scores and paired comparisons |
| `artifacts/figures/` | PNG and vector PDF figures |

## Report details

Author names, affiliation, repository URL, and member-role declarations are intentionally omitted at your request. Before submission, add the correct details and a shareable code link at the end of the abstract. Review and revise the write-up in your own words, as the course instructions require. Nothing has been submitted to D2L or published to GitHub.

This study uses one seed and fixed compact models. Synthetic distortions do not validate real driving, and training includes the same corruption families used for evaluation. No state-of-the-art, significance, or calibrated-confidence claim is made.

## Sources and acknowledgments

- GTSRB: Stallkamp et al., *Man vs. computer*, Neural Networks 32, 323–332, 2012. https://doi.org/10.1016/j.neunet.2012.02.016
- HOG: Dalal and Triggs, CVPR 2005. https://doi.org/10.1109/CVPR.2005.177
- Corruption evaluation: Hendrycks and Dietterich, ICLR 2019. https://arxiv.org/abs/1903.12261
- AdamW: Loshchilov and Hutter, ICLR 2019. https://arxiv.org/abs/1711.05101
- Implementations use PyTorch, scikit-learn, scikit-image, NumPy, Pillow, Matplotlib, and Flask. Original library licenses apply.
- The unmodified official IEEEtran class is by Michael Shell / IEEE, version 1.8b, downloaded from CTAN. License terms are in its source.
- CNNs are trained from scratch; no external model weights or third-party training repository is included.
