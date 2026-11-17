# Lab 3 — AIO: Smart Manufacturing & Preventive Maintenance

Two self-contained mini-projects built for **Session IV: Lab Activity - Exercise**, complete with synthetic datasets, **Pre-AI Optimization** baselines, **Post-AI Optimization** pipelines, visualizations, root-cause interpretation, and actionable recommendations.

---

## Project Structure

```text
Lab 3 - AIO/
├── requirements.txt
├── README.md
├── 01_spc_charting/                          # Use Case 1: Statistical Process Control (SPC)
│   ├── generate_data.py                      # Generates 25-sample synthetic SPC CSV + telemetry
│   ├── run_spc.py                            # Pre-AI (Static 3σ) vs. Post-AI (Robust + EWMA + Nelson + IsolationForest)
│   ├── data/spc_measurements.csv
│   └── outputs/
│       ├── spc_pre_vs_post_comparison.png
│       ├── spc_results_annotated.csv
│       └── spc_metrics_summary.json
└── 02_vision_preventive_maintenance/         # Use Case 2: Vision-Based Preventive Maintenance
    ├── generate_data.py                      # Generates 400 synthetic 64x64 RGB camera images
    ├── run_vision_cnn.py                     # Pre-AI (Naive CNN) vs. Post-AI (Normalized + Augmented + Tuned CNN)
    ├── data/parts_dataset/
    │   ├── normal/                           # 200 normal part images
    │   ├── defective/                        # 200 defective images (scratch, pitting, edge_chip)
    │   └── metadata.csv
    └── outputs/
        ├── sample_images.png
        ├── training_curves_comparison.png
        ├── confusion_matrices_comparison.png
        ├── metrics_pre_vs_post.png
        └── vision_metrics_summary.json
```

---

## Use Case 1: Statistical Process Control (SPC) Charting

### Synthetic Dataset (`01_spc_charting/data/spc_measurements.csv`)
- **25 part diameter measurements** (baseline $\mu = 50.0\text{ mm}$, $\sigma = 1.5\text{ mm}$) paired with contextual factory telemetry (`ambient_temp_c`, `vibration_rms`, `operator`, `shift`).
- Injected faults:
  - **Sample 8 (`41.500 mm`)**: Sudden drop caused by raw material inconsistency (undersized stock).
  - **Samples 17–25 (`51.973 mm` → `55.974 mm`)**: Gradual upward tool-wear and thermal/vibration drift (`Drift-Warning` in Samples 17–22, escalating to `Out-of-Control` in Samples 23–25).

### Pre-AI vs. Post-AI Optimization Results

| Metric | Pre-AI (Classical Static $3\sigma$ SPC) | Post-AI (Robust + EWMA + Nelson + Isolation Forest) |
| :--- | :---: | :---: |
| Centerline / Spread | Mean `51.003`, Std `3.052` | Robust Median `51.167`, MAD Std `2.028` |
| Control Limits (`[LCL, UCL]`) | `[41.847, 60.160]` | `[45.082, 57.252]` |
| Precision | `1.000` | `0.778` |
| Recall | `0.100` | `0.700` |
| **F1 Score** | **`0.182`** | **`0.737`** |
| Flagged Samples | `1` (Sample 8 only) | `9` (Sample 8 + Drift Samples 20–25 + 2 inertia points) |
| First Drift Alert (Samples 17–25) | **Never detected** | **Sample 20** (3 samples *before* `Out-of-Control` at Sample 23) |

### Key Pointers & Interpretation
- **Why Pre-AI failed on the drift:** Computing $\mu$ and $\sigma$ directly on a small 25-sample window that already contains an outlier (`41.5 mm`) and drift inflated $\sigma$ from `1.5` to `3.052`, widening `UCL` to `60.160 mm` and hiding the entire tool-wear trend.
- **Root Causes Identified by Post-AI:**
  - **Sample 8 (`41.500 mm`)**: Sudden single-point drop without sustained vibration/temperature rise → *Raw material inconsistency / undersized stock*.
  - **Samples 20–25 (`52.901` → `55.974 mm`)**: Monotonic diameter climb paired with `vibration_rms` rising from `0.556` to `0.876` and `ambient_temp_c` rising from `22.87°C` to `25.84°C` → *Gradual tool wear, thermal drift, and machine calibration shift*.

### Recommendations & Improvements (Use Case 1)
1. **Lock Baseline Limits on "Golden Run" Data:** Compute $\mu_0$ and $\sigma_0$ from verified in-control batches (or use Median/MAD) rather than re-estimating from fault-contaminated windows.
2. **Reset EWMA After Confirmed Spikes:** Resetting the EWMA accumulator after a confirmed single-point outlier (like Sample 8) prevents inertia false positives on Samples 9–10.
3. **Add Remaining Useful Life (RUL) Forecasting:** Fit a rolling linear/Kalman trend on the Post-AI drift segment to predict *when* the tool will breach `UCL` (e.g., "Schedule tool change within 2 batches").

---

## Use Case 2: Vision-Based Preventive Maintenance (CNN)

### Synthetic Dataset (`02_vision_preventive_maintenance/data/parts_dataset/`)
- **400 synthetic `64x64` RGB camera images** (`200` normal, `200` defective: `81` scratch, `66` pitting, `53` edge_chip), split **70% Train (280) / 15% Val (60) / 15% Test (60)**.

### Pre-AI vs. Post-AI Optimization Results (Test Set)

| Metric | Pre-AI (Naive Baseline CNN) | Post-AI (Normalized + Augmented + Tuned CNN) |
| :--- | :---: | :---: |
| Decision Threshold | `0.50` (default) | `0.70` (validation-tuned for F1) |
| **Accuracy** | `53.33%` | **`88.33%`** |
| Precision | `100.00%` (only 2 positive predictions) | `87.10%` |
| **Recall (Defect Capture)** | `6.67%` (missed 28/30 defects) | **`90.00%`** (caught 27/30 defects) |
| **F1 Score** | **`0.1250`** | **`0.8852`** |

### Key Pointers & Observations
- **Pre-AI Bottlenecks:** Unnormalized `[0, 255]` pixel inputs with a high fixed learning rate (`0.01`) caused unstable initial gradients (`loss: 76.25` in Epoch 1) and a shallow network that collapsed to predicting almost everything as `Normal` (`Recall = 6.67%` — unacceptable in factory QA).
- **Post-AI Gains:**
  - Scaling pixels to `[0, 1]` + `Adam(1e-3)` with `ReduceLROnPlateau` stabilized convergence.
  - Data augmentation (`RandomFlip`, `RandomRotation`, `RandomContrast`) prevented overfitting on the small 280-image training split.
  - Using `MaxPooling2D` + `Flatten` preserved small localized defects (2px scratches and pits) that `GlobalAveragePooling2D` would otherwise average out.

### Recommendations & Improvements (Use Case 2)
1. **Prioritize Recall Over Accuracy in QA:** Missing a defective part (`False Negative`) is far costlier than re-inspecting a flagged part (`False Positive`). Tune the decision threshold to guarantee $\ge 95\%$ Recall if safety-critical.
2. **Add Grad-CAM Heatmaps:** Overlay Grad-CAM activations on test images so operators can see *where* the CNN detected a scratch, pit, or edge chip.
3. **Transfer Learning for Real Factory Cameras:** For higher-resolution real camera feeds (`224x224+`), swap the custom CNN for a lightweight pretrained backbone (`MobileNetV3Small` or `EfficientNetB0`).
4. **Unified Lab Extension:** Connect Use Case 1 and Use Case 2 in a single loop—whenever **SPC** flags an out-of-control or drift sample, automatically trigger the **Vision CNN** to inspect the part's camera frame.

---

## How to Run

```bash
# 1. Activate the virtual environment
source ".venv/bin/activate"

# 2. Run Use Case 1 (SPC Charting)
python 01_spc_charting/generate_data.py
python 01_spc_charting/run_spc.py

# 3. Run Use Case 2 (Vision CNN)
python 02_vision_preventive_maintenance/generate_data.py
python 02_vision_preventive_maintenance/run_vision_cnn.py
```
