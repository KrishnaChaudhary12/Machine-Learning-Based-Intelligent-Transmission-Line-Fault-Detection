# Machine-Learning-Based-Intelligent-Transmission-Line-Fault-Detection
# ML-Based Intelligent Transmission Line Fault Detection

Machine learning system for detecting, classifying, and locating faults on a high-voltage transmission line — built end-to-end from physics-based Simulink simulation through feature engineering to model training, and directly benchmarked against a conventional distance relay.

## Motivation

Conventional distance relays estimate fault location using apparent impedance (Z = V/I) at the relay. This works well for low-resistance faults, but accuracy degrades sharply as fault resistance increases — in a companion project ([Distance Relay vs. Overcurrent Relay Protection Schemes](#)), a conventional distance relay's zone classification **failed outright above ~50 Ω fault resistance** due to zero-sequence impedance inflation.

This project asks: can a machine learning model, trained on the same underlying electrical signals, detect and locate faults more robustly across a wider range of fault resistances?

## Pipeline

```
Simulink (MATLAB)  →  Feature Extraction (MATLAB)  →  Model Training (Python)
   physics-based           RMS, symmetrical              Random Forest,
   fault simulation        components, harmonics,        SVM, XGBoost
                           wavelets, impedance
```

| Stage | Tool | Output |
|---|---|---|
| 1. Data Generation | MATLAB / Simulink | `data/fault_dataset_raw.mat` — 900 labeled fault events (not included in repo, regenerate via script) |
| 2. Feature Extraction | MATLAB | `data/features.csv` — 900 rows × 41 columns |
| 3. Model Training & Evaluation | Python (scikit-learn, XGBoost) | Trained models + evaluation metrics |

## System Simulated

- 220 kV, 150 km transmission line, split into two PI-section segments with a fault block at the junction (allows fault location to be swept programmatically)
- 5 fault types: LG, LL, LLG, LLL, LLLG, plus a no-fault (NONE) baseline — 150 samples per class, 900 total
- Randomized per sample: phase combination, fault location (5–145 km), fault resistance (0–100 Ω, banded low/med/high), fault inception timing, load level (70–130%), and source voltage (±5%)
- Automated validation rejects unstable/invalid simulation runs and guarantees exactly 150 successful samples per class

## Results

| Task | Metric | Result |
|---|---|---|
| Fault Detection | Accuracy (Random Forest) | 100% |
| Fault Type Classification | Overall accuracy | **96.0%** (Random Forest), 95.6% (XGBoost), 80.9% (SVM) |
| Fault Location Estimation | Mean Absolute Error | **9.82 km** (6.5% of line length) |
| Generalization | Holdout set (independent random seed) | 11.5 km MAE — consistent with test set |

### Performance vs. Fault Resistance — the key comparison

| Resistance Band | Classification Accuracy | Location MAE |
|---|---|---|
| 0 – 5 Ω | 95.8% | 6.71 km |
| 5 – 30 Ω | 97.9% | 8.83 km |
| 30 – 100 Ω | **90.9%** | **18.62 km** |

Unlike the conventional distance relay — which failed to trip correctly above 50 Ω — the ML models degrade *gradually* across the full tested resistance range, remaining usable even at high fault resistance.

### Key findings

- The only systematic classification weakness is distinguishing **LLL from LLLG** faults — physically explainable, since a balanced three-phase fault produces almost no zero-sequence current regardless of ground involvement.
- SVM underperforms specifically on that same LLL/LLLG pair (43–44% F1) while matching tree-based models elsewhere — its smooth RBF decision boundary is less suited to this particular overlapping-class distinction.
- Fault-type classification is far more robust to fault resistance than location estimation (7-point accuracy drop vs. nearly tripled location error, low → high resistance).


## Repository Structure

```
├── simulink/
│   └── ml_project.slx              # Transmission line + fault model
├── matlab/
│   ├── generate_fault_dataset.m    # Automated simulation + validation (Stage 1)
│   ├── extract_features.m          # Feature extraction (Stage 2)
│   └── generate_holdout_test.m     # Independent holdout-set generation
├── python/
│   └── train_models.ipynb          # Model training, evaluation, comparison (Stage 3)
├── data/
│   └── features.csv                # Extracted feature dataset (900 x 41)
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance_classification.png
│   ├── feature_importance_location.png
│   └── model_comparison.png
└── docs/
    └── project_report.pdf          # Full written report
```

## How to Run

**Stage 1–2 (requires MATLAB with Simscape Electrical + Wavelet Toolbox):**
```matlab
generate_fault_dataset   % produces fault_dataset_raw.mat
extract_features         % produces features.csv
```

**Stage 3 (Python):**
```bash
pip install -r requirements.txt
jupyter notebook python/train_models.ipynb
```
`features.csv` is already included, so Stage 3 can be run independently without MATLAB.

## Tools & Libraries

MATLAB, Simulink, Simscape Electrical, Wavelet Toolbox · Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, Jupyter Notebook

## Author

Krishna Chaudhary — Electrical Engineering, Motilal Nehru National Institute of Technology Allahabad
