# v1 — Improved Model

v1 extends v0 by improving the model pipeline while reusing the same dataset acquisition
and evaluation protocol.

This version improves upon v0 by introducing a more advanced modeling pipeline.

---

## Objective

- Improve model expressiveness and performance
- Demonstrate iterative model development
- Keep the same dataset and evaluation protocol as v0

---

## Model Improvements

Compared to v0:
- TF-IDF + dimensionality reduction (SVD)
- Two-layer neural network
- Feature normalization
- More expressive decision boundary

---

## Data Consistency

- Uses the **same IMDb dataset** as v0
- Same train/test split
- Same preprocessing logic (via `common`)
- When `--data_dir` is set to `AUTO` (default), v1 reuses v0’s automated dataset
  pipeline for downloading, extracting, and caching the IMDb dataset.

This ensures performance gains come from modeling improvements, not data leakage.

---

## How to Run

### Using a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

**Windows (Command Prompt / cmd)**
```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r ..\requirements.txt
```

### Run v1

```bash
python -m v1.v1_auto
```

By default, the dataset is automatically prepared via **v0’s data pipeline**:
- If `--data_dir` is set to `AUTO` (default), v1 reuses v0’s automated process for
  downloading, extracting, and caching the IMDb dataset.
- This guarantees the same train/test split and data consistency as v0.

---

## Output

- Learning curve visualization (`nn_learning_curve_*.png`)
- Cross-validation results (`nn_cv_curve_*.json`)
- Saved best model artifact (`best_model_v1_nn_*.joblib`)

These artifacts are reused directly by **v2**.
