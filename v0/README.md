# v0 — Baseline Sentiment Model

This version implements a **baseline sentiment analysis model** using classical machine learning.

---

## Objective

- Establish a strong and interpretable baseline
- Provide a reference point for later improvements (v1)
- Demonstrate a standard ML workflow:
  - data loading
  - preprocessing
  - training
  - cross-validation
  - test evaluation

---

## Model

- TF-IDF vectorization
- Logistic Regression (primary)
- Multinomial Naive Bayes (comparison)

---

## Data Handling

- Dataset: Stanford IMDb (aclImdb)
- Data is downloaded and extracted **automatically**
- Uses:
  - `train/` split for training and cross-validation
  - `test/` split for final evaluation

No manual data setup is required.

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
```

### Run v0

```bash
python -m v0.v0_auto
```

---

## Output

After running, v0 will generate:

- Cross-validation metrics printed to stdout
- Test accuracy printed to stdout
- Saved best model artifact: `best_model_v0_*.joblib`
- Saved metrics file: `metrics_v0_*.json`

These artifacts are reused by **v2** for inference and system-level evaluation.
