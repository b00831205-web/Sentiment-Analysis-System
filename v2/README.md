# v2 — System Integration

v2 extends v1 by integrating trained model artifacts into a complete system,
including a CLI and a web-based interactive interface.

No model retraining is performed in this stage.

The purpose of v2 is to demonstrate **systemization**, not modeling.

---

## Key Features

- Command-line interface (CLI)
- Web server with interactive UI
- Automated reuse of trained model artifacts (v0, v1)
- Automated dataset download and reuse
- Logging and basic tests

---

## Feature A — Single Text Prediction

- Input: arbitrary user text
- Output: sentiment label and probability
- Supported models:
  - v0 (Logistic Regression + TF-IDF)
  - v1 (Neural Network)

This feature demonstrates **online inference** using pre-trained models.

---

## Feature B — Keyword-Based Aggregated Analysis (Local Dataset)

- Input:
  - Keyword (any term)
  - Model selection (v0 / v1)
  - Split selection (train / test / both)
  - Maximum number of reviews
- Operation:
  - Match reviews in the local IMDb dataset by keyword
  - Batch inference using v0 or v1
  - Aggregate sentiment statistics
- Output:
  - Average positive probability
  - Positive ratio
  - Recommendation index
  - Representative positive/negative examples

This feature focuses on **topic-level sentiment aggregation**.

> Note: The IMDb dataset does not contain explicit movie-title metadata.
> This feature serves as an offline approximation of keyword-based review analysis.

---

## Feature C — Movie-Level Recommendation (Test Set Only)

- Input:
  - Movie title (free text)
  - Model selection (v0 / v1)
  - Maximum number of reviews
- Constraints:
  - Uses **test split only**
  - Split selection is intentionally fixed and not exposed to the user
- Operation:
  - Match reviews in the test set using robust movie-title phrase matching
  - Batch inference and aggregation
- Output:
  - Overall recommendation index (0–100)
  - Positive ratio
  - Average positive probability

This feature approximates a **movie-level recommendation signal**
under dataset constraints.

Example movie titles that work well with the IMDb dataset include:
- Titanic (positive recommendation)
- Spider-Man (positive recommendation)
- Plan 9 from Outer Space (negative recommendation)
- Battlefield Earth  (negative recommendation)

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

### CLI

```bash
python -m v2.cli predict --model v0 --text "this movie is amazing"
```

### Web Server

```bash
python -m v2.cli serve
```

Open: http://127.0.0.1:8000

### Tests

```bash
python -m pytest -q
```

---

## Configuration (optional)

This project supports an optional JSON configuration file at `config.json` (project root).
CLI arguments always override configuration values.

Example:

```bash
python -m v2.cli --config my_config.json serve
```

---

## Engineering Notes

- Models are **loaded**, not retrained
- Dataset acquisition and preprocessing are fully automated
- No local paths or manual configuration steps are required
- v2 strictly reuses v0/v1 artifacts to ensure consistency and reproducibility
